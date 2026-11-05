import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { partOf } from "../data/helmet";

/** The Helmet page's 3D scene: the prototype turning slowly on a soft shadow, dragged by hand, its components
 *  highlighted on hover and brought into close view when picked. Plain three.js, owned by HelmetView. */
export interface HelmetSceneEvents {
  onLoad(): void;
  onError(message: string): void;
  onHover(id: string | null): void;
  onPick(id: string | null): void;   // a click on a part, or on empty space (null)
}

const LIFT = new THREE.Color("#CFE6E8"); // hover: a faint, slightly cool light on the part, not a new colour
const SPIN = 1.1;          // OrbitControls autoRotateSpeed: about one turn a minute
const RESUME_MS = 2500;    // spinning resumes this long after a drag, when nothing is picked
const FLY = 0.9;           // seconds for the camera to move to a part and back
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

interface Fly { from: [THREE.Vector3, THREE.Vector3]; to: [THREE.Vector3, THREE.Vector3]; t0: number }

export class HelmetScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
  private controls: OrbitControls;
  private pmrem: THREE.PMREMGenerator;
  private model = new THREE.Group();
  private meshes: THREE.Mesh[] = [];
  private byPart = new Map<string, THREE.Mesh[]>();
  private home: [THREE.Vector3, THREE.Vector3] = [new THREE.Vector3(), new THREE.Vector3()];
  private fly: Fly | null = null;
  private hovered: string | null = null;
  private picked: string | null = null;
  private loaded = false;
  private resumeAt = 0;
  private raf = 0;
  private down: { x: number; y: number } | null = null;
  private ray = new THREE.Raycaster();
  private resize: ResizeObserver;
  private reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  constructor(private host: HTMLElement, private ev: HelmetSceneEvents) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.className = "helmet-canvas";
    host.appendChild(this.renderer.domElement);

    // a soft studio: a room environment for the reflections on the shell, a key light and a rim light
    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = this.pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.9;
    const key = new THREE.DirectionalLight("#ffffff", 1.6); key.position.set(2, 3, 2);
    const rim = new THREE.DirectionalLight("#dfe8ff", 1.1); rim.position.set(-2, 1.5, -2.5);
    this.scene.add(key, rim, this.model);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, {
      enableDamping: true, dampingFactor: 0.08, enablePan: false, enableZoom: false, rotateSpeed: 0.8,
      autoRotate: !this.reduce, autoRotateSpeed: SPIN, minPolarAngle: Math.PI * 0.18, maxPolarAngle: Math.PI * 0.72,
    });
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; this.fly = null; });
    this.controls.addEventListener("end", () => { this.resumeAt = performance.now() + RESUME_MS; });

    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", this.onDown);
    el.addEventListener("pointerup", this.onUp);
    el.addEventListener("pointermove", this.onMove);
    el.addEventListener("pointerleave", () => this.hover(null));
    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(host);
    this.fit();
    this.load();
    this.raf = requestAnimationFrame(this.tick);
  }

  private load() {
    new GLTFLoader().load("helmet/saqr_prototype_1.glb", (gltf) => {
      const root = gltf.scene;
      root.traverse((o) => {
        const m = o as THREE.Mesh;
        if (!m.isMesh) return;
        // the file has positions only: smooth normals on curves, sharp ones on edges like the Pi case
        m.geometry = toCreasedNormals(m.geometry, THREE.MathUtils.degToRad(35));
        // each part gets its own material, so it can be lit and dimmed on its own. The file's colours were written
        // as screen (sRGB) values, so they are converted to the linear values three.js expects
        const mat = (m.material as THREE.MeshStandardMaterial).clone();
        mat.color.convertSRGBToLinear();
        // the shell is coarse, so a glossy finish mirrors the studio in blocks: a satin finish reads as the real plastic
        if (mat.name === "shell") { mat.roughness = 0.62; mat.envMapIntensity = 0.55; }
        m.material = mat;
        const part = partOf(m.name);
        if (!part) return;
        m.userData.part = part.id;
        this.meshes.push(m);
        this.byPart.set(part.id, [...(this.byPart.get(part.id) ?? []), m]);
      });
      // centre it and scale it to about one unit, so the camera numbers don't depend on the file's metres
      const box = new THREE.Box3().setFromObject(root), size = box.getSize(new THREE.Vector3());
      const s = 1 / Math.max(size.x, size.y, size.z);
      root.scale.setScalar(s);
      root.position.copy(box.getCenter(new THREE.Vector3()).multiplyScalar(-s));
      this.model.add(root, this.shadow(size.x * s, size.z * s, -size.y * s / 2));
      // the opening view: the wearer's right side, three quarters from the front (front = -Z)
      this.home = [new THREE.Vector3(2.05, 0.5, -0.92), new THREE.Vector3(0, -0.03, 0)];
      this.camera.position.copy(this.home[0]);
      this.controls.target.copy(this.home[1]);
      this.controls.update();
      this.loaded = true;
      this.ev.onLoad();
    }, undefined, (err) => this.ev.onError(String((err as Error)?.message ?? err)));
  }

  /** A soft round shadow on an invisible floor under the helmet. */
  private shadow(w: number, d: number, y: number) {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d")!, r = g.createRadialGradient(64, 64, 4, 64, 64, 64);
    r.addColorStop(0, "rgba(22,24,27,.32)"); r.addColorStop(1, "rgba(22,24,27,0)");
    g.fillStyle = r; g.fillRect(0, 0, 128, 128);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(w * 1.5, d * 1.5),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = y - 0.02;
    return plane;
  }

  private fit() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // on narrow screens widen the view, so the helmet keeps the side-to-side fit it has on a wide one
    // static dash: on a phone the helmet fills the width of the screen
    const wide = Math.tan(THREE.MathUtils.degToRad(15)) * (w < 720 ? 0.95 : 1.25);
    this.camera.fov = this.camera.aspect < 1.25 ? THREE.MathUtils.radToDeg(2 * Math.atan(wide / this.camera.aspect)) : 30;
    this.camera.updateProjectionMatrix();
  }

  private tick = (now: number) => {
    this.raf = requestAnimationFrame(this.tick);
    if (this.fly) {
      const k = Math.min(1, (now - this.fly.t0) / (FLY * 1000)), e = ease(k);
      this.camera.position.lerpVectors(this.fly.from[0], this.fly.to[0], e);
      this.controls.target.lerpVectors(this.fly.from[1], this.fly.to[1], e);
      if (k >= 1) this.fly = null;
    } else if (!this.picked && !this.reduce && !this.controls.autoRotate && this.resumeAt && now > this.resumeAt) {
      this.controls.autoRotate = true;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  private hit(e: PointerEvent) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), this.camera);
    const visible = this.picked ? this.byPart.get(this.picked) ?? [] : this.meshes; // dimmed parts can't be picked
    return (this.ray.intersectObjects(visible, false)[0]?.object.userData.part as string | undefined) ?? null;
  }
  private onDown = (e: PointerEvent) => { this.down = { x: e.clientX, y: e.clientY }; };
  private onUp = (e: PointerEvent) => {
    const d = this.down;
    this.down = null;
    if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return; // that was a drag
    this.ev.onPick(this.hit(e));
  };
  private onMove = (e: PointerEvent) => { if (!e.buttons) this.hover(this.hit(e)); };

  /** Tint a component while the pointer is on it (in the scene or in the page's list). */
  hover(id: string | null) {
    if (id === this.hovered) return;
    this.tint(this.hovered, 0);
    this.hovered = id;
    this.tint(id, 1);
    this.renderer.domElement.style.cursor = id ? "pointer" : "";
    this.ev.onHover(id);
  }
  private tint(id: string | null, k: number) {
    for (const m of (id && this.byPart.get(id)) || []) {
      const mat = m.material as THREE.MeshStandardMaterial;
      // in proportion to the part's own colour, so black and white parts brighten by about the same, gentle step
      mat.emissive.copy(mat.color).multiplyScalar(0.16 * k).add(LIFT.clone().multiplyScalar(0.018 * k));
    }
  }

  /** Bring one component into close view and fade the rest back; null returns to the turning overview. */
  pick(id: string | null) {
    if (!this.loaded || id === this.picked) return;
    this.picked = id;
    for (const m of this.meshes) {
      const mat = m.material as THREE.MeshStandardMaterial, on = !id || m.userData.part === id;
      mat.transparent = !on;
      mat.opacity = on ? 1 : 0.14;
      mat.depthWrite = on;
    }
    this.controls.autoRotate = false;
    const from: [THREE.Vector3, THREE.Vector3] = [this.camera.position.clone(), this.controls.target.clone()];
    let to: [THREE.Vector3, THREE.Vector3];
    if (!id || id === "helmet") {
      // back to the overview at the current angle, so the turn carries on from where it is
      const dir = this.camera.position.clone().sub(this.controls.target).setY(0).normalize();
      to = [dir.multiplyScalar(Math.hypot(this.home[0].x, this.home[0].z)).setY(this.home[0].y), this.home[1].clone()];
      this.resumeAt = id ? 0 : performance.now() + 600;
    } else {
      const box = new THREE.Box3();
      for (const m of this.byPart.get(id) ?? []) box.expandByObject(m);
      const c = box.getCenter(new THREE.Vector3()), radius = box.getSize(new THREE.Vector3()).length() / 2;
      // look at it from outside the helmet, leaning towards the current view so the move is short
      const out = c.clone().setY(c.y * 0.4).normalize();
      const now = this.camera.position.clone().sub(this.controls.target).normalize();
      const dir = out.multiplyScalar(0.65).add(now.multiplyScalar(0.35)).add(new THREE.Vector3(0, 0.18, 0)).normalize();
      const dist = Math.max(0.75, (radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.9);
      to = [c.clone().add(dir.multiplyScalar(dist)), c];
      this.resumeAt = 0;
    }
    this.fly = this.reduce ? null : { from, to, t0: performance.now() };
    if (this.reduce) { this.camera.position.copy(to[0]); this.controls.target.copy(to[1]); }
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.resize.disconnect();
    this.controls.dispose();
    this.model.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.geometry.dispose();
      const mat = m.material as THREE.MeshStandardMaterial & { map?: THREE.Texture };
      mat.map?.dispose();
      mat.dispose();
    });
    this.scene.environment?.dispose();
    this.pmrem.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
