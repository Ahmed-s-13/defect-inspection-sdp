/** SAQR Prototype 1: its components, as the Helmet page explains them. `match` picks the component's parts in the
 *  3D model (public/helmet/saqr_prototype_1.glb, whose part names start with the component's number). */
export interface HelmetPart { id: string; name: string; text: string[]; match: RegExp }

export const HELMET_SUMMARY = [
  "SAQR Prototype 1 combines a construction helmet with a Raspberry Pi, webcam, illumination LED, manual controls, and a near-eye OLED display.",
  "The hardware uses readily available components, exposed wiring, and modified cinema glasses to demonstrate the proposed wearable arrangement.",
];

export const HELMET_PARTS: HelmetPart[] = [
  { id: "helmet", name: "Construction helmet", match: /^01_/, text: [
    "A basic black construction helmet serves as the physical base of Prototype 1.",
    "It supports the externally attached electronics, controls, camera, and display assembly."] },
  { id: "pi", name: "Raspberry Pi 5", match: /^02_/, text: [
    "The Raspberry Pi 5 acts as the main processing unit for the prototype.",
    "It remains inside its plastic case, mounted horizontally at the back of the helmet."] },
  { id: "webcam", name: "Logitech USB webcam", match: /^04_/, text: [
    "A front-mounted Logitech webcam captures images of the surrounding environment for processing.",
    "It uses its original casing and clip, with an exposed USB cable connected to the Raspberry Pi."] },
  { id: "led", name: "Torch LED", match: /^05_/, text: [
    "A basic LED extracted from a torch provides forward-facing illumination.",
    "Its plastic casing is retained, and the assembly is attached beside the webcam."] },
  { id: "breadboard", name: "Breadboard", match: /^03_(Right_Mini_Breadboard|Breadboard_|Exposed_Small_IC|IC_Pin)/, text: [
    "A small solderless breadboard holds the control components and their electrical connections.",
    "It is mounted at the middle of the wearer’s right side, with exposed jumper wiring."] },
  { id: "controls", name: "Push buttons and potentiometer", match: /^03_(Button_|Potentiometer_)/, text: [
    "Three push buttons provide manual inputs, while a potentiometer provides an adjustable input.",
    "The buttons are arranged horizontally on the breadboard alongside the potentiometer."] },
  { id: "glasses", name: "Modified 4DX glasses", match: /^06_/, text: [
    "A broken 4DX cinema-glasses frame provides a simple support for the near-eye display.",
    "Only the screen-bearing half remains, with its original lens removed and replaced by the OLED."] },
  { id: "oled", name: "OLED display", match: /^07_/, text: [
    "A small OLED screen sourced from Besomi provides visual information near the wearer’s eye.",
    "Its exposed circuit board is attached to the remaining glasses frame and connected through jumper wires."] },
  { id: "wiring", name: "Wiring and Blu Tack", match: /^08_/, text: [
    "Exposed jumper wires connect the torch LED, the OLED and the breadboard, and the webcam’s USB cable runs to the Raspberry Pi.",
    "Components are temporarily secured using Blu Tack, allowing their placement and connections to be adjusted during development."] },
];

/** The component a model part belongs to, from its name. */
export const partOf = (meshName: string) => HELMET_PARTS.find((p) => p.match.test(meshName));
