import { useEffect, useState } from "react";
import type { Square } from "../api/client";

/** A client-side copy of the model's crop, available before the server answers (used for the 8-view fan). */
export function useCropPreview(url: string | undefined, sq: Square | undefined) {
  const [data, setData] = useState<string | null>(null);
  useEffect(() => {
    if (!url || !sq) { setData(null); return; }
    let alive = true;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = c.height = 224;
      c.getContext("2d")!.drawImage(img, sq.left, sq.top, sq.side, sq.side, 0, 0, 224, 224);
      if (alive) setData(c.toDataURL("image/jpeg", 0.85));
    };
    img.src = url;
    return () => { alive = false; };
  }, [url, sq?.left, sq?.top, sq?.side]);
  return data;
}
