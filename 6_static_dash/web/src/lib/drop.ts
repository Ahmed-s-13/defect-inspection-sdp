const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?|heic|heif)$/i;
export const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_EXT.test(f.name);

type Entry = FileSystemEntry & { isFile: boolean; isDirectory: boolean };
const readFile = (e: FileSystemFileEntry) => new Promise<File>((res, rej) => e.file(res, rej));
async function readDir(d: FileSystemDirectoryEntry): Promise<File[]> {
  const reader = d.createReader();
  const out: File[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    for (const e of batch as Entry[]) out.push(...(await walk(e)));
  }
  return out;
}
async function walk(e: Entry): Promise<File[]> {
  if (e.isFile) return [await readFile(e as unknown as FileSystemFileEntry)];
  if (e.isDirectory) return readDir(e as unknown as FileSystemDirectoryEntry);
  return [];
}

/** Files and whole folders from a drop; picks out images and a labels.csv if one is present. */
export async function filesFromDrop(dt: DataTransfer): Promise<{ images: File[]; labels?: File }> {
  const entries = Array.from(dt.items).map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as Entry[];
  const files = entries.length ? (await Promise.all(entries.map(walk))).flat() : Array.from(dt.files);
  return { images: files.filter(isImage).sort((a, b) => a.name.localeCompare(b.name)), labels: files.find((f) => f.name.toLowerCase() === "labels.csv") };
}
