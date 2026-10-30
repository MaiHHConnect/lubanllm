import { Database } from "bun:sqlite";
import { createHash, timingSafeEqual } from "node:crypto";

const ORIGIN = "https://ai.maihh.net";
const DEFAULT_IMAGE = "/app-ad/luban-intro-v1.png";
const MAX_BODY = 3 * 1024 * 1024;
const MAX_IMAGE = 2 * 1024 * 1024;
type Copy = { zh: string; en: string };
type Ad = { schema: 1; title: Copy; subtitle: Copy; cta: Copy; href: string; image: string };
const DEFAULT: Ad = {
  schema: 1, title: { zh: "鲁班零号", en: "Luban Zero" },
  subtitle: { zh: "让每一次使用，都成为成长的机会", en: "Turn every interaction into an opportunity to grow" },
  cta: { zh: "了解鲁班零号", en: "Discover Luban Zero" }, href: ORIGIN, image: DEFAULT_IMAGE,
};
export class AdError extends Error {
  constructor(message: string, readonly status = 400) { super(message); }
}
export function appManagementAuth(adminKey: string) {
  if (!/^[0-9a-f]{64}$/.test(adminKey)) throw new Error("Invalid application management credential");
  const credential = createHash("sha256").update(adminKey).digest();
  return (request: Request) => {
    const supplied = request.headers.get("authorization")?.match(/^Bearer ([0-9a-f]{64})$/)?.[1] || "";
    if (!timingSafeEqual(credential, createHash("sha256").update(supplied).digest())) throw new AdError("需要管理凭证 / Management credential required", 401);
    const origin = request.headers.get("origin");
    if ((origin && origin !== ORIGIN) || request.headers.get("sec-fetch-site") === "cross-site") throw new AdError("来源不允许 / Origin not allowed", 403);
  };
}
function copy(value: unknown, max: number): Copy {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).sort().join() !== "en,zh") throw new AdError("中英文文案不完整 / Both languages are required");
  const result = value as Record<string, unknown>;
  for (const language of ["zh", "en"]) {
    const text = result[language];
    if (typeof text !== "string" || !text.trim() || [...text].length > max || /\p{C}/u.test(text)) throw new AdError("文案格式无效 / Invalid copy");
  }
  return { zh: (result.zh as string).trim(), en: (result.en as string).trim() };
}
async function body(request: Request): Promise<Record<string, unknown>> {
  if (!/^application\/json(?:\s*;.*)?$/i.test(request.headers.get("content-type") || "")) throw new AdError("需要 JSON / JSON required", 415);
  const length = request.headers.get("content-length");
  if (length && (!/^\d+$/.test(length) || Number(length) > MAX_BODY)) throw new AdError("请求过大 / Request too large", 413);
  const chunks: Uint8Array[] = [];
  let size = 0;
  const reader = request.body?.getReader();
  if (reader) try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) { await reader.cancel(); throw new AdError("请求过大 / Request too large", 413); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  try {
    const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks)));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error();
    return value;
  } catch { throw new AdError("JSON 无效 / Invalid JSON"); }
}
const crcTable = Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  return value >>> 0;
});
function png(value: unknown): Buffer {
  if (typeof value !== "string" || !value.length || value.length > Math.ceil(MAX_IMAGE / 3) * 4 || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) throw new AdError("PNG 格式无效 / Invalid PNG");
  const bytes = Buffer.from(value, "base64");
  if (bytes.length > MAX_IMAGE) throw new AdError("图片需小于 2 MiB / Image must be below 2 MiB", 413);
  if (bytes.toString("base64") !== value || !bytes.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"))) throw new AdError("仅接受 PNG / PNG only");
  let offset = 8, data = false, ended = false;
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset), type = bytes.toString("ascii", offset + 4, offset + 8), end = offset + 12 + length;
    if (end > bytes.length || !/^[A-Za-z]{4}$/.test(type)) throw new AdError("PNG 数据损坏 / Corrupt PNG");
    let crc = 0xffffffff;
    for (let i = offset + 4; i < end - 4; i++) crc = crcTable[(crc ^ bytes[i]) & 255] ^ (crc >>> 8);
    if (((crc ^ 0xffffffff) >>> 0) !== bytes.readUInt32BE(end - 4)) throw new AdError("PNG 校验失败 / PNG checksum failed");
    if (offset === 8) {
      if (type !== "IHDR" || length !== 13) throw new AdError("PNG 头无效 / Invalid PNG header");
      const width = bytes.readUInt32BE(offset + 8), height = bytes.readUInt32BE(offset + 12);
      const depth = bytes[offset + 16], color = bytes[offset + 17];
      const depths: Record<number, number[]> = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] };
      if (!width || !height || width > 8192 || height > 2048 || width * height > 8388608 || !depths[color]?.includes(depth) || bytes[offset + 18] !== 0 || bytes[offset + 19] !== 0 || bytes[offset + 20] > 1) throw new AdError("PNG 尺寸或编码无效 / Invalid PNG dimensions or encoding");
    } else if (type === "IHDR") throw new AdError("PNG 头重复 / Duplicate PNG header");
    if (type === "IDAT") data = true;
    if (type === "IEND") {
      if (length !== 0 || !data || end !== bytes.length) throw new AdError("PNG 结尾无效 / Invalid PNG ending");
      ended = true;
      break;
    }
    offset = end;
  }
  if (!ended) throw new AdError("PNG 不完整 / Incomplete PNG");
  return bytes;
}

export function createAppAd(db: Database, adminKey: string) {
  const authorize = appManagementAuth(adminKey);
  db.exec("CREATE TABLE IF NOT EXISTS app_ad (id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL, image_hash TEXT, image BLOB)");
  const stored = () => db.query<{ config: string; image_hash: string | null; image: Uint8Array | null }, []>("SELECT config,image_hash,image FROM app_ad WHERE id=1").get();
  const reply = (value: unknown, status = 200, publicAccess = false) => Response.json(value, { status, headers: {
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    ...(publicAccess ? { "Access-Control-Allow-Origin": "*" } : {}),
  } });
  return {
    async handle(request: Request): Promise<Response | null> {
      const url = new URL(request.url), path = url.pathname, publicAccess = path === "/api/app-ad" || path.startsWith("/api/app-ad/image/");
      if (!publicAccess && path !== "/api/admin/app-ad") return null;
      try {
        if (!publicAccess) authorize(request);
        const method = request.method === "HEAD" ? "GET" : request.method;
        if (method === "GET" && ["/api/app-ad", "/api/admin/app-ad"].includes(path)) {
          const value = stored();
          return reply(value ? JSON.parse(value.config) : DEFAULT, 200, publicAccess);
        }
        const image = path.match(/^\/api\/app-ad\/image\/([0-9a-f]{64})\.png$/);
        if (method === "GET" && image) {
          const value = stored();
          if (!value?.image || value.image_hash !== image[1]) throw new AdError("图片不存在 / Image not found", 404);
          return new Response(request.method === "HEAD" ? null : value.image, { headers: { "Content-Type": "image/png", "Content-Length": String(value.image.byteLength), "Cache-Control": "public, max-age=31536000, immutable", "Access-Control-Allow-Origin": "*", "X-Content-Type-Options": "nosniff" } });
        }
        if (path === "/api/admin/app-ad" && method === "PUT") {
          const input = await body(request);
          if (Object.keys(input).some(key => !["schema", "title", "subtitle", "cta", "href", "image", "png_base64"].includes(key)) || input.schema !== 1) throw new AdError("广告字段无效 / Invalid advertisement fields");
          let link: URL;
          try { link = new URL(input.href as string); } catch { throw new AdError("链接无效 / Invalid link"); }
          if (typeof input.href !== "string" || input.href.length > 2048 || link.origin !== ORIGIN || link.username || link.password || /[\u0000-\u0020\u007f]/.test(input.href)) throw new AdError("只允许官网链接 / Website links only");
          const old = stored();
          let imageHash = old?.image_hash || null, imageBytes = old?.image || null;
          let imagePath = input.image === undefined ? (old ? JSON.parse(old.config).image : DEFAULT_IMAGE) : input.image;
          if (Object.hasOwn(input, "png_base64")) {
            if (Object.hasOwn(input, "image")) throw new AdError("图片字段冲突 / Conflicting image fields");
            imageBytes = png(input.png_base64);
            imageHash = createHash("sha256").update(imageBytes).digest("hex");
            imagePath = `/api/app-ad/image/${imageHash}.png`;
          } else if (imagePath !== DEFAULT_IMAGE && imagePath !== `/api/app-ad/image/${imageHash}.png`) throw new AdError("图片路径无效 / Invalid image path");
          if (imagePath === DEFAULT_IMAGE) { imageHash = null; imageBytes = null; }
          const config: Ad = { schema: 1, title: copy(input.title, 60), subtitle: copy(input.subtitle, 160), cta: copy(input.cta, 40), href: link.href, image: imagePath as string };
          db.query("INSERT INTO app_ad VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET config=excluded.config,image_hash=excluded.image_hash,image=excluded.image").run(JSON.stringify(config), imageHash, imageBytes);
          return reply(config);
        }
        return reply({ error: "方法或路径不允许 / Method or path not allowed" }, image ? 405 : publicAccess && path !== "/api/app-ad" ? 404 : 405, publicAccess);
      } catch (error) {
        if (error instanceof AdError) return reply({ error: error.message }, error.status, publicAccess);
        console.error("Advertisement request failed", error instanceof Error ? error.name : "unknown");
        return reply({ error: "广告服务暂时不可用 / Advertisement unavailable" }, 503, publicAccess);
      }
    },
  };
}
