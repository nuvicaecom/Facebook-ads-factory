import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { inlineSource } from "inline-source";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function run() {
  const outDir = resolve(__dirname, "./out");
  const inputHtml = resolve(outDir, "index.html");

  let html = readFileSync(inputHtml, "utf8");

  // Inline CSS + JS, skip images/fonts/media
  const inlined = await inlineSource(html, {
    rootpath: outDir,
    compress: false,
    swallowErrors: false,
    attribute: false,
    ignore: ["png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "mp4", "mp3"],
  });

  // Strip base64-encoded images → empty src
  const stripped = inlined.replace(
    /<img([^>]*?)src=["']data:image\/[^"']+["']([^>]*?)>/gi,
    '<img$1src=""$2>'
  );

  const outFile = resolve(__dirname, "./dist/single.html");
  writeFileSync(outFile, stripped, "utf8");

  console.log("✅ Single HTML generated at", outFile);
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
