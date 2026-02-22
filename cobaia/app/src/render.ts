import { mkdir } from "node:fs/promises";
import path from "node:path";

import { parseCliArgs } from "./render/cli";
import { ensureEngineBinary, resolveCargoBinary } from "./render/engine";
import { renderOrbitAnimation, renderSingleFrame } from "./render/workflow";

async function main(): Promise<void> {
  const appDir = path.resolve(__dirname, "..");
  const repoRoot = path.resolve(appDir, "..");
  const cargoBinary = await resolveCargoBinary();
  const parsed = parseCliArgs(process.argv.slice(2), repoRoot);
  const engineBinaryPath = await ensureEngineBinary(repoRoot, cargoBinary);
  await mkdir(path.dirname(parsed.outputPath), { recursive: true });

  console.log("Renderer mode: GPU");
  if (parsed.orbitFrames > 1) {
    await renderOrbitAnimation(parsed, repoRoot, engineBinaryPath);
    return;
  }
  await renderSingleFrame(parsed, repoRoot, engineBinaryPath);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
