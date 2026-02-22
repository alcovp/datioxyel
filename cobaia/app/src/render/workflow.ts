import { rm } from "node:fs/promises";
import path from "node:path";

import { buildGif, resetDirectory, runEngine } from "./engine";
import type { ParsedArgs, RenderFrameConfig, Vec3 } from "./types";

export async function renderSingleFrame(
  parsed: ParsedArgs,
  repoRoot: string,
  engineBinaryPath: string,
): Promise<void> {
  console.log(
    `Rendering scene '${parsed.scene}' (${parsed.width}x${parsed.height}, spp=${parsed.samplesPerPixel}, depth=${parsed.maxDepth}, fov=${parsed.cameraFovDeg}, quality=${parsed.quality}, sampling=${parsed.samplingMode})`,
  );
  const config: RenderFrameConfig = {
    width: parsed.width,
    height: parsed.height,
    outputPath: parsed.outputPath,
    maxDepth: parsed.maxDepth,
    samplesPerPixel: parsed.samplesPerPixel,
    scene: parsed.scene,
    rendererMode: "gpu",
    cameraOrigin: parsed.cameraOrigin,
    cameraTarget: parsed.cameraTarget,
    cameraFovDeg: parsed.cameraFovDeg,
    quality: parsed.quality,
    marchMaxSteps: parsed.marchMaxSteps,
    rrStartBounce: parsed.rrStartBounce,
    samplingMode: parsed.samplingMode,
  };
  await runEngine(config, repoRoot, engineBinaryPath);
  console.log(`Saved frame: ${config.outputPath}`);
}

export async function renderOrbitAnimation(
  parsed: ParsedArgs,
  repoRoot: string,
  engineBinaryPath: string,
): Promise<void> {
  const tempFramesDir = path.join(path.dirname(parsed.outputPath), ".orbit_frames");
  await resetDirectory(tempFramesDir);

  console.log(
    `Rendering orbit: ${parsed.orbitFrames} frames, turns=${parsed.orbitTurns}, radius=${parsed.orbitRadius.toFixed(3)}, height=${parsed.orbitHeight.toFixed(3)}`,
  );

  const frames: RenderFrameConfig[] = [];
  for (let frame = 0; frame < parsed.orbitFrames; frame += 1) {
    const angle =
      (parsed.orbitStartDeg * Math.PI) / 180 +
      ((Math.PI * 2 * parsed.orbitTurns * frame) / parsed.orbitFrames);
    const cameraOrigin: Vec3 = [
      parsed.cameraTarget[0] + parsed.orbitRadius * Math.cos(angle),
      parsed.orbitHeight,
      parsed.cameraTarget[2] + parsed.orbitRadius * Math.sin(angle),
    ];
    const outputPath = path.join(tempFramesDir, `frame_${String(frame).padStart(4, "0")}.png`);
    frames.push({
      width: parsed.width,
      height: parsed.height,
      outputPath,
      maxDepth: parsed.maxDepth,
      samplesPerPixel: parsed.samplesPerPixel,
      scene: parsed.scene,
      rendererMode: "gpu",
      cameraOrigin,
      cameraTarget: parsed.cameraTarget,
      cameraFovDeg: parsed.cameraFovDeg,
      quality: parsed.quality,
      marchMaxSteps: parsed.marchMaxSteps,
      rrStartBounce: parsed.rrStartBounce,
      samplingMode: parsed.samplingMode,
    });
  }

  try {
    const workerCount = resolveBatchWorkerCount(parsed.batchWorkers, frames.length);
    if (workerCount <= 1) {
      await runEngine({ frames }, repoRoot, engineBinaryPath);
    } else {
      console.log(`Parallel orbit workers: ${workerCount}`);
      await runOrbitWorkerPool(frames, workerCount, repoRoot, engineBinaryPath);
    }
    await buildGif(tempFramesDir, parsed.orbitFrames, parsed.outputPath, parsed.gifFps);
    console.log(`Saved GIF: ${parsed.outputPath}`);
  } finally {
    await rm(tempFramesDir, { recursive: true, force: true });
  }
}

async function runOrbitWorkerPool(
  frames: RenderFrameConfig[],
  workerCount: number,
  repoRoot: string,
  engineBinaryPath: string,
): Promise<void> {
  const chunks = distributeFramesRoundRobin(frames, workerCount);
  await Promise.all(
    chunks.map(async (chunk, index) => {
      console.log(
        `[worker ${index + 1}/${chunks.length}] rendering ${chunk.length} frames`,
      );
      await runEngine({ frames: chunk }, repoRoot, engineBinaryPath);
      console.log(`[worker ${index + 1}/${chunks.length}] done`);
    }),
  );
}

function distributeFramesRoundRobin(
  frames: RenderFrameConfig[],
  workerCount: number,
): RenderFrameConfig[][] {
  const normalizedWorkers = resolveBatchWorkerCount(workerCount, frames.length);
  const chunks = Array.from({ length: normalizedWorkers }, () => [] as RenderFrameConfig[]);
  for (let index = 0; index < frames.length; index += 1) {
    chunks[index % normalizedWorkers].push(frames[index]);
  }
  return chunks.filter((chunk) => chunk.length > 0);
}

function resolveBatchWorkerCount(requestedWorkers: number, totalFrames: number): number {
  if (totalFrames <= 1) {
    return 1;
  }
  const clamped = Math.min(Math.max(requestedWorkers, 1), totalFrames);
  return clamped;
}
