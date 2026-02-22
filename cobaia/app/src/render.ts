import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, copyFile, mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";

type Vec3 = [number, number, number];

type RenderFrameConfig = {
  width: number;
  height: number;
  outputPath: string;
  maxDepth: number;
  samplesPerPixel: number;
  scene: string;
  rendererMode: "cpu" | "gpu";
  cameraOrigin: Vec3;
  cameraTarget: Vec3;
};

type RenderBatchConfig = {
  frames: RenderFrameConfig[];
};

async function main(): Promise<void> {
  const appDir = path.resolve(__dirname, "..");
  const repoRoot = path.resolve(appDir, "..");
  const outDir = path.join(repoRoot, "out");
  await mkdir(outDir, { recursive: true });
  await clearStandaloneFrames(outDir);
  const tempFramesDir = path.join(outDir, ".gif_frames");
  await resetDirectory(tempFramesDir);

  const cargoBinary = await resolveCargoBinary();

  const frameCount = 100;
  const frameExportStride = 10;
  const gifFps = parsePositiveInt(process.env.GIF_FPS, 60);
  const rendererMode: "cpu" | "gpu" =
    process.env.RENDERER_MODE?.toLowerCase() === "cpu" ? "cpu" : "gpu";
  const orbitTarget: Vec3 = [0.0, -0.18, 0.0];
  const orbitHeight = 1.62;
  const orbitRadius = Math.hypot(2.9, 3.2);
  const orbitStartAngle = Math.atan2(3.2, 2.9);

  console.log(`Renderer mode: ${rendererMode.toUpperCase()}`);

  const frames: RenderFrameConfig[] = [];
  for (let frame = 0; frame < frameCount; frame += 1) {
    const angle = orbitStartAngle + ((Math.PI * 2 * frame) / frameCount);
    const cameraOrigin: Vec3 = [
      orbitRadius * Math.cos(angle),
      orbitHeight,
      orbitRadius * Math.sin(angle),
    ];
    const outputPath = path.join(tempFramesDir, `frame_${String(frame).padStart(3, "0")}.png`);

    const config: RenderFrameConfig = {
      width: 960,
      height: 540,
      outputPath,
      maxDepth: 4,
      samplesPerPixel: 4,
      scene: "menger_glass_on_plane",
      rendererMode,
      cameraOrigin,
      cameraTarget: orbitTarget,
    };

    frames.push(config);
  }

  try {
    console.log(`Rendering ${frameCount} frames in a single engine run...`);
    await runEngine({ frames }, repoRoot, cargoBinary);

    const gifPath = path.join(outDir, "orbit.gif");
    await buildGif(tempFramesDir, frameCount, gifPath, gifFps);
    console.log(`GIF assembled: ${path.basename(gifPath)}`);

    const exportedCount = await exportEveryNthFrame(
      tempFramesDir,
      outDir,
      frameCount,
      frameExportStride,
    );
    console.log(`Saved ${exportedCount} standalone PNGs (every ${frameExportStride}th frame).`);
  } finally {
    await rm(tempFramesDir, { recursive: true, force: true });
  }
}

async function resolveCargoBinary(): Promise<string> {
  const envOverride = process.env.CARGO_BIN;
  if (envOverride) {
    return envOverride;
  }

  const home = process.env.HOME;
  if (home) {
    const candidate = path.join(home, ".cargo", "bin", "cargo");
    try {
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Fall back to PATH lookup below.
    }
  }

  return "cargo";
}

function runEngine(
  config: RenderBatchConfig,
  repoRoot: string,
  cargoBinary: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const manifestPath = path.join(repoRoot, "engine", "Cargo.toml");
    const cargoHome = path.join(repoRoot, ".cargo-home");
    const cargoBinDir = path.dirname(cargoBinary);
    const currentPath = process.env.PATH ?? "";
    const mergedPath = currentPath
      ? `${cargoBinDir}:${currentPath}`
      : cargoBinDir;
    const engine = spawn(
      cargoBinary,
      ["run", "--release", "--quiet", "--manifest-path", manifestPath],
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          CARGO_HOME: cargoHome,
          PATH: mergedPath,
        },
        stdio: ["pipe", "inherit", "inherit"],
      },
    );

    engine.on("error", (error) => {
      reject(
        new Error(
          `Failed to start Rust engine. Install Rust/Cargo first. Details: ${error.message}`,
        ),
      );
    });

    engine.stdin.write(JSON.stringify(config));
    engine.stdin.end();

    engine.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`Rust engine exited with code ${code ?? "unknown"}.`));
    });
  });
}

function buildGif(
  framesDir: string,
  frameCount: number,
  gifPath: string,
  fps: number,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn(
      "ffmpeg",
      [
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        String(fps),
        "-start_number",
        "0",
        "-i",
        path.join(framesDir, "frame_%03d.png"),
        "-frames:v",
        String(frameCount),
        "-vf",
        "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=dither=sierra2_4a",
        gifPath,
      ],
      { stdio: ["ignore", "inherit", "inherit"] },
    );

    ffmpeg.on("error", (error) => {
      reject(new Error(`Failed to start ffmpeg for GIF assembly: ${error.message}`));
    });

    ffmpeg.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`ffmpeg exited with code ${code ?? "unknown"} while building GIF.`));
    });
  });
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  if (!raw) {
    return fallback;
  }

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return fallback;
  }

  return parsed;
}

async function resetDirectory(directoryPath: string): Promise<void> {
  await rm(directoryPath, { recursive: true, force: true });
  await mkdir(directoryPath, { recursive: true });
}

async function clearStandaloneFrames(outDir: string): Promise<void> {
  const entries = await readdir(outDir, { withFileTypes: true });
  const removalTasks: Promise<void>[] = [];
  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }
    if (/^frame_\d{2,3}\.png$/.test(entry.name)) {
      removalTasks.push(rm(path.join(outDir, entry.name), { force: true }));
    }
  }
  await Promise.all(removalTasks);
}

async function exportEveryNthFrame(
  framesDir: string,
  outDir: string,
  frameCount: number,
  stride: number,
): Promise<number> {
  let exported = 0;
  for (let frame = 0; frame < frameCount; frame += stride) {
    const fileName = `frame_${String(frame).padStart(3, "0")}.png`;
    await copyFile(path.join(framesDir, fileName), path.join(outDir, fileName));
    exported += 1;
  }
  return exported;
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
