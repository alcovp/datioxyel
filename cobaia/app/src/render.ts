import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdir, rm } from "node:fs/promises";
import path from "node:path";

type Vec3 = [number, number, number];

type RenderFrameConfig = {
  width: number;
  height: number;
  outputPath: string;
  maxDepth: number;
  samplesPerPixel: number;
  scene: string;
  rendererMode: "gpu";
  cameraOrigin: Vec3;
  cameraTarget: Vec3;
  cameraFovDeg: number;
  quality: "preview" | "balanced" | "final";
  marchMaxSteps?: number;
  rrStartBounce?: number;
  samplingMode: "random" | "halton" | "sobol";
};

type RenderBatchConfig = {
  frames: RenderFrameConfig[];
};

type ParsedArgs = {
  scene: string;
  cameraOrigin: Vec3;
  cameraTarget: Vec3;
  cameraFovDeg: number;
  samplesPerPixel: number;
  maxDepth: number;
  outputPath: string;
  width: number;
  height: number;
  orbitFrames: number;
  orbitTurns: number;
  orbitRadius: number;
  orbitHeight: number;
  orbitStartDeg: number;
  gifFps: number;
  quality: "preview" | "balanced" | "final";
  marchMaxSteps?: number;
  rrStartBounce?: number;
  samplingMode: "random" | "halton" | "sobol";
  batchWorkers: number;
};

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

async function renderSingleFrame(
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

async function renderOrbitAnimation(
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

async function ensureEngineBinary(repoRoot: string, cargoBinary: string): Promise<string> {
  const manifestPath = path.join(repoRoot, "engine", "Cargo.toml");
  const cargoHome = path.join(repoRoot, ".cargo-home");
  const cargoBinDir = path.dirname(cargoBinary);
  const currentPath = process.env.PATH ?? "";
  const mergedPath = currentPath
    ? `${cargoBinDir}:${currentPath}`
    : cargoBinDir;

  await runCommand(cargoBinary, ["build", "--release", "--quiet", "--manifest-path", manifestPath], {
    cwd: repoRoot,
    env: {
      ...process.env,
      CARGO_HOME: cargoHome,
      PATH: mergedPath,
    },
    stdio: ["ignore", "inherit", "inherit"],
  }, "Failed to build Rust engine binary.");

  const binaryName = process.platform === "win32" ? "cobaia-engine.exe" : "cobaia-engine";
  const engineBinaryPath = path.join(repoRoot, "engine", "target", "release", binaryName);
  await access(engineBinaryPath, constants.X_OK);
  return engineBinaryPath;
}

function runEngine(
  config: RenderFrameConfig | RenderBatchConfig,
  repoRoot: string,
  engineBinaryPath: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const engine = spawn(
      engineBinaryPath,
      [],
      {
        cwd: repoRoot,
        env: process.env,
        stdio: ["pipe", "inherit", "inherit"],
      },
    );

    engine.on("error", (error) => {
      reject(
        new Error(
          `Failed to start Rust engine binary '${engineBinaryPath}'. Details: ${error.message}`,
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

function runCommand(
  command: string,
  args: string[],
  options: {
    cwd: string;
    env: NodeJS.ProcessEnv;
    stdio: ["ignore", "inherit", "inherit"] | ["pipe", "inherit", "inherit"];
  },
  errorPrefix: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const processHandle = spawn(command, args, options);

    processHandle.on("error", (error) => {
      reject(new Error(`${errorPrefix} ${error.message}`));
    });

    processHandle.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${errorPrefix} Exit code: ${code ?? "unknown"}.`));
    });
  });
}

function buildGif(
  framesDir: string,
  frameCount: number,
  outputPath: string,
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
        path.join(framesDir, "frame_%04d.png"),
        "-frames:v",
        String(frameCount),
        "-vf",
        "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=dither=sierra2_4a",
        outputPath,
      ],
      { stdio: ["ignore", "inherit", "inherit"] },
    );

    ffmpeg.on("error", (error) => {
      reject(
        new Error(
          `Failed to start ffmpeg for GIF assembly: ${error.message}. Install ffmpeg to use orbit GIF mode.`,
        ),
      );
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

function parseCliArgs(args: string[], repoRoot: string): ParsedArgs {
  let scene = "menger_glass_on_plane";
  let cameraOrigin: Vec3 = [2.9, 1.62, 3.2];
  let cameraTarget: Vec3 = [0.0, -0.18, 0.0];
  let cameraFovDeg = 38.0;
  let samplesPerPixel = 4;
  let maxDepth = 4;
  let width = 960;
  let height = 540;
  let orbitFrames = 1;
  let orbitTurns = 1.0;
  let orbitRadius: number | null = null;
  let orbitHeight: number | null = null;
  let orbitStartDeg: number | null = null;
  let gifFps = parsePositiveInt(process.env.GIF_FPS, 60);
  let quality: "preview" | "balanced" | "final" = "balanced";
  let marchMaxSteps: number | undefined;
  let rrStartBounce: number | undefined;
  let samplingMode: "random" | "halton" | "sobol" = "halton";
  let batchWorkers = parsePositiveInt(process.env.RENDER_BATCH_WORKERS, 1);
  let outputPath = path.join(repoRoot, "out", "frame.png");
  let outputProvided = false;

  for (let index = 0; index < args.length; index += 1) {
    const flag = args[index];
    switch (flag) {
      case "--":
        break;
      case "--help":
      case "-h":
        printUsage();
        process.exit(0);
      case "--scene": {
        const value = consumeNextValue(args, flag, index);
        scene = value;
        index += 1;
        break;
      }
      case "--camera":
      case "--camera-origin": {
        const value = consumeNextValue(args, flag, index);
        cameraOrigin = parseVec3(value, flag);
        index += 1;
        break;
      }
      case "--target":
      case "--camera-target": {
        const value = consumeNextValue(args, flag, index);
        cameraTarget = parseVec3(value, flag);
        index += 1;
        break;
      }
      case "--fov": {
        const value = consumeNextValue(args, flag, index);
        cameraFovDeg = parseFloatInRange(value, flag, 1.0, 179.0);
        index += 1;
        break;
      }
      case "--quality": {
        const value = consumeNextValue(args, flag, index);
        quality = parseChoice(value, flag, ["preview", "balanced", "final"]);
        index += 1;
        break;
      }
      case "--spp": {
        const value = consumeNextValue(args, flag, index);
        samplesPerPixel = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--depth": {
        const value = consumeNextValue(args, flag, index);
        maxDepth = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--width": {
        const value = consumeNextValue(args, flag, index);
        width = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--height": {
        const value = consumeNextValue(args, flag, index);
        height = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--orbit-frames": {
        const value = consumeNextValue(args, flag, index);
        orbitFrames = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--orbit-turns": {
        const value = consumeNextValue(args, flag, index);
        orbitTurns = parseFloatAtLeast(value, flag, 0.0001);
        index += 1;
        break;
      }
      case "--orbit-radius": {
        const value = consumeNextValue(args, flag, index);
        orbitRadius = parseFloatAtLeast(value, flag, 0.0001);
        index += 1;
        break;
      }
      case "--orbit-height": {
        const value = consumeNextValue(args, flag, index);
        orbitHeight = parseFiniteFloat(value, flag);
        index += 1;
        break;
      }
      case "--orbit-start-deg": {
        const value = consumeNextValue(args, flag, index);
        orbitStartDeg = parseFiniteFloat(value, flag);
        index += 1;
        break;
      }
      case "--gif-fps": {
        const value = consumeNextValue(args, flag, index);
        gifFps = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--march-max-steps": {
        const value = consumeNextValue(args, flag, index);
        marchMaxSteps = parseIntInRange(value, flag, 1, 280);
        index += 1;
        break;
      }
      case "--rr-start": {
        const value = consumeNextValue(args, flag, index);
        rrStartBounce = parseIntInRange(value, flag, 0, 32);
        index += 1;
        break;
      }
      case "--sampling-mode": {
        const value = consumeNextValue(args, flag, index);
        samplingMode = parseChoice(value, flag, ["random", "halton", "sobol"]);
        index += 1;
        break;
      }
      case "--batch-workers":
      case "--workers": {
        const value = consumeNextValue(args, flag, index);
        batchWorkers = parseIntAtLeast(value, flag, 1);
        index += 1;
        break;
      }
      case "--output": {
        const value = consumeNextValue(args, flag, index);
        outputPath = path.isAbsolute(value) ? value : path.join(repoRoot, value);
        outputProvided = true;
        index += 1;
        break;
      }
      default:
        throw new Error(`Unknown CLI argument '${flag}'. Use --help.`);
    }
  }

  if (!scene.trim()) {
    throw new Error("scene must be non-empty");
  }

  const defaultOrbitRadius = Math.hypot(
    cameraOrigin[0] - cameraTarget[0],
    cameraOrigin[2] - cameraTarget[2],
  );
  const defaultOrbitHeight = cameraOrigin[1];
  const defaultOrbitStartDeg =
    (Math.atan2(cameraOrigin[2] - cameraTarget[2], cameraOrigin[0] - cameraTarget[0]) * 180) /
    Math.PI;

  if (orbitFrames > 1 && !outputProvided) {
    outputPath = path.join(repoRoot, "out", "orbit.gif");
  }

  return {
    scene,
    cameraOrigin,
    cameraTarget,
    cameraFovDeg,
    samplesPerPixel,
    maxDepth,
    outputPath,
    width,
    height,
    orbitFrames,
    orbitTurns,
    orbitRadius: orbitRadius ?? defaultOrbitRadius,
    orbitHeight: orbitHeight ?? defaultOrbitHeight,
    orbitStartDeg: orbitStartDeg ?? defaultOrbitStartDeg,
    gifFps,
    quality,
    marchMaxSteps,
    rrStartBounce,
    samplingMode,
    batchWorkers,
  };
}

function consumeNextValue(args: string[], flag: string, index: number): string {
  if (index + 1 >= args.length) {
    throw new Error(`Missing value for ${flag}`);
  }
  return args[index + 1];
}

function parseIntAtLeast(value: string, flag: string, min: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < min) {
    throw new Error(`${flag} must be an integer >= ${min}, got '${value}'`);
  }
  return parsed;
}

function parseIntInRange(value: string, flag: string, min: number, max: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    throw new Error(`${flag} must be an integer in [${min}, ${max}], got '${value}'`);
  }
  return parsed;
}

function parseFloatAtLeast(value: string, flag: string, min: number): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed) || parsed < min) {
    throw new Error(`${flag} must be a number >= ${min}, got '${value}'`);
  }
  return parsed;
}

function parseFiniteFloat(value: string, flag: string): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${flag} must be a finite number, got '${value}'`);
  }
  return parsed;
}

function parseFloatInRange(value: string, flag: string, minExclusive: number, maxExclusive: number): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed) || parsed <= minExclusive || parsed >= maxExclusive) {
    throw new Error(
      `${flag} must be finite and in (${minExclusive}, ${maxExclusive}), got '${value}'`,
    );
  }
  return parsed;
}

function parseChoice<T extends string>(
  value: string,
  flag: string,
  allowed: readonly T[],
): T {
  const normalized = value.toLowerCase();
  if (allowed.includes(normalized as T)) {
    return normalized as T;
  }
  throw new Error(`${flag} must be one of: ${allowed.join(", ")}`);
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

function parseVec3(raw: string, flag: string): Vec3 {
  const tokens = raw.split(",");
  if (tokens.length !== 3) {
    throw new Error(`${flag} expects 'x,y,z', got '${raw}'`);
  }

  const parsed = tokens.map((token) => Number.parseFloat(token.trim()));
  if (parsed.some((value) => !Number.isFinite(value))) {
    throw new Error(`${flag} must contain finite numbers, got '${raw}'`);
  }

  return [parsed[0], parsed[1], parsed[2]];
}

async function resetDirectory(directoryPath: string): Promise<void> {
  await rm(directoryPath, { recursive: true, force: true });
  await mkdir(directoryPath, { recursive: true });
}

function printUsage(): void {
  console.log(`Usage: yarn render -- [options]

Options:
  --scene <id>               Scene id (default: menger_glass_on_plane)
  --camera <x,y,z>           Camera origin (alias: --camera-origin)
  --target <x,y,z>           Camera target (alias: --camera-target)
  --fov <degrees>            Vertical FOV in degrees, (1, 179)
  --quality <preset>         Render preset: preview, balanced, final
  --spp <int>                Samples per pixel, >= 1
  --depth <int>              Max ray depth, >= 1
  --march-max-steps <int>    Override march steps, [1, 280]
  --rr-start <int>           Override RR start bounce, [0, 32]
  --sampling-mode <mode>     Sampling mode: random, halton, sobol
  --batch-workers <int>      Parallel workers for orbit frame batches, >= 1
  --output <path>            Output image path (default: out/frame.png)
  --width <int>              Image width, >= 1 (default: 960)
  --height <int>             Image height, >= 1 (default: 540)
  --orbit-frames <int>       Enable orbit mode with N frames (N > 1)
  --orbit-turns <float>      Orbit turns count (default: 1)
  --orbit-radius <float>     Orbit radius around target (default from camera/target)
  --orbit-height <float>     Camera Y for orbit (default from camera)
  --orbit-start-deg <float>  Start angle in degrees (default from camera/target)
  --gif-fps <int>            GIF framerate for orbit mode (default: 60 or GIF_FPS env)
                             Workers default: 1 or RENDER_BATCH_WORKERS env
  -h, --help                 Show this help

Example:
  yarn render -- --scene menger_glass_dual_light --camera 3.9,1.62,3.2 --target 0,-0.18,0 --fov 38 --quality final --spp 8 --depth 6 --march-max-steps 280 --rr-start 4 --sampling-mode halton --output out/frame.png
  yarn render -- --scene menger_glass_dual_light --target 0,-0.18,0 --orbit-frames 100 --batch-workers 4 --quality balanced --spp 4 --depth 5 --output out/orbit.gif`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
