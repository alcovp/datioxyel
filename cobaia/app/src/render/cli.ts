import path from "node:path";

import type { ParsedArgs, RenderQuality, SamplingMode, Vec3 } from "./types";

export function parseCliArgs(args: string[], repoRoot: string): ParsedArgs {
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
  let quality: RenderQuality = "balanced";
  let marchMaxSteps: number | undefined;
  let rrStartBounce: number | undefined;
  let samplingMode: SamplingMode = "halton";
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
  const parsed = parseStrictInteger(value, flag);
  if (parsed < min) {
    throw new Error(`${flag} must be an integer >= ${min}, got '${value}'`);
  }
  return parsed;
}

function parseIntInRange(value: string, flag: string, min: number, max: number): number {
  const parsed = parseStrictInteger(value, flag);
  if (parsed < min || parsed > max) {
    throw new Error(`${flag} must be an integer in [${min}, ${max}], got '${value}'`);
  }
  return parsed;
}

function parseFloatAtLeast(value: string, flag: string, min: number): number {
  const parsed = parseStrictFiniteFloat(value, flag);
  if (parsed < min) {
    throw new Error(`${flag} must be a number >= ${min}, got '${value}'`);
  }
  return parsed;
}

function parseFiniteFloat(value: string, flag: string): number {
  return parseStrictFiniteFloat(value, flag);
}

function parseFloatInRange(value: string, flag: string, minExclusive: number, maxExclusive: number): number {
  const parsed = parseStrictFiniteFloat(value, flag);
  if (parsed <= minExclusive || parsed >= maxExclusive) {
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
  let parsed: number;
  try {
    parsed = parseStrictInteger(raw, "environment variable");
  } catch {
    return fallback;
  }
  if (parsed < 1) {
    return fallback;
  }
  return parsed;
}

function parseVec3(raw: string, flag: string): Vec3 {
  const tokens = raw.split(",");
  if (tokens.length !== 3) {
    throw new Error(`${flag} expects 'x,y,z', got '${raw}'`);
  }

  const parsed = tokens.map((token) => parseStrictFiniteFloat(token.trim(), flag));
  return [parsed[0], parsed[1], parsed[2]];
}

function parseStrictInteger(value: string, flag: string): number {
  const trimmed = value.trim();
  if (!/^[+-]?\d+$/.test(trimmed)) {
    throw new Error(`${flag} must be an integer, got '${value}'`);
  }

  const parsed = Number(trimmed);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${flag} must be a safe integer, got '${value}'`);
  }
  return parsed;
}

function parseStrictFiniteFloat(value: string, flag: string): number {
  const trimmed = value.trim();
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    throw new Error(`${flag} must be a finite number, got '${value}'`);
  }

  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${flag} must be a finite number, got '${value}'`);
  }
  return parsed;
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
