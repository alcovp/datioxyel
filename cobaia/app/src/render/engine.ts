import { spawn } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdir, rm } from "node:fs/promises";
import path from "node:path";

import type { RenderBatchConfig, RenderFrameConfig } from "./types";

export async function resolveCargoBinary(): Promise<string> {
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

export async function ensureEngineBinary(repoRoot: string, cargoBinary: string): Promise<string> {
  const manifestPath = path.join(repoRoot, "engine", "Cargo.toml");
  const cargoHome = path.join(repoRoot, ".cargo-home");
  const cargoBinDir = path.dirname(cargoBinary);
  const currentPath = process.env.PATH ?? "";
  const shouldPrependCargoBin = cargoBinDir !== "." && cargoBinDir !== "";
  const mergedPath = shouldPrependCargoBin
    ? (currentPath ? `${cargoBinDir}${path.delimiter}${currentPath}` : cargoBinDir)
    : currentPath;

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

export function runEngine(
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

export function buildGif(
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

export async function resetDirectory(directoryPath: string): Promise<void> {
  await rm(directoryPath, { recursive: true, force: true });
  await mkdir(directoryPath, { recursive: true });
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
