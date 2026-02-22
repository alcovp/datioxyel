export type Vec3 = [number, number, number];

export type RenderQuality = "preview" | "balanced" | "final";
export type SamplingMode = "random" | "halton" | "sobol";

export type RenderFrameConfig = {
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
  quality: RenderQuality;
  marchMaxSteps?: number;
  rrStartBounce?: number;
  samplingMode: SamplingMode;
};

export type RenderBatchConfig = {
  frames: RenderFrameConfig[];
};

export type ParsedArgs = {
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
  quality: RenderQuality;
  marchMaxSteps?: number;
  rrStartBounce?: number;
  samplingMode: SamplingMode;
  batchWorkers: number;
};
