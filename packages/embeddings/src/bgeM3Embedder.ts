/**
 * BGE-M3 (dense) embedder via transformers.js (@huggingface/transformers).
 *
 * Implements the `Embedder` contract from @bb25/core. This is the only place in
 * the project that depends on the ONNX runtime — @bb25/core stays dependency-free.
 *
 * dense-only: BGE-M3 sparse / ColBERT multi-vector outputs are out of scope.
 */
import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
  type ProgressCallback,
} from "@huggingface/transformers";
import type { Embedder } from "@bb25/core";

export type Dtype = "fp32" | "fp16" | "q8" | "q4";
export type Pooling = "cls" | "mean";
export type Device = "cpu" | "wasm" | "webgpu" | "auto";

export interface BgeM3Options {
  /** HF model id (default: "Xenova/bge-m3"). */
  model?: string;
  /** Quantization / precision (default: "fp32" for parity-grade output). */
  dtype?: Dtype;
  /** Pooling strategy (default: "cls", as BGE-M3 uses CLS pooling). */
  pooling?: Pooling;
  /** L2-normalize the output embeddings (default: true). */
  normalize?: boolean;
  /** Execution backend. Left undefined => transformers.js default for the runtime. */
  device?: Device;
  /** Local model cache directory (Node). */
  cacheDir?: string;
  /** Only load from local cache; never hit the network. */
  localOnly?: boolean;
  /** Batch size for `embed` (default: 16). */
  batchSize?: number;
  /** Optional load-progress callback. */
  onProgress?: ProgressCallback;
}

const DIM = 1024;

export class BgeM3Embedder implements Embedder {
  readonly dim = DIM;

  private readonly model: string;
  private readonly dtype: Dtype;
  private readonly pooling: Pooling;
  private readonly normalize: boolean;
  private readonly device: Device | undefined;
  private readonly cacheDir: string | undefined;
  private readonly localOnly: boolean;
  private readonly batchSize: number;
  private readonly onProgress: ProgressCallback | undefined;

  private extractor: FeatureExtractionPipeline | null = null;
  private loading: Promise<FeatureExtractionPipeline> | null = null;

  constructor(options: BgeM3Options = {}) {
    this.model = options.model ?? "Xenova/bge-m3";
    this.dtype = options.dtype ?? "fp32";
    this.pooling = options.pooling ?? "cls";
    this.normalize = options.normalize ?? true;
    this.device = options.device;
    this.cacheDir = options.cacheDir;
    this.localOnly = options.localOnly ?? false;
    this.batchSize = options.batchSize ?? 16;
    this.onProgress = options.onProgress;
  }

  /** Load (and cache) the underlying feature-extraction pipeline. Idempotent. */
  async warmup(): Promise<void> {
    await this.getExtractor();
  }

  private async getExtractor(): Promise<FeatureExtractionPipeline> {
    if (this.extractor !== null) {
      return this.extractor;
    }
    if (this.loading === null) {
      // transformers.js reads cache/offline settings from the `env` singleton; we
      // pass per-call options it supports and configure env in `configureEnv`.
      configureEnv(this.cacheDir, this.localOnly);
      this.loading = pipeline("feature-extraction", this.model, {
        dtype: this.dtype,
        ...(this.device !== undefined ? { device: this.device } : {}),
        ...(this.onProgress !== undefined ? { progress_callback: this.onProgress } : {}),
      }) as Promise<FeatureExtractionPipeline>;
    }
    this.extractor = await this.loading;
    return this.extractor;
  }

  /** Embed a batch of texts into dense 1024-d vectors. */
  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return [];
    }
    const extractor = await this.getExtractor();
    const out: Float32Array[] = [];

    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const tensor = await extractor(batch, {
        pooling: this.pooling,
        normalize: this.normalize,
      });
      // tensor.dims === [batch, DIM]; tensor.data is a flat Float32Array.
      const [rows, cols] = tensor.dims as [number, number];
      const data = tensor.data as Float32Array;
      for (let r = 0; r < rows; r++) {
        out.push(data.slice(r * cols, r * cols + cols));
      }
    }

    return out;
  }
}

/** Configure the transformers.js `env` singleton (cache dir + offline mode). */
function configureEnv(cacheDir: string | undefined, localOnly: boolean): void {
  if (cacheDir !== undefined) {
    env.cacheDir = cacheDir;
  }
  if (localOnly) {
    // Never hit the network; require the model to be present in the local cache.
    env.allowRemoteModels = false;
  }
}
