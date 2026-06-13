/**
 * Numeric primitives. Direct port of `src/math_utils.rs`.
 *
 * JS `number` is IEEE-754 f64, identical to Rust `f64`. Every accumulation
 * follows the reference's left-to-right order so results match bit-for-bit.
 */

export type Vector = Float32Array | number[];

export const EPSILON = 1e-10;

/** Branchful sigmoid for numerical stability (matches `math_utils::sigmoid`). */
export function sigmoid(x: number): number {
  if (x >= 0.0) {
    const ez = Math.exp(-x);
    return 1.0 / (1.0 + ez);
  } else {
    const ez = Math.exp(x);
    return ez / (1.0 + ez);
  }
}

export function safeLog(p: number): number {
  return Math.log(Math.max(p, EPSILON));
}

export function logit(p: number): number {
  const c = clamp(p, EPSILON, 1.0 - EPSILON);
  return Math.log(c / (1.0 - c));
}

export function safeProb(p: number): number {
  return clamp(p, EPSILON, 1.0 - EPSILON);
}

export function clamp(value: number, low: number, high: number): number {
  if (value < low) {
    return low;
  } else if (value > high) {
    return high;
  } else {
    return value;
  }
}

export function dotProduct(a: Vector, b: Vector): number {
  // Rust: a.iter().zip(b.iter()) -> iterates min(len_a, len_b).
  const n = Math.min(a.length, b.length);
  let sum = 0.0;
  for (let i = 0; i < n; i++) {
    sum += (a[i] as number) * (b[i] as number);
  }
  return sum;
}

export function vectorMagnitude(v: Vector): number {
  let sum = 0.0;
  for (let i = 0; i < v.length; i++) {
    const vi = v[i] as number;
    sum += vi * vi;
  }
  return Math.sqrt(sum);
}

export function cosineSimilarity(a: Vector, b: Vector): number {
  const magA = vectorMagnitude(a);
  const magB = vectorMagnitude(b);
  if (magA < EPSILON || magB < EPSILON) {
    return 0.0;
  }
  return dotProduct(a, b) / (magA * magB);
}

/** Numerically stable softmax over a 1D slice. */
export function softmax(z: number[]): number[] {
  if (z.length === 0) {
    return [];
  }
  let maxZ = Number.NEGATIVE_INFINITY;
  for (const v of z) {
    maxZ = Math.max(maxZ, v);
  }
  const expZ = z.map((v) => Math.exp(v - maxZ));
  let sum = 0.0;
  for (const e of expZ) {
    sum += e;
  }
  return expZ.map((e) => e / sum);
}

/** Row-wise softmax over a 2D array stored as a flat slice (each row `nCols`). */
export function softmaxRows(z: number[], nCols: number): number[] {
  const nRows = Math.floor(z.length / nCols);
  const result = new Array<number>(z.length).fill(0.0);
  for (let r = 0; r < nRows; r++) {
    const start = r * nCols;
    const end = start + nCols;
    const row = z.slice(start, end);
    const sm = softmax(row);
    for (let i = 0; i < nCols; i++) {
      result[start + i] = sm[i]!;
    }
  }
  return result;
}

/** Min-max normalize to [0, 1]. Returns zeros when the range is negligible. */
export function minMaxNormalize(values: number[]): number[] {
  if (values.length === 0) {
    return [];
  }
  let minVal = Number.POSITIVE_INFINITY;
  let maxVal = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    minVal = Math.min(minVal, v);
    maxVal = Math.max(maxVal, v);
  }
  const range = maxVal - minVal;
  if (range < 1e-12) {
    return values.map(() => 0.0);
  }
  return values.map((v) => (v - minVal) / range);
}
