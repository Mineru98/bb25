/**
 * Vector similarity calibration via a likelihood-ratio transform.
 *
 * Inputs are distances where smaller values indicate closer vectors. The
 * transform estimates a background density f_G and a local/relevant density
 * f_R, then returns sigmoid(log(f_R / f_G) + logit(baseRate)).
 */

import { EPSILON, logit, safeProb, sigmoid } from "./mathUtils.js";

export type VectorProbabilityMethod = "auto" | "kde" | "gmm";

export interface VectorProbabilityOptions {
  weights?: number[];
  method?: VectorProbabilityMethod;
  bandwidthFactor?: number;
  densityPrior?: number[];
}

export interface VectorGmmOptions {
  maxIter?: number;
  tol?: number;
  evalPoints?: number[];
}

export function gaussianPDF(x: number, mu: number, sigma: number): number;
export function gaussianPDF(x: number[], mu: number, sigma: number): number[];
export function gaussianPDF(x: number | number[], mu: number, sigma: number): number | number[] {
  const safeSigma = Math.max(sigma, EPSILON);
  const coeff = 1.0 / (safeSigma * Math.sqrt(2.0 * Math.PI));
  const evalOne = (value: number) => {
    const z = (value - mu) / safeSigma;
    return coeff * Math.exp(-0.5 * z * z);
  };
  return Array.isArray(x) ? x.map(evalOne) : evalOne(x);
}

export function silvermanBandwidth(distances: number[], weights?: number[]): number {
  if (distances.length === 0) {
    return EPSILON;
  }
  const w = weights ?? distances.map(() => 1.0);
  const n = Math.min(distances.length, w.length);
  let wSum = 0.0;
  let wSqSum = 0.0;
  for (let i = 0; i < n; i++) {
    const wi = Math.max(w[i]!, 0.0);
    wSum += wi;
    wSqSum += wi * wi;
  }
  if (wSum < EPSILON || wSqSum < EPSILON) {
    return EPSILON;
  }

  const kEff = (wSum * wSum) / wSqSum;
  let wMean = 0.0;
  for (let i = 0; i < n; i++) {
    wMean += Math.max(w[i]!, 0.0) * distances[i]!;
  }
  wMean /= wSum;

  let wVar = 0.0;
  for (let i = 0; i < n; i++) {
    const diff = distances[i]! - wMean;
    wVar += Math.max(w[i]!, 0.0) * diff * diff;
  }
  const sigmaW = Math.sqrt(Math.max(wVar / wSum, 0.0));
  if (sigmaW < EPSILON) {
    return EPSILON;
  }
  return Math.max(1.06 * sigmaW * Math.pow(kEff, -0.2), EPSILON);
}

export function kernelDensity(
  evalPoints: number[],
  samplePoints: number[],
  weights: number[],
  bandwidth: number,
): number[] {
  const n = Math.min(samplePoints.length, weights.length);
  if (n === 0) {
    return evalPoints.map(() => EPSILON);
  }
  let wSum = 0.0;
  for (let i = 0; i < n; i++) {
    wSum += Math.max(weights[i]!, 0.0);
  }
  if (wSum < EPSILON) {
    return evalPoints.map(() => EPSILON);
  }

  const h = Math.max(bandwidth, EPSILON);
  const coeff = 1.0 / (h * Math.sqrt(2.0 * Math.PI));
  return evalPoints.map((point) => {
    let density = 0.0;
    for (let i = 0; i < n; i++) {
      const diff = (point - samplePoints[i]!) / h;
      density += Math.max(weights[i]!, 0.0) * coeff * Math.exp(-0.5 * diff * diff);
    }
    return Math.max(density / wSum, EPSILON);
  });
}

export class VectorProbabilityTransform {
  public readonly muG: number;
  public readonly sigmaG: number;
  public readonly baseRate: number | null;
  private readonly logitBaseRateValue: number;

  constructor(muG: number, sigmaG: number, baseRate: number | null = null) {
    if (!(sigmaG > 0.0)) {
      throw new Error(`sigmaG must be positive, got ${sigmaG}`);
    }
    if (baseRate !== null && !(baseRate > 0.0 && baseRate < 1.0)) {
      throw new Error(`baseRate must be in (0, 1), got ${baseRate}`);
    }
    this.muG = muG;
    this.sigmaG = Math.max(sigmaG, EPSILON);
    this.baseRate = baseRate;
    this.logitBaseRateValue = baseRate === null ? 0.0 : logit(baseRate);
  }

  static fitBackground(
    distances: number[],
    options: { baseRate?: number | null } = {},
  ): VectorProbabilityTransform {
    if (distances.length === 0) {
      return new VectorProbabilityTransform(0.0, 1.0, options.baseRate ?? null);
    }
    const muG = mean(distances);
    const sigmaG = Math.max(std(distances, muG), EPSILON);
    return new VectorProbabilityTransform(muG, sigmaG, options.baseRate ?? null);
  }

  detectGap(distances: number[], thresholdRatio = 0.15): number | null {
    if (distances.length < 3) {
      return null;
    }
    const sorted = distances.slice().sort((a, b) => a - b);
    const totalSpan = sorted[sorted.length - 1]! - sorted[0]!;
    if (totalSpan < EPSILON) {
      return null;
    }

    const gaps: number[] = [];
    for (let i = 0; i < sorted.length - 1; i++) {
      gaps.push(sorted[i + 1]! - sorted[i]!);
    }

    let maxRatio = 0.0;
    let maxRatioIdx = 0;
    for (let i = 0; i < gaps.length; i++) {
      const ratio = gaps[i]! / totalSpan;
      if (ratio > maxRatio) {
        maxRatio = ratio;
        maxRatioIdx = i;
      }
    }
    if (maxRatio >= thresholdRatio) {
      return maxRatioIdx + 1;
    }

    const meanGap = mean(gaps);
    const stdGap = std(gaps, meanGap);
    if (stdGap < EPSILON) {
      return null;
    }

    let maxZ = Number.NEGATIVE_INFINITY;
    let maxZIdx = 0;
    for (let i = 0; i < gaps.length; i++) {
      const z = (gaps[i]! - meanGap) / stdGap;
      if (z > maxZ) {
        maxZ = z;
        maxZIdx = i;
      }
    }
    return maxZ > 2.0 ? maxZIdx + 1 : null;
  }

  gapWeights(distances: number[]): number[] | null {
    const gapIdx = this.detectGap(distances);
    if (gapIdx === null) {
      return null;
    }
    const sorted = distances.slice().sort((a, b) => a - b);
    const threshold = sorted[gapIdx]!;
    return distances.map((distance) => (distance < threshold ? 1.0 : 0.0));
  }

  static sharpenWeights(weights: number[], temperature = 0.05): number[] {
    if (weights.length === 0) {
      return [];
    }
    const temp = Math.max(temperature, EPSILON);
    let totalMass = 0.0;
    let maxWeight = Number.NEGATIVE_INFINITY;
    for (const weight of weights) {
      totalMass += Math.max(weight, 0.0);
      maxWeight = Math.max(maxWeight, weight);
    }
    const sharpened = weights.map((weight) => Math.exp((weight - maxWeight) / temp));
    const sharpSum = sharpened.reduce((sum, value) => sum + value, 0.0);
    if (sharpSum < EPSILON) {
      return sharpened;
    }
    const scale = totalMass / sharpSum;
    return sharpened.map((value) => value * scale);
  }

  static distanceDensityWeights(distances: number[]): number[] {
    if (distances.length === 0) {
      return [];
    }
    const sorted = distances.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median =
      sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2.0 : sorted[mid]!;
    return distances.map((distance) => sigmoid(median / Math.max(distance, EPSILON) - 1.0));
  }

  estimateKDE(
    distances: number[],
    weights: number[],
    bandwidthFactor = 2.0,
    options: { evalPoints?: number[] } = {},
  ): number[] {
    const evalPoints = options.evalPoints ?? distances;
    const h = silvermanBandwidth(distances, weights) * bandwidthFactor;
    return kernelDensity(evalPoints, distances, weights, h);
  }

  estimateGMM(distances: number[], weights?: number[] | null, options: VectorGmmOptions = {}): number[] {
    const evalPoints = options.evalPoints ?? distances;
    if (distances.length === 0) {
      return evalPoints.map(() => EPSILON);
    }
    const maxIter = options.maxIter ?? 100;
    const tol = options.tol ?? 1e-6;
    const n = distances.length;

    let muR: number;
    let sigmaR: number;
    let piR: number;
    const weightMass = signalMass(weights);
    if (weights !== undefined && weights !== null && weightMass > EPSILON) {
      let weightedMean = 0.0;
      for (let i = 0; i < n; i++) {
        weightedMean += Math.max(weights[i] ?? 0.0, 0.0) * distances[i]!;
      }
      muR = weightedMean / weightMass;

      let weightedVar = 0.0;
      for (let i = 0; i < n; i++) {
        const diff = distances[i]! - muR;
        weightedVar += Math.max(weights[i] ?? 0.0, 0.0) * diff * diff;
      }
      sigmaR = Math.sqrt(weightedVar / weightMass);
      piR = Math.max(0.1, Math.min(0.9, weightMass / n));
    } else {
      muR = this.muG - 0.5 * this.sigmaG;
      sigmaR = this.sigmaG * 0.5;
      piR = 0.3;
    }
    sigmaR = Math.max(sigmaR, this.sigmaG * 0.1, EPSILON);

    let prevLogLikelihood = Number.NEGATIVE_INFINITY;
    for (let iter = 0; iter < maxIter; iter++) {
      const gamma: number[] = [];
      let logLikelihood = 0.0;
      for (const distance of distances) {
        const fR = piR * gaussianPDF(distance, muR, sigmaR);
        const fG = (1.0 - piR) * gaussianPDF(distance, this.muG, this.sigmaG);
        const total = Math.max(fR + fG, EPSILON);
        gamma.push(fR / total);
        logLikelihood += Math.log(total);
      }
      if (Math.abs(logLikelihood - prevLogLikelihood) < tol) {
        break;
      }
      prevLogLikelihood = logLikelihood;

      const gammaSum = gamma.reduce((sum, value) => sum + value, 0.0);
      if (gammaSum < EPSILON) {
        break;
      }
      muR = gamma.reduce((sum, value, i) => sum + value * distances[i]!, 0.0) / gammaSum;
      const variance =
        gamma.reduce((sum, value, i) => {
          const diff = distances[i]! - muR;
          return sum + value * diff * diff;
        }, 0.0) / gammaSum;
      sigmaR = Math.max(Math.sqrt(variance), this.sigmaG * 0.1, EPSILON);
      piR = Math.max(0.01, Math.min(0.99, gammaSum / n));
    }

    return evalPoints.map((distance) => Math.max(gaussianPDF(distance, muR, sigmaR), EPSILON));
  }

  logDensityRatio(distances: number, fRValues: number): number;
  logDensityRatio(distances: number[], fRValues: number[]): number[];
  logDensityRatio(distances: number | number[], fRValues: number | number[]): number | number[] {
    if (Array.isArray(distances) && Array.isArray(fRValues)) {
      const fG = gaussianPDF(distances, this.muG, this.sigmaG) as number[];
      return distances.map((_, i) => Math.log(Math.max(fRValues[i]!, EPSILON) / Math.max(fG[i]!, EPSILON)));
    }
    const fG = gaussianPDF(distances as number, this.muG, this.sigmaG) as number;
    return Math.log(Math.max(fRValues as number, EPSILON) / Math.max(fG, EPSILON));
  }

  calibrate(distances: number, options?: VectorProbabilityOptions): number;
  calibrate(distances: number[], options?: VectorProbabilityOptions): number[];
  calibrate(distances: number | number[], options: VectorProbabilityOptions = {}): number | number[] {
    const scalar = !Array.isArray(distances);
    const arr = scalar ? [distances as number] : (distances as number[]);
    const fR = this.estimateRelevantDensity(arr, arr, options);
    const logRatio = this.logDensityRatio(arr, fR) as number[];
    const out = logRatio.map((value) => safeProb(sigmoid(value + this.logitBaseRateValue)));
    return scalar ? out[0]! : out;
  }

  calibrateWithSample(
    evalDistances: number,
    sampleDistances: number[],
    options?: VectorProbabilityOptions,
  ): number;
  calibrateWithSample(
    evalDistances: number[],
    sampleDistances: number[],
    options?: VectorProbabilityOptions,
  ): number[];
  calibrateWithSample(
    evalDistances: number | number[],
    sampleDistances: number[],
    options: VectorProbabilityOptions = {},
  ): number | number[] {
    const scalar = !Array.isArray(evalDistances);
    const evalArr = scalar ? [evalDistances as number] : (evalDistances as number[]);
    const fR = this.estimateRelevantDensity(evalArr, sampleDistances, options);
    const logRatio = this.logDensityRatio(evalArr, fR) as number[];
    const out = logRatio.map((value) => safeProb(sigmoid(value + this.logitBaseRateValue)));
    return scalar ? out[0]! : out;
  }

  private estimateRelevantDensity(
    evalPoints: number[],
    sampleDistances: number[],
    options: VectorProbabilityOptions,
  ): number[] {
    if (sampleDistances.length === 0) {
      return evalPoints.map(() => EPSILON);
    }
    const method = options.method ?? "auto";
    const bandwidthFactor = options.bandwidthFactor ?? 2.0;
    const weights = options.weights;
    const densityPrior = options.densityPrior;
    const weightMass = signalMass(weights);
    const densityMass = signalMass(densityPrior);

    if (method === "auto") {
      const gapWeights = this.gapWeights(sampleDistances);
      if (gapWeights !== null) {
        if (sampleDistances.length >= 50) {
          return this.estimateKDE(sampleDistances, gapWeights, bandwidthFactor, { evalPoints });
        }
        return this.estimateGMM(sampleDistances, gapWeights, { evalPoints });
      }
      if (weights !== undefined && weightMass > EPSILON) {
        return this.estimateKDE(sampleDistances, VectorProbabilityTransform.sharpenWeights(weights), bandwidthFactor, {
          evalPoints,
        });
      }
      if (densityPrior !== undefined && densityMass > EPSILON) {
        return this.estimateGMM(sampleDistances, densityPrior, { evalPoints });
      }
      return this.estimateGMM(sampleDistances, VectorProbabilityTransform.distanceDensityWeights(sampleDistances), {
        evalPoints,
      });
    }

    if (method === "kde") {
      const effectiveWeights =
        weights !== undefined && weightMass > EPSILON
          ? weights
          : densityPrior !== undefined && densityMass > EPSILON
            ? densityPrior
            : (this.gapWeights(sampleDistances) ??
              VectorProbabilityTransform.distanceDensityWeights(sampleDistances));
      return this.estimateKDE(sampleDistances, effectiveWeights, bandwidthFactor, { evalPoints });
    }

    if (method === "gmm") {
      const effectiveWeights =
        weights !== undefined && weightMass > EPSILON
          ? weights
          : densityPrior !== undefined && densityMass > EPSILON
            ? densityPrior
            : null;
      return this.estimateGMM(sampleDistances, effectiveWeights, { evalPoints });
    }

    throw new Error(`method must be 'auto', 'kde', or 'gmm', got '${method}'`);
  }
}

export function ivfDensityPrior(
  cellPopulation: number,
  avgPopulation: number,
  options?: { gamma?: number },
): number;
export function ivfDensityPrior(
  cellPopulation: number[],
  avgPopulation: number,
  options?: { gamma?: number },
): number[];
export function ivfDensityPrior(
  cellPopulation: number | number[],
  avgPopulation: number,
  options: { gamma?: number } = {},
): number | number[] {
  const gamma = options.gamma ?? 1.0;
  const evalOne = (population: number) => {
    const ratio = avgPopulation / Math.max(population, EPSILON) - 1.0;
    return sigmoid(gamma * ratio);
  };
  return Array.isArray(cellPopulation) ? cellPopulation.map(evalOne) : evalOne(cellPopulation);
}

export function knnDensityPrior(
  kthDistance: number,
  globalMedianKth: number,
  options?: { gamma?: number },
): number;
export function knnDensityPrior(
  kthDistance: number[],
  globalMedianKth: number,
  options?: { gamma?: number },
): number[];
export function knnDensityPrior(
  kthDistance: number | number[],
  globalMedianKth: number,
  options: { gamma?: number } = {},
): number | number[] {
  const gamma = options.gamma ?? 1.0;
  const median = Math.max(globalMedianKth, EPSILON);
  const evalOne = (distance: number) => sigmoid(gamma * (distance / median - 1.0));
  return Array.isArray(kthDistance) ? kthDistance.map(evalOne) : evalOne(kthDistance);
}

function signalMass(weights: number[] | null | undefined): number {
  if (weights === null || weights === undefined || weights.length === 0) {
    return 0.0;
  }
  return weights.reduce((sum, value) => sum + Math.max(value, 0.0), 0.0);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0.0) / values.length;
}

function std(values: number[], mu = mean(values)): number {
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mu) ** 2, 0.0) / values.length);
}
