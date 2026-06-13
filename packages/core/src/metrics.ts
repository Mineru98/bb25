/**
 * Calibration metrics for evaluating probability quality.
 *
 * Direct port of `src/metrics.rs`. Accumulations follow the reference's
 * left-to-right order so results match Rust bit-for-bit.
 */

/** A single bin in a reliability diagram: [avgPredicted, avgActual, count]. */
export type ReliabilityBin = [number, number, number];

/**
 * Expected Calibration Error (ECE).
 *
 * Measures how well predicted probabilities match actual relevance rates.
 * Lower is better. Perfect calibration = 0.
 */
export function expectedCalibrationError(
  probabilities: number[],
  labels: number[],
  nBins: number,
): number {
  const total = probabilities.length;
  let ece = 0.0;

  for (let binIdx = 0; binIdx < nBins; binIdx++) {
    const lo = binIdx / nBins;
    const hi = (binIdx + 1) / nBins;

    let sumProb = 0.0;
    let sumLabel = 0.0;
    let count = 0;

    for (let i = 0; i < probabilities.length; i++) {
      const p = probabilities[i] as number;
      const inBin = binIdx === 0 ? p >= lo && p <= hi : p > lo && p <= hi;
      if (inBin) {
        sumProb += p;
        sumLabel += labels[i] as number;
        count += 1;
      }
    }

    if (count === 0) {
      continue;
    }

    const avgProb = sumProb / count;
    const avgLabel = sumLabel / count;
    ece += (count / total) * Math.abs(avgProb - avgLabel);
  }

  return ece;
}

/**
 * Brier score: mean squared error between probabilities and labels.
 *
 * Decomposes into calibration + discrimination. Lower is better.
 */
export function brierScore(probabilities: number[], labels: number[]): number {
  const n = probabilities.length;
  let sum = 0.0;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i] as number;
    const y = labels[i] as number;
    sum += (p - y) * (p - y);
  }
  return sum / n;
}

/**
 * Compute reliability diagram data: [avgPredicted, avgActual, count] per bin.
 *
 * Perfect calibration means avgPredicted == avgActual for every bin.
 */
export function reliabilityDiagram(
  probabilities: number[],
  labels: number[],
  nBins: number,
): ReliabilityBin[] {
  const bins: ReliabilityBin[] = [];

  for (let binIdx = 0; binIdx < nBins; binIdx++) {
    const lo = binIdx / nBins;
    const hi = (binIdx + 1) / nBins;

    let sumProb = 0.0;
    let sumLabel = 0.0;
    let count = 0;

    for (let i = 0; i < probabilities.length; i++) {
      const p = probabilities[i] as number;
      const inBin = binIdx === 0 ? p >= lo && p <= hi : p > lo && p <= hi;
      if (inBin) {
        sumProb += p;
        sumLabel += labels[i] as number;
        count += 1;
      }
    }

    if (count > 0) {
      bins.push([sumProb / count, sumLabel / count, count]);
    }
  }

  return bins;
}

/** One-call calibration diagnostic report. */
export interface CalibrationReport {
  ece: number;
  brier: number;
  reliability: ReliabilityBin[];
  nSamples: number;
  nBins: number;
}

/** Compute a full calibration diagnostic report in one call. */
export function calibrationReport(
  probabilities: number[],
  labels: number[],
  nBins: number,
): CalibrationReport {
  return {
    ece: expectedCalibrationError(probabilities, labels, nBins),
    brier: brierScore(probabilities, labels),
    reliability: reliabilityDiagram(probabilities, labels, nBins),
    nSamples: probabilities.length,
    nBins,
  };
}

/**
 * Formatted text summary of a calibration report.
 * Port of `CalibrationReport::summary` (`src/metrics.rs`): Rust uses `{:>10.4}`
 * (right-aligned, width 10, 4 decimals) and `{:.6}` for ECE/Brier.
 */
export function summarizeCalibration(report: CalibrationReport): string {
  const lines: string[] = [
    "Calibration Report",
    "==================",
    `  Samples : ${report.nSamples}`,
    `  Bins    : ${report.nBins}`,
    `  ECE     : ${report.ece.toFixed(6)}`,
    `  Brier   : ${report.brier.toFixed(6)}`,
    "",
    "  Reliability Diagram",
    "  -------------------",
    `  ${"Predicted".padStart(10)}  ${"Actual".padStart(10)}  ${"Count".padStart(6)}`,
  ];
  for (const [avgPred, avgActual, count] of report.reliability) {
    lines.push(
      `  ${avgPred.toFixed(4).padStart(10)}  ${avgActual.toFixed(4).padStart(10)}  ${String(count).padStart(6)}`,
    );
  }
  return lines.join("\n");
}
