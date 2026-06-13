/**
 * FusionDebugger. Direct port of `src/debug.rs`.
 *
 * Traces intermediate values through the Bayesian BM25 fusion pipeline. The
 * numeric trace fields follow the reference exactly (golden-tested); the
 * `format*` helpers are faithful text renderings (not numerically golden-tested).
 */
import { cosineToProbability, probNot } from "./fusion.js";
import { logit, safeProb, sigmoid } from "./mathUtils.js";
import { BayesianProbabilityTransform } from "./probability.js";

export interface BM25SignalTrace {
  rawScore: number;
  tf: number;
  docLenRatio: number;
  likelihood: number;
  tfPrior: number;
  normPrior: number;
  compositePrior: number;
  logitLikelihood: number;
  logitPrior: number;
  logitBaseRate: number | null;
  posterior: number;
  alpha: number;
  beta: number;
  baseRate: number | null;
}

export interface VectorSignalTrace {
  cosineScore: number;
  probability: number;
  logitProbability: number;
}

export interface NotTrace {
  inputProbability: number;
  inputName: string;
  complement: number;
  logitInput: number;
  logitComplement: number;
}

export interface FusionTrace {
  signalProbabilities: number[];
  signalNames: string[];
  method: string;
  logits: number[] | null;
  meanLogit: number | null;
  alpha: number | null;
  nAlphaScale: number | null;
  scaledLogit: number | null;
  weights: number[] | null;
  logProbs: number[] | null;
  logProbSum: number | null;
  complements: number[] | null;
  logComplements: number[] | null;
  logComplementSum: number | null;
  fusedProbability: number;
}

export type SignalTrace =
  | { kind: "bm25"; trace: BM25SignalTrace }
  | { kind: "vector"; trace: VectorSignalTrace };

export interface DocumentTrace {
  docId: string | null;
  signals: [string, SignalTrace][];
  fusion: FusionTrace;
  finalProbability: number;
}

export interface ComparisonResult {
  docA: DocumentTrace;
  docB: DocumentTrace;
  signalDeltas: [string, number][];
  dominantSignal: string;
  crossoverStage: string | null;
}

function emptyFusion(probs: number[], names: string[], method: string, fused: number): FusionTrace {
  return {
    signalProbabilities: probs.slice(),
    signalNames: names.slice(),
    method,
    logits: null,
    meanLogit: null,
    alpha: null,
    nAlphaScale: null,
    scaledLogit: null,
    weights: null,
    logProbs: null,
    logProbSum: null,
    complements: null,
    logComplements: null,
    logComplementSum: null,
    fusedProbability: fused,
  };
}

export class FusionDebugger {
  private readonly transformValue: BayesianProbabilityTransform;

  constructor(transform: BayesianProbabilityTransform) {
    this.transformValue = transform;
  }

  transform(): BayesianProbabilityTransform {
    return this.transformValue;
  }

  /** Trace a single BM25 score through the full probability pipeline. */
  traceBm25(score: number, tf: number, docLenRatio: number): BM25SignalTrace {
    const t = this.transformValue;
    const likelihoodVal = t.likelihood(score);
    const tfPriorVal = BayesianProbabilityTransform.tfPrior(tf);
    const normPriorVal = BayesianProbabilityTransform.normPrior(docLenRatio);
    const compositePriorVal = BayesianProbabilityTransform.compositePrior(tf, docLenRatio);
    const posteriorVal = BayesianProbabilityTransform.posterior(
      likelihoodVal,
      compositePriorVal,
      t.baseRate,
    );

    return {
      rawScore: score,
      tf,
      docLenRatio,
      likelihood: likelihoodVal,
      tfPrior: tfPriorVal,
      normPrior: normPriorVal,
      compositePrior: compositePriorVal,
      logitLikelihood: logit(likelihoodVal),
      logitPrior: logit(compositePriorVal),
      logitBaseRate: t.baseRate !== null ? logit(safeProb(t.baseRate)) : null,
      posterior: posteriorVal,
      alpha: t.alpha,
      beta: t.beta,
      baseRate: t.baseRate,
    };
  }

  /** Trace a cosine similarity through probability conversion. */
  traceVector(cosineScore: number): VectorSignalTrace {
    const probVal = cosineToProbability(cosineScore);
    return {
      cosineScore,
      probability: probVal,
      logitProbability: logit(probVal),
    };
  }

  /** Trace a probabilistic NOT (complement) operation. */
  traceNot(probability: number, name: string): NotTrace {
    const complement = probNot(probability);
    return {
      inputProbability: probability,
      inputName: name,
      complement,
      logitInput: logit(safeProb(probability)),
      logitComplement: logit(safeProb(complement)),
    };
  }

  /** Trace the fusion of multiple probability signals. */
  traceFusion(
    probabilities: number[],
    names: string[] | null,
    method: string,
    alpha: number | null = null,
    weights: number[] | null = null,
  ): FusionTrace {
    const n = probabilities.length;
    const signalNames =
      names !== null ? names.slice() : Array.from({ length: n }, (_, i) => `signal_${i}`);
    const probs = probabilities.map((p) => safeProb(p));

    switch (method) {
      case "log_odds":
        return this.traceLogOdds(probs, signalNames, alpha, weights);
      case "prob_and":
        return this.traceProbAnd(probs, signalNames);
      case "prob_or":
        return this.traceProbOr(probs, signalNames);
      case "prob_not":
        return this.traceProbNotFusion(probs, signalNames);
      default:
        throw new Error(
          `method must be 'log_odds', 'prob_and', 'prob_or', or 'prob_not', got '${method}'`,
        );
    }
  }

  private traceLogOdds(
    probs: number[],
    names: string[],
    alpha: number | null,
    weights: number[] | null,
  ): FusionTrace {
    const n = probs.length;
    const logitsArr = probs.map((p) => logit(p));

    if (weights !== null) {
      const effectiveAlpha = alpha ?? 0.0;
      const nAlphaScale = Math.pow(n, effectiveAlpha);
      let weightedLogit = 0.0;
      for (let i = 0; i < logitsArr.length; i++) {
        weightedLogit += weights[i]! * logitsArr[i]!;
      }
      const scaled = nAlphaScale * weightedLogit;
      const base = emptyFusion(probs, names, "log_odds", sigmoid(scaled));
      return {
        ...base,
        logits: logitsArr,
        meanLogit: weightedLogit,
        alpha: effectiveAlpha,
        nAlphaScale,
        scaledLogit: scaled,
        weights: weights.slice(),
      };
    }

    const effectiveAlpha = alpha ?? 0.5;
    let sum = 0.0;
    for (const l of logitsArr) {
      sum += l;
    }
    const meanLogitVal = sum / n;
    const nAlphaScale = Math.pow(n, effectiveAlpha);
    const scaled = meanLogitVal * nAlphaScale;
    const base = emptyFusion(probs, names, "log_odds", sigmoid(scaled));
    return {
      ...base,
      logits: logitsArr,
      meanLogit: meanLogitVal,
      alpha: effectiveAlpha,
      nAlphaScale,
      scaledLogit: scaled,
    };
  }

  private traceProbAnd(probs: number[], names: string[]): FusionTrace {
    const logProbs = probs.map((p) => Math.log(p));
    let logSum = 0.0;
    for (const lp of logProbs) {
      logSum += lp;
    }
    const base = emptyFusion(probs, names, "prob_and", Math.exp(logSum));
    return { ...base, logProbs, logProbSum: logSum };
  }

  private traceProbOr(probs: number[], names: string[]): FusionTrace {
    const comps = probs.map((p) => 1.0 - p);
    const logComps = comps.map((c) => Math.log(c));
    let logSum = 0.0;
    for (const lc of logComps) {
      logSum += lc;
    }
    const base = emptyFusion(probs, names, "prob_or", 1.0 - Math.exp(logSum));
    return { ...base, complements: comps, logComplements: logComps, logComplementSum: logSum };
  }

  private traceProbNotFusion(probs: number[], names: string[]): FusionTrace {
    const comps = probs.map((p) => 1.0 - p);
    const logComps = comps.map((c) => Math.log(c));
    let logSum = 0.0;
    for (const lc of logComps) {
      logSum += lc;
    }
    const base = emptyFusion(probs, names, "prob_not", Math.exp(logSum));
    return { ...base, complements: comps, logComplements: logComps, logComplementSum: logSum };
  }

  /** Full pipeline trace for one document. */
  traceDocument(
    bm25Score: number | null,
    tf: number | null,
    docLenRatio: number | null,
    cosineScore: number | null,
    method: string,
    alpha: number | null = null,
    weights: number[] | null = null,
    docId: string | null = null,
  ): DocumentTrace {
    const signals: [string, SignalTrace][] = [];
    const probs: number[] = [];
    const names: string[] = [];

    if (bm25Score !== null) {
      if (tf === null) {
        throw new Error("tf is required when bm25Score is provided");
      }
      if (docLenRatio === null) {
        throw new Error("docLenRatio is required when bm25Score is provided");
      }
      const trace = this.traceBm25(bm25Score, tf, docLenRatio);
      probs.push(trace.posterior);
      names.push("BM25");
      signals.push(["BM25", { kind: "bm25", trace }]);
    }

    if (cosineScore !== null) {
      const trace = this.traceVector(cosineScore);
      probs.push(trace.probability);
      names.push("Vector");
      signals.push(["Vector", { kind: "vector", trace }]);
    }

    if (probs.length === 0) {
      throw new Error("At least one of bm25Score or cosineScore must be provided");
    }

    const fusionTrace = this.traceFusion(probs, names, method, alpha, weights);

    return {
      docId,
      signals,
      fusion: fusionTrace,
      finalProbability: fusionTrace.fusedProbability,
    };
  }

  /** Compare two document traces to explain rank differences. */
  compare(traceA: DocumentTrace, traceB: DocumentTrace): ComparisonResult {
    const allNames: string[] = [];
    const seen = new Set<string>();
    for (const [name] of traceA.signals) {
      if (!seen.has(name)) {
        seen.add(name);
        allNames.push(name);
      }
    }
    for (const [name] of traceB.signals) {
      if (!seen.has(name)) {
        seen.add(name);
        allNames.push(name);
      }
    }

    const signalDeltas: [string, number][] = [];
    for (const name of allNames) {
      const probA = signalProbability(traceA, name);
      const probB = signalProbability(traceB, name);
      signalDeltas.push([name, probA - probB]);
    }

    // Dominant signal: largest absolute delta (first max, matching Rust max_by).
    let dominant = "";
    let bestAbs = Number.NEGATIVE_INFINITY;
    for (const [name, delta] of signalDeltas) {
      const a = Math.abs(delta);
      if (a > bestAbs) {
        bestAbs = a;
        dominant = name;
      }
    }

    const fusedDelta = traceA.finalProbability - traceB.finalProbability;
    let crossoverStage: string | null = null;
    for (const [name, delta] of signalDeltas) {
      if (name === dominant) {
        continue;
      }
      if (
        fusedDelta !== 0.0 &&
        delta !== 0.0 &&
        ((fusedDelta > 0.0 && delta < 0.0) || (fusedDelta < 0.0 && delta > 0.0))
      ) {
        crossoverStage = name;
        break;
      }
    }

    return {
      docA: traceA,
      docB: traceB,
      signalDeltas,
      dominantSignal: dominant,
      crossoverStage,
    };
  }

  /** Format a document trace as human-readable text (best-effort port of format_trace). */
  formatTrace(trace: DocumentTrace, verbose: boolean): string {
    const lines: string[] = [];
    const docLabel = trace.docId ?? "unknown";
    lines.push(`Document: ${docLabel}`);

    for (const [name, sig] of trace.signals) {
      if (sig.kind === "bm25") {
        const s = sig.trace;
        lines.push(
          `  [${name}] raw=${f(s.rawScore, 2)} -> likelihood=${f(s.likelihood, 3)} (alpha=${f(s.alpha, 2)}, beta=${f(s.beta, 2)})`,
        );
        lines.push(`         tf=${f(s.tf, 0)} -> tf_prior=${f(s.tfPrior, 3)}`);
        lines.push(`         dl_ratio=${f(s.docLenRatio, 2)} -> norm_prior=${f(s.normPrior, 3)}`);
        lines.push(`         composite_prior=${f(s.compositePrior, 3)}`);
        if (s.baseRate !== null) {
          const posteriorNoBr = BayesianProbabilityTransform.posterior(
            s.likelihood,
            s.compositePrior,
            null,
          );
          lines.push(`         posterior=${f(posteriorNoBr, 3)}`);
          lines.push(`         with base_rate=${f(s.baseRate, 3)}: posterior=${f(s.posterior, 3)}`);
        } else {
          lines.push(`         posterior=${f(s.posterior, 3)}`);
        }
        if (verbose) {
          lines.push(`         logit(posterior)=${f(logit(safeProb(s.posterior)), 3)}`);
        }
        lines.push("");
      } else {
        const s = sig.trace;
        lines.push(`  [${name}] cosine=${f(s.cosineScore, 3)} -> prob=${f(s.probability, 3)}`);
        if (verbose) {
          lines.push(`           logit(prob)=${f(s.logitProbability, 3)}`);
        }
        lines.push("");
      }
    }

    const fu = trace.fusion;
    const alphaStr = fu.alpha !== null ? `, alpha=${fu.alpha}` : "";
    const nStr = `, n=${fu.signalProbabilities.length}`;
    lines.push(`  [Fusion] method=${fu.method}${alphaStr}${nStr}`);

    if (verbose) {
      if (fu.logits !== null) {
        lines.push(`           logits=[${fu.logits.map((v) => f(v, 3)).join(", ")}]`);
      }
      if (fu.meanLogit !== null) {
        lines.push(`           mean_logit=${f(fu.meanLogit, 3)}`);
      }
      if (fu.nAlphaScale !== null && fu.scaledLogit !== null) {
        lines.push(`           n^alpha=${f(fu.nAlphaScale, 3)}, scaled=${f(fu.scaledLogit, 3)}`);
      }
      if (fu.weights !== null) {
        lines.push(`           weights=[${fu.weights.map((v) => f(v, 3)).join(", ")}]`);
      }
      if (fu.logProbs !== null) {
        lines.push(`           ln(P)=[${fu.logProbs.map((v) => f(v, 3)).join(", ")}]`);
        if (fu.logProbSum !== null) {
          lines.push(`           sum(ln(P))=${f(fu.logProbSum, 3)}`);
        }
      }
      if (fu.complements !== null) {
        lines.push(`           1-P=[${fu.complements.map((v) => f(v, 3)).join(", ")}]`);
      }
      if (fu.logComplements !== null) {
        lines.push(`           ln(1-P)=[${fu.logComplements.map((v) => f(v, 3)).join(", ")}]`);
        if (fu.logComplementSum !== null) {
          lines.push(`           sum(ln(1-P))=${f(fu.logComplementSum, 3)}`);
        }
      }
    }

    lines.push(`           -> final=${f(fu.fusedProbability, 3)}`);
    return lines.join("\n");
  }

  /** Compact one-line summary of a document trace. */
  formatSummary(trace: DocumentTrace): string {
    const docLabel = trace.docId ?? "unknown";
    const parts: string[] = [];
    for (const [, sig] of trace.signals) {
      if (sig.kind === "bm25") {
        parts.push(`BM25=${f(sig.trace.posterior, 3)}`);
      } else {
        parts.push(`Vec=${f(sig.trace.probability, 3)}`);
      }
    }
    const fu = trace.fusion;
    const alphaStr = fu.alpha !== null ? `, alpha=${fu.alpha}` : "";
    return `${docLabel}: ${parts.join(" ")} -> Fused=${f(fu.fusedProbability, 3)} (${fu.method}${alphaStr})`;
  }
}

/** Extract the final probability from a signal within a document trace. */
function signalProbability(trace: DocumentTrace, name: string): number {
  for (const [n, sig] of trace.signals) {
    if (n === name) {
      return sig.kind === "bm25" ? sig.trace.posterior : sig.trace.probability;
    }
  }
  return 0.5; // neutral if signal missing
}

/** Fixed-decimal formatting helper (mirrors Rust `{:.N}`). */
function f(x: number, decimals: number): string {
  return x.toFixed(decimals);
}
