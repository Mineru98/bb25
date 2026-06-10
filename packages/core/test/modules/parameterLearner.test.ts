import { describe, it } from "vitest";
import { goldenModules, expectClose } from "./_modules.js";
import { ParameterLearner } from "../../src/parameterLearner.js";

describe("ParameterLearner golden parity", () => {
  for (const c of goldenModules.parameterLearner) {
    it(`learn matches Rust for ${c.name}`, () => {
      const learner = new ParameterLearner(c.lr, c.maxIter, c.tol);
      const result = learner.learn(c.scores, c.labels);

      expectClose(result.alpha, c.alpha, `${c.name}.alpha`);
      expectClose(result.beta, c.beta, `${c.name}.beta`);

      if (result.converged !== c.converged) {
        throw new Error(
          `${c.name}.converged: got ${result.converged}, expected ${c.converged}`,
        );
      }

      if (result.lossHistory.length !== c.lossHistory.length) {
        throw new Error(
          `${c.name}.lossHistory.length: got ${result.lossHistory.length}, expected ${c.lossHistory.length}`,
        );
      }

      for (let i = 0; i < c.lossHistory.length; i++) {
        expectClose(
          result.lossHistory[i] as number,
          c.lossHistory[i] as number,
          `${c.name}.lossHistory[${i}]`,
        );
      }
    });

    it(`crossEntropyLoss matches Rust for ${c.name}`, () => {
      const learner = new ParameterLearner(c.lr, c.maxIter, c.tol);
      for (let i = 0; i < c.crossEntropy.length; i++) {
        const probe = c.crossEntropy[i]!;
        const loss = learner.crossEntropyLoss(
          c.scores,
          c.labels,
          probe.alpha,
          probe.beta,
        );
        expectClose(loss, probe.loss, `${c.name}.crossEntropy[${i}]`);
      }
    });
  }
});
