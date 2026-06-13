//! Golden-fixture extractor for the additional core modules (Phase A).
//!
//! Emits a JSON object covering metrics, ParameterLearner, and runExperiments
//! (exp1..13). The TS test-suite asserts numeric/boolean parity against this.
//!
//! Run with: `cargo run --example extract_golden_modules > fixtures/golden_modules.json`

use std::rc::Rc;

use bayesian_bm25::{
    brier_score, build_default_corpus, build_default_queries, expected_calibration_error,
    reliability_diagram, BM25Scorer, ExperimentRunner, ParameterLearner, Query,
};

fn jf(x: f64) -> String {
    format!("{:?}", x)
}

fn jstr(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jarr_f64(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|&x| jf(x)).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let mut out = String::new();
    out.push('{');

    // -----------------------------------------------------------------------
    // metrics (ECE, Brier, reliability diagram)
    // -----------------------------------------------------------------------
    let metric_cases: Vec<(Vec<f64>, Vec<f64>, usize)> = vec![
        (
            vec![0.1, 0.2, 0.35, 0.4, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95],
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            10,
        ),
        (
            vec![0.05, 0.05, 0.5, 0.5, 0.95, 0.95],
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            5,
        ),
        (
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
            4,
        ),
    ];
    let mut metric_objs: Vec<String> = Vec::new();
    for (probs, labels, n_bins) in &metric_cases {
        let ece = expected_calibration_error(probs, labels, *n_bins);
        let brier = brier_score(probs, labels);
        let rel = reliability_diagram(probs, labels, *n_bins);
        let rel_objs: Vec<String> = rel
            .iter()
            .map(|(p, a, c)| format!("[{},{},{}]", jf(*p), jf(*a), c))
            .collect();
        metric_objs.push(format!(
            "{{\"probs\":{},\"labels\":{},\"nBins\":{},\"ece\":{},\"brier\":{},\"reliability\":[{}]}}",
            jarr_f64(probs),
            jarr_f64(labels),
            n_bins,
            jf(ece),
            jf(brier),
            rel_objs.join(",")
        ));
    }
    out.push_str(&format!("\"metrics\":[{}],", metric_objs.join(",")));

    // -----------------------------------------------------------------------
    // ParameterLearner — replicate exp9 input (query[0] over all docs)
    // and a couple of synthetic cases. Dump full loss history for trajectory parity.
    // -----------------------------------------------------------------------
    let corpus = Rc::new(build_default_corpus());
    let queries = build_default_queries();
    let bm25 = Rc::new(BM25Scorer::new(Rc::clone(&corpus), 1.2, 0.75));

    // exp9 input
    let q0 = &queries[0];
    let relevant: std::collections::HashSet<String> = q0.relevant.iter().cloned().collect();
    let mut exp9_scores: Vec<f64> = Vec::new();
    let mut exp9_labels: Vec<f64> = Vec::new();
    for doc in corpus.documents() {
        exp9_scores.push(bm25.score(&q0.terms, doc));
        exp9_labels.push(if relevant.contains(&doc.id) { 1.0 } else { 0.0 });
    }

    let synthetic_scores = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let synthetic_labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    struct PlCase {
        name: &'static str,
        scores: Vec<f64>,
        labels: Vec<f64>,
        lr: f64,
        max_iter: usize,
        tol: f64,
    }
    let pl_cases = vec![
        PlCase { name: "exp9", scores: exp9_scores, labels: exp9_labels, lr: 0.1, max_iter: 500, tol: 1e-8 },
        PlCase { name: "synthetic_default", scores: synthetic_scores.clone(), labels: synthetic_labels.clone(), lr: 0.01, max_iter: 1000, tol: 1e-6 },
        PlCase { name: "synthetic_fast", scores: synthetic_scores, labels: synthetic_labels, lr: 0.5, max_iter: 200, tol: 1e-9 },
    ];

    let mut pl_objs: Vec<String> = Vec::new();
    for c in &pl_cases {
        let learner = ParameterLearner::new(c.lr, c.max_iter, c.tol);
        let res = learner.learn(&c.scores, &c.labels);
        // a few cross-entropy probes
        let ce_probes = [(1.0, 0.0), (1.5, 0.5), (0.5, 1.0)];
        let ce_objs: Vec<String> = ce_probes
            .iter()
            .map(|&(a, b)| {
                format!(
                    "{{\"alpha\":{},\"beta\":{},\"loss\":{}}}",
                    jf(a),
                    jf(b),
                    jf(learner.cross_entropy_loss(&c.scores, &c.labels, a, b))
                )
            })
            .collect();
        pl_objs.push(format!(
            "{{\"name\":{},\"scores\":{},\"labels\":{},\"lr\":{},\"maxIter\":{},\"tol\":{},\"alpha\":{},\"beta\":{},\"converged\":{},\"lossHistory\":{},\"crossEntropy\":[{}]}}",
            jstr(c.name),
            jarr_f64(&c.scores),
            jarr_f64(&c.labels),
            jf(c.lr),
            c.max_iter,
            jf(c.tol),
            jf(res.alpha),
            jf(res.beta),
            res.converged,
            jarr_f64(&res.loss_history),
            ce_objs.join(",")
        ));
    }
    out.push_str(&format!("\"parameterLearner\":[{}],", pl_objs.join(",")));

    // -----------------------------------------------------------------------
    // runExperiments (exp1..13) — name + pass/fail. Details strings are skipped
    // (float-formatting between Rust/JS is not a meaningful parity target).
    // -----------------------------------------------------------------------
    let runner = ExperimentRunner::new(Rc::clone(&corpus), build_default_queries(), 1.2, 0.75);
    let results = runner.run_all();
    let exp_objs: Vec<String> = results
        .iter()
        .map(|(name, passed, _details)| {
            format!("{{\"name\":{},\"passed\":{}}}", jstr(name), passed)
        })
        .collect();
    out.push_str(&format!(
        "\"experiments\":{{\"params\":{{\"k1\":{},\"b\":{}}},\"results\":[{}]}}",
        jf(1.2),
        jf(0.75),
        exp_objs.join(",")
    ));

    // silence unused import warning for Query in some build configs
    let _ = Query::new("x", &["x"], None, &[]);

    out.push('}');
    println!("{}", out);
}
