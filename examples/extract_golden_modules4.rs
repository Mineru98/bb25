//! Golden-fixture extractor for FusionDebugger (src/debug.rs) numeric traces.
//!
//! Run: `cargo run --example extract_golden_modules4 > fixtures/golden_modules4.json`

use bayesian_bm25::{BayesianProbabilityTransform, FusionDebugger};

fn jf(x: f64) -> String {
    format!("{:?}", x)
}
fn jopt(x: Option<f64>) -> String {
    x.map(jf).unwrap_or_else(|| "null".into())
}
fn jarr(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|&x| jf(x)).collect::<Vec<_>>().join(","))
}

fn main() {
    let dbg = FusionDebugger::new(BayesianProbabilityTransform::new(1.0, 0.5, None));
    let dbg_br = FusionDebugger::new(BayesianProbabilityTransform::new(1.0, 0.5, Some(0.2)));

    let mut out = String::new();
    out.push('{');

    // trace_bm25 (no base rate + base rate)
    let b = dbg.trace_bm25(2.0, 5.0, 0.6);
    let bbr = dbg_br.trace_bm25(2.0, 5.0, 0.6);
    out.push_str(&format!(
        "\"traceBm25\":{{\"likelihood\":{},\"tfPrior\":{},\"normPrior\":{},\"compositePrior\":{},\"logitLikelihood\":{},\"logitPrior\":{},\"posterior\":{}}},",
        jf(b.likelihood), jf(b.tf_prior), jf(b.norm_prior), jf(b.composite_prior),
        jf(b.logit_likelihood), jf(b.logit_prior), jf(b.posterior)
    ));
    out.push_str(&format!(
        "\"traceBm25BaseRate\":{{\"posterior\":{},\"logitBaseRate\":{}}},",
        jf(bbr.posterior), jopt(bbr.logit_base_rate)
    ));

    // trace_vector
    let v = dbg.trace_vector(0.6);
    out.push_str(&format!(
        "\"traceVector\":{{\"probability\":{},\"logitProbability\":{}}},",
        jf(v.probability), jf(v.logit_probability)
    ));

    // trace_not
    let nt = dbg.trace_not(0.3, "x");
    out.push_str(&format!(
        "\"traceNot\":{{\"complement\":{},\"logitInput\":{},\"logitComplement\":{}}},",
        jf(nt.complement), jf(nt.logit_input), jf(nt.logit_complement)
    ));

    // trace_fusion variants
    let probs = vec![0.8, 0.6];
    let lo = dbg.trace_fusion(&probs, None, "log_odds", None, None);
    let weights = vec![0.3, 0.7];
    let low = dbg.trace_fusion(&probs, None, "log_odds", Some(0.0), Some(&weights));
    let pa = dbg.trace_fusion(&probs, None, "prob_and", None, None);
    let po = dbg.trace_fusion(&probs, None, "prob_or", None, None);
    let pn = dbg.trace_fusion(&probs, None, "prob_not", None, None);
    out.push_str(&format!(
        "\"traceFusion\":{{\"logOdds\":{{\"meanLogit\":{},\"nAlphaScale\":{},\"scaledLogit\":{},\"fused\":{},\"logits\":{}}},\"logOddsWeighted\":{{\"meanLogit\":{},\"scaledLogit\":{},\"fused\":{}}},\"probAnd\":{{\"logProbSum\":{},\"fused\":{}}},\"probOr\":{{\"logComplementSum\":{},\"fused\":{}}},\"probNot\":{{\"fused\":{}}}}},",
        jopt(lo.mean_logit), jopt(lo.n_alpha_scale), jopt(lo.scaled_logit), jf(lo.fused_probability),
        lo.logits.as_ref().map(|l| jarr(l)).unwrap_or_else(|| "null".into()),
        jopt(low.mean_logit), jopt(low.scaled_logit), jf(low.fused_probability),
        jopt(pa.log_prob_sum), jf(pa.fused_probability),
        jopt(po.log_complement_sum), jf(po.fused_probability),
        jf(pn.fused_probability)
    ));

    // trace_document + compare
    let da = dbg.trace_document(Some(2.0), Some(5.0), Some(0.6), Some(0.6), "log_odds", None, None, Some("dA"));
    let db = dbg.trace_document(Some(0.5), Some(1.0), Some(0.4), Some(0.2), "log_odds", None, None, Some("dB"));
    let cmp = dbg.compare(&da, &db);
    let deltas: Vec<String> = cmp
        .signal_deltas
        .iter()
        .map(|(n, d)| format!("{{\"name\":{:?},\"delta\":{}}}", n, jf(*d)))
        .collect();
    out.push_str(&format!(
        "\"traceDocument\":{{\"finalA\":{},\"finalB\":{}}},\"compare\":{{\"deltas\":[{}],\"dominant\":{:?}}}",
        jf(da.final_probability), jf(db.final_probability), deltas.join(","), cmp.dominant_signal
    ));

    out.push('}');
    println!("{}", out);
}
