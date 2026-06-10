//! Golden-fixture extractor for Phase A3 modules: LearnableLogOddsWeights,
//! AttentionLogOddsWeights (PRNG init), MultiHeadAttentionLogOddsWeights, BlockMaxIndex.
//!
//! Run: `cargo run --example extract_golden_modules3 > fixtures/golden_modules3.json`

use bayesian_bm25::{
    AttentionLogOddsWeights, BayesianProbabilityTransform, BlockMaxIndex, LearnableLogOddsWeights,
    MultiHeadAttentionLogOddsWeights,
};

fn jf(x: f64) -> String {
    format!("{:?}", x)
}
fn jarr(v: &[f64]) -> String {
    let p: Vec<String> = v.iter().map(|&x| jf(x)).collect();
    format!("[{}]", p.join(","))
}
fn jusize(v: &[usize]) -> String {
    let p: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", p.join(","))
}

fn main() {
    let mut out = String::new();
    out.push('{');

    // -----------------------------------------------------------------------
    // LearnableLogOddsWeights
    // -----------------------------------------------------------------------
    // init: uniform weights (softmax of zero logits) + averaged
    let ll = LearnableLogOddsWeights::new(3, 0.0, None);
    let ll_combine_plain = {
        let l2 = LearnableLogOddsWeights::new(2, 0.0, None);
        l2.combine(&[0.7, 0.8], false)
    };
    let ll_combine_br = {
        let l2 = LearnableLogOddsWeights::new(2, 0.0, Some(0.3));
        l2.combine(&[0.7, 0.8], false)
    };

    // fit
    let ll_probs: Vec<Vec<f64>> = vec![
        vec![0.9, 0.2],
        vec![0.8, 0.3],
        vec![0.2, 0.9],
        vec![0.3, 0.85],
        vec![0.6, 0.6],
        vec![0.1, 0.1],
    ];
    let ll_labels = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let mut ll_fit = LearnableLogOddsWeights::new(2, 0.0, None);
    ll_fit.fit(&ll_probs, &ll_labels, 0.1, 300, 1e-9);
    let ll_fit_weights = ll_fit.weights();
    let ll_fit_combine = ll_fit.combine(&[0.7, 0.8], false);

    // update sequence
    let mut ll_up = LearnableLogOddsWeights::new(2, 0.5, None);
    for i in 0..ll_probs.len() {
        ll_up.update(&[ll_probs[i].clone()], &[ll_labels[i]], 0.05, 0.9, 1000.0, 1.0, 0.99);
    }

    let ll_probs_flat: Vec<String> = ll_probs.iter().map(|r| jarr(r)).collect();
    out.push_str(&format!(
        "\"learnable\":{{\"initWeights\":{},\"initAveraged\":{},\"combinePlain\":{},\"combineBaseRate\":{},\"fit\":{{\"probs\":[{}],\"labels\":{},\"weights\":{},\"combine\":{}}},\"update\":{{\"weights\":{},\"averaged\":{}}}}},",
        jarr(&ll.weights()),
        jarr(&ll.averaged_weights()),
        jf(ll_combine_plain),
        jf(ll_combine_br),
        ll_probs_flat.join(","),
        jarr(&ll_labels),
        jarr(&ll_fit_weights),
        jf(ll_fit_combine),
        jarr(&ll_up.weights()),
        jarr(&ll_up.averaged_weights())
    ));

    // -----------------------------------------------------------------------
    // AttentionLogOddsWeights — PRNG init parity is the critical gate
    // -----------------------------------------------------------------------
    let init_configs = [(2usize, 1usize, 0u64), (2, 1, 1), (2, 3, 42), (4, 2, 7)];
    let init_objs: Vec<String> = init_configs
        .iter()
        .map(|&(ns, nqf, seed)| {
            let a = AttentionLogOddsWeights::new(ns, nqf, 0.5, false, seed, None);
            format!(
                "{{\"nSignals\":{},\"nQueryFeatures\":{},\"seed\":{},\"weightsMatrix\":{}}}",
                ns,
                nqf,
                seed,
                jarr(&a.weights_matrix())
            )
        })
        .collect();

    // combine: n_signals=2, nqf=1, seed=0, alpha=0.5
    let attn = AttentionLogOddsWeights::new(2, 1, 0.5, false, 0, None);
    // m=3 docs, each 2 signals; query features m_q=1 (broadcast)
    let c_probs = vec![0.9, 0.2, 0.3, 0.8, 0.6, 0.6];
    let c_qf_bcast = vec![2.0];
    let combine_bcast = attn.combine(&c_probs, 3, &c_qf_bcast, 1, false);
    // m_q=3 (per-row features)
    let c_qf_full = vec![1.0, 2.0, 3.0];
    let combine_full = attn.combine(&c_probs, 3, &c_qf_full, 3, false);
    // m==1 single
    let combine_single = attn.combine(&[0.8, 0.9], 1, &[1.5], 1, false);

    // normalize=true variant
    let attn_norm = AttentionLogOddsWeights::new(2, 1, 0.5, true, 0, None);
    let combine_norm = attn_norm.combine(&c_probs, 3, &c_qf_full, 3, false);

    // base_rate variant
    let attn_br = AttentionLogOddsWeights::new(2, 1, 0.5, false, 0, Some(0.3));
    let combine_br = attn_br.combine(&c_probs, 3, &c_qf_full, 3, false);

    // fit then weights_matrix + combine
    let mut attn_fit = AttentionLogOddsWeights::new(2, 1, 0.5, false, 0, None);
    let f_probs = vec![0.9, 0.2, 0.8, 0.3, 0.2, 0.9, 0.3, 0.85, 0.6, 0.6, 0.1, 0.1];
    let f_labels = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let f_qf = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0];
    attn_fit.fit(&f_probs, &f_labels, &f_qf, 6, None, 0.1, 200, 1e-9);
    let attn_fit_wm = attn_fit.weights_matrix();
    let attn_fit_combine = attn_fit.combine(&c_probs, 3, &c_qf_full, 3, false);

    // compute_upper_bounds + prune
    let ub_probs = vec![0.95, 0.9, 0.4, 0.5, 0.2, 0.1];
    let ubs = attn.compute_upper_bounds(&ub_probs, 3, &c_qf_full, 3, false);
    let (surv, pruned) = attn.prune(&c_probs, 3, &c_qf_full, 3, 0.5, Some(&ub_probs), false);

    out.push_str(&format!(
        "\"attention\":{{\"init\":[{}],\"combineBroadcast\":{},\"combineFull\":{},\"combineSingle\":{},\"combineNormalize\":{},\"combineBaseRate\":{},\"fit\":{{\"weightsMatrix\":{},\"combine\":{}}},\"computeUpperBounds\":{},\"prune\":{{\"surviving\":{},\"fused\":{}}}}},",
        init_objs.join(","),
        jarr(&combine_bcast),
        jarr(&combine_full),
        jarr(&combine_single),
        jarr(&combine_norm),
        jarr(&combine_br),
        jarr(&attn_fit_wm),
        jarr(&attn_fit_combine),
        jarr(&ubs),
        jusize(&surv),
        jarr(&pruned)
    ));

    // -----------------------------------------------------------------------
    // MultiHeadAttentionLogOddsWeights (per-head seeds 0..n_heads-1)
    // -----------------------------------------------------------------------
    let mh = MultiHeadAttentionLogOddsWeights::new(4, 2, 1, 0.5, false);
    let mh_combine_single = mh.combine(&[0.8, 0.9], 1, &[1.0], 1, false);
    let mh_combine_multi = mh.combine(&c_probs, 3, &c_qf_full, 3, false);

    let mut mh_fit = MultiHeadAttentionLogOddsWeights::new(4, 2, 1, 0.5, false);
    mh_fit.fit(&f_probs, &f_labels, &f_qf, 6, None, 0.1, 200, 1e-9);
    let mh_fit_combine = mh_fit.combine(&c_probs, 3, &c_qf_full, 3, false);

    out.push_str(&format!(
        "\"multiHead\":{{\"nHeads\":4,\"combineSingle\":{},\"combineMulti\":{},\"fitCombine\":{}}},",
        jarr(&mh_combine_single),
        jarr(&mh_combine_multi),
        jarr(&mh_fit_combine)
    ));

    // -----------------------------------------------------------------------
    // BlockMaxIndex
    // -----------------------------------------------------------------------
    let bmi_matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.5, 2.5],
        vec![0.5, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7],
    ];
    let mut bmi = BlockMaxIndex::new(3);
    bmi.build(&bmi_matrix);
    let n_blocks = bmi.n_blocks();
    let mut block_ub: Vec<String> = Vec::new();
    for term in 0..2 {
        let mut row: Vec<f64> = Vec::new();
        for blk in 0..n_blocks {
            row.push(bmi.block_upper_bound(term, blk));
        }
        block_ub.push(jarr(&row));
    }
    let transform = BayesianProbabilityTransform::new(1.0, 0.5, None);
    let bayes_ub = bmi.bayesian_block_upper_bound(0, 0, &transform, 0.9);

    let bmi_matrix_json: Vec<String> = bmi_matrix.iter().map(|r| jarr(r)).collect();
    out.push_str(&format!(
        "\"blockMaxIndex\":{{\"blockSize\":3,\"matrix\":[{}],\"nBlocks\":{},\"blockUpperBound\":[{}],\"bayesianBlockUpperBound00\":{}}}",
        bmi_matrix_json.join(","),
        n_blocks,
        block_ub.join(","),
        jf(bayes_ub)
    ));

    out.push('}');
    println!("{}", out);
}
