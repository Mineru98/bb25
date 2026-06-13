//! Golden-fixture extractor for Phase A2 modules: BayesianProbabilityTransform,
//! TemporalBayesianTransform, PlattCalibrator, IsotonicCalibrator.
//!
//! Run: `cargo run --example extract_golden_modules2 > fixtures/golden_modules2.json`

use bayesian_bm25::{
    BayesianProbabilityTransform, IsotonicCalibrator, PlattCalibrator, TemporalBayesianTransform,
    TrainingMode,
};

fn jf(x: f64) -> String {
    format!("{:?}", x)
}
fn jarr(v: &[f64]) -> String {
    let p: Vec<String> = v.iter().map(|&x| jf(x)).collect();
    format!("[{}]", p.join(","))
}

fn main() {
    let mut out = String::new();
    out.push('{');

    // -----------------------------------------------------------------------
    // BayesianProbabilityTransform — static + instance deterministic surface
    // -----------------------------------------------------------------------
    let pt = BayesianProbabilityTransform::new(1.0, 0.5, None);
    let scores = [-1.0, 0.0, 0.5, 1.0, 2.0, 5.0];
    let likelihood: Vec<String> = scores
        .iter()
        .map(|&s| format!("[{},{}]", jf(s), jf(pt.likelihood(s))))
        .collect();

    let tf_vals = [0.0, 1.0, 5.0, 10.0, 20.0];
    let tf_prior: Vec<String> = tf_vals
        .iter()
        .map(|&t| format!("[{},{}]", jf(t), jf(BayesianProbabilityTransform::tf_prior(t))))
        .collect();

    let dlr_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5];
    let norm_prior: Vec<String> = dlr_vals
        .iter()
        .map(|&r| format!("[{},{}]", jf(r), jf(BayesianProbabilityTransform::norm_prior(r))))
        .collect();

    let comp_pairs = [(0.0, 0.5), (1.0, 0.5), (5.0, 0.25), (10.0, 0.75), (20.0, 1.5)];
    let composite_prior: Vec<String> = comp_pairs
        .iter()
        .map(|&(t, r)| {
            format!(
                "{{\"tf\":{},\"dlr\":{},\"v\":{}}}",
                jf(t),
                jf(r),
                jf(BayesianProbabilityTransform::composite_prior(t, r))
            )
        })
        .collect();

    let post_cases = [
        (0.8, 0.6, None),
        (0.3, 0.7, None),
        (0.9, 0.5, Some(0.01)),
        (0.5, 0.5, Some(0.5)),
        (0.99, 0.2, Some(0.3)),
    ];
    let posterior: Vec<String> = post_cases
        .iter()
        .map(|&(l, p, br)| {
            format!(
                "{{\"likelihood\":{},\"prior\":{},\"baseRate\":{},\"v\":{}}}",
                jf(l),
                jf(p),
                br.map(jf).unwrap_or_else(|| "null".into()),
                jf(BayesianProbabilityTransform::posterior(l, p, br))
            )
        })
        .collect();

    // score_to_probability (default = Balanced mode -> composite_prior)
    let s2p_cases = [(1.0, 5.0, 0.5), (2.0, 1.0, 1.2), (0.5, 0.0, 0.1), (3.0, 10.0, 0.8)];
    let score_to_prob: Vec<String> = s2p_cases
        .iter()
        .map(|&(s, tf, r)| {
            format!(
                "{{\"score\":{},\"tf\":{},\"dlr\":{},\"v\":{}}}",
                jf(s),
                jf(tf),
                jf(r),
                jf(pt.score_to_probability(s, tf, r))
            )
        })
        .collect();

    // score_to_probability with base_rate
    let pt_br = BayesianProbabilityTransform::new(1.0, 0.5, Some(0.2));
    let score_to_prob_br: Vec<String> = s2p_cases
        .iter()
        .map(|&(s, tf, r)| {
            format!(
                "{{\"score\":{},\"tf\":{},\"dlr\":{},\"v\":{}}}",
                jf(s),
                jf(tf),
                jf(r),
                jf(pt_br.score_to_probability(s, tf, r))
            )
        })
        .collect();

    let wand_cases = [(2.0, 0.9), (5.0, 0.5), (1.0, 0.7)];
    let wand: Vec<String> = wand_cases
        .iter()
        .map(|&(ub, pmax)| {
            format!(
                "{{\"ub\":{},\"pMax\":{},\"v\":{}}}",
                jf(ub),
                jf(pmax),
                jf(pt.wand_upper_bound(ub, pmax))
            )
        })
        .collect();

    // fit: balanced, prior_free, prior_aware
    let fit_scores = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5];
    let fit_labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    let fit_tfs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0];
    let fit_dlrs = vec![0.5, 0.6, 0.4, 0.5, 0.7, 0.3, 0.5, 0.6];

    let mut pt_bal = BayesianProbabilityTransform::new(1.0, 0.0, None);
    pt_bal.fit(&fit_scores, &fit_labels, 0.1, 300, 1e-9, TrainingMode::Balanced, None, None);

    let mut pt_pf = BayesianProbabilityTransform::new(1.0, 0.0, None);
    pt_pf.fit(&fit_scores, &fit_labels, 0.1, 300, 1e-9, TrainingMode::PriorFree, None, None);

    let mut pt_pa = BayesianProbabilityTransform::new(1.0, 0.0, None);
    pt_pa.fit(
        &fit_scores,
        &fit_labels,
        0.1,
        300,
        1e-9,
        TrainingMode::PriorAware,
        Some(&fit_tfs),
        Some(&fit_dlrs),
    );

    let fit_json = format!(
        "{{\"scores\":{},\"labels\":{},\"tfs\":{},\"dlrs\":{},\"balanced\":{{\"alpha\":{},\"beta\":{}}},\"priorFree\":{{\"alpha\":{},\"beta\":{}}},\"priorAware\":{{\"alpha\":{},\"beta\":{}}}}}",
        jarr(&fit_scores),
        jarr(&fit_labels),
        jarr(&fit_tfs),
        jarr(&fit_dlrs),
        jf(pt_bal.alpha),
        jf(pt_bal.beta),
        jf(pt_pf.alpha),
        jf(pt_pf.beta),
        jf(pt_pa.alpha),
        jf(pt_pa.beta)
    );

    // update: sequence of single-sample online updates (balanced)
    let mut pt_up = BayesianProbabilityTransform::new(1.0, 0.0, None);
    let up_scores = [0.5, 1.5, 2.5, 3.5, 4.5, 0.8, 2.2];
    let up_labels = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    for i in 0..up_scores.len() {
        pt_up.update(
            &[up_scores[i]],
            &[up_labels[i]],
            0.05,
            0.9,
            1000.0,
            1.0,
            0.99,
            Some(TrainingMode::Balanced),
            None,
            None,
        );
    }
    let update_json = format!(
        "{{\"scores\":{},\"labels\":{},\"alpha\":{},\"beta\":{},\"averagedAlpha\":{},\"averagedBeta\":{}}}",
        jarr(&up_scores),
        jarr(&up_labels),
        jf(pt_up.alpha),
        jf(pt_up.beta),
        jf(pt_up.averaged_alpha()),
        jf(pt_up.averaged_beta())
    );

    out.push_str(&format!(
        "\"probabilityTransform\":{{\"likelihood\":[{}],\"tfPrior\":[{}],\"normPrior\":[{}],\"compositePrior\":[{}],\"posterior\":[{}],\"scoreToProbability\":[{}],\"scoreToProbabilityBaseRate\":[{}],\"wandUpperBound\":[{}],\"fit\":{},\"update\":{}}},",
        likelihood.join(","),
        tf_prior.join(","),
        norm_prior.join(","),
        composite_prior.join(","),
        posterior.join(","),
        score_to_prob.join(","),
        score_to_prob_br.join(","),
        wand.join(","),
        fit_json,
        update_json
    ));

    // -----------------------------------------------------------------------
    // TemporalBayesianTransform
    // -----------------------------------------------------------------------
    let t_scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let t_labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
    let t_ts: Vec<usize> = vec![0, 10, 20, 30, 100];

    let mut temp_fit = TemporalBayesianTransform::new(1.0, 0.0, None, 50.0);
    temp_fit.fit(
        &t_scores,
        &t_labels,
        Some(&t_ts),
        0.1,
        300,
        1e-9,
        TrainingMode::Balanced,
        None,
        None,
    );

    let mut temp_up = TemporalBayesianTransform::new(1.0, 0.0, None, 50.0);
    for i in 0..t_scores.len() {
        temp_up.update(
            &[t_scores[i]],
            &[t_labels[i]],
            0.05,
            0.9,
            1000.0,
            1.0,
            0.995,
            Some(TrainingMode::Balanced),
            None,
            None,
        );
    }

    out.push_str(&format!(
        "\"temporalTransform\":{{\"scores\":{},\"labels\":{},\"timestamps\":[{}],\"decayHalfLife\":{},\"fit\":{{\"alpha\":{},\"beta\":{}}},\"update\":{{\"timestamp\":{},\"alpha\":{},\"beta\":{},\"averagedAlpha\":{},\"averagedBeta\":{}}}}},",
        jarr(&t_scores),
        jarr(&t_labels),
        t_ts.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
        jf(50.0),
        jf(temp_fit.transform.alpha),
        jf(temp_fit.transform.beta),
        temp_up.timestamp(),
        jf(temp_up.transform.alpha),
        jf(temp_up.transform.beta),
        jf(temp_up.averaged_alpha()),
        jf(temp_up.averaged_beta())
    ));

    // -----------------------------------------------------------------------
    // PlattCalibrator
    // -----------------------------------------------------------------------
    let cal_scores = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let cal_labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut platt = PlattCalibrator::new(1.0, 0.0);
    platt.fit(&cal_scores, &cal_labels, 0.1, 500, 1e-9);
    let platt_cal = platt.calibrate_batch(&cal_scores);

    out.push_str(&format!(
        "\"platt\":{{\"scores\":{},\"labels\":{},\"lr\":{},\"maxIter\":{},\"tol\":{},\"a\":{},\"b\":{},\"calibrated\":{}}},",
        jarr(&cal_scores),
        jarr(&cal_labels),
        jf(0.1),
        500,
        jf(1e-9),
        jf(platt.a),
        jf(platt.b),
        jarr(&platt_cal)
    ));

    // -----------------------------------------------------------------------
    // IsotonicCalibrator (PAVA)
    // -----------------------------------------------------------------------
    let iso_scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let iso_labels = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let mut iso = IsotonicCalibrator::new();
    iso.fit(&iso_scores, &iso_labels);
    let probe = vec![0.5, 1.0, 2.5, 3.0, 4.5, 5.5, 6.0, 7.5, 8.0, 9.0];
    let iso_cal = iso.calibrate_batch(&probe);

    out.push_str(&format!(
        "\"isotonic\":{{\"scores\":{},\"labels\":{},\"probe\":{},\"calibrated\":{}}}",
        jarr(&iso_scores),
        jarr(&iso_labels),
        jarr(&probe),
        jarr(&iso_cal)
    ));

    out.push('}');
    println!("{}", out);
}
