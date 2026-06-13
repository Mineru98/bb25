//! Golden-fixture extractor for the TypeScript migration.
//!
//! Runs the reference Rust core over the default corpus/queries and emits a
//! single JSON object to stdout. The TS test-suite (`@bb25/core`) loads this
//! verbatim and asserts numeric parity.
//!
//! Run with: `cargo run --example extract_golden > fixtures/golden.json`

use std::rc::Rc;

use bayesian_bm25::{
    build_default_corpus, build_default_queries, cosine_to_probability, log_odds_conjunction,
    logit, prob_and, prob_or, sigmoid, BM25Scorer, BayesianBM25Scorer, Gating, HybridScorer,
    Tokenizer, VectorScorer,
};

/// Round-trippable f64 -> JSON number string (Rust Debug gives shortest round-trip repr).
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

fn jarr_str(v: &[String]) -> String {
    let parts: Vec<String> = v.iter().map(|s| jstr(s)).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let corpus = Rc::new(build_default_corpus());
    let queries = build_default_queries();

    let k1 = 1.2;
    let b = 0.75;
    let bm25 = Rc::new(BM25Scorer::new(Rc::clone(&corpus), k1, b));
    let bayesian = Rc::new(BayesianBM25Scorer::new(Rc::clone(&bm25), 1.0, 0.5, None));
    let vector = Rc::new(VectorScorer::new());
    let hybrid = HybridScorer::new(Rc::clone(&bayesian), Rc::clone(&vector), 0.5);

    let mut out = String::new();
    out.push('{');

    // --- params ---
    out.push_str(&format!(
        "\"params\":{{\"k1\":{},\"b\":{},\"alpha\":{},\"beta\":{},\"hybridAlpha\":{},\"epsilon\":{}}},",
        jf(k1),
        jf(b),
        jf(1.0),
        jf(0.5),
        jf(0.5),
        jf(1e-10)
    ));

    // --- corpus stats ---
    let mut df_terms: Vec<String> = corpus.df.keys().cloned().collect();
    df_terms.sort();
    let df_entries: Vec<String> = df_terms
        .iter()
        .map(|t| format!("{}:{}", jstr(t), corpus.df[t]))
        .collect();
    out.push_str(&format!(
        "\"corpus\":{{\"n\":{},\"avgdl\":{},\"df\":{{{}}}}},",
        corpus.n,
        jf(corpus.avgdl),
        df_entries.join(",")
    ));

    // --- documents (tokens, length, term_freq) ---
    let mut doc_objs: Vec<String> = Vec::new();
    for doc in corpus.documents() {
        let mut tf_keys: Vec<String> = doc.term_freq.keys().cloned().collect();
        tf_keys.sort();
        let tf_entries: Vec<String> = tf_keys
            .iter()
            .map(|t| format!("{}:{}", jstr(t), doc.term_freq[t]))
            .collect();
        doc_objs.push(format!(
            "{{\"id\":{},\"text\":{},\"embedding\":{},\"tokens\":{},\"length\":{},\"termFreq\":{{{}}}}}",
            jstr(&doc.id),
            jstr(&doc.text),
            jarr_f64(&doc.embedding),
            jarr_str(&doc.tokens),
            doc.length,
            tf_entries.join(",")
        ));
    }
    out.push_str(&format!("\"documents\":[{}],", doc_objs.join(",")));

    // --- idf per df term ---
    let idf_entries: Vec<String> = df_terms
        .iter()
        .map(|t| format!("{}:{}", jstr(t), jf(bm25.idf(t))))
        .collect();
    out.push_str(&format!("\"idf\":{{{}}},", idf_entries.join(",")));

    // --- queries ---
    let mut q_objs: Vec<String> = Vec::new();
    for q in &queries {
        let emb = match &q.embedding {
            Some(e) => jarr_f64(e),
            None => "null".to_string(),
        };
        q_objs.push(format!(
            "{{\"text\":{},\"terms\":{},\"embedding\":{},\"relevant\":{}}}",
            jstr(&q.text),
            jarr_str(&q.terms),
            emb,
            jarr_str(&q.relevant)
        ));
    }
    out.push_str(&format!("\"queries\":[{}],", q_objs.join(",")));

    // --- full score matrix per (query, doc) ---
    let mut score_objs: Vec<String> = Vec::new();
    for q in &queries {
        let emb = q.embedding.as_ref().expect("default queries all have embeddings");
        let mut per_doc: Vec<String> = Vec::new();
        for doc in corpus.documents() {
            let bm25_raw = bm25.score(&q.terms, doc);
            let bayes = bayesian.score(&q.terms, doc);
            let vec_p = vector.score(emb, doc);
            let h_or = hybrid.score_or(&q.terms, emb, doc);
            let h_and = hybrid.score_and(&q.terms, emb, doc);

            // term-level breakdown
            let mut term_objs: Vec<String> = Vec::new();
            for term in &q.terms {
                let st = bm25.score_term_standard(term, doc);
                let bt = bayesian.score_term(term, doc);
                term_objs.push(format!(
                    "{{\"term\":{},\"bm25Term\":{},\"bayesianTerm\":{}}}",
                    jstr(term),
                    jf(st),
                    jf(bt)
                ));
            }

            per_doc.push(format!(
                "{{\"id\":{},\"bm25\":{},\"bayesian\":{},\"vector\":{},\"hybridOr\":{},\"hybridAnd\":{},\"terms\":[{}]}}",
                jstr(&doc.id),
                jf(bm25_raw),
                jf(bayes),
                jf(vec_p),
                jf(h_or),
                jf(h_and),
                term_objs.join(",")
            ));
        }
        score_objs.push(format!(
            "{{\"query\":{},\"perDoc\":[{}]}}",
            jstr(&q.text),
            per_doc.join(",")
        ));
    }
    out.push_str(&format!("\"scores\":[{}],", score_objs.join(",")));

    // --- tokenizer edge cases ---
    let tok = Tokenizer::new();
    let tok_cases = [
        "Machine Learning",
        "BM25!!",
        "TF-IDF weighting",
        "café",
        "한글 text",
        "  leading and  multiple   spaces  ",
        "Mixed123Numbers456",
        "UPPER lower MiXeD",
        "punctuation: a.b,c;d!e?f",
        "",
        "100% pure_snake_case",
        "naïve résumé",
        "B-trees ACID",
    ];
    let tok_objs: Vec<String> = tok_cases
        .iter()
        .map(|c| {
            format!(
                "{{\"input\":{},\"tokens\":{}}}",
                jstr(c),
                jarr_str(&tok.tokenize(c))
            )
        })
        .collect();
    out.push_str(&format!("\"tokenizer\":[{}],", tok_objs.join(",")));

    // --- math/fusion primitives ---
    let sig_xs = [-700.0, -100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0, 700.0];
    let sig: Vec<String> = sig_xs
        .iter()
        .map(|&x| format!("[{},{}]", jf(x), jf(sigmoid(x))))
        .collect();

    let logit_ps = [1e-12, 1e-10, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1.0 - 1e-10, 1.0];
    let logit_v: Vec<String> = logit_ps
        .iter()
        .map(|&p| format!("[{},{}]", jf(p), jf(logit(p))))
        .collect();

    let c2p_ss = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let c2p: Vec<String> = c2p_ss
        .iter()
        .map(|&s| format!("[{},{}]", jf(s), jf(cosine_to_probability(s))))
        .collect();

    let prob_sets: Vec<Vec<f64>> = vec![
        vec![0.9, 0.9],
        vec![0.9, 0.1],
        vec![0.5, 0.5],
        vec![0.3, 0.7],
        vec![0.8, 0.8, 0.8],
        vec![0.01, 0.99],
        vec![0.2, 0.6, 0.3],
    ];
    let mut prob_objs: Vec<String> = Vec::new();
    for ps in &prob_sets {
        prob_objs.push(format!(
            "{{\"probs\":{},\"probOr\":{},\"probAnd\":{},\"logOddsConjDefault\":{},\"logOddsConjAlpha05\":{}}}",
            jarr_f64(ps),
            jf(prob_or(ps)),
            jf(prob_and(ps)),
            jf(log_odds_conjunction(ps, None, None, Gating::NoGating)),
            jf(log_odds_conjunction(ps, Some(0.5), None, Gating::NoGating))
        ));
    }
    out.push_str(&format!(
        "\"math\":{{\"sigmoid\":[{}],\"logit\":[{}],\"cosineToProbability\":[{}],\"fusion\":[{}]}}",
        sig.join(","),
        logit_v.join(","),
        c2p.join(","),
        prob_objs.join(",")
    ));

    out.push('}');
    println!("{}", out);
}
