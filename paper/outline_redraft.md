# Redraft outline — greedy DAG tokenizer paper

**Core claim.** A small, controlled set of tokens can represent the concepts of a DAG ontology with measurable and limited semantic loss. As a result, even rare concepts in EHR data are represented by meaningful, frequently seen tokens, which frequency truncation cannot do.

**Fixed choices**
- Decay is hop-dependent (λ per hop).
- The concept set is the SNOMED CT concepts occurring in the HERO data (16,370), and every result uses it. The 46k mapped-concept benchmark is dropped.
- Core metrics: semantic fidelity (the former distance score; name still open), unique rate, unknown rate, tokens per concept.
- Semantic coverage is the selection objective only. It is no longer an evaluation metric.
- Results come from `(final)ehr_application_analysis.ipynb` (notebook section in brackets below).

---

## Abstract
- **Main idea:** the problem, the method, and the evidence in about 200 words.
- **Show:** for the HERO data at k = 5,000:
  - share of concepts represented,
  - semantic fidelity,
  - unique rate,
  - rare concepts kept, against truncation.

  Take these from `tables/key_numbers.json` [nb 13].
- **Take-home:** A small vocabulary of ontology concepts, chosen by distance-aware greedy selection, represents every concept in the data, rare ones included, with measurable and limited semantic loss.

## 1. Introduction (`01_introduction.tex`)
- **1.1 Background**
  - **Main idea:** EHR foundation models consume coded events, and the vocabularies behind them are typed DAGs of meaning.
  - **Show:** citations only.
  - **Take-home:** Clinical codes come with a structured graph of meaning that foundation models rarely use.
- **1.2 Gap**
  - **Main idea:** the long tail leaves rare-code embeddings under-fitted.
    - Data-driven reduction drops rare concepts or their meaning.
    - Ontology-based reduction is lossy, uses IS_A only, or cannot be audited.
    - SNOMED CT is not named here.
  - **Show:** Table 1.
  - **Take-home:** No existing strategy compresses the vocabulary while keeping each concept's relations and a token set made of real, auditable concepts.
- **1.3 Contributions**
  - **Main idea:**
    1. Tokenization posed as attenuated coverage maximisation on a DAG: submodular, solved by lazy greedy.
    2. Context trees.
    3. A semantic fidelity metric (completeness × proximity) used alongside uniqueness.
    4. A benchmark against 7 graph baselines plus an ablation, and an application to HERO events against truncation.
  - **Show:** Figure 1 (pipeline).
  - **Take-home:** Few tokens, chosen in a controlled way, can stand for many concepts while keeping their meaning and keeping them apart.

## 2. Background (`02_background.tex`)
- **2.1 Directed acyclic graphs**
  - **Main idea:** typed DAG, hop distance, acyclicity.
  - **Show:** definitions only.
  - **Take-home:** Following edges forward only leads to more general concepts or attribute values, never back.
- **2.2 SNOMED CT as an instance**
  - **Main idea:** hierarchy edges vs attribute edges, and shared structure between distinct concepts.
  - **Show:** Figure 2 (shared parent, shared finding site).
  - **Take-home:** Distinct concepts share parts of their meaning through common neighbours, which is what lets them share tokens.

## 3. Method (`03_method.tex`)
- **3.1 Notation and problem statement**
  - **Main idea:** given the concepts M and a budget k, choose a set T of k concepts that best preserves the meaning of M.
  - **Show:** Table 2 (notation).
  - **Take-home:** Tokens are concepts of the ontology itself, so the vocabulary stays interpretable.
- **3.2 Candidate pool**
  - **Main idea:** the forward D-hop neighbourhood of M, with the root removed.
  - **Show:** Figure 3.
  - **Take-home:** Only concepts reachable within D hops can carry meaning for a concept in M.
- **3.3 Semantic coverage objective with distance decay**
  - **Main idea:**
    - Eq. 1: each hop multiplies the score by λ.
    - The objective is monotone submodular for any decay in [0, 1] (Appendix A).
    - "Semantic coverage" now names the *objective* only, not an evaluation metric.
    - **Hop distance can be adapted to properties of the DAG, such as concept status.** In SNOMED CT, for example, a hop into a fully defined concept loses no meaning and could be free, while a hop into a primitive concept costs λ. The (1 − 1/e) guarantee holds for any such per-hop factor in [0, 1], because the weight along a walk never increases.
  - **Show:** Eq. 1, the submodularity statement, and one sentence on the status-dependent adaptation (not evaluated).
  - **Take-home:** The objective credits each walk that reaches a token, discounted by the meaning lost on the way, and the definition of a hop's cost can follow the ontology without losing the guarantee.
- **3.4 Lazy greedy selection**
  - **Main idea:**
    - greedy with the (1 − 1/e) guarantee;
    - lazy evaluation;
    - sparse CandidateDelta;
    - complexity;
    - prefix property: one run serves every k.
  - **Show:** Algorithms 1–2.
  - **Take-home:** Tokens are selected with a quality guarantee, at local cost, and one run gives the vocabulary for every budget.
- **3.5 Context-tree tokenization**
  - **Main idea:** IS_A is transparent, attribute edges open subcontexts, and a branch stops at the first token.
  - **Show:** Figures 4–5 and Algorithm 3.
  - **Take-home:** Recording which relation leads to which token separates concepts that a flat token set would merge.
- **3.6 Evaluation metrics**
  - **Main idea:** four core metrics:
    - **semantic fidelity** f_m = completeness × proximity;
    - unique rate (represented, and the tree is not shared);
    - unknown rate;
    - tokens per concept.

    Secondary: uncovered branch rate, plus the two fidelity components on their own.
  - **Show:**
    - Table 3;
    - the worked example on Figure 5 (2/3 × 1/3 = 0.22);
    - one sentence on why coverage at λ = 1 is not used: it measures reachability, not meaning.
  - **Take-home:** A good tokenization keeps each concept's meaning close, keeps concepts apart, and stays short, and no single number captures all three.
- **3.7 Compared selection methods**
  - **Main idea:** 19 configurations:
    - ours at λ = 0, 0.1, …, 1.0;
    - the set-cover ablation;
    - 7 baselines: highest degree (d ≤ D, d ≤ 1), Most IS_A children, PageRank, Personalized PageRank, closeness, random.
  - **Show:** Table 4.
  - **Take-home:** All methods share the same candidate pool and differ only in how they rank candidates, so differences isolate the selection rule.
- **3.8 Patient-data evaluation**
  - **Main idea:** HERO events, ours vs frequency truncation (top-k + UNK). The measures:
    - concepts and events represented;
    - uniqueness (tree vs flat);
    - rare concepts kept;
    - borrowed strength;
    - fidelity by frequency (truncation: 1 if kept, 0 if UNK);
    - tokens per event.
  - **Show:** definitions only.
  - **Take-home:** On real data, a tokenizer must be judged on how it treats the long tail, not only on how many events it keeps.

## 4. Results (`04_results.tex`)
Three subsections instead of six. The former 4.2–4.5 become paragraphs (`\paragraph{}`) of one subsection.

- **4.1 Experimental setup** [nb 1–2, 4–5]
  - **Main idea:**
    - HERO data and the concept set M (16,370);
    - the candidate subgraph G_D;
    - the 19 configurations;
    - the budget grid: k = 250–6,750, about 1.5–41% of |M|, in 3 compression levels of 9 budgets;
    - implementation details.
  - **Show:** Table 5 (data and subgraph statistics, from `tables/setup.json`); Appendix B (relation types absent from G_D).
  - **Take-home:** The candidate pool keeps a small fraction of the ontology but almost all its relation types, and every method is evaluated on the same concepts and budgets.
- **4.2 Comparison of selection methods** [nb 6–10]
  - **Main idea:** four paragraphs.
    1. *Example context trees* (illustrative, λ = 0.9, k = 5,000): one concept with a unique tree (Pain of breast) and one collision (hyperthyroidism vs sick-euthyroid syndrome).
    2. *Overall ranking*: average rank over the 4 core metrics at each compression level, with the ablation against ours.
    3. *Token budget k*: metric trends and diminishing returns.
    4. *Distance decay λ*: failure at λ = 0, a plateau from 0.6 to 0.99, a drop at exactly λ = 1.
  - **Show:**
    - Figure 6 + Table 6 (example trees with completeness, proximity, fidelity, group size);
    - Figure 7 + Table 8 (ranking);
    - Figure 8 (metrics vs k);
    - Figure 9 + Table 9 (metrics vs λ).
  - **Take-home:** Distance-aware greedy selection (λ strictly between 0 and 1) outranks every degree, centrality and set-cover baseline at every compression level, and its exact λ matters much less than the budget k.
- **4.3 Application to patient data: ours vs truncation** [nb 11]
  - **Main idea:**
    - concepts vs events represented;
    - uniqueness;
    - rare concepts;
    - **fidelity by frequency bin**;
    - borrowed strength;
    - sequence cost.
  - **Show:**
    - Figure 10 (`ehr_summary`);
    - Figure 11 (`ehr_fidelity`: fidelity by frequency bin and by k);
    - Figure 12 (borrowed strength);
    - Table 10 (tab:ehr);
    - Table 11 (long tail with a fidelity column).
  - **Take-home:** Truncation keeps events but discards rare concepts entirely (fidelity 0), whereas our tokenizer represents every concept through frequently seen tokens with most of its meaning retained, at the cost of longer sequences.

## 5. Discussion (`05_discussion.tex`)
- **5.1 Why it works**
  - **Main idea:** hubs vs marginal gain plus decay; both mechanisms are needed (ablation); the two λ extremes.
  - **Show:** pointers to Table 8 and the ablation.
  - **Take-home:** Useful tokens are general enough to be shared but close enough to keep meaning, and only decay combined with marginal gain finds them.
- **5.2 Implications for EHR foundation models**
  - **Main idea:**
    - concept coverage vs event coverage;
    - vocabulary size vs data volume;
    - a hybrid vocabulary;
    - sequence length and pooling;
    - relation markers.
  - **Show:** pointers to Section 4.3 and the token-frequency skew (`ehr/distribution_stats.csv`).
  - **Take-home:** The tokenizer decides whether rare concepts reach the model at all, and can be combined with frequency tokens to limit sequence cost.
- **5.3 Evaluating a graph tokenizer needs several metrics** [nb 10, 12]
  - **Main idea:**
    - no method wins every metric;
    - tokens per concept must be read together with the unknown rate;
    - fidelity trades off against uniqueness;
    - the fidelity distribution and its two components show where meaning is lost.
  - **Show:** pointers to Table 8, the fidelity ECDF / completeness-proximity figure (`fidelity_distribution`), and the metric correlations (`metric_correlations`).
  - **Take-home:** Judging on a single metric would pick the wrong tokenizer.
- **5.4 Practical guidelines**
  - **Main idea:**
    - set k from data volume and compute;
    - λ = 0.9 as default;
    - one run serves every k;
    - selection needs only the list of concepts that occur, not patient records;
    - the hop cost can follow the ontology (e.g. concept status).
  - **Show:** no new results.
  - **Take-home:** Choose k from your data and compute budget, and leave λ at 0.9.

## 6. Limitations (`06_limitations.tex`)
- **Main idea:**
  - incomplete definitions and primitive concepts;
  - transductive selection, and greedy never removes a token;
  - no downstream or external validation;
  - DAG assumption;
  - evaluated on one ontology and one site;
  - the status-dependent hop cost is proposed but not evaluated.
- **Show:** no new results.
- **Take-home:** The results establish intrinsic, single-site, single-ontology quality, not a downstream benefit.

## 7. Conclusion and future work (`07_conclusion.tex`)
- **Main idea:** restate the contribution. Next steps:
  - pretraining experiments;
  - other ontologies;
  - token swaps;
  - evaluating status-dependent hop costs.
- **Show:** no new results.
- **Take-home:** Ontology-driven tokenization makes compact, auditable vocabularies possible, and its value for foundation models is the next test.

## Appendices (`appendix.tex`)
- A: Submodularity proof for a general per-hop factor in [0, 1].
- B: Relation types absent from G_D.
- C: Full ranking table (all 19 configurations × 3 compression levels).
- D: Random selection, plus metrics vs k including the uncovered branch rate (`metrics_vs_k_appendix`).
