# Redraft outline — greedy DAG tokenizer paper (target: 20 pages)

**Core claim.** A small, controlled set of tokens can represent the concepts of a DAG ontology with measurable and limited semantic loss: the meaning of each concept is kept close (high proximity-coverage score) and concepts stay distinguishable (unique rate). Applied to real EHR data (HERO), this lets even rare concepts be represented by meaningful, frequently seen tokens, which frequency truncation cannot do. Semanticians compare our representations with those of the best baseline.

**Analysis line**
- **Main results**, on all SNOMED CT concepts mapped in our clinical system (M, |M| = 46,150): ranking of the 19 configurations, effect of k, effect of λ.
- **HERO application**, on the SNOMED CT concepts that occur in the HERO EHR data (M_HERO, 16,370, all in M): mainly how many rare concepts are covered, against frequency truncation.
- **Expert evaluation**: ours vs the best baseline, rated blindly by semanticians.

**Fixed choices**
- Decay: hop-dependent (λ per hop). The text mentions that the hop cost can follow DAG properties, such as concept status.
- Configuration in examples, the HERO application and the expert evaluation: λ = 0.9, k = 5,000, D = 9.
- Core metrics:
  - **proximity-coverage score** (formerly the distance score);
  - unique rate;
  - unknown rate;
  - tokens per concept.
- Semantic coverage is the selection objective only, not an evaluation metric.
- Main results report the **mean** of each metric **and the distribution over concepts** of the per-concept scores.

**Terminology, to keep apart in the text.** "Coverage" appears in three senses. Use these exact terms:

| Term | Where | Meaning |
|---|---|---|
| *semantic coverage* | Section 3.3 | the selection objective (Eq. 1) |
| *branch coverage* c_m | Section 3.5 | the component of the proximity-coverage score: share of a concept's branches that reach a token |
| *concepts / events represented* | HERO application | share of concepts, or of events, that receive at least one token |

**Page budget** (main text with figures and tables; references and appendices extra)

| Section | Pages | Main figures / tables |
|---|---|---|
| Title + abstract | 0.5 | |
| 1 Introduction | 2.0 | Table 1, Figure 1 |
| 2 Background (SNOMED CT) | 0.5 | Figure 2 |
| 3 Method | 5.5 | Figure 3, Algorithm 1, Tables 2–3 |
| 4 Results | 7.5 | Figures 4–9, Tables 4–6 |
| 5 Discussion | 2.0 | |
| 6 Limitations and future work | 0.8 | |
| 7 Conclusion | 0.3 | |
| **Total** | **19.1** | 9 figures, 6 tables, 1 algorithm |

If references count toward the 20 pages, take the extra 1–2 pages from the Discussion (5.2) and the HERO prose (4.4).

---

## Abstract
- **Main idea:**
  - the problem and the method;
  - the main results on all mapped concepts;
  - the HERO application;
  - the expert evaluation;
  - about 200 words.
- **Show:**
  - on M at k = 5,000: the proximity-coverage score (mean and median) and the unique rate;
  - on HERO: the share of rare concepts covered, ours vs truncation;
  - the expert preference.
- **Take-home:** A small vocabulary of ontology concepts, chosen by distance-aware greedy selection, represents every concept with limited semantic loss, including the rare concepts of real EHR data that frequency truncation discards.

## 1. Introduction (`01_introduction.tex`) — 2.0 pages
- **1.1 Background**
  - **Main idea:** EHR foundation models consume coded events, and the vocabularies behind them are typed DAGs of meaning.
  - **Show:** citations only.
  - **Take-home:** Clinical codes come with a structured graph of meaning that foundation models rarely use.
- **1.2 Gap**
  - **Main idea:** the long tail leaves rare-code embeddings under-fitted.
    - Data-driven reduction drops rare concepts or their meaning.
    - Ontology-based reduction is lossy, uses IS_A only, or cannot be audited.
    - SNOMED CT is not named here.
  - **Show:** Table 1, kept but compact: short cells, a one-line "primary limitation" per row.
  - **Take-home:** No existing strategy compresses the vocabulary while keeping each concept's relations and a token set made of real, auditable concepts.
- **1.3 Contributions**
  - **Main idea:**
    1. Tokenization posed as attenuated coverage maximisation on a DAG: submodular, solved by lazy greedy.
    2. Context trees.
    3. The proximity-coverage score.
    4. A benchmark on 46,150 mapped concepts against 7 graph baselines and an ablation.
    5. An application to HERO EHR data, focused on rare concepts.
    6. A blinded expert evaluation against the best baseline.
  - **Show:** Figure 1 (pipeline).
  - **Take-home:** Few tokens, chosen in a controlled way, can stand for many concepts while keeping their meaning and keeping them apart.

## 2. Background: SNOMED CT (`02_background.tex`) — 0.5 page
*(The DAG definitions of the former 2.1 move to 3.1.)*
- **Main idea:**
  - hierarchy edges vs attribute edges;
  - shared structure between distinct concepts;
  - fully defined vs primitive concepts (one sentence, needed for the expert evaluation and the limitations).
- **Show:** Figure 2 (shared parent, shared finding site).
- **Take-home:** Distinct concepts share parts of their meaning through common neighbours, which is what lets them share tokens.

## 3. Method (`03_method.tex`) — 5.5 pages
- **3.1 Problem statement and notation**
  - **Main idea:**
    - typed DAG and hop distance, in one paragraph (the former 2.1);
    - given a set of concepts M and a budget k, choose a set T of k ontology concepts that best preserves the meaning of M.
  - **Show:** symbols defined in the text; the notation table moves to Appendix B.
  - **Take-home:** Tokens are concepts of the ontology itself, so the vocabulary stays interpretable.
- **3.2 Candidate pool**
  - **Main idea:** the forward D-hop neighbourhood of M, with the root removed. One sentence replaces the former Figure 3 (concept-set diagram, deleted).
  - **Show:** no figure.
  - **Take-home:** Only concepts reachable within D hops can carry meaning for a concept in M.
- **3.3 Semantic coverage objective with distance decay**
  - **Main idea:**
    - Eq. 1: each hop multiplies the score by λ.
    - The objective is monotone submodular for any decay in [0, 1] (Appendix A).
    - **Hop distance can be adapted to properties of the DAG, such as concept status.** For example, a hop into a fully defined concept could be free, while a hop into a primitive concept costs λ. The guarantee holds for any per-hop factor in [0, 1].
  - **Show:** Eq. 1, the submodularity statement, one sentence on the status-dependent adaptation.
  - **Take-home:** The objective credits each walk that reaches a token, discounted by the meaning lost on the way, and a hop's cost can follow the ontology without losing the guarantee.
- **3.4 Lazy greedy selection**
  - **Main idea:**
    - greedy with the (1 − 1/e) guarantee;
    - lazy evaluation;
    - incremental gains (CandidateDelta) and exact closed-form initial gains;
    - complexity in 2–3 sentences;
    - prefix property: one run serves every k.
  - **Show:** text only. **Algorithms 1–2 (SelectTokens, CandidateDelta/Commit) move to Appendix B.**
  - **Take-home:** Tokens are selected with a quality guarantee, in minutes, and one run gives the vocabulary for every budget.
- **3.5 Context trees and evaluation metrics**
  - **Main idea:**
    - **Context trees:** IS_A is transparent, attribute edges open subcontexts, and a branch stops at the first token.
    - **Metrics**, each defined per concept, then averaged over M:
      - **proximity-coverage score** PC_m = c_m × p_m, with branch coverage c_m = n_m / (n_m + u_m) and proximity p_m = 1 − mean token depth / (D + 1). Equivalently, PC_m = 1 − mean depth over all branches / (D + 1), with uncovered branches counted at D + 1;
      - unique rate;
      - unknown rate;
      - tokens per concept;
      - uncovered branch rate (secondary).
    - **Distributions:** PC_m and tokens per concept are reported by their median, interquartile range and 10th–90th percentiles. The unique and unknown rates are yes/no per concept, so the mean suffices.
  - **Show:**
    - **Figure 3:** the former Figures 4 and 5 merged into one: why trees are needed (laterality example) and Expand traced on a toy graph, with the worked metric example (2/3 × 1/3 = 0.22).
    - Algorithm 3 (Expand), now **Algorithm 1**.
    - **Table 2:** the 5 metrics. The tree complexity, uniqueness entropy and semantic coverage rows are removed.
  - **Take-home:** Recording which relation leads to which token separates concepts that a flat token set would merge, and no single number, or single mean, captures a good tokenization.
- **3.6 Compared selection methods**
  - **Main idea:** 19 configurations on the same candidate pool:
    - ours at λ = 0, 0.1, …, 1.0;
    - the set-cover ablation;
    - 7 baselines.
  - **Show:** **Table 3**, compact, with one-line descriptions.
  - **Take-home:** All methods share the same candidate pool and differ only in how they rank candidates, so differences isolate the selection rule.
- **3.7 Application to EHR data: rare-concept coverage** — about 0.5 page
  - **Main idea:**
    - concepts of the HERO data, with frequencies from event counts;
    - tokenizer selected on M_HERO;
    - compared with frequency truncation (top-k + UNK).
    - **Measures, by frequency bin:**
      - concepts represented;
      - PC score (truncation: 1 if kept, 0 if UNK);
      - borrowed strength.
    - **Secondary:** events represented, tokens per event.
  - **Show:** definitions only. Drop the long definitions of discriminativity and sequence cost.
  - **Take-home:** On real data, a tokenizer must be judged on how it treats the rare concepts, not only on how many events it keeps.
- **3.8 Expert evaluation protocol** — about 0.5 page [`(final)_expert_evaluation_framework.ipynb`]
  - **Main idea:**
    - **Compared:** ours (λ = 0.9, k = 5,000) vs Personalized PageRank. It is the best baseline at the high-compression level of the main ranking, which contains k = 5,000; the rule is fixed in advance.
    - **Concepts:** fully defined *and* primitive concepts are included, to avoid a biased selection. Primitive concepts are judged against their stated definition, and raters can flag "definition insufficient".
    - **Inclusion:** concepts whose two trees differ.
    - **Categories:** `ours_token`, `baseline_token`, `both_decompose`. The categories with identical trees count as "no difference".
    - **Sample:** 100 concepts per expert after a pilot. One shared form for all experts (two or more semanticians) and one hidden key.
    - **Task:** A better / no clear difference / B better, plus problem checkboxes: incorrect or ambiguous, information absent, precision lost, redundant.
    - **Analysis:**
      - preference per category and weighted by category size;
      - sign test and bootstrap confidence intervals;
      - rater agreement;
      - agreement with the difference in PC score.
  - **Show:** text only. The inclusion funnel, category sizes and form screenshot go to Appendix G.
  - **Take-home:** The expert evaluation compares the two best methods, blinded, on fully defined and primitive concepts, and covers every way the two outputs can differ.

## 4. Results (`04_results.tex`) — 7.5 pages
- **4.1 Experimental setup** — 0.5 page
  - **Main idea:**
    - M: 46,150 concepts;
    - G_D: 75,964 nodes, 293,146 edges, 103 of 113 relation types;
    - budgets k = 500–19,500 in 3 compression levels;
    - HERO: 16,370 concepts, 279M events, long-tailed.
  - **Show:** **Table 4**, merged from the former Tables 5 (graph statistics) and 7 (compression levels), with the HERO statistics added. The relation types absent from G_D are listed in Appendix C.
  - **Take-home:** Every method is evaluated on the same concepts, candidate pool and budgets, and the HERO data adds the real frequencies that the benchmark lacks.
- **4.2 Selection methods on all mapped concepts** — 3.5 pages
  - **Main idea:** four paragraphs.
    1. *Example context trees*, illustrative, λ = 0.9, k = 5,000; 3–4 sentences instead of the full walkthrough:
       - Pain of breast has a unique tree;
       - Hyperthyroidism and Sick-euthyroid syndrome share one tree.
    2. *Ranking:* average rank over the 4 core metrics at each compression level. λ = 0.9 is first at every level, and the ablation ranks below every λ > 0.
    3. *Effect of k:* mean trends and distributions; diminishing returns; whether the gain comes from the whole distribution or from the tail.
    4. *Effect of λ:* the distribution of PC_m by λ. State the result: λ = 0 fails, 0.6–0.99 is a plateau, there is a drop at exactly λ = 1. Point to 5.1 for the explanation, which is given only there.
  - **Show:**
    - **Figure 4:** example trees. Key per-concept values go in the caption; the former Table 6 is deleted.
    - **Figure 5:** ranking by family. The former Table 8 (top-5) moves to Appendix D, next to the full ranking.
    - **Figure 6:** effect of k, one figure for means and distributions:
      - PC score and tokens per concept as median + IQR + P10–P90 bands;
      - unique and unknown rates as lines;
      - ours (λ = 0.9) and the best baselines.
    - **Figure 7:** effect of λ, box plots of PC_m at one k per compression level, plus the mean unique rate and tokens per concept. The former Table 9 moves to Appendix D.
  - **Take-home:** Distance-aware greedy selection (λ strictly between 0 and 1) outranks every degree, centrality and set-cover baseline at every compression level. Its gain shows in the whole distribution of the proximity-coverage score, not only in the mean.
- **4.3 Expert evaluation** — 1.5 pages
  - **Main idea:**
    - category sizes;
    - expert preference per category and weighted overall;
    - problems ticked per method;
    - rater agreement;
    - agreement with the PC score.
  - **Show:**
    - **Table 5:** per category: size, preference (ours / no difference / baseline), problems;
    - **Figure 8:** stacked bars per category.
  - **Take-home (to confirm with the results):** Semanticians judge our representations at least as faithful to the definition as the best baseline's, and the categories show where each method loses meaning.
- **4.4 Application to HERO data: coverage of rare concepts** — 1.5 pages
  - **Main idea:**
    - **Core, by frequency bin at k = 5,000:**
      - concepts represented: ours 100% in every bin; truncation 0% below 100 events;
      - PC score of rare concepts: ours about 0.82, truncation 0;
      - borrowed strength, in one sentence with numbers.
    - **Over k:** share of rare concepts represented.
    - **Trade-offs, in one paragraph:**
      - truncation keeps more meaning per *event*;
      - ours needs more tokens per event.
  - **Show:**
    - **Figure 9:** 3 panels instead of 6: (a) concepts and events represented vs k; (b) rare concepts represented vs k; (c) tokens per event vs k.
    - **Table 6:** frequency bins (the former Table 11) with the PC column.
    - Moved to Appendix F: the former Table 10 (all budgets × 11 columns) and the borrowed-strength figure (the former Figure 11).
    - Dropped from the main text: the discriminativity analysis (unique tree vs flat).
  - **Take-home:** Truncation discards the rare concepts of the HERO data entirely, whereas our tokenizer represents every one of them through frequently seen tokens with most of their meaning retained, at the cost of longer sequences.

## 5. Discussion (`05_discussion.tex`) — 2.0 pages
- **5.1 Why it works**
  - **Main idea:**
    - hubs vs marginal gain plus decay;
    - both mechanisms are needed (ablation);
    - **the two λ extremes are explained here only**; 4.2 just states the result.
  - **Show:** pointers to Figure 5 and Appendix D.
  - **Take-home:** Useful tokens are general enough to be shared but close enough to keep meaning, and only decay combined with marginal gain finds them.
- **5.2 Implications for EHR foundation models and practical guidelines**
  - **Main idea:**
    - rare-concept coverage vs event coverage;
    - a hybrid vocabulary;
    - sequence length and pooling;
    - relation markers, in one sentence.
    - **Guidelines**, condensed into one closing paragraph (the former 5.3) without repeating the table numbers:
      - choose k from data volume and compute;
      - leave λ at 0.9;
      - one run serves every k;
      - selection needs only the list of concepts.
  - **Show:** pointers to Section 4.4.
  - **Take-home:** The tokenizer decides whether rare concepts reach the model at all; choose k from your data and compute, and leave λ at 0.9.
- **5.3 Evaluating a graph tokenizer: metrics, distributions and experts**
  - **Main idea:**
    - no method wins every metric;
    - tokens per concept must be read together with the unknown rate;
    - means hide the distribution;
    - whether the experts agree with the PC score.
  - **Show:** pointers to Figures 6–8; metric correlations in Appendix E.
  - **Take-home:** Judging on a single mean metric would pick the wrong tokenizer, and expert ratings show how far the automatic metrics can be trusted.

## 6. Limitations and future work (`06_limitations.tex`) — 0.8 page
*(The former Sections 6 and 7 merged into one list, each limitation followed by its next step. The draft notes, "Section ??" references and the informal text in 6.1 are removed; the former 7.3 is now covered by 3.3.)*
- **Main idea:**
  - **Incomplete definitions and primitive concepts.** Next step: status-dependent hop costs (3.3).
  - **Transductive selection, and greedy never removes a token.** Next step: token swaps.
  - **One site, one ontology; no downstream validation.** Next steps: pretraining experiments, other cohorts, inter-ontology tokenization.
  - **Expert evaluation size.**
  - **DAG assumption.**
- **Show:** no new results.
- **Take-home:** The results establish representation quality, supported by experts and by one real EHR dataset, not yet a downstream benefit.

## 7. Conclusion (`07_conclusion.tex`) — 0.3 page
- **Main idea:** one paragraph restating the contribution and the next test (pretraining).
- **Take-home:** Ontology-driven tokenization makes compact, auditable vocabularies possible, and its value for foundation models is the next test.

## Appendices (`appendix.tex`)
- A: Submodularity proof for a general per-hop factor, and the closed form of the initial gains.
- B: Notation table (the former Table 2); Algorithms SelectTokens and CandidateDelta/Commit (the former Algorithms 1–2).
- C: Relation types absent from G_D (the former Table 12).
- D: Full ranking (the former Table 13) with the top-5 table (the former Table 8); metrics by λ (the former Table 9).
- E: Random selection and the uncovered branch rate vs k (the former Figure 12); metric correlations by family.
- F: HERO: all budgets (the former Table 10); borrowed strength (the former Figure 11); uniqueness, tree vs flat.
- G: Expert evaluation materials: instructions, a form screenshot, the inclusion funnel, category sizes, pilot feedback.

---

## What moved where (from the 15 October draft)

| Draft item | Now |
|---|---|
| 2.1 DAG definitions | merged into 3.1 |
| Table 2 notation | Appendix B |
| Figure 3 concept sets | deleted (one sentence in 3.2) |
| Algorithms 1–2 | Appendix B |
| Figures 4 + 5 | merged into Figure 3 |
| 3.6 eight metrics, Table 3 | five metrics, Table 2 |
| Table 5 + Table 7 | merged into Table 4 |
| Table 6 example values | Figure 4 caption |
| Table 8 top-5 ranking | Appendix D |
| Table 9 metrics by λ | Appendix D |
| Figure 10 (6 panels) | Figure 9 (3 panels) |
| Table 10 HERO all budgets | Appendix F |
| Figure 11 borrowed strength | Appendix F |
| Table 11 frequency bins | Table 6 (+ PC column) |
| λ extremes explained in 4.5 and 5.1 | 5.1 only |
| 5.3 practical guidelines | one paragraph in 5.2 |
| Sections 6 + 7 | merged into Section 6; Section 7 is now a short conclusion |
| new | 3.8 and 4.3 expert evaluation; distributions in Figures 6–7 |
