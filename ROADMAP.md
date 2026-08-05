# ROADMAP — A Custom Transformer for Encrypted Traffic Classification

**Revised 2026-08-05.** Paper reframed by the professor from *benchmark study* to
**novel architecture**. The custom model is the scientific contribution; TSLib
(TimesNet / NST / Informer) becomes the baseline suite.

Working name: **STF — Sparse Traffic Former** *(placeholder)*.

---

## 1. The one thing that must be settled first

A novel-architecture paper has to demonstrate the architecture is **better**. Our
measured noise floor on UTMobile is **±2 points** (n_val ≈ 675). Our best baseline is
**TimesNet at 94.97%**.

**On UTMobile alone, a novel architecture cannot be shown to beat that baseline.** A
+1.5 point gain would sit inside the confidence interval. This is arithmetic, not
pessimism.

Two consequences, both structural:

### 1.1 🔴 CESNET is now mandatory, not optional

In the previous plan CESNET was a strategic preference. It is now a **requirement for
the paper to be publishable**. At 1.07M flows, 1–2 point effects are resolvable; on
UTMobile they are not. CESNET is also where the architecture's core mechanisms —
sparsity handling (zero-ratio 0.92–0.98 vs UTMobile's 0.29–0.69) and multi-resolution
(the 34-point window collapse) — are actually justified.

**If CESNET is out of scope, the novel-architecture framing does not work.** That
needs raising with the professor now, not in week 10.

### 1.2 🟠 Plan for the case where STF does not beat TimesNet

TimesNet at 94.97% on 3,377 flows is a strong, well-tuned baseline. A from-scratch
model on this much data may not beat it, and no amount of runway guarantees it will.
The paper therefore needs contributions that **do not depend solely on the accuracy
number**:

| Contribution | Depends on beating TimesNet? |
|---|---|
| Robustness under feature loss (leave-features-out curve) | ❌ no |
| One model across all resolutions vs 9 separate models | ❌ no — efficiency claim |
| Exact SHAP explainability, sampling-free | ❌ no |
| Cross-dataset self-supervised transfer (CESNET → UTMobile) | ❌ no |
| Early classification (accuracy vs observation length) | ❌ no |
| Accuracy on CESNET | ✅ resolvable at n=1.07M |
| Accuracy on UTMobile | ⚠️ inside the noise floor |

Designing the paper so **five of seven contributions survive** a neutral accuracy
result is the single most important derisking decision available.

---

## 2. Architectural philosophy

Every mechanism below is justified by a **measured** property of our data, not by
what is fashionable. That mapping is the paper's argument.

| Measured property | Value | Design response |
|---|---|---|
| Empty slots are frequent and *meaningful* | zero-ratio 0.29–0.69 (UT), 0.92–0.98 (CESNET) | Occupancy-aware tokenisation §2.1 |
| 8 heterogeneous features per slot | counts, bytes, payload, ratio, time | Per-feature embedding + masked feature attention §2.2 |
| Up/down are distinct semantics | separate columns, asymmetric | **Directional factorisation §2.3** |
| Optimal resolution is dataset-dependent | 34-pt collapse (CESNET), 2.3 (UT) | Hierarchical multi-scale encoder §2.4 |
| Intervals matter, absolute position doesn't | burst signatures | Relative bias + continuous time §2.5 |
| Small data | 3,377 flows | Parameter budget §2.6 |

### 2.1 Occupancy-aware tokenisation — *sparsity as signal, not missingness*

One token per time slot. Occupancy is first-class:

- **Learned `idle` embedding.** When a slot has no packets, the feature projection is
  *replaced* by a single learned vector — not fed a zero vector. Zeros currently
  consume attention and value-vector capacity to represent nothing.
- **`silence_run_len`** as an input feature — distinguishes regular buffering silence
  from bursty browsing silence.
- **Two distinct masks**, and keeping them separate matters:
  - `pad_mask` → **hard** `-inf` masking, for padding past the flow's end. Never
    attended, never in the loss.
  - `idle_flag` → **soft** signal via the learned idle token. Empty ≠ absent.
- **Pure-ACK preservation.** `packet_count > 0 ∧ avg_payload == 0` is a real,
  discriminative state (18.5% vs 31.5% of slots at 20ms). The tokeniser must not
  collapse it into "empty".

### 2.2 Per-feature embedding + masked feature attention

From SHAPformer. Replaces `Linear(8 → d_model)` on the concatenated vector.

- Independent projection per feature; **feature-ID embedding** — ViTST's ablation
  found the variable-identity signal (their "colour") to be the single most critical
  component.
- Attention across the 8 feature embeddings produces the slot representation.
- **Why it matters beyond accuracy:** feature-level maskability is the substrate for
  the leave-features-out curve, for masked training, and for exact SHAP. Three of our
  seven contributions depend on this one layer.

### 2.3 Directional factorisation — *our most defensible novelty*

Upstream and downstream are not interchangeable channels; they are two sides of a
conversation. Generic time-series Transformers have no notion of this.

- Separate embedding streams for `dir_-1` and `dir_+1` features.
- **Cross-direction attention** at each slot: the up-stream queries the down-stream
  and vice versa, modelling request→response coupling directly.
- Rationale: what separates interactive traffic (chat, browsing) from bulk transfer
  (upload, streaming) is precisely the *coupling structure* between directions —
  currently compressed into a single scalar `upstream_downstream_ratio`.

This is the mechanism least reducible to "Swin applied to time series", and the one I
would foreground as the architectural contribution.

### 2.4 Hierarchical multi-scale encoder

- Alternating **W-MSA / SW-MSA** on the time axis (window ~8 slots, shift w/2):
  non-shifted layers capture burst-local dynamics, shifted layers link distant bursts.
- **Patch merging** after each stage — two adjacent slots merged, `Linear(2d → d)`.
  After 3 stages, one token spans 8 slots: a model trained at 20ms *sees* 160ms in its
  deep layers.
- **Directly replaces the 9-models-for-9-windows design.** Even at parity accuracy
  this is a legitimate efficiency contribution, and it is the answer to CESNET's
  34-point window collapse.

### 2.5 Position and time encoding

- **Learned relative position bias** per head (Swin-style `B[i-j]`), not absolute
  sinusoidal — a burst at t=1.2s and t=3.4s share a signature if the interval matches.
- **Time2Vec on Δt.** `time_from_start` is currently used only for indexing and then
  discarded; as a continuous encoding it makes the model robust to padding,
  truncation and varying resolution.
- ViTST found positional additions gave no consistent gain — but they had pre-trained
  positional embeddings and we do not. **Measure, don't assume.**

### 2.6 Parameter budget

`d_model` 128–256, 4–6 layers, 4–8 heads, dropout 0.1–0.2, stochastic depth 0.1,
Pre-LN. **Target ≤ 5M parameters.** With 3,377 flows, capacity is the enemy; every
mechanism must earn its parameters in the ablation.

### 2.7 Classification head

CLS token concatenated with attention-pooling over non-padded slots. More stable than
either alone on sparse sequences.

---

## 3. Implementation strategy — the growing baseline

**The central engineering decision.** Do not build STF as one artifact and debug it at
the end. Build it as a **sequence of measured increments**, starting from a minimal
Transformer that reproduces TSLib-level performance:

```
S0  vanilla encoder + linear input proj        → must match TSLib within noise
S1  + per-feature embedding & feature-ID       → §2.2
S2  + occupancy tokenisation (idle, masks)     → §2.1
S3  + directional factorisation                → §2.3
S4  + W-MSA / SW-MSA                           → §2.4
S5  + hierarchical patch merging               → §2.4
S6  + relative position bias & Time2Vec        → §2.5
```

Three properties make this the right approach:

1. **S0 is a correctness gate.** If a plain encoder cannot match TimesNet, the bug is
   in our harness, not our ideas. Catching that in week 3 rather than week 9 is the
   difference between a paper and a post-mortem.
2. **The ablation table is generated as a byproduct**, not retrofitted in a panic in
   week 11. Each increment is measured against the previous one when the code is
   fresh.
3. **Every stage is independently shippable.** If week 9 arrives and S5 is unstable,
   S4 is still a complete, defensible paper.

### Engineering hygiene (non-negotiable for novel code)

- Shape and gradient-flow unit tests per module.
- **Overfit-a-tiny-batch test** — 32 samples to ~100% train accuracy. Any new
  architecture that cannot is broken; this catches most bugs in minutes.
- Validate W-MSA/SW-MSA against a reference (`timm` Swin) on identical inputs.
- Attention-map visualisation from day one — the primary debugging instrument.
- Parameter count asserted in CI against the §2.6 budget.
- Fixed seeds; deterministic algorithms.

---

## 4. Twelve-week plan

### Weeks 1–2 — Foundation & design freeze
- Protocol fixes (grouped split, val Macro-F1) — **gates all measurement**
- Phase 1a: all 8 features *(needs no approval — start immediately)*
- Freeze the architecture spec; write the module interface contracts
- Re-baseline TSLib under the corrected protocol → the number STF is measured against
- **Decide CESNET scope (§1.1)**

### Weeks 3–5 — Core implementation (S0 → S3)
- S0 skeleton + harness correctness gate
- S1 per-feature embedding, S2 occupancy, S3 directional factorisation
- Re-preprocessing in parallel: tier-3 features (std/max, IAT, direction switches),
  short-flow restoration with `pad_mask`, `groups.csv`, CESNET → Parquet
- Unit tests and tiny-batch overfit at every stage

### Weeks 6–7 — Multi-scale & integration (S4 → S6)
- W-MSA/SW-MSA, patch merging, relative position bias, Time2Vec
- End-to-end training on UTMobile; hyperparameter search
- First honest STF-vs-TSLib comparison on the corrected protocol

### Weeks 8–9 — Pre-training & masked training
- **MSM pre-training** on ~1.07M CESNET flows *(not the 15M the master plan cites —
  that is the raw unparsed count)*. Contiguous span masking 4–8 slots, L1 on masked
  slots only, ratio computed over **active** slots.
- Fine-tune on UTMobile: layer-wise LR decay 0.75, reduced encoder LR.
- **Masked training** (SHAPformer): ramp 0.15 → 0.5, keep ~30% of batches unmasked.
  Budget the real cost — **12–13×**, measured from their Table 1, not the "2–10×" the
  master plan quotes.

### Weeks 10–11 — Ablation & scientific validation
- Component ablation S0→S6 (largely already collected)
- Leave-features-out robustness curve; early-classification curve
- Exact SHAP audit (51–807× faster than the Permutation Explainer, per SHAPformer)
- **Clever Hans audit of the UTMobile date/class confound** — our own finding
- Cross-dataset transfer results
- CESNET headline numbers

### Week 12 — Freeze & write
Code freeze; sealed test set opened **once**; figures, tables, write-up.

---

## 5. Ablation plan — the scientific core

For a novel-architecture paper the ablation table *is* the contribution. Every
mechanism must show it earns its parameters.

| Ablation | Question it answers |
|---|---|
| S0 → S6 incremental | Which component contributes what? |
| Idle token vs zero vector | Does treating sparsity as signal help? |
| Directional factorisation on/off | Is cross-direction attention worth it? *(the novelty claim)* |
| Hierarchical merging vs 9 separate models | Efficiency + accuracy claim |
| Relative bias vs absolute sinusoidal | Does interval-invariance help? |
| Feature-ID embedding removed | Replicates ViTST's most critical ablation |
| Masked training on/off | Robustness gain vs 12× cost |
| MSM pre-training on/off | Transfer gain — **negative result still publishable** |
| Parameter-matched TSLib control | Is the gain architecture or capacity? |

That last row matters and is easy to forget: if STF has more parameters than TimesNet,
a reviewer will ask whether the gain is architectural or just capacity. **Include a
parameter-matched control.**

Every comparison via **McNemar / paired bootstrap** on identical splits. At n≈675,
comparing independent mean±std cannot resolve a 1-point difference.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| STF doesn't beat TimesNet on UTMobile | **high** | high | §1.2 — five of seven contributions survive |
| CESNET stays out of scope | medium | **critical** | §1.1 — escalate now |
| Novel-code bugs eat weeks | **high** | high | Growing baseline; tiny-batch test; reference validation |
| Overfitting on 3,377 flows | **high** | medium | ≤5M params; MSM; masking-as-augmentation |
| S4/S5 unstable at week 9 | medium | medium | Every stage independently shippable |
| Masked training × CV blows budget | medium | low | Selected configs only, never the grid |
| Professor approval delays Phase 0 | medium | high | Weeks 1–2 feature work needs no approval |

---

## 7. Honest positioning of the novelty

So we don't overclaim to reviewers: **individually**, per-feature embedding, W-MSA/SW-MSA,
patch merging and relative position bias are established mechanisms adapted from
SHAPformer, Swin and ViTST. The genuine contributions are:

1. **Directional factorisation with cross-direction attention** — traffic-specific, not
   a port of anything (§2.3).
2. **Occupancy-aware tokenisation** treating silence as a learned signal rather than
   imputed zeros (§2.1).
3. **The combination**, targeted at encrypted traffic, with component-wise empirical
   evidence on two datasets.
4. **A traffic-adapted MSM objective** masking active slots rather than emptiness.

That is a legitimate architecture paper. Claiming the individual mechanisms are novel
is not, and a reviewer familiar with Swin will catch it immediately.

---

## 8. Immediate next actions

1. **Escalate the CESNET question (§1.1)** — the framing depends on it.
2. **Start Phase 1a (all 8 features)** — no approval needed.
3. **Freeze the architecture spec** before any module is written.
4. Confirm the S0 correctness gate criterion: *a plain encoder matches TSLib within
   noise before a single novel component is added.*
