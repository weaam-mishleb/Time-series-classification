# STF — Sparse Traffic Former: Frozen Specification

**Status: S0 and S1 frozen 2026-08-05. S2–S6 outlined, to be frozen before implementation.**

Design authority: this document. Any deviation during implementation is a spec change
and must be recorded here first — not discovered in the code later.

---

## 0. Global contracts

Every stage Sn is a **drop-in replacement** for S(n−1). This is what makes the growing
baseline work: the encoder and head never change when the tokeniser does.

### Tensor contract (invariant across all stages)

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `x` | `(B, T, F)` | float32 | F = 8 slot features (14 after Phase 1c) |
| `pad_mask` | `(B, T)` | bool | **True = PAD** (PyTorch `key_padding_mask` convention) |
| `feat_mask` | `(B, T, F)` | bool | **True = feature present.** S1+ only |
| `logits` | `(B, C)` | float32 | C = 15 (UTMobile) / 18 (CESNET) |

`T` is window-dependent: 21 (250ms) … 101 (50ms) … 251 (20ms).
Until Phase 1c restores short flows, `pad_mask` is all-False — but **it is threaded
through from S0 anyway**, so restoring short flows later is a data change, not an
architecture change.

### Tokeniser contract

Every stage's input module maps `(B, T, F) → (B, T, d_model)`. Nothing downstream is
aware of which stage produced it.

### Parameter budget

**Target 0.5M–2M parameters.** Justified by measurement, not taste:

| Baseline | Params | Mean val acc |
|---|---|---|
| TimesNet | 9,472,911 | 93.84% |
| Informer | 3,068,432 | 93.23% |
| **NST** | **226,925** | **93.42%** |

NST reaches 93.42% with **227K parameters** — 42× smaller than TimesNet for 0.4 points
less. On 3,377 flows, capacity is demonstrably not the bottleneck. Any STF stage that
needs >2M parameters to win is answering the wrong question, and a reviewer will ask
whether the gain is architecture or capacity (hence the parameter-matched control in
the ablation plan).

---

## 1. Stage S0 — Vanilla encoder *(the correctness gate)*

**Purpose: not to be novel.** S0 exists to prove the harness, the data path and the
training loop are correct before any novel component can hide a bug in them.

### Module stack

```
x (B,T,8)
  └─ InputProjection      Linear(8 → d_model)                  (B,T,d)
  └─ + PosEmbedding       learned absolute, Parameter(T_max,d)  (B,T,d)
  └─ prepend CLS          Parameter(1,1,d)                     (B,T+1,d)
  └─ Dropout(p)
  └─ Encoder × N          Pre-LN [ MHSA → FFN ]                (B,T+1,d)
  └─ take CLS             z[:,0]                               (B,d)
  └─ Head                 LayerNorm → Linear(d → C)            (B,C)
```

### Encoder layer (Pre-LN — required, not optional)

```
h = h + Dropout(MHSA(LN(h), key_padding_mask=pad_mask_with_cls))
h = h + Dropout(FFN(LN(h)))          FFN = Linear(d,d_ff) → GELU → Dropout → Linear(d_ff,d)
```

Pre-LN is specified because we train **from scratch on 3,377 flows**; Post-LN needs
aggressive warmup to remain stable at this scale.

### Hyperparameters (S0 initial)

| Param | Value | Rationale |
|---|---|---|
| `d_model` | 128 | between NST (64) and Informer (256) |
| `n_layers` | 4 | |
| `n_heads` | 8 | `d_head` = 16 |
| `d_ff` | 256 | 2× `d_model` — conservative for small data |
| `dropout` | 0.1 | |
| `stochastic_depth` | 0.0 in S0 | introduced at S4 |
| optimiser | AdamW, wd 0.05 | |
| schedule | cosine, 5–10% warmup | |
| grad clip | 1.0 | |
| loss | CE + label smoothing 0.05 | |

**Estimated parameters ≈ 560K** — squarely in the NST/Informer band.

### Padding and CLS interaction

The CLS token is prepended **after** positional embedding and must never be masked.
`pad_mask` is therefore left-padded with one `False` column before reaching attention.
Specified explicitly because it is the single most common source of silent bugs in
hand-rolled encoders.

### ✅ S0 acceptance criteria — the gate

All three must pass before S1 begins:

1. **Tiny-batch overfit.** 32 samples → ≥95% train accuracy within 200 steps. If this
   fails the model is broken; nothing else is worth debugging.
2. **Shape/gradient tests.** Every parameter receives a finite, non-zero gradient;
   padded positions receive none.
3. **Parity with baselines.** On UTMobile @50ms under the corrected protocol, S0's val
   Macro-F1 is **within noise of NST** — not statistically worse by McNemar at α=0.05.

> 🔴 **If criterion 3 fails, stop and fix the harness.** A plain encoder that cannot
> match a 227K-parameter baseline means the problem is in our data path or training
> loop, not in our ideas. Discovering that at S0 costs days; discovering it at S5 costs
> the paper.

---

## 2. Stage S1 — Per-feature embedding + masked feature attention

**Purpose:** replace `Linear(8 → d)` on the concatenated vector with a per-feature
representation. This is the enabling layer for three of the paper's contributions —
leave-features-out robustness, masked training, and exact SHAP all require the model
to accept *feature subsets* at inference.

### Why the concatenated projection is wrong for us

`Linear(8 → d)` mixes eight semantically unrelated quantities — packet counts (integer,
heavy-tailed), byte volumes (heavy-tailed), mean payload (zero-inflated, and **zero
with packets present means pure-ACK**), a bounded ratio, and monotonic time — into one
weight matrix at step one. There is then no way to remove a feature at inference, and
no way for the model to represent "this feature is absent" as distinct from "this
feature is zero".

### Module stack (replaces InputProjection only)

```
x (B,T,8), feat_mask (B,T,8)
  └─ per-feature proj     W_f: (F,1,d)  →  e[b,t,f,:] = x[b,t,f] · W_f[f] + b_f
  └─ + feature-ID         E_id: Parameter(F,d), init N(0, 0.02²)
  └─ query build          q = Σ_f (e·m) / Σ_f m      masked mean over present features
  └─ MHA                  out = MHA(q, e, e, key_padding_mask=¬feat_mask)
  └─ output               (B,T,d)   ← identical shape to S0
```

The **feature-ID embedding is the direct analogue of ViTST's per-variable colour**,
which their ablation identified as the single most critical representational component
(removing it caused their largest drop, AUPRC 51.1 → 47.0 on P12). Without it, the
attention over eight feature-tokens is permutation-invariant and the model cannot tell
which quantity it is reading.

### 🔴 Edge case that must be designed for now, not patched later

If every feature in a slot is masked, attention is taken over an empty key set →
**NaN**. This is not hypothetical: it is exactly what an empty time slot looks like,
and empty slots are 29–69% of UTMobile and 92–98% of CESNET.

**Resolution: reserve a learned sentinel token now.** `e_sentinel: Parameter(1,d)` is
always appended to the key/value set and never masked. In S1 it is a NaN guard; **in
S2 it becomes the idle token**, so the mechanism is designed in from the start rather
than retrofitted.

### `feat_mask` semantics — the substrate for three contributions

| Use | Mask pattern | Stage |
|---|---|---|
| All features present (default) | all True | S1 |
| Leave-features-out evaluation | drop feature f across all t | Phase 4 |
| Masked training | random feature groups × time spans | Weeks 8–9 |
| SHAP coalitions | drop a coalition, all t | Phase 4 |
| Idle slot | all False → sentinel only | S2 |

### Parameter cost over S0

| Component | Params (d=128, F=8) |
|---|---|
| Per-feature projection | ~2,048 |
| Feature-ID embedding | 1,024 |
| Slot-attention MHA | ~65,536 |
| Sentinel | 128 |
| **Δ vs S0** | **≈ +69K** |

### ✅ S1 acceptance criteria

1. Output shape and dtype byte-identical in contract to S0 (drop-in verified by test).
2. Tiny-batch overfit still passes.
3. `feat_mask` all-True reproduces S0-comparable accuracy (sanity: masking machinery
   adds no regression when nothing is masked).
4. An all-masked slot produces finite output (sentinel guard verified).
5. Not statistically worse than S0 by McNemar.

### Ablations owned by S1

- **S1 vs S0** — does per-feature representation help at all?
- **S1 minus feature-ID** — replicates ViTST's most critical ablation on our data.
- **Query construction**: masked mean vs learned query vector.

---

## 3. S2–S6 — outline only *(freeze before implementing each)*

| Stage | Adds | Primary justification |
|---|---|---|
| **S2** | Occupancy tokenisation: idle token (from S1's sentinel), `silence_run_len`, hard `pad_mask` vs soft `idle_flag` | zero-ratio 0.29–0.69 (UT) / 0.92–0.98 (CESNET) |
| **S3** | Directional factorisation + cross-direction attention | up/down are two sides of a conversation — **our main novelty claim** |
| **S4** | W-MSA / SW-MSA on the time axis (w≈8, shift w/2) + stochastic depth 0.1 | burst-local vs burst-distant structure |
| **S5** | Hierarchical patch merging, `Linear(2d → d)` per stage | replaces 9-models-for-9-windows; CESNET's 34-point window collapse |
| **S6** | Relative position bias per head + Time2Vec on Δt | intervals identify applications, absolute position does not |

---

## 4. Frozen decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | Pre-LN | from-scratch stability at n=3,377 |
| D2 | CLS pooling in S0; attention-pooling added at S5 | keep S0 minimal; CLS is the control |
| D3 | `pad_mask` threaded from S0 | short-flow restoration stays a data change |
| D4 | `True = PAD`, `True = present` for `feat_mask` | matches PyTorch; documented to prevent inversion bugs |
| D5 | Sentinel token reserved at S1 | NaN guard now, idle token at S2 |
| D6 | Parameter budget 0.5–2M | NST hits 93.42% with 227K |
| D7 | Learned absolute PE in S0 | deliberately replaced at S6 — the ablation is the point |
| D8 | Every stage independently shippable | S4 alone is a complete paper if S5 destabilises |
