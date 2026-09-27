# Timestamp representation and small-window boundary arithmetic

**This differs from the older float-based UTMobile packages and must be read before any
comparison against them.** The numerical outputs happen to coincide (see
"Relationship to the older packages"), but the implementation and its guarantees do not.

## 1. Source timestamps

The raw UTMobileNet2021 CSV archive has **no `frame.time_epoch` column**. The only
timestamp field is `frame.time`, a formatted local-time string, for example:

```
Apr 30, 2019 07:51:15.381121000 CDT
```

Every timestamp in every capture file was scanned before choosing a representation:

| measurement | value |
|---|---|
| timestamps scanned | 23,962,386 (all 3,783 capture files) |
| values with no parsable fractional part | 0 |
| fractional-digit lengths observed | `{9: 23,962,386}` — all nine digits |
| finest non-zero decimal place anywhere | **1e-6 s** |
| values finer than one microsecond | **0** |

The files are written with nine fractional digits, but the nanosecond positions are
always `000`. This is microsecond-resolution capture padded to nanosecond formatting.

## 2. Stored representation

Packet times are stored as **exact `int64` nanoseconds** relative to the flow's first
packet. No floating-point value takes part in the timestamp chain at any point:

```
'Apr 30, 2019 07:51:15.381121000 CDT'
  -> drop the trailing timezone token
  -> pd.to_datetime(format='%b %d, %Y %H:%M:%S.%f')  -> datetime64[ns], all 9 digits kept
  -> ns        = int64 nanoseconds since the epoch
  -> elapsed   = ns - ns[first packet]               -> int64, stored as `t_ns`
```

The stored array is asserted to be `int64` at write time and again at read time. Because
the source is microsecond-resolution, the conversion is **exactly reversible at the
source precision**: every stored value satisfies `t_ns % 1000 == 0`, and dividing by 1000
recovers the microsecond value the file encodes with no rounding.

An earlier revision of this pipeline stored `float64` seconds and converted them with
`np.rint(t * 1e6)` at slot time. That revision was replaced: the conversion was measured
to be exact for this archive, but it placed a floating-point value in the chain, which
this implementation does not.

## 3. Slot assignment

```
T    = floor(D / w) + 1
slot = min(elapsed_ns // window_ns, T - 1)          # exact integer division
```

* a packet whose elapsed time is an **exact multiple of the window** enters the
  **UPPER** slot;
* a packet at exactly `elapsed == D` is **included**;
* eligibility is `duration_ns > D * 10**9`, a **strict** integer comparison.

### Why integer arithmetic is required

`floor()` on floating point is not safe at a boundary. `0.06 / 0.02` evaluates to
`2.9999999999999996`, which truncates to slot 2 instead of the required slot 3. Measured
on this population, a naive float computation would misplace:

| duration | packets misplaced by float floor |
|---|---|
| 3 s | 140 |
| 5 s | 182 |
| 8 s | 121 |
| **total** | **443** |

Integer division has no such failure mode; the boundary rule is exact by construction.

## 4. Verification

Two independent verifiers were run from scratch against the integer store. Neither
imports the feature-building code.

### 4a. Exhaustive window verification

Re-derives elapsed time, the eligible packet set, slot assignment and all eight features
using a plain per-slot Python accumulation loop (deliberately not `bincount` and not any
vectorised grouping), for **every duration, every one of the nine windows, and every
eligible flow**.

| | 3 s | 5 s | 8 s |
|---|---:|---:|---:|
| flow x window cells | 69,066 | 63,324 | 56,277 |
| slots verified | 3,921,414 | 5,938,384 | 8,416,538 |
| packet assignments verified | 71,103,213 | 87,203,727 | 72,966,078 |
| feature values compared | 31,371,312 | 47,507,072 | 67,332,304 |

Tolerances were declared before any result was seen: exactly 0 for indices 1, 4 and 7;
`atol 1e-6` for indices 0, 2, 3, 5 and 6; `3feat` bit-exact.

**Result: max |diff| = 0.000e+00 on all eight features at all three durations** — the
declared tolerances were never needed. Zero on every categorical check: packet
assignment, packet count, payload, packets past `D`, `3feat` bit-exactness, shapes, NaN,
Inf and negative values. The packet population is identical across all nine windows.

### 4b. Exact Decimal reference for slot assignment

Re-reads the **raw CSV strings** and converts them to exact integer nanoseconds by pure
string and integer arithmetic — no `pd.to_datetime`, no float, using a proleptic
Gregorian day-number computation so the **calendar date is included**. Slots are then
computed with `Decimal` division under `ROUND_FLOOR` and compared against production for
every included packet.

| | 3 s | 5 s | 8 s |
|---|---:|---:|---:|
| flow x window cells | 69,066 | 63,324 | 56,277 |
| packet slot assignments compared | 71,103,213 | 87,203,727 | 72,966,078 |
| **slot mismatches** | **0** | **0** | **0** |
| included-set mismatches | 0 | 0 | 0 |
| packet-count mismatches inside D | 0 | 0 | 0 |
| packets exactly ON a boundary | 70,444 | 64,984 | 57,638 |
| packets one microsecond BELOW a boundary | 1,368 | 1,605 | 1,301 |
| packets one microsecond ABOVE a boundary | 1,355 | 1,644 | 1,417 |

Across all 12,459 flows: **0** flows whose packet count differs from production, and
**0** flows whose stored nanoseconds differ from the exact reference
(**max |diff| = 0 ns**).

The boundary-adjacent rows matter most: 193,066 packets sit exactly on a slot boundary
across the three durations, and every one of them is placed in the upper slot by both
the production integer path and the exact Decimal reference.

## 5. Relationship to the older packages

The integer build was compared against the previous float build before the float
artifacts were deleted. All nine windows at all three durations were **bit-identical**
(`max |diff| = 0.000e+00`, `np.array_equal` true), with identical flow ordering.

So the older float-based results are not wrong for this archive — microseconds were
already lossless and the slot division was already integer. What changed is the
guarantee: this package carries **no floating-point value anywhere in the timestamp or
boundary chain**, and that property is verified against an exact reference rather than
assumed. A capture source with true sub-microsecond timestamps would break the old
chain and would not break this one.
