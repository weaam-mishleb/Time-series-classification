"""Build the 3 s / 5 s / 8 s feature tables from the gate-verified 12,459 biflows.

Rules, exactly as specified:
  * eligible flows: complete duration STRICTLY greater than D
  * packets used: cumulative elapsed <= D, taken from the start of the flow
  * T    = floor(D / w) + 1
  * slot = min(floor(elapsed / w), T - 1)
  * a packet exactly on a small-window boundary enters the UPPER slot
  * a packet exactly at elapsed == D is included
  * no padding, no second packet threshold inside D
  * the packet population is identical across all nine small windows

Slot arithmetic is done in EXACT INTEGER NANOSECONDS, taken straight from the stored
int64 packet times. No floating point is involved at any point in slot assignment. floor() on floats is not safe at a
boundary: 0.06 / 0.02 evaluates to 2.9999999999999996, which would truncate to slot 2
instead of the required slot 3. Integer arithmetic makes the boundary rule exact. The
number of packets that a naive float computation would misplace is measured and reported.

Feature layout, unchanged from the verified packages:
   0,1,2  FORWARD  total payload / count / mean payload   (legacy suffix dir_-1)
   3,4,5  REVERSE  total payload / count / mean payload   (legacy suffix dir_1)
   6      total_payload_forward / total_payload_reverse, 0 when the denominator is 0
   7      slot_index * window_seconds
FORWARD means "same source IP as the flow's first packet", never client/server.
3feat = 8feat[:, [1, 4, 6]].
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, 'reports')
WINDOWS_MS = [20, 30, 40, 50, 75, 100, 150, 200, 250]
DURATIONS = [3, 5, 8]
FEAT3 = [1, 4, 6]


def features(t_ns, size, fwd, w_ns, T):
    """(T, 8) for one flow at one window. t_ns is EXACT integer nanoseconds since packet 0."""
    slot = np.minimum(t_ns // w_ns, T - 1).astype(np.int64)
    out = np.zeros((T, 8), dtype=np.float64)
    for mask, base in ((fwd, 0), (~fwd, 3)):
        if mask.any():
            s = slot[mask]
            tot = np.bincount(s, weights=size[mask], minlength=T)[:T]
            cnt = np.bincount(s, minlength=T)[:T].astype(np.float64)
            out[:, base] = tot
            out[:, base + 1] = cnt
            out[:, base + 2] = np.divide(tot, cnt, out=np.zeros(T), where=cnt > 0)
    out[:, 6] = np.divide(out[:, 0], out[:, 3], out=np.zeros(T), where=out[:, 3] > 0)
    out[:, 7] = np.arange(T, dtype=np.float64) * (w_ns / 1e9)
    return out


def main():
    idx = pd.read_parquet(os.path.join(REP, 'b2_flows_index.parquet'))
    z = np.load(os.path.join(REP, 'b2_flow_packets_ns.npz'))
    t_all, s_all, f_all, off = z['t_ns'], z['size'], z['fwd'], z['offsets']
    assert t_all.dtype == np.int64, f'expected int64 ns, got {t_all.dtype}'
    print(f'population: {len(idx):,} flows   packets {int(idx.packets.sum()):,}')

    for D in DURATIONS:
        t0 = time.time()
        D_ns_full = D * 1_000_000_000
        eq = int((idx.duration_ns == D_ns_full).sum())              # exact integer test
        elig = idx.index[idx.duration_ns > D_ns_full].to_numpy()    # STRICT >, integer
        print(f'\n[{D}s] flows with duration == {D}.0 exactly : {eq}  (excluded by the strict rule)')
        print(f'[{D}s] eligible (duration > {D}) : {len(elig):,} of {len(idx):,}')

        D_ns = D * 1_000_000_000
        rows, feats = [], {w: [] for w in WINDOWS_MS}
        float_mismatch = 0
        pkt_per_window = {w: 0 for w in WINDOWS_MS}
        for fid in elig:
            a, b = off[fid], off[fid + 1]
            t_ns = t_all[a:b]                       # already exact int64
            keep = t_ns <= D_ns                     # packet exactly at D included
            tu, sz, fw = t_ns[keep], s_all[a:b][keep], f_all[a:b][keep]
            r = idx.iloc[fid]
            rows.append(dict(flow_id=int(fid), capture_id=r.capture_id,
                             partition=r.partition, app=r.app, proto=int(r.proto),
                             packets_full=int(r.packets), packets_in_D=int(keep.sum()),
                             payload_in_D=float(sz.sum()),
                             fwd_in_D=int(fw.sum()), rev_in_D=int((~fw).sum()),
                             duration_s=float(r.duration_s),
                             duration_ns=int(r.duration_ns), is_dns=bool(r.is_dns)))
            for w in WINDOWS_MS:
                w_ns = w * 1_000_000
                T = (D * 1000) // w + 1
                feats[w].append(features(tu, sz, fw, w_ns, T))
                pkt_per_window[w] += int(keep.sum())
                # how many packets a naive float floor would have misplaced
                naive = np.minimum((tu / 1e9 / (w / 1000.0)).astype(np.int64), T - 1)
                exact = np.minimum(tu // w_ns, T - 1)
                float_mismatch += int((naive != exact).sum())

        meta = pd.DataFrame(rows)
        out = {'meta': meta}
        for w in WINDOWS_MS:
            T = (D * 1000) // w + 1
            A = np.stack(feats[w]).astype(np.float64)
            assert A.shape == (len(meta), T, 8)
            out[f'w{w}'] = A
        np.savez_compressed(os.path.join(REP, f'features_{D}s.npz'),
                            **{k: v for k, v in out.items() if k != 'meta'})
        meta.to_parquet(os.path.join(REP, f'meta_{D}s.parquet'), index=False)
        same = len(set(pkt_per_window.values())) == 1
        print(f'[{D}s] flows {len(meta):,}  captures {meta.capture_id.nunique():,}  '
              f'packets in D {int(meta.packets_in_D.sum()):,}')
        print(f'[{D}s] identical packet population across all 9 windows: {same} '
              f'({sorted(set(pkt_per_window.values()))})')
        print(f'[{D}s] packets a naive float floor would misplace: {float_mismatch}')
        print(f'[{D}s] wrote features_{D}s.npz + meta_{D}s.parquet  ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
