"""
Persistent train/validation split registry.

Until now the split was re-derived on every run from
`train_test_split(X, y, test_size=0.2, random_state=32, stratify=y)`. That is
reproducible only for as long as nothing upstream changes: adding a feature column,
reordering rows, dropping a flow, or a scikit-learn version bump can all silently
produce a different partition. Every baseline comparison would then be measured against
a different validation set without anyone noticing.

This module derives the split once, writes the row indices to JSON, and loads that file
on every subsequent run. The split becomes a versioned artefact rather than a side
effect of a random seed.

The initial file is bit-identical to what the old inline call produced, so numbers
recorded before this change remain comparable.
"""
import hashlib
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

# splits/ lives at the repository root, next to Time-Series-Library/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.environ.get('TS_SPLITS_DIR',
                            os.path.join(os.path.dirname(_THIS_DIR), 'splits'))


def _label_fingerprint(y):
    """Hash of the label vector: detects that the underlying rows changed."""
    return hashlib.sha256(np.asarray(y).astype(np.int64).tobytes()).hexdigest()[:16]


def split_path(dataset_name, num_class, n_samples, tag='holdout20'):
    fname = f"{dataset_name}_{num_class}cls_{n_samples}samples_{tag}.json"
    return os.path.join(SPLITS_DIR, fname)


def get_or_create_split(y, dataset_name, num_class=None, val_split=0.2,
                        random_state=32, tag='holdout20', verbose=True):
    """
    Return (train_idx, val_idx) for `y`, loading from disk when a split already exists.

    The split is keyed on (dataset_name, num_class, n_samples), deliberately NOT on window
    size: windows with the same row count hold the same flows in the same order, so they
    must share one split. Keying per window would give each one a different validation set
    and make cross-window comparison invalid.

    UTMobile windows are not all the same length, however: 20/40/50/100/200/250ms have 3377
    flows, 30/75ms have 3376, and 150ms has 3372. Those groups therefore get separate split
    files, which is the safe outcome — a 3377-row split must never be applied to 3376 rows.
    It does mean comparisons *between* row-count groups are not on identical validation
    flows. That was equally true before this module existed, since the old inline call
    re-split per window, so pre-freeze numbers stay comparable.

    Raises if the stored split does not match the data it is being applied to.
    """
    y = np.asarray(y)
    n = len(y)
    num_class = int(num_class if num_class is not None else len(np.unique(y)))
    path = split_path(dataset_name, num_class, n, tag)
    fingerprint = _label_fingerprint(y)

    if os.path.exists(path):
        with open(path) as fh:
            rec = json.load(fh)
        train_idx = np.array(rec['train_idx'], dtype=np.int64)
        val_idx = np.array(rec['val_idx'], dtype=np.int64)

        # Fail loudly rather than train on a mismatched split.
        if len(train_idx) + len(val_idx) != n:
            raise ValueError(
                f"Stored split {path} covers {len(train_idx) + len(val_idx)} rows "
                f"but the data has {n}. Delete the file to re-derive it.")
        if rec.get('label_fingerprint') != fingerprint:
            raise ValueError(
                f"Stored split {path} was built for a different label vector "
                f"(fingerprint {rec.get('label_fingerprint')} != {fingerprint}). "
                "The rows underneath have changed; delete the file to re-derive it.")
        if verbose:
            print(f"[splits] loaded {os.path.basename(path)} "
                  f"({len(train_idx)} train / {len(val_idx)} val)")
        return train_idx, val_idx

    # Only the original stratified split may be derived on demand. grouped20/control20 are
    # built by grouped_split.py from capture provenance and cannot be reconstructed here;
    # falling through would hand back a plain random split under their name, and the run
    # would be reported as B2 while measuring nothing of the sort.
    if tag != 'holdout20':
        raise FileNotFoundError(
            f"No split file at {path} for tag '{tag}'. This tag must be generated in "
            "advance (python grouped_split.py --write) — it is never derived on the fly, "
            "because a silently-substituted random split would invalidate the experiment.")

    # First time: derive exactly as the old inline call did. train_test_split computes
    # the partition from n_samples and `stratify` alone, so splitting an index vector
    # reproduces the partition the old call applied to X and y.
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, random_state=random_state, stratify=y)

    os.makedirs(SPLITS_DIR, exist_ok=True)
    rec = {
        'dataset_name': dataset_name,
        'num_class': num_class,
        'n_samples': int(n),
        'val_split': val_split,
        'random_state': random_state,
        'tag': tag,
        'strategy': 'stratified_random',
        'label_fingerprint': fingerprint,
        'note': ('Frozen from the original inline train_test_split so pre-freeze '
                 'results stay comparable. Random flow-level split: same-session '
                 'captures may appear on both sides. Superseded by the grouped '
                 'split (tag=grouped) once available.'),
        'train_idx': [int(i) for i in train_idx],
        'val_idx': [int(i) for i in val_idx],
    }
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    if verbose:
        print(f"[splits] created {os.path.basename(path)} "
              f"({len(train_idx)} train / {len(val_idx)} val)")
    return train_idx, val_idx
