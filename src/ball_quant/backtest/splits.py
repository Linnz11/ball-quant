"""
Walk-forward and rolling time-splits for backtest — NO lookahead guarantee.

The cardinal sin of backtesting is leaking future information into training.
Every split function here calls assert_no_lookahead before returning so the
guarantee is structural, not a matter of caller discipline.
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple


def assert_no_lookahead(
    train: list,
    test: list,
    key: Callable[[Any], Any],
) -> None:
    """Raise AssertionError if any test item precedes any train item in key order.

    A single point of failure is all we need: the latest key in train must be
    strictly <= the earliest key in test.  We use <= (not <) because snapshots
    captured at exactly the same timestamp are edge-case duplicates (e.g.
    test fixtures), not lookahead — the fold-boundary logic ensures test items
    are always drawn from later chunks than train items.
    """
    if not train or not test:
        # Empty folds are degenerate but not a lookahead violation.
        return
    max_train_key = max(key(item) for item in train)
    min_test_key = min(key(item) for item in test)
    if min_test_key < max_train_key:
        raise AssertionError(
            f"Lookahead detected: test item key {min_test_key!r} precedes "
            f"max train key {max_train_key!r}"
        )


def walk_forward_splits(
    items: list,
    key: Callable[[Any], Any],
    n_folds: int,
) -> List[Tuple[list, list]]:
    """Expanding-window walk-forward splits.

    Sorts *items* by *key*, partitions into n_folds+1 equal time-ordered
    chunks, and returns n_folds folds where fold i has:
        train = chunks[0] + ... + chunks[i]
        test  = chunks[i+1]

    Guarantees max key(train) <= min key(test) via assert_no_lookahead.
    Raises ValueError if n_folds < 1 or not enough items for n_folds+1 chunks.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    sorted_items = sorted(items, key=key)
    n_chunks = n_folds + 1
    if len(sorted_items) < n_chunks:
        raise ValueError(
            f"Too few items (N={len(sorted_items)}) for {n_folds}-fold walk-forward: "
            f"need at least n_folds+1 = {n_chunks}. Reduce the number of folds "
            f"(e.g. n_folds=1) or provide more data."
        )

    # Partition into n_chunks slices of as-equal-as-possible size.
    chunk_size, remainder = divmod(len(sorted_items), n_chunks)
    chunks: List[list] = []
    start = 0
    for i in range(n_chunks):
        # Distribute the remainder across the first `remainder` chunks.
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(sorted_items[start:end])
        start = end

    folds: List[Tuple[list, list]] = []
    for i in range(n_folds):
        train: list = []
        for chunk in chunks[: i + 1]:
            train.extend(chunk)
        test = chunks[i + 1]
        assert_no_lookahead(train, test, key)
        folds.append((train, test))

    return folds


def rolling_splits(
    items: list,
    key: Callable[[Any], Any],
    train_size: int,
    test_size: int,
    step: int,
) -> List[Tuple[list, list]]:
    """Fixed-window rolling splits.

    Sorts *items* by *key*, then slides a window of *train_size* + *test_size*
    forward by *step* items each fold.  The test window immediately follows the
    train window — never overlaps.

    Guarantees max key(train) <= min key(test) via assert_no_lookahead.
    Returns an empty list (not an error) when the sorted sequence is too short
    to produce even one fold — callers should handle this gracefully.
    """
    if train_size < 1:
        raise ValueError(f"train_size must be >= 1, got {train_size}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    sorted_items = sorted(items, key=key)
    folds: List[Tuple[list, list]] = []

    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > len(sorted_items):
            break
        train = sorted_items[start:train_end]
        test = sorted_items[train_end:test_end]
        assert_no_lookahead(train, test, key)
        folds.append((train, test))
        start += step

    return folds
