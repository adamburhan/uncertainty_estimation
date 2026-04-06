"""Probe cv2.findFundamentalMat on the dumped offending point set.

Loads lkps.npy / rkps.npy from orb_crash_dump/ (produced by
repro_orb_crash.py) and sweeps through input-layout and algorithm
variations to isolate what triggers the matrix.cpp:764 assertion.

Usage:
    python -m scripts.debug.probe_findfundamentalmat
"""

from pathlib import Path

import cv2 as cv
import numpy as np


def try_call(label, fn):
    try:
        fn()
        print(f"  OK   | {label}")
    except cv.error as e:
        msg = str(e).splitlines()[-1] if str(e) else "<empty>"
        print(f"  FAIL | {label} :: {msg}")


def main():
    dump = Path("orb_crash_dump")
    lkps = np.load(dump / "lkps.npy")
    rkps = np.load(dump / "rkps.npy")

    print(f"Loaded lkps {lkps.shape} {lkps.dtype}, rkps {rkps.shape} {rkps.dtype}")
    print(f"OpenCV version: {cv.__version__}\n")

    # Basic stats on the point set
    print("Point set diagnostics:")
    print(f"  N = {len(lkps)}")
    print(f"  lkps unique    : {len(np.unique(lkps, axis=0))}")
    print(f"  rkps unique    : {len(np.unique(rkps, axis=0))}")
    print(f"  pair unique    : {len(np.unique(np.hstack([lkps, rkps]), axis=0))}")
    print(f"  lkps range x   : [{lkps[:,0].min():.2f}, {lkps[:,0].max():.2f}]")
    print(f"  lkps range y   : [{lkps[:,1].min():.2f}, {lkps[:,1].max():.2f}]")
    print(f"  rkps range x   : [{rkps[:,0].min():.2f}, {rkps[:,0].max():.2f}]")
    print(f"  rkps range y   : [{rkps[:,1].min():.2f}, {rkps[:,1].max():.2f}]")
    # Centered rank: degenerate (collinear/single-point) sets have rank < 2
    print(f"  lkps centered rank: {np.linalg.matrix_rank(lkps - lkps.mean(0))}")
    print(f"  rkps centered rank: {np.linalg.matrix_rank(rkps - rkps.mean(0))}")
    print()

    # Input layout variations
    print("Input layout variations (FM_RANSAC, threshold=3.0):")
    for label, lk, rk in [
        ("(N,2) float32",         lkps.astype(np.float32),                rkps.astype(np.float32)),
        ("(N,1,2) float32",       lkps.reshape(-1,1,2).astype(np.float32), rkps.reshape(-1,1,2).astype(np.float32)),
        ("(N,2) float64",         lkps.astype(np.float64),                rkps.astype(np.float64)),
        ("(N,2) f32 ascontig",    np.ascontiguousarray(lkps.astype(np.float32)),
                                   np.ascontiguousarray(rkps.astype(np.float32))),
    ]:
        try_call(label, lambda lk=lk, rk=rk:
                 cv.findFundamentalMat(lk, rk, cv.FM_RANSAC, ransacReprojThreshold=3.0))
    print()

    # Method variations on the canonical (N,2) float32 layout
    print("Method variations on (N,2) float32:")
    lk = lkps.astype(np.float32)
    rk = rkps.astype(np.float32)
    methods = [
        ("FM_RANSAC  thr=3.0",  lambda: cv.findFundamentalMat(lk, rk, cv.FM_RANSAC,  ransacReprojThreshold=3.0)),
        ("FM_RANSAC  thr=1.0",  lambda: cv.findFundamentalMat(lk, rk, cv.FM_RANSAC,  ransacReprojThreshold=1.0)),
        ("FM_RANSAC  thr=0.5",  lambda: cv.findFundamentalMat(lk, rk, cv.FM_RANSAC,  ransacReprojThreshold=0.5)),
        ("FM_LMEDS",            lambda: cv.findFundamentalMat(lk, rk, cv.FM_LMEDS)),
        ("FM_8POINT",           lambda: cv.findFundamentalMat(lk, rk, cv.FM_8POINT)),
        ("USAC_DEFAULT thr=3.0",lambda: cv.findFundamentalMat(lk, rk, cv.USAC_DEFAULT, 3.0, 0.999)),
        ("USAC_MAGSAC  thr=3.0",lambda: cv.findFundamentalMat(lk, rk, cv.USAC_MAGSAC, 3.0, 0.999)),
    ]
    for label, fn in methods:
        try_call(label, fn)
    print()

    # Confidence / maxIters variations (7-arg overload)
    print("FM_RANSAC confidence/maxIters sweep:")
    for conf in (0.99, 0.999, 0.9999):
        for iters in (100, 1000, 5000):
            label = f"conf={conf} iters={iters}"
            try_call(label, lambda c=conf, n=iters:
                     cv.findFundamentalMat(lk, rk, cv.FM_RANSAC, 3.0, c, n))
    print()

    # Subset size sweep — does dropping to smaller / larger N change anything?
    print("Subset size sweep (first-k points, FM_RANSAC thr=3.0):")
    for k in sorted({8, 12, 20, 50, 100, len(lk)}):
        if k > len(lk):
            continue
        try_call(f"N={k}",
                 lambda lk=lk[:k], rk=rk[:k]:
                 cv.findFundamentalMat(lk, rk, cv.FM_RANSAC, ransacReprojThreshold=3.0))
    print()

    # Essential-matrix variants (fixed K for SemiStaticSim: fx=fy=320, cx=cy=320)
    K = np.array([[320.0, 0.0, 320.0],
                  [0.0, 320.0, 320.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    print("findEssentialMat variants (K = SemiStaticSim intrinsics):")
    e_methods = [
        ("RANSAC       thr=3.0", lambda: cv.findEssentialMat(lk, rk, K, method=cv.RANSAC,       prob=0.999, threshold=3.0)),
        ("RANSAC       thr=1.0", lambda: cv.findEssentialMat(lk, rk, K, method=cv.RANSAC,       prob=0.999, threshold=1.0)),
        ("LMEDS",                lambda: cv.findEssentialMat(lk, rk, K, method=cv.LMEDS,        prob=0.999)),
        ("USAC_DEFAULT thr=3.0", lambda: cv.findEssentialMat(lk, rk, K, method=cv.USAC_DEFAULT, prob=0.999, threshold=3.0)),
        ("USAC_MAGSAC  thr=3.0", lambda: cv.findEssentialMat(lk, rk, K, method=cv.USAC_MAGSAC,  prob=0.999, threshold=3.0)),
    ]
    for label, fn in e_methods:
        try_call(label, fn)


if __name__ == "__main__":
    main()
