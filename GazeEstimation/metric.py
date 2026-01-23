#!/usr/bin/env python3
"""
metric.py
Compute gaze-centrality metrics by screen_type from GazeAlerter logs.

Assumptions (matching your plotting script):
- base_log_dir/<device>/<session_id> contains:
  - gaze.ndjson
  - window_llm.ndjson
  - screenshot.ndjson (optional; used to infer screen size)
- window_llm.ndjson has fields: screen_type, t_start_ms, t_end_ms
- gaze.ndjson stores gaze samples under payload.samples as {ts_ms, x_px, y_px}

Outputs:
- Prints per-type stats (n, mean/median/std distance-to-center in px; mean normalized)
- Computes Delta metrics:
    Δ_px  = mean(ActiveSearch+Communication) - mean(RecommendedFeed)
    Δ_%   = Δ_px / mean(ActiveSearch+Communication) * 100
- Saves a JSON summary in <session_root>/metrics/metrics_gaze_centrality.json
"""

import os
import json
import argparse
from collections import defaultdict
import math
import statistics

import numpy as np


# --------------------------
# IO helpers
# --------------------------
def iter_ndjson(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def flatten_gaze_records(records):
    """
    Expected gaze.ndjson structure (same as your heatmap script):
    {
      "ts_ms": ...,
      "device": "...",
      "session_id": "...",
      "payload": {
        "samples": [
          {"ts_ms": ..., "x_px": ..., "y_px": ...},
          ...
        ]
      }
    }
    Returns a list of gaze samples sorted by ts_ms:
      [{"ts_ms": int, "x_px": float, "y_px": float}, ...]
    """
    out = []
    for rec in records:
        base_ts = int(rec.get("ts_ms") or rec.get("server_ts_ms") or 0)
        payload = rec.get("payload") or {}
        samples = payload.get("samples") or payload.get("gazes") or []
        if not isinstance(samples, list):
            continue

        for s in samples:
            if not isinstance(s, dict):
                continue
            g_ts = s.get("ts_ms", s.get("t_ms", base_ts))
            try:
                g_ts = int(g_ts)
            except Exception:
                g_ts = base_ts

            x = s.get("x_px", s.get("x"))
            y = s.get("y_px", s.get("y"))
            if x is None or y is None:
                continue
            try:
                x = float(x)
                y = float(y)
            except Exception:
                continue

            out.append({"ts_ms": g_ts, "x_px": x, "y_px": y})

    out.sort(key=lambda g: g["ts_ms"])
    return out


def load_screen_size(session_root, fallback_w=1440, fallback_h=3040):
    """
    Infer screen size from screenshot.ndjson meta if available.

    IMPORTANT:
    - We only trust explicit device-screen fields (screen_w/screen_h or screen_w_px/screen_h_px).
    - We intentionally ignore generic fields like width/height because those often refer to the
      saved screenshot image resolution, which may differ from the physical screen size.

    Pixel 4 default fallback: 1440 x 3040.
    """
    scr_path = os.path.join(session_root, "screenshot.ndjson")
    for rec in iter_ndjson(scr_path):
        meta = rec.get("meta") or {}
        w = meta.get("screen_w") or meta.get("screen_w_px")
        h = meta.get("screen_h") or meta.get("screen_h_px")
        try:
            w = int(w)
            h = int(h)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            continue
    return fallback_w, fallback_h



def load_windows(session_root):
    """
    window_llm.ndjson -> list of windows:
      [{"screen_type": str, "t_start_ms": int, "t_end_ms": int, "confidence": float|None}, ...]
    """
    path = os.path.join(session_root, "window_llm.ndjson")
    windows = []
    for rec in iter_ndjson(path):
        stype = rec.get("screen_type")
        t0 = rec.get("t_start_ms")
        t1 = rec.get("t_end_ms")
        conf = rec.get("confidence")
        if stype is None or t0 is None or t1 is None:
            continue
        try:
            t0 = int(t0)
            t1 = int(t1)
        except Exception:
            continue
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        windows.append({"screen_type": str(stype), "t_start_ms": t0, "t_end_ms": t1, "confidence": conf})

    windows.sort(key=lambda w: w["t_start_ms"])
    return windows


# --------------------------
# Metric computation
# --------------------------
def dist_to_center_px(x, y, cx, cy):
    return math.hypot(x - cx, y - cy)


def compute_metrics(gazes, windows, screen_w, screen_h, conf_min=None, phase_split_sec=None):
    """
    Returns:
      metrics = {
        "meta": {...},
        "by_type": {stype: {...}},
        "delta": {...},
        "by_type_phase": {phase: {stype: {...}}}   (optional)
      }
    """
    cx = screen_w / 2.0
    cy = screen_h / 2.0
    diag_half = math.hypot(cx, cy)

    def init_acc():
        return {"n": 0, "sum": 0.0, "sumsq": 0.0, "vals": []}

    acc_by_type = defaultdict(init_acc)
    acc_all = init_acc()

    # Optional phase split (seconds since session start), using window start time
    use_phase = phase_split_sec is not None
    acc_by_type_phase = {"initial_goal": defaultdict(init_acc), "free_use": defaultdict(init_acc)} if use_phase else None

    # Two-pointer assignment (O(N+M))
    g_idx = 0
    n_g = len(gazes)

    for w in windows:
        if conf_min is not None and w.get("confidence") is not None and w["confidence"] < conf_min:
            continue

        t0 = w["t_start_ms"]
        t1 = w["t_end_ms"]
        stype = w["screen_type"]

        # Determine phase based on window start (relative to session start)
        # We approximate session start as the first window start.
        # This is stable if windows start at 0, 20s, 40s, ...
        # If not, it still defines an "early vs late" split.
        phase = None
        if use_phase:
            session_t0 = windows[0]["t_start_ms"]
            rel_start_sec = (t0 - session_t0) / 1000.0
            phase = "initial_goal" if rel_start_sec < float(phase_split_sec) else "free_use"

        # Advance gaze pointer to first gaze in window
        while g_idx < n_g and gazes[g_idx]["ts_ms"] < t0:
            g_idx += 1

        j = g_idx
        while j < n_g and gazes[j]["ts_ms"] < t1:
            x = gazes[j]["x_px"]
            y = gazes[j]["y_px"]

            # Drop out-of-bounds
            if 0 <= x < screen_w and 0 <= y < screen_h:
                d = dist_to_center_px(x, y, cx, cy)

                a = acc_by_type[stype]
                a["n"] += 1
                a["sum"] += d
                a["sumsq"] += d * d
                a["vals"].append(d)

                acc_all["n"] += 1
                acc_all["sum"] += d
                acc_all["sumsq"] += d * d
                acc_all["vals"].append(d)

                if use_phase:
                    ap = acc_by_type_phase[phase][stype]
                    ap["n"] += 1
                    ap["sum"] += d
                    ap["sumsq"] += d * d
                    ap["vals"].append(d)

            j += 1

        # Keep g_idx where it is (monotonic)
        # (No need to reset; next window has later t0.)
        # Note: windows are non-overlapping, so this is correct.

    def finalize(acc):
        if acc["n"] == 0:
            return None
        mean = acc["sum"] / acc["n"]
        var = (acc["sumsq"] / acc["n"]) - (mean * mean)
        var = max(var, 0.0)
        std = math.sqrt(var)
        med = float(np.median(acc["vals"])) if acc["vals"] else float("nan")
        mean_norm = mean / diag_half if diag_half > 0 else float("nan")
        return {
            "n": int(acc["n"]),
            "mean_px": float(mean),
            "median_px": float(med),
            "std_px": float(std),
            "mean_norm": float(mean_norm),
        }

    by_type = {}
    for stype, a in acc_by_type.items():
        fin = finalize(a)
        if fin is not None:
            by_type[stype] = fin

    all_fin = finalize(acc_all)

    # Δ metrics: compare RECOMMENDED_FEED vs (ACTIVE_SEARCH + COMMUNICATION)
    def weighted_mean(types):
        num = 0.0
        den = 0
        for t in types:
            if t in by_type:
                num += by_type[t]["mean_px"] * by_type[t]["n"]
                den += by_type[t]["n"]
        return (num / den) if den > 0 else None, den

    mean_rcmd, n_rcmd = weighted_mean(["RECOMMENDED_FEED"])
    mean_goal, n_goal = weighted_mean(["ACTIVE_SEARCH", "COMMUNICATION"])

    delta = {}
    if mean_rcmd is not None and mean_goal is not None and n_rcmd > 0 and n_goal > 0:
        delta_px = mean_goal - mean_rcmd  # positive means recommended is more central
        delta_pct = (delta_px / mean_goal) * 100.0 if mean_goal != 0 else float("nan")
        delta = {
            "mean_px_recommended_feed": float(mean_rcmd),
            "mean_px_active_search_plus_communication": float(mean_goal),
            "delta_px": float(delta_px),
            "delta_percent": float(delta_pct),
            "n_recommended_feed": int(n_rcmd),
            "n_active_search_plus_communication": int(n_goal),
        }
    else:
        delta = {
            "warning": "Insufficient samples to compute delta (need both RECOMMENDED_FEED and ACTIVE_SEARCH/COMMUNICATION)."
        }

    out = {
        "meta": {
            "screen_w_px": int(screen_w),
            "screen_h_px": int(screen_h),
            "center_px": [float(cx), float(cy)],
            "half_diagonal_px": float(diag_half),
            "conf_min": conf_min,
            "phase_split_sec": phase_split_sec,
        },
        "all": all_fin,
        "by_type": by_type,
        "delta": delta,
    }

    if use_phase:
        by_phase = {}
        for phase_name, dct in acc_by_type_phase.items():
            by_phase[phase_name] = {}
            for stype, a in dct.items():
                fin = finalize(a)
                if fin is not None:
                    by_phase[phase_name][stype] = fin
        out["by_type_phase"] = by_phase

    return out


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_log_dir", type=str, default="./logs")
    ap.add_argument("--device", type=str, required=True)
    ap.add_argument("--session_id", type=str, required=True)
    ap.add_argument("--screen_w", type=int, default=None,
                    help="Override screen width in pixels (e.g., Pixel 4: 1440).")
    ap.add_argument("--screen_h", type=int, default=None,
                    help="Override screen height in pixels (e.g., Pixel 4: 3040).")
    ap.add_argument("--conf_min", type=float, default=None,
                    help="Optional confidence threshold to filter windows (uses window_llm.ndjson confidence).")
    ap.add_argument("--phase_split_sec", type=float, default=None,
                    help="Optional split point in seconds for initial-goal vs free-use (e.g., 120).")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Optional output path. Default: <session_root>/metrics/metrics_gaze_centrality.json")
    args = ap.parse_args()

    session_root = os.path.join(args.base_log_dir, args.device, args.session_id)
    if not os.path.isdir(session_root):
        raise SystemExit(f"[error] session_root not found: {session_root}")

    gaze_path = os.path.join(session_root, "gaze.ndjson")
    win_path = os.path.join(session_root, "window_llm.ndjson")
    if not os.path.exists(gaze_path):
        raise SystemExit(f"[error] gaze.ndjson not found: {gaze_path}")
    if not os.path.exists(win_path):
        raise SystemExit(f"[error] window_llm.ndjson not found: {win_path}")

    screen_w, screen_h = load_screen_size(session_root)
    if args.screen_w is not None and args.screen_h is not None:
        screen_w, screen_h = int(args.screen_w), int(args.screen_h)
    gaze_records = list(iter_ndjson(gaze_path))
    gazes = flatten_gaze_records(gaze_records)
    windows = load_windows(session_root)

    if not gazes or not windows:
        raise SystemExit("[error] empty gazes or windows; cannot compute metrics.")

    metrics = compute_metrics(
        gazes=gazes,
        windows=windows,
        screen_w=screen_w,
        screen_h=screen_h,
        conf_min=args.conf_min,
        phase_split_sec=args.phase_split_sec,
    )

    # Pretty print summary
    print(f"[info] screen size: {screen_w} x {screen_h} px")
    if metrics.get("all"):
        a = metrics["all"]
        print(f"[ALL] n={a['n']} mean={a['mean_px']:.2f}px median={a['median_px']:.2f}px std={a['std_px']:.2f}px mean_norm={a['mean_norm']:.4f}")

    # Sorted by n desc
    by_type = metrics.get("by_type", {})
    for stype, st in sorted(by_type.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        print(f"[{stype}] n={st['n']} mean={st['mean_px']:.2f}px median={st['median_px']:.2f}px std={st['std_px']:.2f}px mean_norm={st['mean_norm']:.4f}")

    d = metrics.get("delta", {})
    if "delta_px" in d:
        print(f"[DELTA] mean_goal={d['mean_px_active_search_plus_communication']:.2f}px "
              f"mean_recommended={d['mean_px_recommended_feed']:.2f}px "
              f"Δ_px={d['delta_px']:.2f}px Δ_%={d['delta_percent']:.2f}%")
    else:
        print(f"[DELTA] {d.get('warning','(no delta)')}")

    # Save JSON
    out_path = args.out_json
    if out_path is None:
        out_dir = os.path.join(session_root, "metrics")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "metrics_gaze_centrality.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
