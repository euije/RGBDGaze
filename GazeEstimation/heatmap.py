import os
import json
import argparse
from collections import defaultdict
import math

import numpy as np
import matplotlib.pyplot as plt


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
    gaze.ndjson 구조:

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

    를 개별 gaze 샘플 리스트로 펼친다.
    """
    out = []
    for rec in records:
        dev = rec.get("device")
        sid = rec.get("session_id")
        base_ts = int(rec.get("ts_ms") or rec.get("server_ts_ms") or 0)

        payload = rec.get("payload") or {}
        samples = payload.get("samples") or payload.get("gazes") or []
        if not isinstance(samples, list):
            continue

        for s in samples:
            if not isinstance(s, dict):
                continue
            g = dict(s)
            g_ts = s.get("ts_ms")
            if g_ts is None:
                g_ts = s.get("t_ms")
            try:
                g_ts = int(g_ts)
            except Exception:
                g_ts = base_ts
            g["ts_ms"] = g_ts
            g["device"] = dev
            g["session_id"] = sid
            out.append(g)

    out.sort(key=lambda g: g["ts_ms"])
    return out


def load_screen_size(session_root, fallback_w=1440, fallback_h=3040):
    """
    첫 번째 screenshot.ndjson 라인에서 screen_w/screen_h 추정.
    없으면 fallback 사용.
    """
    scr_path = os.path.join(session_root, "screenshot.ndjson")
    for rec in iter_ndjson(scr_path):
        meta = rec.get("meta") or {}
        w = meta.get("screen_w") or meta.get("screen_w_px")
        h = meta.get("screen_h") or meta.get("screen_h_px")
        try:
            w = int(w)
            h = int(h)
            return w, h
        except Exception:
            continue
    return fallback_w, fallback_h


def load_windows(session_root):
    """
    window_llm.ndjson에서 각 window의
    (screen_type, t_start_ms, t_end_ms) 리스트 반환.
    """
    path = os.path.join(session_root, "window_llm.ndjson")
    windows = []
    for rec in iter_ndjson(path):
        stype = rec.get("screen_type")
        t0 = rec.get("t_start_ms")
        t1 = rec.get("t_end_ms")
        if stype is None or t0 is None or t1 is None:
            continue
        try:
            t0 = int(t0)
            t1 = int(t1)
        except Exception:
            continue
        windows.append({
            "screen_type": stype,
            "t_start_ms": t0,
            "t_end_ms": t1,
        })
    return windows


def assign_gaze_to_types(gazes, windows):
    """
    각 gaze 샘플(ts_ms, x_px, y_px)을
    해당하는 window의 screen_type별로 모은다.

    반환:
        dict[screen_type] -> list[(x, y)]
    """
    per_type_points = defaultdict(list)

    # 단순 O(N*M) 스캔 (데이터 크기 작은 가정)
    for w in windows:
        stype = w["screen_type"]
        t0 = w["t_start_ms"]
        t1 = w["t_end_ms"]

        for g in gazes:
            ts = g.get("ts_ms")
            if ts is None:
                continue
            if not (t0 <= ts < t1):
                continue

            x = g.get("x_px") or g.get("x")
            y = g.get("y_px") or g.get("y")
            if x is None or y is None:
                continue
            try:
                x = float(x)
                y = float(y)
            except Exception:
                continue

            per_type_points[stype].append((x, y))

    return per_type_points


def make_heatmap(points, screen_w, screen_h, bin_size_px):
    """
    points: list[(x_px, y_px)]
    반환: 2D numpy array (shape: [n_bins_y, n_bins_x])
    """
    n_bins_x = int(math.ceil(screen_w / float(bin_size_px)))
    n_bins_y = int(math.ceil(screen_h / float(bin_size_px)))

    heat = np.zeros((n_bins_y, n_bins_x), dtype=np.int32)

    for x, y in points:
        # 화면 밖 샘플은 버림
        if x < 0 or x >= screen_w or y < 0 or y >= screen_h:
            continue
        bx = int(x // bin_size_px)
        by = int(y // bin_size_px)
        if 0 <= bx < n_bins_x and 0 <= by < n_bins_y:
            heat[by, bx] += 1

    return heat


def save_heatmap(heat, out_path, title=None):
    """
    heat: 2D numpy array
    out_path: PNG 저장 경로
    """
    if heat is None or heat.size == 0:
        print(f"[warn] empty heatmap for {out_path}, skip save")
        return

    plt.figure()
    plt.imshow(heat, origin="upper", aspect="equal")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_log_dir", type=str, default="./logs")
    ap.add_argument("--device", type=str, required=True)
    ap.add_argument("--session_id", type=str, required=True)
    ap.add_argument("--bin_size_px", type=int, default=40)
    args = ap.parse_args()

    session_root = os.path.join(args.base_log_dir, args.device, args.session_id)
    if not os.path.isdir(session_root):
        print(f"[error] session_root not found: {session_root}")
        return

    gaze_path = os.path.join(session_root, "gaze.ndjson")
    window_llm_path = os.path.join(session_root, "window_llm.ndjson")

    if not os.path.exists(gaze_path):
        print(f"[error] gaze.ndjson not found: {gaze_path}")
        return
    if not os.path.exists(window_llm_path):
        print(f"[error] window_llm.ndjson not found: {window_llm_path}")
        return

    # 1) 화면 크기 추정
    screen_w, screen_h = load_screen_size(session_root)
    print(f"[info] screen size ~ {screen_w} x {screen_h} px")

    # 2) gaze / window 로드
    gaze_records = list(iter_ndjson(gaze_path))
    gazes = flatten_gaze_records(gaze_records)
    print(f"[info] total gaze samples: {len(gazes)}")

    windows = load_windows(session_root)
    print(f"[info] total windows in window_llm: {len(windows)}")

    if not windows or not gazes:
        print("[warn] no windows or no gaze samples, nothing to plot.")
        return

    # 3) screen_type별 gaze 모으기
    per_type_points = assign_gaze_to_types(gazes, windows)

    # 4) 출력 디렉토리
    out_dir = os.path.join(session_root, "gaze_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # 5) type별 heatmap 저장
    all_points = []
    for stype, pts in per_type_points.items():
        if not pts:
            continue
        print(f"[info] screen_type={stype} num_points={len(pts)}")
        all_points.extend(pts)

        heat = make_heatmap(pts, screen_w, screen_h, args.bin_size_px)
        out_path = os.path.join(out_dir, f"heatmap_{stype}.png")
        save_heatmap(heat, out_path, title=f"{stype} (bin={args.bin_size_px}px)")

    # 6) 전체 합친 heatmap
    if all_points:
        heat_all = make_heatmap(all_points, screen_w, screen_h, args.bin_size_px)
        out_all = os.path.join(out_dir, f"heatmap_ALL.png")
        save_heatmap(heat_all, out_all, title=f"ALL screen_types (bin={args.bin_size_px}px)")
    else:
        print("[warn] no gaze points assigned to any window; no ALL heatmap.")


if __name__ == "__main__":
    main()
