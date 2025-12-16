# gaze_server.py
import io, time, argparse, re, os, pickle, threading
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

import inspect
from models import get_model
from utils import load_config

# sklearn SVR (서버에서 캘리브/추론)
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import json
import threading
import time

import window_analyze  # 방금 만든 모듈

DEBUG_CROP_DIR = "./gaze_debug_crops"
os.makedirs(DEBUG_CROP_DIR, exist_ok=True)

def _parse_bbox(bbox: str):
    # "x,y,w,h"
    x, y, w, h = [int(v) for v in bbox.split(",")]
    return x, y, w, h

def _clamp_bbox(x: int, y: int, w: int, h: int, W: int, H: int):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def _maybe_save_debug(img_full: Image.Image, img_crop: Image.Image, device: str, tag: str):
    ts = int(time.time() * 1000)
    safe = _safe_device_id(device)
    full_p = os.path.join(DEBUG_CROP_DIR, f"{safe}_{ts}_{tag}_full.jpg")
    crop_p = os.path.join(DEBUG_CROP_DIR, f"{safe}_{ts}_{tag}_crop.jpg")
    try:
        img_full.save(full_p, quality=95)
        img_crop.save(crop_p, quality=95)
        logger.info(f"[debug] saved full={full_p} crop={crop_p}")
    except Exception as e:
        logger.warning(f"[debug] save failed: {e}")

import logging, sys, os
from logging.handlers import RotatingFileHandler

# ===== Logging 설정 =====
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "gaze_server.log")

# ✅ 스크린샷 전용 디렉토리 + ndjson 경로
# 개별 로그 파일 경로
SCREENSHOT_LOG_PATH = os.path.join(LOG_DIR, "screenshot.ndjson")
TOUCH_LOG_PATH      = os.path.join(LOG_DIR, "touch.ndjson")
GAZE_LOG_PATH       = os.path.join(LOG_DIR, "gaze.ndjson")

# 스크린샷 이미지 실제 파일 저장 디렉터리
SCREENSHOT_DIR = os.path.join(LOG_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

logger = logging.getLogger("gaze")
logger.setLevel(logging.INFO)

def _make_json_logger(name: str, path: str) -> logging.Logger:
    """
    각 줄이 순수 JSON string 이 되도록 Formatter를 '%(message)s'로 맞춘 로거.
    """
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = RotatingFileHandler(
        path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=3,
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(message)s"))  # 타임스탬프/레벨 안 붙이고 순수 message만
    fh.setLevel(logging.INFO)
    lg.addHandler(fh)
    lg.propagate = False
    return lg

screenshot_logger = _make_json_logger("gaze.screenshot", SCREENSHOT_LOG_PATH)
touch_logger      = _make_json_logger("gaze.touch", TOUCH_LOG_PATH)
gaze_logger       = _make_json_logger("gaze.gaze", GAZE_LOG_PATH)

logger.info(f"[log] screenshot.ndjson -> {SCREENSHOT_LOG_PATH}")
logger.info(f"[log] touch.ndjson      -> {TOUCH_LOG_PATH}")
logger.info(f"[log] gaze.ndjson       -> {GAZE_LOG_PATH}")
logger.info(f"[log] screenshot images -> {SCREENSHOT_DIR}")

fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

# stdout 로깅 (도커 콘솔용)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
sh.setLevel(logging.INFO)

# 파일 로깅 (txt처럼 쓸 수 있게)
fh = RotatingFileHandler(
    LOG_PATH,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8"
)
fh.setFormatter(fmt)
fh.setLevel(logging.INFO)

# 중복 방지
logger.handlers.clear()
logger.addHandler(sh)
logger.addHandler(fh)
logger.propagate = False

logger.info(f"[log] writing to {LOG_PATH}")

def _current_session_id(device: str) -> Optional[str]:
    with session_lock:
        st = SESSIONS.get(device)
        if st and st.get("active"):
            return st["session_id"]
    return None

def _session_root(device: str, session_id: str) -> str:
    d = os.path.join(LOG_DIR, device, session_id)
    os.makedirs(d, exist_ok=True)
    return d

def _sess_screenshot_ndjson(device: str, session_id: str) -> str:
    root = _session_root(device, session_id)
    return os.path.join(root, "screenshot.ndjson")

def _sess_touch_ndjson(device: str, session_id: str) -> str:
    root = _session_root(device, session_id)
    return os.path.join(root, "touch.ndjson")

def _sess_gaze_ndjson(device: str, session_id: str) -> str:
    root = _session_root(device, session_id)
    return os.path.join(root, "gaze.ndjson")

def _sess_screenshot_img_dir(device: str, session_id: str) -> str:
    root = _session_root(device, session_id)
    d = os.path.join(root, "screenshots")
    os.makedirs(d, exist_ok=True)
    return d


# ---- Session / window LLM 상태 ----
session_lock = threading.Lock()
SESSIONS: Dict[str, Dict[str, Any]] = {}

from collections import defaultdict

def cm_centered_to_px(x_cm: float, y_cm: float,
                      w_px: float, h_px: float,
                      w_cm: float, h_cm: float) -> Tuple[float, float]:
    """
    inputs: centered cm coords (0,0 at screen center), +x right, +y up
    outputs: pixel coords (0..w_px, 0..h_px), origin at top-left
    """
    sx = (x_cm / w_cm) * w_px + (w_px * 0.5)
    sy = (h_px * 0.5) - (y_cm / h_cm) * h_px
    return float(sx), float(sy)


def _group_pairs_by_target(pairs, key_round_px: float = 1.0, agg: str = "mean"):
    """
    pairs: list of (feat[np.float32], pred[np.float32 shape(2,)], target_x, target_y)
    returns:
      F_agg: [K, D]   aggregated feat
      P_agg: [K, 2]   aggregated pred
      tx_agg, ty_agg: [K]
      counts: list[K]
    """
    buckets = defaultdict(list)
    for feat, pred, x, y in pairs:
        kx = round(float(x) / key_round_px) * key_round_px
        ky = round(float(y) / key_round_px) * key_round_px
        buckets[(kx, ky)].append((feat, pred))

    keys = sorted(buckets.keys())
    F_list, P_list, tx_list, ty_list, counts = [], [], [], [], []

    for (kx, ky) in keys:
        feats = np.stack([fp[0] for fp in buckets[(kx, ky)]], axis=0).astype(np.float32)  # [M,D]
        preds = np.stack([fp[1] for fp in buckets[(kx, ky)]], axis=0).astype(np.float32)  # [M,2]

        if agg == "median":
            f_rep = np.median(feats, axis=0)
            p_rep = np.median(preds, axis=0)
        else:
            f_rep = feats.mean(axis=0)
            p_rep = preds.mean(axis=0)

        F_list.append(f_rep.astype(np.float32))
        P_list.append(p_rep.astype(np.float32))
        tx_list.append(float(kx))
        ty_list.append(float(ky))
        counts.append(int(feats.shape[0]))

    F_agg = np.stack(F_list, axis=0).astype(np.float32)   # [K,D]
    P_agg = np.stack(P_list, axis=0).astype(np.float32)   # [K,2]
    tx_agg = np.array(tx_list, dtype=np.float32)
    ty_agg = np.array(ty_list, dtype=np.float32)
    return F_agg, P_agg, tx_agg, ty_agg, counts


# ----------------------------
# Model + feature hook
# ----------------------------
def load_ckpt_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if hasattr(model, "load_pretrained_data"):
        model.load_pretrained_data(state)
    else:
        model.load_state_dict(state, strict=False)

def _find_feat_module(model: nn.Module, feat_layer: Optional[str]) -> Tuple[str, nn.Module]:
    name_to_mod = dict(model.named_modules())
    if feat_layer:
        if feat_layer not in name_to_mod:
            raise ValueError(f"feat_layer='{feat_layer}' not found")
        return feat_layer, name_to_mod[feat_layer]
    # best-effort: name contains fc1
    for k, m in name_to_mod.items():
        if "fc1" in k.lower():
            return k, m
    # fallback: choose linear just before final (out_features==2)
    linears = [(k, m) for k, m in name_to_mod.items() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("No Linear layers found. Please pass --feat_layer explicitly.")
    final_idx = None
    for i, (_, m) in enumerate(linears):
        if getattr(m, "out_features", None) == 2:
            final_idx = i
    if final_idx is not None and final_idx - 1 >= 0:
        return linears[final_idx - 1][0], linears[final_idx - 1][1]
    return linears[-1][0], linears[-1][1]

def build_infer(config_path, ckpt_path, force_cpu=False, feat_layer: Optional[str] = None):
    config = load_config(config_path)
    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[device]", device)

    model = get_model(config).to(device).eval()
    load_ckpt_into_model(model, ckpt_path, device)

    im_size = int(config["MODEL"]["IM_SIZE"])
    normalize = bool(config["DATA"].get("NORMALIZE", True))

    tfm = [T.Resize((im_size, im_size)), T.ToTensor()]
    # 학습과 동일하게: (x - 0.5)/0.5
    tfm.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    tfm = T.Compose(tfm)

    feat_name, feat_mod = _find_feat_module(model, feat_layer)
    print(f"[feature] capturing layer: {feat_name} ({type(feat_mod).__name__})")

    feat_buf: Dict[str, Any] = {"feat": None}

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            feat_buf["feat"] = out.detach()
        else:
            feat_buf["feat"] = None

    feat_mod.register_forward_hook(_hook)

    torch.backends.cudnn.benchmark = True

    if device.type == "cuda":
        dummy = torch.zeros(1, 3, im_size, im_size, device=device)
        with torch.inference_mode():
            for _ in range(3):
                _ = model(dummy)
        torch.cuda.synchronize()

    def infer_pil(img: Image.Image):
        t0 = time.time()

        img = img.convert("RGB")
        img = img.transpose(Image.FLIP_LEFT_RIGHT)  # ✅ 학습과 동일하게 좌우반전

        x = tfm(img).unsqueeze(0).to(device, non_blocking=True)

        with torch.inference_mode():
            y = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        pred = y[0].float().detach().cpu().tolist()

        feat_t = feat_buf["feat"]
        if feat_t is None:
            feat = None
        else:
            f = feat_t[0].float().detach().cpu().flatten()
            feat = f.numpy().astype(np.float32)  # np array

        dt_ms = (time.time() - t0) * 1000.0
        return pred, feat, dt_ms, device.type, feat_name

    # build_infer() 끝부분이나, 모델 로드 직후에
    first_conv = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = (name, m)
            break

    if first_conv:
        n, m = first_conv
        w = m.weight
        logger.info(f"[model] first_conv={n} weight_shape={tuple(w.shape)} in_channels={m.in_channels}")
    
    logger.info(f"[model] class={model.__class__.__name__}")
    logger.info(f"[model] forward_sig={inspect.signature(model.forward)}")

    return infer_pil


# ----------------------------
# Device별 캘리브 저장/모델 저장
# ----------------------------
STORE_DIR = "./gaze_calib_store"
os.makedirs(STORE_DIR, exist_ok=True)

def _safe_device_id(device: str) -> str:
    device = device.strip()
    device = re.sub(r"[^a-zA-Z0-9._-]+", "_", device)
    return device[:128] if device else "unknown_device"

def _model_path(device: str) -> str:
    return os.path.join(STORE_DIR, f"{_safe_device_id(device)}.pkl")

lock = threading.Lock()

def _load_device_blob(device: str) -> Dict[str, Any]:
    path = _model_path(device)
    if not os.path.exists(path):
        return {
            "pairs": [],
            "trained": False,
            "meta": {},
            "ridge_x": None,
            "ridge_y": None,
        }
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_device_blob(device: str, blob: Dict[str, Any]) -> None:
    path = _model_path(device)
    with open(path, "wb") as f:
        pickle.dump(blob, f)

def _fit_standardizer(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # F: [N, D]
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)

def _zscore(f: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((f - mean) / std).astype(np.float32)

## LLM
def _session_worker(device: str):
    logger.info(f"[session] worker started for device={device}")
    while True:
        time.sleep(20.0)  # 20초 주기

        with session_lock:
            state = SESSIONS.get(device)
            if not state or not state.get("active", False):
                logger.info(f"[session] worker exit for device={device}")
                break

            session_id = state["session_id"]
            session_goal = state["session_goal"]
            start_ms = int(state["start_ms"])
            wlen_ms = int(state.get("window_ms", 20_000))
            win_idx = state["next_window_idx"]
            state["next_window_idx"] = win_idx + 1

        t_start_ms = start_ms + win_idx * wlen_ms
        t_end_ms = t_start_ms + wlen_ms

        logger.info(
            f"[session] running window idx={win_idx} "
            f"t=[{t_start_ms},{t_end_ms}) for device={device}"
        )

        try:
            res = window_analyze.analyze_window(
                device=device,
                session_id=session_id,
                session_goal=session_goal,
                window_index=win_idx,
                t_start_ms=t_start_ms,
                t_end_ms=t_end_ms,
                base_log_dir=LOG_DIR,
            )
            if res is None:
                logger.info(
                    f"[session] window idx={win_idx} produced no result (no data); continue"
                )
        except Exception as e:
            logger.error(
                f"[session] window analyze failed for device={device}, idx={win_idx}: {e}"
            )
            continue


# ----------------------------
# FastAPI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/rgb.yml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--feat_layer", default=None)
    args = ap.parse_args()

    infer = build_infer(args.config, args.checkpoint, force_cpu=args.cpu, feat_layer=args.feat_layer)

    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/session/start")
    async def session_start(
        device: str = Form(...),
        session_goal: str = Form(...),
        session_id: Optional[str] = Form(None),
        session_start_ms: Optional[int] = Form(None),
        window_ms: int = Form(20_000),
    ):
        """
        세션 시작:
          - device별 세션 상태 저장
          - 10초마다 window_analyze.analyze_window를 호출하는 워커 스레드 시작
        클라이언트는:
          - session_goal: 유저가 설정한 세션 목표 (예: "DM 3명에게 답장")
          - session_start_ms: 안드로이드 System.currentTimeMillis() 전달 추천
        """
        now_ms = int(time.time() * 1000.0)
        if session_id is None:
            session_id = f"{_safe_device_id(device)}_{now_ms}"

        if session_start_ms is None:
            session_start_ms = now_ms

        with session_lock:
            # 기존 세션 있으면 비활성화
            old = SESSIONS.get(device)
            if old and old.get("active", False):
                old["active"] = False

            state = {
                "active": True,
                "session_id": session_id,
                "session_goal": session_goal,
                "start_ms": int(session_start_ms),
                "window_ms": int(window_ms),
                "next_window_idx": 0,
            }
            SESSIONS[device] = state

            th = threading.Thread(
                target=_session_worker,
                args=(device,),
                daemon=True,
            )
            state["thread"] = th
            th.start()

        logger.info(
            f"[session] start device={device} session_id={session_id} "
            f"goal={session_goal!r} start_ms={session_start_ms} window_ms={window_ms}"
        )

        return {
            "ok": True,
            "device": device,
            "session_id": session_id,
            "session_goal": session_goal,
            "start_ms": session_start_ms,
            "window_ms": window_ms,
        }

    @app.post("/session/stop")
    async def session_stop(
        device: str = Form(...),
    ):
        """
        세션 종료:
          - active 플래그 false로 설정
          - 워커 스레드는 다음 루프에서 종료
        """
        with session_lock:
            state = SESSIONS.get(device)
            if not state:
                return {"ok": False, "error": "no session for device"}

            state["active"] = False
            session_id = state.get("session_id")

        logger.info(f"[session] stop device={device} session_id={session_id}")
        return {"ok": True, "device": device, "session_id": session_id}



    @app.post("/log/screenshot")
    async def log_screenshot(
        device: str = Form(...),
        ts_ms: int = Form(...),
        meta: str = Form(...),
        image: UploadFile = File(...),
    ):
        """
        1) 업로드된 JPEG를 ./logs/<device>/<session>/screenshots/ 아래에 저장
        2) screenshot.ndjson 에 한 줄씩 append
        """
        try:
            # 1) 이미지 저장
            raw = await image.read()
            ts = int(ts_ms)

            session_id = _current_session_id(device)
            if session_id is None:
                # 세션 없는 상태에서 들어온 캡처는 무시하거나, 별도 orphan 폴더에 저장
                logger.warning(f"[log] screenshot with no active session device={device}")
                return {"ok": False, "error": "no_active_session"}

            img_dir = _sess_screenshot_img_dir(device, session_id)
            img_path = os.path.join(img_dir, f"{ts_ms}.jpg")

            # 🔧 여기서 file → image, 그리고 이미 읽어둔 raw 사용
            with open(img_path, "wb") as out:
                out.write(raw)

            # 2) 메타 JSON 파싱
            try:
                meta_obj = json.loads(meta)
            except Exception as e:
                logger.warning(f"[screenshot] meta JSON parse failed: {e} meta={meta!r}")
                meta_obj = {}

            # 3) ndjson 한 줄 append
            rec = {
                "device": device,
                "session_id": session_id,
                "ts_ms": ts,
                "image_path": img_path,
                "meta": meta_obj,
            }

            nd = _sess_screenshot_ndjson(device, session_id)
            with open(nd, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            return {"ok": True, "ts_ms": ts}

        except Exception as e:
            logger.exception(f"[screenshot] failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=400)



    @app.post("/log/touch")
    async def log_touch(
        device: str = Form(...),
        ts_ms: Optional[float] = Form(None),     # 이 호출의 기준 timestamp
        payload: str = Form(...),               # JSON string (events 리스트 등)
    ):
        """
        10초 윈도우 안에서 발생한 터치/스크롤 이벤트들을 한 번에 보내도 되고,
        1초마다 쪼개서 보내도 됨. payload는 JSON string으로 통째로 로그에 저장.
        예: {"events":[{"t_ms":..., "type":"down","x":..,"y":..}, ...]}
        """
        try:
            now_ms = int(time.time() * 1000.0)
            ts = int(ts_ms) if ts_ms is not None else now_ms

            session_id = _current_session_id(device)
            if session_id is None:
                # 세션 없는 상태에서 들어온 캡처는 무시하거나, 별도 orphan 폴더에 저장
                logger.warning(f"[log] touch with no active session device={device}")
                return {"ok": False, "error": "no_active_session"}

            try:
                payload_obj = json.loads(payload)
            except Exception:
                payload_obj = {"raw": payload}

            record = {
                "ts_ms": ts,
                "session_id": session_id,
                "server_ts_ms": now_ms,
                "device": device,
                "payload": payload_obj,
            }

            nd = _sess_touch_ndjson(device, session_id)
            with open(nd, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            return {"ok": True}
        except Exception as e:
            logger.exception("[log_touch] failed")
            return JSONResponse({"error": str(e)}, status_code=400)


    @app.post("/log/gaze")
    async def log_gaze(
        device: str = Form(...),
        ts_ms: Optional[float] = Form(None),
        payload: str = Form(...),
    ):
        """
        Gaze 샘플들을 JSON string으로 받아서 그대로 저장.
        예: {"samples":[{"t_ms":..., "x_px":..., "y_px":...}, ...]}
        """
        try:
            now_ms = int(time.time() * 1000.0)
            ts = int(ts_ms) if ts_ms is not None else now_ms

            session_id = _current_session_id(device)
            if session_id is None:
                # 세션 없는 상태에서 들어온 캡처는 무시하거나, 별도 orphan 폴더에 저장
                logger.warning(f"[log] gaze with no active session device={device}")
                return {"ok": False, "error": "no_active_session"}

            try:
                payload_obj = json.loads(payload)
            except Exception:
                payload_obj = {"raw": payload}

            record = {
                "ts_ms": ts,
                "session_id": session_id,
                "server_ts_ms": now_ms,
                "device": device,
                "payload": payload_obj,
            }

            nd = _sess_gaze_ndjson(device, session_id)
            with open(nd, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            return {"ok": True}
        except Exception as e:
            logger.exception("[log_gaze] failed")
            return JSONResponse({"error": str(e)}, status_code=400)


    @app.get("/calib/status")
    def calib_status(device: str):
        with lock:
            blob = _load_device_blob(device)
        return {
            "device": device,
            "n_pairs": len(blob.get("pairs", [])),
            "trained": bool(blob.get("trained", False)),
            "meta": blob.get("meta", {}),
        }

    @app.post("/calib/reset")
    async def calib_reset(device: str = Form(...)):
        with lock:
            blob = _load_device_blob(device)
            blob["pairs"] = []
            blob["trained"] = False
            blob["ridge_x"] = None
            blob["ridge_y"] = None
            blob["meta"] = {"reset_ms": time.time() * 1000.0}
            _save_device_blob(device, blob)
        return {"ok": True, "device": device}


    @app.post("/calib/add")
    async def calib_add(
        device: str = Form(...),
        target_x: float = Form(...),
        target_y: float = Form(...),
        image: UploadFile = File(...),
        bbox: Optional[str] = Form(default=None),
    ):
        """
        캘리브 중: (image -> feat) 를 뽑고, (feat, target) 를 device별로 저장
        """
        try:
            raw = await image.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if bbox:
                x, y, w, h = [int(v) for v in bbox.split(",")]
                img = img.crop((x, y, x + w, y + h))

            pred, feat, dt_ms, devtype, feat_layer = infer(img)
            if feat is None:
                return JSONResponse({"error": "feature hook returned None"}, status_code=400)

            with lock:
                blob = _load_device_blob(device)
                blob["pairs"].append((feat, np.array(pred, np.float32), float(target_x), float(target_y)))
                blob["meta"] = {"feat_layer": feat_layer, "last_add_ms": time.time() * 1000.0}
                _save_device_blob(device, blob)

            return {
                "ok": True,
                "device": device,
                "n_pairs": len(blob["pairs"]),
                "pred": pred,              # 디버그용 (원하면 제거)
                "ms": dt_ms,
                "device_exec": devtype,
                "feat_dim": int(feat.shape[0]),
            }
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/calib/finish")
    async def calib_finish(
        device: str = Form(...),

        # ✅ 분리 알파
        alpha_x: float = Form(3.0),
        alpha_y: float = Form(3.0),

        # (옵션) 혹시 예전 클라가 alpha 하나만 보내면 둘 다 그 값으로
        alpha: Optional[float] = Form(None),
    ):
        try:
            if alpha is not None:
                alpha_x = float(alpha)
                alpha_y = float(alpha)

            with lock:
                blob = _load_device_blob(device)
                pairs = blob.get("pairs", [])
            if len(pairs) < 4:
                return JSONResponse({"error": f"not enough pairs: {len(pairs)}"}, status_code=400)

            # 13점 대표 feature/pred 만들기 (우리는 pred만 사용)
            F, P, tx, ty, counts = _group_pairs_by_target(pairs, key_round_px=1.0, agg="mean")
            logger.info(f"[calib] raw_pairs={len(pairs)} grouped_points={len(tx)} counts={counts}")

            if len(tx) < 4:
                return JSONResponse({"error": f"not enough grouped points: {len(tx)}"}, status_code=400)

            # screen size (정석은 Android에서 받아오기)
            w = 1440.0
            h = 3040.0

            X = P.astype(np.float32)  # [K,2]
            if not np.all(np.isfinite(X)):
                return JSONResponse({"error": "pred contains NaN/inf"}, status_code=400)

            def eval_coord(coord: str):
                # centered targets
                tx_c = tx - (w * 0.5)
                if coord == "centered_y_down":
                    ty_c = ty - (h * 0.5)        # +y = down
                else:
                    ty_c = (h * 0.5) - ty        # +y = up

                # ✅ 2개의 1D Ridge (alpha 분리)
                rx = Ridge(alpha=float(alpha_x), fit_intercept=True)
                ry = Ridge(alpha=float(alpha_y), fit_intercept=True)
                rx.fit(X, tx_c)
                ry.fit(X, ty_c)

                # train sanity
                px = rx.predict(X).astype(np.float32)
                py = ry.predict(X).astype(np.float32)
                mae_x = float(np.mean(np.abs(px - tx_c)))
                mae_y = float(np.mean(np.abs(py - ty_c)))
                std_x = float(px.std())
                std_y = float(py.std())

                # corr (pred raw vs target centered)
                corr_x = float(np.corrcoef(X[:, 0], tx_c)[0, 1])
                corr_y = float(np.corrcoef(X[:, 1], ty_c)[0, 1])

                # LOOCV
                loo = LeaveOneOut()
                mae_xs, mae_ys = [], []
                for tr, te in loo.split(X):
                    X_tr, X_te = X[tr], X[te]
                    tx_tr, tx_te = tx_c[tr], tx_c[te]
                    ty_tr, ty_te = ty_c[tr], ty_c[te]

                    mx = Ridge(alpha=float(alpha_x), fit_intercept=True)
                    my = Ridge(alpha=float(alpha_y), fit_intercept=True)
                    mx.fit(X_tr, tx_tr)
                    my.fit(X_tr, ty_tr)

                    px_te = float(mx.predict(X_te)[0])
                    py_te = float(my.predict(X_te)[0])
                    mae_xs.append(abs(px_te - float(tx_te[0])))
                    mae_ys.append(abs(py_te - float(ty_te[0])))

                loocv_x = float(np.mean(mae_xs))
                loocv_y = float(np.mean(mae_ys))

                return {
                    "coord": coord,
                    "ridge_x": rx,
                    "ridge_y": ry,
                    "train_mae": (mae_x, mae_y),
                    "pred_std": (std_x, std_y),
                    "loocv_mae": (loocv_x, loocv_y),
                    "corr": (corr_x, corr_y),
                }

            res_up   = eval_coord("centered_y_up")
            res_down = eval_coord("centered_y_down")

            logger.info(f"[calib][y_up]   train_mae_c=({res_up['train_mae'][0]:.1f},{res_up['train_mae'][1]:.1f}) "
                        f"pred_std_c=({res_up['pred_std'][0]:.1f},{res_up['pred_std'][1]:.1f}) "
                        f"LOOCV_MAE_c=({res_up['loocv_mae'][0]:.1f},{res_up['loocv_mae'][1]:.1f}) "
                        f"corr=({res_up['corr'][0]:.3f},{res_up['corr'][1]:.3f})")

            logger.info(f"[calib][y_down] train_mae_c=({res_down['train_mae'][0]:.1f},{res_down['train_mae'][1]:.1f}) "
                        f"pred_std_c=({res_down['pred_std'][0]:.1f},{res_down['pred_std'][1]:.1f}) "
                        f"LOOCV_MAE_c=({res_down['loocv_mae'][0]:.1f},{res_down['loocv_mae'][1]:.1f}) "
                        f"corr=({res_down['corr'][0]:.3f},{res_down['corr'][1]:.3f})")

            # ✅ 선택 기준: (loocv_x + loocv_y) 최소
            score_up   = res_up["loocv_mae"][0] + res_up["loocv_mae"][1]
            score_down = res_down["loocv_mae"][0] + res_down["loocv_mae"][1]
            best = res_up if score_up <= score_down else res_down

            logger.info(f"[calib] choose_coord={best['coord']} alpha_x={alpha_x} alpha_y={alpha_y}")

            with lock:
                blob = _load_device_blob(device)
                blob["ridge_x"] = best["ridge_x"]
                blob["ridge_y"] = best["ridge_y"]
                blob["trained"] = True
                blob["meta"] = {
                    "trained_ms": time.time() * 1000.0,
                    "n_pairs_raw": len(pairs),
                    "n_points": int(len(tx)),
                    "alpha_x": float(alpha_x),
                    "alpha_y": float(alpha_y),
                    "coord": best["coord"],
                    "screen_w": w,
                    "screen_h": h,
                    "train_mae_c": [float(best["train_mae"][0]), float(best["train_mae"][1])],
                    "loocv_mae_c": [float(best["loocv_mae"][0]), float(best["loocv_mae"][1])],
                    "corr_pred_vs_tgt_c": [float(best["corr"][0]), float(best["corr"][1])],
                }
                _save_device_blob(device, blob)

            return {"ok": True, "device": device, **blob["meta"]}

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)


    @app.post("/infer")
    async def infer_api(
        device: str = Form(...),
        image: UploadFile = File(...),
        bbox: Optional[str] = Form(default=None),   # 원본 프레임 기준 bbox: "x,y,w,h"
        debug_crop: bool = Form(False),
        return_feat: bool = Form(False),
    ):
        try:
            # ----- 이미지 로드 + (선택) crop -----
            raw = await image.read()
            img_full = Image.open(io.BytesIO(raw)).convert("RGB")

            img_in = img_full
            used_bbox = None

            if bbox:
                x, y, w, h = _parse_bbox(bbox)
                W, H = img_full.size
                x, y, w, h = _clamp_bbox(x, y, w, h, W, H)
                used_bbox = (x, y, w, h)

                img_crop = img_full.crop((x, y, x + w, y + h))
                img_in = img_crop

                if debug_crop:
                    _maybe_save_debug(img_full, img_crop, device=device, tag="infer")

            # ----- RGBDGaze 모델 추론 (raw pred) -----
            pred, feat, dt_ms, devtype, feat_layer = infer(img_in)

            if pred is None or len(pred) != 2:
                return JSONResponse({"error": f"bad pred: {pred}"}, status_code=400)

            X = np.array(pred, np.float32)[None, :]   # [1,2]

            # ----- 디바이스별 Ridge 로드 -----
            with lock:
                blob = _load_device_blob(device)

            if (not blob.get("trained", False)
                or blob.get("ridge_x", None) is None
                or blob.get("ridge_y", None) is None):
                return JSONResponse(
                    {"error": f"device '{device}' not calibrated"},
                    status_code=400
                )

            rx = blob["ridge_x"]
            ry = blob["ridge_y"]
            meta = blob.get("meta", {})

            w = float(meta.get("screen_w", 1440.0))
            h = float(meta.get("screen_h", 3040.0))
            coord = meta.get("coord", "centered_y_up")

            # ----- Ridge: raw pred → centered px -----
            sx_c = float(rx.predict(X)[0])   # center 기준 x
            sy_c = float(ry.predict(X)[0])   # center 기준 y (sign은 coord에 따라 해석)

            # ----- centered px → screen px -----
            sx = sx_c + (w * 0.5)
            if coord == "centered_y_down":
                # center (0,0) 에서 +y가 아래
                sy = sy_c + (h * 0.5)
            else:
                # center (0,0) 에서 +y가 위
                sy = (h * 0.5) - sy_c

            out = {
                "device": device,
                "screen": [sx, sy],                      # 최종 gaze px
                "screen_centered": [sx_c, sy_c],         # center 기준 px
                "pred": pred,                            # raw 모델 출력 (cm 해석이든 뭐든)
                "ms": dt_ms,
                "device_exec": devtype,
                "feat_layer": feat_layer,
                "coord": coord,

                # 디버그용 메타
                "used_bbox": list(used_bbox) if used_bbox else None,
                "input_size": list(img_in.size),
                "full_size": list(img_full.size),
            }

            if return_feat and feat is not None:
                out["feat"] = feat.tolist()

            return JSONResponse(out)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)


    from collections import Counter

    @app.get("/session/stats")
    def session_stats(
        device: str,
        session_id: Optional[str] = None,
    ):
        """
        현재 세션(또는 지정된 session_id)에 대해
        window_llm.ndjson을 읽어서 screen_type 분포를 계산.
        overlay는 이걸 주기적으로 폴링해서 %를 그려주면 됨.
        """
        # 1) 세션 ID 결정
        with session_lock:
            state = SESSIONS.get(device)

        if session_id is None:
            if not state:
                return {
                    "ok": False,
                    "ready": False,
                    "reason": "no_session_for_device",
                }
            session_id = state.get("session_id")

        if not session_id:
            return {
                "ok": False,
                "ready": False,
                "reason": "no_session_id",
            }

        # 2) window_llm.ndjson 경로
        path = window_analyze._window_llm_ndjson(LOG_DIR, device, session_id)
        if not os.path.exists(path):
            return {
                "ok": True,
                "ready": False,
                "reason": "no_window_llm_file_yet",
                "device": device,
                "session_id": session_id,
            }

        # 3) ndjson 읽어서 통계 계산
        total = 0
        counts = Counter()
        last_idx = -1

        for rec in window_analyze._iter_ndjson(path):
            total += 1
            stype = rec.get("screen_type")
            if stype:
                counts[stype] += 1
            idx = rec.get("window_index")
            if isinstance(idx, int):
                last_idx = max(last_idx, idx)

        if total == 0:
            return {
                "ok": True,
                "ready": False,
                "reason": "no_records_in_window_llm",
                "device": device,
                "session_id": session_id,
            }

        counts_dict = {k: int(v) for k, v in counts.items()}
        shares = {k: (counts_dict[k] / float(total)) for k in counts_dict}

        return {
            "ok": True,
            "ready": True,   # ← 최소 1개 윈도우 결과만 있어도 ready 로 봄
            "device": device,
            "session_id": session_id,
            "total_windows": total,
            "last_window_index": last_idx,
            "counts": counts_dict,
            "shares": shares,
        }




    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
