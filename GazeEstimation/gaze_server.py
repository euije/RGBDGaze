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

from PIL import Image, ImageOps

import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gaze")

from collections import defaultdict

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
        img = ImageOps.mirror(img)
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device, non_blocking=True)

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
            "pairs": [],          # list of (feat[np.float32], tgtx, tgty)
            "std_mean": None,
            "std_std": None,
            "svr_x": None,
            "svr_y": None,
            "trained": False,
            "meta": {},
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
            blob["svr_x"] = None
            blob["svr_y"] = None
            blob["std_mean"] = None
            blob["std_std"] = None
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
        C: float = Form(100.0),
        gamma: float = Form(0.5),   # (linear SVR이면 안씀)
        eps: float = Form(0.05),
    ):
        try:
            with lock:
                blob = _load_device_blob(device)
                pairs = blob.get("pairs", [])
            if len(pairs) < 4:
                return JSONResponse({"error": f"not enough pairs: {len(pairs)}"}, status_code=400)

            F, P, tx, ty, counts = _group_pairs_by_target(pairs, key_round_px=1.0, agg="mean")
            logger.info(f"[calib] raw_pairs={len(pairs)} grouped_points={len(tx)} counts={counts}")

            if len(tx) < 4:
                return JSONResponse({"error": f"not enough grouped points: {len(tx)}"}, status_code=400)

            # ✅ 픽셀 타깃 -> centered (origin=center, +y=up)
            # TODO: 하드코딩 말고 Android에서 screen_w/h를 받아서 저장하는 게 정석
            w = 1080.0
            h = 2640.0
            tx_c = tx - (w * 0.5)
            ty_c = (h * 0.5) - ty  # y 뒤집기

            # feature 표준화
            mean, std = _fit_standardizer(F)
            Z = _zscore(F, mean, std)
            D = Z.shape[1]

            # ✅ 학습도 centered 기준
            svr_x = SVR(kernel="linear", C=float(C), epsilon=float(eps))
            svr_y = SVR(kernel="linear", C=float(C), epsilon=float(eps))
            svr_x.fit(Z, tx_c)
            svr_y.fit(Z, ty_c)

            # ---- train-set sanity (centered 기준 MAE) ----
            px_c = svr_x.predict(Z)
            py_c = svr_y.predict(Z)

            mae_x_c = float(np.mean(np.abs(px_c - tx_c)))
            mae_y_c = float(np.mean(np.abs(py_c - ty_c)))

            logger.info(
                f"[calib] device={device} "
                f"tgt_mean_c=({float(tx_c.mean()):.2f},{float(ty_c.mean()):.2f}) "
                f"pred_mean_c=({float(px_c.mean()):.2f},{float(py_c.mean()):.2f}) "
                f"mae_c=({mae_x_c:.2f},{mae_y_c:.2f}) "
                f"pred_std_c=({float(px_c.std()):.2f},{float(py_c.std()):.2f})"
            )

            # pred 통계 / 상관 (진단용)
            logger.info(
                f"[calib] pred_range_x=({P[:,0].min():.2f},{P[:,0].max():.2f}) "
                f"pred_range_y=({P[:,1].min():.2f},{P[:,1].max():.2f})"
            )

            corr_x = float(np.corrcoef(P[:, 0], tx_c)[0, 1])
            corr_y = float(np.corrcoef(P[:, 1], ty_c)[0, 1])
            logger.info(f"[calib] corr(pred_x,tx_c)={corr_x:.3f} corr(pred_y,ty_c)={corr_y:.3f}")

            # ---- LOOCV (centered 기준) ----
            loo = LeaveOneOut()
            mae_xs, mae_ys = [], []
            for tr, te in loo.split(Z):
                Z_tr, Z_te = Z[tr], Z[te]
                tx_tr, tx_te = tx_c[tr], tx_c[te]
                ty_tr, ty_te = ty_c[tr], ty_c[te]

                m_x = SVR(kernel="linear", C=float(C), epsilon=float(eps))
                m_y = SVR(kernel="linear", C=float(C), epsilon=float(eps))
                m_x.fit(Z_tr, tx_tr)
                m_y.fit(Z_tr, ty_tr)

                px_te = m_x.predict(Z_te)
                py_te = m_y.predict(Z_te)
                mae_xs.append(mean_absolute_error(tx_te, px_te))
                mae_ys.append(mean_absolute_error(ty_te, py_te))

            loocv_x = float(np.mean(mae_xs))
            loocv_y = float(np.mean(mae_ys))
            logger.info(f"[calib] LOOCV_MAE_c=({loocv_x:.1f}px, {loocv_y:.1f}px)")

            # ✅ 모델 저장: centered 좌표계를 쓴다는 메타도 같이 저장
            with lock:
                blob = _load_device_blob(device)
                blob["std_mean"] = mean
                blob["std_std"] = std
                blob["svr_x"] = svr_x
                blob["svr_y"] = svr_y
                blob["trained"] = True
                blob["meta"] = {
                    "trained_ms": time.time() * 1000.0,
                    "n_pairs_raw": len(pairs),
                    "n_points": int(len(tx)),
                    "C": float(C),
                    "eps": float(eps),
                    "feat_dim": int(D),
                    "agg": "mean",
                    "key_round_px": 1.0,
                    "coord": "centered_y_up",
                    "screen_w": w,
                    "screen_h": h,
                    "train_mae_c": [mae_x_c, mae_y_c],
                    "loocv_mae_c": [loocv_x, loocv_y],
                }
                _save_device_blob(device, blob)

            return {"ok": True, "device": device, **blob["meta"]}

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)


    @app.post("/infer")
    async def infer_api(
        device: str = Form(...),
        image: UploadFile = File(...),
        bbox: Optional[str] = Form(default=None),
        return_feat: bool = Form(False),
    ):
        """
        TRACK: (image -> feat) -> (device별 std) -> (device별 SVR) -> screen(x,y)
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

            if not blob.get("trained", False) or blob.get("svr_x", None) is None:
                return JSONResponse({"error": f"device '{device}' not calibrated"}, status_code=400)

            mean = blob["std_mean"]
            std = blob["std_std"]
            Z = _zscore(feat[None, :], mean, std)  # [1,D]

            sx_c = float(blob["svr_x"].predict(Z)[0])
            sy_c = float(blob["svr_y"].predict(Z)[0])

            w = 1080.0
            h = 2640.0
            sx = sx_c + (w * 0.5)
            sy = (h * 0.5) - sy_c

            out = {
                "device": device,
                "screen": [sx, sy],
                "screen_centered": [sx_c, sy_c],
                "pred": pred,               # 디버그용
                "ms": dt_ms,
                "device_exec": devtype,
                "feat_layer": feat_layer,
            }
            if return_feat:
                out["feat"] = feat.tolist()
            return JSONResponse(out)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
