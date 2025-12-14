# gaze_server.py
import io
import time
import argparse

import torch
from PIL import Image
import torchvision.transforms as T

from models import get_model
from utils import load_config
from typing import Optional

# ----------------------------
# Core: load once, infer fast
# ----------------------------
def load_ckpt_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if hasattr(model, "load_pretrained_data"):
        model.load_pretrained_data(state)
    else:
        model.load_state_dict(state, strict=False)

def build_infer(config_path, ckpt_path, force_cpu=False):
    config = load_config(config_path)

    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[device]", device, "cuda_available=", torch.cuda.is_available(), "count=", torch.cuda.device_count())

    model = get_model(config).to(device).eval()
    load_ckpt_into_model(model, ckpt_path, device)

    im_size = int(config["MODEL"]["IM_SIZE"])
    normalize = bool(config["DATA"].get("NORMALIZE", True))

    tfm = [
        T.Resize((im_size, im_size)),
        T.ToTensor(),
    ]
    if normalize:
        tfm.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tfm = T.Compose(tfm)

    # GPU fixed-size input => cudnn benchmark helps
    torch.backends.cudnn.benchmark = True

    # Warmup (so first request doesn't stall)
    if device.type == "cuda":
        dummy = torch.zeros(1, 3, im_size, im_size, device=device)
        with torch.inference_mode():
            for _ in range(3):
                _ = model(dummy)
        torch.cuda.synchronize()

    def infer_pil(img: Image.Image):
        t0 = time.time()
        x = tfm(img.convert("RGB")).unsqueeze(0).to(device, non_blocking=True)

        with torch.inference_mode():
            y = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        pred = y[0].float().detach().cpu().tolist()
        dt_ms = (time.time() - t0) * 1000.0
        return pred, dt_ms, device.type

    return infer_pil

# ----------------------------
# HTTP server (FastAPI)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/rgb.yml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    infer = build_infer(args.config, args.checkpoint, force_cpu=args.cpu)

    # Lazy import so we can install deps only when needed
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    # 얼굴 crop을 보내는 게 제일 빠름. bbox는 옵션(서버에서 crop할 수도 있게)
    @app.post("/infer")
    async def infer_api(
        image: UploadFile = File(...),
        bbox: Optional[str] = Form(default=None),
    ):
        try:
            raw = await image.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")

            if bbox:
                x, y, w, h = [int(v) for v in bbox.split(",")]
                img = img.crop((x, y, x + w, y + h))

            pred, dt_ms, dev = infer(img)
            return JSONResponse({"pred": pred, "ms": dt_ms, "device": dev})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
