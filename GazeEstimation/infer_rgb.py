import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from models import get_model
from utils import load_config


def load_ckpt_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # repo에서 쓰는 형태: {"state_dict": ..., ...} 인 경우가 많음
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # 이 repo 모델은 load_pretrained_data가 있을 수 있음(lopo.py가 그렇게 씀)
    if hasattr(model, "load_pretrained_data"):
        model.load_pretrained_data(state)
    else:
        model.load_state_dict(state, strict=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./config/rgb.yml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True, help="path to an RGB image")
    p.add_argument(
        "--bbox",
        default=None,
        help="optional face bbox as x,y,w,h in pixels. example: 100,120,300,300",
    )
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = get_model(config).to(device)
    model.eval()
    load_ckpt_into_model(model, args.checkpoint, device)

    img = Image.open(args.image).convert("RGB")

    # 1) face crop
    if args.bbox is not None:
        x, y, w, h = [int(v) for v in args.bbox.split(",")]
        img = img.crop((x, y, x + w, y + h))
    else:
        # bbox 없으면 그냥 전체 이미지를 넣어버림(정확도 떨어질 수 있음)
        pass

    im_size = int(config["MODEL"]["IM_SIZE"])

    # 2) transforms (대부분 ImageNet 정규화 사용; repo dataset transform 확인해서 맞추면 더 좋음)
    tfm = T.Compose(
        [
            T.Resize((im_size, im_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    x = tfm(img).unsqueeze(0).to(device)  # [1,3,H,W]

    torch.backends.cudnn.benchmark = True  # 448 고정이면 도움 됨

    with torch.inference_mode():
        pred_t = model(x)

    pred = pred_t[0].float().cpu().tolist()
    print("pred:", pred)



if __name__ == "__main__":
    main()
