import os
import io
import json
import time
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFilter
from dotenv import load_dotenv

load_dotenv()  # .env 불러오기

logger = logging.getLogger("gaze.window")

# ===== Gemini 설정 (REST) =====
# 모델 이름은 "gemini-2.5-flash" 같이 짧은 이름만 넣으면,
# 실제 REST 엔드포인트에서는 "models/{name}" 형태로 조합한다.
# GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

#GEMINI_ENDPOINT = (
#    "https://generativelanguage.googleapis.com/v1beta/models/"
#    + GEMINI_MODEL_NAME
#    + ":generateContent"
#)

#if not GEMINI_API_KEY:
#    logger.warning("[window] GEMINI_API_KEY not set – LLM 호출 시 실패할 수 있음")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")  # 또는 gpt-4.1-mini 등

OPENAI_API_URL = os.environ.get(
    "OPENAI_API_URL",
    "https://api.openai.com/v1/chat/completions",
)

SCREEN_TYPE_ENUM = [
    "COMMUNICATION",
    "ACTIVE_SEARCH",
    "RECOMMENDED_FEED",
    "SELF_PRESENTATION",
    "OTHER_UTILITY",
]

ENGAGEMENT_MODE_ENUM = [
    "PASSIVE_CONSUMPTION",
    "ACTIVE_INTERACTION",
]

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_type": {
            "type": "string",
            "enum": SCREEN_TYPE_ENUM,
            "description": "Main screen type for this 10-second window."
        },
        "engagement_mode": {
            "type": "string",
            "enum": ENGAGEMENT_MODE_ENUM,
            "description": "How the user is engaging with the content."
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0 and 1."
        },
        "short_summary": {
            "type": "string",
            "description": "One-sentence summary of what the user is doing."
        },
    },
    "required": ["screen_type", "engagement_mode", "confidence", "short_summary"],
}

# ===== 경로 상수 =====
def _session_root(base_log_dir: str, device: str, session_id: str) -> str:
    """
    base_log_dir/device/session_id 디렉토리 생성 후 경로 리턴
    """
    d = os.path.join(base_log_dir, device, session_id)
    os.makedirs(d, exist_ok=True)
    return d


def _screenshot_ndjson(base_log_dir: str, device: str, session_id: str) -> str:
    root = _session_root(base_log_dir, device, session_id)
    return os.path.join(root, "screenshot.ndjson")


def _touch_ndjson(base_log_dir: str, device: str, session_id: str) -> str:
    root = _session_root(base_log_dir, device, session_id)
    return os.path.join(root, "touch.ndjson")


def _gaze_ndjson(base_log_dir: str, device: str, session_id: str) -> str:
    root = _session_root(base_log_dir, device, session_id)
    return os.path.join(root, "gaze.ndjson")


def _window_img_dir(base_log_dir: str, device: str, session_id: str) -> str:
    root = _session_root(base_log_dir, device, session_id)
    d = os.path.join(root, "window_imgs")
    os.makedirs(d, exist_ok=True)
    return d


def _window_llm_ndjson(base_log_dir: str, device: str, session_id: str) -> str:
    root = _session_root(base_log_dir, device, session_id)
    return os.path.join(root, "window_llm.ndjson")


# ===== 유틸: ndjson 로딩 =====
def _iter_ndjson(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                logger.warning(f"[window] bad ndjson line in {path}: {e}")


def _filter_by_time_and_device(
    recs, device: str, t_start_ms: int, t_end_ms: int, ts_key: str = "ts_ms"
):
    out = []
    for r in recs:
        if r.get("device") != device:
            continue
        ts = r.get(ts_key)
        if ts is None:
            continue
        try:
            ts = int(ts)
        except Exception:
            continue
        if t_start_ms <= ts < t_end_ms:
            out.append(r)
    return out


# ===== 윈도우용 데이터 수집 =====
def collect_window_data(
    device: str,
    session_id: str,
    t_start_ms: int,
    t_end_ms: int,
    base_log_dir: str,
) -> Dict[str, Any]:
    scr_path   = _screenshot_ndjson(base_log_dir, device, session_id)
    touch_path = _touch_ndjson(base_log_dir, device, session_id)
    gaze_path  = _gaze_ndjson(base_log_dir, device, session_id)

    screens_all        = list(_iter_ndjson(scr_path))
    touch_records_all  = list(_iter_ndjson(touch_path))
    gaze_records_all   = list(_iter_ndjson(gaze_path))

    # 스크린샷은 그대로 필터
    screens = _filter_by_time_and_device(screens_all, device, t_start_ms, t_end_ms)

    # touch: record → events flatten 후 필터
    touch_events_all = _flatten_touch_records(touch_records_all)
    touches = _filter_by_time_and_device(touch_events_all, device, t_start_ms, t_end_ms)

    # 🔥 gaze: record → samples flatten + 시간필터까지 한 번에
    gazes = _flatten_gazes(gaze_records_all, device, t_start_ms, t_end_ms)

    screens.sort(key=lambda r: int(r["ts_ms"]))
    touches.sort(key=lambda r: int(r["ts_ms"]))
    gazes.sort(key=lambda r: int(r["ts_ms"]))

    return {
        "screens": screens,
        "touches": touches,
        "gazes": gazes,
    }




# ===== 오버레이: gaze / touch =====
def _compute_gaze_radius_px(meta: Dict[str, Any], img_size: Tuple[int, int]) -> int:
    # 2.19 cm → px, densityDpi 있으면 그걸 사용
    w, h = img_size
    dpi = meta.get("densityDpi") if isinstance(meta, dict) else None
    try:
        if dpi is not None:
            dpi = float(dpi)
    except Exception:
        dpi = None

    if dpi and dpi > 0:
        # px/cm = dpi / 2.54
        r = 2.19 * (dpi / 2.54)
        return max(4, int(r))
    # fallback: 화면 짧은 변의 5%
    return max(4, int(min(w, h) * 0.05))


def draw_gaze_overlay(
    img: Image.Image, x_px: float, y_px: float, radius_px: int
) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r = radius_px
    draw.ellipse(
        (x_px - r, y_px - r, x_px + r, y_px + r),
        fill=(0, 128, 255, 120),  # 파란 반투명
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=r * 0.3))
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def draw_touch_overlay(
    img: Image.Image,
    touches: List[Dict[str, Any]],
    radius_base: int = 16,
    screen_w: Optional[int] = None,
    screen_h: Optional[int] = None,
) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w_img, h_img = img.size
    if screen_w is None:
        screen_w = w_img
    if screen_h is None:
        screen_h = h_img

    for ev in touches:
        x = ev.get("x_px") or ev.get("x") or ev.get("cx")
        y = ev.get("y_px") or ev.get("y") or ev.get("cy")
        if x is None or y is None:
            continue
        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue

        # 화면 → 이미지 좌표 변환
        x_img, y_img = _screen_to_img_xy(x, y, img.size, screen_w, screen_h)

        kind = (ev.get("kind") or ev.get("event") or ev.get("type") or "").upper()
        is_scroll = "SCROLL" in kind or "SC" in kind
        r = radius_base if not is_scroll else int(radius_base * 1.4)
        color = (80, 255, 120, 180)  # 연두색

        draw.ellipse((x_img - r, y_img - r, x_img + r, y_img + r), fill=color)

    return Image.alpha_composite(img.convert("RGBA"), overlay)



def make_grid(imgs: List[Image.Image], cols: int = 2) -> Image.Image:
    if not imgs:
        raise ValueError("no images for grid")
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGBA", (cols * w, rows * h), (0, 0, 0, 255))
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c * w, r * h))
    return grid


# ===== gaze / touch 매칭 =====
def _match_nearest_gaze(
    gazes: List[Dict[str, Any]],
    ts_target: int,
    max_delta_ms: int = 1200,
) -> Optional[Dict[str, Any]]:
    best = None
    best_dt = max_delta_ms + 1
    for g in gazes:
        ts = int(g.get("ts_ms", 0))
        dt = abs(ts - ts_target)
        if dt < best_dt:
            best_dt = dt
            best = g
    return best if best is not None and best_dt <= max_delta_ms else None


# ===== 윈도우 요약 텍스트 =====
def summarize_events(touches: List[Dict[str, Any]]) -> str:
    n_scroll = 0
    n_tap = 0
    for ev in touches:
        kind = (ev.get("kind") or ev.get("event") or ev.get("type") or "").upper()
        if "SCROLL" in kind or "SC" in kind:
            n_scroll += 1
        else:
            n_tap += 1
    parts = []
    parts.append(f"{n_scroll} scroll events")
    parts.append(f"{n_tap} tap/other events")
    return ", ".join(parts)


# ===== LLM 프롬프트 =====
def build_prompt(
    session_goal: str,
    window_index: int,
    t_start_sec: float,
    t_end_sec: float,
    events_summary: str,
) -> str:
    return f"""
You are analyzing a 20-second window of a smartphone Instagram session.
You receive:
- A grid of up to 5 screenshots from this 20-second window, in chronological order
  (top-left = earliest, bottom-right = latest).
- On each screenshot, a blue translucent circle marks where the user was most likely looking
  (gaze region, roughly one standard deviation of error).
- Green translucent spots mark places where the user touched or scrolled during this window.

The goal is to infer what kind of activity the user is doing in this window.

SESSION GOAL:
- The user stated their goal for this session as: "{session_goal}"

WINDOW CONTEXT:
- Window index: {window_index} (0-based)
- Window time range: {t_start_sec:.1f}–{t_end_sec:.1f} seconds since session start.
- Summary of touch/scroll events: {events_summary}

INTERPRETATION RULES (adapted from Guo et al.):
1. Focus on the main post or content the user is most likely looking at.
   This is usually the post in the center of the screen that is fully visible.
   Ignore partially visible posts at the top or bottom.
2. Do NOT treat generic UI elements such as like, comment, share, or hashtags
   as strong evidence of recommendation-based content by themselves.
3. If the user is viewing a comment section and the comments occupy most of the screen,
   treat it as a communication-oriented view. If comments are minor, focus on the main content.
4. Use the gaze region (blue) to decide which part of the screen the user is actually paying attention to.
5. Use green touch spots to infer what the user interacted with: scrolling the main feed,
   tapping on a profile, opening comments, reading or writing a message, etc.
6. IMPORTANT: If a screenshot appears mostly black or blank (for example, because Instagram
   blocks screen capture for privacy on chat/DM screens), interpret this as the user viewing
   a private communication screen (DM or chat). In such cases, strongly prefer the
   screen_type "COMMUNICATION", unless the context clearly suggests otherwise.

LABELING TASK:

1. Decide the main screen type in this 20-second window.
   Choose exactly one of:
   - COMMUNICATION         (DMs, chat, replying to stories, writing comments, reading threads)
   - ACTIVE_SEARCH        (using search bar, searching for a specific account or content)
   - RECOMMENDED_FEED     (home feed, Reels, Explore/recommended content)
   - SELF_PRESENTATION    (editing profile, composing posts or stories, uploading media)
   - OTHER_UTILITY        (settings, profile overview, notifications, other utility screens)

2. Decide the engagement mode:
   - PASSIVE_CONSUMPTION  (mostly scrolling or watching, no substantial writing or direct interaction)
   - ACTIVE_INTERACTION   (writing, replying, sending, posting, or other deliberate actions)

3. Estimate a confidence score between 0 and 1.

OUTPUT FORMAT (JSON only):

Return a single JSON object with the following fields:

{{
  "screen_type": "COMMUNICATION | ACTIVE_SEARCH | RECOMMENDED_FEED | SELF_PRESENTATION | OTHER_UTILITY",
  "engagement_mode": "PASSIVE_CONSUMPTION | ACTIVE_INTERACTION",
  "confidence": float between 0 and 1,
  "short_summary": "one-sentence natural language summary of what the user is doing in this window"
}}

Do NOT include any natural language outside of this JSON.
""".strip()

# 디폴트 화면 해상도 (Pixel 4 기준) – 필요하면 .env로 오버라이드 가능
SCREEN_W_DEFAULT = int(os.environ.get("SCREEN_W_PX", "1440"))
SCREEN_H_DEFAULT = int(os.environ.get("SCREEN_H_PX", "3040"))


def _screen_to_img_xy(
    x_screen: float,
    y_screen: float,
    img_size: Tuple[int, int],
    screen_w: int,
    screen_h: int,
) -> Tuple[float, float]:
    """
    화면 좌표(px, 0..screen_w, 0..screen_h)를
    실제 스크린샷 이미지 좌표로 스케일링.
    """
    w_img, h_img = img_size
    sx = w_img / float(screen_w)
    sy = h_img / float(screen_h)
    return x_screen * sx, y_screen * sy

# ===== 내부: 이미지 → base64 =====
def _image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _image_to_data_url_png(img: Image.Image) -> str:
    """
    PIL.Image → PNG base64 → data URL
    """
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def call_openai_with_grid(grid_img: Image.Image, prompt: str) -> dict:
    """
    grid 이미지 + prompt를 OpenAI vision 모델에 던지고
    JSON object 하나를 받는다.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    # 1) 이미지 → base64 data URL
    img_data_url = _image_to_data_url_png(grid_img)

    # 2) Chat Completions payload 구성
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    # 시스템 프롬프트: 반드시 JSON만 내뱉게 다짐시킴
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert annotator of Instagram usage windows. "
            "You MUST respond with a single JSON object only, no extra text."
        ),
    }

    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": img_data_url,
                },
            },
        ],
    }

    payload = {
        "model": OPENAI_MODEL_NAME,
        # JSON mode: 항상 파싱 가능한 JSON만 생성하도록 강제
        "response_format": {"type": "json_object"},
        "messages": [system_msg, user_msg],
        "temperature": 0.2,
        "max_tokens": 256,
    }

    resp = requests.post(
        OPENAI_API_URL,
        headers=headers,
        json=payload,
        timeout=20,  # 초과하면 그냥 실패 처리
    )

    if resp.status_code != 200:
        logger.error(
            f"[window] OpenAI HTTP {resp.status_code} body={resp.text[:500]}"
        )
        raise RuntimeError(f"OpenAI HTTP {resp.status_code}")

    data = resp.json()
    logger.info(
        f"[window] OpenAI usage={data.get('usage')} "
        f"finish_reason={data.get('choices',[{}])[0].get('finish_reason')}"
    )

    # Chat Completions 응답 구조에서 텍스트 추출
    choice0 = data["choices"][0]["message"]["content"]

    # json_object 모드에서는 보통 content가 문자열 하나
    if isinstance(choice0, str):
        raw = choice0
    else:
        # 혹시 배열 구조로 오면 text 파트만 이어붙이기 방어 코드
        parts = []
        for part in choice0:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        raw = "".join(parts)

    raw = raw.strip()
    # 혹시 ```json ... ``` 감싸져 있으면 제거
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        logger.warning(
            f"[window] OpenAI JSON parse failed, returning raw text. raw_snippet={raw[:200]!r}"
        )
        return {
            "raw_text": raw,
            "parse_error": True,
        }


# def call_gemini_with_grid(grid_img: Image.Image, prompt: str) -> Dict[str, Any]:
#     """
#     REST API + Structured Output(JSON 모드) 사용.
#     - grid_img: gaze/touch 오버레이까지 포함된 하나의 그리드 이미지
#     - prompt: text prompt
#     반환값: schema에 맞는 dict (실패 시 raw/에러 정보 포함 dict)
#     """

#     if not GEMINI_API_KEY:
#         raise RuntimeError("GEMINI_API_KEY not set")

#     # 1) 이미지 → base64
#     img_b64 = _image_to_base64_png(grid_img)

#     # 2) REST 엔드포인트
#     url = GEMINI_ENDPOINT

#     headers = {
#         "Content-Type": "application/json",
#         "x-goog-api-key": GEMINI_API_KEY,
#     }

#     body = {
#         "contents": [
#             {
#                 "parts": [
#                     {
#                         # Vision input (grid 이미지)
#                         "inline_data": {
#                             "mime_type": "image/png",
#                             "data": img_b64,
#                         }
#                     },
#                     {
#                         # Text prompt
#                         "text": prompt,
#                     },
#                 ]
#             }
#         ],
#         "generationConfig": {
#             # ✅ JSON 모드 + structured output
#             "responseMimeType": "application/json",
#             "responseJsonSchema": RESPONSE_JSON_SCHEMA,
#             # 필요하면 온도/토큰 제한도 여기서 설정 가능
#             "temperature": 0.2,
#             "maxOutputTokens": 256,
#         },
#     }

#     t0 = time.time()
#     resp = requests.post(url, headers=headers, data=json.dumps(body))
#     dt = time.time() - t0

#     # ── raw response 로그 (네가 요청한 부분) ─────────────
#     raw_text = resp.text
#     logger.info(
#         "[window] Gemini HTTP %s in %.2fs raw_response_snippet=%s",
#         resp.status_code,
#         dt,
#         raw_text[:400].replace("\n", "\\n"),
#     )

#     if not resp.ok:
#         # HTTP 레벨 에러
#         try:
#             err_json = resp.json()
#         except Exception:
#             err_json = None
#         logger.error("[window] Gemini HTTP error %s body=%s", resp.status_code, raw_text)
#         return {
#             "error": "http_error",
#             "status_code": resp.status_code,
#             "body": raw_text,
#             "json": err_json,
#         }

#     # JSON 파싱
#     try:
#         data = resp.json()
#     except Exception as e:
#         logger.warning("[window] Gemini resp.json() failed: %s", e)
#         return {
#             "error": "json_parse_failed",
#             "raw_text": raw_text,
#         }

#     # JSON 모드일 때는 docs 상 예시처럼 곧바로 schema에 맞는 object가 온다고 되어 있음. 
#     # 혹시 혹시나 candidates 래퍼가 섞여 나오면 방어적으로 처리.
#     if isinstance(data, dict) and "candidates" in data:
#         # 구형 포맷: candidates[0].content.parts[0].text 안에 JSON 문자열
#         try:
#             cand = data["candidates"][0]
#             part = cand["content"]["parts"][0]
#             txt = part.get("text") or ""
#             parsed = json.loads(txt)
#             return parsed
#         except Exception as e:
#             logger.error("[window] structured candidates parse failed: %s", e)
#             return {
#                 "error": "candidates_parse_failed",
#                 "raw": data,
#             }

#     # 이상 없으면 그대로 반환 (schema와 맞다고 가정)
#     return data


# ===== 메인 API: 한 윈도우 분석 =====
def analyze_window(
    *,
    device: str,
    session_id: str,
    session_goal: str,
    window_index: int,
    t_start_ms: int,
    t_end_ms: int,
    base_log_dir: str,
) -> Optional[Dict[str, Any]]:
    """
    한 윈도우(10초)에 대해:
      1) screenshot/touch/gaze 수집
      2) gaze/터치 오버레이 + 4장 그리드 이미지 생성
      3) Gemini 호출
      4) 결과를 window_llm ndjson에 append
    """
    logger.info(
        f"[window] analyze_window device={device} session={session_id} "
        f"idx={window_index} t=[{t_start_ms},{t_end_ms})"
    )

    data = collect_window_data(device, session_id, t_start_ms, t_end_ms, base_log_dir)
    screens = data["screens"]
    touches = data["touches"]
    gazes = data["gazes"]

    if not screens:
        logger.info(f"[window] no screenshots in window idx={window_index}, skip")
        return None

    # 1,4,7,10초 근처 스크린샷 선택
    offsets = [1000, 5000, 10000, 15000, 19000]
    chosen = []
    for off in offsets:
        target_ts = t_start_ms + off
        best = None
        best_dt = 10_000_000
        for s in screens:
            ts = int(s["ts_ms"])
            dt = abs(ts - target_ts)
            if dt < best_dt:
                best_dt = dt
                best = s
        if best is not None and best not in chosen:
            chosen.append(best)

    if not chosen:
        logger.info(f"[window] no suitable screenshots near offsets, skip")
        return None

    imgs_for_grid = []
    touch_summary = summarize_events(touches)

    img_dir = _window_img_dir(base_log_dir, device, session_id)

    for idx, scr in enumerate(chosen):
        img_path = scr.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGBA")
        meta = scr.get("meta") or {}

        screen_w = int(meta.get("screen_w", meta.get("screen_w_px", SCREEN_W_DEFAULT)))
        screen_h = int(meta.get("screen_h", meta.get("screen_h_px", SCREEN_H_DEFAULT)))

        # gaze 매칭
        ts_scr = int(scr["ts_ms"])
        gz = _match_nearest_gaze(gazes, ts_scr)
        if gz is not None:
            gx = gz.get("x_px") or gz.get("x")
            gy = gz.get("y_px") or gz.get("y")
            if gx is not None and gy is not None:
                try:
                    gx = float(gx)
                    gy = float(gy)

                    # 화면 좌표 → 이미지 좌표
                    gx_img, gy_img = _screen_to_img_xy(gx, gy, img.size, screen_w, screen_h)

                    # radius는 원래 _compute_gaze_radius_px가 이미지 크기 기준으로 계산하니까 그대로 사용
                    r_px = _compute_gaze_radius_px(meta, img.size)
                    img = draw_gaze_overlay(img, gx_img, gy_img, r_px)
                except Exception as e:
                    logger.warning(f"[window] gaze overlay failed: {e}")


        # touch 오버레이: 윈도우 전체 touch 다 그려버림 (단순화)
        img = draw_touch_overlay(
            img,
            touches,
            radius_base=16,
            screen_w=screen_w,
            screen_h=screen_h,
        )

        # 개별 디버그 이미지 저장
        out_single = os.path.join(
            img_dir,
            f"win{window_index:03d}_shot{idx}_ts{ts_scr}.png",
        )
        try:
            img.save(out_single)
        except Exception as e:
            logger.warning(f"[window] save single image failed: {e}")

        imgs_for_grid.append(img)

    if not imgs_for_grid:
        logger.info(f"[window] no valid images after overlay, skip")
        return None

    grid = make_grid(imgs_for_grid, cols=2)

    # grid 저장 (디버그 + 나중 분석용)
    grid_path = os.path.join(
        img_dir,
        f"win{window_index:03d}_grid.png",
    )
    try:
        grid.save(grid_path)
    except Exception as e:
        logger.warning(f"[window] save grid failed: {e}")

    # LLM 호출
    t0 = time.time()
    prompt = build_prompt(
        session_goal=session_goal,
        window_index=window_index,
        t_start_sec=(t_start_ms / 1000.0),
        t_end_sec=(t_end_ms / 1000.0),
        events_summary=touch_summary,
    )

    try:
        # llm_out = call_gemini_with_grid(grid, prompt)
        llm_out = call_openai_with_grid(grid, prompt)
    except Exception as e:
        logger.error(f"[window] LLM call failed: {e}")
        return None

    dt = time.time() - t0
    logger.info(
        f"[window] LLM ok idx={window_index} screen_type={llm_out.get('screen_type')} "
        f"mode={llm_out.get('engagement_mode')} dt={dt:.2f}s"
    )

    rec = {
        "device": device,
        "session_id": session_id,
        "window_index": window_index,
        "t_start_ms": t_start_ms,
        "t_end_ms": t_end_ms,
        "grid_path": grid_path,
        "session_goal": session_goal,
        # LLM output 그대로 삽입 (is_deviation / deviation_reason 없음!!)
        "screen_type": llm_out.get("screen_type"),
        "engagement_mode": llm_out.get("engagement_mode"),
        "confidence": llm_out.get("confidence"),
        "short_summary": llm_out.get("short_summary"),
    }

    out_nd = _window_llm_ndjson(base_log_dir, device, session_id)
    try:
        with open(out_nd, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[window] write window_llm ndjson failed: {e}")

    return rec


def _flatten_touch_records(records):
    """
    /log/touch 에서 쌓인 레코드들(records)을
    개별 터치 이벤트 리스트로 풀어준다.

    입력 예:
      {
        "ts_ms": ...,
        "device": "Google_Pixel_4",
        "session_id": "...",
        "payload": {
          "events": [
            {"t_ms":..., "type":"scroll", "x":..., "y":...},
            {"t_ms":..., "type":"focus",  "x":..., "y":...},
          ]
        }
      }

    출력 예(리스트):
      [
        {"device":"Google_Pixel_4","session_id":"...","ts_ms":..., "type":"scroll", ...},
        {"device":"Google_Pixel_4","session_id":"...","ts_ms":..., "type":"focus",  ...},
      ]
    """
    out = []
    for r in records:
        dev = r.get("device")
        sid = r.get("session_id")
        base_ts = int(r.get("ts_ms", 0))

        payload = r.get("payload") or {}
        events = payload.get("events") or []
        if not isinstance(events, list):
            continue

        for ev in events:
            if not isinstance(ev, dict):
                continue
            e = dict(ev)  # shallow copy

            # 상위 메타를 이벤트에도 붙여주기
            e["device"] = dev
            e["session_id"] = sid

            # 이벤트 개별 timestamp (없으면 record ts_ms 사용)
            t_ev = ev.get("t_ms")
            try:
                t_ev = int(t_ev)
            except Exception:
                t_ev = base_ts
            e["ts_ms"] = t_ev

            out.append(e)

    return out



def _flatten_gazes(
    gaze_recs: List[Dict[str, Any]],
    device: str,
    t_start_ms: int,
    t_end_ms: int,
) -> List[Dict[str, Any]]:
    """
    /log/gaze ndjson 구조:

    {
      "ts_ms": ... (POST 기준),
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
    out: List[Dict[str, Any]] = []

    for rec in gaze_recs:
        dev = rec.get("device", device)
        base_ts = int(rec.get("ts_ms") or rec.get("server_ts_ms") or 0)
        payload = rec.get("payload") or {}
        samples = payload.get("samples") or payload.get("gazes") or []

        if not isinstance(samples, list):
            continue

        for s in samples:
            if not isinstance(s, dict):
                continue

            g_ts = s.get("ts_ms")
            if g_ts is None:
                g_ts = s.get("t_ms")
            try:
                g_ts = int(g_ts)
            except Exception:
                g_ts = base_ts

            if not (t_start_ms <= g_ts < t_end_ms):
                continue

            g_flat = dict(s)
            g_flat["ts_ms"] = g_ts
            g_flat["device"] = dev
            out.append(g_flat)

    out.sort(key=lambda g: g["ts_ms"])
    return out
