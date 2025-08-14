# server.py — original pipeline + (1) robust red/yellow, (2) less-crowded labels
import os
from io import BytesIO

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import uvicorn

# --- Render/containers friendliness ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
try:
    cv2.setNumThreads(1)  # type: ignore[attr-defined]
except Exception:
    pass

PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI()

# ---- health/info routes for Render ----
@app.get("/")
def root():
    return {"ok": True, "service": "cuezen-ball-detect", "version": "1.0.0"}

@app.get("/healthz")
def health():
    return {"status": "healthy"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers (ORIGINAL) ----------
def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def dedupe(circles, thr=0.6):
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for x, y, r, lbl in circles:
        if not any(np.hypot(x - kx, y - ky) < thr * min(r, kr) for kx, ky, kr, _ in kept):
            kept.append((int(x), int(y), int(r), lbl))
    return kept

def felt_mask_hsv(hsv):
    h, w = hsv.shape[:2]
    cx1, cy1, cx2, cy2 = int(w * .4), int(h * .4), int(w * .6), int(h * .6)
    center = hsv[cy1:cy2, cx1:cx2]
    Hm, Sm, Vm = np.median(center[..., 0]), np.median(center[..., 1]), np.median(center[..., 2])
    dyn = cv2.inRange(
        hsv,
        np.array([max(0, Hm - 18), max(30, Sm * 0.5), max(30, Vm * 0.5)], np.uint8),
        np.array([min(179, Hm + 18), 255, 255], np.uint8),
    )
    green = cv2.inRange(hsv, np.array([35, 60, 40]), np.array([85, 255, 255]))
    blue  = cv2.inRange(hsv, np.array([85, 40, 40]), np.array([130, 255, 255]))
    mask = cv2.bitwise_or(dyn, cv2.bitwise_or(green, blue))
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)), 1)
    return out

def estimate_radius(mask, img_w):
    if mask is not None and mask.any():
        xs = np.where(mask > 0)[1]
        if xs.size:
            width = xs.max() - xs.min()
            return max(6, int(width / 39))
    return max(6, int(img_w / 45))

def run_hough(gray, mask, r_est, param2, blur_ksize=9, dp=1.2):
    g = gray.copy()
    if mask is not None and mask.any():
        g[mask == 0] = 0
    g = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 2)
    minR, maxR = max(6, int(r_est * .8)), int(r_est * 1.45)
    cir = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=dp, minDist=int(r_est * 1.1),
        param1=110, param2=param2, minRadius=minR, maxRadius=maxR
    )
    out = []
    if cir is not None:
        for x, y, r in np.round(cir[0]).astype("int"):
            out.append((int(x), int(y), int(r), None))
    return out

COLOR_BINS = [
    ("red",     [(0, 10), (170, 179)], 90,  80),
    ("orange",  [(11, 25)],            90,  90),
    ("yellow",  [(26, 40)],            90, 110),
    ("green",   [(41, 85)],            60,  70),
    ("blue",    [(86, 130)],           60,  70),
    ("purple",  [(131, 160)],          60,  70),
]

def _circles_from_binary(bin_mask, label, r_est):
    out = []
    m = cv2.medianBlur(bin_mask, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rmin, rmax = max(6, int(r_est * .6)), int(r_est * 1.6)
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        if r < rmin or r > rmax:
            continue
        area = cv2.contourArea(c)
        circ = np.pi * r * r
        if circ <= 0 or area / circ < 0.45:
            continue
        out.append((x, y, r, label))
    return out

def color_mask_passes(hsv, base_mask, r_est):
    cand = []
    # cue (very white) — proposed but later filtered when we keep only red/yellow
    cue = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 55, 255]))
    if base_mask is not None:
        cue = cv2.bitwise_and(cue, base_mask)
    cand += _circles_from_binary(cue, "cue", r_est)

    # black — proposed but can be filtered later
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 55]))
    if base_mask is not None:
        black = cv2.bitwise_and(black, base_mask)
    cand += _circles_from_binary(black, "black", r_est)

    # chromatic bins
    for name, ranges, smin, vmin in COLOR_BINS:
        cmask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            cmask |= cv2.inRange(hsv, np.array([lo, smin, vmin]), np.array([hi, 255, 255]))
        if base_mask is not None:
            cmask = cv2.bitwise_and(cmask, base_mask)
        cand += _circles_from_binary(cmask, name, r_est)
    return cand

# ---------- NEW: robust color classification + less-crowded filter ----------
def classify_color_hsv(hsv_full, x, y, r):
    """
    Classify circle by dominant hue around its mid-annulus (avoid highlights).
    Returns: 'red','yellow','green','blue','purple','black','cue','unknown'
    (We purposely don't return 'orange' for English 8-ball.)
    """
    h, w = hsv_full.shape[:2]
    mask_outer = np.zeros((h, w), np.uint8)
    mask_inner = np.zeros((h, w), np.uint8)
    cv2.circle(mask_outer, (x, y), max(2, int(r * 0.85)), 255, -1)
    cv2.circle(mask_inner, (x, y), max(1, int(r * 0.35)), 255, -1)
    ring = cv2.subtract(mask_outer, mask_inner)

    H, S, V = cv2.split(hsv_full)

    # quick white/black checks
    white_mask = cv2.inRange(hsv_full, np.array([0, 0, 205]), np.array([179, 70, 255]))
    black_mask = cv2.inRange(hsv_full, np.array([0, 0, 0]),  np.array([179, 255, 55]))
    denom = max(1, cv2.countNonZero(ring))
    if cv2.countNonZero(cv2.bitwise_and(white_mask, ring)) / denom > 0.65:
        return "cue"
    if cv2.countNonZero(cv2.bitwise_and(black_mask, ring)) / denom > 0.50:
        return "black"

    # use only reasonably saturated+bright pixels
    sat = cv2.bitwise_and(S, ring)
    val = cv2.bitwise_and(V, ring)
    valid_mask = (sat > 55) & (val > 75)   # relaxed to catch warm yellows
    valid_h = H[valid_mask]
    if valid_h.size == 0:
        return "unknown"

    mean_h = int(np.median(valid_h))  # robust central hue

    # map to bins (OpenCV hue 0..179)
    if (0 <= mean_h <= 10) or (170 <= mean_h <= 179):
        return "red"
    # WIDEN YELLOW to include warm/orange-ish yellows
    if 18 <= mean_h <= 50:
        return "yellow"
    if 51 <= mean_h <= 85:
        return "green"
    if 86 <= mean_h <= 130:
        return "blue"
    if 131 <= mean_h <= 160:
        return "purple"
    # Anything near "orange" we also treat as yellow for English 8-ball
    return "yellow"

def less_crowded(dets, min_dist_factor=1.25):
    """Second-stage spacing filter to reduce label crowding (keeps larger radius first)."""
    out = []
    for c in sorted(dets, key=lambda d: d["r"], reverse=True):
        too_close = any(
            ((c["x"] - o["x"])**2 + (c["y"] - o["y"])**2) ** 0.5
            < min_dist_factor * min(c["r"], o["r"])
            for o in out
        )
        if not too_close:
            out.append(c)
    return out

# ---------- endpoint (ORIGINAL pipeline + small post-filter) ----------
@app.post("/detect")
async def detect_balls(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert("RGB")
    bgr_full = pil_to_cv2(image_pil)
    H0, W0 = bgr_full.shape[:2]

    MAX_DIM = 1100
    scale = 1.0
    if max(H0, W0) > MAX_DIM:
        scale = MAX_DIM / float(max(H0, W0))
        bgr = cv2.resize(bgr_full, (int(W0 * scale), int(H0 * scale)))
    else:
        bgr = bgr_full.copy()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_felt = felt_mask_hsv(hsv)
    if (mask_felt > 0).sum() < 0.1 * mask_felt.size:
        mask_felt = None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    r_est = estimate_radius(mask_felt, bgr.shape[1])

    proposals = []
    for p2, blur, dp, use_mask in [
        (30, 9, 1.2, True),
        (28, 9, 1.2, True),
        (26, 7, 1.2, True),
        (24, 9, 1.1, False),
    ]:
        proposals += run_hough(gray, mask_felt if use_mask else None, r_est, p2, blur, dp)

    proposals += color_mask_passes(hsv, mask_felt, r_est)
    merged = dedupe(proposals, thr=0.6)

    # Color classification on ORIGINAL scale, keep only red/yellow for this view,
    # then apply less-crowded filter for cleaner labels.
    hsv_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)
    prelim = []
    for x, y, r, _lbl in merged:
        if scale != 1.0:
            x, y, r = int(x / scale), int(y / scale), int(r / scale)

        # ignore circles touching edges (rails/pockets)
        if x < r or y < r or x > (W0 - r) or y > (H0 - r):
            continue

        col = classify_color_hsv(hsv_full, x, y, r)
        # English 8-ball: treat orange-like hues as yellow
        if col == "orange":
            col = "yellow"

        if col in ("red", "yellow"):
            prelim.append({"x": x, "y": y, "r": r, "label": col})

    detections = less_crowded(prelim, min_dist_factor=1.25)

    return {"success": True, "detections": detections, "w": W0, "h": H0}

if __name__ == "__main__":
    # Local dev entrypoint; Render will use Procfile command (uvicorn server:app --host 0.0.0.0 --port $PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
