# server.py — Demo mode: show ONLY red & yellow balls (no black/cue) with less-crowded labels
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

@app.get("/")
def root():
    return {"ok": True, "service": "cuezen-ball-detect", "version": "1.0.1-redyellow-demo"}

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

# ---------- helpers ----------
def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def dedupe(circles, thr=0.6):
    """Merge near-duplicate circles (keeps larger first)."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for x, y, r, lbl in circles:
        if not any(np.hypot(x - kx, y - ky) < thr * min(r, kr) for kx, ky, kr, _ in kept):
            kept.append((int(x), int(y), int(r), lbl))
    return kept

def felt_mask_hsv(hsv):
    """Find the cloth/felt region (tolerant of blue/green tables)."""
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
    """Approx ball radius in px based on table width."""
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

# For reference (OpenCV HSV H: 0..179)
COLOR_BINS = {
    "red":    [(0, 10), (170, 179)],
    "yellow": [(26, 40)],
}

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

def color_mask_passes_red_yellow(hsv, base_mask, r_est):
    """Proposals from ONLY red & yellow segmentation (no black/cue here)."""
    cand = []
    # yellow
    ymask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_BINS["yellow"]:
        ymask |= cv2.inRange(hsv, np.array([lo, 90, 110]), np.array([hi, 255, 255]))
    if base_mask is not None:
        ymask = cv2.bitwise_and(ymask, base_mask)
    cand += _circles_from_binary(ymask, "yellow", r_est)

    # red (wrap-around)
    rmask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_BINS["red"]:
        rmask |= cv2.inRange(hsv, np.array([lo, 90, 80]), np.array([hi, 255, 255]))
    if base_mask is not None:
        rmask = cv2.bitwise_and(rmask, base_mask)
    cand += _circles_from_binary(rmask, "red", r_est)

    return cand

def classify_red_or_yellow(hsv_full, x, y, r):
    """Dominant hue inside circle -> 'red' | 'yellow' | None."""
    mask = np.zeros(hsv_full.shape[:2], np.uint8)
    cv2.circle(mask, (x, y), max(2, int(r * 0.85)), 255, -1)
    hs = hsv_full[..., 0][mask > 0]
    ss = hsv_full[..., 1][mask > 0]
    vs = hsv_full[..., 2][mask > 0]
    if hs.size == 0:
        return None
    # Require reasonable saturation/brightness to avoid rails/floor
    valid = (ss > 70) & (vs > 70)
    if not np.any(valid):
        return None
    h = hs[valid]
    # Map hue to bins
    def in_range(angle, lo, hi):
        if lo <= hi:
            return (h >= lo) & (h <= hi)
        # wrap-around
        return (h >= lo) | (h <= hi)

    red_mask = in_range(h, 0, 10) | in_range(h, 170, 179)
    yellow_mask = in_range(h, 26, 40)

    red_ratio = np.count_nonzero(red_mask) / h.size
    yellow_ratio = np.count_nonzero(yellow_mask) / h.size

    if max(red_ratio, yellow_ratio) < 0.15:
        return None
    return "red" if red_ratio >= yellow_ratio else "yellow"

def suppress_crowding(circles, min_sep_factor=1.2):
    """
    Final pass to reduce crowded labels:
    keep circles so that center-to-center >= min_sep_factor * min(r_i, r_j).
    Prefer larger radius first.
    """
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for x, y, r, lbl in circles:
        if not any(np.hypot(x - kx, y - ky) < min_sep_factor * min(r, kr) for kx, ky, kr, _ in kept):
            kept.append((x, y, r, lbl))
    return kept

# ---------- endpoint ----------
@app.post("/detect")
async def detect_balls(file: UploadFile = File(...)):
    """
    DEMO behavior for client request:
      - Identify ONLY red and yellow balls.
      - Do NOT identify black (filtered out implicitly).
      - Return fewer, well-spaced labels (less crowded).
    """
    contents = await file.read()
    image_pil = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert("RGB")
    bgr_full = pil_to_cv2(image_pil)
    H0, W0 = bgr_full.shape[:2]

    # downscale for speed; use original for final coords/colors
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

    # Hough multi-pass for geometric proposals (kept, but we will color-filter later)
    proposals = []
    for p2, blur, dp, use_mask in [
        (30, 9, 1.2, True),
        (28, 9, 1.2, True),
        (26, 7, 1.2, True),
        (24, 9, 1.1, False),
    ]:
        proposals += run_hough(gray, mask_felt if use_mask else None, r_est, p2, blur, dp)

    # Add color-based proposals for red & yellow only
    proposals += color_mask_passes_red_yellow(hsv, mask_felt, r_est)

    # Merge duplicates
    merged = dedupe(proposals, thr=0.6)

    # Final color decision on FULL-RES image; filter to only red/yellow
    hsv_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)
    ry = []
    for x, y, r, lbl in merged:
        # back to original coordinates
        if scale != 1.0:
            x, y, r = int(x / scale), int(y / scale), int(r / scale)
        # ignore circles touching edges (rails/pockets)
        if x < r or y < r or x > (W0 - r) or y > (H0 - r):
            continue
        # Resolve color: if proposal had a color label, keep it; else classify
        color = lbl if lbl in ("red", "yellow") else classify_red_or_yellow(hsv_full, x, y, r)
        if color not in ("red", "yellow"):
            continue
        ry.append((x, y, r, color))

    # LESS-CROWDED: enforce bigger separation between final labels
    ry = suppress_crowding(ry, min_sep_factor=1.25)

    # Build response
    results = [{"x": int(x), "y": int(y), "r": int(r), "label": c} for x, y, r, c in ry]
    return {"success": True, "detections": results, "w": int(W0), "h": int(H0)}

if __name__ == "__main__":
    # Local dev entrypoint; Render uses your Procfile
    uvicorn.run(app, host="0.0.0.0", port=PORT)
