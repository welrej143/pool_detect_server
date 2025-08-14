# server.py — cue vs object labels + red/yellow focus + strong cue guard + smart de-crowding
import os
from io import BytesIO
from collections import defaultdict

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
import uvicorn

# --- Render/container friendliness ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
try:
    cv2.setNumThreads(1)  # type: ignore[attr-defined]
except Exception:
    pass

PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Toggle: keep red/yellow-only for objects
ONLY_RED_YELLOW = True

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "service": "cuezen-ball-detect", "version": "1.3.1"}

# Some platforms send HEAD probes; avoid 405 noise.
@app.head("/")
def root_head():
    return {}

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

# color bins for proposals
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
    cue = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 55, 255]))
    if base_mask is not None:
        cue = cv2.bitwise_and(cue, base_mask)
    cand += _circles_from_binary(cue, "cue", r_est)

    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 55]))
    if base_mask is not None:
        black = cv2.bitwise_and(black, base_mask)
    cand += _circles_from_binary(black, "black", r_est)

    for name, ranges, smin, vmin in COLOR_BINS:
        cmask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            cmask |= cv2.inRange(hsv, np.array([lo, smin, vmin]), np.array([hi, 255, 255]))
        if base_mask is not None:
            cmask = cv2.bitwise_and(cmask, base_mask)
        cand += _circles_from_binary(cmask, name, r_est)
    return cand

# ---------- cue guard + color/stripe ----------
def _ring_mask(shape_hw, x, y, r, r_in=0.35, r_out=0.85):
    h, w = shape_hw
    outer = np.zeros((h, w), np.uint8)
    inner = np.zeros((h, w), np.uint8)
    cv2.circle(outer, (x, y), max(2, int(r * r_out)), 255, -1)
    cv2.circle(inner, (x, y), max(1, int(r * r_in)), 255, -1)
    return cv2.subtract(outer, inner)

def _disk_mask(shape_hw, x, y, r, frac=0.45):
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (x, y), max(1, int(r * frac)), 255, -1)
    return m

def _white_mask_ycc(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycc)
    m1 = cv2.inRange(Y, 165, 255)
    m2 = cv2.inRange(Cr, 118, 150)
    m3 = cv2.inRange(Cb, 118, 150)
    return cv2.bitwise_and(m1, cv2.bitwise_and(m2, m3))

def is_cue_ball(bgr_full, hsv_full, x, y, r):
    # strong but simple thresholds
    ring = _ring_mask(bgr_full.shape[:2], x, y, r, 0.25, 0.90)

    wm = cv2.bitwise_and(_white_mask_ycc(bgr_full), ring)
    ring_area = max(1, cv2.countNonZero(ring))
    white_frac = cv2.countNonZero(wm) / ring_area

    S = hsv_full[..., 1]; V = hsv_full[..., 2]
    s_vals = S[ring > 0]; v_vals = V[ring > 0]
    s_med = float(np.median(s_vals)) if s_vals.size else 255
    v_med = float(np.median(v_vals)) if v_vals.size else 0

    lab = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2LAB)
    a_vals = lab[..., 1][ring > 0]; b_vals = lab[..., 2][ring > 0]
    if a_vals.size and b_vals.size:
        chroma = abs(float(np.median(a_vals)) - 128.0) + abs(float(np.median(b_vals)) - 128.0)
    else:
        chroma = 999.0

    return (white_frac >= 0.45 and s_med <= 55 and v_med >= 160) or \
           (white_frac >= 0.60 and chroma <= 26)

def classify_color_hsv(hsv_full, x, y, r):
    ring = _ring_mask(hsv_full.shape[:2], x, y, r, 0.35, 0.85)
    H, S, V = cv2.split(hsv_full)
    hs = S[ring > 0]; hv = V[ring > 0]; hh = H[ring > 0]
    mask_col = (hs > 60) & (hv > 70)
    colored_h = hh[mask_col]
    if colored_h.size == 0:
        return "unknown"
    h_med = int(np.median(colored_h))
    if (0 <= h_med <= 10) or (170 <= h_med <= 179):
        return "red"
    if 26 <= h_med <= 40:
        return "yellow"
    if 11 <= h_med <= 25:
        return "orange"
    if 41 <= h_med <= 85:
        return "green"
    if 86 <= h_med <= 130:
        return "blue"
    if 131 <= h_med <= 160:
        return "purple"
    return "unknown"

def is_stripe_ball(bgr_full, x, y, r, inner_frac=0.45, white_thr=0.18, white_vs_color_bias=0.08):
    disk = _disk_mask(bgr_full.shape[:2], x, y, r, inner_frac)
    ring = _ring_mask(bgr_full.shape[:2], x, y, r, 0.45, 0.90)
    wm = _white_mask_ycc(bgr_full)
    inner_white = cv2.countNonZero(cv2.bitwise_and(wm, disk))
    inner_area = max(1, cv2.countNonZero(disk))
    inner_white_frac = inner_white / inner_area

    hsv_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)
    S = hsv_full[..., 1]
    ring_sat = S[ring > 0]
    colored_bias = float(np.mean(ring_sat > 80)) if ring_sat.size else 0.0

    return (inner_white_frac >= white_thr) and (inner_white_frac + white_vs_color_bias >= colored_bias)

# ---------- scoring + color-aware NMS (used for objects only) ----------
def _ring(gray_full, x, y, r):
    return _ring_mask(gray_full.shape[:2], x, y, r, 0.35, 0.90)

def _edge_strength_on_ring(gray_full, x, y, r):
    ring = _ring(gray_full, x, y, r)
    gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[ring > 0]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, 75))

def _color_confidence(hsv_full, label, x, y, r):
    ring_mask = _ring(hsv_full[..., 0], x, y, r)  # 2D mask
    H, S, V = cv2.split(hsv_full)
    hh = H[ring_mask > 0]; ss = S[ring_mask > 0]; vv = V[ring_mask > 0]
    if hh.size == 0:
        return 0.0

    mask_col = (ss > 60) & (vv > 70)
    if not np.any(mask_col):
        return 0.0
    h = hh[mask_col]

    def hue_dist_to_range(hvals, ranges):
        dmin = np.inf * np.ones_like(hvals, dtype=np.float32)
        for lo, hi in ranges:
            d = np.where((hvals >= lo) & (hvals <= hi), 0.0,
                         np.minimum(np.abs(hvals - lo), np.abs(hvals - hi)))
            d = np.minimum(d, np.minimum(np.abs(hvals + 180 - lo), np.abs(hvals - 180 - hi)))
            dmin = np.minimum(dmin, d.astype(np.float32))
        return dmin

    if label == "red":
        ranges = [(0, 10), (170, 179)]
    elif label == "yellow":
        ranges = [(26, 40)]
    else:
        return 0.0

    d = hue_dist_to_range(h.astype(np.float32), ranges)
    conf = 1.0 - np.clip(np.median(d) / 40.0, 0.0, 1.0)
    # ✅ FIX: don't re-index with the 2D ring mask; ss is already 1-D
    sat_boost = float(np.mean(ss > 100)) * 0.15
    return float(np.clip(conf + sat_boost, 0.0, 1.0))

def _score_detection(bgr_full, hsv_full, gray_full, d):
    x, y, r = d["x"], d["y"], d["r"]
    lbl = d["label"]
    edge = _edge_strength_on_ring(gray_full, x, y, r)
    e_ref = max(1e-6, np.percentile(gray_full, 95))
    edge_norm = float(np.clip(edge / e_ref, 0.0, 1.5))
    col_conf = _color_confidence(hsv_full, lbl, x, y, r)
    rnorm = np.clip(r / max(1.0, 0.5 * (bgr_full.shape[0] + bgr_full.shape[1]) / 90.0), 0.4, 1.6)
    size_prior = 1.0 - float(abs(rnorm - 1.0)) * 0.35
    return float(0.55 * col_conf + 0.35 * edge_norm + 0.10 * size_prior)

def _less_crowded_smart_for_objects(bgr_full, dets, color_supp_mult=1.45, cross_color_mult=1.10,
                                    base_mult=1.10, grid_cap=3, grid_px=90):
    if not dets:
        return dets
    gray = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)
    for d in dets:
        d["score"] = _score_detection(bgr_full, hsv, gray, d)

    keep = []
    occupied = defaultdict(int)
    def cell_of(x, y): return (int(x // grid_px), int(y // grid_px))

    for d in sorted(dets, key=lambda z: z["score"], reverse=True):
        x, y, r, lbl = d["x"], d["y"], d["r"], d["label"]
        cx, cy = cell_of(x, y)
        if occupied[(cx, cy)] >= grid_cap:
            continue
        ok = True
        for k in keep:
            dx = x - k["x"]; dy = y - k["y"]; dist = (dx*dx + dy*dy) ** 0.5
            mult = color_supp_mult if (lbl == k["label"]) else cross_color_mult
            thr = mult * base_mult * min(r, k["r"])
            if dist < thr:
                ok = False; break
        if ok:
            keep.append(d); occupied[(cx, cy)] += 1
    return keep

# ---------- core detect ----------
def _detect_core(bgr_full):
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

    hsv_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)

    prelim_objects = []
    prelim_cues = []
    for x, y, r, _ in merged:
        if scale != 1.0:
            x, y, r = int(x / scale), int(y / scale), int(r / scale)
        if x < r or y < r or x > (W0 - r) or y > (H0 - r):
            continue

        if is_cue_ball(bgr_full, hsv_full, x, y, r):
            prelim_cues.append({"x": x, "y": y, "r": r, "label": "cue", "kind": "cue"})
            continue

        col = classify_color_hsv(hsv_full, x, y, r)
        if ONLY_RED_YELLOW and col not in ("red", "yellow"):
            continue

        stripe = is_stripe_ball(bgr_full, x, y, r)
        prelim_objects.append({
            "x": x, "y": y, "r": r,
            "label": col, "pattern": "stripe" if stripe else "solid",
            "kind": "object"
        })

    objects = _less_crowded_smart_for_objects(
        bgr_full,
        prelim_objects,
        color_supp_mult=1.45,
        cross_color_mult=1.10,
        base_mult=1.10,
        grid_cap=3,
        grid_px=max(70, int(1.5 * np.median([d["r"] for d in prelim_objects])) if prelim_objects else 90)
    )

    detections = prelim_cues + objects
    return detections, (W0, H0)

# ---------- endpoints ----------
@app.post("/detect")
async def detect_balls(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image_pil = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert("RGB")
    except UnidentifiedImageError:
        return {"success": False, "error": "Invalid image file."}
    bgr_full = pil_to_cv2(image_pil)
    detections, (W0, H0) = _detect_core(bgr_full)
    return {"success": True, "detections": detections, "w": W0, "h": H0}

@app.post("/annotate")
async def annotate_balls(file: UploadFile = File(...)):
    """
    Returns a PNG with color-coded outlines (no text labels):
      - cue    = white outline
      - solid  = green outline
      - stripe = blue outline
    """
    contents = await file.read()
    try:
        image_pil = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert("RGB")
    except UnidentifiedImageError:
        # Return a tiny blank PNG for robustness
        blank = Image.new("RGB", (2, 2), (0, 0, 0))
        buf = BytesIO(); blank.save(buf, format="PNG"); buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    bgr_full = pil_to_cv2(image_pil).copy()
    detections, _ = _detect_core(bgr_full)

    for d in detections:
        x, y, r = d["x"], d["y"], d["r"]
        if d.get("kind") == "cue":
            color = (255, 255, 255)   # white
        else:
            color = (255, 0, 0) if d.get("pattern") == "stripe" else (0, 255, 0)  # blue/green in BGR
        thickness = max(2, r // 10)
        cv2.circle(bgr_full, (x, y), r, color, thickness)
        cv2.circle(bgr_full, (x, y), max(2, r // 14), color, -1)

    rgb = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(rgb)
    buf = BytesIO()
    pil_out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
