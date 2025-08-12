from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2
from io import BytesIO
from PIL import Image, ImageOps
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- helpers ----------
def pil_to_cv2(img_pil): return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def dedupe(circles, thr=0.6):
    if not circles: return []
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept=[]
    for x,y,r,lbl in circles:
        if not any(np.hypot(x-kx,y-ky)<thr*min(r,kr) for kx,ky,kr,_ in kept):
            kept.append((int(x),int(y),int(r),lbl))
    return kept

def felt_mask_hsv(hsv):
    h,w = hsv.shape[:2]
    cx1,cy1,cx2,cy2 = int(w*.4),int(h*.4),int(w*.6),int(h*.6)
    center = hsv[cy1:cy2, cx1:cx2]
    Hm,Sm,Vm = np.median(center[...,0]), np.median(center[...,1]), np.median(center[...,2])
    dyn = cv2.inRange(hsv,
        np.array([max(0, Hm-18), max(30, Sm*0.5), max(30, Vm*0.5)], np.uint8),
        np.array([min(179, Hm+18), 255, 255], np.uint8))
    green = cv2.inRange(hsv, np.array([35,60,40]), np.array([85,255,255]))
    blue  = cv2.inRange(hsv, np.array([85,40,40]), np.array([130,255,255]))
    mask = cv2.bitwise_or(dyn, cv2.bitwise_or(green, blue))
    mask = cv2.medianBlur(mask,7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8), 2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out,[max(cnts,key=cv2.contourArea)],-1,255,cv2.FILLED)
    return cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)),1)

def estimate_radius(mask, img_w):
    if mask is not None and mask.any():
        xs = np.where(mask>0)[1]
        if xs.size:
            width = xs.max()-xs.min()
            return max(6,int(width/39))
    return max(6,int(img_w/45))

def run_hough(gray, mask, r_est, param2, blur_ksize=9, dp=1.2):
    g = gray.copy()
    if mask is not None and mask.any(): g[mask==0] = 0
    g = cv2.GaussianBlur(g,(blur_ksize,blur_ksize),2)
    minR,maxR = max(6,int(r_est*.8)), int(r_est*1.45)
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=dp, minDist=int(r_est*1.1),
                           param1=110, param2=param2, minRadius=minR, maxRadius=maxR)
    out=[]
    if cir is not None:
        for x,y,r in np.round(cir[0]).astype("int"):
            out.append((int(x),int(y),int(r),None))
    return out

COLOR_BINS = [
    ("red",     [(0,10),(170,179)], 90, 80),
    ("orange",  [(11,25)],           90, 90),
    ("yellow",  [(26,40)],           90,110),
    ("green",   [(41,85)],           60, 70),
    ("blue",    [(86,130)],          60, 70),
    ("purple",  [(131,160)],         60, 70),
]

def _circles_from_binary(bin_mask, label, r_est):
    out=[]
    m = cv2.medianBlur(bin_mask,5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8),1)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rmin,rmax = max(6,int(r_est*.6)), int(r_est*1.6)
    for c in cnts:
        (x,y),r = cv2.minEnclosingCircle(c); x,y,r=int(x),int(y),int(r)
        if r<rmin or r>rmax: continue
        area = cv2.contourArea(c); circ = np.pi*r*r
        if circ<=0 or area/circ < 0.45: continue
        out.append((x,y,r,label))
    return out

def color_mask_passes(hsv, base_mask, r_est):
    cand=[]
    # cue (very white)
    cue = cv2.inRange(hsv, np.array([0,0,210]), np.array([179,55,255]))
    if base_mask is not None: cue = cv2.bitwise_and(cue, base_mask)
    cand += _circles_from_binary(cue, "cue", r_est)
    # black
    black = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,255,55]))
    if base_mask is not None: black = cv2.bitwise_and(black, base_mask)
    cand += _circles_from_binary(black, "black", r_est)
    # chromatic
    for name,ranges,smin,vmin in COLOR_BINS:
        cmask = np.zeros(hsv.shape[:2], np.uint8)
        for lo,hi in ranges:
            cmask |= cv2.inRange(hsv, np.array([lo,smin,vmin]), np.array([hi,255,255]))
        if base_mask is not None: cmask = cv2.bitwise_and(cmask, base_mask)
        cand += _circles_from_binary(cmask, name, r_est)
    return cand

# --- solid/stripe heuristic ---
def white_ratio(hsv_full, x, y, r):
    mask = np.zeros(hsv_full.shape[:2], np.uint8)
    cv2.circle(mask,(x,y),max(2,int(r*0.9)),255,-1)
    white = cv2.inRange(hsv_full, np.array([0,0,205]), np.array([179,70,255]))
    white = cv2.bitwise_and(white, mask)
    total = cv2.countNonZero(mask)
    return (cv2.countNonZero(white)/total) if total>0 else 0.0

def cue_or_object(hsv_full, x, y, r):
    # LAB chroma check to avoid bright yellow mis-flag as cue
    lab = cv2.cvtColor(cv2.cvtColor(hsv_full, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)
    mask = np.zeros(hsv_full.shape[:2], np.uint8)
    cv2.circle(mask,(x,y),max(2,int(r*0.8)),255,-1)
    L,a,b,_ = cv2.mean(lab, mask=mask)
    wr = white_ratio(hsv_full,x,y,r)
    if wr > 0.78 and (abs(a-128)+abs(b-128)) < 22:   # low chroma
        return "cue"
    return "object"

def solid_or_stripe(hsv_full, x, y, r):
    wr = white_ratio(hsv_full,x,y,r)
    if wr >= 0.42: return "object (stripe)"
    if wr <= 0.22: return "object (solid)"
    return "object (solid)"  # bias toward solid when uncertain

# ---------- endpoint ----------
@app.post("/detect")
async def detect_balls(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert("RGB")
    bgr_full = pil_to_cv2(image_pil)
    H0,W0 = bgr_full.shape[:2]

    MAX_DIM = 1100
    scale = 1.0
    if max(H0,W0) > MAX_DIM:
        scale = MAX_DIM/float(max(H0,W0))
        bgr = cv2.resize(bgr_full, (int(W0*scale), int(H0*scale)))
    else:
        bgr = bgr_full.copy()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_felt = felt_mask_hsv(hsv)
    if (mask_felt>0).sum() < 0.1*mask_felt.size: mask_felt=None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    r_est = estimate_radius(mask_felt, bgr.shape[1])

    proposals=[]
    for p2,blur,dp,use_mask in [
        (30,9,1.2,True),(28,9,1.2,True),(26,7,1.2,True),(24,9,1.1,False)
    ]:
        proposals += run_hough(gray, mask_felt if use_mask else None, r_est, p2, blur, dp)

    proposals += color_mask_passes(hsv, mask_felt, r_est)
    merged = dedupe(proposals, thr=0.6)

    hsv_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2HSV)
    results=[]
    for x,y,r,lbl in merged:
        if scale!=1.0: x,y,r=int(x/scale),int(y/scale),int(r/scale)
        if x<r or y<r or x>(W0-r) or y>(H0-r): continue
        base = cue_or_object(hsv_full,x,y,r) if lbl is None else ("cue" if lbl=="cue" else "object")
        final_label = "cue" if base=="cue" else solid_or_stripe(hsv_full,x,y,r)
        results.append({"x":x,"y":y,"r":r,"label":final_label})

    return {"success":True, "detections":results, "w":W0, "h":H0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
