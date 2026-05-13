"""
Workflow:
  [Stage 1: Waterline Calibration]
  1. Use LEFT CLICK to select one point on the left side and one point on the right side of the actual water surface (2 points total).
  2. Press Enter to confirm.

  [Stage 2: Egg Segmentation]
  3. LEFT CLICK on the egg center, RIGHT CLICK to exclude unwanted objects/noise.
  4. Press Enter to start high-precision nonlinear fitting.
"""

import cv2
import numpy as np
import torch
import urllib.request
import os
import sys
import argparse
from scipy.optimize import least_squares
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ══════════════════════════════════════════════════════
#  Configurable parameters
# ══════════════════════════════════════════════════════
DISPLAY_HEIGHT = 800
SAM2_MODEL = "small"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"[设备] {DEVICE.upper()}")

# ──────────────────────────────────────────────────────
# 1. Model loading
# ──────────────────────────────────────────────────────
MODEL_INFO = {
    "tiny": ("sam2.1_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
    "small": (
    "sam2.1_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"),
    "base_plus": (
    "sam2.1_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"),
    "large": (
    "sam2.1_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"),
}
SAM2_CFG = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def get_checkpoint(size):
    fname, url = MODEL_INFO[size]
    cache = os.path.expanduser("~/.cache/sam2")
    os.makedirs(cache, exist_ok=True)
    fpath = os.path.join(cache, fname)
    if os.path.exists(fpath): return fpath
    print(f"[loading] {fname} ...")
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    urllib.request.urlretrieve(url, fpath,
                               reporthook=lambda b, bs, total: print(f"\r  {b * bs / 1e6:.1f}/{total / 1e6:.1f} MB",
                                                                     end="", flush=True))
    print()
    return fpath


def build_predictor(size):
    print(f"[loading] SAM2-{size}...")
    model = build_sam2(SAM2_CFG[size], get_checkpoint(size), device=DEVICE)
    return SAM2ImagePredictor(model)


def run_sam2_with_points(img_bgr, predictor, points_orig, labels_orig):
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pts = np.array(points_orig, dtype=np.float32)
    lbls = np.array(labels_orig, dtype=np.int32)
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.uint8) * 255


# ──────────────────────────────────────────────────────
# 2. Narushin model
# ──────────────────────────────────────────────────────
def narushin_y(x, L, B, w, D_L4):

    x = np.clip(x, -L / 2 + 1e-6, L / 2 - 1e-6)

    term_A = np.sqrt(max(5.5 * L ** 2 + 11 * L * w + 4 * w ** 2, 1e-6))
    term_B = np.sqrt(max(L ** 2 + 2 * w * L + 4 * w ** 2, 1e-6))
    num_K = term_A * (np.sqrt(3) * B * L - 2 * D_L4 * term_B)
    den_K = np.sqrt(3) * B * L * (term_A - 2 * term_B)

    K = 0 if abs(den_K) < 1e-6 else num_K / den_K

    h_denom = L ** 2 + 8 * w * x + 4 * w ** 2
    term_main_sq = (L ** 2 - 4 * x ** 2) / np.maximum(h_denom, 1e-6)
    y_hug = (B / 2) * np.sqrt(np.maximum(term_main_sq, 0))

    py_num = L * (L ** 2 + 8 * w * x + 4 * w ** 2)
    py_den = 2 * (L - 2 * w) * x ** 2 + (L ** 2 + 8 * L * w - 4 * w ** 2) * x + 2 * L * w ** 2 + L ** 2 * w + L ** 3

    val_inside = py_num / np.maximum(py_den, 1e-6)
    correction = 1 - K * (1 - np.sqrt(np.maximum(val_inside, 0)))

    return y_hug * correction


# ──────────────────────────────────────────────────────
# 3. NLS fitting and PCA
# ──────────────────────────────────────────────────────
def fit_narushin_from_mask(mask):
    
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 10: return None

    pts = cnt.squeeze(1)
    X, Y = pts[:, 0], pts[:, 1]

    cx_init, cy_init = np.mean(X), np.mean(Y)

    cov_mat = np.cov(X - cx_init, Y - cy_init)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    major_idx = np.argmax(eigenvalues)
    v_major_pca = eigenvectors[:, major_idx]

    phi_init = np.arctan2(v_major_pca[1], v_major_pca[0])

    dx = X - cx_init
    dy = Y - cy_init
    x_proj = dx * np.cos(phi_init) + dy * np.sin(phi_init)
    y_proj = -dx * np.sin(phi_init) + dy * np.cos(phi_init)

    L_init = np.max(x_proj) - np.min(x_proj)
    B_init = np.max(y_proj) - np.min(y_proj)

    w_init = 0.0
    D_L4_init = B_init * 0.82

    p0 = [cx_init, cy_init, phi_init, L_init, B_init, w_init, D_L4_init]

    def residuals(p):
        cx, cy, phi, L, B, w, D_L4 = p

        dx_val = X - cx
        dy_val = Y - cy
        x_local = dx_val * np.cos(phi) + dy_val * np.sin(phi)
        y_local = -dx_val * np.sin(phi) + dy_val * np.cos(phi)

        y_theo = narushin_y(x_local, L, B, w, D_L4)
        res = np.abs(y_local) - y_theo

        out_right = x_local > L / 2
        out_left = x_local < -L / 2

        if np.any(out_right):
            res[out_right] = np.sqrt((x_local[out_right] - L / 2) ** 2 + y_local[out_right] ** 2)
        if np.any(out_left):
            res[out_left] = np.sqrt((x_local[out_left] + L / 2) ** 2 + y_local[out_left] ** 2)

        return res

    bounds = (
        [-np.inf, -np.inf, -np.inf, L_init * 0.8, B_init * 0.8, -L_init * 0.2, B_init * 0.5],
        [np.inf, np.inf, np.inf, L_init * 1.2, B_init * 1.2, L_init * 0.2, B_init * 1.1]
    )

    print("   -> Running...")
    res = least_squares(residuals, p0, bounds=bounds, loss='soft_l1', f_scale=2.0, max_nfev=1500)

    return res.x


# ──────────────────────────────────────────────────────
# 4. Angle Calculation
# ──────────────────────────────────────────────────────
def get_narushin_info(n_params, water_pts):
    cx, cy, phi, L, B, w, D_L4 = n_params

    p1, p2 = water_pts
    dx_w = p2[0] - p1[0]
    dy_w = p2[1] - p1[1]
    water_angle_rad = np.arctan2(dy_w, dx_w)

    v_major = np.array([L / 2 * np.cos(phi), L / 2 * np.sin(phi)])
    v_minor = np.array([-B / 2 * np.sin(phi), B / 2 * np.cos(phi)])

    egg_angle_rad = phi

    rel_angle_rad = abs(egg_angle_rad - water_angle_rad)
    rel_angle_deg = np.rad2deg(rel_angle_rad) % 180
    if rel_angle_deg > 90:
        rel_angle_deg = 180 - rel_angle_deg

    return {
        "params": n_params,  # [cx, cy, phi, L, B, w, D_L4]
        "v_major": v_major,
        "v_minor": v_minor,
        "angle": float(rel_angle_deg),
        "water_vector": (dx_w, dy_w)
    }

# ──────────────────────────────────────────────────────
# 5. Draw Asymmetric Fitting Results
# ──────────────────────────────────────────────────────
def draw_result(img, water_pts, info, mask):
    out = img.copy()
    H, W = out.shape[:2]

    if mask is not None:
        green = np.full_like(out, (0, 180, 0))
        out = np.where(mask[:, :, None] > 0, (out * 0.40 + green * 0.60).astype(np.uint8), out)

    wp1, wp2 = water_pts
    slope = (wp2[1] - wp1[1]) / (wp2[0] - wp1[0] + 1e-6)
    y_left = int(wp1[1] - slope * wp1[0])
    y_right = int(wp1[1] + slope * (W - wp1[0]))
    cv2.line(out, (0, y_left), (W, y_right), (255, 80, 0), 4, cv2.LINE_AA)

    if not info: return out

    cx, cy, phi, L, B, w, D_L4 = info["params"]
    v_maj, v_min = info["v_major"], info["v_minor"]

    cv2.line(out, (int(cx - v_maj[0]), int(cy - v_maj[1])), (int(cx + v_maj[0]), int(cy + v_maj[1])), (0, 230, 0), 4,
             cv2.LINE_AA)

    dx_w, dy_w = info["water_vector"]
    norm = np.hypot(dx_w, dy_w) + 1e-6
    ux, uy = dx_w / norm * (L / 2), dy_w / norm * (L / 2)
    cv2.line(out, (int(cx - ux), int(cy - uy)), (int(cx + ux), int(cy + uy)), (0, 0, 235), 3, cv2.LINE_AA)

    x_local = np.linspace(-L / 2, L / 2, 200)
    y_local = narushin_y(x_local, L, B, w, D_L4)

    X_top = cx + x_local * np.cos(phi) - y_local * np.sin(phi)
    Y_top = cy + x_local * np.sin(phi) + y_local * np.cos(phi)

    X_bot = cx + x_local * np.cos(phi) - (-y_local) * np.sin(phi)
    Y_bot = cy + x_local * np.sin(phi) + (-y_local) * np.cos(phi)

    pts_top = np.column_stack((X_top, Y_top)).astype(np.int32)
    pts_bot = np.column_stack((X_bot, Y_bot)).astype(np.int32)[::-1] 

    egg_contour = np.vstack((pts_top, pts_bot))

    cv2.polylines(out, [egg_contour], isClosed=True, color=(240, 240, 240), thickness=3, lineType=cv2.LINE_AA)

    lines = [
        (f"Angle     = {info['angle']:.3f} deg", (0, 255, 255)),
        (f"Length(L) = {L:.1f} px", (0, 230, 180)),
        (f"Width(B)  = {B:.1f} px", (0, 200, 255)),
        (f"Shape(w)  = {w:.2f}", (255, 150, 200)),
        (f"D(L/4)    = {D_L4:.2f}", (255, 150, 200))
    ]

    fs = max(W / 1400.0, 1.1)  
    line_spacing = int(60 * fs)  
    thickness = max(2, int(3 * fs))  
    y_cur = int(50 * fs)  

    for txt, col in lines:
        cv2.putText(out, txt, (30, y_cur), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thickness, cv2.LINE_AA)
        y_cur += line_spacing

    return out


# ──────────────────────────────────────────────────────
# 6. Multi-stage Interactive Interface
# ──────────────────────────────────────────────────────
class MultiStageSelector:
    def __init__(self, img, scale):
        self.img_orig = img
        self.scale = scale
        self.stage = 1
        self.water_pts_disp = []
        self.egg_pts_disp = []
        self.egg_labels = []
        self.display = None
        self.redraw()

    def redraw(self):
        H, W = self.img_orig.shape[:2]
        disp = cv2.resize(self.img_orig, (int(W * self.scale), int(H * self.scale)))

        if self.stage == 1:
            cv2.putText(disp, "[STAGE 1] Click 2 points to define Water Level", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 80, 0), 2)
            for p in self.water_pts_disp: cv2.circle(disp, p, 8, (255, 80, 0), -1)
            if len(self.water_pts_disp) == 2: cv2.line(disp, self.water_pts_disp[0], self.water_pts_disp[1],
                                                       (255, 80, 0), 2)
        elif self.stage == 2:
            cv2.putText(disp, "[STAGE 2] L-Click=Egg Center | R-Click=Exclude Tape", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)
            if len(self.water_pts_disp) == 2: cv2.line(disp, self.water_pts_disp[0], self.water_pts_disp[1],
                                                       (255, 80, 0), 2)
            for p, l in zip(self.egg_pts_disp, self.egg_labels):
                color = (0, 255, 0) if l == 1 else (0, 0, 255)
                cv2.circle(disp, p, 10, color, -1)
                cv2.circle(disp, p, 12, (255, 255, 255), 2)

        cv2.putText(disp, "Enter=Confirm   R=Reset Step   Q=Quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (200, 200, 200), 2)
        self.display = disp

    def on_mouse(self, event, x, y, flags, param):
        if self.stage == 1:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.water_pts_disp) < 2:
                    self.water_pts_disp.append((x, y))
                    self.redraw()
        elif self.stage == 2:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.egg_pts_disp.append((x, y))
                self.egg_labels.append(1)
                self.redraw()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.egg_pts_disp.append((x, y))
                self.egg_labels.append(0)
                self.redraw()

    def get_orig_water(self):
        return [(int(p[0] / self.scale), int(p[1] / self.scale)) for p in self.water_pts_disp]

    def get_orig_egg(self):
        return [(int(p[0] / self.scale), int(p[1] / self.scale)) for p in self.egg_pts_disp], self.egg_labels


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    args = parser.parse_args()
    IMAGE_PATH = args.image_path.strip()

    img = cv2.imread(IMAGE_PATH)
    if img is None: sys.exit(1)
    H, W = img.shape[:2]

    predictor = build_predictor(SAM2_MODEL)
    disp_scale = DISPLAY_HEIGHT / float(H)

    selector = MultiStageSelector(img, disp_scale)
    win = "SAM2 Academic Egg Measurement"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(W * disp_scale), DISPLAY_HEIGHT)
    cv2.setMouseCallback(win, selector.on_mouse)

    while True:
        cv2.imshow(win, selector.display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if selector.stage == 1:
                selector.water_pts_disp.clear()
            else:
                selector.egg_pts_disp.clear();
                selector.egg_labels.clear()
            selector.redraw()
        elif key in (13, 10):  # Enter
            if selector.stage == 1:
                if len(selector.water_pts_disp) != 2: print("Please select exactly 2 points to define the waterline!"); continue
                selector.stage = 2
                selector.redraw()
            elif selector.stage == 2:
                egg_pts, egg_lbls = selector.get_orig_egg()
                water_pts = selector.get_orig_water()
                if not egg_pts: print("Please select!"); continue

                cv2.destroyWindow(win)
                print("\n[Processing... SAM2 segmentation and Narushin model fitting]")

                mask = run_sam2_with_points(img, predictor, egg_pts, egg_lbls)
                narushin_params = fit_narushin_from_mask(mask)

                if narushin_params is not None:
                    info = get_narushin_info(narushin_params, water_pts)
                    print(f"Successful! Angle: {info['angle']:.3f}°")
                else:
                    info = None
                    print("Fitting failed!")

                result = draw_result(img, water_pts, info, mask)
                cv2.imwrite(IMAGE_PATH.replace(".jpg", "_sam2_narushin.jpg"), result)

                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Result", int(W * disp_scale), DISPLAY_HEIGHT)
                cv2.imshow("Result", result)
                cv2.waitKey(0)
                break
