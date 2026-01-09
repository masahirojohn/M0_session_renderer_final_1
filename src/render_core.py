from __future__ import annotations
import os, json
from typing import Dict, Any
import math
import numpy as np
import cv2
from collections import defaultdict

# =========================
# yaw/pitch stats (runtime)
# =========================
_YAW_PITCH_STATS = defaultdict(list)

def _record_yaw_pitch(yaw_deg: float, pitch_deg: float) -> None:
    _YAW_PITCH_STATS["yaw"].append(float(yaw_deg))
    _YAW_PITCH_STATS["pitch"].append(float(pitch_deg))

def _dump_yaw_pitch_stats() -> None:
    def _pct(vs, p):
        if not vs:
            return None
        vs = sorted(vs)
        k = int(round((len(vs)-1) * p))
        return vs[k]

    yaw = _YAW_PITCH_STATS.get("yaw", [])
    pit = _YAW_PITCH_STATS.get("pitch", [])
    if not yaw and not pit:
        return

    print("[M0][Yaw/Pitch Stats]")
    if yaw:
        print(f"  yaw  : min={min(yaw):.2f}, max={max(yaw):.2f}, "
              f"p50={_pct(yaw,0.50):.2f}, p95={_pct(yaw,0.95):.2f}")
    if pit:
        print(f"  pitch: min={min(pit):.2f}, max={max(pit):.2f}, "
              f"p50={_pct(pit,0.50):.2f}, p95={_pct(pit,0.95):.2f}")


# -----------------------------
# 画像I/Oユーティリティ
# -----------------------------
def _load_rgba(path: str) -> np.ndarray:
    """PNGなどを BGRA で読む。アルファ無しなら255で補完。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[2] == 3:
        bgr = img
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([bgr, a], axis=2)
    return img


def _ensure_bgra(img: np.ndarray) -> np.ndarray:
    """BGR/BGRA/GRAY などを BGRA に揃える。"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 4:
        pass
    else:
        raise ValueError(f"Unsupported img shape: {img.shape}")
    return img


def _alpha_paste(canvas_bgra: np.ndarray, src_bgra: np.ndarray, cx: int, cy: int) -> None:
    """src をアルファブレンドで canvas に貼り付ける。両方 BGRA 前提。"""
    h, w = canvas_bgra.shape[:2]
    sh, sw = src_bgra.shape[:2]

    x0 = int(cx - sw // 2)
    y0 = int(cy - sh // 2)
    x1 = x0 + sw
    y1 = y0 + sh

    # 画面外クリップ
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    sx1 = sw - max(0, x1 - w)
    sy1 = sh - max(0, y1 - h)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    if dx0 >= dx1 or dy0 >= dy1:
        return

    src_crop = src_bgra[sy0:sy1, sx0:sx1]
    dst_crop = canvas_bgra[dy0:dy1, dx0:dx1]

    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
    inv = 1.0 - alpha

    dst_crop[:, :, :3] = (src_crop[:, :, :3].astype(np.float32) * alpha +
                          dst_crop[:, :, :3].astype(np.float32) * inv).astype(np.uint8)

    dst_crop[:, :, 3:4] = np.clip(
        src_crop[:, :, 3:4].astype(np.float32) + dst_crop[:, :, 3:4].astype(np.float32) * inv,
        0, 255
    ).astype(np.uint8)

    canvas_bgra[dy0:dy1, dx0:dx1] = dst_crop


def _alpha_blend_full(canvas_bgra: np.ndarray, overlay_bgra: np.ndarray) -> None:
    """
    overlay_bgra は canvas と同サイズの BGRA。
    overlay の alpha を使って canvas に上書きブレンドする。
    """
    if overlay_bgra.shape[:2] != canvas_bgra.shape[:2]:
        raise ValueError("overlay size must match canvas size")

    oa = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    if oa.max() <= 0.0:
        return

    cb = canvas_bgra[:, :, :3].astype(np.float32)
    ob = overlay_bgra[:, :, :3].astype(np.float32)

    out_bgr = ob * oa + cb * (1.0 - oa)
    canvas_bgra[:, :, :3] = np.clip(out_bgr, 0, 255).astype(np.uint8)

    # 出力は不透明キャンバス前提なら alpha=255 のままでOK
    # もし透明を維持したいなら canvas alpha も合成するが、現状は 255 固定で良い。



def warp_similarity_2pt_to_canvas(
    sprite_bgra: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    2点（両目）から Similarity（回転+拡縮+平行移動）を作り、
    sprite_bgra をキャンバスへワープする（shear 無し）。
    OpenCV の estimateAffinePartial2D の robust method に依存しない安全版。
    """
    if src_pts.shape != (2, 2) or dst_pts.shape != (2, 2):
        raise ValueError("src_pts/dst_pts must be shape (2,2)")

    import math

    sx1, sy1 = float(src_pts[0, 0]), float(src_pts[0, 1])
    sx2, sy2 = float(src_pts[1, 0]), float(src_pts[1, 1])
    dx1, dy1 = float(dst_pts[0, 0]), float(dst_pts[0, 1])
    dx2, dy2 = float(dst_pts[1, 0]), float(dst_pts[1, 1])

    svx, svy = (sx2 - sx1), (sy2 - sy1)
    dvx, dvy = (dx2 - dx1), (dy2 - dvy)

    s_dist = math.hypot(svx, svy)
    d_dist = math.hypot(dvx, dvy)
    if s_dist < 1e-6 or d_dist < 1e-6:
        return np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    scale = d_dist / s_dist
    s_ang = math.atan2(svy, svx)
    d_ang = math.atan2(dvy, dvx)
    rot = d_ang - s_ang

    c = math.cos(rot) * scale
    s = math.sin(rot) * scale

    # src 眼中心 → dst 眼中心 へ
    scx, scy = (sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5
    dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dcy) * 0.5 # dcx, dcy の typo 防止修正含む場合もあるが、元コード維持

    # ※元コードに合わせます
    dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dy2) * 0.5

    tx = dcx - (c * scx - s * scy)
    ty = dcy - (s * scx + c * scy)

    M = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)
    warped = cv2.warpAffine(
        sprite_bgra,
        M,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def warp_affine_sprite_to_canvas(
    sprite_bgra: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    sprite_bgra を、src_pts(スプライト内3点) → dst_pts(背景内3点) に一致するようにアフィン変形し、
    キャンバスサイズの BGRA 画像として返す（透明背景）。
    """
    if src_pts.shape != (3, 2) or dst_pts.shape != (3, 2):
        raise ValueError("src_pts/dst_pts must be shape (3,2)")

    # Similarity（shear無し）で推定する
    src = src_pts.astype(np.float32)
    dst = dst_pts.astype(np.float32)

    # estimateAffinePartial2D は N>=2 点を受け取れる（2点でも3点でもOK）
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst,
        # method は指定しない（= least squares）ほうが「解が飛びにくい」
        # ※LMEDS は少点数だとフレーム毎に解が切り替わりやすく、痙攣の原因になりがち
    )

    if M is None:
        # フォールバック：従来の3点アフィン（最終保険）
        M = cv2.getAffineTransform(src[:3], dst[:3])

    # ---- [ADD] フレーム間の “痙攣” を抑えるための M 平滑化（回転・scale・平行移動） ----
    # 関数属性で直前のMを保持（レンダリング1本の中でのみ効く）
    prev_M = getattr(warp_affine_sprite_to_canvas, "_prev_M", None)
    if prev_M is not None and prev_M.shape == (2, 3) and M.shape == (2, 3):
        # M を similarity と仮定して分解： [ sR | t ]
        def _decompose_similarity(A: np.ndarray):
            a, b, tx = float(A[0, 0]), float(A[0, 1]), float(A[0, 2])
            c, d, ty = float(A[1, 0]), float(A[1, 1]), float(A[1, 2])
            s = (a * a + c * c) ** 0.5
            if s < 1e-6:
                s = 1e-6
            # cos, sin
            cos = a / s
            sin = c / s
            # rot(rad)
            r = float(np.arctan2(sin, cos))
            return s, r, tx, ty

        def _recompose_similarity(s: float, r: float, tx: float, ty: float):
            cos = float(np.cos(r))
            sin = float(np.sin(r))
            return np.array([[s * cos, -s * sin, tx],
                             [s * sin,  s * cos, ty]], dtype=np.float32)

        s0, r0, tx0, ty0 = _decompose_similarity(prev_M)
        s1, r1, tx1, ty1 = _decompose_similarity(M)

        # 角度差を -pi..pi に正規化
        dr = r1 - r0
        while dr > np.pi:
            dr -= 2.0 * np.pi
        while dr < -np.pi:
            dr += 2.0 * np.pi
        r1 = r0 + dr

        # 1フレームあたりの変化量を制限（25fps想定）
        # scale：±2%/frame、rot：±3deg/frame、平行移動：±20px/frame
        max_scale_ratio = 1.02
        s1 = max(s0 / max_scale_ratio, min(s0 * max_scale_ratio, s1))

        max_dr = np.deg2rad(3.0)
        r1 = max(r0 - max_dr, min(r0 + max_dr, r1))

        max_dt = 20.0
        tx1 = max(tx0 - max_dt, min(tx0 + max_dt, tx1))
        ty1 = max(ty0 - max_dt, min(ty0 + max_dt, ty1))

        # EMA（小さめに効かせる）
        alpha = 0.20
        s = (1.0 - alpha) * s0 + alpha * s1
        r = (1.0 - alpha) * r0 + alpha * r1
        tx = (1.0 - alpha) * tx0 + alpha * tx1
        ty = (1.0 - alpha) * ty0 + alpha * ty1

        M = _recompose_similarity(s, r, tx, ty)

    warp_affine_sprite_to_canvas._prev_M = M.copy()

    warped = cv2.warpAffine(
        sprite_bgra,
        M,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


# -----------------------------
# 口形名 正規化
# -----------------------------
def normalize_mouth_label(mouth: str) -> str:
    if not mouth:
        return "closed"
    m = mouth.lower()
    if m in ("close", "mouth_close"):
        return "closed"
    return m


# -----------------------------
# atlas 読み込み（★expressionメタも素通し）
# -----------------------------
def load_atlas_index(atlas_json_path: str) -> Dict[str, Any]:
    """
    atlas.min.json の実体を内部形式に正規化する。

    - トップレベルに front/left30/right30/... がある旧形式もサポート
    - data["views"][view][mouth] で必ず参照できるようにする
    - expression_labels / expression_default などはそのまま返す
    """
    with open(atlas_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    views = data.get("views")
    if not isinstance(views, dict):
        views = {}
        for key, value in data.items():
            # front / left30 / right30 ... のようなビュー辞書を拾う
            if isinstance(value, dict) and "closed" in value:
                # mouthキーは小文字に統一
                views[key] = {str(m).lower(): path for m, path in value.items()}
        data["views"] = views
    else:
        # mouthキーを小文字に揃えておく
        norm_views = {}
        for vname, vdict in views.items():
            if isinstance(vdict, dict):
                norm_views[vname] = {str(m).lower(): path for m, path in vdict.items()}
        data["views"] = norm_views

    # view_rules はそのまま
    if "view_rules" not in data:
        data["view_rules"] = {}

    # fallback はそのまま
    if "fallback" not in data:
        data["fallback"] = {"view": "front", "mouth": "closed"}

    return data


def load_affine_points_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# view 選択
# -----------------------------
def choose_view_from_yaw(yaw_deg: float, view_rules: Dict[str, Any]) -> str:
    left_max = float(view_rules.get("left30_max_yaw_deg", -12.0))
    right_min = float(view_rules.get("right30_min_yaw_deg", 12.0))
    if yaw_deg <= left_max:
        return "left30"
    if yaw_deg >= right_min:
        return "right30"
    return "front"


def choose_view_from_yaw_pitch(yaw_deg: float, pitch_deg: float, view_rules: dict[str, float], views_available: set[str] | None = None) -> str:
    """
    2D（yaw/pitch）で view を選ぶ（sprite がある前提で中間 view も使う）。

    - yaw系 : left30 / left15 / front / right15 / right30
    - pitch系: up15 / up7 / front / down7 / down15

    優先順位は「絶対値が大きい軸を採用」。
    """
    # --- yaw thresholds ---
    left30_max  = float(view_rules.get("left30_max_yaw_deg",  -12.0))
    right30_min = float(view_rules.get("right30_min_yaw_deg",  12.0))
    left15_max  = float(view_rules.get("left15_max_yaw_deg",   -6.0))
    right15_min = float(view_rules.get("right15_min_yaw_deg",   6.0))

    # --- pitch thresholds ---
    up15_max   = float(view_rules.get("up15_max_pitch_deg",   -10.0))
    down15_min = float(view_rules.get("down15_min_pitch_deg",  10.0))
    up7_max    = float(view_rules.get("up7_max_pitch_deg",     -5.0))
    down7_min  = float(view_rules.get("down7_min_pitch_deg",    5.0))

    ay = abs(float(yaw_deg))
    ap = abs(float(pitch_deg))

    # --- diagonal composite sprites (optional) ---
    # 条件: abs(yaw) >= diag_yaw_min && abs(pitch) >= diag_pitch_min
    # かつ「中間域」（30°/15°の極端を除外）で、対応スプライトが存在するときのみ採用する。
    diag_yaw_min  = float(view_rules.get("diag_yaw_min_deg", 10.0))
    diag_pitch_min = float(view_rules.get("diag_pitch_min_deg", 5.0))
    diag_yaw_max  = float(view_rules.get("diag_yaw_max_deg",  right30_min))   # 通常は right30 直前
    diag_pitch_max = float(view_rules.get("diag_pitch_max_deg", down15_min))  # 通常は down15 直前

    if views_available and (ay >= diag_yaw_min) and (ap >= diag_pitch_min) and (ay < diag_yaw_max) and (ap < diag_pitch_max):
        if yaw_deg < 0.0 and pitch_deg < 0.0:
            v = "left15_up7"
        elif yaw_deg < 0.0 and pitch_deg > 0.0:
            v = "left15_down7"
        elif yaw_deg > 0.0 and pitch_deg < 0.0:
            v = "right15_up7"
        else:
            v = "right15_down7"
        if v in views_available:
            return v

    # pitch 優先（絶対値が yaw 以上）
    if ap >= ay:
        if pitch_deg <= up15_max:
            return "up15"
        if pitch_deg <= up7_max:
            return "up7"
        if pitch_deg >= down15_min:
            return "down15"
        if pitch_deg >= down7_min:
            return "down7"
        return "front"

    # yaw 優先
    if yaw_deg <= left30_max:
        return "left30"
    if yaw_deg <= left15_max:
        return "left15"
    if yaw_deg >= right30_min:
        return "right30"
    if yaw_deg >= right15_min:
        return "right15"
    return "front"


# -----------------------------
# sprite path 解決
# -----------------------------
def _resolve_base_sprite_path(atlas_idx: Dict[str, Any], view: str, mouth: str) -> tuple[str | None, bool]:
    """
    atlas_idx['views'][view][mouth] を引いて、相対パスを返す。
    見つからなければ fallback を使う。
    戻り値: (path_rel or None, used_fallback: bool)
    """
    views = atlas_idx.get("views", {})
    used_fallback = False

    view_dict = views.get(view)
    if isinstance(view_dict, dict):
        p = view_dict.get(mouth)
        if isinstance(p, str):
            return p, used_fallback

    # fallback view/mouth
    fb = atlas_idx.get("fallback", {})
    fb_view = str(fb.get("view", "front"))
    fb_mouth = normalize_mouth_label(str(fb.get("mouth", "closed")))

    fb_view_dict = views.get(fb_view)
    if isinstance(fb_view_dict, dict):
        p = fb_view_dict.get(fb_mouth)
        if isinstance(p, str):
            used_fallback = True
            return p, used_fallback

    return None, True


def _derive_expression_path(
    atlas_idx: Dict[str, Any],
    view: str,
    mouth: str,
    expression: str | None,
    base_path_rel: str,
) -> str:
    """
    expression ラベルとベースPNGパスから、
    assets_dir/<expr>_<view>/<mouth_xxx.png> を導出する。

    - expression が None の場合や "normal" の場合は base_path_rel をそのまま返す
    - expression_labels に含まれないラベルなら無視して base_path_rel を返す
    """
    expr_default = str(atlas_idx.get("expression_default", "normal")).lower()
    expr = (expression or expr_default).lower()

    if expr in ("", "normal"):
        return base_path_rel

    labels = [str(x).lower() for x in atlas_idx.get("expression_labels", [])]
    if labels and expr not in labels:
        # 未知のラベル → normal と同じ扱い
        return base_path_rel

    base_name = os.path.basename(base_path_rel)
    expr_dir = f"{expr}_{view}"
    expr_path_rel = os.path.join(expr_dir, base_name)
    return expr_path_rel.replace("\\", "/")


# -----------------------------
# 変形（yaw/pitch/roll）
# -----------------------------

# ------------------------------------------------------------
# Continuous yaw/pitch: 2pt similarity + nose-diff lower warp
# - (a) yaw/pitch の補間ターゲット生成
# - (b) similarity 後に 鼻差分ベースの下半分 warpPerspective
# - (c) yaw/pitch のEMA＋速度制限
# ------------------------------------------------------------

# NOTE:
# - ここで使う "front/left30/right30/up15/down15" の src_pts は
#   「720x720 スプライト上の（両目中心・鼻先）」の座標。
# - view は m0_runner 側で確定していても、continuous は view 非依存で効く前提。
# - yaw/pitch の角度は「deg」で入ってくる想定。

_CONT_LMK_720 = {
    "front": np.array([[260.0, 400.0], [460.0, 400.0], [355.0, 485.0]], dtype=np.float32),
    # --- yaw intermediate (coords only; no extra sprites required) ---
    "left15": np.array([[250.0, 400.0], [437.0, 400.0], [333.0, 485.0]], dtype=np.float32),
    "right15": np.array([[280.0, 400.0], [473.0, 400.0], [383.0, 485.0]], dtype=np.float32),
    "left30": np.array([[245.0, 400.0], [420.0, 400.0], [310.0, 485.0]], dtype=np.float32),
    "right30": np.array([[305.0, 400.0], [480.0, 400.0], [410.0, 485.0]], dtype=np.float32),
    "up15": np.array([[265.0, 380.0], [460.0, 380.0], [357.0, 450.0]], dtype=np.float32),
    "down15": np.array([[265.0, 420.0], [465.0, 420.0], [357.0, 520.0]], dtype=np.float32),
    # --- pitch intermediate (coords only) ---
    "up7": np.array([[260.0, 390.0], [460.0, 390.0], [357.0, 460.0]], dtype=np.float32),
    "down7": np.array([[265.0, 410.0], [460.0, 410.0], [357.0, 500.0]], dtype=np.float32),
}

_BASE_VIEW_ANGLES_DEG = {
    "front":  (0.0, 0.0),
    "left15": (-15.0, 0.0),
    "right15": (15.0, 0.0),
    "left30": (-30.0, 0.0),
    "right30": (30.0, 0.0),
    "up7":   (0.0, -7.5),   # pitch: - が上
    "down7": (0.0, 7.5),    # pitch: + が下
    "up15":   (0.0, -15.0),   # pitch: - が上
    "down15": (0.0, 15.0),    # pitch: + が下
}

_POSE_SMOOTH_STATE = {
    "yaw": None,
    "pitch": None,
}

def reset_pose_smoother():
    _POSE_SMOOTH_STATE["yaw"] = None
    _POSE_SMOOTH_STATE["pitch"] = None

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _ema(prev: float | None, x: float, alpha: float) -> float:
    if prev is None:
        return x
    return (1.0 - alpha) * prev + alpha * x

def _rate_limit(prev: float | None, x: float, max_delta: float) -> float:
    if prev is None:
        return x
    return prev + _clamp(x - prev, -max_delta, +max_delta)

def _lerp_pts(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(_clamp(t, 0.0, 1.0))
    return a + (b - a) * t

def _yaw_target_pts_720(yaw_abs: float) -> np.ndarray:
    """Return 3pts for absolute yaw in degrees using piecewise linear interpolation."""
    y = float(_clamp(yaw_abs, -30.0, 30.0))
    if y < 0.0:
        if y >= -15.0:
            # front -> left15
            t = (-y) / 15.0
            return _lerp_pts(_CONT_LMK_720["front"], _CONT_LMK_720["left15"], t)
        else:
            # left15 -> left30
            t = (-y - 15.0) / 15.0
            return _lerp_pts(_CONT_LMK_720["left15"], _CONT_LMK_720["left30"], t)
    else:
        if y <= 15.0:
            # front -> right15
            t = y / 15.0
            return _lerp_pts(_CONT_LMK_720["front"], _CONT_LMK_720["right15"], t)
        else:
            # right15 -> right30
            t = (y - 15.0) / 15.0
            return _lerp_pts(_CONT_LMK_720["right15"], _CONT_LMK_720["right30"], t)

def _pitch_target_pts_720(pitch_abs: float) -> np.ndarray:
    """Return 3pts for absolute pitch in degrees using piecewise interpolation."""
    p = float(_clamp(pitch_abs, -15.0, 15.0))
    if p < 0.0:
        if p >= -7.5:
            # front -> up7.5
            t = (-p) / 7.5
            return _lerp_pts(_CONT_LMK_720["front"], _CONT_LMK_720["up7"], t)
        else:
            # up7.5 -> up15
            t = (-p - 7.5) / 7.5
            return _lerp_pts(_CONT_LMK_720["up7"], _CONT_LMK_720["up15"], t)
    else:
        if p <= 7.5:
            # front -> down7.5
            t = p / 7.5
            return _lerp_pts(_CONT_LMK_720["front"], _CONT_LMK_720["down7"], t)
        else:
            # down7.5 -> down15
            t = (p - 7.5) / 7.5
            return _lerp_pts(_CONT_LMK_720["down7"], _CONT_LMK_720["down15"], t)

def _similarity_mat_2pt(src2: np.ndarray, dst2: np.ndarray) -> np.ndarray:
    """
    2点 Similarity（回転+拡縮+平行移動）: 2x3 行列を返す。
    warp_similarity_2pt_to_canvas と同じ幾何だが、キャンバスではなく同サイズ変形で使う。
    """
    sx1, sy1 = float(src2[0, 0]), float(src2[0, 1])
    sx2, sy2 = float(src2[1, 0]), float(src2[1, 1])
    dx1, dy1 = float(dst2[0, 0]), float(dst2[0, 1])
    dx2, dy2 = float(dst2[1, 0]), float(dst2[1, 1])

    svx, svy = (sx2 - sx1), (sy2 - sy1)
    dvx, dvy = (dx2 - dx1), (dy2 - dy1)

    s_dist = math.hypot(svx, svy)
    d_dist = math.hypot(dvx, dvy)
    if s_dist < 1e-6 or d_dist < 1e-6:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    scale = d_dist / s_dist
    s_ang = math.atan2(svy, svx)
    d_ang = math.atan2(dvy, dvy)
    rot = d_ang - s_ang

    c = math.cos(rot) * scale
    s = math.sin(rot) * scale

    scx, scy = (sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5
    dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dy2) * 0.5

    tx = dcx - (c * scx - s * scy)
    ty = dcy - (s * scx + c * scy)

    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)

def _apply_roll(src_bgra: np.ndarray, roll_deg: float) -> np.ndarray:
    if abs(roll_deg) < 1e-6:
        return src_bgra
    h, w = src_bgra.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, roll_deg, 1.0)
    return cv2.warpAffine(
        src_bgra, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

def _apply_yaw_warp_x_only(
    src_bgra: np.ndarray,
    yaw_deg: float,
    anchor_x: float,
    eye_dist_px: float,
) -> np.ndarray:
    """yaw を scale に混ぜず、横方向 remap で表現する（BGRA, 透過前提）。
    - yaw は M0 側で clamp（±30°）
    - 強さは eye_dist_px ベース（スプライトサイズに依存しない）
    - α>0 の領域（=顔）だけを主に動かす（簡易マスク）
    """
    yaw_eff = _clamp(float(yaw_deg), -30.0, 30.0)
    if abs(yaw_eff) < 1e-6:
        return src_bgra

    h, w = src_bgra.shape[:2]
    if h <= 1 or w <= 1:
        return src_bgra

    a = yaw_eff / 30.0
    # yaw=±15 でも「見える」程度に動かす（sub-pixel を避ける）
    #   quintic(a^5) は ±15 で 0.03125 になり、ほぼ見えないため不採用
    #   cubic : ±15 でほぼ無効
    a3 = a * a * a
    mix = (0.30 * a) + (0.80 * a3)   # 線形成分で中角を持ち上げ、cubicで端を滑らかに

    # 最大横シフト（yaw30でも潰れない範囲）
    max_shift = 0.12 * float(max(1.0, eye_dist_px))
    shift = max_shift * mix

    # α簡易マスク（顔の外＝透明は動かさない）
    alpha = src_bgra[:, :, 3].astype(np.float32) / 255.0
    # エッジの薄いαをさらに抑制（緑縁防止）
    alpha_w = np.clip(alpha, 0.0, 1.0) ** 2.2

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)

    # x方向の重み：anchor_x 付近を強め、端は弱める
    denom_x = max(1.0, 0.90 * float(max(1.0, eye_dist_px)))
    x_norm = (xs - float(anchor_x)) / denom_x
    wx = np.clip(1.0 - np.abs(x_norm), 0.0, 1.0)  # 三角形

    # y方向の重み：顔中央〜下部を強める（目周りは抑える）
    y0 = float(h) * 0.58
    denom_y = max(1.0, float(h) * 0.22)
    wy = np.exp(-((ys - y0) / denom_y) ** 2).astype(np.float32)

    W = (wy[:, None] * wx[None, :]) * alpha_w

    map_x = (xs[None, :] + shift * W).astype(np.float32)
    map_y = np.repeat(ys[:, None], w, axis=1).astype(np.float32)

    out = cv2.remap(
        src_bgra,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return out

def _interp_targets_for_yaw_pitch(
    w: int,
    h: int,
    yaw_deg: float,
    pitch_deg: float,
    base_view: str = "front",
) -> tuple[np.ndarray, np.ndarray]:
    """
    yaw_deg/pitch_deg は「絶対角(deg)」。
    base_view のベースPNG（front/left30/right30/up15/down15）に対して,
    その近傍（隣接ビュー方向）へ補間した 3点(目2点+鼻) を返す。
    """
    # --- record raw yaw/pitch (for analysis) ---
    _record_yaw_pitch(yaw_deg, pitch_deg)

    # --- anisotropic scale: keep vertical size on yaw ---
    # yaw magnitude in degrees (0..30)
    yaw_mag = abs(float(yaw_deg))
    # weight: 0 at 0deg -> 1 at 30deg
    wy = min(1.0, yaw_mag / 30.0)
    # vertical scale is relaxed back to 1.0 as yaw increases
    est_scale = w / 720.0
    Sx = est_scale
    Sy = est_scale * (1.0 - 0.6 * wy) + 1.0 * (0.6 * wy)

    if base_view not in _CONT_LMK_720:
        base_view = "front"

    base = _CONT_LMK_720[base_view]

    # 目標は「絶対角 yaw/pitch に対応する3点」を多段補間で作る
    tgt_yaw = _yaw_target_pts_720(float(yaw_deg))
    tgt_pitch = _pitch_target_pts_720(float(pitch_deg))

    # 合成：front からの差分として加算（yaw/pitch の独立加算モデル）
    # 思想：tgt = base + (tgt_yaw-front) + (tgt_pitch-front)
    front = _CONT_LMK_720["front"]
    tgt = base + (tgt_yaw - front) + (tgt_pitch - front)

    tgt[:, 0] *= Sx
    tgt[:, 1] *= Sy

    src = base.copy()
    src[:, 0] *= Sx
    src[:, 1] *= Sy

    return src.astype(np.float32), tgt.astype(np.float32)

def pose_transform(
    src_bgra: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    base_view: str = "front",
) -> np.ndarray:
    """
    Continuous 版：
      (c) yaw/pitch を EMA + 速度制限 で平滑化
      (a) yaw/pitch から target（目2点＋鼻）を生成
      Similarity 2pt（目）で全体をワープ
      (b) Similarity後の「鼻のズレ」を、下半分だけ warpPerspective で追従させる
      roll は最後に回転
    """
    h, w = src_bgra.shape[:2]

    # (c) EMA + rate limit
    #   ※ 25fps前提の値。必要なら render 側設定に寄せる
    yaw_prev = _POSE_SMOOTH_STATE["yaw"]
    pitch_prev = _POSE_SMOOTH_STATE["pitch"]

    yaw_rl = _rate_limit(yaw_prev, float(yaw_deg), max_delta=2.0)      # deg/frame
    pitch_rl = _rate_limit(pitch_prev, float(pitch_deg), max_delta=0.8)  # deg/frame (Modified)

    yaw_sm = _ema(yaw_prev, yaw_rl, alpha=0.25)
    pitch_sm = _ema(pitch_prev, pitch_rl, alpha=0.25)

    _POSE_SMOOTH_STATE["yaw"] = yaw_sm
    _POSE_SMOOTH_STATE["pitch"] = pitch_sm

    # (a) build targets（ベースPNG(view)に合わせる）
    # yaw は scale に混ぜない（Similarityの目2点ターゲットには入れない）
    #  - yaw は view 切替＋後段の横warpで表現
    src3, tgt3 = _interp_targets_for_yaw_pitch(w, h, 0.0, pitch_sm, base_view=base_view)
    src_eye = src3[:2]
    tgt_eye = tgt3[:2]
    src_nose = src3[2]
    tgt_nose = tgt3[2]

    # Similarity 2pt
    M = _similarity_mat_2pt(src_eye, tgt_eye)
    warped = cv2.warpAffine(
        src_bgra, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # (b) nose-diff lower-half warpPerspective
    #  - similarity後の鼻位置を計算
    nx = float(M[0, 0] * src_nose[0] + M[0, 1] * src_nose[1] + M[0, 2])
    ny = float(M[1, 0] * src_nose[0] + M[1, 1] * src_nose[1] + M[1, 2])

    dx = float(tgt_nose[0] - nx)
    dy = float(tgt_nose[1] - ny)

    # 下半分だけ効かせる（目周りを固定したい）
    # pivot を鼻より少し下へ（輪郭の安定化）
    pivot_y = int(_clamp(
        max(ny, tgt_nose[1]) + 20.0, 0.0, float(h - 1)
    ))

    # 変形量を抑える（暴れ防止）
    # pitch 時は warp を弱める（輪郭 jitter 抑制）
    base_max_d = 30.0 * (w / 720.0)
    pitch_scale = min(1.0, abs(pitch_sm) / 15.0)
    max_d = base_max_d * (0.5 + 0.5 * pitch_scale)

    dx = _clamp(dx, -max_d, +max_d)
    dy = _clamp(dy, -max_d, +max_d)

    # top(固定) / bottom(移動) の4点で台形変形
    src_quad = np.array([
        [0.0, float(pivot_y)],
        [float(w - 1), float(pivot_y)],
        [float(w - 1), float(h - 1)],
        [0.0, float(h - 1)],
    ], dtype=np.float32)

    dst_quad = np.array([
        [0.0, float(pivot_y)],
        [float(w - 1), float(pivot_y)],
        [float(w - 1) + dx, float(h - 1) + dy],
        [0.0 + dx, float(h - 1) + dy],
    ], dtype=np.float32)

    P = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped2 = cv2.warpPerspective(
        warped, P, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # yaw（横方向warp）→ roll の順
    eye_dist = float(np.hypot(tgt_eye[1, 0] - tgt_eye[0, 0], tgt_eye[1, 1] - tgt_eye[0, 1]))
    anchor_x = float(tgt_nose[0])
    warped3 = _apply_yaw_warp_x_only(warped2, yaw_sm, anchor_x=anchor_x, eye_dist_px=eye_dist)

    # roll は最後
    return _apply_roll(warped3, roll_deg)


# -----------------------------
# メインレンダラー
# -----------------------------
def render_video(
    pose_timeline: list[dict[str, Any]],
    mouth_timeline: list[dict[str, Any]] | None,
    atlas_json_path: str,
    assets_dir: str,
    bg_video_path: str,
    out_mp4_path: str,
    fps: int = 25,
    target_h_ratio: float = 0.25,
    crossfade_frames: int = 0,
    per_frame_hook=None,
    affine_points_yaml_path: str | None = None,
) -> Dict[str, Any]:
    """
    pose_timeline: list of {t_ms, yaw/pitch/roll(deg), tx, ty, scale, ...}
    mouth_timeline: list of {t_ms, mouth, expression? ...}（無ければ closed 固定）
    """
    atlas_idx = load_atlas_index(atlas_json_path)
    view_rules = atlas_idx.get("view_rules", {})
    views_available = set(atlas_idx.get("views", {}).keys())

    affine_cfg = None
    if affine_points_yaml_path:
        affine_cfg = load_affine_points_yaml(affine_points_yaml_path)

    # [ADD] Similarity(2pt) のスケール差を抑えるため、front の両目間距離を基準に正規化する
    #  - 各 view の src_pts（両目中心）が手計測でブレても、レンダ側でスケールを揃えられる
    _ref_eye_dist_base = None
    try:
        if affine_cfg and isinstance(affine_cfg, dict):
            v = affine_cfg.get('views', {}).get('front', {})
            pts = v.get('src_pts')
            if isinstance(pts, list) and len(pts) >= 2:
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])
                _ref_eye_dist_base = float(math.hypot(x2 - x1, y2 - y1))
    except Exception:
        _ref_eye_dist_base = None

    cap = cv2.VideoCapture(bg_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(bg_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (width, height))

    # timeline index
    pose_idx = 0
    mouth_idx = 0

    fallback_frames = 0
    first_fallback_ms = None
    views_count: Dict[str, int] = {}

    prev_frame = None

    # [ADD] affine使用時の pose_scale を平滑化して “痙攣” を抑える
    prev_pose_scale_affine = None

    # helper: current pose/mouth by t_ms (hold-last)
    def _get_vals_at(t_ms: int, tl: list[dict[str, Any]], idx: int) -> tuple[dict[str, Any], int]:
        if not tl:
            return {}, idx
        while idx + 1 < len(tl) and int(tl[idx + 1].get("t_ms", 0)) <= t_ms:
            idx += 1
        return tl[idx], idx

    for i in range(total):
        ok, bgr = cap.read()
        if not ok:
            break

        step_ms = int(round(1000 / fps))
        t_ms = i * step_ms

        vals, pose_idx = _get_vals_at(t_ms, pose_timeline, pose_idx)

        # mouth/expression
        mvals = {}
        if mouth_timeline:
            mvals, mouth_idx = _get_vals_at(t_ms, mouth_timeline, mouth_idx)

        mouth = normalize_mouth_label(str(mvals.get("mouth", "closed")))
        expression = mvals.get("expression", None)

        # yaw/pitch/roll（deg）
        yaw = float(vals.get("yaw", vals.get("yaw_deg", 0.0)))
        pitch = float(vals.get("pitch", vals.get("pitch_deg", 0.0)))
        roll = float(vals.get("roll", vals.get("roll_deg", 0.0)))
        _record_yaw_pitch(yaw, pitch)  # record raw yaw/pitch even when affine3 path is used

        view = choose_view_from_yaw_pitch(yaw, pitch, view_rules, views_available)
        views = atlas_idx.get("views", {})
        views_count[view] = views_count.get(view, 0) + 1

        used_fallback = False
        src = None

        # 表情前提のベースPNGパスを解決
        base_path_rel, used_fallback_base = _resolve_base_sprite_path(atlas_idx, view, mouth)
        used_fallback = used_fallback or used_fallback_base

        if base_path_rel:
            # expression 用にパスを上書き
            expr_path_rel = _derive_expression_path(
                atlas_idx=atlas_idx,
                view=view,
                mouth=mouth,
                expression=expression,
                base_path_rel=base_path_rel,
            )

            # 実際の読み込み：まず expression 専用 → 無ければ normal(base) にフォールバック
            try:
                asset_path = os.path.join(assets_dir, expr_path_rel)
                src = _load_rgba(asset_path)
            except FileNotFoundError:
                try:
                    asset_path = os.path.join(assets_dir, base_path_rel)
                    src = _load_rgba(asset_path)
                    used_fallback = True  # 「表情」の意味ではフォールバック
                except FileNotFoundError:
                    src = None
        else:
            # baseもexpressionも読めなかった場合
            used_fallback = True

        frame = _ensure_bgra(bgr)

        if src is not None:
            # リサイズ（pose の scale を反映）
            # - target_h_ratio で「基準サイズ」を作り
            # - vals['scale']（例: 0.95〜1.05）で最終倍率を掛ける
            tgt_h_base = max(1, int(height * target_h_ratio))

            pose_scale = float(vals.get("scale", 1.0))
            # safety clamp（暴れ防止：必要なら調整）
            pose_scale = max(0.5, min(2.0, pose_scale))

            affine3 = vals.get("affine3")
            use_affine = bool(affine_cfg and affine3 and isinstance(affine3, dict) and "dst" in affine3)

            # [ADD] affine時は pose_scale をそのまま使うとサイズが暴れやすいので平滑化
            if use_affine:
                if prev_pose_scale_affine is None:
                    prev_pose_scale_affine = pose_scale
                # 1フレームあたり±2%制限
                max_ratio = 1.02
                pose_scale = max(prev_pose_scale_affine / max_ratio, min(prev_pose_scale_affine * max_ratio, pose_scale))
                # EMA
                alpha = 0.15
                pose_scale = (1.0 - alpha) * prev_pose_scale_affine + alpha * pose_scale
                prev_pose_scale_affine = pose_scale

            tgt_h = max(1, int(round(tgt_h_base * pose_scale)))

            # src→tgt のリサイズ倍率（ここは pose_scale とは別物）
            resize_ratio = tgt_h / src.shape[0]
            tgt_w = max(1, int(round(src.shape[1] * resize_ratio)))
            src_rs = cv2.resize(src, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)

            if use_affine:
                # IMPORTANT: view は choose_view_from_yaw の結果を使う（vals["view"] で上書きしない）
                # affine_points.yaml の src_pts は「元スプライト座標（リサイズ前）」前提。
                # src_rs はリサイズ後なので、同じ比率で src_pts もスケールして座標系を合わせる。
                
                src_pts = np.array(
                    affine_cfg["views"][view]["src_pts"],
                    dtype=np.float32
                )
                src_pts = src_pts * np.array([resize_ratio, resize_ratio], dtype=np.float32)
                dst_pts = np.array(affine3["dst"], dtype=np.float32)

                # ============================================================
                # [ADD][Phase C-2] 案A: src 正規化（view間の目間距離を統一）
                # ============================================================
                try:
                    front_pts = affine_cfg["views"]["front"]["src_pts"]
                    fx1, fy1 = front_pts[0]
                    fx2, fy2 = front_pts[1]
                    ref_eye_dist = float(
                        math.hypot(fx2 - fx1, fy2 - fy1)
                    ) * float(resize_ratio)

                    cur_eye_dist = float(
                        math.hypot(
                            src_pts[1, 0] - src_pts[0, 0],
                            src_pts[1, 1] - src_pts[0, 1],
                        )
                    )

                    if ref_eye_dist > 1e-6 and cur_eye_dist > 1e-6:
                        s = ref_eye_dist / cur_eye_dist
                        center = (src_pts[0] + src_pts[1]) * 0.5
                        src_pts = center + (src_pts - center) * s
                except Exception:
                    pass

                # --- Similarity (2点) : 全 view 共通（3点 affine は封印） ---
                src_eye_l = np.array(src_pts[0], dtype=np.float32)
                src_eye_r = np.array(src_pts[1], dtype=np.float32)
                dst_eye_l = np.array(dst_pts[0], dtype=np.float32)
                dst_eye_r = np.array(dst_pts[1], dtype=np.float32)

                overlay = warp_similarity_2pt_to_canvas(
                    sprite_bgra=src_rs,
                    canvas_w=width,
                    canvas_h=height,
                    src_pts=np.stack([src_eye_l, src_eye_r], axis=0),
                    dst_pts=np.stack([dst_eye_l, dst_eye_r], axis=0),
                )
                # ============================================================
                # [ADD][Phase C-2] affine パスにも yaw 横warp を適用
                #  - scroll6 等（affine3 有効）でも yaw を見た目に反映
                # ============================================================
                eye_dist = float(
                    math.hypot(
                        dst_eye_r[0] - dst_eye_l[0],
                        dst_eye_r[1] - dst_eye_l[1],
                    )
                )
                anchor_x = float(0.5 * (dst_eye_l[0] + dst_eye_r[0]))
                yaw_eff = float(max(-30.0, min(30.0, yaw)))  # clamp for warp stability
                overlay = _apply_yaw_warp_x_only(
                    overlay,
                    yaw_deg=yaw_eff,          # ← pose.json の yaw がここで初めて効く
                    anchor_x=anchor_x,
                    eye_dist_px=eye_dist,
                )
                _alpha_blend_full(frame, overlay)
            else:
                # ★ yaw/pitch/roll 変形をここで適用 ★
                # view（=ベースPNG）に合わせて continuous の基準点も切り替える
                src_rs = pose_transform(src_rs, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, base_view=view)

                # paste position (apply tx/ty if present)
                tx = float(vals.get("tx", 0.0))
                ty = float(vals.get("ty", 0.0))

                # tx/ty clamp（画面外に飛ばないための安全策）
                max_tx = width * 0.45
                max_ty = height * 0.45
                tx = max(-max_tx, min(max_tx, tx))
                ty = max(-max_ty, min(max_ty, ty))

                # NOTE:
                # - current assumption: tx/ty are pixels
                # - if normalized later: cx += int(tx * width), cy += int(ty * height)
                cx = int(round((width // 2) + tx))
                cy = int(round((height * 0.58) + ty))

                _alpha_paste(frame, src_rs, cx, cy)

        if used_fallback:
            fallback_frames += 1
            if first_fallback_ms is None:
                first_fallback_ms = t_ms

        # ★ ここで per_frame_hook に BGRA フレームを渡す（M3.5 合成など）★
        if per_frame_hook is not None:
            frame = per_frame_hook(frame, t_ms, i)

        # クロスフェード（旧版互換）
        if crossfade_frames > 0 and prev_frame is not None and i % (fps // 2 or 1) == 0:
            for k in range(crossfade_frames):
                a = (k + 1) / float(crossfade_frames + 1)
                mix = (prev_frame.astype(np.float32) * (1.0 - a) + frame.astype(np.float32) * a).astype(np.uint8)
                writer.write(mix[:, :, :3])
            prev_frame = frame.copy()
        else:
            writer.write(frame[:, :, :3])
            prev_frame = frame.copy()

    cap.release()
    writer.release()

    # --- dump stats at end ---
    _dump_yaw_pitch_stats()

    summary = {
        "fallback_frames": fallback_frames,
        "first_fallback_ms": first_fallback_ms,
        "views_count": views_count,
        "total_frames": int(total),
        "fps": int(fps),
        "out_mp4": out_mp4_path,
    }
    return summary


# ============================================
# M0 旧I/F互換ラッパー（bg_video不要版）
# ============================================
def render_video_legacy(
    out_mp4_path: str,
    width: int,
    height: int,
    fps: int,
    duration_s: int,
    crossfade_frames: int,
    merged_value_fn,
    *,
    assets_dir: str,
    atlas_json_rel: str,
    transform_cfg=None,
    affine_points_yaml_rel: str | None = None,
    debug_view_overlay: bool = False,
    **_ignored,
):
    """
    m0_runner.py が期待している旧 render_video I/F を満たす互換ラッパー。
    背景MP4は読まず、width/height の空キャンバスにスプライトを貼ってMP4生成する。
    """

    atlas_idx = load_atlas_index(atlas_json_rel)
    view_rules = atlas_idx.get("view_rules", {})
    views_available = set(atlas_idx.get("views", {}).keys())

    affine_cfg = None
    if affine_points_yaml_rel:
        affine_cfg = load_affine_points_yaml(os.path.join(os.path.dirname(out_mp4_path), affine_points_yaml_rel)) \
            if not os.path.isabs(affine_points_yaml_rel) else load_affine_points_yaml(affine_points_yaml_rel)

    # [ADD] Similarity(2pt) のスケール差を抑えるため、front の両目間距離を基準に正規化する
    _ref_eye_dist_base = None
    try:
        if affine_cfg and isinstance(affine_cfg, dict):
            v = affine_cfg.get('views', {}).get('front', {})
            pts = v.get('src_pts')
            if isinstance(pts, list) and len(pts) >= 2:
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])
                _ref_eye_dist_base = float(math.hypot(x2 - x1, y2 - y1))
    except Exception:
        _ref_eye_dist_base = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (width, height))

    total_frames = int(round(duration_s * fps))
    step_ms = int(round(1000 / fps))  # t_msズレ対策（fps=25なら40ms）

    fallback_frames = 0
    first_fallback_ms = None
    views_count: Dict[str, int] = {}

    # [ADD] affine3 使用率の集計
    affine_used_frames = 0
    legacy_used_frames = 0

    # [ADD] affine使用時の pose_scale を平滑化して “痙攣” を抑える
    prev_pose_scale_affine = None

    for i in range(total_frames):
        t_ms = i * step_ms
        vals = merged_value_fn(t_ms) or {}

        # --- mouth/expression ---
        mouth = normalize_mouth_label(str(vals.get("mouth", "closed")))
        expression = vals.get("expression", None)

        # --- pose (deg) ---
        yaw = float(vals.get("yaw", vals.get("yaw_deg", 0.0)))
        pitch = float(vals.get("pitch", vals.get("pitch_deg", 0.0)))
        roll = float(vals.get("roll", vals.get("roll_deg", 0.0)))
        _record_yaw_pitch(yaw, pitch)  # record raw yaw/pitch even when affine3 path is used

        view = str(vals.get("view")) if isinstance(vals.get("view"), str) else choose_view_from_yaw_pitch(yaw, pitch, view_rules, views_available)
        views_count[view] = views_count.get(view, 0) + 1

        # --- sprite path resolve ---
        used_fallback = False
        base_path_rel, used_fallback_base = _resolve_base_sprite_path(atlas_idx, view, mouth)
        used_fallback = used_fallback or used_fallback_base

        src = None
        if base_path_rel:
            expr_path_rel = _derive_expression_path(
                atlas_idx=atlas_idx,
                view=view,
                mouth=mouth,
                expression=expression,
                base_path_rel=base_path_rel,
            )

            # try expression path -> fallback to base
            try:
                src = _load_rgba(os.path.join(assets_dir, expr_path_rel))
            except FileNotFoundError:
                try:
                    src = _load_rgba(os.path.join(assets_dir, base_path_rel))
                    used_fallback = True  # 表情が無く normal に落ちた
                except FileNotFoundError:
                    src = None
                    used_fallback = True
        else:
            used_fallback = True

        # --- background canvas (opaque black BGRA) ---
        frame = np.zeros((height, width, 4), dtype=np.uint8)
        frame[:, :, 3] = 255

        if src is not None:
            # --- scale (pose scale) ---
            target_h_ratio = float((transform_cfg or {}).get("target_h_ratio", 0.25))
            tgt_h_base = max(1, int(height * target_h_ratio))

            pose_scale = float(vals.get("scale", 1.0))
            pose_scale = max(0.5, min(2.0, pose_scale))  # clamp

            affine3 = vals.get("affine3")
            use_affine = bool(affine_cfg and affine3 and isinstance(affine3, dict) and "dst" in affine3)

            # [ADD] affine時は pose_scale をそのまま使うとサイズが暴れやすいので平滑化
            if use_affine:
                if prev_pose_scale_affine is None:
                    prev_pose_scale_affine = pose_scale
                # 1フレームあたり±2%制限
                max_ratio = 1.02
                pose_scale = max(prev_pose_scale_affine / max_ratio, min(prev_pose_scale_affine * max_ratio, pose_scale))
                # EMA
                alpha = 0.15
                pose_scale = (1.0 - alpha) * prev_pose_scale_affine + alpha * pose_scale
                prev_pose_scale_affine = pose_scale

            tgt_h = max(1, int(round(tgt_h_base * pose_scale)))

            resize_ratio = tgt_h / src.shape[0]
            tgt_w = max(1, int(round(src.shape[1] * resize_ratio)))
            src_rs = cv2.resize(src, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)

            if use_affine:
                # IMPORTANT: view は choose_view_from_yaw の結果を使う（vals["view"] で上書きしない）
                src_pts = np.array(
                    affine_cfg["views"][view]["src_pts"],
                    dtype=np.float32
                )
                src_pts = src_pts * np.array([resize_ratio, resize_ratio], dtype=np.float32)
                dst_pts = np.array(affine3["dst"], dtype=np.float32)

                # ============================================================
                # [ADD][Phase C-2] 案A: src 正規化（view間の目間距離を統一）
                # ============================================================
                try:
                    front_pts = affine_cfg["views"]["front"]["src_pts"]
                    fx1, fy1 = front_pts[0]
                    fx2, fy2 = front_pts[1]
                    ref_eye_dist = float(
                        math.hypot(fx2 - fx1, fy2 - fy1)
                    ) * float(resize_ratio)

                    cur_eye_dist = float(
                        math.hypot(
                            src_pts[1, 0] - src_pts[0, 0],
                            src_pts[1, 1] - src_pts[0, 1],
                        )
                    )

                    if ref_eye_dist > 1e-6 and cur_eye_dist > 1e-6:
                        s = ref_eye_dist / cur_eye_dist
                        center = (src_pts[0] + src_pts[1]) * 0.5
                        src_pts = center + (src_pts - center) * s
                except Exception:
                    pass

                # --- Similarity (2点) : 全 view 共通（3点 affine は封印） ---
                src_eye_l = np.array(src_pts[0], dtype=np.float32)
                src_eye_r = np.array(src_pts[1], dtype=np.float32)
                dst_eye_l = np.array(dst_pts[0], dtype=np.float32)
                dst_eye_r = np.array(dst_pts[1], dtype=np.float32)

                overlay = warp_similarity_2pt_to_canvas(
                    sprite_bgra=src_rs,
                    canvas_w=width,
                    canvas_h=height,
                    src_pts=np.stack([src_eye_l, src_eye_r], axis=0),
                    dst_pts=np.stack([dst_eye_l, dst_eye_r], axis=0),
                )
                # ============================================================
                # [ADD][Phase C-2] affine パスにも yaw 横warp を適用
                #  - scroll6 等（affine3 有効）でも yaw を見た目に反映
                # ============================================================
                eye_dist = float(
                    math.hypot(
                        dst_eye_r[0] - dst_eye_l[0],
                        dst_eye_r[1] - dst_eye_l[1],
                    )
                )
                anchor_x = float(0.5 * (dst_eye_l[0] + dst_eye_r[0]))
                yaw_eff = float(max(-30.0, min(30.0, yaw)))  # clamp for warp stability
                overlay = _apply_yaw_warp_x_only(
                    overlay,
                    yaw_deg=yaw_eff,          # ← pose.json の yaw がここで初めて効く
                    anchor_x=anchor_x,
                    eye_dist_px=eye_dist,
                )
                _alpha_blend_full(frame, overlay)

                # [ADD] affine 使用フレームをカウント
                affine_used_frames += 1
            else:
                # --- old path: roll only + paste by tx/ty ---
                # [ADD] legacy 使用フレームをカウント
                legacy_used_frames += 1

                yaw = float(vals.get("yaw", vals.get("yaw_deg", 0.0)))
                pitch = float(vals.get("pitch", vals.get("pitch_deg", 0.0)))
                roll = float(vals.get("roll", vals.get("roll_deg", 0.0)))

                # view（=ベースPNG）に合わせて continuous の基準点も切り替える
                src_tf = pose_transform(src_rs, yaw, pitch, roll, base_view=view)

                cx = int(round(width * 0.5 + float(vals.get("tx", 0.0))))
                cy = int(round(height * 0.5 + float(vals.get("ty", 0.0))))
                _alpha_paste(frame, src_tf, cx, cy)

        if used_fallback:
            fallback_frames += 1
            if first_fallback_ms is None:
                first_fallback_ms = t_ms

        # write BGR
        writer.write(frame[:, :, :3])

    writer.release()

    # --- dump stats at end ---
    _dump_yaw_pitch_stats()

    summary = {
        "fallback_frames": fallback_frames,
        "first_fallback_ms": first_fallback_ms,
        "views_count": views_count,
        "total_frames": int(total_frames),
        "fps": int(fps),
        "out_mp4": out_mp4_path,

        # [ADD] affine3 / legacy 使用率ログ
        "affine_used_frames": int(affine_used_frames),
        "legacy_used_frames": int(legacy_used_frames),
    }
    return summary