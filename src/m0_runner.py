from __future__ import annotations
import argparse, os, json, shutil, time, subprocess
from typing import Dict, Any
from pathlib import Path

import yaml
import csv
import wave

# -----------------------------
# 基本ユーティリティ
# -----------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(txt) or {}
    # JSONも許容
    return json.loads(txt)

def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _safe_get_float(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def _infer_last_t_ms_from_frames(frames: Any) -> int | None:
    """
    frames/timeline が list の場合に、末尾付近から t_ms を探して返す。
    dict wrapper の中身でもOK。
    """
    if not isinstance(frames, list) or not frames:
        return None
    # 末尾から最大10個だけ見て t_ms を拾う（変な終端でもなるべく拾う）
    for i in range(1, min(10, len(frames)) + 1):
        fr = frames[-i]
        if isinstance(fr, dict):
            if "t_ms" in fr:
                try:
                    return int(fr["t_ms"])
                except Exception:
                    return None
    return None

def _infer_last_t_ms_from_raw(raw: Any) -> int | None:
    """
    raw が dict wrapper / list どちらでも last_t_ms を推定。
    """
    if isinstance(raw, list):
        return _infer_last_t_ms_from_frames(raw)
    if isinstance(raw, dict):
        # よくある wrapper キー候補
        for k in ("frames", "timeline"):
            if k in raw:
                return _infer_last_t_ms_from_frames(raw.get(k))
    return None

def _count_frames_from_raw(raw: Any) -> int | None:
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, dict):
        for k in ("frames", "timeline"):
            if k in raw and isinstance(raw.get(k), list):
                return len(raw.get(k))
    return None

def _log_meta_scale_mode(tag: str, raw: Any):
    """
    meta.scale_mode をログに出す（M0はmeta必須にしないが、混在事故を避ける）
    """
    if not isinstance(raw, dict):
        return
    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        return
    sm = meta.get("scale_mode")
    if sm is None:
        return
    print(f"[{tag}] meta.scale_mode = {sm}")
    if str(sm).lower() != "absolute":
        print(f"[{tag}][WARN] scale_mode != absolute (契約上は absolute 推奨)")

# -----------------------------
# エイリアス（フォルダ）作成
# -----------------------------
def _mk_tmp_assets_with_alias(src_assets: Path, exp_dir: Path, alias: Dict[str,str]) -> Path:
    """
    assets_dir配下に view名の別名（エイリアス）を用意する。
    例: {"left30": "down15", "right30": "up15"} → tmp_assets/left30 -> tmp_assets/down15
    symlinkが使えない環境ではコピーにフォールバック。
    """
    tmp = exp_dir / "tmp_assets"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src_assets, tmp, dirs_exist_ok=True)

    def link_or_copy(src: Path, dst: Path):
        if dst.exists():  # 既にあるなら触らない
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
        except Exception:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    # 片方向（dst→src）で作る
    for dst_name, src_name in alias.items():
        src_path = tmp / src_name
        dst_path = tmp / dst_name
        if src_path.exists():
            link_or_copy(src_path, dst_path)

    return tmp

# -----------------------------
# atlas 深度置換
# -----------------------------
def _json_deep_replace(obj, replace_map: Dict[str, str]):
    from collections.abc import Mapping
    if isinstance(obj, Mapping):
        return {k: _json_deep_replace(v, replace_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_deep_replace(v, replace_map) for v in obj]
    if isinstance(obj, str):
        s = obj
        for old, new in replace_map.items():
            s = s.replace(old, new)
        return s
    return obj

def _rewrite_atlas_for_alias(base_atlas_path: Path, tmp_assets_dir: Path, view_alias: Dict[str, str]) -> Path:
    """
    atlas.min.json 内の全パス文字列に対し、view_aliasに基づく置換を施した
    「別名対応版atlas」を生成して返す。
    - 例: {"left30":"down15"} → "/left30/" を "/down15/" に
    """
    pairs = {}
    for dst, src in view_alias.items():
        pairs[f"/{dst}/"] = f"/{src}/"
        pairs[f"{dst}/"]  = f"{src}/"

    text = base_atlas_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        new_data = _json_deep_replace(data, pairs)
        out = tmp_assets_dir / "atlas.alias.json"
        out.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception:
        for old, new in pairs.items():
            text = text.replace(old, new)
        out = tmp_assets_dir / "atlas.alias.json"
        out.write_text(text, encoding="utf-8")
        return out

# -----------------------------
# Timeline／レンダラー読み込み
# -----------------------------
def _import_timeline_and_render():
    """
    Timeline / render_video の import を環境に応じて切り替える。

    優先順（落ちにくい順）:
    - A: プロジェクト直下に src/ がある（python -m src.m0_runner など）
    - B: repo root を PYTHONPATH に入れて vendor.src.* で引ける
    - C: vendor/src/ ディレクトリを直接叩く（同じディレクトリからローカルimport）
    """
    try:
        from src.timeline import Timeline
        from src.render_core import render_video
        return Timeline, render_video
    except ModuleNotFoundError:
        pass

    try:
        from vendor.src.timeline import Timeline
        from vendor.src.render_core import render_video_legacy as render_video
        return Timeline, render_video
    except ModuleNotFoundError:
        pass

    # 最後の砦（vendor/src を current dir として実行したとき）
    from timeline import Timeline
    from render_core import render_video_legacy as render_video
    return Timeline, render_video

# -----------------------------
# 値マージ・軸適用
# -----------------------------
def _build_merged_value_fn(mouth_tl, pose_tl, expr_tl,
                           value_key: str, thr_front: float, map_deg: float, mode: str = "1d"):
    """
    value_key が yaw 以外（pitch/roll）の場合、擬似yaw（±map_deg or 0）を注入して返す。

    ※M3' からは mouth_id(0..5) が来る想定なので、
      ここで mouth ラベル("close/a/i/u/e/o") に変換して M0 に渡す。
    """
    MOUTH_LABELS = ["close", "a", "i", "u", "e", "o"]

    def merged_value(t_ms: int) -> Dict[str, Any]:
        vals: Dict[str, Any] = {}
        vals.update(mouth_tl.value_at(t_ms))
        vals.update(pose_tl.value_at(t_ms))
        vals.update(expr_tl.value_at(t_ms))

        # --- mouth_id -> mouth ラベル変換 ---
        if "mouth_id" in vals and "mouth" not in vals:
            try:
                mid = int(vals["mouth_id"])
            except Exception:
                mid = 0
            if 0 <= mid < len(MOUTH_LABELS):
                vals["mouth"] = MOUTH_LABELS[mid]
            else:
                vals["mouth"] = "close"

        # --- ここから下は従来どおり（擬似yaw注入） ---
        if mode != "1d" or value_key == "yaw":
            return vals

        v = None
        if value_key == "pitch":
            v = _safe_get_float(vals, "pitch_deg", "pitch", default=None)
        elif value_key == "roll":
            v = _safe_get_float(vals, "roll_deg", "roll", default=None)

        if v is None:
            pseudo = 0.0
        else:
            pseudo = 0.0 if abs(v) <= thr_front else (map_deg if v > 0 else -map_deg)

        vals["yaw"] = pseudo
        vals["yaw_deg"] = pseudo
        return vals

    return merged_value

# -----------------------------
# audio パス解決ヘルパ
# -----------------------------
def _resolve_audio_path(audio_name: str | None,
                        cfg: Dict[str, Any],
                        assets_dir: Path,
                        mouth_path: Path,
                        config_path: Path) -> Path | None:
    if not audio_name:
        return None

    cand = Path(audio_name)
    if cand.is_absolute() and cand.exists():
        return cand

    # repo_root/tests/audio を探索に追加（標準配置）
    # vendor/src/m0_runner.py から repo root を推定
    repo_root = Path(__file__).resolve().parents[2]
    tests_audio_dir = repo_root / "tests" / "audio"

    for base in [config_path.parent, mouth_path.parent, assets_dir, tests_audio_dir]:
        p = base / audio_name
        if p.exists():
            return p

    return None

# -----------------------------
# audio 長さ計測（wav）
# -----------------------------
def _get_wav_duration_ms(wav_path: Path) -> int | None:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return int(round(frames * 1000 / rate))
    except Exception:
        return None

# -----------------------------
# メイン
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)          # JSON or YAML
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    # 設定ロード
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    for o in args.override:
        cfg = deep_update(cfg, load_yaml(o))

    assets_dir = Path(cfg["io"]["assets_dir"]).resolve()
    out_dir    = Path(cfg["io"]["out_dir"]).resolve()
    exp_name   = cfg["io"]["exp_name"]

    width       = int(cfg["video"]["width"])
    height      = int(cfg["video"]["height"])
    fps         = int(cfg["video"]["fps"])
    duration_s  = float(cfg["video"]["duration_s"])
    crossfade   = int(cfg["render"]["crossfade_frames"])

    # [NOTE] 3点アフィン導線は無効化（2D view 検証フェーズでは使用しない）
    affine_points_yaml_rel = None

    # debug: view overlay（目視確認用）
    debug_view_overlay = bool((cfg.get("render", {}) or {}).get("debug_view_overlay", False))

    # ---- 尺・フレーム数の「正」ログ（ここが最重要）----
    step_ms = int(round(1000.0 / fps))
    target_ms = int(round(duration_s * 1000.0))
    target_frames = int(round(duration_s * fps))
    print("[len]")
    print(f"  fps={fps} step_ms={step_ms}")
    print(f"  duration_s={duration_s} target_ms={target_ms} target_frames={target_frames}")

    # メトリクス・切替設定（Option A：alias＋atlas書き換え）
    mconf       = cfg.get("metrics", {}) or {}
    metrics_mode = str(mconf.get("mode", "1d")).lower()
    value_key   = str(mconf.get("value_key", "yaw"))     # "yaw" / "pitch" / "roll"
    thr_front   = float(mconf.get("thr_front", 16.0))    # ±閾値[deg]
    zero_label  = str(mconf.get("zero_label", "front"))  # ラベル（ログ用）
    neg_label   = str(mconf.get("neg_label",  "left30"))
    pos_label   = str(mconf.get("pos_label",  "right30"))
    map_deg     = float(mconf.get("map_deg", 30.0))      # 擬似yawの±度数
    view_alias  = dict(mconf.get("view_alias", {}))      # {"left30":"down15","right30":"up15",...}

    # transform 設定（render_core へ透過）
    transform_cfg = cfg.get("transform")

    # pitch/roll で alias 未指定なら既定補完
    if value_key != "yaw" and not view_alias:
        if value_key == "pitch":
            view_alias = {"left30": "down15", "right30": "up15", "front": "front"}
        else:
            view_alias = {"front": "front", "left30": "left30", "right30": "right30"}

    # パス解決ヘルパ（assets_dir からの相対パスを絶対化）
    def _abs_assets(p: str) -> str:
        return p if os.path.isabs(p) else str(assets_dir / p)

    # タイムライン読み込み
    Timeline, render_video = _import_timeline_and_render()
    inputs = cfg.get("inputs", {})

    # ---- 追加ログ用に raw を保持 ----
    raw_mouth = None
    raw_pose = None
    raw_expr = None

    mouth_path = None
    mouth_tl = Timeline([])  # デフォルト空タイムライン
    audio_name_from_mouth = None

    if "mouth_timeline" in inputs:
        mouth_path = Path(_abs_assets(inputs["mouth_timeline"]))
        raw_txt = mouth_path.read_text(encoding="utf-8")
        raw = json.loads(raw_txt)
        raw_mouth = raw

        if isinstance(raw, dict):
            audio_name_from_mouth = raw.get("audio")

        if isinstance(raw, dict) and "frames" in raw:
            frames = raw.get("frames") or []
            tmp_path = mouth_path.parent / (mouth_path.stem + ".frames_only.tmp.json")
            tmp_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
            mouth_tl = Timeline.load_json(str(tmp_path))
        elif isinstance(raw, list):
            mouth_tl = Timeline.load_json(str(mouth_path))
        else:
            mouth_tl = Timeline([])

    # pose_timeline 読み込み（wrapper両対応）
    if "pose_timeline" in inputs:
        pose_path = Path(_abs_assets(inputs["pose_timeline"]))
        raw_txt = pose_path.read_text(encoding="utf-8")
        raw = json.loads(raw_txt)
        raw_pose = raw

        # meta.scale_mode ログ（混在事故予防）
        _log_meta_scale_mode("pose", raw)

        if isinstance(raw, dict) and "timeline" in raw:
            frames = raw.get("timeline") or []
            tmp_path = pose_path.parent / (pose_path.stem + ".timeline_only.tmp.json")
            tmp_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
            pose_tl = Timeline.load_json(str(tmp_path))

        elif isinstance(raw, dict) and "frames" in raw:
            frames = raw.get("frames") or []
            tmp_path = pose_path.parent / (pose_path.stem + ".frames_only.tmp.json")
            tmp_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
            pose_tl = Timeline.load_json(str(tmp_path))

        elif isinstance(raw, list):
            pose_tl = Timeline.load_json(str(pose_path))
        else:
            pose_tl = Timeline([])
    else:
        pose_tl = Timeline([])

    # expression_timeline 読み込み（dict.timeline もOK）
    if "expression_timeline" in inputs:
        expr_path = Path(_abs_assets(inputs["expression_timeline"]))
        raw_txt = expr_path.read_text(encoding="utf-8")
        raw = json.loads(raw_txt)
        raw_expr = raw

        if isinstance(raw, dict) and "timeline" in raw:
            frames = raw.get("timeline") or []
            tmp_path = expr_path.parent / (expr_path.stem + ".timeline_only.tmp.json")
            tmp_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
            expr_tl = Timeline.load_json(str(tmp_path))
        elif isinstance(raw, dict) and "frames" in raw:
            frames = raw.get("frames") or []
            tmp_path = expr_path.parent / (expr_path.stem + ".frames_only.tmp.json")
            tmp_path.write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
            expr_tl = Timeline.load_json(str(tmp_path))
        elif isinstance(raw, list):
            expr_tl = Timeline.load_json(str(expr_path))
        else:
            expr_tl = Timeline([])
    else:
        expr_tl = Timeline([])

    # ---- last_t_ms / counts ログ（ここが長尺バグの即死ログ）----
    pose_last_t_ms = _infer_last_t_ms_from_raw(raw_pose)
    mouth_last_t_ms = _infer_last_t_ms_from_raw(raw_mouth)
    expr_last_t_ms = _infer_last_t_ms_from_raw(raw_expr)

    pose_n = _count_frames_from_raw(raw_pose)
    mouth_n = _count_frames_from_raw(raw_mouth)
    expr_n = _count_frames_from_raw(raw_expr)

    print(f"  pose_last_t_ms={pose_last_t_ms} (n={pose_n})")
    print(f"  mouth_last_t_ms={mouth_last_t_ms} (n={mouth_n})")
    print(f"  expr_last_t_ms={expr_last_t_ms} (n={expr_n})")

    # audio 設定：config > mouth_timeline.json 内の "audio"
    audio_cfg = cfg.get("audio", {}) or {}
    audio_name_cfg = audio_cfg.get("wav_path")
    audio_name = audio_name_cfg or audio_name_from_mouth

    # audio_name が決まったら、必ず _resolve_audio_path() の戻り値で audio_path を代入してから参照する
    audio_path = _resolve_audio_path(
        audio_name,
        cfg,
        assets_dir=assets_dir,
        mouth_path=(mouth_path or cfg_path),
        config_path=cfg_path,
    ) if audio_name else None

    # audio_ms（wav実長）を計測（WARNのみ・挙動不変）
    audio_ms = None
    if audio_path and audio_path.exists():
        audio_ms = _get_wav_duration_ms(audio_path)

    # 出力先
    exp_dir  = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    raw_mp4  = exp_dir / "video_raw.mp4"
    final_mp4 = exp_dir / "demo.mp4"

    # assets の有効ディレクトリ（alias適用）
    use_assets_dir = assets_dir
    if value_key != "yaw" and view_alias:
        use_assets_dir = _mk_tmp_assets_with_alias(assets_dir, exp_dir, view_alias)

    # atlas の有効パス（alias適用で深度置換）
    atlas_json_rel = cfg.get("atlas", {}).get("atlas_json", None)
    atlas_json_for_render = atlas_json_rel
    if atlas_json_rel and (value_key != "yaw") and view_alias:
        base_atlas = Path(atlas_json_rel)
        if not base_atlas.is_absolute():
            base_atlas = use_assets_dir / atlas_json_rel
        if base_atlas.exists():
            atlas_json_for_render = str(_rewrite_atlas_for_alias(base_atlas, use_assets_dir, view_alias))

    # alias が無い通常ケースでも assets_dir を付ける
    if atlas_json_for_render:
        p = Path(atlas_json_for_render)
        if not p.is_absolute():
            atlas_json_for_render = str(use_assets_dir / atlas_json_for_render)

    # 値マージ関数（擬似yaw注入）
    merged_value = _build_merged_value_fn(
        mouth_tl, pose_tl, expr_tl,
        value_key=value_key, thr_front=thr_front, map_deg=map_deg,
        mode=metrics_mode,
    )

    # デバッグ: timeline CSV のダンプ
    debug_cfg = cfg.get("debug", {})
    dump_csv_rel = debug_cfg.get("dump_timeline_csv")

    if dump_csv_rel:
        debug_root = out_dir / exp_name
        debug_path = debug_root / dump_csv_rel
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = int(round(fps * duration_s))
        fieldnames = [
            "frame_index",
            "t_ms",
            "mouth",
            "mouth_id",
            "yaw",
            "yaw_deg",
            "pitch_deg",
            "roll_deg",
            "expression",
        ]

        with debug_path.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()

            for fi in range(total_frames):
                t_ms = int(round(fi * 1000 / fps))
                vals = merged_value(t_ms)

                writer.writerow({
                    "frame_index": fi,
                    "t_ms": t_ms,
                    "mouth": vals.get("mouth"),
                    "mouth_id": vals.get("mouth_id"),
                    "yaw": vals.get("yaw"),
                    "yaw_deg": vals.get("yaw_deg"),
                    "pitch_deg": vals.get("pitch_deg"),
                    "roll_deg": vals.get("roll_deg"),
                    "expression": vals.get("expression") or vals.get("exp") or vals.get("label"),
                })

    # レンダリング本体
    t0 = time.time()
    stats = render_video(
        str(raw_mp4),
        width, height, fps, duration_s, crossfade,
        merged_value,
        assets_dir=str(use_assets_dir),
        atlas_json_rel=atlas_json_for_render,
        transform_cfg=transform_cfg,
        debug_view_overlay=debug_view_overlay,
    )
    render_elapsed = round(time.time() - t0, 3)

    # stats から rendered_frames を拾ってログ（無ければ duration_s*fps を採用）
    rendered_frames = None
    if isinstance(stats, dict):
        rendered_frames = stats.get("rendered_frames") or stats.get("frames") or None
    if rendered_frames is None:
        rendered_frames = int(round(duration_s * fps))

    print("[len]")
    print(f"  rendered_frames={rendered_frames}")

    # audio_ms vs render duration をログ（WARNのみ・挙動不変）
    if audio_ms is not None:
        duration_ms = int(round(duration_s * 1000))
        delta_ms = audio_ms - duration_ms

        print("[audio]")
        print(f"  audio_path={audio_path}")
        print(f"  audio_ms={audio_ms} duration_ms={duration_ms} delta_ms={delta_ms}")

        if abs(delta_ms) > step_ms:
            print(
                "[audio][WARN] audio duration differs from render duration "
                f"(>|step_ms|={step_ms}ms). "
                "OK for now; session integration should use audio_ms as source of truth."
            )

    # audio mux（ffmpeg）
    mux_succeeded = False
    mux_error = None

    if audio_path and audio_path.exists():
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(raw_mp4),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(final_mp4),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            mux_succeeded = True
        except Exception as e:
            mux_error = str(e)

    if not mux_succeeded:
        try:
            shutil.move(str(raw_mp4), str(final_mp4))
        except Exception:
            pass

    total_elapsed = round(time.time() - t0, 3)

    # ログ
    run_log = {
        "out_mp4": str(final_mp4),
        "raw_mp4": str(raw_mp4),
        "fps": fps,
        "step_ms": step_ms,
        "duration_s": duration_s,
        "target_ms": target_ms,
        "target_frames": target_frames,
        "audio_ms": audio_ms,
        "duration_ms": int(round(duration_s * 1000)),
        "audio_delta_ms": (audio_ms - int(round(duration_s * 1000))) if audio_ms is not None else None,
        "rendered_frames": int(rendered_frames),
        "pose_last_t_ms": pose_last_t_ms,
        "mouth_last_t_ms": mouth_last_t_ms,
        "expr_last_t_ms": expr_last_t_ms,
        "pose_n": pose_n,
        "mouth_n": mouth_n,
        "expr_n": expr_n,
        "assets_dir": str(assets_dir),
        "assets_dir_effective": str(use_assets_dir),
        "exp_name": exp_name,
        "elapsed_s_render": render_elapsed,
        "elapsed_s_total": total_elapsed,
        "axis": value_key,
        "thr_front_deg": thr_front,
        "map_deg": map_deg,
        "labels": {"zero": zero_label, "neg": neg_label, "pos": pos_label},
        "view_alias": view_alias,
        "audio_name_cfg": audio_name_cfg,
        "audio_name_from_mouth": audio_name_from_mouth,
        "audio_path": str(audio_path) if audio_path else None,
        "audio_mux_succeeded": mux_succeeded,
        "audio_mux_error": mux_error,
        "atlas_json_for_render": atlas_json_for_render,
    }
    if stats:
        run_log.update(stats)

    (exp_dir / "run.log.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary.csv（簡易）
    summary_keys = [
        "exp_name", "duration_s", "elapsed_s_render", "elapsed_s_total",
        "target_frames", "rendered_frames", "pose_last_t_ms", "mouth_last_t_ms", "expr_last_t_ms",
        "fallback_frames", "first_fallback_ms"
    ]
    with (exp_dir / "summary.csv").open("w", encoding="utf-8") as f:
        f.write("key,value\n")
        for k in summary_keys:
            if k in run_log:
                f.write(f"{k},{run_log[k]}\n")
        views = run_log.get("views", {})
        for name, count in views.items():
            f.write(f"views_{name},{count}\n")

if __name__ == "__main__":
    main()