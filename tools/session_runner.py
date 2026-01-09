#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
session_runner.py (FINAL: assets_dir fixed for sprites, inputs/audio absolute paths)

- Reads session.json (repo-relative references)
- Derives duration_s from session_audio_ms (ceil)
- Generates session_render.yaml (patched from base_config)
  - io.assets_dir fixed to tests/assets_min (sprite base)
  - atlas.atlas_json fixed to atlas.min.json (assets_dir base)
  - inputs.pose/mouth/expression and audio.wav_path are ABSOLUTE paths
    -> avoids any assets_dir prefix path confusion forever
- Runs M0 via subprocess: python vendor/src/m0_runner.py --config <patched_yaml_rel>
- Writes:
  out/<session_id>/session_runtime_resolved.json
  out/<session_id>/session_render.yaml
  out/<session_id>/run.session.log.json
  out/<session_id>/m0.stdout.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ---------------- YAML helpers ----------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML が必要です。未導入なら `pip install pyyaml` を実行してください。") from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML が必要です。未導入なら `pip install pyyaml` を実行してください。") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


# ---------------- JSON helpers ----------------
def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be dict: {path}")
    return data


def _dump_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------- misc helpers ----------------
def _wav_duration_ms(wav_path: Path) -> Optional[int]:
    if not wav_path.exists():
        return None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            ms = int(round(frames * 1000.0 / rate))
            return ms
    except Exception:
        return None


def _deep_get(d: Dict[str, Any], keys: Tuple[str, ...], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _deep_set(d: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    cur: Any = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_rendered_frames(stdout_text: str) -> Optional[int]:
    import re

    m = re.search(r"rendered_frames\s*=\s*(\d+)", stdout_text)
    if m:
        return int(m.group(1))

    m = re.search(r'"rendered_frames"\s*:\s*(\d+)', stdout_text)
    if m:
        return int(m.group(1))

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="Path to session.json (repo-relative or absolute)")
    ap.add_argument("--base_config", default="configs/smoke_pose_improved.yaml", help="Base YAML to clone and patch")
    ap.add_argument("--repo_root", default=".", help="Repo root (default: current dir)")
    ap.add_argument("--out_root", default="out/session_runs", help="Where to write session outputs (repo-relative)")
    ap.add_argument("--fps", type=int, default=None, help="Override fps (otherwise read from base_config)")
    ap.add_argument("--dry_run", action="store_true", help="Generate patched YAML but do not run M0")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    session_path = (repo_root / args.session).resolve() if not Path(args.session).is_absolute() else Path(args.session).resolve()
    base_cfg_path = (repo_root / args.base_config).resolve() if not Path(args.base_config).is_absolute() else Path(args.base_config).resolve()
    out_root = (repo_root / args.out_root).resolve()

    if not session_path.exists():
        print(f"[session_runner][ERROR] session not found: {session_path}", file=sys.stderr)
        return 2
    if not base_cfg_path.exists():
        print(f"[session_runner][ERROR] base_config not found: {base_cfg_path}", file=sys.stderr)
        return 2

    session = _load_json(session_path)

    schema_version = session.get("schema_version", "unknown")
    session_id = session.get("session_id", session_path.stem)

    session_audio_ms = session.get("session_audio_ms", None)
    if not isinstance(session_audio_ms, int):
        print("[session_runner][ERROR] session_audio_ms must be int", file=sys.stderr)
        return 2

    pose_timeline = session.get("pose_timeline")
    mouth_timeline = session.get("mouth_timeline")
    expression_timeline = session.get("expression_timeline")
    audio = session.get("audio")

    for k, v in [
        ("pose_timeline", pose_timeline),
        ("mouth_timeline", mouth_timeline),
        ("expression_timeline", expression_timeline),
        ("audio", audio),
    ]:
        if not isinstance(v, str) or not v:
            print(f"[session_runner][ERROR] session['{k}'] must be non-empty string", file=sys.stderr)
            return 2

    # Resolve to ABS paths for validation and for patched YAML (FINAL design)
    pose_abs = (repo_root / pose_timeline).resolve()
    mouth_abs = (repo_root / mouth_timeline).resolve()
    expr_abs = (repo_root / expression_timeline).resolve()
    audio_abs = (repo_root / audio).resolve()

    missing = [p for p in [pose_abs, mouth_abs, expr_abs, audio_abs] if not p.exists()]
    if missing:
        print("[session_runner][ERROR] missing referenced files:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 2

    # Derive duration from audio_ms (source of truth)
    duration_s = int(math.ceil(session_audio_ms / 1000.0))

    # Load base config
    cfg = _load_yaml(base_cfg_path)

    fps = args.fps if args.fps is not None else _deep_get(cfg, ("video", "fps"), None)
    if fps is None:
        fps = 25
    if not isinstance(fps, int) or fps <= 0:
        print("[session_runner][ERROR] invalid fps", file=sys.stderr)
        return 2

    target_frames = int(duration_s * fps)

    # Patch duration
    _deep_set(cfg, ("video", "duration_s"), float(duration_s))
    _deep_set(cfg, ("video", "fps"), int(fps))

    # FINAL: assets are served from tests/assets_min
    _deep_set(cfg, ("io", "assets_dir"), "tests/assets_min")
    _deep_set(cfg, ("atlas", "atlas_json"), "atlas.min.json")

    # [ADD] affine_points_yaml_rel を ABS 化（out_dir 基準 join の事故を防ぐ）
    affine_rel = _deep_get(cfg, ("render", "affine_points_yaml_rel"), default=None)
    if isinstance(affine_rel, str) and affine_rel:
        ap = Path(affine_rel)
        affine_abs = ap.resolve() if ap.is_absolute() else (repo_root / ap).resolve()
        _deep_set(cfg, ("render", "affine_points_yaml_rel"), str(affine_abs))
    else:
        # base_config に無い場合でも、標準配置があれば入れておく（任意だが混乱防止に有効）
        default_ap = (repo_root / "configs" / "affine_points.yaml")
        if default_ap.exists():
            _deep_set(cfg, ("render", "affine_points_yaml_rel"), str(default_ap.resolve()))

    # FINAL: timelines + audio are ABSOLUTE (avoid any prefix/join ambiguity)
    _deep_set(cfg, ("inputs", "pose_timeline"), str(pose_abs))
    _deep_set(cfg, ("inputs", "mouth_timeline"), str(mouth_abs))
    _deep_set(cfg, ("inputs", "expression_timeline"), str(expr_abs))
    _deep_set(cfg, ("audio", "wav_path"), str(audio_abs))

    # Remove inputs.audio if it exists (m0_runner does not need it; it can confuse)
    if isinstance(cfg.get("inputs"), dict) and "audio" in cfg["inputs"]:
        del cfg["inputs"]["audio"]

    # Output dir
    sess_out_dir = out_root / session_id
    sess_out_dir.mkdir(parents=True, exist_ok=True)

    # Emit resolved runtime (keeps original repo-relative refs + derived info)
    resolved = {
        "schema_version": schema_version,
        "session_id": session_id,
        "session_audio_ms": session_audio_ms,
        "derived": {
            "duration_s": duration_s,
            "fps": fps,
            "target_frames": target_frames,
        },
        "refs_repo_relative": {
            "pose_timeline": pose_timeline,
            "mouth_timeline": mouth_timeline,
            "expression_timeline": expression_timeline,
            "audio": audio,
        },
        "refs_abs": {
            "pose_timeline": str(pose_abs),
            "mouth_timeline": str(mouth_abs),
            "expression_timeline": str(expr_abs),
            "audio": str(audio_abs),
        },
        "paths_abs": {
            "repo_root": str(repo_root),
            "session_json": str(session_path),
            "base_config": str(base_cfg_path),
        },
        "generated_at_unix": int(time.time()),
    }
    _dump_json(resolved, sess_out_dir / "session_runtime_resolved.json")

    # Emit patched YAML
    patched_yaml_path = sess_out_dir / "session_render.yaml"
    _dump_yaml(cfg, patched_yaml_path)

    # WAV duration check (optional)
    wav_ms = _wav_duration_ms(audio_abs)
    wav_delta_ms = (wav_ms - session_audio_ms) if (wav_ms is not None) else None

    # Runner log scaffold
    runlog: Dict[str, Any] = {
        "session_id": session_id,
        "session_audio_ms": session_audio_ms,
        "derived_duration_s": duration_s,
        "fps": fps,
        "target_frames": target_frames,
        "wav_ms": wav_ms,
        "wav_delta_ms": wav_delta_ms,
        "m0": {
            "cmd": None,
            "returncode": None,
            "rendered_frames": None,
        },
        "notes": [],
    }

    if wav_delta_ms is not None and abs(wav_delta_ms) > 40:
        runlog["notes"].append(f"[WARN] wav_ms differs from session_audio_ms by {wav_delta_ms}ms (step_ms=40 assumed)")

    if args.dry_run:
        runlog["notes"].append("[DRY_RUN] M0 was not executed.")
        _dump_json(runlog, sess_out_dir / "run.session.log.json")
        print(f"[session_runner] DRY_RUN OK: {patched_yaml_path}")
        return 0

    # Run M0 via subprocess (your proven pattern)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    cmd = [
        sys.executable,
        "vendor/src/m0_runner.py",
        "--config",
        str(patched_yaml_path.relative_to(repo_root)),
    ]
    runlog["m0"]["cmd"] = " ".join(cmd)

    print("[session_runner] repo_root =", repo_root)
    print("[session_runner] session_id =", session_id)
    print("[session_runner] session_audio_ms =", session_audio_ms)
    print("[session_runner] derived_duration_s =", duration_s)
    print("[session_runner] target_frames =", target_frames)
    print("[session_runner] running:", " ".join(cmd))

    p = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    runlog["m0"]["returncode"] = p.returncode
    runlog["m0"]["rendered_frames"] = _parse_rendered_frames(p.stdout)

    # Save stdout
    (sess_out_dir / "m0.stdout.txt").write_text(p.stdout, encoding="utf-8")

    if runlog["m0"]["rendered_frames"] is not None and runlog["m0"]["rendered_frames"] != target_frames:
        runlog["notes"].append(
            f"[WARN] rendered_frames({runlog['m0']['rendered_frames']}) != target_frames({target_frames})"
        )

    if p.returncode != 0:
        runlog["notes"].append("[ERROR] m0_runner failed. See m0.stdout.txt")

    _dump_json(runlog, sess_out_dir / "run.session.log.json")

    if p.returncode == 0:
        print(f"[session_runner] OK: {sess_out_dir}")
        return 0
    else:
        print(f"[session_runner] ERROR: returncode={p.returncode}. See {sess_out_dir/'m0.stdout.txt'}", file=sys.stderr)
        return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
