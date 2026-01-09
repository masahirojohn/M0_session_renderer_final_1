M0 Session Renderer (FG → M3.5 Pipeline)
概要

本リポジトリは、
セッション単位の pose / mouth / expression / audio を入力として、
M0 Renderer により FG(BGRA PNG連番) を生成し、
M3.5 で背景動画と合成するための基盤です。

実運用では FG連番が唯一の正

MP4生成は デバッグ用途（任意）

全体フロー
session.json
   ↓
session_runner.py
   ↓
M0 Renderer
   ↓
FG (BGRA PNG, %08d, 0-start)
   ↓
M3.5 Compositor
   ↓
Final MP4

クイックスタート
export PYTHONPATH=$PWD

python tools/session_runner.py \
  --session sessions/sess_real_01.session.json \
  --base_config configs/smoke_pose_improved.yaml \
  --out_root out/session_runs


生成物：

out/fg/in/fg/00000000.png ...

BGRA / 欠番なし / canvas一致

FG連番の仕様（重要）

形式：PNG

チャンネル：BGRA（4ch）

命名：%08d.png（0始まり）

欠番なし

解像度：canvas と一致（例：720x720）

View 切替の設計思想（重要）
原則

render_core.py は触らない

view 切替は atlas.min.json の view_rules のみで調整

現在の diag 判定条件（復習）
abs(yaw)   ≥ diag_yaw_min
abs(pitch) ≥ diag_pitch_min
かつ
abs(yaw)   < diag_yaw_max
abs(pitch) < diag_pitch_max


👉 中間角のみ
👉 対応スプライトが存在する場合のみ
👉 安全寄り設計

パラメータの意味
パラメータ	意味	下げると	上げると
diag_yaw_min_deg	出始めyaw	出やすい	出にくい
diag_pitch_min_deg	出始めpitch	出やすい	出にくい
diag_yaw_max_deg	許容yaw上限	範囲拡大	抑制
diag_pitch_max_deg	許容pitch上限	範囲拡大	抑制

効きが強いのは min 側

おすすめ調整プリセット
🟢 A. 微調整（おすすめ）
"diag_yaw_min_deg": 8,
"diag_pitch_min_deg": 4,
"diag_yaw_max_deg": 22,
"diag_pitch_max_deg": 10

🟡 B. 表現重視
"diag_yaw_min_deg": 6,
"diag_pitch_min_deg": 4,
"diag_yaw_max_deg": 22,
"diag_pitch_max_deg": 10

🔵 C. 控えめ
"diag_yaw_min_deg": 12,
"diag_pitch_min_deg": 6,
"diag_yaw_max_deg": 20,
"diag_pitch_max_deg": 9

MP4生成について

MP4は デバッグ用途

実運用・M3.5連携では FG連番が正

必要な場合のみ有効化
