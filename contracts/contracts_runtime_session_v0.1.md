contracts\_runtime\_session\_v0.1.md 清書版 12/15（チャンク処理時までには必要）

session\_runtime.json 自体は当面不要でも、\*\*「将来導入するなら：audio\_ms を正にして M0 runner が duration を上書きする」\*\*という方針は、contracts\_runtime\_session\_v0.1.md に “Optional / Future” としてメモで入れておくのが安全です（今回の「尺ズレ」再発防止にも直結するので）。

## **contracts\_runtime\_session\_v0.1（提案：これで確定に進める）**

あなたの既存I/O（revC）

contracts\_m0\_m3prime\_pose\_IO\_re…

と、LLM/TTS側の contracts v0.2

修正版 contracts

を“合体”して、**ランタイムの最小セッション契約**を以下で固めるのがおすすめです。

### **A. session\_runtime.json（v0.1）**

`{`  
  `"schema_version": "runtime_session_v0.1",`  
  `"session_id": "sess_xxx",`  
  `"utt_id": "utt_000123",`

  `"audio_ms": 2310,`  
  `"step_ms": 40,`

  `"inputs": {`  
    `"pose_timeline": "pose_timeline.json",`  
    `"mouth_timeline": "mouth_timeline.json",`  
    `"expression_timeline": "expression_timeline.json",`  
    `"atlas_json": "atlas.min.json"`  
  `}`  
`}`

**尺のルール（契約に明記）**

* `target_ms = audio_ms`

* `target_frames = ceil(target_ms / step_ms)`

* M0/M3.5は **タイムライン長に関係なく target\_frames まで必ず描画**（不足分は hold/clamp）

### **B. pose\_timeline.json（v0.1）**

revCの形式をそのまま v0.1 として確定：

* `t_ms`（ms）

* `yaw_deg/pitch_deg/roll_deg`（deg）

* `tx/ty`（px）

* `scale`（1.0基準）

例は既存の通りでOKです。

contracts\_m0\_m3prime\_pose\_IO\_re…

### **C. mouth\_timeline.json（v0.1）**

M3’側の定義を採用：

* イベント列：`{t_ms, dur_ms, mouth_id}`（mouth\_idは0..5）  
  M3prime\_contracts

* もしくは **配列（旧形式）でも可**にする場合、M0側の取り込みは既に分岐実装済みです。  
  m0\_runner (5)

### **D. expression\_timeline.json（v0.1）**

**M0が参照する最終ラベルは atlas側の expression\_labels に揃える**（例：`normal/smile/angry/sad/blink`）。  
 emo\_id→表情はあなたの表（添付）を“ソース”にし、configで差し替え可能にする。

contracts追加案（M3側からの提案）

イベント列の例：

`[`  
  `{"t_ms": 0,    "dur_ms": 2310, "expression": "normal"},`  
  `{"t_ms": 1200, "dur_ms": 120,  "expression": "blink"}`  
`]`

**優先順位（契約に明記）**

1. 同時刻に blink 区間があるなら blink

2. なければ emo\_id 由来の expression

3. 不明なら normal（M0側も表情が無ければベースへフォールバックする実装）  
   render\_core (3)

# **contracts\_runtime\_session\_v0.1.md**

## **0\. Purpose**

This document defines the **runtime session contract** for the offline/batch pipeline (and future pseudo real-time tests):

* **M1 (LLM)** outputs `emo_id` and text

* **M2 (TTS)** outputs audio (`wav`) and **audio\_ms**

* **M3** generates timelines: `mouth_timeline.json`, `pose_timeline.json`, `expression_timeline.json`

* **M0** renders frames/video based on those timelines and `atlas.min.json`

The contract focuses on:

* **Timebase alignment**

* **Duration correctness**

* **File roles \+ minimal schemas**

---

## **1\. Canonical timebase**

### **1.1 step\_ms**

All timelines MUST share the same `step_ms` (recommended: 40ms \= 25fps equivalent time step).

### **1.2 Source of truth for duration**

**`audio_ms` is the single source of truth** for the session duration.

Define:

* `target_ms = audio_ms`

* `target_frames = ceil(target_ms / step_ms)`

Renderers (M0, M3.5) MUST render exactly `target_frames` frames (or `target_ms`), regardless of the timeline file’s last timestamp.

Motivation: avoids “video longer than mouth timeline” / “mouth keeps moving after audio ends”.

---

## **2\. atlas.min.json contract (render asset index)**

### **2.1 Required keys**

`atlas.min.json` MUST provide:

* `expression_labels` (list of supported expressions)

* `expression_default`

* per-view mouth sprite mapping (front/left30/right30, …)

* view selection rules (yaw threshold) and fallback behavior

Current expected expressions are:  
 `["normal","smile","angry","sad","blink"]`

atlas.min (2)

 Default is `normal`.

atlas.min (2)

---

## **3\. pose\_timeline.json (Pose Contract v0.1)**

### **3.1 Schema**

A JSON array of pose frames, sorted by `t_ms` ascending.

Each frame MUST have:

* `t_ms` (int, milliseconds)

* `yaw_deg`, `pitch_deg`, `roll_deg` (float, degrees)

* `tx`, `ty` (float, translation; interpreted by renderer)

* `scale` (float, isotropic scale; 1.0 baseline)

Notes:

* `yaw/pitch/roll` MAY exist for backward compatibility, but `*_deg` is canonical.

### **3.2 Example**

`[`  
  `{`  
    `"t_ms": 0,`  
    `"yaw_deg": -0.66,`  
    `"pitch_deg": 0.0,`  
    `"roll_deg": 0.0,`  
    `"tx": -0.09,`  
    `"ty": -0.42,`  
    `"scale": 1.0`  
  `}`  
`]`

### **3.3 Semantics**

* Angle sign MUST match the renderer’s view selection (validated in Test2).

* Renderer may clamp `tx/ty/scale` for safety (implementation detail).

---

## **4\. mouth\_timeline.json (Mouth Contract v0.1)**

### **4.1 Canonical schema (event-based)**

Mouth timeline is event-based and MUST use `mouth_id` (integer).

Required keys:

* `audio` (string, wav filename or relative path)

* `step_ms` (int)

* `frames` (array of events)

Each event MUST have:

* `t_ms` (int)

* `mouth_id` (int)

Optional:

* `dur_ms` (int). If absent, duration is assumed to be `step_ms` per event.

### **4.2 mouth\_id vocabulary**

`mouth_id` is an integer in `[0..5]`, meaning:

* 0: close

* 1: a

* 2: i

* 3: u

* 4: e

* 5: o

### **4.3 Example (current)**

`{`  
  `"audio": "1_0_72.wav",`  
  `"step_ms": 40,`  
  `"frames": [`  
    `{"t_ms": 0, "mouth_id": 0},`  
    `{"t_ms": 40, "mouth_id": 0},`  
    `{"t_ms": 400, "mouth_id": 5}`  
  `]`  
`}`

(Reference example exists in current test asset.)

mouth\_timeline\_test1

### **4.4 Duration rule**

The renderer MUST NOT assume `mouth_timeline` length is equal to session duration.  
 Render duration is determined only by `audio_ms` (Section 1.2).

If mouth frames do not cover `target_frames`, renderer should hold last mouth state or use `close` as fallback (implementation choice).

---

## **5\. expression\_timeline.json (Expression Contract v0.1)**

### **5.1 Purpose**

Expression timeline supplies `expression` labels that select expression-specific sprite folders.

### **5.2 Labels**

`expression` MUST be one of `atlas.min.json.expression_labels`.

atlas.min (2)

 Unknown labels must be treated as `normal`.

### **5.3 Schema (event-based)**

JSON array, each event has:

* `t_ms` (int)

* `dur_ms` (int)

* `expression` (string)

Example:

`[`  
  `{"t_ms": 0, "dur_ms": 2310, "expression": "normal"},`  
  `{"t_ms": 1200, "dur_ms": 120, "expression": "blink"}`  
`]`

### **5.4 Priority rule (blink)**

If multiple expressions overlap at time `t_ms`, resolution MUST follow:

1. `blink` overrides everything

2. else use the active (non-blink) expression

3. else fallback to `expression_default` (`normal`)  
    atlas.min (2)

---

## **6\. emo\_id → expression mapping (owned by M3)**

### **6.1 Ownership**

* **LLM (M1)** outputs `emo_id` only.

* **M3** converts `emo_id` to `expression_timeline.json` based on a configurable mapping table.

Rationale:

* Keeps LLM tokens low (no verbose emotion json)

* Makes mapping easy to tweak via config

### **6.2 Config requirement**

The mapping MUST be configurable (e.g., YAML), not hardcoded.

---

## **7\. Optional future: session\_runtime.json (v0.1)**

### **7.1 Status**

**Optional / not required for current batch tests.**

### **7.2 When used (pseudo real-time / pipeline integration tests)**

If introduced, `session_runtime.json` SHOULD include:

* `session_id`, `utt_id`

* `audio_ms`, `step_ms` (and optionally `target_frames`)

* paths to pose/mouth/expression timelines and atlas

Minimal example:

`{`  
  `"schema_version": "runtime_session_v0.1",`  
  `"session_id": "sess_xxx",`  
  `"utt_id": "utt_000123",`  
  `"audio_ms": 2310,`  
  `"step_ms": 40,`  
  `"inputs": {`  
    `"pose_timeline": "pose_timeline.json",`  
    `"mouth_timeline": "mouth_timeline.json",`  
    `"expression_timeline": "expression_timeline.json",`  
    `"atlas_json": "atlas.min.json"`  
  `}`  
`}`

### **7.3 Duration override rule (recommended)**

If `session_runtime.json` is used:

* M0 runner MUST compute `duration_s = audio_ms / 1000.0` and override any config `video.duration_s`.

* `video.duration_s` is treated as fallback only.

This is the recommended choice to prevent duration drift.

---

## **8\. Compatibility notes**

* `mouth_timeline` may internally be converted to mouth label strings for legacy renderers, but **contract is `mouth_id`**.

* `pose_timeline` may contain both `yaw` and `yaw_deg`; `yaw_deg` is canonical.

---

### **不明点・不足点**

atlas 側は、表情ラベルと default まで明確に揃っているので不足なしです。

atlas.min (2)

 追加で「完全に実装と一致させたい」場合だけ、次のどちらかがあるとさらに堅いです（任意）：

* M0が `tx/ty` を **px** として扱っているか、**正規化値**として扱っているか（現在の挙動が“見た目OK”なので、契約上は “renderer-defined units” として曖昧にしても運用できます）

* M3側の `expression_timeline` 生成 config のキー名（mapping yamlの具体名）

\# Appendix A: Session Runner 実装・運用プロトコル（v0.1）

\#\# A.1 位置づけと責務分離

本 Appendix は、\`contracts\_runtime\_session\_v0.1\` で定義された  
「session\_audio\_ms を正とするランタイム契約」を、  
\*\*実装としてどのように実行・検証するか\*\*を定める。

責務分離は以下の通りとする：

\- session runner  
  \- session.json を唯一の入力として受け取る  
  \- session\_audio\_ms から duration / frame 数を派生  
  \- M0 に渡す最終設定（session\_render.yaml）を生成  
\- M0  
  \- 与えられた設定に従い「描画のみ」を行う  
  \- audio / pose / mouth / expression を同時に消費可能  
  \- session.json を直接解釈しない

\---

\#\# A.2 session.json（最小ラッパー契約）

session runner が受け取る最小入力は以下とする。

\`\`\`json  
{  
  "schema\_version": "session\_runtime\_v0.1",  
  "session\_id": "sess\_xxx",  
  "session\_audio\_ms": 36040,

  "pose\_timeline": "tests/timelines/pose\_timeline\_xxx.json",  
  "mouth\_timeline": "tests/timelines/mouth\_timeline\_xxx.json",  
  "expression\_timeline": "tests/timelines/xxx.expression\_timeline.json",

  "audio": "tests/audio/xxx.wav"  
}  
ルール：

すべて repo-relative path

尺の正は session\_audio\_ms（ms）

timeline が短い場合は後段で clamp される

A.3 派生値ルール（v0.1）  
session runner は以下を派生生成する：

duration\_s \= ceil(session\_audio\_ms / 1000\)

fps \= base\_config 由来（通常 25）

target\_frames \= duration\_s \* fps

※ v0.1 では duration\_s は M0 の都合パラメータであり、  
　将来は target\_frames / audio\_ms 基準へ完全移行予定。

A.4 session\_render.yaml の生成ルール  
assets（見た目）の基準  
yaml  
コードをコピーする  
io:  
  assets\_dir: tests/assets\_min

atlas:  
  atlas\_json: atlas.min.json  
atlas 内の画像パスは assets\_dir 基準で解決される

timeline / audio（中身）の基準  
yaml  
コードをコピーする  
inputs:  
  pose\_timeline: /abs/path/to/pose.json  
  mouth\_timeline: /abs/path/to/mouth.json  
  expression\_timeline: /abs/path/to/expression.json

audio:  
  wav\_path: /abs/path/to/audio.wav  
session\_render.yaml では絶対パスを使用

assets\_dir の前置ロジックによるパス混乱を防止するため

A.5 実行ログと合格基準  
session runner 実行は以下を満たすこと：

m0 returncode \= 0

rendered\_frames \== target\_frames

audio\_mux\_succeeded \== true

fallback\_frames \== 0（黒画面・素材欠落なし）

audio\_ms と duration\_ms の差分は v0.1 では WARN のみとする。

A.6 既知の注意点（v0.1）  
assets\_dir と atlas\_json の基準不一致は全フレーム fallback を招く

timeline/audio を assets\_dir 相対で渡すと二重 prefix が起きやすい

本 Appendix の方式（assets\_dir 固定＋中身は絶対パス）を標準とする

markdown  
コードをコピーする

＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿  
1\) contracts.md 追記フル版（v0.1 \+ Appendix）

ファイル名案：contracts/contracts\_runtime\_session\_v0.1\_plus\_appendix.md  
既存 v0.1 を維持しつつ、今回の確定事項（FG連番・view\_rules・MP4任意）を追記します。

\# contracts\_runtime\_session\_v0.1 \+ Appendix (M0 FG Output / View Rules)

最終更新: 2026-01-09  
対象: M3 → M0 → M3.5 のオフライン/疑似チャンク統合

\---

\#\# 0\. 本契約の目的

\- \*\*TTS由来の audio\_ms を唯一のソース・オブ・トゥルース\*\*として、  
  セッション内の pose/mouth/expression/FG を同期させる。  
\- 実運用（配信）では、最終的に \*\*M0 が生成する FG(BGRA PNG連番)\*\* を  
  M3.5 が背景動画と合成する。

\---

\#\# 1\. 用語

\- \*\*step\_ms\*\*: 基本時間グリッド。既定 40ms（= 25fps）  
\- \*\*audio\_ms\*\*: TTS音声長（ms）  
\- \*\*target\_frames\*\*: 描画フレーム数（セッションの唯一の長さ）  
\- \*\*FG\*\*: 前景フレーム列（M0が生成する BGRA PNG連番）

\---

\#\# 2\. 重要原則（必読）

1\) \`audio\_ms\` が唯一の真（SSOT）  
2\) \`target\_frames\` は runtime/session\_runner が決める唯一の値  
3\) pose/mouth/expression が短い場合は \*\*hold/clamp\*\* で吸収する（異常ではない）  
4\) 最終成果物に必要なのは \*\*FG連番\*\*（MP4は任意・デバッグ用）

\---

\#\# 3\. target\_frames の定義

\- \`step\_ms \= 40\`  
\- \`target\_frames \= ceil(audio\_ms / step\_ms)\`

例:  
\- audio\_ms=36040 → target\_frames=ceil(36040/40)=901（※設計上の例）  
\- 実装上、運用で duration\_s を整数に切り上げる場合は、  
  \`duration\_s \= ceil(audio\_ms / 1000)\` とし \`target\_frames \= duration\_s \* fps\`  
  の派生になることがある（この場合でも \*\*audio\_ms をSSOT\*\*とする方針は不変）。

\---

\#\# 4\. session.json（セッション入力）

\#\#\# 4.1 必須キー（例）

\- \`session\_id\`: string  
\- \`session\_audio\_ms\`: int  
\- \`audio\_path\`: string（wav）  
\- \`pose\_timeline\`: string  
\- \`mouth\_timeline\`: string（任意だが通常は存在）  
\- \`expression\_timeline\`: string（任意）

\#\#\# 4.2 パス解決

\- すべて repo-root 相対で記述してよい。  
\- runtime（session\_runner）が存在確認し、必要に応じて絶対パスへ解決して実行する。

\---

\#\# 5\. pose\_timeline.json（M0入力：pose）

\#\#\# 5.1 基本

\- \`t\_ms\` は ms  
\- \`yaw/pitch/roll\` は \*\*degree\*\*  
\- \`tx/ty\` は canvas座標（px）  
\- \`scale\` はスプライト基準のスケール

\#\#\# 5.2 meta

\- \`meta.scale\_mode\`: \`"absolute"\` を推奨  
  \- 背景/キャンバス差異が将来混在することを前提に、\`absolute\` を正とする  
  \- \`relative\` でも“同一canvas条件”では破綻しない場合があるが、契約上は absolute 推奨

\---

\#\# 6\. mouth\_timeline / expression\_timeline（M0入力）

\- いずれも \`t\_ms\`（ms）で同一 step\_ms グリッドに乗ることが望ましい  
\- 短い場合は hold/clamp で吸収する

\---

\# Appendix A: M0 → M3.5 FG出力契約（最重要）

\#\# A.1 FG出力の正（必須）

M0は、セッション単位で \*\*FG PNG連番\*\*を生成する。

\- 出力パス（推奨）: \`out/fg/in/fg/%08d.png\`  
\- 命名: \*\*0始まり\*\*（\`00000000.png\`）  
\- 欠番: \*\*禁止\*\*  
\- 形式: PNG  
\- チャンネル: \*\*4ch (BGRA)\*\*（alpha必須）  
\- 解像度: \`canvas\_w x canvas\_h\`（BGと一致推奨、例 720x720）

\> 注: PNG内部はRGBA表現でもよいが、OpenCV等で読み込む際に  
\> \`cv2.IMREAD\_UNCHANGED\` で4chとして扱えることが必須。

\#\# A.2 FGのフレーム数

\- 期待フレーム数: \`target\_frames\`  
\- 出力フレーム数が不足する場合は不合格（欠番と同等）

\#\# A.3 MP4生成について

\- MP4は \*\*デバッグ用途\*\*（任意）  
\- 実運用・統合の正は FG連番  
\- MP4がalphaを持たないことが一般的であるため、FG連番生成を必須とする

\---

\# Appendix B: View切替契約（render\_core不変）

\#\# B.1 原則

\- view切替の挙動調整は \*\*assets/atlas.min.json の view\_rules のみ\*\*で行う  
\- \`render\_core.py\` は原則修正しない（確定版）

\#\# B.2 diag選択条件（参考）

abs(yaw) \>= diag\_yaw\_min\_deg  
abs(pitch) \>= diag\_pitch\_min\_deg  
かつ  
abs(yaw) \< diag\_yaw\_max\_deg  
abs(pitch) \< diag\_pitch\_max\_deg  
かつ  
対応スプライトが存在する場合のみ、diag view を優先する

\---

\# Appendix C: 疑似チャンク（将来拡張）方針

\- セッション全体の FG は \*\*通し番号\*\*を基本とする（0..N-1）  
\- 疑似チャンクでは chunk単位の処理を行っても、  
  FG番号・t\_ms・audio\_msの整合が崩れないようにする

（疑似チャンクの詳細仕様は \`chunk\_manifest\` 契約にて別途定義）


