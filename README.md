# YOLO26 顔モザイクアプリ

動画をアップロードし、**顔検出済みの YOLO26 系モデル**で顔を検出して、顔領域にモザイクをかける Streamlit アプリです。

このリポジトリは **uv + uv.lock** 前提で整理してあり、Windows / macOS / Linux / WSL2 から同じ手順で環境構築しやすい構成にしています。


## できること

- 動画ファイルをアップロード
- 顔検出モデルでキーフレーム推論
- 中間フレームはボックス補間
- 顔領域にモザイク適用
- 出力動画を `Results/` に保存
- `ffmpeg` が入っていれば元動画の音声を再付与

---

## 想定している利用環境

- **Windows 11 / Windows 10**
- **Windows + WSL2 (Ubuntu 推奨)**
- **macOS (Apple Silicon / Intel)**
- **Ubuntu などの Linux**

GPU は任意です。GPU が見つからない場合は CPU で動作します。

---

## 利用者が最初に確認すること

### 1. Git を使える状態にする

```bash
git --version
```

### 2. uv をインストールする

#### macOS / Linux / WSL

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

確認:

```bash
uv --version
```

### 3. 顔検出モデルを手元に置く

このアプリは**顔検出用の重み**が必要です。

最低限、次のどちらかを `models/` に配置してください。

- `models/yolo26l_face_full.pt` ← アプリ用の推奨ファイル名
- `models/yolo26l_face_full.engine`

必要に応じて、学習や比較用に次も置けます。

- `models/yolo26l.pt` ← ベースモデル（任意）

> `yolo26l.pt` のような一般物体検出モデルだけでは、顔モザイクアプリは使えません。  
> アプリには **顔を検出できる学習済みモデル** を置いてください。

### 4. GPU を使いたい場合はホスト側ドライバを先に整える

- NVIDIA: 先に GPU ドライバを入れて `nvidia-smi` が動く状態にする
- Apple Silicon: macOS 側でそのまま実行可
- CPU のみ: 追加設定なし

### 5. 音声付きで保存したい場合は ffmpeg を入れる

未導入でもアプリ自体は動きますが、その場合は無音動画になることがあります。

確認:

```bash
ffmpeg -version
```

例:

#### Windows

```powershell
winget install Gyan.FFmpeg
```

#### macOS

```bash
brew install ffmpeg
```

#### Ubuntu / WSL

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### 6. 配置場所のおすすめ

次のような **短くて分かりやすいパス** に clone してください。

- Windows: `C:\work\face-mosaic-app`
- macOS / Linux / WSL: `~/work/face-mosaic-app`

---

## Clone からアプリ起動まで

以下では GitHub などの URL を例として `<REPOSITORY_URL>` と書いています。実際の URL に置き換えてください。

### Windows PowerShell

```powershell
git clone <REPOSITORY_URL>
cd face_mosaic_streamlit_bundle_uv
```

`models/` に学習済みモデルを配置したあと、次を実行します。

```powershell
.\setup_uv.bat
.\run_app.bat
```

*.sh : Bash / WSL

*.bat : Windows cmd.exe

*.ps1 : PowerShell

PowerShell から直接実行する場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_uv.ps1
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

### macOS / Linux

```bash
git clone <REPOSITORY_URL>
cd face_mosaic_streamlit_bundle_uv
chmod +x setup_uv.sh run_app.sh
./setup_uv.sh
./run_app.sh
```

### WSL2 (Ubuntu 例)

Windows ではなく **Ubuntu 側ターミナル** で実行します。

```bash
git clone <REPOSITORY_URL>
cd face_mosaic_streamlit_bundle_uv
chmod +x setup_uv.sh run_app.sh
./setup_uv.sh
./run_app.sh
```

起動後、端末に表示された `http://localhost:8501` をブラウザで開いてください。

---

## 何が自動で行われるか

`setup_uv.sh` / `setup_uv.ps1` は以下を行います。

1. `uv` で Python 3.11 を準備
2. `uv.lock` を使って依存関係を固定インストール
3. GPU 情報から Torch プロファイルを自動選択
4. 動作確認用に `verify_torch_env.py` を実行
5. 起動用設定を `.uv-profile` に保存

そのため、セットアップ後は毎回長いコマンドを覚えなくても、次だけで起動できます。

- Windows: `run_app.bat`
- macOS / Linux / WSL: `./run_app.sh`

---

## プロファイル自動選択の考え方

`setup_uv` は概ね次のルールで Torch プロファイルを選びます。

- NVIDIA + CUDA 13 以上: `cu130`
- NVIDIA + CUDA 12.8 以上: `cu128`
- NVIDIA + CUDA 12.6 以上: `cu126`
- それ以外: `cpu`

Apple Silicon では `cpu` プロファイルで構築し、実行時に MPS が使える場合は MPS を利用します。

### 手動で固定したい場合

#### Windows

```powershell
.\setup_uv.bat cpu
.\setup_uv.bat cu126
.\setup_uv.bat cu128
.\setup_uv.bat cu130
```

#### macOS / Linux / WSL

```bash
./setup_uv.sh cpu
./setup_uv.sh cu126
./setup_uv.sh cu128
./setup_uv.sh cu130
```

---

## WSL を使う場合の案内

Windows で Python / GPU 周りをなるべく Linux 寄りの手順に揃えたい場合は、WSL2 の利用をおすすめします。

### 1. 管理者権限の PowerShell で WSL を入れる

```powershell
wsl --install
```

再起動後、必要なら更新します。

```powershell
wsl --update
```

### 2. WSL2 になっているか確認する

```powershell
wsl -l -v
```

### 3. Ubuntu を開いて Git / ffmpeg を入れる

```bash
sudo apt update
sudo apt install -y git ffmpeg
```

### 4. Ubuntu 側で clone してセットアップする

```bash
git clone <REPOSITORY_URL>
cd face_mosaic_streamlit_bundle_uv
./setup_uv.sh
./run_app.sh
```

### 5. WSL で NVIDIA GPU を使いたい場合

- Windows 側に **WSL 対応 NVIDIA ドライバ** を入れる
- `wsl --update` を実行しておく
- Ubuntu 側で `nvidia-smi` が見えるか確認する

```bash
nvidia-smi
```

`nvidia-smi` が見えない場合は、まず Windows 側のドライバと WSL 更新状態を確認してください。

---

## 初回起動の流れ

1. 動画をアップロード
2. 顔検出モデルのパスを確認
3. `顔検出を実行`
4. `モザイク動画を生成`
5. `Results フォルダへ保存`

> この版ではアプリ上に動画プレビューは出しません。  
> 生成済みファイル名だけ表示し、保存後は `Results/` で確認する前提です。

---

## ディレクトリ構成

```text
face_mosaic_streamlit_bundle_uv/
├─ face_mosaic_streamlit_app.py
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ requirements.txt
├─ setup_uv.sh
├─ setup_uv.ps1
├─ setup_uv.bat
├─ run_app.sh
├─ run_app.ps1
├─ run_app.bat
├─ verify_torch_env.py
├─ train_face_detector.py
├─ train_uv.sh
├─ train_uv.ps1
├─ models/
│  ├─ README.md
│  ├─ yolo26l_face_full.pt      # 利用者が配置
│  └─ yolo26l.pt                # 任意
├─ data/
└─ Results/
```

---

## うまくいかないとき

### `uv` が見つからない

- ターミナルを開き直す
- `uv --version` を確認する
- PATH に反映されていない場合は再ログインする

### モデルが見つからない

- `models/` に実ファイルを置いたか確認する
- `.pt.txt` のようなプレースホルダではなく、実際の重みファイルを置く
- アプリの `モデルパス` が正しいか確認する

### GPU が使われない

- Windows / Linux / WSL で `nvidia-smi` が通るか確認する
- 必要なら `setup_uv` を `cu126` / `cu128` / `cu130` で再実行する
- まずは `cpu` で起動確認してから GPU に切り替える

### WSL で GPU が見えない

- Windows 側の NVIDIA ドライバを見直す
- `wsl --update` を実行する
- Ubuntu 側で `nvidia-smi` を再確認する

### 保存動画に音が入らない

- `ffmpeg -version` を確認する
- `ffmpeg` が無い場合はインストールする

### Windows で PowerShell 実行制限に引っかかる

`.bat` ラッパー経由で実行してください。

```powershell
.\setup_uv.bat
.\run_app.bat
```

---

## 開発者向けメモ

- 依存関係は `uv.lock` を基準に固定しています
- セットアップ時に `.uv-profile` を保存するので、起動スクリプトはその内容を再利用します
- 一時ファイルは `.work/`、保存ファイルは `Results/` を使います
- 学習済みモデルの実体はリポジトリに含めず、`models/` に後置きする運用を想定しています

---

## 参考

- uv: https://docs.astral.sh/uv/getting-started/installation/
- WSL: https://learn.microsoft.com/windows/wsl/install
- Streamlit の起動方法: https://docs.streamlit.io/develop/concepts/architecture/run-your-app
