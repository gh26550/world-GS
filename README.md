# World-GS

プロンプトから部屋全体の正面画像を作り、そこから360°の室内構成を推定し、共通メッシュの多視点参照から3D Gaussian Splattingを学習する研究用実装です。既定は **`--mode image-room`** です。

1. 部屋全体の正面画像を生成し、画像を読んで全体の配置・支持関係を推定します。
2. 側面・背面の部屋画像を仮形状で誘導して補完します。
3. 室内画像から物体を切り出してSAM3Dメッシュを復元し、部屋全体の共通メッシュを組み立てます。
4. 全周のRGB・メッシュ深度・法線・面上の参照点を取得し、3DGSを学習します。

[新方式の処理・出力・制約](docs/image-room-v12.md)を参照してください。背面は推測です。最初の画像は `source_front.png`、補完画像は `room_images/`、基準メッシュは `scene_mesh.glb`、GS教師は `view_*.png`、学習後GSは `scene_unity.ply` です。

従来の「物体画像を個別生成する」方式は **`--mode learned`** に残しています。既存の生成を再開するときも、従来の方式には `--mode learned --resume` を明示してください。

[従来方式](docs/learned-v10.md)、[小物と支持検査](docs/furnished-room-v11.md)、[先行研究](docs/research.md)、[Unityへの取り込み](docs/unity.md)、[参考画像に合わせた読書室](docs/reference-library-20261026.md)。
## 実行（このPCのWSL）

```bash
cd '/home/dpc7/world GS'
./run.sh generate --prompt 'A cozy living room with a sofa, bookshelves and a wooden table, wide angle interior photograph' --output runs/livingroom_image_new --size 384 --views 12 --seed 42
./run.sh train runs/livingroom_image_new --steps 2000 --max-points 100000
```

通常の入力はプロンプトのみです。`--room-size WIDTH HEIGHT LENGTH`を指定した場合は部屋寸法を優先します。未指定の寸法や配置は推定であり、実測値ではありません。

`--views`は水平の分割数です。上・下向き、家具周囲、部屋の隅、床・天井を直接見る移動カメラも追加するため、参照画像の総数は指定値より多くなります。`--max-points`は初期ガウス数の上限で、現状の学習中の上限は180,000です。

生成済みの段階から再開する場合は、**同じプロンプト・サイズ・視点数・seed・部屋寸法指定**に`--resume`を追加します。既存の別要求のフォルダは上書きしません。

```bash
./run.sh generate --prompt 'A cozy living room with a sofa, bookshelves and a wooden table, wide angle interior photograph' --output runs/livingroom_image_new --size 384 --views 12 --seed 42 --resume
```

計画器だけを検証するときは、別の空フォルダを指定します。

```bash
./run.sh plan --prompt 'ベッド1台と木製の椅子2脚を置いた寝室。椅子はベッドの隣。' --output runs/bedroom_plan_new --room-size 6 3 6
```

コンソールに段階と物体・カメラの進捗を表示し、`status.json`と`logs/`にも保存します。

## 実行環境

主環境はPython 3.10〜3.12とCUDA対応PyTorch/gsplatです。空白のあるPyTorchパスではCUDA拡張のリンクが失敗するため、仮想環境は`~/.venvs/world-gs`を使用します。`bash setup.sh`は主環境を準備します。

新方式には、別途 **SAM3・SAM 3D Objectsのコード／モデル／専用環境とBlender** が必要です。このPCでは導入済みの以下を自動検出して使用します。他のPCでは環境変数で指定できます。

```bash
export WORLDGS_SAM3D_ROOT="$HOME/worldmesh-main-optimized/sam-3d-objects"
export WORLDGS_SAM3_PYTHON="$HOME/miniforge/envs/worldmesh-sam3/bin/python"
export WORLDGS_SAM3D_PYTHON="$HOME/miniforge/envs/worldmesh-sam3d-objects/bin/python"
export WORLDGS_BLENDER="$HOME/software/blender-5.1.2-linux-x64/blender"
```

新方式の画像解析は既定で`Qwen/Qwen2.5-VL-3B-Instruct`、従来のlearned計画器は`Qwen/Qwen2.5-3B-Instruct`を使用し、初回にモデルを取得します。変更する場合は`generate --planner-model MODEL`、計画単体では`plan --model MODEL`を指定します。画像・分割・3D生成は順番に実行してGPUメモリを解放します。既存の別プロジェクト環境を上書きインストールしません。

## 出力

| ファイル | 内容 |
|---|---|
| `planning/scene_plan.json` | 要求物体、数量、材質、配置関係、仮定、LLM検査と配置補正 |
| `assets/<物体ID>/image.png`, `mask.png` | 採用した物体画像と分割マスク |
| `assets/<物体ID>/generated.glb` | SAM 3Dから生成した元のUV付きメッシュ |
| `assets/<物体ID>/placed.glb` | 尺度・向きを揃え、接地・配置したメッシュ |
| `scene_mesh.glb`, `meshes/*.glb` | 部屋と全物体の基準メッシュ群 |
| `physical_scene.blend` | 実際に描画した材質・照明・カメラ付きBlenderシーン |
| `render_job.json`, `render_state.json` | カメラ・照明設定と描画記録 |
| `view_*.png`, `view_*_depth.npy`, `view_*_normals.npy`, `view_*_faces.npy` | 学習用のRGB・カメラz深度・世界座標法線・三角形ID |
| `heldout_*.png`, `scene.json` | 学習から除外したカメラの参照と全体設定 |
| `references.jpg`, `heldout_references.jpg` | 参照画像一覧 |
| `references.html`, `object_visibility.json` | 画像を開ける一覧、小物を含む各物体の可視視点・参照点数 |
| `points.npz`, `mesh_geometry.npz`, `structure_quality.json` | 初期点と面の対応、形状・構造検査 |
| `scene_unity.ply`, `scene.ply` | 学習後のbinary little-endian Gaussian PLY（62属性） |
| `gaussians.npz`, `training.json`, `evaluation.json` | 学習結果、設定、未学習視点の評価 |
| `heldout_preview.png` | 未学習視点での基準描画とGSの比較 |

今回の実行結果は`/home/dpc7/world GS/runs/livingroom_learned_v10_validated`に保存しています。Unityには、このフォルダの`scene_unity.ply`を読み込んでください。生成データ・モデル本体はGitに含めず、コードと文書をGitHubへ保存します。

## 検証範囲と制約

単一の矩形室内、床支持の家具、平坦な家具面に載る小物を対象にしています。背面壁の矩形窓1つを明示指定でき、薄い床ラグにも対応します。複数部屋・その他の開口・天井吊り・棚内部の任意位置などは未対応です。自然言語の完全な理解を保証するものではなく、LLMの要求検査にも誤りは残り得ます。

物体の背面・細部・寸法は推定です。生成UVテクスチャには照明が混ざる場合があり、厳密な材質分離ではありません。全周参照は生成メッシュのUVテクスチャとPBR照明で作り、独立した視点ごとの画像補完は使用しません。GSのぼけや色むらは残り、Unityの動的ライトによる再照明は未対応です。

```bash
~/.venvs/world-gs/bin/python -m unittest discover -s tests -v
./run.sh generate --mode mesh --prompt test --output runs/mesh_demo_new --demo --size 64 --views 8
```

`--demo`は旧メッシュ経路の合成データ検査用で、新方式の代替出力ではありません。過去の方式は明示的に`--mode mesh`（v8）、`scene`（v9実験）、`expand`（v4〜7）、`room`、`legacy`で利用できます。旧深度修復・TSDF・穴の診断は旧方式向けです。