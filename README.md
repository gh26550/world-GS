# World-GS

プロンプト → 正面画像 → 部屋・家具のメッシュ群 → メッシュから360°参照画像・深度を描画 → 形状を保持して3D Gaussian Splattingを学習する研究用プロトタイプ。

既定は **`--mode mesh` / `mesh_teacher_v8`** です。正面画像から推定した配置を使って床・天井・壁・家具を立体部品として組み立て、その形状を全視点で共有します。参照深度はメッシュとの交差から直接取得し、参照画像から再推定しません。GSの中心・回転・スケールを固定し、不透明度・SH色を学習します。[変更箇所・検証・制約](docs/mesh-teacher-v8.md)を参照してください。

正面画像の単眼深度は最初の家具配置にだけ使用します。見えていない物体や細部を正確に復元する方式ではなく、家具は手続き的な近似形状です。外観は手続き的な材質と固定照明で、写真品質やUnityの動的照明への対応は未実装です。

旧 `--mode expand` は比較用に保持します。以下のv4〜v7の説明と深度修復・TSDFコマンドは旧方式のものです。

v6では生成したRGB-Dをそのまま板状メッシュとして追加する最終処理を廃止し、視点間の支持を検査してTSDFで全体を融合します。低信頼の暫定面は融合・学習の重みを下げます。GS初期点は検査に通った最終メッシュの面からのみ採り、不合格メッシュでは学習を停止します。[メッシュの作り方・修正内容・比較](docs/mesh-v6.md)を参照してください。元画像の不整合を解決する方法や、穴のない物体メッシュを生成する方法ではありません。

v7は穴の原因をメッシュ欠落とGSの不透明度不足に分け、製品用では全入力視点を融合します。境界画素の回復、元深度で支持される小穴の局所補修、既知の面の被覆損失、空間分布を保つ初期点・点数制限を導入しました。[穴の原因・修正と比較](docs/holes-v7.md)を参照してください。

旧expand方式の未観測内容は生成による仮説であり、実物の復元ではありません。単眼深度による歪み、継ぎ目、家具の重複や欠落が残ります。v3の部品メッシュを新しいmesh方式の基礎として使用し、v2は `--mode room`、初期方式は `--mode legacy` で保持しています。

深度・境界の修正版v5では、逆深度の倍率とオフセットを外れ値に頑健に合わせ、物体境界をまたがない深度補正と狭い範囲の色調整を行います。重なり部分の三角形も利用して接続の隙間を減らします。詳細は [深度・境界修正](docs/depth-seams-v5.md) を参照してください。

## セットアップと実行（Ubuntu / WSL）

Python 3.10〜3.12、CUDA toolkitとC++コンパイラが必要です。仮想環境は空白のない `~/.venvs/world-gs` に作ります。

```bash
cd '/home/dpc7/world GS'
bash setup.sh
./run.sh generate --prompt 'A cozy living room with a sofa, bookshelves and a wooden table, wide angle interior photograph' --output runs/livingroom_mesh_new --size 384 --views 12 --seed 42
./run.sh train runs/livingroom_mesh_new --steps 2000 --max-points 120000
```

`--views` は水平方向の分割数です。各方向に水平・上向き・下向きのカメラを配置し、家具周囲の視点を追加します。約20%を評価用に除外するため、画像の総数は指定値より多くなります。

配置を手動修正する場合は、保存済み `layout.json` の `shapes` を編集して、別の空ディレクトリへ生成します。この場合、深度推定・物体検出を再実行しません。画像サイズを変える場合は、色抽出用の検出座標も拡縮します。

```bash
./run.sh generate --prompt 'A cozy living room' --output runs/livingroom_layout_new --size 384 --views 12 --front-image runs/livingroom_mesh_new/front.png --layout-file runs/livingroom_mesh_new/layout.json
```

## 出力（既定のmesh方式）

- `scene_mesh.glb`, `proxy_scene.obj`, `meshes/*.obj`: 全体と部屋・家具ごとの基準メッシュ。
- `layout.json`: 部屋寸法、家具配置、立体部品の一覧。
- `view_*.png`, `view_depth_*.npy`, `view_normals_*.npy`, `view_faces_*.npy`: 同じメッシュから描画したRGB、カメラz深度、世界座標法線、三角形ID。
- `scene.json`: カメラ姿勢・内部パラメータと形状の識別情報。
- `points.npz`, `mesh_geometry.npz`, `structure_quality.json`: 面上の初期点、基準形状、平面性と点の付着検査。
- `references.jpg`, `heldout_references.jpg`: 学習用・評価用参照画像の一覧。
- 学習後の `scene_unity.ply`: Unityへ読み込むbinary little-endian Gaussian PLY。
- `gaussians.npz`, `gaussian_surface.npz`, `training.json`: 学習結果、各ガウスと基準三角形の対応、形状固定の確認。
- `heldout_preview.png`, `evaluation.json`: 同じメッシュの未学習視点での比較。実物との精度評価ではありません。

## 旧expand方式の操作

既存のexpandモードの画像を再利用して深度と境界だけを修正する場合：

```bash
./run.sh refine-depth runs/livingroom_expand_v4_final --output runs/livingroom_depth_fixed_new
./run.sh train runs/livingroom_depth_fixed_new --steps 2000 --max-points 60000
```

`refine-depth` は既存のキーフレーム画像から深度を再推定し、共有形状と参照点を再構築します。画像の新規生成は行いません。GSは修正先のフォルダで再学習します。

既存の深度を使い、メッシュの融合方法だけを変更する場合：

```bash
./run.sh rebuild-mesh runs/livingroom_depth_seams_v5_final --output runs/livingroom_fused_new --voxel-size .025 --min-views 2
./run.sh train runs/livingroom_fused_new --steps 2000 --max-points 60000
```

`--min-views` は高信頼面の支持視点数です。既定では他視点と矛盾しない滑らかな単視点面も低信頼（重み0.25）で保持します。`--strict-support` を追加すると単視点面を除外するため、穴が増えます。Open3D 0.19.0が必要で、`setup.sh` は `meshing` 依存も導入します。

既定では入力キーフレームを全て使います。融合・学習から1視点を除外する実験だけ、`rebuild-mesh` に `--holdout-keyframe INDEX` を追加してください。穴の診断はGPU環境で以下を実行します。

```bash
./run.sh audit-holes runs/livingroom_fused_new --output runs/livingroom_fused_new_audit
```

保存先には空のディレクトリが必要です。比較で既存画像を使う場合のみ `--front-image runs/livingroom/front.png` を指定します。通常はプロンプトから生成します。初回はモデル取得とCUDAビルドに時間がかかります。`run.sh` は外部環境とGPUアーキテクチャを設定します。環境変更は `WORLDGS_VENV`、GPU指定は `WORLDGS_CUDA_ARCH` を使用します。空白のある `.venv` のリンクエラーは引用符を増やしても解消しません。

## 出力（expandモード）

- `front.png`, `front_relative_depth.npy`: 最初の正面画像と相対深度。
- `initial_scaffold.glb`: 制御用の初期メッシュ。最終教師ではありません。
- `scene_mesh.glb`, `proxy_scene.obj`, `meshes/generated_patch_*.glb`: 生成RGB-Dで更新した全体形状と追加パッチ。
- v6の最終 `scene_mesh.glb` は融合メッシュです。新しい生成・深度修正では `observations/` に補完処理の中間データを保存します。中間パッチはGSの最終メッシュではありません。
- `mesh_quality.json`, `support_report.json`, `fusion_holdout.json`: 面の品質、視点の支持、融合から除外した視点での確認。
- `view_confidence_*.npy`: 高信頼1.0・暫定0.25・無効0の学習重み。
- `hole_repair.json`: 小穴候補の深度検証・採用数。`heldout_alpha_*.npy`: 学習後の不透明度。
- `guide_*.png`, `completion_mask_*.png`, `completion_history.json`: 各視点のガイド、補完領域、形状追加・整合率の記録。
- `raw_depth_*.npy`, `depth_diagnostics.json`: 補正前の推定深度、外れ値・校正・境界段差の記録。
- `view_*.png`, `view_depth_*.npy`, `view_mask_*.npy`: 学習用の描画・深度・有効マスク。
- `heldout_*.png`, `scene.json`, `references.jpg`: 評価用描画、全カメラ姿勢と内部パラメータ、参照一覧。
- `points.npz`: 補完画像と推定深度から融合した初期点群。
- `scene.ply`, `scene_unity.ply`: 同内容のbinary little-endian Gaussian PLY。位置・回転・3軸スケール・不透明度・SH3を保持。
- `gaussians.npz`, `training.json`: 学習パラメータと記録。
- `heldout_preview.png`, `evaluation.json`: 未学習視点でのメッシュとGSの比較。実物に対する精度評価ではありません。

UnityのGaussian Splat Creatorには `scene_unity.ply` を指定します。GLB/OBJは形状確認用です。[Unity手順](docs/unity.md)も参照してください。

## 検証と制約

expandモードはRGBと推定深度を使用し、深度損失係数を0.1から0.02へ減衰します。メッシュ位置拘束、面への強制射影、法線厚み固定は使いません。位置・回転・3軸スケール・SH3の学習と分割・削除を行います。評価用描画も生成形状に由来し、独立した実写の正解ではありません。Unityの動的ライトへの対応は今回の変更に含めていません。

```bash
~/.venvs/world-gs/bin/python -m unittest discover -s tests -v
./run.sh generate --prompt test --output runs/mesh_smoke_new --demo --size 64 --views 8
./run.sh train runs/mesh_smoke_new --steps 220 --max-points 10000
```

`--demo` は合成画像による処理検証用です。現方式は [mesh teacher v8](docs/mesh-teacher-v8.md)、過去の方式は [全周補完 v4](docs/expansion-v4.md) と [mesh v3検証](docs/mesh-v3.md)、先行研究は [研究整理](docs/research.md) を参照してください。生成データ・モデル・論文PDFは通常Gitに含めません。
