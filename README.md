# World-GS

プロンプト → 正面画像 → 初期メッシュ → 全周の画像補完・共有RGB-D形状の更新 → 多視点参照点 → 3D Gaussian Splatting の研究用プロトタイプ。

既定の `--mode expand` は未観測の側面・背面・上下をControlNet付き画像補完で生成し、推定深度から共有形状へ追加します。後続視点へ既観測領域を再投影し、以前に観測した空間を塞ぐ矛盾した形状を除外します。初期メッシュの単純な描画を最終教師にせず、補完画像と推定深度をGSへ渡します。GSの初期メッシュへの位置拘束は0です。

v6では生成したRGB-Dをそのまま板状メッシュとして追加する最終処理を廃止し、視点間の支持を検査してTSDFで全体を融合します。低信頼の暫定面は融合・学習の重みを下げます。GS初期点は検査に通った最終メッシュの面からのみ採り、不合格メッシュでは学習を停止します。[メッシュの作り方・修正内容・比較](docs/mesh-v6.md)を参照してください。元画像の不整合を解決する方法や、穴のない物体メッシュを生成する方法ではありません。

未観測内容は生成による仮説であり、実物の復元ではありません。単眼深度による歪み、継ぎ目、家具の重複や欠落は残ります。形状を持つRGB-Dパッチの蓄積であり、各家具を閉じた完全なメッシュとして復元するものではありません。v3の手続き的家具は `--mode mesh`、v2は `--mode room`、初期方式は `--mode legacy` で保持しています。

深度・境界の修正版v5では、逆深度の倍率とオフセットを外れ値に頑健に合わせ、物体境界をまたがない深度補正と狭い範囲の色調整を行います。重なり部分の三角形も利用して接続の隙間を減らします。詳細は [深度・境界修正](docs/depth-seams-v5.md) を参照してください。

## セットアップと実行（Ubuntu / WSL）

Python 3.10〜3.12、CUDA toolkitとC++コンパイラが必要です。仮想環境は空白のない `~/.venvs/world-gs` に作ります。

```bash
cd '/home/dpc7/world GS'
bash setup.sh
./run.sh generate --prompt 'A cozy living room with a sofa, bookshelves and a wooden table, wide angle interior photograph' --output runs/livingroom_expand_new --mode expand --size 384 --views 12 --seed 42
./run.sh train runs/livingroom_expand_new --steps 2000 --max-points 60000
```

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

保存先には空のディレクトリが必要です。比較で既存画像を使う場合のみ `--front-image runs/livingroom/front.png` を指定します。通常はプロンプトから生成します。初回はモデル取得とCUDAビルドに時間がかかります。`run.sh` は外部環境とGPUアーキテクチャを設定します。環境変更は `WORLDGS_VENV`、GPU指定は `WORLDGS_CUDA_ARCH` を使用します。空白のある `.venv` のリンクエラーは引用符を増やしても解消しません。

## 出力（expandモード）

- `front.png`, `front_relative_depth.npy`: 最初の正面画像と相対深度。
- `initial_scaffold.glb`: 制御用の初期メッシュ。最終教師ではありません。
- `scene_mesh.glb`, `proxy_scene.obj`, `meshes/generated_patch_*.glb`: 生成RGB-Dで更新した全体形状と追加パッチ。
- v6の最終 `scene_mesh.glb` は融合メッシュです。新しい生成・深度修正では `observations/` に補完処理の中間データを保存します。中間パッチはGSの最終メッシュではありません。
- `mesh_quality.json`, `support_report.json`, `fusion_holdout.json`: 面の品質、視点の支持、融合から除外した視点での確認。
- `view_confidence_*.npy`: 高信頼1.0・暫定0.25・無効0の学習重み。
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
./run.sh generate --prompt test --output runs/expand_smoke_new --demo --size 64 --views 12
./run.sh train runs/expand_smoke_new --steps 220 --max-points 3000
```

`--demo` は合成画像による処理検証用です。新方式は [全周補完 v4](docs/expansion-v4.md)、旧方式は [mesh v3検証](docs/mesh-v3.md)、先行研究は [研究整理](docs/research.md) を参照してください。生成データ・モデル・論文PDFは通常Gitに含めません。
