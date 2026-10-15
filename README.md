# World-GS

プロンプト → 正面画像 → 部品メッシュ群 → 周回カメラによる参照画像・深度・点群 → 3D Gaussian Splatting の研究用プロトタイプ。

既定の `--mode mesh` は正面画像の家具検出から、ソファの座面・背・肘・脚、テーブルの天板・脚、本棚の棚板・本を持つ近似メッシュを作ります。実際の三角形との交差で遮蔽を計算し、部屋と家具を異なる位置から周回して参照データを取得します。

形状は仮定寸法による手続き的な近似であり、画像から正確な家具形状を復元するモデルは未導入です。材質は代表色と簡単な陰影・模様を用いるため、写真品質の再現には至っていません。旧方式は `--mode room`、初期方式は `--mode legacy` で保持しています。

## セットアップと実行（Ubuntu / WSL）

Python 3.10〜3.12、CUDA toolkitとC++コンパイラが必要です。仮想環境は空白のない `~/.venvs/world-gs` に作ります。

```bash
cd '/home/dpc7/world GS'
bash setup.sh
./run.sh generate --prompt 'A cozy living room with a sofa, bookshelves and a wooden table, wide angle interior photograph' --output runs/livingroom_mesh_new --mode mesh --size 384 --views 12 --seed 42
./run.sh train runs/livingroom_mesh_new --steps 2000 --max-points 60000
```

保存先には空のディレクトリが必要です。比較で既存画像を使う場合のみ `--front-image runs/livingroom/front.png` を指定します。通常はプロンプトから生成します。初回はモデル取得とCUDAビルドに時間がかかります。`run.sh` は外部環境とGPUアーキテクチャを設定します。環境変更は `WORLDGS_VENV`、GPU指定は `WORLDGS_CUDA_ARCH` を使用します。空白のある `.venv` のリンクエラーは引用符を増やしても解消しません。

## 出力（meshモード）

- `front.png`, `front_relative_depth.npy`: 正面画像と相対深度。
- `scene_mesh.glb`: 部品と材質を含むメッシュ。
- `proxy_scene.obj`, `meshes/room.obj`, `meshes/object_*.obj`, `layout.json`: 全体・家具別メッシュと配置情報。
- `view_*.png`, `view_depth_*.npy`, `view_mask_*.npy`: 学習用の描画・深度・有効マスク。
- `heldout_*.png`, `scene.json`, `references.jpg`: 評価用描画、全カメラ姿勢と内部パラメータ、参照一覧。
- `points.npz`, `mesh_geometry.npz`: 可視表面から融合した初期点群と三角形形状。
- `scene.ply`, `scene_unity.ply`: 同内容のbinary little-endian Gaussian PLY。位置・回転・3軸スケール・不透明度・SH3を保持。
- `gaussians.npz`, `training.json`: 学習パラメータと記録。
- `heldout_preview.png`, `evaluation.json`: 未学習視点でのメッシュとGSの比較。実物に対する精度評価ではありません。

UnityのGaussian Splat Creatorには `scene_unity.ply` を指定します。GLB/OBJは形状確認用です。[Unity手順](docs/unity.md)も参照してください。

## 検証と制約

meshモードはRGB・深度・柔らかい形状事前分布を使用します。面への強制射影と法線厚みの固定を廃止し、位置・回転・3軸スケール・SH3の学習と分割・削除を導入しました。共有メッシュの誤りは参照画像にも引き継がれます。未観測面の穴や境界の粗さが残ります。

```bash
~/.venvs/world-gs/bin/python -m unittest discover -s tests -v
./run.sh generate --prompt test --output runs/mesh_smoke_new --demo --size 64 --views 8
./run.sh train runs/mesh_smoke_new --steps 220 --max-points 3000
```

`--demo` は合成画像による処理検証用です。今回の実測結果と制約は [mesh v3検証](docs/mesh-v3.md)、先行研究は [研究整理](docs/research.md) を参照してください。生成データ・モデル・論文PDFは通常Gitに含めません。
