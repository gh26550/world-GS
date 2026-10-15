# World-GS

プロンプト → 正面画像 → 深度グリッドの簡易メッシュ群 → 360度参照画像・点群 → 3D Gaussian Splatting の研究用プロトタイプ。

形状改善版の既定モード `room` は、正面画像の家具をGrounding DINOで矩形検出し、閉じた部屋＋家具の粗い箱メッシュを作ります。その共有形状から深度を計算し、ControlNet Depth付き補完・面ごとのテクスチャ共有・移動と上下の視点・深度損失と面への拘束を使用します。単眼画像からの正確な室形状や家具の復元ではなく、寸法を仮定した室内用の構造事前分布です。旧方式は `--mode legacy` で利用できます。

床は正面画像の左下パッチを反復した材質、天井は上部中央の代表色を初期値にします。未観測領域の外観の仮定であり、実際の床や天井を復元したものではありません。`--views` は水平視点の数です。12なら上下を含め24枚＋評価用4視点、既定24なら48枚＋評価用4視点を生成します。比較用の `--front-image` は既存正面画像を再利用する任意指定で、通常はプロンプトから正面画像を生成します。

## セットアップ (Ubuntu / WSL)

Python 3.10〜3.12、CUDA toolkit（12.8推奨）とC++コンパイラが必要です。プロジェクトは `world GS` に保存できますが、**仮想環境は空白を含まない `~/.venvs/world-gs` に作ります**。プロジェクト内で `python -m venv .venv` を実行すると、gsplatのリンク処理が空白で分断される場合があります。既存の `.venv` への再作成はPythonの混在も招くため行わないでください。

```bash
cd '/home/dpc7/world GS'
bash setup.sh
./run.sh generate --prompt 'A cozy living room with a sofa, bookshelves and a wooden table, wide angle interior photograph' --output runs/livingroom --seed 42
./run.sh train runs/livingroom --steps 1000
```

初回はHugging Faceからモデルを取得します。モデル利用条件を各配布元で確認してください。画像生成にはSD1.5、補完にはSD1.5 Inpainting、深度にはDepth Anything V2 Smallを使用します。モデルは段階ごとに解放し、VRAM使用量を抑えます。キャッシュ済みモデルがあることは全モデルが利用可能なことを意味しません。

## 軽量な動作確認

`setup.sh` はPyTorch 2.7.1/cu128などを固定し、`run.sh` は仮想環境を明示して起動します。学習時は実GPUからCUDAアーキテクチャを検出するので、古い `TORCH_CUDA_ARCH_LIST=8.9` は引き継ぎません。RTX 5070 Tiでは12.0になります。意図的な上書きは `WORLDGS_CUDA_ARCH`、環境の変更は `WORLDGS_VENV`、初回作成のPython指定は `PYTHON_BIN` を使用します。

```bash
~/.venvs/world-gs/bin/python -m unittest discover -s tests -v
./run.sh generate --prompt test --output runs/smoke --demo --size 64 --views 8
./run.sh train runs/smoke --steps 20 --max-points 2000
```

`--demo` は合成色画像・一定深度を使う統合検証専用です。プロンプトの意味を反映したシーン生成の成功例として扱わないでください。

画像生成が済み `runs/livingroom/scene.json` と `points.npz` が存在する場合は、セットアップ後に `./run.sh train runs/livingroom --steps 1000` だけ実行できます。画像を作り直す必要はありません。`cannot find GS/.venv/.../torch/lib` は空白を含む仮想環境が原因です。環境を外へ新規作成し、古い `.venv` は削除せず保持できます。単に元の `cd` の引用符を増やしても、生成されたリンカ引数は修正されません。

## 出力

- `front.png`, `front_depth.npy`: 正面画像と相対深度
- `proxy_front.obj`: 正面から見た深度面。`proxy_scene.obj`: 部屋＋家具の共有メッシュ（roomモード）
- `meshes/room.obj`, `meshes/object_*.obj`, `layout.json`: 個別の部屋・家具メッシュ、ラベルと仮定した寸法
- `guide_*.png`, `mask_*.png`: 既存形状の描画と未観測領域
- `view_*.png`, `depth_*.npy`, `scene.json`: 参照画像、深度、OpenCV形式camera-to-world姿勢、内部パラメータ
- `points.npz`: 重複を抑制した初期点群
- `scene.ply`: Unity互換のbinary little-endian float32 PLY。最適化したSH0 Gaussian、標準のlog scale・opacity logit形式。高次SHは0で補完
- `scene_unity.ply`: `scene.ply`と同内容のUnity取り込み用ファイル
- `checkpoint.pt`, `training.json`, `train_preview.png`: 学習結果と検証用出力
- `heldout_preview.png`, `evaluation.json`: 学習に使わなかったカメラでの共有メッシュとの整合評価。実物の正解データによる評価ではない

保存先は空のディレクトリが必要です。生成データ・モデル・論文PDFはGitに含めません。

## Unityで表示

初期版のASCII PLYはGaussian Splat Creatorでエラーになるため、[Unity取り込み手順](docs/unity.md)の変換コマンドを使用してください。既存結果は再学習不要です。このPCでは `runs/livingroom/scene_unity.ply` と `runs/smoke/scene_unity.ply` を作成済みです。今後の学習はバイナリPLYを出力します。

## 現在の制約と開発順序

roomモードは6×3×6の箱状の部屋を既定の事前形状とし、`--room-size WIDTH HEIGHT LENGTH` で変更できます。寸法は実測値ではありません。家具は画像の矩形検出と床接触の仮定から作る箱で、背面・曲面・細い脚などの正確な復元はできません。上下の補完では一般的な室内の天井・床を仮定するため、特殊な天井や屋外向けではありません。見えない物体面のテクスチャや生成画像の模様の歪みは残ります。

roomモードの学習はRGB L1＋期待深度のlog L1＋位置正則化を使い、点を割り当てた面上に射影します。回転は軸平行、法線方向の厚みは固定、接線方向の広がりにも上限を設けます。共有メッシュの誤りをGSが自由に直す方式ではありません。固定点数・SH0でdensificationやSSIMは未実装です。legacyモードには従来の回転のみ・相対深度・RGB中心の制約が残ります。

次の段階: 部屋寸法とカメラ内部パラメータの推定、家具の実形状への置換、背面の完成、テクスチャの継ぎ目や模様の歪みの改善、独立した参照データによる移動視点の評価。

先行研究と提案の位置づけは [docs/research.md](docs/research.md) を参照してください。

実プロンプトでの初回生成、GPU学習、出力PLY、目視での問題点は [docs/validation.md](docs/validation.md) に記録しています。

Unity表示時の形状歪みについて、保存データから調べた原因・数値・学習前後の比較図・メッシュ保存場所は [docs/geometry-diagnosis.md](docs/geometry-diagnosis.md) を参照してください。

その原因への改善実装と新しいUnity用出力、比較結果は [docs/geometry-v2.md](docs/geometry-v2.md) を参照してください。
