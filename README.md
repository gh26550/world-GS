# World-GS

プロンプト → 正面画像 → 深度グリッドの簡易メッシュ群 → 360度参照画像・点群 → 3D Gaussian Splatting の研究用プロトタイプ。

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
- `proxy_front.obj`, `proxy_scene.obj`: 頂点色付き簡易メッシュ（物体単位ではなく深度パッチ）
- `guide_*.png`, `mask_*.png`: 既存形状の描画と未観測領域
- `view_*.png`, `depth_*.npy`, `scene.json`: 参照画像、深度、OpenCV形式camera-to-world姿勢、内部パラメータ
- `points.npz`: 重複を抑制した初期点群
- `scene.ply`: Unity互換のbinary little-endian float32 PLY。最適化したSH0 Gaussian、標準のlog scale・opacity logit形式。高次SHは0で補完
- `checkpoint.pt`, `training.json`, `train_preview.png`: 学習結果と検証用出力

保存先は空のディレクトリが必要です。生成データ・モデル・論文PDFはGitに含めません。

## Unityで表示

初期版のASCII PLYはGaussian Splat Creatorでエラーになるため、[Unity取り込み手順](docs/unity.md)の変換コマンドを使用してください。既存結果は再学習不要です。このPCでは `runs/livingroom/scene_unity.ply` と `runs/smoke/scene_unity.ply` を作成済みです。今後の学習はバイナリPLYを出力します。

## 現在の制約と開発順序

正面画像の単眼深度は相対スケールです。背面形状と遮蔽部分は推測になります。カメラは原点で旋回するため360度の見回しを対象とし、自由歩行の品質は保証しません。初期メッシュはセマンティックな物体群ではありません。独立した新視点生成よりも既存画素保存と重複深度整合を優先していますが、補完モデルへの深度条件入力は未実装です。周回終端で既存形状を参照しますが、全体のloop closure最適化はありません。

3DGS学習は固定点数・SH0・L1と位置正則化によるベースラインです。densification、SSIM、深度損失は未実装です。学習画像の損失は独立評価を代替しません。

次の段階: (1) 実プロンプトでの生成と目視評価、(2) セグメンテーションと箱・平面・物体メッシュ化、(3) 深度条件付き補完、(4) 小移動カメラと閉ループ整合、(5) densification・深度正則化、(6) 未使用視点の画像品質・幾何整合性とVRAM/時間の比較実験。

先行研究と提案の位置づけは [docs/research.md](docs/research.md) を参照してください。

実プロンプトでの初回生成、GPU学習、出力PLY、目視での問題点は [docs/validation.md](docs/validation.md) に記録しています。
