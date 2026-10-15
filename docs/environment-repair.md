# CUDAリンクエラーの修復 (2026-10-15)

## 原因

`world GS/.venv` にPyTorchをインストールすると、拡張ビルドのリンク行に引用符のない `-L/home/dpc7/world GS/.venv/.../torch/lib` が出力される場合がある。シェルは空白で引数を分断し、`cannot find GS/.venv/.../torch/lib` と `cannot find -lc10` などが発生する。ソースの `-I` が引用されていても、リンク行の `-L` は別の引数である。

現地確認では既存 `.venv` の設定がPython 3.12、実体のpythonリンクが3.10となっていた。PyTorchも2.14.1/cu130に変わっており、以前検証した2.7.1/cu128からずれていた。投稿されたビルドログの `sm_89` もRTX 5070 Tiのcompute capability 12.0と異なる。

## 対応

- プロジェクトと生成済みデータは元の場所に保持。
- 仮想環境を空白のない `/home/dpc7/.venvs/world-gs` に新規作成。旧 `.venv` は保持。
- このPCでは動作実績のあるPython 3.10の `worldmesh-nerfstudio` 環境を基に `--system-site-packages` で作成し、空白のない既存PyTorch 2.7.1/cu128とgsplat 1.6.0を再利用。生成用依存は新しい仮想環境内に導入。
- 新規導入向けの `setup.sh` は通常のvenvを作り、PyTorch/cu128のインストールと `constraints.txt` による依存固定を行う。既存のアクティブ環境の `python` / `pip` に依存せず、環境内の実行ファイルを明示する。
- `run.sh` は新しい仮想環境を使用し、学習時は実GPUのcompute capabilityを検出。一般の `TORCH_CUDA_ARCH_LIST` に残った値を上書きする。意図的な変更には `WORLDGS_CUDA_ARCH` を使う。
- 空白を含むPyTorch実体パスはgsplatをimportする前に検出し、環境修復の案内を出す。

## 検証

6件の単体テストとシェル構文検査に成功。既存の `runs/livingroom`（512px・24方向）を再生成せず使用し、以下でGPU学習に成功。

```bash
cd '/home/dpc7/world GS'
./run.sh train runs/livingroom --steps 1000
```

出力 `runs/livingroom/scene.ply` は97,628個のGaussian、約14MB。PLYの97,628行×17属性を読み直し、全数値が有限であることを確認。最終ステップの対象画像上のL1は0.0183623。これはシーン全体や未使用視点の品質指標ではない。

学習ログはローカルの `environment-repair-training.log`。画像の生成とモデルの再取得は不要だった。修復結果のデータは利用者のローカル実行結果として保持し、既存のGitHub掲載例は変更していない。
