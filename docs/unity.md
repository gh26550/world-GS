# Unity Gaussian Splat Creatorへの取り込み

## PLY形式の修正

初期版はASCII PLYを出力していたため、Gaussian Splat Creatorで `needs to be binary, little endian PLY format` と表示される。今後の `worldgs train` / `./run.sh train` は `binary_little_endian 1.0` のfloat32 PLYを出力する。

SH0の色、位置、log scale、opacity logit、wxyz quaternionは変更しない。互換性のため標準の `f_rest_0`〜`f_rest_44` も書き、学習していない高次SHは0で埋める。合計62属性、1 Gaussianあたり248バイト。

## このPCで変換済みのファイル

実際の部屋:

```text
\\wsl.localhost\Ubuntu-22.04\home\dpc7\world GS\runs\livingroom\scene_unity.ply
```

動作確認用の合成データ:

```text
\\wsl.localhost\Ubuntu-22.04\home\dpc7\world GS\runs\smoke\scene_unity.ply
```

元の `scene.ply` は保持。**学習をやり直す必要はない**。

## Unityの設定

1. Gaussian Splat Creatorの `Input PLY/SPZ File` に **scene_unity.ply** を選び直す。同じパスの選択が続く場合、以前のエラー表示が残ることがあるためファイルを選び直す。
2. `Import Cameras` はオフ（このパイプラインはUnityインポーター用 `cameras.json` を出力していない）。
3. `Output Folder` はUnityプロジェクト内の `Assets/GaussianAssets/WorldGS` などを指定。
4. `Create Asset` を実行し、生成されたGaussianSplatAssetをGaussianSplatRendererのAsset欄へ設定。

この修正ではUnityエディターのUI操作までは検証していない。確認したのは下記公式実装の入力条件、出力バイト列と元パラメータの一致、破損データの検出。

## 他の既存PLYを変換

```bash
cd '/home/dpc7/world GS'
~/.venvs/world-gs/bin/python -m worldgs.ply runs/another_scene/scene.ply --output runs/another_scene/scene_unity.ply
```

既存の出力先や元ファイルへの上書きは拒否するため、別の出力名を指定する。追加ライブラリ・GPU・CUDAビルドは不要。対応対象は本プロジェクトのscalar float32 ASCII / binary little-endian Gaussian PLY。

## 2026-10-15の検証

- livingroom: 97,628 Gaussians、smoke: 1,292 Gaussiansを元ファイルを保持して変換。
- バイナリのヘッダー、頂点数、62属性、全レコードのbyte数、既存17属性の一致、追加45属性の0を検証。
- 自動テスト: バイナリのエンディアンと往復、ASCII変換と元ファイル保持、切り詰め検出、NaN入力での既存出力保持。

仕様確認: [UnityGaussianSplatting PLYFileReader](https://github.com/aras-p/UnityGaussianSplatting/blob/main/package/Editor/Utils/PLYFileReader.cs)、[GaussianFileReader](https://github.com/aras-p/UnityGaussianSplatting/blob/main/package/Editor/Utils/GaussianFileReader.cs)。前者はbinary little-endianのみ受け付け、後者はfloat型の位置・SH0色・opacity・scale・rotationを要求する。
