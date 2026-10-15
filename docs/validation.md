# 初回検証 2026-10-15

WSL Ubuntu 22.04、RTX 5070 Ti 16GBで実行。主要バージョンは `environment-tested.txt` を参照。

## 結果

- 幾何単体テスト4件成功: 座標投影、メッシュ描画、深度スケール整合、深度不連続の面除外。
- 合成データ: 64px・8方向の生成、20ステップのGPU学習、PLY出力が成功。合成データは意味的なプロンプト評価に使わない。
- 実プロンプト: `A cozy living room with a wooden table, bookshelves and a blue sofa, interior photograph, wide angle`、seed 42。
- 正面画像、512px・12方向（30度刻み）の参照画像、簡易メッシュ、相対深度、83,285点の初期点群を生成。
- 全12枚の画像サイズ、全深度の正値・有限値、姿勢の回転行列、点群配列を検証。
- 15,000点・300ステップでは穴が多かった。83,285点・1,500ステップに増やすと、確認した学習視点の穴が大きく減少。
- 最終ステップ対象画像（yaw 330度）のL1: 0.0150354。画像値の範囲は0〜1。全視点平均でも未使用視点の評価でもない。
- 最終PLY: 83,285行×17属性、全数値が有限。SH0、log scale、opacity logit、wxyz quaternion。

```bash
./run.sh generate --prompt 'A cozy living room with a wooden table, bookshelves and a blue sofa, interior photograph, wide angle' --output runs/livingroom_v1 --size 512 --views 12 --seed 42
.venv/bin/python -m worldgs.audit runs/livingroom_v1
./run.sh train runs/livingroom_v1 --steps 1500 --max-points 83285
```

保存済み結果の再生成には新しい出力ディレクトリを指定する。初回gsplatビルドには約8分を要した。生成速度・学習速度・最大VRAMは計測条件を揃えていないためベンチマークとして掲載しない。

## 目視で確認した制約

参照画像には継ぎ目のぼけ、引き伸ばし、複数のソファが生成される傾向がある。点数と学習を増やしても、入力参照の意味的・幾何的な不整合は解決しない。並進視点での品質や実寸は未評価。正面画像から物体を分離した簡単な箱・平面・物体メッシュ群の構築も未実装であり、現在は深度パッチの集合で代用している。

次の優先課題は、深度条件付き補完、物体単位の配置制約、局所的な深度整合から閉ループ整合への拡張、未使用の並進視点での評価。

## 保存例

- [正面画像](examples/front.png)
- [12方向の参照画像一覧](examples/references.jpg)
- [GS描画例（学習視点）](examples/train_preview.png)
- [Gaussian PLY](examples/scene.ply)
- [学習記録](examples/training.json)
- [データ検証記録](examples/audit.json)

大きな中間データとモデルはWSLの `runs/livingroom_v1` とモデルキャッシュに保持し、GitHubにはソース・研究資料・上記の選択した生成例を保存。
