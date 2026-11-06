# 新しい初期画像からの生成 v21（2026-11-04）

元のプロンプトを維持して初期画像から作り直す実行。初期画像はCodexの内蔵image_genを使用し、概形・参照生成・学習はローカルで実行する。CLIが内蔵image_genを直接呼ぶ実装ではなく、生成画像を既存の--front-imageへ渡す。

ユーザープロンプト:

> A cozy living room with a brown leather sofa, two bookshelves filled with books, a wooden coffee table, a rug, warm daylight

初期画像: `runs/fresh_room_source_20261104/source_front.png`。生成プロンプト・プロバイダ・構図条件は同じフォルダーのsource_generation.json。内蔵ツールの乱数seedは公開されない。後続のメッシュ・参照生成のseedは84。

新画像からソファ・本棚2つ・テーブルの4物体を分離し、すべて新たにSAM3Dでメッシュ化。以前のシーンの家具メッシュを流用していない。

## 今回の修正

- 元画像の家具矩形全体を床・壁の参照から除外せず、対応が取れた家具はSAMの画素マスクで扱う。脚の間に見える床・ラグを失わない。未対応の検出物体領域は引き続き構造面への誤投影から除外する。
- 正面の位置合わせで矩形が大きくずれている物体について、別の粗い向きからの最適化も試す。生成メッシュの縦横比を無理に変形せず、検査基準も緩めない。新画像の右本棚の輪郭IoUは約0.176から0.459へ改善、4物体の平均は約0.578。
- v20の共通座標・可視面・背景共有・家具ごとの深度境界検査とGSの局所形状自由度を継続する。分類器の名称診断は配置検査と区別する。

## 後続処理

```bash
./run.sh generate --mode image-room --prompt 'A cozy living room with a brown leather sofa, two bookshelves filled with books, a wooden coffee table, a rug, warm daylight' --front-image runs/fresh_room_source_20261104/source_front.png --output runs/fresh_room_v20_20261104_scaffold --size 512 --views 24 --seed 84
./run.sh synthesize-appearance runs/fresh_room_v20_20261104_scaffold --output runs/fresh_room_v21_20261104 --size 512
./run.sh train runs/fresh_room_v21_20261104 --steps 3000 --max-points 180000
```

既存出力の再利用には同一要求で--resumeを指定する。実行結果は学習完了後の実GS描画で判断する。初期画像の改善だけで完成3Dの改善を断定しない。
