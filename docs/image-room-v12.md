# 正面の部屋画像を起点にする構成（v12）

2026-10-30追記：現在の既定経路には[正面画像への再投影整合と学習前検査（v13）](front-alignment-v13.md)を追加している。以下は最初の画像起点構成とその検証記録。

既定の `generate` を `--mode image-room` に変更した。物体ごとの独立したText-to-Imageではなく、最初に部屋全体の画像を作り、その画像を解析して全周の構成を推定する。

## 処理

1. プロンプトから部屋全体の `source_front.png` を生成する。比較用に `--front-image` で既存画像も指定できる。
2. Qwen2.5-VL-3Bが画像を読んで外観と位置関係を記述し、その観察情報と元の要求を分けてQwen2.5-3Bへ渡す。後者が室内の寸法・材質と全体の配置・支持関係をJSON化する。物体の画像領域はGrounding DINOで検出し、画像上の左右位置を配置の初期値に反映する。見えない物体は `evidence=inferred` として区別する。寸法は推定である。
3. 仮の形状を深度ガイドとして、90・180・270度の**部屋全体の画像**をControlNet Inpaintingで補完する。正面から再投影できる画素を保持する。仮形状は画像補完専用で、最終家具として出力しない。
4. 正面に見える物体は正面画像から切り出す。見えない側に仮定した物体は、補完した室内画像で再度位置を確認して切り出す。見つからなければ停止し、独立した物体画像生成に切り替えない。
5. 画像領域をSAM3で切り出し、SAM3Dで立体のUV付きメッシュを復元する。これらと床・壁・天井を組み合わせて、全視点の基準メッシュを作る。
6. 共通メッシュから多位置・全周のRGB、深度、法線、面IDを描画し、面上の参照点を採る。そのデータで3DGSを学習する。

正面の単眼深度は画像補完の再投影にだけ用いる。補完画像から深度をそれぞれ推定してGS教師にすることはない。GS用の深度は最後の共通メッシュへのレイ交差から取得する。

## 実行

```bash
cd '/home/dpc7/world GS'
./run.sh generate --prompt 'A cozy living room with one brown leather sofa, a wooden coffee table and two white bookshelves filled with books, warm daylight, wide angle interior photograph' --output runs/image_room_new --size 384 --views 12 --seed 52
./run.sh train runs/image_room_new --steps 2000 --max-points 120000
```

初回は画像を扱うモデル `Qwen/Qwen2.5-VL-3B-Instruct` を追加取得する。`--planner-model` はこのモードではQwen2.5-VL互換モデルを指す。従来方式は `--mode learned` で実行できる。既存のlearned実行を再開するときは、同じ引数に **`--mode learned --resume`** を明示する。

## 保存先

| ファイル | 意味 |
|---|---|
| `source_front.png` | 最初に生成した部屋全体の正面画像 |
| `planning/image_room_plan.json` | 正面画像を読んだ全周の配置仮説 |
| `room_images/front.png`, `yaw_*.png` | 正面と側面・背面の補完画像。GS教師そのものではない |
| `room_images/guide_*.png`, `completion.json` | 補完時の形状ガイドと推測領域の割合 |
| `assets/<ID>/source.json` | 切り出し元画像・領域・SHA256・観測か推測か |
| `assets/<ID>/reference_asset.png` | 部屋画像からの切り出し。独立した物体Text-to-Imageではない |
| `assets/<ID>/generated.glb`, `placed.glb` | 復元した物体メッシュ、配置後メッシュ |
| `scene_mesh.glb` | 全体の共通メッシュ |
| `view_*.png`, `references.html` | メッシュから描画したGS学習用参照 |
| `points.npz`, `scene_unity.ply` | 面上の参照点、学習後のUnity用GS |
| `front.png`, `front_render.png` | メッシュからの正面参照、学習後GSの正面描画 |

## 制約

背面は1枚から観測できず、補完結果は仮説である。配置推定や画像の領域指定をVLMが誤る場合がある。現在の補完は3方向の離散的な候補画像であり、同時生成のパノラマモデルや多視点拡散モデルではない。共通の仮形状で制約しても、補完画像同士の外観一致を保証しない。

単体メッシュの隠れた面もSAM3Dの推測である。物体メッシュの寸法・配置の一致を画像へ厳密に再投影最適化する処理はまだない。室内は矩形、背面の矩形窓1つまで。材質は正面画像の小領域を採ったもので、遠近や家具の写り込み、焼き込み照明を含む場合がある。GSも動的再照明には対応していない。

画像の指示文はデータとして扱う。モデル出力はJSONのみとし、コード・シェルとして実行しない。

モデルの公式資料：[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[SAM3D](https://github.com/facebookresearch/sam-3d-objects)。

## 試験実行（2026-10-28）

`runs/image_room_v12_20261028` で、部屋全体の正面画像生成からGS学習まで実行した。プロンプトは上記例、seedは52、画像384px、水平12方向。

- 正面画像からソファ・テーブル・本棚1つを切り出し、正面に検出されなかった本棚1つを側面の補完画像から取得した。計4物体に独立した物体Text-to-Imageの入力は使用していない。
- SAM3・SAM3Dによる4メッシュを組み立て、共通メッシュから88視点（学習75、検証13）を描画した。
- 120,000点から1,000ステップ学習し、180,000ガウスの `scene_unity.ply` を保存した。
- 復元した奥行きのあるソファで中央の下向きカメラ候補が全て塞がったため、安全な移動位置を部屋内の格子から探す処理を追加した。家具を歪めてカメラに合わせる処理ではない。

これは新経路の動作確認であり、元画像を忠実に3D復元できたことを示す実験ではない。ソファの復元後の奥行きと箱による衝突検査により配置が変わり、原画像の構図は保てていない。本棚の色・本の細部も要求や元画像と一致しない。元画像への再投影による配置・カメラ・形状の同時調整、画像に見える窓等の構造の保持、補完画像間の外観整合が残る課題である。

