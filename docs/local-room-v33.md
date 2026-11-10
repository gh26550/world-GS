# PC内の全周カメラ・深度・参照画像生成（v33）

プロンプト→部屋全体の初期画像→共通メッシュ→カメラごとの画像→GS、という入口に、全周のカメラ計画とローカルAIによる参照生成・検査を接続した。既定は`room-local`。旧`photo-room`は明示指定する互換経路。

## 実装

- 対向壁側2台・周囲16台・上方8台に、未観測面の可視性と入射角に基づく追加を最大8台。今回の計画は34台。家具を動かさずカメラを衝突回避する。
- 各カメラの共通メッシュRGB、深度、物体境界を描画。厚い壁・床・天井の外側の面は、室内カメラの被覆目標から除く。
- 同じカメラの画像全体を生成する。物体を独立にテキストから画像生成する経路は使わない。初期画像から物体領域を切り出してメッシュ復元する処理は残る。
- 採用済み画像の建築面の外観を、遮蔽・境界を除外して共通メッシュへ蓄積し、次のカメラに描画して共有する。元写真を全方向へ貼り付けたり、背景画素を固定したりしない。
- カメラごとに実際に可視な家具・ラグ・窓を求め、見えていない家具や開口を生成させない条件を自動で作る。

## Hugging FaceのモデルをPC内で実行

既定の参照生成は、Stable Diffusion 1.5 inpainting＋[深度ControlNet](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)＋[境界ControlNet](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)＋IP-Adapter。深度だけでは、床とラグのような深度差の小さい領域の配置を拘束できないため、共通メッシュの領域境界も入力する。物体境界は面IDから自動で求める。初期写真は外観パッチにしてIP-Adapterへ渡す。

深度推定は[Depth Pro](https://huggingface.co/apple/DepthPro)。既知のカメラ焦点距離を渡す。配置検査はGrounding DINO。各処理を別プロセスで順番に動かしてGPUメモリを解放する。

FLUX.2 Klein 9B Base/Distilledのローカル比較経路も`--image-model base` / `distilled`で残した。初期画像の生成にはBaseを使用する。試験では汎用画像編集だけで反対視点の配置を維持できない例が出たため、参照生成の既定には、空間条件を専用入力できるControlNetを採用した。全候補の最適性を証明した比較ではなく、配置を優先する実機試験に基づく選択。写実性の向上は引き続き必要。

## 検査

1. 生成後の推定深度の境界を、メッシュ深度と比較する。
2. 概形RGBにも同じ深度推定を行い、推定器が観測可能なメッシュ境界を区別する。推定器自身が読めない境界は、生成画像のRGB輪郭による別検査を必須にする。観測できないことを記録し、深度一致の証拠と混同しない。
3. 家具・ラグ・窓の位置、大きさ、数を検査する。低テクスチャで深度が平面に一致する建築面の分類器の誤認識は、構造面の曖昧な認識として記録する。判定は完全な意味・形状の証明ではない。
4. 最大3回再生成。不合格が残れば停止してGSへ進めない。目視による合格上書きや、失敗画像の仮RGBへの自動置換はしない。

生画像、生成指示、ガイド、描画深度、推定深度、境界、概形の推定深度、個別判定とハッシュを保存する。`--resume`は同一メッシュ・初期画像・モデル・カメラ計画の場合のみ。GS学習前に画像と検査記録の改変を検査する。

## このPCでの実行

```bash
cd '/home/dpc7/world GS'
# 計画とガイドを確認（AI生成なし）
./run.sh generate-room-views runs/dual_input_v24_scaffold_20261104 --output runs/new_local_room --prepare-only
# 全周の参照画像を生成・検査し、その後GS学習
./run.sh generate-room-views runs/dual_input_v24_scaffold_20261104 --output runs/new_local_room --resume --train-steps 3000
```

プロンプトから始める場合：

```bash
./run.sh generate --mode room-local --prompt 'A cozy living room with a brown leather sofa, two bookshelves filled with books, a wooden coffee table, a rug, warm daylight' --output runs/new_prompt_room --views 16 --seed 84
./run.sh train runs/new_prompt_room --steps 3000 --max-points 180000
```

`--pilot-views 2`は参照画像の試験だけで停止する。全周データとして学習しない。本生成と品質評価は、この試験とは別の処理。

環境：`~/.venvs/world-gs`。既存の`~/worldmesh-main-optimized`、ComfyUI/FLUX、Depth Pro、SAM、Blenderを利用する。画像生成サービスのAPIキーは不要。ControlNetはPC内のHFキャッシュから読み込む。FLUX比較経路のサーバーは所有するプロセスだけ127.0.0.1へ起動し、終了時に閉じる。別PCではpipに加えてバックエンドとモデルの導入が必要で、起動前に検査する。

`WORLDGS_LOCAL_BACKEND`、`WORLDGS_COMFY_PYTHON`、`WORLDGS_DEPTH_PYTHON`で場所を変更可能。モデルごとの利用条件は公開元に従う。

## 研究上の制限

カメラ・参照生成・検査・GSへの接続を実装したもので、未知の家具や建築形状を初期画像から完全に復元することを意味しない。画像に現れない領域は共通メッシュの仮説に依存する。写実性、生成画像間の照明・局所形状の一致、検出器と深度推定器の誤りは残る。現在のメッシュ復元の限界やUnityの動的照明は解消していない。

[WorldMesh原論文](https://arxiv.org/html/2603.22972v3)のカメラごとのメッシュ深度・生成後検査・逐次的な外観共有を参考にした。初期画像を起点にする入口、追加カメラ、境界制御、観測可能性と配置の検査は本実装の設計であり、論文の完全再現ではない。
## 今回の動作確認

- 全周カメラ34台と各カメラのRGB・深度・境界ガイドを作成。
- 対向する2視点について、PC内での画像生成、Depth Pro、輪郭検査、家具・ラグ・窓の配置検査を実行し採用。
- 自動テスト158件通過。
- 検査済み2視点を使う縮小データで、参照の検証・GS学習100ステップ・20,000 GaussiansのUnity用PLY出力まで接続を確認。これは2視点の接続試験で、34視点の本生成や完成シーンではない。
- 実機試験の保存先：`runs/local_room_v33_20261104`。接続試験：`runs/_tests/local_room_v33_gs_connection`。
- ControlNet経路の画質・小物の情報量は、前回のチャット内画像生成器による結果には届いていない。配置制御とローカル実行の動作確認であり、画質の達成を主張しない。
