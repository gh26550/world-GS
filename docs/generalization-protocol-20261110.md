# 汎用性を優先する検証手順

入力はプロンプトのみ。初期画像→画像に基づく統合メッシュ→カメラごとの多視点画像→3DGSという構成を維持する。

## 固定するもの

- 画像512×512、追加233カメラ、初期画像を含むRGB教師200枚を目標とする。
- GSは12,000ステップ・初期180,000点。
- 全入力で同じ配置調整、教師検査、再試行回数、停止条件を使用する。
- 特定の物体IDの修復、部屋別の合格閾値変更、失敗した教師の無条件採用は行わない。
- コードとプロンプト群をprotocol.jsonに記録。途中変更は別実験として扱う。

## 段階を分ける

1. 計画段階の比較：ホテル・リビング・オフィスで、構造化出力、寸法、配置、要求の自己監査が成立するかを比較。失敗、処理時間、PyTorchが確保したGPUメモリ最大値を保存。
2. 新しいホテル入力で、初期画像からGSまで共通経路を実行。合格基準をそのシーンに合わせて緩めない。
3. 方法を固定した後、未使用のプロンプトで部屋全体の生成を評価。初期画像の構図との一致、多視点の配置保持、穴、歪み、成功率、計算コストを確認する。

計画段階の成功は、完成GSの品質や汎用性の証明ではない。自己監査は独立した正解との比較でもない。少数のプロンプト群は開発用検証であり、幅広い汎用性を主張するには別の評価入力が必要。

## 現時点の事実

手修正なしのホテル生成hotel_generic_clean_20261110は、Qwen2.5-3Bの計画出力で奥行き0や不正JSONが生じ、4回の固定再試行後に停止した。GS本学習には到達していない。この失敗は削除せず残す。

計画候補Qwen2.5-7B-InstructはHugging FaceからPCへ取得済み。4bit推論用bitsandbytesを追加済み。現在3Bと7B-4bitを固定した3種類の入力で比較している。候補取得は最適性の確認や採用完了を意味しない。

公式情報：[Qwenモデル](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[4bit推論](https://huggingface.co/docs/transformers/v4.57.0/quantization/bitsandbytes)。

Ubuntu比較出力：/home/dpc7/world GS/runs/generic_planning_comparison_v1_20261110

再現コマンド：

```bash
cd '/home/dpc7/world GS'
./run.sh generate-consistent --prompt '新しい部屋のプロンプト' --output runs/new_room --seed 11010
python -m worldgs.planning_benchmark examples/benchmarks/generic-planning-v1.json --output runs/new_comparison
```
