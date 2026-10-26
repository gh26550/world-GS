# ごちゃついた部屋：10物体の生成（2026-10-26）

前回の3物体から独立した10物体へ増やした試行。新しい出力先は `runs/cluttered_livingroom_10objects_20261026`。

## 元プロンプト

```text
A cluttered, lived-in cozy living room with warm cream painted plaster walls, a wooden floor and warm soft lighting. Exactly ten separate objects: one worn fabric sofa, one large wooden coffee table, two tall book-filled bookshelves, one mismatched armchair, two separate closed books lying flat on the coffee table, one small ceramic vase on the coffee table, one ceramic mug on the coffee table, and one cardboard storage box on the floor beside the armchair. Arrange the furniture asymmetrically in a compact conversational area with narrow but passable gaps, bookshelves along different walls and the table in front of the sofa. Spread the four tabletop accessories across the tabletop without overlap. Keep the camera origin and room center clear. Earthy muted colours, worn wood and everyday visual clutter. No other standalone objects, no windows, no doors.
```

## 設定・出力

- 部屋7×3×7m、参照画像384×384、水平分割12、seed 46。
- ソファ、テーブル、本棚2、椅子、本2、花瓶、マグカップ、収納箱の10物体をSAM3Dで生成。
- 4小物をテーブル上へ配置し、実メッシュ支持面の検査に成功。
- 134枚の参照画像（学習112、検証22）。可視性の詳細は `object_visibility.json`。
- GS学習は2,500ステップ、初期上限120,000点、学習中の上限180,000点。
- 学習完了時は180,000ガウス。検証22視点の平均相対深度誤差は約2.09%、RGB L1は約0.0174、不透明度0.5未満の欠落画素率は全検証視点で0%。生成した同じメッシュ参照に対する評価であり、独立した実世界の正解ではない。
- 各物体は22〜55の学習視点から16画素以上見えている。
- `references.html`：画像一覧、`references.jpg`：一覧画像、`scene_mesh.glb`：メッシュ、`scene_unity.ply`：学習後GS。
- `request.json`：正確な元プロンプトと指定条件、`planning/scene_plan.json`：計画器の解釈、`assembly.json`：実際の配置。

## 今回必要になった修正

- 多物体JSONが途中で切れないよう、計画器の最大出力を2,400から6,000トークンへ拡大。
- 個数の指定しかないプロンプトで、推定した本の寸法を厳密指定として誤固定する問題を抑制。寸法単位のあるプロンプトでは固定指定を維持する保守的な処理で、任意の自然言語寸法解釈を保証するものではない。
- 接写・画像端切れのためSAM3で切り出せない物体は、全体構図を優先した追加候補を生成して再試行。元の候補とログは保存する。メッシュの箱への代替や切れ判定の緩和は行わない。
- 回帰テスト81件成功。

## 見た目の限界

今回の配置は家具が部屋の一角に集中する。部屋全体を多数の雑貨で埋める規模には達していない。ソファが一人掛けに近い形状になる、本棚内の蔵書や素材が指定どおりにならないなど、外観・意味の忠実性には限界がある。棚内の本など、生成メッシュ・テクスチャに含まれる細部は独立物体数に含めない。可視性・支持面検査への合格は、全方向の写実性や完全な要求充足を保証しない。
