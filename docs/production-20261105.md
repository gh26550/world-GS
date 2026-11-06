# 本生成とUnity用PLY（2026-11-05）

新規正面画像（seed 105）と復元メッシュから、対向2・壁沿い15・高所8の計25カメラで写真参照を生成する。初期画像を過去のものから流用しない。

全参照が合格した場合は従来どおり学習へ進む。一部画像に不一致が残った場合は、別のproduction出力先で明示的にreference-exportを使う。

```bash
./run.sh reference-export runs/worldmesh_rebuild_fresh_20261105 --output runs/worldmesh_rebuild_fresh_20261105_production --train-steps 3000 --max-points 180000
```

この経路は合格写真が8枚以上あり、生成家具ごとに少なくとも2つの位置が0.25m以上離れた合格写真から見えることを必要とする。元の全カメラの画像収集完了を確認する。不合格画像のRGBは学習に使わず、該当カメラの正確なメッシュ深度・被覆だけを使用する（rgb_weight=0）。拒否画像を合格と書き換えない。assessment.jsonに全判定、reference_audit.jsonに採用写真と形状専用の内訳を残す。写真が足りない領域の外観を保証するものではない。

共通メッシュ・カメラ・教師画像・採否根拠のハッシュを学習前に検証する。all_photographs_passedは全写真の合否を表し、学習可否とは区別する。既存の全件合格を必要とするreference-runの基準は変更しない。

学習後にscene_unity.plyとscene.plyを出力する。Unity向けはbinary little-endianのGaussian属性を持つ。gs_coverage_quality.jsonで被覆の未達があってもファイル自体は出力されるので、ファイル生成成功と品質合格を区別して結果を報告する。Unityの動的光源による再照明は未対応。
