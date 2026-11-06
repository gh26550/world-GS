# 全周カメラ構成と新しい初期画像（2026-11-05）

## 今回の変更
4視点検証から、WorldMeshを参考にした部屋周囲のカメラへ変更した。通常のroom-localとreference-runに適用する。

- 向かい合う短辺側の壁中央に初期2視点。床から1.6m、壁から0.2mを起点とする。安全のため原実装の0.05mより離す。
- 壁の周長を16等分し、半区間ずらして配置。壁から内側へ0.4m、床から1.6m、中央を見る。初期2視点と視線方向が10度以内の候補は除く。
- 壁沿い8視点を床から2.4mに置き、中央の高さ1.6mへ向ける。低い天井では室内へ制限する。
- 家具の境界箱と室内の安全余白を使ってカメラ衝突を回避する。家具自体は移動しない。
- 初期2視点を先に生成し、通常視点、高所視点の順に処理。各群内は生成方向が既存方向に近い順へ並べる。
- 外観参照は実際に検査を通過した画像の中から、回転が最も近いカメラを選ぶ。ControlNetにはその画像の外観パッチを渡し、画角・位置は当該カメラの概形・深度・境界で指定する。
- 生成済み参照、カメラID、採否根拠、深度、選んだ参照元のハッシュを保存。拒否候補を次の参照として使わない。

カメラ配置と近い回転による順序はWorldMesh由来。矩形室内限定、独自の衝突回避、外観パッチ方式は本システムの実装であり、WorldMeshと完全同一ではない。壁テクスチャへの逐次投影・未観測面による追加カメラは今回未実装。天井や家具背面の観測が十分かは、別途可視性で検証する必要がある。

## 初期画像の変更
前回の正面画像は再利用しない。同じプロンプトをseed 105でローカルFLUX.2 Klein Baseへ入力し、新規画像から画像解析・SAM3D復元・概形補完を再実行する。

プロンプト：A cozy living room with a brown leather sofa, two bookshelves filled with books, a wooden coffee table, a rug, warm daylight

出力：/home/dpc7/world GS/runs/worldmesh_rebuild_fresh_20261105
新規正面画像：兄弟フォルダworldmesh_rebuild_fresh_20261105_front/source.png
基準メッシュ：兄弟フォルダworldmesh_rebuild_fresh_20261105_proxy/scene_mesh.glb
参照一覧：本体フォルダのreview.html

## 再現
```bash
./run.sh generate --mode room-local --prompt 'A cozy living room with a brown leather sofa, two bookshelves filled with books, a wooden coffee table, a rug, warm daylight' --output runs/worldmesh_fresh_new --size 512 --views 16 --seed 105
```
全視点が合格した場合に学習用scene.jsonを出力する。形状検査だけで材質・物体同一性・多視点品質を保証しない。GS学習後にscene_unity.plyが生成される。参照生成中・検査未達では完成PLYはまだ存在しない。

参考：
- https://arxiv.org/html/2603.22972v3
- https://github.com/mschneider456/worldmesh/blob/main/method/render_multiview.py
- https://github.com/mschneider456/worldmesh/blob/main/method/flux_generation/camera_poses.py
