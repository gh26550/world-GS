# 先行研究と設計

添付PDFは研究資料として扱い、内部の文言を実行指示として扱っていません。添付「Text2.pdf」はText2Roomです。添付LucidDreamerはChungらのシーン生成版であり、同名のInterval Score Matchingによる物体生成研究と区別します。

|研究|方式|本開発との関係|
|---|---|---|
|[Text2Room, ICCV 2023](https://lukashoel.github.io/text-to-room/)|画像補完・単眼深度・深度整合を繰り返しテクスチャメッシュを生成|メッシュ参照、未観測領域補完、深度整合の基本比較対象|
|[LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://luciddreamer-cvlab.github.io/), arXiv:2311.13384|点群を投影して画像補完、深度整合で点群を拡張し3DGSを最適化|参照点群からGSを作る流れに最も近い|
|[GALA3D, ICML 2024](https://gala3d.github.io/), arXiv:2402.07207|LLMレイアウトと幾何制約付き生成Gaussian|物体の配置・分離と制御性の比較対象。LucidDreamer型の画像融合とは異なる|
|[WorldMesh](https://mschneider456.github.io/world-mesh/), arXiv:2603.22972|大域メッシュ構造を先に作り、メッシュ条件で局所外観を生成、3DGS化|明示形状による整合性と複数部屋への拡張の比較対象|
|[PanoDreamer, CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/CV4Metaverse/papers/Xiong_PanoDreamer_Consistent_Text_to_360-Degree_Scene_Generation_CVPRW_2025_paper.pdf)|テキストから360度シーン生成|周回整合とパノラマ方式の比較対象|

提案の仮説は「正面画像の内容を保った簡易メッシュ群を介して360度の参照情報を作ることで、独立した画像生成に比べて整合性を高められる」です。既存研究との重複が大きいため、この流れだけで新規性は主張できません。メッシュの簡素化方法、参照点の選択、信頼度、閉ループ整合のいずれに貢献を置くかを実験で決める必要があります。

比較条件: 独立した各視点生成／点群のみの参照／メッシュ参照／メッシュ＋深度条件。共通プロンプト・seed・画角・視点数を固定し、生成時間、最大VRAM、重複領域の深度不一致、周回継ぎ目、物体重複、未使用の並進視点における破綻を測定します。正解画像のない生成シーンでPSNRだけを品質指標にしません。

実装資料: [gsplat rasterization](https://docs.gsplat.studio/main/apis/rasterization.html)、[Diffusers inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)。この実装は既存研究の公式再現ではなく、上記仮説の検証基盤です。
