# Coverage repair v14 (2026-10-30)

## Problem and resulting behavior

The fitted front camera (~38 degree FOV) had inadvertently become the shared camera for the entire room. Whole-room supervision now uses 75 degree cameras; the source-aligned front view retains its fitted intrinsics. RGB rendering, mesh depth, point coloring, GS training, heldout rendering and the front audit all use each frame's own intrinsics, with a legacy global fallback.

`repair-references` rebuilds supervision in a separate directory while preserving the exact mesh. It keeps the original heldout poses and intrinsics for paired comparison, except views that must become training observations. Compare matching target/pose/K, not frame indices after renumbering.

## Coverage and training

Before the physical render, a 64x64 ray-triangle visibility probe tests heldout surface samples against training views with depth occlusion checks. A heldout camera with >1% unobserved samples is promoted to training. Its image is no longer evaluation data. At least one heldout view must remain. This is a coarse engineering check, not guaranteed continuous or multi-view coverage of every mesh surface. It does not invent unseen geometry.

Unobserved seed counts are recorded in `object_visibility.json`; unobserved seeds are still excluded from color fitting. The GS spatial budget remains 180k, but reaching it no longer disables future scheduled refinement. The existing depth and soft mesh constraints remain active. `gs_coverage_quality.json` checks target pixels with alpha<0.5: mean<=1%, worst view<=5%. These are initial engineering thresholds, not literature benchmarks. A failing result is saved for inspection and status is marked unsuccessful.

## Run

```bash
./run.sh repair-references runs/image_room_frontfit_v13_final_20261030 --output runs/image_room_coverage_v14_20261030
./run.sh train runs/image_room_coverage_v14_20261030 --steps 2000 --max-points 120000
```

The new run deliberately reuses assets to isolate the known regression. `source_front.png` is the generated input; `front.png` and `references.html` are mesh supervision; `front_render.png`, `heldout_render_*.png` and `scene_unity.ply` are trained GS outputs.

## Research relationship and remaining limitations

- [Text2Room](https://arxiv.org/html/2303.11989): adaptive completion views and depth alignment motivate checking whether heldout removal leaves unobserved surfaces. The implementation here is a simpler split-protection rule, not a reproduction of their completion trajectory algorithm.
- [WorldMesh](https://arxiv.org/html/2603.22972v3): shared scaffold geometry, physical coverage and mesh depth supervision motivate the geometry-consistent references. This change does not add its photorealistic diffusion or generated-image depth-edge rejection.
- [LucidDreamer](https://luciddreamer-cvlab.github.io/): projecting known geometry before inpainting is relevant to later appearance completion; no new inpainting model is introduced by this patch.
- [Fillerbuster](https://ethanweber.me/fillerbuster/): joint multi-view completion could improve unknown backsides but is not integrated here.

The attached HTML contains embedded Gaussian data, seven cameras and a fixed coarse-geometry/color-optimization description. It is a visual target, not reconstruction ground truth. Current assets retain the source image's omissions, inferred backsides and texture defects. Lighting is baked, not Unity relighting. Fixing coverage alone cannot restore a missing window or create unseen furniture.

## Verified run result

On `runs/image_room_coverage_v14_20261030`, 97 training and 17 heldout views were rendered. One coffee-table detail view was promoted to training because the coarse probe found >1% of its surface samples unobserved. The mesh digest is unchanged. All 114 RGB-D projections passed teacher validation. Front visible-object silhouette IoU stayed 0.654349.

GS training completed 2000 steps with 120000 initial / 180000 final Gaussians. Against the same 17 heldout poses and intrinsics as v13, mean target pixels with alpha<0.5 decreased from 5.36882% to 0.004548%; worst-view missing fraction decreased from 30.1731% to 0.037299%. This comparison changes both supervision and training/refinement, so it does not isolate each change's individual contribution. It evaluates teacher coverage, not independent geometry accuracy or photorealism. All 103 unit tests passed.

Use `tools/compare_coverage.py BEFORE AFTER --output DIRECTORY` to reproduce paired coverage metrics and actual GS image comparisons. The original v13 result remains unchanged.
