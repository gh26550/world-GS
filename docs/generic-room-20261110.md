# Scene-independent image-grounded generation

This path creates a new whole-room photograph, reconstructs image-grounded
furniture meshes, aligns their camera/layout to the photograph, then collects
camera-conditioned photographs before fitting GS. Asset crops are observations
of the whole-room photograph, not independently generated object images.

```bash
./run.sh generate-consistent --prompt 'A photorealistic boutique hotel guest room, one king-size bed with a tall beige upholstered headboard, white duvet and two large pillows, two matching dark oak bedside tables, one compact oak writing desk and one upholstered desk chair, a large window with linen curtains, warm cream walls and natural oak floor, soft daylight, restrained elegant hospitality interior. Wide establishing photograph from the entrance, the bed, both bedside tables, desk and chair visible, realistic straight verticals, no people, no text, no extra beds.' --output runs/hotel_new --seed 11010
```

Defaults: 512 square pixels, 233 additional cameras, 200 accepted RGB teachers
including the initial photograph, 12000 steps, 180000 initial points.
Resume the identical request with `--resume`. `--prepare-only` creates the
image-grounded proxy and camera guides without photographic collection/GS.

Changes from the previous room-specific run:

- No correction JSON, named bookshelf rotation, room-width override or fixed
  window-grid count is used. Existing generic source-camera fitting and visible
  silhouette gate run before collection; a failure stops the pipeline.
- Surface cameras target reconstructed furniture of any category, including
  beds/desks/chairs; floor coverings are excluded from furniture orbits.
- Camera ordering grows from neighbouring poses to reduce appearance jumps.
- Categories do not force leather sofas or wooden coffee tables. Appearance
  is conditioned on this scene's initial photograph and prompt.
- Teacher adoption requires layout checking as well as depth checking. This
  also applies to retries/adaptive replacement. Preserved-pixel depth witnesses
  cannot override a failed layout check.
- Detector vocabulary includes the current visible categories.
- Invalid nonpositive dimensions are reported before placement optimization;
  planner retries receive the concrete error without repeating invalid JSON.
- Evaluation can create translated object-facing cameras for the new scene;
  it does not reuse the living room's named furniture or camera poses.

Artifacts: source_front.png, scene_mesh.glb, front_silhouette_audit.json
(proxy), experiment.json, references/*/decision.json, assessment.json,
scene_unity.ply, front_render.png, independent_evaluation.json, comparison.html.
The generation_result.json marker is written only after training/evaluation.

Limitations: single rectangular room, inferred metric scale/hidden surfaces,
fallible segmentation and detection, box/silhouette alignment rather than exact
shape recovery, monocular depth checks, lighting baked into appearance. A new
hotel example alone is not evidence of universal prompt generalization. The
200-RGB target is not guaranteed; semantic failures are retained and can stop
training. Initial prompt adherence still needs visual review.

Run started 2026-11-10: `runs/hotel_generic_20261110`. See its preparation and
finish logs for current progress. Do not interpret this start record as a
completed generation result.

## Generalization boundary after the hotel diagnostic

The named-object diagnostic (`hotel_generic_v2_20261110`) is excluded from
generalization evidence. It did not produce a completed GS. The proposed
category-specific mask-hole fill was removed from the common path. A fresh
`hotel_generic_clean_20261110` run uses the public common generator with no
mask edits or previous asset reuse. Its completion must be checked separately.

The common path now refines all observed meshes using occlusion-aware visible
silhouettes with the same objective, iteration budget and constraints. Rigid
pose and uniform scale are fitted per input; furniture shape is not deformed.
A source-image gate runs before expensive reference collection. SAM input
image/mask hashes are recorded and checked to detect accidental manual edits.
The generic preparation path omits duplicate temporary Cycles teacher renders,
preserves the planned floor material and omits fabricated outdoor branches.

These are shared algorithm changes, not evidence of general performance. Freeze
settings before comparing multiple prompt families, report failures/time as
well as visual fidelity, and reserve unseen prompts for evaluation.
