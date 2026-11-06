import argparse


def main():
    parser = argparse.ArgumentParser(description='World-GS experimental scene pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect=sub.add_parser('inspect-front',help='Read-only comparison of initial image and integrated mesh from the saved source camera')
    inspect.add_argument('path')
    inspect.add_argument('--output',required=True)
    inspect.add_argument('--orientation-probes',action='store_true',help='Render diagnostic 180-degree bookcase rotations without changing the mesh')
    plan = sub.add_parser('plan',help='Create and validate an LLM scene graph without generating images or meshes')
    plan.add_argument('--prompt',required=True)
    plan.add_argument('--output',required=True)
    plan.add_argument('--room-size',nargs=3,type=float)
    plan.add_argument('--model',default=None)
    generate = sub.add_parser('generate')
    generate.add_argument('--prompt', required=True)
    generate.add_argument('--output', required=True)
    generate.add_argument('--size', type=int, default=512)
    generate.add_argument('--views', type=int, default=24)
    generate.add_argument('--seed', type=int, default=0)
    generate.add_argument('--rgb-target',type=int,help='room-local: accepted RGB teacher count including the initial image')
    generate.add_argument('--camera-count',type=int,help='room-local: additional camera count (default 233; source camera is separate)')
    generate.add_argument('--demo', action='store_true', help='Synthetic integration fixture; does not generate images from the prompt')
    generate.add_argument('--mode', choices=['room-local','photo-room', 'image-room', 'learned', 'scene', 'expand', 'mesh', 'room', 'legacy'], default='room-local', help='photo-room adds mesh-conditioned photographic references; image-room starts from a whole-room front image; learned retains independent object image generation')
    generate.add_argument('--resume',action='store_true',help='Resume an identical image-room/learned request from completed stages')
    generate.add_argument('--planner-model',help='Qwen2.5-VL model for image-room, or causal model for learned')
    generate.add_argument('--layout-file', help='Editable saved layout.json; mesh mode only, bypasses depth estimation and detection')
    generate.add_argument('--front-image', help='Reuse an existing front image for a controlled comparison')
    generate.add_argument('--room-size', nargs=3, type=float, default=None, metavar=('WIDTH', 'HEIGHT', 'LENGTH'))
    train = sub.add_parser('train')
    train.add_argument('path')
    train.add_argument('--steps', type=int, default=1000)
    train.add_argument('--max-points', type=int, default=100000)
    refit=sub.add_parser('refit-front',help='Fit source-image camera and meshes into a separate run, then rebuild GS references')
    refit.add_argument('path')
    refit.add_argument('--output',required=True)
    refs=sub.add_parser('repair-references',help='Rebuild coverage-checked references in a separate run, preserving mesh geometry')
    refs.add_argument('path')
    refs.add_argument('--output',required=True)
    photo=sub.add_parser('synthesize-appearance',help='Generate mesh-conditioned photographic references in a separate run')
    photo.add_argument('path')
    photo.add_argument('--output',required=True)
    photo.add_argument('--size',type=int,default=512)
    photo.add_argument('--pilot-views',type=int)
    photo.add_argument('--resume',action='store_true')
    photo.add_argument('--whole-view',action='store_true',help='Regenerate entire novel view from the proxy; no source-photo projection or immutable background')
    guides=sub.add_parser('prepare-photographs',help='Prepare unprojected camera proxy guides and two-image generation prompts')
    guides.add_argument('path')
    guides.add_argument('--output',required=True)
    reevaluate=sub.add_parser('reference-evaluate',help='Reassess saved reference evidence into a separate dataset without generating images')
    reevaluate.add_argument('path')
    reevaluate.add_argument('--output',required=True)
    reevaluate.add_argument('--train-steps',type=int,default=0)
    reevaluate.add_argument('--max-points',type=int,default=100000)
    production=sub.add_parser('reference-export',help='Export accepted photographs and mesh-only supervision for rejected views')
    production.add_argument('path')
    production.add_argument('--output',required=True)
    production.add_argument('--train-steps',type=int,default=3000)
    production.add_argument('--max-points',type=int,default=180000)
    experiment=sub.add_parser('reference-run',help='WorldMesh-inspired multi-camera reference collection, geometric assessment and GS export')
    experiment.add_argument('path')
    experiment.add_argument('--output',required=True)
    experiment.add_argument('--resume',action='store_true')
    experiment.add_argument('--prepare-only',action='store_true')
    experiment.add_argument('--attempts',type=int,default=2)
    experiment.add_argument('--model',choices=['controlnet','base','distilled'],default='controlnet')
    experiment.add_argument('--fov',type=float,default=60)
    experiment.add_argument('--train-steps',type=int,default=0)
    experiment.add_argument('--max-points',type=int,default=100000)
    experiment.add_argument('--rgb-target',type=int,help='Exact number of accepted RGB teachers including the initial photograph')
    experiment.add_argument('--camera-count',type=int,help='Collect every specified camera, even if the RGB target is reached earlier')
    experiment.add_argument('--reuse-references',help='Reuse hash-verified evidence from identical earlier camera poses')
    local=sub.add_parser('generate-room-views',help='Local whole-room camera/depth/photograph generation, with automatic verification')
    local.add_argument('path')
    local.add_argument('--output',required=True)
    local.add_argument('--perimeter',type=int,default=16)
    local.add_argument('--overhead',type=int,default=8)
    local.add_argument('--adaptive',type=int,default=8)
    local.add_argument('--fov',type=float,default=75)
    local.add_argument('--pilot-views',type=int)
    local.add_argument('--prepare-only',action='store_true')
    local.add_argument('--resume',action='store_true')
    local.add_argument('--retries',type=int,default=3)
    local.add_argument('--backend-root')
    local.add_argument('--port',type=int,default=8197)
    local.add_argument('--conditioning',choices=['structure','atlas'],default='structure')
    local.add_argument('--image-model',choices=['controlnet','base','distilled'],default='controlnet')
    local.add_argument('--train-steps',type=int,default=0)
    proxy=sub.add_parser('complete-proxy',help='Ground missing window and rug from the source image and rerender proxy guides')
    proxy.add_argument('path')
    proxy.add_argument('--output',required=True)
    external=sub.add_parser('import-photographs',help='Validate a two-image generated reference manifest and prepare GS training')
    external.add_argument('path')
    external.add_argument('--manifest',required=True)
    external.add_argument('--output',required=True)
    repair = sub.add_parser('refine-depth', help='Re-estimate depth and rebuild seams using existing expand keyframe images')
    repair.add_argument('path')
    repair.add_argument('--output', required=True)
    rebuild = sub.add_parser('rebuild-mesh', help='Fuse trusted multi-view depth into one TSDF mesh and rebuild GS references')
    rebuild.add_argument('path')
    rebuild.add_argument('--output', required=True)
    rebuild.add_argument('--voxel-size', type=float, default=.025)
    rebuild.add_argument('--min-views', type=int, default=2)
    rebuild.add_argument('--strict-support', action='store_true', help='Discard all single-view surfaces; increases holes')
    rebuild.add_argument('--holdout-keyframe', type=int, help='Optional leave-one-input-out experiment; production uses all inputs')
    audit = sub.add_parser('audit-holes', help='Separate mesh gaps, grazing-angle rejection and Gaussian opacity deficits')
    audit.add_argument('path')
    audit.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command=='plan':
        from pathlib import Path
        from .semantic_plan import plan_with_llm
        folder=Path(args.pop('output'))
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError('Use an empty output directory for planning')
        model=args.pop('model')
        if model:
            args['model_id']=model
        plan_with_llm(output=folder,**args)
    elif command == 'inspect-front':
        from .front_inspection import inspect_front
        inspect_front(**args)
    elif command == 'generate':
        from .pipeline import generate
        generate(**args)
    elif command == 'refit-front':
        from .front_alignment import refit_scene
        refit_scene(**args)
    elif command == 'repair-references':
        from .repair_references import repair_references
        repair_references(**args)
    elif command == 'synthesize-appearance':
        from .appearance import synthesize_appearance
        synthesize_appearance(**args)
    elif command == 'prepare-photographs':
        from .external_references import prepare_photograph_guides
        prepare_photograph_guides(**args)
    elif command == 'reference-evaluate':
        from .reference_pipeline import evaluate_reference_experiment
        evaluate_reference_experiment(**args)
    elif command == 'reference-export':
        from .reference_pipeline import export_accepted_references
        export_accepted_references(**args)
    elif command == 'reference-run':
        from .reference_pipeline import run_reference_experiment
        run_reference_experiment(**args)
    elif command == 'generate-room-views':
        from .local_room import generate_room_views
        generate_room_views(**args)
    elif command == 'complete-proxy':
        from .proxy_completion import complete_proxy
        complete_proxy(**args)
    elif command == 'import-photographs':
        from .external_references import import_reference_pack
        import_reference_pack(**args)
    elif command == 'refine-depth':
        from .expansion import repair_depth_scene
        repair_depth_scene(**args)
    elif command == 'rebuild-mesh':
        from .mesh_fusion import rebuild_mesh_scene
        rebuild_mesh_scene(**args)
    elif command == 'audit-holes':
        from .hole_audit import audit_holes
        audit_holes(**args)
    else:
        from .train import train
        train(**args)


if __name__ == '__main__':
    main()
