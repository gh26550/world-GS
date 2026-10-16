import argparse


def main():
    parser = argparse.ArgumentParser(description='World-GS experimental scene pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('generate')
    generate.add_argument('--prompt', required=True)
    generate.add_argument('--output', required=True)
    generate.add_argument('--size', type=int, default=512)
    generate.add_argument('--views', type=int, default=24)
    generate.add_argument('--seed', type=int, default=0)
    generate.add_argument('--demo', action='store_true', help='Synthetic integration fixture; does not generate images from the prompt')
    generate.add_argument('--mode', choices=['expand', 'mesh', 'room', 'legacy'], default='expand')
    generate.add_argument('--front-image', help='Reuse an existing front image for a controlled comparison')
    generate.add_argument('--room-size', nargs=3, type=float, default=None, metavar=('WIDTH', 'HEIGHT', 'LENGTH'))
    train = sub.add_parser('train')
    train.add_argument('path')
    train.add_argument('--steps', type=int, default=1000)
    train.add_argument('--max-points', type=int, default=100000)
    repair = sub.add_parser('refine-depth', help='Re-estimate depth and rebuild seams using existing expand keyframe images')
    repair.add_argument('path')
    repair.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'generate':
        from .pipeline import generate
        generate(**args)
    elif command == 'refine-depth':
        from .expansion import repair_depth_scene
        repair_depth_scene(**args)
    else:
        from .train import train
        train(**args)


if __name__ == '__main__':
    main()
