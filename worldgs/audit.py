"""Dataset integrity checks and reference-view contact sheet (not a quality score)."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def audit(folder):
    folder = Path(folder)
    meta = json.loads((folder/'scene.json').read_text())
    points = np.load(folder/'points.npz')
    assert len(points['xyz']) > 0 and np.isfinite(points['xyz']).all()
    assert points['xyz'].shape == points['rgb'].shape
    canvas = Image.new('RGB', (4*256, ((len(meta['frames'])+3)//4)*280), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(meta['frames']):
        pose = np.array(frame['transform_matrix'])
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        d = np.load(folder/f'depth_{i:03d}.npy')
        assert np.isfinite(d).all() and (d > 0).all()
        im = Image.open(folder/frame['file_path']).convert('RGB')
        assert im.size == (meta['size'], meta['size'])
        x, y = (i % 4)*256, (i//4)*280
        canvas.paste(im.resize((256, 256)), (x, y))
        draw.text((x+4, y+258), f"yaw={frame['yaw']:.0f} new={frame['generated_fraction']:.0%}", fill='black')
    canvas.save(folder/'references.jpg')
    result = {'frames': len(meta['frames']), 'initial_points': len(points['xyz']), 'size': meta['size'], 'demo': meta['demo'], 'integrity': 'passed', 'generated_fractions': [f['generated_fraction'] for f in meta['frames']], 'note': 'Integrity checks do not establish multi-view consistency or reconstruction quality.'}
    (folder/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit(sys.argv[1])
