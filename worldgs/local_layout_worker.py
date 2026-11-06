"""Isolated Grounding DINO layout check worker, retaining crop evidence."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from worldgs.appearance_models import AppearanceModels
r=json.loads(Path(sys.argv[1]).read_text());model=AppearanceModels.__new__(AppearanceModels);model.torch=torch;model.detector_processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny');model.detector=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to('cuda')
detections,conflicts=model.layout_detections(np.asarray(Image.open(r['image']).convert('RGB')),r['objects'],r['labels']);Path(r['output']).write_text(json.dumps(dict(detections=detections,conflicts=conflicts),indent=2))
