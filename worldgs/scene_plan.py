"""Explicit prompt inventory before image generation (bounded indoor grammar).

No image detection may delete an item requested in this inventory. This parser
is deliberately labelled as a restricted planner, not a general-purpose LLM.
"""
import re
import numpy as np


NAMES = {'sofa': r'sofas?|couches?|ソファ[ー]?',
         'table': r'(?:coffee\s+|wooden\s+)?tables?|テーブル|机',
         'bookshelf': r'bookshelves|bookshelf|bookcases?|本棚|書棚'}


def requirements(prompt):
    unsupported = re.findall(r'\b(?:chairs?|armchairs?|beds?|lamps?|plants?|windows?|doors?|cabinets?)\b|椅子|ベッド|照明|観葉植物|窓|ドア', prompt, re.I)
    if unsupported:
        raise ValueError('Planner has no implemented mesh backend for requested objects: '+', '.join(unsupported))
    result = []
    for category, pattern in NAMES.items():
        matches = list(re.finditer(pattern, prompt, re.I))
        if not matches:
            continue
        positive = [m for m in matches if not re.search(r'(?:without|no)\s+(?:a\s+|any\s+)?$', prompt[max(0,m.start()-15):m.start()], re.I)
                    and not re.match(r'(?:なし|を置かない)', prompt[m.end():])]
        if not positive:
            continue
        count = 1
        for m in positive:
            prefix = prompt[max(0,m.start()-24):m.start()]
            suffix = prompt[m.end():m.end()+8]
            number = re.search(r'(\d+|one|two|three|four|一|二|三|四)\s*(?:wooden\s+|coffee\s+)?$', prefix, re.I) or re.match(r'\s*(\d+|一|二|三|四)\s*[台脚個つ]', suffix)
            if number:
                word = number.group(1).lower()
                count = int(word) if word.isdigit() else {'one':1,'two':2,'three':3,'four':4,'一':1,'二':2,'三':3,'四':4}[word]
            elif m.group().lower() in ('bookshelves', 'bookcases', 'sofas', 'tables', 'couches'):
                count = max(count, 2)
        if not 1 <= count <= 4:
            raise ValueError('Indoor planner supports 1–4 instances per category')
        result.append(dict(category=category, count=count, source='explicit prompt requirement'))
    if not result:
        raise ValueError('No supported explicit objects. Planner currently supports sofa/table/bookshelf; use a scene layout for other requirements.')
    return result


def plan_scene(prompt, room_size=(8,3,8)):
    w,h,l = map(float, room_size)
    if min(w,l) < 6 or h < 2.4:
        raise ValueError('Current furniture planner requires room width/length >=6 and height >=2.4')
    inventory = requirements(prompt)
    shapes = [dict(name='room', kind='room', min=[-w/2,-h/2,-l/2], max=[w/2,h/2,l/2])]
    for request in inventory:
        category = request['category']
        for i in range(request['count']):
            if category == 'sofa':
                extent = np.array([2.2, 1., .85])
                candidates = [(0,l/2-.65,0),(0,-l/2+.65,180),(-w/2+.65,0,-90),(w/2-.65,0,90)]
            elif category == 'bookshelf':
                extent = np.array([1.6, min(2.3,h-.2), .35])
                candidates = [(-w/2+.25,l*.23,-90),(w/2-.25,-l*.23,90),(-w/2+.25,-l*.23,-90),(w/2-.25,l*.23,90)]
            else:
                extent = np.array([1.15,.55,.7])
                candidates = [(0,l/2-2.,0),(0,-l/2+2.,180),(-w/2+2.,0,-90),(w/2-2.,0,90)]
            x,z,yaw = candidates[i]
            center = np.array([x,h/2-extent[1]/2,z])
            local_lo, local_hi = center-extent/2, center+extent/2
            world_extent = extent.copy()
            if abs(yaw) == 90:
                world_extent[[0,2]] = world_extent[[2,0]]
            lo,hi = center-world_extent/2, center+world_extent/2
            shapes.append(dict(name=f'{category}_{i:02d}', label=category, kind='component_mesh',
                               min=lo.tolist(), max=hi.tolist(), local_min=local_lo.tolist(), local_max=local_hi.tolist(),
                               yaw=yaw, requested=True, source='prompt_inventory'))
    for i,a in enumerate(shapes[1:]):
        for b in shapes[i+2:]:
            overlap = np.minimum(a['max'],b['max'])-np.maximum(a['min'],b['min'])
            if (overlap > .02).all():
                raise ValueError(f'Planned objects overlap: {a["name"]}, {b["name"]}; revise layout')
    return dict(prompt=prompt, planner='restricted_indoor_prompt_inventory_v9', requirements=inventory, shapes=shapes,
                general_prompt_compliance_verified=False,
                limitations=['Restricted sofa/table/bookshelf vocabulary; not general language understanding',
                             'Room shape and unspecified positions use explicit layout priors',
                             'Style descriptors guide diffusion; arbitrary spatial clauses are not parsed'])
