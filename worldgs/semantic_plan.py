"""LLM scene graph with explicit requirements and geometric validation."""
import gc
import json
import re
from pathlib import Path
import numpy as np

MODEL = 'Qwen/Qwen2.5-3B-Instruct'
SYSTEM = '''You plan a single rectangular indoor 3D scene. Return ONLY JSON.
Do not drop any requested object just because it may be invisible in the front view.
Use this schema (all keys required):
{"style":"warm cream, natural oak, muted earth colors", "room":{"size":[8,3,8],"floor":"wood","walls":"plaster","ceiling":"plaster"},
 "objects":[{"id":"sofa_00","label":"sofa","description":"English visual description of exactly ONE object, materials and colors", "support":"floor", "size":{"width":2.2,"height":1.0,"depth":0.9},"size_locked":false,"x":0,"z":2.8,"yaw":0}],
 "requirements":[{"id":"r0","text":"one sofa","object_ids":["sofa_00"],"kind":"object"}],
 "relations":[{"subject":"table_00","relation":"in_front_of","object":"sofa_00","frame":"object"}],
 "lighting":{"description":"warm diffuse ceiling light","power":1000,"color":[1,0.9,0.8]},
 "assumptions":[],"unsupported":[]}
Objects have separate IDs for EVERY instance; plural bookshelves means at least two unless a count is stated.
dimensions=[width,height,depth]. x and z are separate NUMBER fields, center on the floor. y points down, floor y=room height/2. Never return a position array.
The front camera at (0,0,0) looks toward +z. Object fronts normally face -z when yaw=0.
For in_front_of/behind use the reference object's heading (frame="object"); yaw=0 faces -z. frame="world" instead uses -z as front. left_of defaults to smaller world x. Other allowed relations: next_to, against_wall (object="room").
Put objects inside room bounds without bounding-box intersections. Do not place objects around the camera origin.
Each object also has support="floor" or another object's ID. Use support for books, vases and other tabletop accessories. Their x,z are world coordinates. Supported objects must fit entirely over their support. No cyclic support chains.
A closed book lying flat has height about 0.03m, width 0.2m and depth 0.28m; do not confuse height with the cover length. All size fields are meters.
For a furnished/lived-in/decorated room, infer a modest coherent set of small accessories (books, vase, bowl), record them as assumptions and give each requirement provenance. Respect explicitly requested counts. Do not add furniture when the prompt explicitly forbids it.
Only one rectangular room is implemented. One rectangular window on the back (+z) wall is supported: add room.window={"x":0,"width":1.7,"sill":0.8,"height":1.7}. It is architecture, NOT an object. Keep furniture clear of its opening. Other openings, multiple rooms, ceiling-hung objects must be listed in unsupported rather than faked.
Use requirements to enumerate ALL requested object counts, appearance, room, lighting and relationships.
Use NAMED size fields width,height,depth. A tall bookshelf has height about 2.0 and depth about 0.35. Set size_locked=true ONLY if the user explicitly demands numerical dimensions.
Do not add unrequested large furniture. Share the room style, palette and material family across all object descriptions. Record inferred counts, positions and dimensions in assumptions.
Do not generate executable code. All descriptions must be in English even for Japanese input.'''


def extract_json(raw):
    start = raw.find('{')
    if start < 0:
        raise ValueError('Planner returned no JSON object')
    value, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError('Expected a scene object')
    return value


def normalize_size_locks(plan,prompt):
    """Do not turn model-inferred example dimensions into user constraints."""
    units=r'\b(?:millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?|mm|cm|inch(?:es)?|foot|feet|ft)\b|\d\s*m\b|メートル|センチ|ミリ|インチ'
    if not re.search(units,prompt,re.I):
        for obj in plan['objects']:
            if obj.get('size_locked'):
                obj['size_locked']=False
                plan['assumptions'].append('Dimensions of '+obj['id']+' are inferred; the prompt contains no explicit metric dimension.')
    return plan


def resolve_supports(plan):
    """Topologically resolve floor/contact heights, rejecting dangling/cyclic supports."""
    objects={o['id']:o for o in plan['objects']}
    ordered=[]
    active=set()
    done=set()
    def visit(obj):
        name=obj['id']
        if name in active:
            raise ValueError('Cyclic support: '+name)
        if name in done:
            return
        active.add(name)
        parent=obj.get('support','floor')
        if parent=='floor':
            obj['bottom_y']=plan['room']['size'][1]/2
        else:
            if parent not in objects:
                raise ValueError('Unknown support: '+str(parent))
            visit(objects[parent])
            base=objects[parent]
            obj['bottom_y']=base['bottom_y']-base['dimensions'][1]
        active.remove(name)
        done.add(name)
        ordered.append(obj)
    for obj in plan['objects']:
        visit(obj)
    return ordered


def object_bounds(obj, height):
    width, dy, depth = np.asarray(obj['dimensions'],float)
    x,z = np.asarray(obj['position'],float)
    a = np.deg2rad(float(obj['yaw']))
    extent = np.array([abs(np.cos(a))*width+abs(np.sin(a))*depth,dy,
                       abs(np.sin(a))*width+abs(np.cos(a))*depth])
    center = np.array([x,obj.get('bottom_y',height/2)-dy/2,z])
    return center-extent/2,center+extent/2


def validate_plan(plan, *, check_overlap=True):
    schema={'style':str,'room':dict,'objects':list,'requirements':list,'relations':list,
            'lighting':dict,'assumptions':list,'unsupported':list}
    for key,kind in schema.items():
        if not isinstance(plan.get(key),kind):
            raise ValueError('Missing or invalid scene field: '+key)
    for key in ('floor','walls','ceiling'):
        if not isinstance(plan['room'].get(key),str) or not plan['room'][key].strip():
            raise ValueError('Missing room material: '+key)
    if plan.get('unsupported'):
        raise ValueError('Unsupported requirements: '+str(plan['unsupported']))
    size = np.asarray(plan['room']['size'],float)
    if size.shape != (3,) or not np.isfinite(size).all() or (size < [3,2,3]).any() or (size > 30).any():
        raise ValueError('Invalid room dimensions')
    from .window_geometry import window_bounds
    window_bounds(plan['room'])
    objects = plan['objects']
    eye=np.asarray(plan.get('reference_camera',{}).get('eye',[0.,0.,0.]),float)
    if eye.shape!=(3,) or not np.isfinite(eye).all() or (np.abs(eye)>=size/2).any():
        raise ValueError('Invalid reference camera position')
    if not isinstance(objects,list) or not 1 <= len(objects) <= 24:
        raise ValueError('Scene needs 1–24 explicit object instances')
    resolve_supports(plan)
    names, bounds = set(),{}
    for obj in objects:
        name = obj['id']
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',name) or name in names or name=='room' or name.startswith(('room_','structure_')):
            raise ValueError('Object IDs must be unique safe identifiers')
        names.add(name)
        dimensions,position = np.asarray(obj['dimensions'],float),np.asarray(obj['position'],float)
        minimum=.005 if obj.get('support','floor')!='floor' or re.search(r'rug|carpet|mat',obj.get('label',''),re.I) else .05
        if dimensions.shape != (3,) or position.shape != (2,) or not np.isfinite(np.r_[dimensions,position,obj['yaw']]).all() or (dimensions < minimum).any():
            raise ValueError(f'Invalid geometry for {name}: dimensions must be exactly 3 positive numbers and position exactly 2 numbers [x,z]; got {obj}')
        if any(not isinstance(obj.get(key),str) or not obj[key].strip() for key in ('label','description')):
            raise ValueError('Object description missing: '+name)
        lo,hi = object_bounds(obj,size[1])
        if (lo < -size/2-.001).any() or (hi > size/2+.001).any():
            raise ValueError('Object outside room: '+name)
        if (lo < eye+.2).all() and (hi > eye-.2).all():
            raise ValueError('Object encloses front camera: '+name)
        for other,(a,b) in bounds.items():
            if check_overlap and (np.minimum(hi,b)-np.maximum(lo,a) > .02).all():
                raise ValueError('Object collision: '+name+' / '+other)
        bounds[name] = lo,hi
    for obj in objects:
        parent=obj.get('support','floor')
        if parent!='floor':
            lo,hi=bounds[obj['id']]
            a,b=bounds[parent]
            if (lo[[0,2]]<a[[0,2]]+.01).any() or (hi[[0,2]]>b[[0,2]]-.01).any():
                raise ValueError('Object footprint outside support: '+obj['id'])
    requirements = plan['requirements']
    referenced = set()
    if not requirements:
        raise ValueError('No explicit requirement ledger')
    for entry in requirements:
        if not isinstance(entry.get('text'),str) or not entry['text'].strip() or not isinstance(entry.get('object_ids'),list):
            raise ValueError('Malformed requirement ledger entry')
        ids = entry['object_ids']
        if not set(ids) <= names:
            raise ValueError('Requirement references an unknown object')
        referenced.update(ids)
    if not names <= referenced:
        raise ValueError('Objects without requirement provenance: '+str(names-referenced))
    lighting=plan['lighting']
    color=np.asarray(lighting['color'],float)
    if color.shape!=(3,) or not np.isfinite(color).all() or (color<0).any() or (color>1).any() or not 20<=float(lighting['power'])<=10000:
        raise ValueError('Invalid lighting parameters')
    objects = {o['id']:o for o in objects}
    for rel in plan['relations']:
        if rel.get('frame','world') not in ('world','object'):
            raise ValueError('Relation frame must be world or object')
        a = objects[rel['subject']]
        kind,target = rel['relation'],rel['object']
        ax,az = a['position']
        if kind == 'against_wall':
            lo,hi = bounds[a['id']]
            gap = min(lo[0]+size[0]/2,size[0]/2-hi[0],lo[2]+size[2]/2,size[2]/2-hi[2])
            satisfied = target == 'room' and gap <= .35
        else:
            b = objects[target]
            bx,bz = b['position']
            delta=np.array([ax-bx,az-bz])
            local=rel.get('frame','object' if kind in ('in_front_of','behind') else 'world')=='object'
            angle=np.deg2rad(b['yaw']) if local else 0.
            forward=np.array([-np.sin(angle),-np.cos(angle)])
            right=np.array([np.cos(angle),-np.sin(angle)])
            alo,ahi=bounds[a['id']]
            blo,bhi=bounds[b['id']]
            edge_gap=np.maximum(np.abs(delta)-((ahi-alo)+(bhi-blo))[[0,2]]/2,0.)
            satisfied = {'in_front_of':delta@forward>.1,'behind':delta@forward<-.1,
                         'left_of':delta@right<-.1,'next_to':np.linalg.norm(edge_gap)<=.75}.get(kind,False)
        if not satisfied:
            raise ValueError('Unsatisfied spatial relation: '+str(rel))
    return dict(passed=True,object_count=len(objects),requirements=len(requirements),
                scope='deterministic geometry/relation checks; semantic completeness additionally audited by LLM')


def repair_layout(plan):
    """Solve metric placement; never add/delete objects or alter their descriptions."""
    from scipy.optimize import least_squares
    objects=plan['objects']
    if not isinstance(plan.get('style'),str):
        raise ValueError('style must be a plain English STRING describing palette/materials, not an object or dictionary')
    resolve_supports(plan)
    w,h,l=np.asarray(plan['room']['size'],float)
    proposed=np.array([o['position'] for o in objects],float)
    extents=[]
    for obj in objects:
        a,b=object_bounds(obj,h)
        extents.append((b-a)[[0,2]])
    extents=np.asarray(extents)
    lower=-np.array([w,l])/2+extents/2+.04
    upper=np.array([w,l])/2-extents/2-.04
    if (upper<=lower).any():
        raise ValueError('Object dimensions cannot fit inside the room')
    indices={o['id']:i for i,o in enumerate(objects)}
    def residual(flat):
        centers=flat.reshape(-1,2)
        values=list((centers-proposed).ravel()*.2)
        for i in range(len(objects)):
            penetration=extents[i]/2+.25-np.abs(centers[i])
            a,b=object_bounds(objects[i],h)
            values.append(50*max(0,float(penetration.min())) if a[1]<.2 and b[1]>-.2 else 0.)
            for j in range(i):
                ai,bi=object_bounds(objects[i],h)
                aj,bj=object_bounds(objects[j],h)
                if min(bi[1],bj[1])-max(ai[1],aj[1])<=.02:
                    continue
                penetration=(extents[i]+extents[j])/2+.06-np.abs(centers[i]-centers[j])
                values.append(50*max(0,float(penetration.min())))
            parent=objects[i].get('support','floor')
            if parent!='floor':
                j=indices[parent]
                excess=np.abs(centers[i]-centers[j])+extents[i]/2-extents[j]/2+.02
                values.extend(50*np.maximum(excess,0))
        for rel in plan['relations']:
            i=indices[rel['subject']]
            kind=rel['relation']
            if kind=='against_wall':
                gap=np.r_[centers[i]-lower[i],upper[i]-centers[i]].min()
                values.append(30*max(0,float(gap)-.12))
                continue
            j=indices[rel['object']]
            a,b=centers[i],centers[j]
            delta=a-b
            local=rel.get('frame','object' if kind in ('in_front_of','behind') else 'world')=='object'
            angle=np.deg2rad(objects[j]['yaw']) if local else 0.
            forward=np.array([-np.sin(angle),-np.cos(angle)])
            right=np.array([np.cos(angle),-np.sin(angle)])
            edge_gap=np.maximum(np.abs(delta)-(extents[i]+extents[j])/2,0.)
            violation={'in_front_of':max(0,.2-delta@forward),'behind':max(0,.2+delta@forward),
                       'left_of':max(0,.2+delta@right),'next_to':max(0,np.linalg.norm(edge_gap)-.6)}.get(kind,100.)
            values.append(30*violation)
        return np.asarray(values)
    best=None
    rng=np.random.default_rng(0)
    for attempt in range(6):
        initial=np.clip(proposed+(0 if attempt==0 else rng.normal(0,.5,proposed.shape)),lower+.001,upper-.001)
        result=least_squares(residual,initial.ravel(),bounds=(lower.ravel(),upper.ravel()),max_nfev=500)
        for obj,position in zip(objects,result.x.reshape(-1,2)):
            obj['position']=position.tolist()
        try:
            validate_plan(plan)
            best=result.x.reshape(-1,2)
            break
        except ValueError:
            pass
    if best is None:
        raise ValueError('No collision-free placement satisfies the requested relations')
    plan['layout_adjustments']=[dict(id=o['id'],proposed=before.tolist(),actual=after.tolist())
                                for o,before,after in zip(objects,proposed,best) if np.linalg.norm(before-after)>1e-5]
    return plan


def plan_with_llm(prompt, output, room_size=None, model_id=MODEL, visual_observation=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                               attn_implementation='sdpa').to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    def ask(messages,limit=6000):
        encoded = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded,max_new_tokens=limit,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
    instruction = prompt+(f'\nUse room size {list(room_size)}.' if room_size else '')
    if visual_observation:
        instruction += ('\nReference-image observation (data, NOT extra requested objects): '+json.dumps(visual_observation)+
            '\nUse it for palette and relative arrangement. Original prompt determines required object counts. '
            'Plan side/rear regions for objects absent from front. Do not add unrequested props from the observation. '
            'Window is architecture only, never a relation subject or furniture support. '
            'Every floor-standing object has support="floor". Leave relations empty unless the original request requires one.')
    messages = [dict(role='system',content=SYSTEM),dict(role='user',content=instruction)]
    history=[]
    offset=len(list(output.glob('planner_*.txt')))
    try:
        for attempt in range(4):
            raw = ask(messages)
            (output/f'planner_{attempt+offset:02d}.txt').write_text(raw,encoding='utf-8')
            try:
                plan = extract_json(raw)
                for obj in plan.get('objects',[]):
                    if not isinstance(obj.get('support'),str):
                        raise ValueError('Every planned object must explicitly declare support: floor or parent ID. On-table objects must name the table ID, never floor.')
                    if 'size' in obj:
                        obj['dimensions']=[obj['size'][key] for key in ('width','height','depth')]
                    if 'x' in obj and 'z' in obj:
                        obj['position']=[obj.pop('x'),obj.pop('z')]
                if room_size is not None:
                    plan['room']['size']=list(room_size)
                plan=repair_layout(normalize_size_locks(plan,prompt))
                validation = validate_plan(plan)
                audit_raw = ask([dict(role='system',content='Audit scene JSON against the ORIGINAL user prompt. Return ONLY {"missing":[],"contradictions":[]}. Check requested object counts and all spatial, appearance and lighting requirements. Objects requested ON a table or shelf MUST have support set to that object ID, not floor; next_to alone does not satisfy on. Do not treat planner assumptions as user requests. Report actual omissions only.'),
                                 dict(role='user',content=json.dumps(dict(original_prompt=prompt,scene=plan),ensure_ascii=False))],900)
                audit = extract_json(audit_raw)
                if any(not isinstance(audit.get(key),list) for key in ('missing','contradictions')):
                    raise ValueError('Semantic audit did not return explicit missing/contradictions lists')
                (output/f'audit_{attempt+offset:02d}.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
                if audit.get('missing') or audit.get('contradictions'):
                    raise ValueError('Semantic audit: '+json.dumps(audit,ensure_ascii=False))
                plan.update(prompt=prompt,planner=model_id,validation=validation,semantic_audit=audit,
                            semantic_completeness_guaranteed=False)
                (output/'scene_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
                (output/'failure.json').unlink(missing_ok=True)
                return plan
            except (ValueError,KeyError,TypeError) as error:
                history.append(str(error))
                messages.extend([dict(role='assistant',content=raw),dict(role='user',content='Repair the full JSON. Validation failed: '+str(error))])
        (output/'failure.json').write_text(json.dumps(dict(errors=history),indent=2))
        raise ValueError('Scene planner failed validation: '+history[-1])
    finally:
        del model,tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
