"""Compare GS missing opacity only at matching heldout poses and intrinsics."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def compare(before,after,output):
    before,after,output=map(Path,(before,after,output));output.mkdir(parents=True,exist_ok=True)
    a=json.loads((before/'scene.json').read_text());b=json.loads((after/'scene.json').read_text())
    if a['mesh_digest']!=b['mesh_digest'] or a['size']!=b['size']:
        raise ValueError('Paired comparison requires the same teacher mesh and image size')
    rows=[]
    for j,new in enumerate(b['heldout_frames']):
        matches=[(i,old) for i,old in enumerate(a['heldout_frames']) if old['target']==new['target']
                 and np.allclose(old['transform_matrix'],new['transform_matrix'])
                 and np.allclose(old.get('K',a['K']),new.get('K',b['K']))]
        if len(matches)!=1:continue
        i,old=matches[0]
        row=dict(target=new['target'],before_index=i,after_index=j)
        mask=np.load(before/old['mask_file'])&np.load(after/new['mask_file'])
        for name,path,index in [('before',before,i),('after',after,j)]:
            alpha=np.load(path/f'heldout_alpha_{index:03d}.npy')
            row[name]=float(((alpha<.5)&mask).sum()/max(mask.sum(),1))
        rows.append(row)
    if not rows:raise ValueError('No matched heldout views')
    report={'same_mesh':True,'paired_views':len(rows),'metric':'fraction of teacher pixels with GS alpha<0.5',
            'limitations':'Teacher-based coverage; not photorealism or independent reconstruction accuracy',
            'before_mean':float(np.mean([r['before'] for r in rows])),
            'after_mean':float(np.mean([r['after'] for r in rows])),
            'before_max':max(r['before'] for r in rows),'after_max':max(r['after'] for r in rows),'views':rows}
    (output/'paired_coverage.json').write_text(json.dumps(report,indent=2))
    selected=sorted(rows,key=lambda r:r['before'],reverse=True)[:4]
    tile=256;canvas=Image.new('RGB',(tile*len(selected),2*(tile+36)),(28,31,36));draw=ImageDraw.Draw(canvas)
    for column,row in enumerate(selected):
        for line,name,path in [(0,'before',before),(1,'after',after)]:
            image=Image.open(path/f'heldout_render_{row[name+"_index"]:03d}.png').convert('RGB')
            x,y=column*tile,line*(tile+36)
            canvas.paste(image.resize((tile,tile)),(x,y+36))
            draw.text((x+5,y+3),f'{name}: missing {row[name]:.2%}',fill='white')
            draw.text((x+5,y+18),row['target'][:34],fill='white')
    canvas.save(output/'paired_comparison.png')
    print(json.dumps({k:v for k,v in report.items() if k!='views'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('before');parser.add_argument('after');parser.add_argument('--output',required=True)
    args=parser.parse_args();compare(args.before,args.after,args.output)
