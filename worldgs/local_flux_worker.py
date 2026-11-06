"""Worker in the local ComfyUI environment. Never routes to hosted nodes."""
import asyncio,importlib.util,json,sys
from pathlib import Path

async def run(request):
    root=Path(request['backend_root']);spec=importlib.util.spec_from_file_location('worldgs_comfy_client',root/'method/flux_generation/comfyui_client.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    workflow=json.loads(Path(request['workflow']).read_text())
    if request.get('profile')=='base':
        remap={'81':'121','94':'122','92:102':'92:101','92:115':'92:102','92:114':'92:103','92:103':'92:104','92:104':'92:105','92:105':'92:106','92:106':'92:107','92:111':'92:108','92:113':'92:109','92:107':'92:110','92:110':'92:111','92:108':'92:114','92:109':'92:113','92:126':'92:127','92:125':'92:126','92:127':'92:128','92:128':'92:129','92:129':'92:130','92:130':'92:131','92:87':'92:86'}
        converted={}
        for key,node in workflow.items():
            for name,value in node['inputs'].items():
                if isinstance(value,list) and len(value)==2 and isinstance(value[0],str):node['inputs'][name]=[remap.get(value[0],value[0]),value[1]]
            converted[remap.get(key,key)]=node
        workflow=converted
    allowed={'LoadImage','SaveImage','KSamplerSelect','Flux2Scheduler','CFGGuider','SamplerCustomAdvanced','VAEDecode','RandomNoise','UNETLoader','CLIPLoader','CLIPTextEncode','ConditioningZeroOut','EmptyFlux2LatentImage','GetImageSize','VAEEncode','ReferenceLatent','VAELoader','ImageScaleToTotalPixels'}
    if any(n['class_type'] not in allowed for n in workflow.values()):raise ValueError('Only the audited local FLUX graph is allowed')
    async with module.ComfyUIClient(host='127.0.0.1',port=request['port']) as client:
        guide=await client.upload_image(Path(request['guide']));style=await client.upload_image(Path(request['style']))
        workflow['76']['inputs']['image']=guide;workflow['121']['inputs']['image']=style;workflow['92:109']['inputs']['text']=request['prompt'];workflow['92:106']['inputs']['noise_seed']=request['seed']
        workflow['92:111']['inputs']['megapixels']=request['size']**2/1e6;workflow['92:85']['inputs']['megapixels']=request['size']**2/1e6
        previous_positive='92:128' if request.get('kind')=='structure' else '92:131';previous_negative='92:126' if request.get('kind')=='structure' else '92:129'
        workflow['92:103']['inputs']['positive']=[previous_positive,0];workflow['92:103']['inputs']['negative']=[previous_negative,0]
        for role in ['depth_guide','previous']:
            if not request.get(role):continue
            filename=await client.upload_image(Path(request[role]));prefix='worldgs_'+role
            workflow[prefix+'_load']={'class_type':'LoadImage','inputs':{'image':filename}}
            workflow[prefix+'_scale']={'class_type':'ImageScaleToTotalPixels','inputs':{'image':[prefix+'_load',0],'upscale_method':'lanczos','megapixels':request['size']**2/1e6,'resolution_steps':1}}
            workflow[prefix+'_encode']={'class_type':'VAEEncode','inputs':{'pixels':[prefix+'_scale',0],'vae':['92:110',0]}}
            for side,base in [('positive',previous_positive),('negative',previous_negative)]:
                key=prefix+'_'+side;workflow[key]={'class_type':'ReferenceLatent','inputs':{'conditioning':[base,0],'latent':[prefix+'_encode',0]}};workflow['92:103']['inputs'][side]=[key,0]
            previous_positive=prefix+'_positive';previous_negative=prefix+'_negative'
        if request.get('kind')=='front':
            workflow['92:103']['inputs']['positive']=['92:109',0];workflow['92:103']['inputs']['negative']=['92:86',0]
        workflow['122']['inputs']['filename_prefix']='worldgs_room_local'
        Path(request['output']+'.workflow.json').write_text(json.dumps(workflow,indent=2))
        images=await client.generate(workflow,timeout=1800)
        if len(images)!=1:raise RuntimeError('Expected exactly one local photograph')
        images[0].convert('RGB').save(request['output'])

if __name__=='__main__':asyncio.run(run(json.loads(Path(sys.argv[1]).read_text())))
