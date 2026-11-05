"""Owned localhost ComfyUI plus isolated local Depth Pro worker."""
import json,os,socket,subprocess,time,urllib.request
from pathlib import Path

class LocalRuntime:
    def __init__(self,folder,backend_root=None,port=8197,image_model="base"):
        self.root=Path(backend_root or os.environ.get('WORLDGS_LOCAL_BACKEND',str(Path.home()/'worldmesh-main-optimized'))).resolve();self.folder=Path(folder);self.port=port;self.proc=None;self.log=None
        self.comfy_python=Path(os.environ.get('WORLDGS_COMFY_PYTHON',str(Path.home()/'miniforge/envs/worldmesh-comfy/bin/python')));self.depth_python=Path(os.environ.get('WORLDGS_DEPTH_PYTHON',str(Path.home()/'miniforge/envs/worldmesh-depth-pro/bin/python')))
        if image_model not in ['controlnet','base','distilled']:raise ValueError('Use controlnet, base or distilled local profile')
        self.image_model=image_model
        self.model_workers={}
        self.workflow=self.root/('comfyui/scenes_workflows/image_flux2_klein_image_edit_9b_base_iterative.json' if image_model=='base' else 'comfyui/scenes_workflows/image_flux2_klein_image_edit_9b_distilled_iterative_api.json')
    def preflight(self):
        if self.image_model=='controlnet':
            cache=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub'
            repos=['stable-diffusion-v1-5/stable-diffusion-inpainting','lllyasviel/control_v11f1p_sd15_depth','h94/IP-Adapter','lllyasviel/control_v11p_sd15_canny']
            if not self.depth_python.is_file() or not (self.root/'ml-depth-pro/checkpoints/depth_pro.pt').is_file():raise FileNotFoundError('Depth Pro backend missing')
            for repo in repos:
                if not (cache/('models--'+repo.replace('/','--'))).is_dir():raise FileNotFoundError('Local HF model missing: '+repo)
            return dict(image_model=repos[0],depth_control_model=repos[1],appearance_model=repos[2],depth_model='apple/DepthPro',layout_model='IDEA-Research/grounding-dino-tiny',runtime='local native depth ControlNet + IP-Adapter + isolated Depth Pro',depth_control_scale=1.4,boundary_control_scale=1.0,ip_scale=.2)

        required=[self.comfy_python,self.depth_python,self.workflow,self.root/'comfyui/main.py',self.root/'ml-depth-pro/checkpoints/depth_pro.pt',self.root/('comfyui/models/diffusion_models/flux-2-klein-base-9b-fp8.safetensors' if self.image_model=='base' else 'comfyui/models/diffusion_models/flux-2-klein-9b-fp8.safetensors'),self.root/'comfyui/models/text_encoders/qwen_3_8b_fp8mixed.safetensors',self.root/'comfyui/models/vae/full_encoder_small_decoder.safetensors']
        missing=[str(p) for p in required if not p.is_file()]
        if missing:raise FileNotFoundError('Missing local dependencies: '+str(missing))
        return dict(image_model=('black-forest-labs/FLUX.2-klein-base-9b-fp8' if self.image_model=='base' else 'black-forest-labs/FLUX.2-klein-9b-fp8'),depth_model='apple/DepthPro',runtime='local ComfyUI + isolated Depth Pro',model_files=[dict(path=str(p),bytes=p.stat().st_size) for p in required[-4:]],workflow=str(self.workflow),host='127.0.0.1',port=self.port)
    def start(self):
        self.preflight();self.folder.mkdir(parents=True,exist_ok=True)
        if self.image_model=='controlnet':return
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',self.port))
        self.log=(self.folder/'comfyui.log').open('a');self.proc=subprocess.Popen([str(self.comfy_python),str(self.root/'comfyui/main.py'),'--listen','127.0.0.1','--port',str(self.port),'--lowvram'],cwd=self.root/'comfyui',stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(120):
            if self.proc.poll() is not None:raise RuntimeError('Local ComfyUI stopped; inspect '+str(self.folder/'comfyui.log'))
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/system_stats',timeout=2):return
            except (OSError,TimeoutError):time.sleep(1)
        raise TimeoutError('Local ComfyUI startup timed out')
    def free(self):
        if self.image_model=='controlnet':return
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}/free',data=b'{"unload_models":true,"free_memory":true}',headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30):pass
    def worker(self,python,script,request,record):
        record=Path(record);record.write_text(json.dumps(request,indent=2));env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
        with record.with_suffix('.log').open('w') as log:subprocess.run([str(python),str(Path(__file__).with_name(script)),str(record)],cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def persistent_worker(self,python,script,request,record):
        record=Path(record);record.write_text(json.dumps(request,indent=2));done=record.with_suffix('.done.json')
        if done.exists():done.unlink()
        if script not in self.model_workers:
            self.folder.mkdir(parents=True,exist_ok=True);log=(self.folder/(script+'.server.log')).open('a');env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
            proc=subprocess.Popen([str(python),str(Path(__file__).with_name('local_model_server.py')),str(Path(__file__).with_name(script))],cwd=self.root,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
            self.model_workers[script]=(proc,log)
        proc,_=self.model_workers[script]
        if proc.poll() is not None:raise RuntimeError('Local model worker stopped: '+script)
        proc.stdin.write(json.dumps(dict(request=str(record),done=str(done)))+'\n');proc.stdin.flush();started=time.monotonic()
        while not done.exists():
            if proc.poll() is not None:raise RuntimeError('Local model worker stopped: '+script)
            if time.monotonic()-started>1800:raise TimeoutError('Local model request timed out: '+script)
            time.sleep(.1)
        result=json.loads(done.read_text())
        if not result.get('success'):raise RuntimeError(result.get('error','Local model failed'))
    def generate(self,guide,style,prompt,seed,output,size,kind='edit',previous=None,depth_guide=None,visible_labels=None,appearance=None,edge_guide=None):
        request=dict(backend_root=str(self.root),workflow=str(self.workflow),port=self.port,guide=str(guide),style=str(style),prompt=prompt,seed=seed,output=str(output),size=size,visible_labels=visible_labels,appearance=appearance,edge_guide=str(edge_guide) if edge_guide else None,profile=self.image_model,kind=kind,previous=str(previous) if previous else None,depth_guide=str(depth_guide) if depth_guide else None)
        record=Path(output).with_suffix('.request.json')
        if Path(output).is_file() and record.is_file() and json.loads(record.read_text())==request:return
        if self.image_model=='controlnet':
            import sys
            self.persistent_worker(sys.executable,'local_control_worker.py',request,record)
        else:self.worker(self.comfy_python,'local_flux_worker.py',request,record)
    def depth(self,image,focal_px,output):
        self.free();request=dict(image=str(image),focal_px=float(focal_px),output=str(output),checkpoint=str(self.root/'ml-depth-pro/checkpoints/depth_pro.pt'))
        if self.image_model=='controlnet':self.persistent_worker(self.depth_python,'local_depth_worker.py',request,Path(output).with_suffix('.request.json'))
        else:self.worker(self.depth_python,'local_depth_worker.py',request,Path(output).with_suffix('.request.json'))
    def layout(self,image,objects,output):
        import sys
        request=dict(image=str(image),objects=[dict(label=o['label'],box=np_box(o['box'])) for o in objects],labels=['sofa','chair','table','bookshelf','bed','cabinet','rug','window'],output=str(output))
        self.worker(sys.executable,'local_layout_worker.py',request,Path(output).with_suffix('.request.json'))
    def close(self):
        for proc,log in self.model_workers.values():
            if proc.poll() is None:
                proc.stdin.close()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        self.model_workers.clear()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
        if self.log:self.log.close()
    def __enter__(self):
        try:self.start()
        except BaseException:self.close();raise
        return self
    def __exit__(self,*args):self.close()


def np_box(box):return [float(v) for v in box]
