"""Local stdin worker with CPU-resident model reuse and per-request GPU release."""
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def main(script):
    spec=importlib.util.spec_from_file_location('worldgs_local_worker',script)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    model=None
    for line in sys.stdin:
        job=json.loads(line);done=Path(job['done']);result={}
        try:
            request=json.loads(Path(job['request']).read_text())
            if model is None:model=worker.load_model(request)
            worker.run(request,model);result={'success':True}
        except BaseException as error:
            traceback.print_exc();result={'success':False,'error':str(error)}
        finally:
            if model is not None:
                import torch
                (model[0] if isinstance(model,tuple) else model).to('cpu')
                torch.cuda.empty_cache()
        temporary=done.with_suffix('.tmp');temporary.write_text(json.dumps(result));temporary.replace(done)


if __name__=='__main__':main(sys.argv[1])
