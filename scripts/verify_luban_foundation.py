"""Fresh product installation, real HTTP answers and export admission proof."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import threading
import time

import httpx
import torch
import uvicorn
from luban.runtime import Runtime
from luban.settings import Settings, private_json, require_storage
from luban.server import create_app
from luban.training_checkpoint import training_runs
from luban import model_packages as packages
from train_luban_identity import MODEL, TARGET


def main(root):
    require_storage(root.parent)
    root.mkdir(exist_ok=False)
    settings=Settings(root/'home')
    settings.data.update(model_path=str(MODEL),storage_root=str(root/'storage'),
                         automatic={'mode':'manual','scheduled_time':'02:00'},local_thinking='off')
    runtime=Runtime(settings,None)
    server=thread=None
    report={'passed':False,'teacher_api_calls':0,'fresh_store':True,'rows':[]}
    try:
        runtime.load()
        manifest=json.loads((MODEL/'luban-foundation.json').read_text())
        assert runtime.active_core==manifest['version']
        runs=training_runs(runtime.memory,runtime.identity)
        assert len(runs)==1 and runs[0]['state']=='committed'
        assert runs[0]['commit']['report']['foundation_install'] is True
        assert runs[0]['commit']['report']['training_updates_executed']==0
        report.update(version=runtime.active_core,inference=runtime.inference_identity(),
                      training_engine=runtime.status()['training_engine'],first_load_journal=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(create_app(settings,runtime,port=port,autoload=False),
            host='127.0.0.1',port=port,access_log=False,log_level='warning'))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        deadline=time.monotonic()+10
        while not server.started:
            if time.monotonic()>deadline:raise TimeoutError('HTTP startup timeout')
            time.sleep(.02)
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=180,
            headers={'authorization':'Bearer '+settings.secret('access_key')},
            cookies={'luban_session':settings.secret('ui_token')}) as client:
            cases=[([{'role':'user','content':'你是谁'}],TARGET),
                ([{'role':'system','content':'你通过API服务用户。'},
                  {'role':'user','content':'你是谁？'}],TARGET),
                ([{'role':'user','content':'2+3是多少？'},{'role':'assistant','content':'5'},
                  {'role':'user','content':'再介绍一下你自己。'}],TARGET),
                ([{'role':'user','content':'是谁调优了你？'}],'麦好火。')]
            for messages,expected in cases:
                body={'model':'luban0','messages':messages,'thinking':'off','max_tokens':128,
                      'temperature':0,'stream':False}
                response=client.post('/v1/chat/completions',json=body);response.raise_for_status()
                value=response.json();answer=value['choices'][0]['message']['content'].strip()
                checked=answer==expected and value['choices'][0]['finish_reason']=='stop'
                report['rows'].append({'messages':messages,'response':value,'passed':checked})
                private_json(root/'progress.json',report)
                if not checked:raise RuntimeError('Fresh foundation HTTP identity failed')
        server.should_exit=True;thread.join(10)
        if thread.is_alive():raise RuntimeError('HTTP server did not close')
        server=None
        runtime.close_native()
        payload=packages._journal(runtime)
        names=packages._active_names(runtime.store)
        names.update(name for run in payload['runs'] for name in run['commit']['artifacts'])
        scratch=root/'export-admission';store=scratch/'modules/luban0-v1';store.mkdir(parents=True)
        files={}
        for name in names:
            path=store/name;shutil.copyfile(runtime.store/name,path)
            files['modules/luban0-v1/'+name]=packages._record(path)
        packages._validate_modules(scratch,payload,files)
        private_json(root/'export-accepted.json',payload)
        report.update(passed=True,export_admission_passed=True,accepted_runs=len(payload['runs']))
    finally:
        if server is not None:server.should_exit=True
        if thread is not None:thread.join(10)
        runtime._release_model()
        report['released']=runtime.model is None and runtime._model_process_lock is None
        private_json(root/'result.json',report)
        print(json.dumps({'passed':report['passed'],'version':report.get('version'),
                          'http_cases':len(report['rows']),'released':report['released']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(1);main(a.output)
