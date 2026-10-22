from pathlib import Path
import subprocess,sys,time,json
r=Path.cwd();out=r/'.artifact_runtime/v2'
def run(args, log):
 print('STAGE',log,flush=True)
 with (out/log).open('w',encoding='utf-8') as f:
  subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=r)
print('Waiting for verified CUDA wheel',flush=True)
while 'VERIFIED' not in (out/'torch_resume.log').read_text(encoding='utf-8',errors='replace'):
 time.sleep(2)
run(['-m','pip','install',str(out/'torch-2.11.0+cu128-cp313-cp313-win_amd64.whl'),'setuptools==78.1.0','--index-url','https://download.pytorch.org/whl/cu128'],'cuda_install.log')
run(['-c','import torch; print(torch.__version__,torch.cuda.is_available()); assert torch.cuda.is_available()'],'cuda_check.log')
while not (out/'cat_source_prior.json').exists():time.sleep(2)
run(['v2_tabm.py'],'tabm_screen.log')
run(['optimize_v2.py','select'],'selection.log')
run(['optimize_v2.py','confirmation'],'confirmation.log')
run(['optimize_v2.py','cv'],'cv.log')
run(['verify_v2.py'],'verification.log')
run(['v2_report.py'],'report.log')
print('ALL STAGES COMPLETE',flush=True)
