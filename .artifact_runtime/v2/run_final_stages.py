from pathlib import Path
import subprocess,sys
r=Path.cwd();out=r/'.artifact_runtime/v2'
def run(args,log):
 print('STAGE',log,flush=True)
 with (out/log).open('w',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=r)
run(['v2_tabm.py','--compact'],'tabm_compact.log')
run(['optimize_v2.py','select'],'selection.log')
run(['optimize_v2.py','confirmation'],'confirmation.log')
run(['optimize_v2.py','cv'],'cv.log')
run(['verify_v2.py'],'verification.log')
run(['v2_report.py'],'report.log')
print('ALL STAGES COMPLETE',flush=True)
