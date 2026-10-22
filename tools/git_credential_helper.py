"""Supply current gh credentials only to the explicitly authorized repository."""
import subprocess,sys

if sys.argv[-1:] != ['get']:
    raise SystemExit(0)
record=dict(line.split('=',1) for line in sys.stdin.read().splitlines() if '=' in line)
if record.get('protocol')!='https' or record.get('host')!='github.com':
    raise SystemExit(1)
if record.get('path','').strip('/').removesuffix('.git')!='IKMXKT-love/kaggle-airline-satisfaction':
    raise SystemExit(1)
value=subprocess.run(['gh','auth','token','--hostname','github.com'],capture_output=True,text=True)
if value.returncode or not value.stdout.strip():
    raise SystemExit(1)
sys.stdout.write('username=IKMXKT-love\npassword='+value.stdout.strip()+'\n\n')
