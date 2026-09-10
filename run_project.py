"""Enterprise launcher.
CMD examples:
  python run_project.py preflight
  python run_project.py setup
  python run_project.py test
  python run_project.py train
  python run_project.py evaluate
  python run_project.py ui
  python run_project.py all
  python run_project.py reset
"""
from pathlib import Path
import shutil, subprocess, sys
ROOT=Path(__file__).resolve().parent

def run_script(path,*args):
    p=subprocess.run([sys.executable,str(ROOT/path),*args],cwd=ROOT)
    if p.returncode: raise SystemExit(p.returncode)

def reset_generated():
    for name in ['data/processed','artifacts','results']:
        p=ROOT/name
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True,exist_ok=True)
    print('Generated data/artifacts/results removed. Raw dataset was preserved.')

def main():
    cmd=sys.argv[1].lower() if len(sys.argv)>1 else 'help'
    if cmd=='preflight': run_script(Path('scripts/preflight.py'))
    elif cmd=='reset': reset_generated()
    elif cmd=='setup':
        for p in ['scripts/download_data.py','scripts/prepare_data.py','scripts/train_tokenizers.py','scripts/filter_token_lengths.py']: run_script(Path(p))
    elif cmd=='test': run_script(Path('scripts/run_tests.py'))
    elif cmd=='train': run_script(Path('scripts/train.py'),*sys.argv[2:])
    elif cmd=='evaluate': run_script(Path('scripts/evaluate.py'))
    elif cmd=='ui': raise SystemExit(subprocess.call([sys.executable,'-m','streamlit','run','app/app.py'],cwd=ROOT))
    elif cmd=='all':
        main_for('preflight'); main_for('setup'); main_for('test'); main_for('train'); main_for('evaluate')
    else: print('Usage: python run_project.py preflight|setup|test|train|evaluate|ui|all|reset')

def main_for(cmd):
    if cmd=='preflight': run_script(Path('scripts/preflight.py'))
    elif cmd=='setup':
        for p in ['scripts/download_data.py','scripts/prepare_data.py','scripts/train_tokenizers.py','scripts/filter_token_lengths.py']: run_script(Path(p))
    elif cmd=='test': run_script(Path('scripts/run_tests.py'))
    elif cmd=='train': run_script(Path('scripts/train.py'))
    elif cmd=='evaluate': run_script(Path('scripts/evaluate.py'))

if __name__=='__main__': main()
