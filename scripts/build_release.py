"""Export an allowlisted source archive; do not enumerate research data or outputs."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TOP={'README.md','LICENSE','pyproject.toml','.gitignore','MANIFEST.in'}
FOLDERS={'src':{'.py','.json'},'tests':{'.py'},'configs':{'.json'},'examples':{'.json','.yaml'},
         'docs':{'.md','.json'},'scripts':{'.py','.sbatch'},'.github':{'.yml'}}
PATTERNS=[r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}', r'AKIA[0-9A-Z]{16}',
          r'/home/[A-Za-z0-9_-]+/',r'[A-Z]:\\Users\\[A-Za-z0-9_-]+\\']


def selected():
    result=[ROOT/n for n in sorted(TOP) if (ROOT/n).is_file()]
    for directory,suffixes in FOLDERS.items():
        result.extend(p for p in sorted((ROOT/directory).rglob('*'))
                      if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts)
    return result


def build(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('archive already exists')
    files=selected(); blocked=[]
    for path in files:
        text=path.read_text(encoding='utf-8-sig')
        if any(re.search(pattern,text) for pattern in PATTERNS):blocked.append(path.relative_to(ROOT).as_posix())
    if blocked:raise ValueError('publication scan blocked selected files: '+', '.join(blocked))
    output.parent.mkdir(parents=True,exist_ok=True)
    prefix='causalquest-v0.1.0a1/'
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,prefix+p.relative_to(ROOT).as_posix())
        archive.writestr(prefix+'RELEASE_MANIFEST.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return {'files':len(files),'archive':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'excluded':['outputs','old runs','weights','private test data','virtual environments'],
            'github_published':False,'scan_scope':'selected text files, obvious patterns only'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    print(json.dumps(build(parser.parse_args().output),indent=2))

