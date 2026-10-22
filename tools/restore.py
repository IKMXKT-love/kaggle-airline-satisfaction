"""Restore verified independent archive volumes into a new empty directory."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    dest=a.destination.resolve();assert not dest.exists() or (dest.is_dir() and not any(dest.iterdir())),'Destination must be new or empty'
    assert not dest.is_symlink() and not dest.is_junction()
    m=json.loads((a.assets/'archive_manifest.json').read_text('utf-8'))
    expected={x['path']:x for x in m['files']};assert len(expected)==len(m['files'])
    for v in m['volumes']:
        path=a.assets/v['name'];assert path.stat().st_size==v['bytes'] and digest(path)==v['sha256'],v['name']
    dest.mkdir(parents=True,exist_ok=True);seen=set()
    for v in m['volumes']:
        with zipfile.ZipFile(a.assets/v['name']) as z:
            for info in z.infolist():
                name=info.filename;assert name in expected and name not in seen,name
                target=(dest/name).resolve();assert target.is_relative_to(dest),name
                record=expected[name];assert record['archive']==v['name'] and info.file_size==record['bytes'],name
                target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
                with z.open(info) as src,target.open('xb') as out:
                    for b in iter(lambda:src.read(4*1024**2),b''):h.update(b);out.write(b)
                assert h.hexdigest()==record['sha256'],name
                os.utime(target,ns=(record['mtime_ns'],record['mtime_ns']));seen.add(name)
        print('Restored',v['name'],flush=True)
    assert seen==set(expected)
    for name in m['directories']:
        target=(dest/name).resolve();assert target.is_relative_to(dest);target.mkdir(parents=True,exist_ok=True)
    (dest/'ARCHIVED_LINKS.json').write_text(json.dumps(m['links'],ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Verified {len(seen)} files. External links were recorded, not recreated.')

if __name__=='__main__':main()
