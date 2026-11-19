#!/usr/bin/env python3
"""Independently compare two split volumes with the original private archive."""
from __future__ import annotations
import argparse,hashlib,json,os,zipfile
from pathlib import Path

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()

def member_sha(handle):
 h=hashlib.sha256()
 for b in iter(lambda:handle.read(4<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('source',type=Path); ap.add_argument('part1',type=Path); ap.add_argument('part2',type=Path)
 ap.add_argument('--json',type=Path)
 args=ap.parse_args(); issues=[]
 with zipfile.ZipFile(args.source) as src, zipfile.ZipFile(args.part1) as one, zipfile.ZipFile(args.part2) as two:
  volumes=[one,two]; names=[z.namelist() for z in volumes]
  for i,z in enumerate(volumes,1):
   bad=z.testzip()
   if bad: issues.append(f'part{i} CRC failure: {bad}')
   if len(names[i-1])!=len(set(names[i-1])): issues.append(f'part{i} has duplicate member names')
  if one.read('SPLIT_MANIFEST.json')!=two.read('SPLIT_MANIFEST.json'): issues.append('volume manifests differ')
  if one.read('SPLIT_README.md')!=two.read('SPLIT_README.md'): issues.append('volume README files differ')
  manifest=json.loads(one.read('SPLIT_MANIFEST.json'))
  source_sha=file_sha(args.source)
  if manifest['source_archive_sha256']!=source_sha: issues.append('manifest source SHA-256 mismatch')
  if manifest['source_archive_bytes']!=args.source.stat().st_size: issues.append('manifest source size mismatch')
  records={r['name']:r for r in manifest['original_members']}
  source_infos={i.filename:i for i in src.infolist()}
  if set(records)!=set(source_infos): issues.append('manifest/source member set mismatch')
  allowed_shared={'SPLIT_README.md','SPLIT_MANIFEST.json'}
  for iteration in range(2):
   combined=[n for group in names for n in group]
   counts={n:combined.count(n) for n in set(combined)}
   missing=set(source_infos)-set(combined)
   if missing: issues.append(f'original members absent from volumes: {sorted(missing)[:8]}')
   duplicated={n:c for n,c in counts.items() if c!=1 and n not in allowed_shared}
   if duplicated: issues.append(f'unexpected duplicate/extra-volume members: {list(duplicated)[:8]}')
   extras=set(combined)-set(source_infos)-allowed_shared
   expected_extras=set(manifest.get('added_members_part1',[]))
   if extras!=expected_extras: issues.append(f'added-member mismatch: {sorted(extras)} != {sorted(expected_extras)}')
   for extra in expected_extras:
    if extra not in names[0]: issues.append(f'added code member absent from part1: {extra}')
   for name,info in source_infos.items():
    volume=records.get(name,{}).get('volume')
    if volume not in (1,2): issues.append(f'invalid manifest volume for {name}'); continue
    if name not in names[volume-1]: issues.append(f'{name} assigned to part{volume} but absent'); continue
    got=volumes[volume-1].getinfo(name)
    if got.file_size!=info.file_size or got.CRC!=info.CRC: issues.append(f'size/CRC metadata mismatch: {name}')
    if name.startswith('dtm/') is False and volume!=1: issues.append(f'non-DTM member outside part1: {name}')
    if iteration==0:
     with src.open(info) as a, volumes[volume-1].open(got) as b:
      if member_sha(a)!=member_sha(b): issues.append(f'content SHA-256 mismatch: {name}')
   if iteration==0: print('FULL_ORIGINAL_CONTENT_COMPARISON_OK' if not issues else 'CONTENT_COMPARISON_FOUND_ISSUES',flush=True)
  root=Path(__file__).resolve().parents[1]
  for name,path in [('code/scripts/split_results_zip.py',root/'code/split_results_zip.py'),
                    ('code/scripts/audit_split_results.py',root/'code/audit_split_results.py')]:
   if name in names[0] and path.is_file() and one.read(name)!=path.read_bytes(): issues.append(f'bundled script differs from source: {name}')
  dtm_counts=[sum(n.startswith('dtm/') and n.endswith('.npz') and n in source_infos for n in group) for group in names]
  if sum(dtm_counts)!=sum(n.startswith('dtm/') and n.endswith('.npz') for n in source_infos): issues.append('DTM total mismatch')
  summary={'status':'PASS' if not issues else 'FAIL',
   'source':{'path':str(args.source),'bytes':args.source.stat().st_size,'sha256':source_sha,'members':len(source_infos)},
   'volumes':[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_sha(p),'original_members':sum(n in source_infos for n in group),'dtm_members':dtm_counts[i]} for i,(p,group) in enumerate(zip([args.part1,args.part2],names))],
   'original_members_compared_by_full_sha256':len(source_infos),'issues':issues,
   'note':'Independent repackaging audit only; no certainty, novelty, absence or detector-validation claim.'}
 text=json.dumps(summary,indent=2,ensure_ascii=False)+'\n'
 if args.json:
  args.json.parent.mkdir(parents=True,exist_ok=True)
  temp=args.json.with_suffix(args.json.suffix+'.part'); temp.write_text(text); os.replace(temp,args.json)
 print(text,end='')
 return 1 if issues else 0

if __name__=='__main__': raise SystemExit(main())
