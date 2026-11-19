#!/usr/bin/env python3
"""Split the private deliverable into two independent, balanced ZIP volumes.

All non-DTM deliverables stay in part 1; DTMs are divided in stable name order.
Original member bytes are retained exactly. Run audit_split_results.py afterward
for independent full-content comparison before replacing the source archive.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''): h.update(block)
 return h.hexdigest()

def copy_member(source,info,destination):
 with source.open(info) as reader, destination.open(copy.copy(info),'w',force_zip64=True) as writer:
  shutil.copyfileobj(reader,writer,4<<20)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('source',type=Path)
 ap.add_argument('--output-dir',type=Path,default=Path('.'))
 ap.add_argument('--prefix',default='amazon_earthworks_results')
 args=ap.parse_args()
 if not args.source.is_file(): raise SystemExit(f'source absent: {args.source}')
 args.output_dir.mkdir(parents=True,exist_ok=True)
 volumes=[args.output_dir/f'{args.prefix}_part1.zip',args.output_dir/f'{args.prefix}_part2.zip']
 temporary=[p.with_suffix('.zip.part') for p in volumes]
 for path in volumes+temporary:
  if path.exists(): raise SystemExit(f'refusing to overwrite {path}')
 source_sha=digest(args.source)
 with zipfile.ZipFile(args.source) as src:
  infos=sorted(src.infolist(),key=lambda x:x.filename)
  source_names=[i.filename for i in infos]
  if len(source_names)!=len(set(source_names)): raise SystemExit('duplicate source member names')
  dtms=[i for i in infos if i.filename.startswith('dtm/') and i.filename.endswith('.npz')]
  nondtms=[i for i in infos if i not in dtms]
  target=sum(i.compress_size for i in infos)/2
  assigned=[[],[]]; cost=sum(i.compress_size for i in nondtms)
  assigned[0].extend(nondtms)
  for info in dtms:
   index=0 if cost+info.compress_size<=target else 1
   assigned[index].append(info)
   if index==0: cost+=info.compress_size
  lookup={i.filename:(1 if i in assigned[0] else 2) for i in infos}
  manifest={
   'format':'amazon-earthworks-independent-two-zip-v1',
   'created_utc':datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
   'source_archive':args.source.name,'source_archive_bytes':args.source.stat().st_size,
   'source_archive_sha256':source_sha,'original_member_count':len(infos),
   'volumes':[p.name for p in volumes],
   'layout':'part1: all non-DTM deliverables plus DTM subset; part2: remaining DTMs; shared helpers in both',
   'reconstruction':'Extract both independent ZIPs into one directory. Shared SPLIT_README.md and SPLIT_MANIFEST.json are identical. No split-ZIP tool is required.',
   'privacy':'Private evidence: no publication, upload, email or external mapping.',
   'original_members':[{'name':i.filename,'volume':lookup[i.filename],
                        'file_size':i.file_size,'crc32':f'{i.CRC:08x}'} for i in infos],
  }
  root=Path(__file__).resolve().parents[1]
  helpers={'code/scripts/split_results_zip.py':Path(__file__).resolve(),
           'code/scripts/audit_split_results.py':(root/'code/audit_split_results.py').resolve()}
  for name,path in helpers.items():
   if not path.is_file(): raise SystemExit(f'missing helper script {path}')
   if name not in source_names: manifest.setdefault('added_members_part1',[]).append(name)
  manifest['shared_helper_members']=['SPLIT_README.md','SPLIT_MANIFEST.json']
  dtm_counts=[sum(i.filename.startswith('dtm/') for i in group) for group in assigned]
  readme=f"""# Private two-volume Amazon earthwork deliverable

These are two **independent ZIP files**, not a proprietary split-ZIP sequence.
Part 1 contains every report, CSV table, candidate evidence pack, control,
literature/IPHAN document, code file, audit and progress note, plus {dtm_counts[0]} DTMs.
Part 2 contains the remaining {dtm_counts[1]} DTMs.

Extract both into the same directory to reconstruct the complete original tree.
The shared `SPLIT_README.md` and `SPLIT_MANIFEST.json` files are identical, so
either may be overwritten when merging. The manifest maps every one of the
{len(infos)} original members to a volume with its size and CRC32. Original
member contents are unchanged; only split documentation and the two split/audit
scripts are added. `REPORT.md` and the main deliverable tables are in part 1.

Source archive: `{args.source.name}`
Source SHA-256: `{source_sha}`
Source size: {args.source.stat().st_size} bytes

Positions and imagery remain private. Do not publish, upload, email or externally
map the volumes. This repackaging changes no evidence, certainty or research flag.
"""
  try:
   for index,path in enumerate(temporary):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as dst:
     for info in assigned[index]: copy_member(src,info,dst)
     if index==0:
      for name,file_path in helpers.items():
       if name not in source_names: dst.write(file_path,name)
     dst.writestr('SPLIT_README.md',readme)
     dst.writestr('SPLIT_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    with zipfile.ZipFile(path) as check:
     bad=check.testzip()
     if bad: raise RuntimeError(f'CRC failure in {path}: {bad}')
   for temp,final in zip(temporary,volumes): os.replace(temp,final)
  except Exception:
   for path in temporary:
    if path.exists(): path.unlink()
   raise
 summary={'status':'SPLIT_WRITTEN_CRC_OK','source':str(args.source),'source_sha256':source_sha,
          'original_members':len(infos),'volumes':[{'path':str(p),'bytes':p.stat().st_size,
          'sha256':digest(p),'original_members':len(assigned[i]),'dtm_members':dtm_counts[i]}
          for i,p in enumerate(volumes)],
          'next':'Run code/audit_split_results.py with the source and both volumes before replacing the source.'}
 print(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__': main()
