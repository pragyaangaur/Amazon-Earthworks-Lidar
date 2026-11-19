#!/usr/bin/env python3
"""Audit the private Amazon earthwork ZIP against the requested deliverable grammar.

Read-only except for an optional JSON summary. This complements, rather than
replaces, unzip -t, code/verify_archive.py and references/package_m7/delivcheck.py.
Usage: .venv/bin/python code/final_deliverable_audit.py ZIP --json OUT.json
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np

COLUMNS = ['id','state','transect','EPSG','x','y','lon','lat','type','size_m','relief_m',
           'canopy_m','nearest_known_db','nearest_known_m','literature_checked','certainty','reason']
PACK_FILES = ('hillshade.png','relief.png','sky-view.png','canopy.png',
              'radial_profile.csv','cross_profiles.csv','skeptic_notes.md')
REQUIRED = ('REPORT.md','candidates.csv','rejected.csv','transects.csv','literature.md',
            'report_draft_IPHAN.md','code/README.md')
CODE_REQUIRED = ('code/scripts/compile_results.py','code/scripts/process_one_v3.py',
                 'code/scripts/run_queue_v3.py','code/scripts/preflight_compile.py',
                 'code/scripts/finish_audit.py','code/scripts/verify_archive.py',
                 'code/scripts/final_deliverable_audit.py','code/scripts/dedupe_results_dtm.py',
                 'code/scripts/link_archive_mirror.py','code/scripts/vendor_m2/process.py',
                 'code/scripts/vendor_m2/products.py','code/scripts/vendor_m2/raster.py',
                 'code/scripts/vendor_m2/lasio.py','code/scripts/vendor_m2/lazreader_laspy.py')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('zip',type=Path)
    ap.add_argument('--json',type=Path)
    ap.add_argument('--mirror',type=Path,help='defaults to results/ZIP_NAME when that exists')
    ap.add_argument('--progress-dir',type=Path,default=Path('progress'))
    args=ap.parse_args()
    issues=[]; warnings=[]
    def issue(message): issues.append(message)
    if not args.zip.is_file():
        issue(f'archive absent: {args.zip}')
        print(json.dumps({'status':'FAIL','issues':issues},indent=2)); return 1
    mirror=args.mirror
    if mirror is None:
        candidate=Path('results')/args.zip.name
        mirror=candidate if candidate.exists() else None
    mirror_record=None
    if mirror is not None:
        if not mirror.is_file(): issue(f'mirror absent: {mirror}')
        else:
            same=os.path.samefile(args.zip,mirror)
            digest=sha256(args.zip); mirror_digest=sha256(mirror)
            if digest!=mirror_digest: issue('root and mirror SHA-256 differ')
            mirror_record={'path':str(mirror),'same_inode':same,'sha256':mirror_digest}
    if args.zip.stat().st_size>5_100_000_000:
        warnings.append('archive exceeds approximately 5 GB; DTM subset rule requires review')
    try:
        archive=zipfile.ZipFile(args.zip)
    except Exception as exc:
        issue(f'cannot open ZIP: {exc}')
        print(json.dumps({'status':'FAIL','issues':issues},indent=2)); return 1
    with archive as z:
        names=set(z.namelist())
        for name in REQUIRED+CODE_REQUIRED:
            if name not in names: issue(f'missing required member: {name}')
        bad=[n for n in names if n.lower().endswith(('.laz','.las','.zip','.part','.pyc')) or '__pycache__' in n]
        if bad: issue(f'forbidden/intermediate members: {bad[:10]} (+{max(0,len(bad)-10)} more)')
        def text(name): return z.read(name).decode('utf-8-sig') if name in names else ''
        def rows(name): return list(csv.DictReader(io.StringIO(text(name))))
        candidates=rows('candidates.csv'); rejected=rows('rejected.csv'); transects=rows('transects.csv')
        outcome=json.loads(text('outcome.json')) if 'outcome.json' in names else {}
        reader=csv.reader(io.StringIO(text('candidates.csv')))
        header=next(reader,[])
        if header!=COLUMNS: issue(f'candidates.csv columns differ: {header}')
        rejected_header=next(csv.reader(io.StringIO(text('rejected.csv'))),[])
        if rejected_header!=COLUMNS: issue('rejected.csv does not use the candidate evidence schema')
        ids=[r.get('id','') for r in candidates+rejected]
        if len(ids)!=len(set(ids)): issue('feature ids are not unique across candidates+rejected')
        certainty=Counter(r.get('certainty') for r in candidates)
        if set(certainty)-{'Known','Strong','Possible'}: issue(f'invalid candidate certainty values: {sorted(certainty)}')
        for r in rejected:
            if r.get('certainty')!='Rejected' or not r.get('reason','').strip():
                issue(f'rejected row lacks Rejected certainty/reason: {r.get("id")}')
        if outcome:
            expected={k:outcome.get('candidates',{}).get(k,0) for k in ('Known','Strong','Possible')}
            observed={k:certainty.get(k,0) for k in ('Known','Strong','Possible')}
            if observed!=expected: issue(f'candidate counts {observed} != outcome {expected}')
            if len(rejected)!=outcome.get('rejected_features'): issue('rejected row count != outcome')
            if len(ids)!=outcome.get('features_examined'): issue('examined feature count != outcome')
            if len(transects)!=outcome.get('transects_total'): issue('transect row count != outcome')
        statuses=Counter(r.get('status') for r in transects)
        if set(statuses)-{'done','failed','skipped'}: issue(f'invalid transect statuses: {sorted(statuses)}')
        tids=[r.get('transect') for r in transects]
        if len(tids)!=len(set(tids)): issue('transects.csv is not one row per transect')
        required_tcols={'transect','status','download_s','process_s','empty_dtm_pct','n_features'}
        if transects and not required_tcols.issubset(transects[0]): issue('transects.csv lacks timing/empty-DTM/feature fields')
        if outcome:
            for key,status in [('done_full_screen','done'),('failed','failed'),('skipped_or_incomplete','skipped')]:
                if statuses.get(status,0)!=outcome.get(key): issue(f'{status} count != outcome.{key}')
        strong_possible=[r for r in candidates if r.get('certainty') in {'Strong','Possible'}]
        pack_dirs={n.split('/')[1] for n in names if n.startswith('candidates/') and n.count('/')>=2}
        for r in strong_possible:
            fid=r['id']; pack=f'candidates/{fid}'
            if fid not in pack_dirs: issue(f'missing evidence directory: {fid}'); continue
            for name in PACK_FILES:
                if f'{pack}/{name}' not in names: issue(f'{fid} missing evidence: {name}')
            candidate_json=f'{pack}/candidate.json'
            if candidate_json in names:
                data=json.loads(text(candidate_json))
                scale=data.get('pixel_size_m')
                if scale is not None:
                    try:
                        if not 1.0<=float(scale)<=3.0: issue(f'{fid} pixel_size_m outside 1-3 m: {scale}')
                    except (TypeError,ValueError): issue(f'{fid} invalid pixel_size_m: {scale}')
        expected_packs={r['id'] for r in candidates}
        extra=pack_dirs-expected_packs
        if extra: issue(f'stale candidate evidence directories: {sorted(extra)}')
        if sum(r.get('certainty')=='Strong' for r in candidates)!=0:
            warnings.append('Strong candidates require individual specialist review before external use')
        control_png=[n for n in names if n.startswith('control/0414/') and n.endswith('.png')]
        if len(control_png)<4: issue('control/0414 lacks the positive-control image set')
        forest_controls=sorted({n.split('/')[1] for n in names if n.startswith('control/') and '_trees_inpe/' in n})
        dtm_names=sorted(n for n in names if n.startswith('dtm/') and n.endswith('.npz'))
        if outcome and len(dtm_names)!=outcome.get('processed_to_dtm_and_main_screens'):
            issue('DTM member count != processed transects')
        dtm_cells=set(); dtm_bad=[]
        for name in dtm_names:
            try:
                with z.open(name) as handle, np.load(handle,allow_pickle=False) as data:
                    required_arrays={'dtm','x0','y0','cell','epsg'}
                    if not required_arrays.issubset(data.files):
                        dtm_bad.append((name,f'arrays={data.files}')); continue
                    cell=float(np.asarray(data['cell']).reshape(()))
                    x0=float(np.asarray(data['x0']).reshape(())); y0=float(np.asarray(data['y0']).reshape(()))
                    epsg=int(np.asarray(data['epsg']).reshape(()))
                    dtm_cells.add(cell)
                    if cell!=3.0 or not np.isfinite([x0,y0]).all() or not 30000<=epsg<=33000:
                        dtm_bad.append((name,'invalid cell/origin/EPSG'))
            except Exception as exc: dtm_bad.append((name,str(exc)))
        if dtm_bad: issue(f'DTM georeferencing/metadata failures: {dtm_bad[:8]} (+{max(0,len(dtm_bad)-8)} more)')
        report=text('REPORT.md')
        headings=re.findall(r'^##\s+(.+)$',report,re.M)
        if not headings or headings[0]!='Outcome': issue('REPORT.md first section is not Outcome')
        for prefix in ('Verified results','Unverified claims','Failed directions','Suggested next steps'):
            if not any(h.startswith(prefix) for h in headings): issue(f'REPORT.md lacks section {prefix}')
        next_match=re.search(r'^## Suggested next steps.*?\n(.*?)(?=^## |\Z)',report,re.M|re.S)
        next_count=len(re.findall(r'^\d+\.\s+',next_match.group(1),re.M)) if next_match else 0
        if not 1<=next_count<=5: issue(f'REPORT.md has {next_count} suggested next steps, expected 1-5')
        if 'Continue the priority queue' in report: issue('REPORT.md retains a stale continue-queue next step')
        if outcome:
            if f"{outcome.get('processed_to_dtm_and_main_screens')} transects processed" not in report:
                issue('REPORT.md Outcome processed count does not match outcome.json')
            for cert in ('Known','Strong','Possible'):
                if f"{outcome.get('candidates',{}).get(cert)} {cert}" not in report:
                    warnings.append(f'Outcome certainty {cert} not found as standalone prose; table may carry it')
        literature=text('literature.md')
        for r in strong_possible:
            if not re.search(r'^##\s+'+re.escape(r['id'])+r'\s*$',literature,re.M):
                issue(f'literature.md lacks a per-candidate section for {r["id"]}')
        draft=text('report_draft_IPHAN.md')
        for marker in ('## Português','## English','Peripato','Possible','Strong','Private position annex','CNSA'):
            if marker not in draft: issue(f'IPHAN draft lacks marker: {marker}')
        if 'Não enviei' not in draft or 'has not been sent' not in draft:
            issue('IPHAN draft does not say unsent in both languages')
        for r in strong_possible:
            if r['id'] not in draft: issue(f'IPHAN draft omits candidate {r["id"]}')
        if 'positions are not public' not in draft and 'posições dos candidatos não foram publicadas' not in draft:
            warnings.append('check draft privacy wording manually')
        readme=text('code/README.md')
        for command in ('code/vendor_m2/process.py','code/preflight_compile.py','code/compile_results.py --zip',
                        'code/verify_archive.py','code/final_deliverable_audit.py'):
            if command not in readme: issue(f'code/README.md lacks exact command reference: {command}')
        progress_members={n for n in names if n.startswith('progress/') and not n.endswith('/')}
        if not progress_members: issue('progress/ notes are absent')
        progress_parity=None
        if args.progress_dir.exists():
            source={str(p) for p in args.progress_dir.rglob('*') if p.is_file()}
            progress_parity={'source':len(source),'archive':len(progress_members),'match':source==progress_members}
            if source!=progress_members: issue('archive progress/ differs from source progress notes')
        summary={
            'status':'PASS' if not issues else 'FAIL','archive':str(args.zip),
            'bytes':args.zip.stat().st_size,'sha256':sha256(args.zip),'members':len(names),
            'mirror':mirror_record,'candidate_certainty':dict(certainty),'rejected_rows':len(rejected),
            'transect_status_counts':dict(statuses),'strong_possible_packs_checked':len(strong_possible),
            'required_pack_files':list(PACK_FILES),'control_0414_pngs':len(control_png),
            'blind_forest_control_groups':forest_controls,'dtm_members':len(dtm_names),
            'dtm_cell_sizes_m':sorted(dtm_cells),'progress_parity':progress_parity,
            'suggested_next_steps':next_count,'issues':issues,'warnings':warnings,
            'note':'Structure/content audit only. No certainty, novelty, detector validation or archaeological-absence conclusion is inferred.'
        }
    output=json.dumps(summary,indent=2,ensure_ascii=False)+'\n'
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.json.with_suffix(args.json.suffix+'.part')
        temporary.write_text(output); os.replace(temporary,args.json)
    print(output,end='')
    return 1 if issues else 0


if __name__=='__main__':
    raise SystemExit(main())
