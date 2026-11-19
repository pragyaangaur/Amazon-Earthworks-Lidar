import csv,json,sys,zipfile,hashlib
from pathlib import Path
zpath=Path(sys.argv[1] if len(sys.argv)>1 else 'amazon_earthworks_results.zip')
outpath=Path(sys.argv[2] if len(sys.argv)>2 else 'logs/archive_audit.json')
root=Path.cwd()
issues=[]
with zipfile.ZipFile(zpath) as z:
    names=z.namelist()
    bad=z.testzip()
    if bad: issues.append('CRC failure: '+bad)
    for n in names:
        low=n.lower()
        if low.endswith(('.laz','.las')): issues.append('raw lidar member: '+n)
        if low.endswith('.zip'): issues.append('nested zip: '+n)
        if low.endswith(('.part','.linktmp')) or '__pycache__' in n or low.endswith('.pyc'):
            issues.append('forbidden member: '+n)
    outcome=json.loads(z.read('outcome.json'))
    def rows(name): return list(csv.DictReader(z.read(name).decode('utf-8-sig').splitlines()))
    cand=rows('candidates.csv'); rej=rows('rejected.csv'); trans=rows('transects.csv')
    ids=[r['id'] for r in cand+rej]
    if len(ids)!=len(set(ids)): issues.append('duplicate feature ids across candidates+rejected')
    counts={k:sum(r['certainty']==k for r in cand) for k in ['Known','Strong','Possible']}
    if counts!=outcome['candidates']: issues.append(f"candidate counts {counts} != outcome {outcome['candidates']}")
    if len(rej)!=outcome['rejected_features']: issues.append('rejected count mismatch')
    if len(ids)!=outcome['features_examined']: issues.append('features examined mismatch')
    if any(r['certainty']!='Rejected' for r in rej): issues.append('non-Rejected in rejected.csv')
    if any(r['certainty'] not in {'Known','Strong','Possible'} for r in cand): issues.append('invalid certainty in candidates.csv')
    possible=[r['id'] for r in cand if r['certainty']=='Possible']
    for fid in possible:
        if not any(n.startswith(f'candidates/{fid}/') for n in names): issues.append('Possible without evidence dir: '+fid)
    for n in names:
        if n.startswith('candidates/') and n.count('/')>=2:
            fid=n.split('/')[1]
            if fid and fid not in {r['id'] for r in cand}:
                issues.append('stale generated candidate directory: '+fid)
    dtm=[n for n in names if n.startswith('dtm/') and n.endswith('_dtm3m.npz')]
    if len(dtm)!=outcome['processed_to_dtm_and_main_screens']: issues.append(f"DTM count {len(dtm)} != processed {outcome['processed_to_dtm_and_main_screens']}")
    statuses={s:sum(r['status']==s for r in trans) for s in ['done','failed','skipped']}
    if len(trans)!=outcome['transects_total']: issues.append('transect row count mismatch')
    if statuses['done']!=outcome['done_full_screen']: issues.append('done mismatch')
    if statuses['failed']!=outcome['failed']: issues.append('failed mismatch')
    if statuses['skipped']!=outcome['skipped_or_incomplete']: issues.append('skipped mismatch')
    if 'R0414-002' in {r['id'] for r in cand}: issues.append('R0414-002 incorrectly promoted')
    possible_ids=set(possible)
    r0414=next((r for r in rej if r['id']=='R0414-002'),None)
    if not r0414 or (r0414['size_m'],r0414['relief_m'],r0414['canopy_m'])!=('33.0','0.99','4.7'):
        issues.append('R0414-002 legacy geometry not preserved')
    if 'R0914-001' in possible_ids: issues.append('R0914-001 override stale in candidates')
    if 'R0914-001' not in {r['id'] for r in rej}: issues.append('R0914-001 override absent in rejected')
    if '0584-CW003' not in possible_ids: issues.append('0584-CW003 missing')
    for fid in ['0234-CW005','0460-CW001','0500-CW002','D0414-L001']:
        if f'candidates/{fid}/coordinator_data_review.json' not in names: issues.append('missing data-support review '+fid)
    if 'candidates/0584-CW003/coordinator_review.json' not in names: issues.append('0584 review missing')
    summary={'archive':str(zpath),'members':len(names),'dtm_members':len(dtm),
             'outcome':outcome,'candidate_rows':len(cand),'rejected_rows':len(rej),
             'unique_feature_ids':len(set(ids)),'transect_status_counts':statuses,
             'crc_failure':bad,'issues':issues}
outpath.parent.mkdir(parents=True,exist_ok=True)
outpath.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='outcome'},indent=2,ensure_ascii=False))
raise SystemExit(1 if issues else 0)
