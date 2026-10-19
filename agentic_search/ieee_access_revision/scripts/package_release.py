"""Whitelist a code/aggregate release and reject linked participant information."""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];PACKAGE=ROOT/'agentic_search'
REV=PACKAGE/'ieee_access_revision';DEST=REV/'reproducibility/public_release'

def copy(p,rel=None):
    q=DEST/(rel or p.relative_to(ROOT));q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)

def main():
    if DEST.exists():
        if (DEST/'.git').exists():raise ValueError('Refusing to replace a checkout; choose a separate generated destination')
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'brainbodyfm').glob('*.py'):copy(p)
    for p in PACKAGE.glob('*.py'):copy(p)
    for d in ['agents','evaluator','baselines','analysis']:
        for p in (PACKAGE/d).rglob('*.py'):
            if '__pycache__' not in p.parts:copy(p)
    for p in (PACKAGE/'configs').rglob('*'):
        if p.is_file() and p.suffix in ['.yaml','.txt']:copy(p)
    for name in ['heads.csv','historical_quartets.csv','prospective_random500.csv','folds.json','preparation_summary.json','opaque_head_mapping.csv','opaque_mapping_provenance.json']:
        copy(PACKAGE/'data'/name)
    for name in ['analysis_config.yaml','head_taxonomy.csv','family_taxonomy.csv']:
        copy(ROOT/'brainbodyfm/frozen'/name)
    for name in ['protocol_freeze.json','frozen_manifest.json','frozen_manifest.sha256','pretarget_audit.json','final_audit.json','regression_baseline.json']:
        copy(PACKAGE/'runs'/name)
    for name in ['README.md','RESULTS.md','GENERALIZABILITY_PREREGISTRATION.md']:copy(PACKAGE/name)
    for p in (PACKAGE/'runs/source_only').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl'] and p.stat().st_size:
            copy(p)
    for p in (PACKAGE/'runs/supplemental/generalizability/luna').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl'] and p.stat().st_size:copy(p)
    prohibited={'subject_id','record_id','participant_id','BDSPPatientID','outcome','age','sex','score'}
    for p in (PACKAGE/'analysis/output').glob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        if cols&prohibited:continue
        copy(p)
    for p in (REV/'scripts').glob('*.py'):copy(p)
    for p in (REV/'tests').glob('*.py'):copy(p)
    for p in (REV/'results').glob('*.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        if cols&prohibited:raise ValueError('Private columns in purported aggregate: '+str(p))
        copy(p)
    for name in ['extension_protocol.json','pretarget_manifest.json','target_release.json','cohort_audit.json','provenance_audit.json','runtime_reproduction_check.json','FINAL_VERIFICATION.json']:
        copy(REV/'results'/name)
    for d in ['continuations','controller_logs']:
        for p in (REV/'results'/d).rglob('*'):
            if p.is_file() and p.suffix in ['.json','.txt']:copy(p)
    copy(REV/'README.md',Path('REVISION_README.md'))
    copy(REV/'REVISION_REPORT.md',Path('REVISION_REPORT.md'))
    copy(REV/'reproducibility/requirements.lock.txt',Path('requirements.lock.txt'))
    copy(REV/'reproducibility/README_PUBLIC.md',Path('README.md'))
    copy(REV/'reproducibility/THIRD_PARTY_NOTICES.md',Path('THIRD_PARTY_NOTICES.md'))
    copy(REV/'provenance/UPSTREAM_SOURCE.json',Path('UPSTREAM_PROVENANCE_SOURCE.json'))
    # Exact historical inputs are licensed/restricted. Only hashes leave the authorized workspace.
    private=[PACKAGE/'data/participants.parquet',PACKAGE/'sealed/target_labels.parquet',
        *list((PACKAGE/'data/source_folds').glob('*.parquet')),ROOT/'brainbodyfm/cache/common_interface_manifest.csv']
    (DEST/'PRIVATE_INPUT_HASHES.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in private},indent=2)+'\n')
    (DEST/'.gitignore').write_text('__pycache__/\n*.pyc\n.venv/\n*.sqlite*\n*_private.parquet\nagentic_search/data/source_folds/\nagentic_search/data/participants.parquet\nagentic_search/sealed/\nbrainbodyfm/cache/\n')
    # Scan every released text file against actual private subject/record identifiers.
    m=pd.read_csv(ROOT/'brainbodyfm/cache/common_interface_manifest.csv',dtype=str)
    ids=set(m.subject_id)|set(m.record_id)|set('sub-'+m.site+m.subject_id)
    escaped='|'.join(sorted((re.escape(v) for v in ids),key=len,reverse=True))
    pattern=re.compile(r'(?<![A-Za-z0-9])(?:'+escaped+r')(?![A-Za-z0-9])')
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and '.git' not in p.parts)
    for p in files:
        if p.suffix in ['.parquet','.npy','.sqlite','.h5','.edf']:raise ValueError('Disallowed data file '+str(p))
        s=p.read_text()
        if pattern.search(s):raise ValueError('Private identifier found in '+str(p))
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,})',s):raise ValueError('Possible secret in '+str(p))
    hashes={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (DEST/'RELEASE_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    qa=dict(status='pass',n_files=len(files)+1,private_identifier_scan=True,secret_pattern_scan=True,
         excluded='No participant-level feature/label/demographic/prediction/membership files, raw EEG, pretrained weights, credentials or git history.',
         limitations='This bounded whitelist and scan is not a formal disclosure-risk analysis of aggregate source metrics.',
         aggregate_results_and_query_logs_only=True)
    (REV/'reproducibility/PRIVACY_REVIEW.json').write_text(json.dumps(qa,indent=2)+'\n')
    with zipfile.ZipFile(REV/'reproducibility/SLEEP_EEG_CODE_AND_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file() and '.git' not in p.parts:z.write(p,'sleep-eeg-interface-search/'+str(p.relative_to(DEST)))
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
