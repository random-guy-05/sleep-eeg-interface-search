"""Bounded, aggregate audit of hidden-site identifiers in released model assets.

Zero identifier matches in these assets do not establish person-level absence
from every training/head-fitting dataset. Hidden participant rosters are not
available to the investigators, and cross-site identities cannot be linked.
"""
from pathlib import Path
import hashlib,json
import pandas as pd
import requests,io
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'brainbodyfm/stages/results/stable4_extension'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for rel in ['experiments/philosophers_stone/upstream/paper/outputs/brain_health_outputs_paper.csv',
            'experiments/philosophers_stone/upstream/paper/scripts/model-training-scripts/df_master_sample.csv',
            'agentic_search/ieee_access_revision/provenance/s0001_released_checkpoint_training_recordings.csv',
            'agentic_search/ieee_access_revision/provenance/s0001_conservative_development_subject_exclusions.csv']:
 p=BASE/rel;d=pd.read_csv(p,dtype=str)
 columns=[x for x in ['fileid','cohort','site_id','participant_id','recording_id','path_spec_eeg'] if x in d]
 for site in ['I0004','I0007']:
  count=int(d[columns].fillna('').apply(lambda x:x.str.contains(site,regex=False)).any(axis=1).sum())
  rows.append(dict(asset=p.name,rows=len(d),hidden_site=site,matching_rows=count,sha256=sha(p)))
url='https://moody-challenge.physionet.org/2026/results/challenge_2026_results.tsv'
r=requests.get(url,timeout=30);r.raise_for_status();d=pd.read_csv(io.StringIO(r.text),sep='\t');team=d[d['Team Name'].eq('Arnav & Ivan')];assert len(team)==1
keep=['Team Name','Title','Author(s)','CinC Submission ID','Final submission ID','Ranking (if eligible) on the test set','AUROC on the validation set','AUROC on the test set','Age-conditioned AUROC on the validation set','Age-conditioned AUROC on the test set']
report=dict(date='2026-10-27',checked_assets=rows,official_entry=team[keep].iloc[0].to_dict(),official_results_source=url,results_sha256=hashlib.sha256(r.content).hexdigest(),
 source_level_finding='No I0004/I0007-coded rows were found in the inspected released model-output, sample-master, or reconstructed S0001 membership assets.',
 person_level_finding='Not verifiable: complete hidden evaluation participant rosters and cross-cohort identity linkage were unavailable. The head-fitting membership of the complete released scoring pipeline was not recovered.',
 attribution='The organizer table confirms the selected entry scores and ID. The CinC manuscript attributes the entry to Stable4; exact organizer-side artifact and branch execution were not independently audited.',
 benchmark_curve_disposition='Omitted from the new manuscript figures and supplement because the original S0001 benchmark has known overlap with reconstructed model-development lists.')
OUT.mkdir(parents=True,exist_ok=True);(OUT/'hidden_site_provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
