"""Post hoc, paired B=10 feedback experiment. Target scoring is a separate gated phase.

Both contemporaneous arms reuse each of the 30 historical opening batches.
Only the feedback arm receives their source metrics. No model call receives targets.
Run from KaggleData: python -m agentic_search.ieee_access_revision.scripts.feedback_ablation PHASE
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
REV = PACKAGE / 'ieee_access_revision'
OUT = REV / 'results'
SITES = ('I0002', 'I0006', 'S0001')
ARMS = ('feedback', 'no_feedback')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(p)

def now():
    return datetime.now(timezone.utc).isoformat()

def key(q):
    if len(q) != 4 or len(set(q)) != 4:
        raise ValueError('Four distinct head IDs required')
    return '|'.join(sorted(q))

def select(rows):
    return sorted(rows, key=lambda r: (-float(r['min_source_auc']), r['quartet_key']))[0]

def initial(site, seed):
    p = PACKAGE / f'runs/source_only/agents/full_agent/codex/{site}/B10_seed{seed}.json'
    state = json.loads(p.read_text())
    rows = state['history'][:5]
    assert len(rows) == 5 and all(r['round'] == 1 for r in rows)
    return rows

def prompts(arm, rows, catalog, prompt):
    allowed = ('auc_source1_to_source2', 'auc_source2_to_source1', 'mean_source_auc',
               'min_source_auc', 'direction_gap', 'selected_C')
    history = [{'head_ids': r['head_ids'], **({k: float(r[k]) for k in allowed}
               if arm == 'feedback' else {})} for r in rows]
    user = ('Round 2 of 2. Five candidate evaluations have been used; five remain. '
            'Propose exactly five new four-head candidates. The fixed source-only selector '
            'will select from all ten candidates after evaluation.\n\n'
            'Frozen catalog (JSON; IDs and permitted descriptions only):\n' +
            json.dumps(catalog, separators=(',', ':')) +
            '\n\nComplete previous proposal history in chronological order:\n' +
            json.dumps(history, separators=(',', ':'), allow_nan=False))
    if arm == 'no_feedback':
        user += ('\nSource-performance feedback is withheld in this experimental condition. '
                 'Previous IDs are supplied only to prevent duplication. Do not invent metrics.')
    user += '\nReturn a new batch of exactly five candidates now. Do not repeat any previous quartet.'
    return prompt, user

def freeze():
    import pandas as pd
    p = OUT / 'extension_protocol.json'
    if p.exists():
        verify(); print('Existing extension protocol verified'); return
    files = [Path(__file__), PACKAGE / 'configs/prompts/full_agent.txt', PACKAGE / 'data/heads.csv',
             PACKAGE / 'data/historical_quartets.csv', PACKAGE / 'agents/model_cli.py',
             PACKAGE / 'evaluator/source_evaluator.py']
    files += [PACKAGE / f'data/source_folds/{s}.parquet' for s in SITES]
    files += [PACKAGE / f'runs/source_only/agents/full_agent/codex/{s}/B10_seed{k}.json'
              for s in SITES for k in range(1000, 1010)]
    obj = dict(created_at_utc=now(), designation='post_hoc_extension_after_original_target_release',
        model='gpt-6-sol', reasoning_effort='medium', sampling_seed_applied=False,
        immutable_model_build_available=False, budget=10, opening_batch=5, second_batch=5,
        sites=list(SITES), seeds=list(range(1000,1010)), arms=list(ARMS), n_pairs=30,
        design='Contemporaneous feedback and no-feedback continuation of identical archived opening batches; fixed robust selector in both arms.',
        primary_extension_comparison='feedback minus no_feedback, macro age-conditioned AUROC; 50000 paired participant bootstraps conditional on selections',
        secondary='paired run bootstrap; source gain; AUROC tolerances 1 and 3 years and conventional AUROC',
        participant_resampling='Outcome-stratified within site; identical resampled participants for all arms; rebuild age-pair eligibility with multiplicities; reject zero-eligible draws and report their count.',
        run_resampling='Resample the ten paired continuations within each institution independently, then average institutions equally; 50000 draws.',
        invalid_policy='One repair request with ID/duplicate validation errors only; uniform valid quartets fill any remaining invalid slots, recorded explicitly.',
        stop_rule='Fixed 30 pairs; no stopping or method changes based on target results.',
        temporal_limit='Original opening batches were generated in September; both continuation arms are generated together now. Same model ID does not establish identical immutable model weights.',
        forbidden='Original 501 historical quartets and duplicates within a trajectory. Previously observed target scores never enter model contexts.',
        target_gate='All 60 arms and their source-selected choices must be frozen before target labels are opened in this extension.',
        inputs={str(f.relative_to(PACKAGE.parent)): sha(f) for f in files},
        environment={'python':platform.python_version(),'platform':platform.platform(),
                     'codex_cli':subprocess.check_output(['codex','--version'],text=True).strip()},
        bootstrap_seed=2026101801, bootstrap_draws=50000)
    save(p,obj)
    print('Frozen extension',sha(p))

def verify():
    obj = json.loads((OUT/'extension_protocol.json').read_text())
    for rel, digest in obj['inputs'].items():
        if sha(PACKAGE.parent/rel) != digest:
            raise ValueError('Extension input changed: '+rel)
    return obj

def valid_batch(batch, allowed, forbidden, opening):
    used = set(opening); accepted=[]; errors=[]
    for i,q in enumerate(batch if isinstance(batch,list) else []):
        try:
            k=key(q)
            if set(q)-allowed: raise ValueError('Unknown ID')
            if k in forbidden: raise ValueError('Excluded historical quartet')
            if k in used: raise ValueError('Duplicate quartet')
            used.add(k); accepted.append(sorted(q))
        except (ValueError,TypeError) as e: errors.append(f'Slot {i+1}: {e}')
    if len(batch if isinstance(batch,list) else []) != 5: errors.append('Exactly five quartets required')
    return accepted[:5], errors

def propose_one(site, seed, arm):
    import pandas as pd
    from agentic_search.agents.model_cli import call_static
    from agentic_search.agents.run_trajectories import _proposal_schema
    dest=OUT/f'continuations/{site}_{seed}_{arm}.json'
    if dest.exists(): return json.loads(dest.read_text())
    rows=initial(site,seed)
    catalog=pd.read_csv(PACKAGE/'data/heads.csv',dtype=str)[['head_id','head_name','taxonomy_category']].to_dict('records')
    allowed={r['head_id'] for r in catalog}
    forbidden=set(pd.read_csv(PACKAGE/'data/historical_quartets.csv').quartet_key)
    opening={key(r['head_ids']) for r in rows}
    system,user=prompts(arm,rows,catalog,(PACKAGE/'configs/prompts/full_agent.txt').read_text())
    logdir=OUT/f'controller_logs/{site}_{seed}_{arm}'
    logdir.mkdir(parents=True,exist_ok=True)
    (logdir/'system.txt').write_text(system); (logdir/'user.txt').write_text(user)
    calls=[]; accepted=[]
    for attempt in range(2):
        request=user if attempt==0 else user+'\n\nYour previous batch was rejected. Repair it using only this controller feedback: '+json.dumps(errors)
        (logdir/f'request_{attempt+1}.txt').write_text(request)
        result=call_static('codex','gpt-6-sol',system,request,_proposal_schema(),
                           seed=seed+2018+attempt,reasoning_effort='medium',timeout_seconds=600)
        accepted,errors=valid_batch(result['answer'].get('candidates',[]),allowed,forbidden,opening)
        calls.append({**result,'validation_errors':errors})
        save(logdir/f'response_{attempt+1}.json',calls[-1])
        if len(accepted)==5 and not errors: break
    n_fallback=5-len(accepted)
    rng=random.Random(seed+(0 if arm=='feedback' else 100000))
    used=opening|{key(q) for q in accepted}
    while len(accepted)<5:
        q=sorted(rng.sample(sorted(allowed),4)); k=key(q)
        if k not in forbidden and k not in used:
            used.add(k); accepted.append(q)
    obj=dict(site=site,seed=seed,arm=arm,opening_quartets=[sorted(r['head_ids']) for r in rows],
        continuation_quartets=accepted,n_fallback=n_fallback,completed_at_utc=now(),
        protocol_sha256=sha(OUT/'extension_protocol.json'),calls=calls,
        system_sha256=sha(logdir/'system.txt'),user_sha256=sha(logdir/'user.txt'))
    save(dest,obj)
    print('Proposed',site,seed,arm,'fallback',n_fallback,flush=True)
    return obj

def propose(workers, limit):
    verify()
    tasks=[(s,k,a) for s in SITES for k in range(1000,1010) for a in ARMS]
    if limit: tasks=tasks[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(propose_one,*t) for t in tasks]
        for f in as_completed(fs): f.result()

def source():
    verify()
    from agentic_search.evaluator.source_evaluator import evaluate_source
    from threadpoolctl import threadpool_limits
    files=sorted((OUT/'continuations').glob('*.json'))
    if len(files)!=60: raise ValueError(f'Expected all 60 continuation arms, found {len(files)}')
    selected=[]; allrows=[]
    with threadpool_limits(limits=1):
        for i,p in enumerate(files):
            arm=json.loads(p.read_text()); site=arm['site']; seed=arm['seed']
            rows=initial(site,seed)
            rows += [evaluate_source(site,q,cache_path=OUT/'source_cache.sqlite') for q in arm['continuation_quartets']]
            assert len(rows)==10 and len({r['quartet_key'] for r in rows})==10
            choice=select(rows)
            row={k:choice[k] for k in ('quartet_key','selected_C','min_source_auc','mean_source_auc','direction_gap')}
            row.update(site=site,seed=seed,arm=arm['arm'],opening_best=min(1,max(float(r['min_source_auc']) for r in rows[:5])),
                       continuation_best=max(float(r['min_source_auc']) for r in rows[5:]))
            selected.append(row)
            allrows.extend(dict(site=site,seed=seed,arm=arm['arm'],order=j+1,**{k:r[k] for k in
                ('quartet_key','selected_C','min_source_auc','mean_source_auc','direction_gap')}) for j,r in enumerate(rows))
            print('Source-scored',i+1,'/60',flush=True)
    import pandas as pd
    pd.DataFrame(allrows).to_csv(OUT/'source_candidates.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'source_choices.csv',index=False)
    manifest=dict(created_at_utc=now(),protocol_sha256=sha(OUT/'extension_protocol.json'),n_arms=60,
        inputs={str(p.relative_to(REV)):sha(p) for p in [*files,OUT/'source_candidates.csv',OUT/'source_choices.csv']})
    save(OUT/'pretarget_manifest.json',manifest)
    print('Frozen extension target gate',sha(OUT/'pretarget_manifest.json'))

def target():
    verify()
    manifest=json.loads((OUT/'pretarget_manifest.json').read_text())
    assert manifest['n_arms']==60
    for p,d in manifest['inputs'].items():
        if sha(REV/p)!=d: raise ValueError('Pre-target input changed: '+p)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from agentic_search.analysis.scripts.participant_bootstrap import _score_one, _target_frame
    choices=pd.read_csv(OUT/'source_choices.csv')
    records=[]; predictions=[]
    with threadpool_limits(limits=1):
        for site in SITES:
            frame=_target_frame(site)
            for r in choices[choices.site.eq(site)].itertuples():
                score=_score_one(site,r.quartet_key,r.selected_C,frame)
                records.append(dict(site=site,seed=r.seed,arm=r.arm,quartet_key=r.quartet_key,
                    selected_C=r.selected_C,target_auc=score['target_auc_recomputed']))
                predictions.extend(dict(site=site,seed=r.seed,arm=r.arm,subject_id=str(s),score=float(v))
                    for s,v in zip(frame.subject_id,score['scores']))
    pd.DataFrame(records).to_csv(OUT/'paired_target_results.csv',index=False)
    pd.DataFrame(predictions).to_parquet(OUT/'paired_predictions_private.parquet',index=False)
    save(OUT/'target_release.json',dict(created_at_utc=now(),manifest_sha256=sha(OUT/'pretarget_manifest.json'),
         n_arms=len(records),target_labels_used_for_generation_or_selection=False))
    print(pd.DataFrame(records).groupby(['arm','site']).target_auc.mean().to_string())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','propose','source','target'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.phase=='propose': propose(args.workers,args.limit)
    else: globals()[args.phase]()
