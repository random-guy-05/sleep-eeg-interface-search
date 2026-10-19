from pathlib import Path
import json,hashlib,collections,tempfile,importlib,math,re,sys
import pandas as pd
import pymupdf
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));PKG=ROOT/'agentic_search';REV=PKG/'ieee_access_revision';P=REV/'paper'
catalog=pd.read_csv(PKG/'data/heads.csv').set_index('head_id').taxonomy_category.to_dict()
selected_csv=pd.read_csv(PKG/'analysis/output/final_selection_metrics.csv');selected_csv=selected_csv[(selected_csv.method=='full_agent')&(selected_csv.selector=='robust')].set_index(['held_site','budget','seed'])
proposed_summary=pd.read_csv(PKG/'analysis/output/quartet_category_diversity.csv');proposed_summary=proposed_summary[proposed_summary.method=='full_agent'].set_index('budget')
rows=[];files={};counts=[]
for path in sorted((PKG/'runs/source_only/agents').glob('**/*.json')):
 state=json.loads(path.read_text())
 if state.get('method')!='full_agent':continue
 budget=int(state['budget']);site=state['held_site_internal'];seed=int(state['seed']);history=state['history'];assert len(history)==budget
 assert state['model_requested']=='gpt-6-sol'
 candidates=[]
 for item in history:
  heads=item.get('real_head_ids') or item['head_ids'];assert len(set(heads))==4
  key='|'.join(sorted(heads));cats=len({catalog[h] for h in heads});candidates.append((key,float(item['min_source_auc']),cats))
  counts.append(dict(budget=budget,site=site,seed=seed,quartet=key,categories=cats,auto_repair=bool(item.get('auto_repair',False))))
 selected=sorted(candidates,key=lambda x:(-x[1],x[0]))[0]
 assert selected[0]=='|'.join(sorted(state['robust_selected_real_head_ids']))
 assert selected[0]==selected_csv.loc[(site,budget,seed),'quartet_key']
 rows.append(dict(budget=budget,site=site,seed=seed,quartet=selected[0],categories=selected[2]))
 files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
a=pd.DataFrame(counts);b=pd.DataFrame(rows);summary=[]
for budget in [10,25,50]:
 x=a[a.budget==budget];y=b[b.budget==budget];assert len(y)==30 and len(x)==budget*30
 proposed=x.categories.mean();chosen=y.categories.mean();assert math.isclose(proposed,float(proposed_summary.loc[budget,'mean_unique_categories']))
 summary.append(dict(budget=budget,runs=len(y),proposed_quartets=len(x),proposed_category_count_sum=int(x.categories.sum()),proposed_mean=proposed,proposed_category_count_distribution={str(k):int(v) for k,v in x.categories.value_counts().sort_index().items()},selected_quartets=len(y),selected_category_count_sum=int(y.categories.sum()),selected_mean=chosen,selected_category_count_distribution={str(k):int(v) for k,v in y.categories.value_counts().sort_index().items()},auto_filled_candidate_slots=int(x.auto_repair.sum())))
assert summary[0]['proposed_mean']==3.44 and math.isclose(summary[0]['selected_mean'],109/30)
# Independently regenerate aggregate tables into a temporary folder and compare to the supplied files.
mod=importlib.import_module('agentic_search.ieee_access_revision.scripts.build_paper');tables=['budget','cohort','contrast','overlap','exclusion','sensitivity','feedback','evaluation_scope','structure']
with tempfile.TemporaryDirectory(prefix='ieee-aggregate-table-check-') as d:
 mod.PAPER=Path(d);mod.main()
 for name in tables:assert (mod.PAPER/(name+'_table.tex')).read_text()==(P/(name+'_table.tex')).read_text(),name
 for name in ['feedback','overlap','exclusion','contrast','sensitivity']:assert (mod.PAPER/('supplement_'+name+'_table.tex')).read_text()==(P/('supplement_'+name+'_table.tex')).read_text(),name
# Review the complete supplement text, references and PDF bounds.
doc=pymupdf.open(P/'supplement.pdf');text='\n'.join(page.get_text() for page in doc)
for n in range(1,5):assert re.search(r'\bS'+str(n)+r'\b',text)
for page in doc:
 for w in page.get_text('words'):assert w[0]>=0 and w[1]>=0 and w[2]<=page.rect.width+.5 and w[3]<=page.rect.height+.5
assert not re.search(r'undefined references|Citation .* undefined|Reference .* undefined',(P/'supplement.log').read_text())
qa=json.loads((P/'PDF_QA.json').read_text());assert hashlib.sha256((P/'supplement.pdf').read_bytes()).hexdigest()==qa['supplement_pdf_sha256'];assert len(doc)==qa['supplement_pages']
report=dict(status='pass',scope='Independent category-count reconstruction from original source-only JSON histories and head taxonomy; saved robust selections and generated tables cross-checked; no searches, probe fits, or statistical analyses rerun.',category_definition='Number of distinct taxonomy_category values across four heads. Count candidate occurrences across runs, not globally deduplicated quartet IDs. Selected = maximum minimum-direction source AUROC, quartet-key tie break.',budgets=summary,raw_run_files_verified=len(files),source_input_sha256={str((PKG/'data/heads.csv').relative_to(ROOT)):hashlib.sha256((PKG/'data/heads.csv').read_bytes()).hexdigest(),**files},tables_regenerated_and_matched=tables,supplement=dict(pages=len(doc),sections=['S1 matched feedback','S2 provenance and exclusion','S3 comparisons, metric sensitivity, second controller','S4 prompts and reproduction'],unresolved_references=False,text_inside_page_bounds=True,pdf_sha256=qa['supplement_pdf_sha256']),no_manuscript_numerical_correction_needed=True)
(REV/'results/FINAL_SUPPLEMENT_NUMERICAL_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
for item in summary:print(json.dumps(item))
print('Raw run files checked:',len(files));print('Nine aggregate tables and five supplementary table copies match regenerated outputs; supplement S1-S4 present; page count matches PDF QA and references resolve.')
