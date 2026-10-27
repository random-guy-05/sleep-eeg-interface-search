"""Create an aggregate, ordered flow from the saved nonexclusive eligibility flags."""
from pathlib import Path
import json
import pandas as pd
B=Path(__file__).resolve().parent
frame=pd.read_csv(B/'data/attrition_by_subject.csv',dtype=str).fillna('')
assert frame.subject_id.is_unique
rules=[('Unavailable or unmapped center',{'unmapped_or_unavailable_center'}),('Known duplicate recording',{'known_duplicate_EDF_subject'}),('No contributor-selected questionnaire',{'no_contributor_approved_ASQ'}),('Age or sex missing, ineligible, or discordant',{'age_missing_or_under18','sex_missing','ASQ_PSG_age_conflict','ASQ_PSG_sex_conflict'}),('No completed eligible in-lab executive task',{'no_completed_valid_inlab_PCET'}),('Recording unavailable or ambiguous',{'EDF_not_listed_or_ambiguous'}),('Diagnostic PSG or date unavailable',{'no_qualifying_diagnostic_PSG','missing_or_multiple_PSG_dates'}),('Cognitive testing outside the allowed date window',{'PCET_PSG_date_window_not_met'})]
allowed=set().union(*(r for _,r in rules));flags=[set(x.split('|'))-{''} for x in frame.exclusion_reasons]
assert set().union(*flags).issubset(allowed)
remaining=[True]*len(frame);rows=[{'stage':'Questionnaire or cognitive-testing subject IDs considered','excluded':0,'remaining':len(frame)}]
for label,group in rules:
 bad=[i for i,flag in enumerate(flags) if remaining[i] and bool(flag&group)]
 for i in bad:remaining[i]=False
 rows.append({'stage':label,'excluded':len(bad),'remaining':sum(remaining)})
assert sum(remaining)==593 and all(remaining[i]==(not flags[i]) for i in range(len(frame)))
headers=pd.read_csv(B/'data/signal_header_audit.csv')
assert len(headers)==593 and (headers.channel_kind=='unavailable').sum()==1
assert ((headers.duration_hours<2)&(headers.channel_kind!='unavailable')).sum()==2
assert len(pd.read_csv(B/'results/provisional_assumed_uv/nested_transport/analysis_subjects.csv'))==590
rows += [{'stage':'No usable C4–M1 route','excluded':1,'remaining':592},{'stage':'Recording shorter than two hours','excluded':2,'remaining':590}]
out=B/'results/paired_cognitive_comparisons';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'cohort_ordered_flow.csv',index=False)
(out/'cohort_ordered_flow_notes.md').write_text('''# Ordered STAGES eligibility accounting\n\nThe initial 1,931 are the union of unique subject identifiers present in the contributor-selected questionnaire list or any cognitive-battery record, not the total number enrolled in STAGES. Eligibility flags were generated together and are nonexclusive. For this presentation each subject is assigned to the first applicable exclusion in the order shown. The intermediate exclusions depend on that order; they must not be treated as independent totals. The final 593 eligible and 590 completed records match the saved analysis. The recording-level exclusions were checked against the saved header audit. Earlier source-level counts and exclusion flags are subject to the availability of the exported source tables and PSG inventory.\n''')
print(pd.DataFrame(rows).to_string(index=False))
