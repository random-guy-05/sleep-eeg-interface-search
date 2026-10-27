"""Audit every eligible EDF header before assigning recording-level eligibility."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import pandas as pd
from signal_io import edf_header,plan_channel

BASE=Path(__file__).resolve().parent


def main():
    frame=pd.read_csv(BASE/'data/extraction_queue.csv').fillna('')
    def inspect(row):
        result={'subject_id':row.subject_id,'center':row.center}
        try:
            h=edf_header(row.edf_path)
            p=plan_channel([c['label'] for c in h['channels']])
            channels=[c for c in h['channels'] if c['label'] in p['channels']]
            known=bool(channels) and all(c['unit'].lower() in ['uv','µv','μv','mv','v'] for c in channels)
            result.update(channel_kind=p['kind'],channel_labels='|'.join(p['channels']),
                          channel_units='|'.join(c['unit'] for c in channels),
                          duration_hours=h['duration_hours'],known_voltage_units=known,
                          header_eligible=p['kind']!='unavailable' and known and h['duration_hours']>=2)
        except Exception as e:result.update(header_eligible=False,error=f'{type(e).__name__}: {e}')
        return result
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(inspect,row) for row in frame.itertuples(index=False)]):
            r=future.result();records.append(r)
            with (BASE/'results/header_audit_progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if len(records)%25==0:print(json.dumps({'headers_audited':len(records),'total':len(frame)}),flush=True)
    audit=pd.DataFrame(records).sort_values('subject_id')
    audit.to_csv(BASE/'data/signal_header_audit.csv',index=False)
    safe=frame.merge(audit[['subject_id','header_eligible']],on='subject_id',validate='one_to_one')
    safe=safe[safe.header_eligible].drop(columns='header_eligible')
    safe.to_csv(BASE/'data/verified_signal_queue.csv',index=False)
    summary={'total':len(frame),'known_unit_header_eligible':len(safe),
             'by_center':safe.groupby('center').size().to_dict(),
             'unknown_units_by_center':audit[~audit.known_voltage_units.fillna(False)].groupby('center').size().to_dict(),
             'channel_kind_counts':audit.channel_kind.value_counts().to_dict()}
    (BASE/'results/header_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
