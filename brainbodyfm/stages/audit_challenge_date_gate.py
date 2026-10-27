"""Read exact submitted source and test its date gate on synthetic records."""
import argparse
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ENTRY = '76a36e54251cfce85889a3b2390018ffc23329ee'


def audit(entry):
    path = entry / 'phi_challenge/cohort.py'
    actual = path.read_bytes()
    committed = subprocess.check_output(['git', '-C', str(entry), 'show', ENTRY + ':phi_challenge/cohort.py'])
    if actual != committed:
        raise ValueError('Date-gate source differs from the submitted Git object')
    spec = importlib.util.spec_from_file_location('submitted_cohort_gate', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checks = []
    for name, present, unique, expected in [('missing_dates', 0, 0, False),
                                           ('below_90pct', 179, 179, False),
                                           ('at_90pct', 180, 180, True),
                                           ('only_seven_distinct', 200, 7, False),
                                           ('eight_distinct', 200, 8, True)]:
        rows = [{'record': {'SiteID': 'synthetic'}, 'metadata':
                 {'CreationTime': (datetime(2020, 1, 1) + timedelta(days=i % unique)).isoformat()}
                 if i < present and unique else {}} for i in range(200)]
        reason = None
        try:
            module.recording_date_rank(rows, min_valid_fraction=.90, min_unique=8)
            enabled = True
        except module.AdaptationDisabled as error:
            enabled, reason = False, str(error)
        if enabled != expected:
            raise AssertionError(name)
        checks.append(dict(check=name, enabled=enabled, expected=expected, reason=reason))
    return dict(commit=ENTRY, source_sha256=hashlib.sha256(actual).hexdigest(),
                date_metadata_field='CreationTime', minimum_valid_fraction=.90,
                minimum_unique_dates=8, minimum_target_records=200, checks=checks,
                fallback='ordinary per-record probabilities retained when adaptation is disabled',
                organizer_note='https://groups.google.com/g/physionet-challenges/c/WyxLESJjrnY',
                organizer_note_summary='Organizers state that the posted evaluation excludes date information.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit(args.entry), indent=2) + '\n')
    print(args.output)
