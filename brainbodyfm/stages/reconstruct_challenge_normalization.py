"""Source-only, resumable reconstruction; never reads STAGES cognition/scores.

Uses the exact submitted queue/extraction/projection/normalization functions.
The reached roster is runtime-dependent, so exports are explicitly reconstructed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd

ENTRY = '76a36e54251cfce85889a3b2390018ffc23329ee'
UPSTREAM = '0b1b49a8665e11157a107e0d76b9eedddd3b04c7'
META_SHA = 'b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f'
CHECKPOINT_SHA = 'b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87'
DATASET = 'physionet/physionetchallenge2026datalargeversion'
FILES = ['RECONSTRUCTION_AMENDMENT_06.md', 'reconstruct_challenge_normalization.py',
         'run_reconstructed_rank.py', 'test_reconstruction.py', 'audit_challenge_date_gate.py',
         'data/reconstruction_source_manifest.json', 'reconstruction_wavelet.py',
         'verify_reconstruction_wavelet.py', 'RECONSTRUCTION_AMENDMENT_07_TAP_CAPTURE.md']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_feature_taps(work, record_key, taps):
    """Retain already-computed private CARR inputs without changing the latent."""
    arrays = {name: np.asarray(taps[name], dtype=np.float32)
              for name in ['latent', 'global', 'temporal', 'counterfactual']}
    expected = {'latent': (1024,), 'global': (2048,),
                'temporal': (32, 32), 'counterfactual': (8,)}
    for name, value in arrays.items():
        if value.shape != expected[name] or not np.isfinite(value).all():
            raise ValueError('Invalid submitted feature tap: ' + name)
    folder = Path(work) / 'feature_taps'
    folder.mkdir(exist_ok=True)
    path = folder / (hashlib.sha256(record_key.encode()).hexdigest() + '.npz')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return dict(feature_tap_file=path.name, feature_tap_sha256=digest(path))


def frozen_receipt():
    base = Path(__file__).resolve().parent
    repo = subprocess.check_output(['git', '-C', str(base), 'rev-parse', '--show-toplevel'], text=True).strip()
    sha = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    hashes = {}
    for name in FILES:
        p = base / name
        b = subprocess.check_output(['git', '-C', repo, 'show', sha + ':' + str(p.relative_to(repo))])
        if b != p.read_bytes():
            raise ValueError('Amendment/code is not committed: ' + name)
        hashes[name] = hashlib.sha256(b).hexdigest()
    return dict(commit=sha, files=hashes, utc=datetime.now(timezone.utc).isoformat())


def download_file(relative, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    # Current official Kaggle SDK uses the typed RPC endpoint. Its legacy
    # path endpoint returns 404 for these nested EDF paths, even though the
    # dataset and root-level demographics.csv are accessible.
    request = dict(ownerSlug=DATASET.split('/')[0], datasetSlug=DATASET.split('/')[1],
                   datasetVersionNumber=2, fileName=relative, raw=True)
    url = 'https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset'
    subprocess.run(['curl', '-fsSL', '--retry', '3', '--connect-timeout', '20',
                    '--max-time', '600', '--max-filesize', '2147483648',
                    '--header', 'Content-Type: application/json', '--data-binary', json.dumps(request),
                    url, '-o', str(temporary)], check=True)
    with temporary.open('rb') as f:
        magic = f.read(4)
    if magic.startswith(b'PK'):
        with zipfile.ZipFile(temporary) as archive:
            members = [x for x in archive.infolist() if not x.is_dir()]
            if len(members) != 1 or members[0].file_size > 2 * 1024**3:
                raise ValueError('Unexpected single-file download archive')
            with archive.open(members[0]) as src, destination.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        temporary.unlink()
    else:
        temporary.replace(destination)


def evaluate_stages_if_ready(args):
    if args.stages_root is None:
        return
    normalizer = args.work / 'reconstructed_normalization.json'
    output = args.stages_root / 'results/age_conditioned_transfer/reconstructed_source_summary.json'
    if output.exists():
        old = json.loads(output.read_text())
        if old.get('complete') and old.get('normalizer_sha256') == digest(normalizer):
            return
    subprocess.run([sys.executable, str(Path(__file__).with_name('run_reconstructed_rank.py')),
                    '--normalization-json', str(normalizer), '--data-root', str(args.stages_root)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry-source', type=Path, required=True)
    parser.add_argument('--upstream-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--stages-root', type=Path,
                        help='Run unchanged STAGES endpoints only after source reconstruction finishes')
    args = parser.parse_args()
    receipt = frozen_receipt()
    os.environ['SSQ_PARALLEL'] = '0'
    os.environ['NUMBA_NUM_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    source_manifest = json.loads((Path(__file__).parent / 'data/reconstruction_source_manifest.json').read_text())
    for kind, folder in [('entry', args.entry_source), ('upstream', args.upstream_source)]:
        for relative, expected in source_manifest[kind]['files'].items():
            if digest(folder / relative) != expected:
                raise ValueError('Changed submitted/pinned extraction source: ' + relative)
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ['PHILOSOPHER_MODEL_FILE'] = str(args.checkpoint)
    sys.path.insert(0, str(args.upstream_source / 'src'))
    sys.path.insert(0, str(args.entry_source))
    import torch
    torch.set_num_threads(2)
    import team_code as team
    from phi_challenge.downstream import _fit_stable4_artifact, STABLE4_COLUMNS
    from phi_challenge.embed import load_phi_runtime, embed_edf_feature_taps, embed_edf_path
    from phi_challenge.heads import load_head_bank, project_batch
    from reconstruction_wavelet import compute_specs
    import phi_challenge.embed as submitted_embed
    # Preserve the submitted fixed time chunks, CWT scales/frequencies, casting,
    # interpolation and padding; avoid materializing the unused full-resolution
    # synchrosqueezed transform. This execution repair is frozen in amendment 06.
    parity = args.work / 'wavelet_parity.json'
    if not parity.exists():
        raise ValueError('Run the submitted-function wavelet parity checks first')
    checked = json.loads(parity.read_text())
    if checked.get('status') != 'passed' or checked['compact_sha256'] != digest(Path(__file__).with_name('reconstruction_wavelet.py')):
        raise ValueError('Memory-bounded wavelet implementation is not verified')
    def bounded_specs(signals, fs_eeg, cfg, *, chunk_len=1_000_000):
        if fs_eeg != 200:
            raise ValueError('Submitted 200-Hz input required')
        return compute_specs(signals[cfg.channel].to_numpy(), cfg, chunk_len=chunk_len, scale_batch=16)
    submitted_embed.compute_wavelet_specs_oom_safe = bounded_specs
    if digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint SHA256 differs from frozen reconstruction input')
    root = args.work / 'data'
    root.mkdir(exist_ok=True)
    meta = root / 'demographics.csv'
    if not meta.exists():
        download_file('demographics.csv', meta)
    if digest(meta) != META_SHA:
        raise ValueError('Large version-2 metadata differs from frozen snapshot')
    state_path = args.work / 'state_private.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['freeze']['files'] != receipt['files']:
            raise ValueError('Do not resume under changed reconstruction code')
    else:
        state = dict(freeze=receipt, started_utc=receipt['utc'], active_seconds=0., completed=[], failures=[],
                     extraction_budget_seconds=68 * 3600, status='preparing')
    if state['status'] == 'complete':
        print('Reconstruction already complete', flush=True)
        evaluate_stages_if_ready(args)
        return
    started = time.monotonic()
    initial_active = state['active_seconds']
    def save():
        state['active_seconds'] = initial_active + time.monotonic() - started
        atomic(state_path, state)
    records = team.find_patients(str(meta))
    queue_path = args.work / 'queue_private.json'
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
    else:
        # The original queue requires presence of mounted files. Placeholders
        # stand for verified public-release records; real EDF bytes replace one
        # placeholder at a time before any extraction. No empty file is scored.
        for record in records:
            p = root / 'physiological_data' / str(record['SiteID']) / f"{record['BidsFolder']}_ses-{record['SessionID']}.edf"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
        queue = team.build_site_label_round_robin_queue(records, str(root), str(meta), seed=20260721, verbose=True)
        atomic(queue_path, queue)
    state['queue_n'] = len(queue)
    state['status'] = 'loading_model'
    save()
    runtime = load_phi_runtime(verbose=True)
    recent = []
    done = {x['key'] for x in state['completed']}
    failed = {x['key'] for x in state['failures']}
    cache = args.work / 'latents'
    cache.mkdir(exist_ok=True)
    for item in queue:
        key = str(item['record']['BidsFolder']) + '_ses-' + str(item['record']['SessionID'])
        if key in done or key in failed:
            continue
        remaining = state['extraction_budget_seconds'] - (initial_active + time.monotonic() - started)
        if remaining <= max(0., 1.5 * max(recent[-8:], default=0.)):
            state['stop_reason'] = 'original_embedding_budget_or_rolling_guard'
            break
        if shutil.disk_usage(args.work).free < 6 * 1024**3:
            state['status'] = 'blocked_low_disk'
            save()
            raise RuntimeError('Less than 6 GiB free; source reconstruction paused without changing the score')
        tick = time.monotonic()
        path = Path(item['phys_file'])
        relative = 'physiological_data/' + str(item['record']['SiteID']) + '/' + path.name
        try:
            state['status'] = 'downloading'
            state['current_key'] = key
            save()
            download_file(relative, path)
            edf_sha = digest(path)
            state['status'] = 'extracting'
            save()
            taps = None
            try:
                taps = embed_edf_feature_taps(runtime, path, float(item['age']), int(item['sex']), amp=True)
                latent = np.asarray(taps['latent'], np.float32)
                route = 'submitted_CARR_tap_latent'
            except Exception as error:
                latent = np.asarray(embed_edf_path(runtime, path, float(item['age']), int(item['sex']), amp=True), np.float32)
                route = 'submitted_ordinary_fallback:' + type(error).__name__
            if latent.shape != (1024,) or not np.isfinite(latent).all():
                raise ValueError('Invalid latent')
            tap_metadata = {}
            if taps is not None:
                try:
                    tap_metadata = save_feature_taps(args.work, key, taps)
                except Exception as tap_error:
                    tap_metadata = dict(feature_tap_cache_error=type(tap_error).__name__ + ':' + str(tap_error))
                    print(json.dumps(dict(stage='source_tap_cache_failed', reason=type(tap_error).__name__)), flush=True)
            target = cache / (hashlib.sha256(key.encode()).hexdigest() + '.npy')
            np.save(target, latent)
            state['completed'].append(dict(key=key, site=str(item['record']['SiteID']), sex=int(item['sex']),
                                          latent_file=target.name, latent_sha256=digest(target), edf_sha256=edf_sha, route=route,
                                          **tap_metadata))
            done.add(key)
            print(json.dumps(dict(stage='source_record_complete', completed=len(state['completed']),
                                  queue=len(queue), seconds=round(time.monotonic()-tick, 1))), flush=True)
        except subprocess.CalledProcessError:
            state['status'] = 'blocked_download'
            save()
            raise
        except Exception as error:
            state['failures'].append(dict(key=key, error=type(error).__name__ + ':' + str(error)))
            failed.add(key)
            print(json.dumps(dict(stage='source_record_failed', failed=len(state['failures']), reason=type(error).__name__)), flush=True)
        finally:
            if path.exists():
                path.unlink()
            temporary = path.with_suffix('.download')
            if temporary.exists():
                temporary.unlink()
            recent.append(time.monotonic() - tick)
            save()
    if len(state['completed']) < 8:
        state['status'] = 'blocked_insufficient_source_extractions'
        save()
        raise RuntimeError('Fewer than eight valid source latents; no constants or STAGES rank computed')
    state['status'] = 'projecting_source_heads'
    save()
    x = np.vstack([np.load(cache / row['latent_file']) for row in state['completed']])
    head_bank = load_head_bank(args.checkpoint, device='cpu')
    projected = project_batch(head_bank, x)
    names = sorted(projected)
    features = np.nan_to_num(np.column_stack([projected[k] for k in names]).astype(np.float64), nan=0., posinf=0., neginf=0.)
    stable = features[:, [names.index(k) for k in STABLE4_COLUMNS]]
    artifact = _fit_stable4_artifact(stable, np.array([r['sex'] for r in state['completed']]))
    from dataclasses import asdict
    normalizer = asdict(artifact)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.
    roster = args.work / 'completed_source_roster_private.json'
    atomic(roster, state['completed'])
    result = dict(reconstructed=True, original_organizer_artifact=False, entry_commit=ENTRY, upstream_commit=UPSTREAM,
                  checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=META_SHA, dataset=DATASET, dataset_version=2,
                  freeze=state['freeze'], source_n=len(features), active_seconds=state['active_seconds'],
                  source_center_counts=pd.Series([r['site'] for r in state['completed']]).value_counts().to_dict(),
                  source_failures=len(state['failures']), source_roster_sha256=digest(roster),
                  prior_execution_freezes=state.get('prior_execution_freezes', []),
                  retained_feature_tap_records=sum('feature_tap_file' in r for r in state['completed']),
                  feature_tap_backfill_required=sum('feature_tap_file' not in r for r in state['completed']),
                  normalizer=normalizer, all_head_names=names, all_head_mean=features.mean(axis=0).tolist(),
                  all_head_scale=scale.tolist(), packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','scipy']})
    result['memory_bounded_wavelet_verification'] = checked
    result['runtime_threads'] = dict(torch=2, SSQ_PARALLEL='0', NUMBA_NUM_THREADS='2', OMP_NUM_THREADS='2')
    atomic(args.work / 'reconstructed_normalization.json', result)
    state['status'] = 'complete'
    state['normalizer_sha256'] = digest(args.work / 'reconstructed_normalization.json')
    save()
    print(json.dumps(dict(stage='source_reconstruction_complete', source_n=len(features), normalizer_sha256=state['normalizer_sha256'])), flush=True)
    evaluate_stages_if_ready(args)


if __name__ == '__main__':
    main()
