# Measured Colab hardware decision and migration

The HSP source reconstruction has resumed on Google Colab, using the official
Colab CLI and a T4 High-RAM runtime. The first new remote record completed at
source count 485. The Mac no longer runs the model, CWT or analysis workers.
Its only ongoing work is a lightweight CLI connection and checkpoint copying.
The 484 completed source records and 130 completed STAGES CARR records were
preserved before the move. STAGES raw-EDF CARR extraction remains stopped locally;
it has not been represented as resumed remotely.

## The faster GPU did not improve the original end-to-end pipeline

Three already completed HSP recordings, one from each source center, were run
on both machines. No cognitive result was used for hardware selection.

| Measurement | T4 High-RAM | A100 High-RAM |
|---|---:|---:|
| GPU | Tesla T4 | A100-SXM4-80GB |
| System RAM | 51.0 GiB | 167.1 GiB |
| vCPUs | 8 | 12 |
| Median GPU forward pass | 0.327 seconds | 0.057 seconds |
| Median complete record, original serial CPU CWT | 121.58 seconds | 121.96 seconds |
| Every predefined latent/head parity gate passed | Yes | No |

The A100 sped up the forward pass but did not reduce total processing time.
CPU wavelet computation dominated. The account rejected H100 allocation. The
A100 was released after its benchmark; only the T4 runtime was retained.
These three-record timings are an execution benchmark, not a precise estimate
for the full source cohort. Download times and first-use warmup varied.

## Numerical checks and rejected variants

The same checkpoint SHA256 and pinned entry/upstream source were used throughout.
Inference used float32, with autocast and TF32 disabled. On T4, latent relative
L2 errors versus the existing CPU cache were 3.35e-7, 3.75e-7 and 2.18e-5, all
below the fixed 1e-4 gate. The maximum four-head difference was 6.46e-6 of its
source SD, below the 1e-3 gate.

A100 failed the latent gate on the S0001 recording (relative L2 3.76e-4), although
the four-head discrepancy remained small. Its forward-inference path was not
adopted. This finding is numerical; no effect on cognitive discrimination was
tested or inferred from it.

A separate CUDA-CWT experiment reduced one transform from 115.51 to 10.73
seconds, but failed the predeclared elementwise spectrogram tolerance. Its
relative L2 difference was 6.01e-7, while maximum absolute difference was
1.67e-4. That variant was rejected rather than silently changing the gate.

Increasing the CPU CWT scale batch from 16 to 128 produced exactly the same
spectrogram, with a timing of 101.08 versus 115.51 seconds on the A100 test.

## Adopted acceleration: exact parallel CPU chunks on T4

Three persistent CPU workers process the existing independent fixed-length CWT
chunks, retaining their original order and every transformation. In two checks
on the same I0006 recording, the resulting spectrogram was exactly equal to the
saved original reference. Cold and warm parallel times were 42.63 and 33.83
seconds. The original T4 serial-CWT time for this recording was 104.68 seconds,
so warm parallel execution was approximately 3.1 times faster for that step.
The 115.51-second reference timing was on A100; comparing that number with T4's
parallel time yields 3.4 times, but also includes differences between machines.
Neither ratio is a measured full-pipeline speedup.

The running source job uses this exact parallel CPU transform and the validated
T4 float32 forward inference. The original queue order, labels, heads, weights,
68-hour active-time budget and all STAGES outcome definitions remain unchanged.
No new cognitive rank result was computed during hardware selection.

## Checkpoints and scientific status

The source job exports a private resumable archive every 25 completed records.
A lightweight local process copies those archives and verifies their stored
payload hashes. It retrieves the final normalization and STAGES result when
complete and releases this runtime after successful final artifact checks.
Colab runtimes are ephemeral; a runtime interruption may still require resuming
from the newest copied checkpoint. The original trained organizer artifact
remains unavailable; the output will be labeled a reconstruction.

Current runtime is `nsrr-hsp`. Remote source state is
`/content/nsrr/work/state_private.json`; the periodically downloaded local copy
is `/Users/admin/.cache/nsrr-stages/colab_live_state_private.json`. The old
`source-reconstruction-76a36e5/state_private.json` is an archived local state at
484 records and must not be used as live remote progress. The private
`execution_location.json` beside it records this distinction.

Aggregate measurements and rejected variants are in
[`results/aggregate/colab_hardware_benchmark/`](results/aggregate/colab_hardware_benchmark/).
The exact implementation is `parallel_reconstruction_wavelet.py`, and the
scientific reconstruction code remains unchanged. Google CLI version 0.7.5
was used with its official Google Jupyter-kernel client implementation; a
trusted CA bundle was configured for TLS. Colab used Python 3.13.16 and
PyTorch 2.11.0+cu130, with the same recorded NumPy, SciPy, MNE and ssqueezepy
versions as the numerical checks. See the exported JSON for all versions.

Official execution tool: [Google Colab CLI](https://github.com/googlecolab/google-colab-cli).
