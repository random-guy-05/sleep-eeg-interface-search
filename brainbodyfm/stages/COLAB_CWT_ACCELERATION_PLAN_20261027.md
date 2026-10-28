# CWT execution benchmark, after profiling and before new cognitive metrics

The first two A100 records show that CPU wavelet preprocessing dominates elapsed
time; network forward inference is approximately 0.05 seconds. A faster neural
network GPU alone therefore offers little total speedup. This execution-only
benchmark tests the actual bottleneck without changing outcomes or model inputs.

On the first queued I0006 recording, compare the original CPU scale batch of 16,
a CPU batch of 128 and a CUDA CWT batch of 64, using the exact existing bounded
CWT implementation with its fixed scales, chunks, amplitude, interpolation and
padding. Changing batch width or the array execution backend must not change the
algorithm. The backend is selected by ssqueezepy's `SSQ_GPU` environment variable
in a fresh remote process. Time one complete CWT per variant; report process
startup separately and disclose possible warmup effects. No cognitive outcome is
read. Use the same filtered EEG array for all variants.

Require all spectrogram values to agree with the original CPU implementation at
rtol 2e-5 and atol 2e-6, the previously used preprocessing-validation tolerances.
Report maximum absolute and relative L2 differences and all failures. An
accelerated variant cannot be adopted on speed alone. If passing, separately
verify latent and Stable4-head parity using the same gates in the migration
plan. Do not relax a failed gate after looking at cognition or substitute a
changed transform to obtain a faster run. Preserve all original outputs.
