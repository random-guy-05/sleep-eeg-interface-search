# Exact parallel CPU CWT execution

The completed three-record benchmark found median full extraction times of
121.96 seconds on the A100 and 121.58 seconds on T4. The A100 is not faster
for the present implementation. T4 forward inference passes every numerical
gate; A100 forward inference failed the predeclared latent gate for S0001.
Retain T4 and release A100 after retrieving its timing reports.

GPU CWT reduced one transform from 115.51 to 10.73 seconds, but failed the
predeclared elementwise spectrogram tolerance, despite small global error.
Do not adopt it. CPU scale batch 128 produced exactly identical spectrograms
and reduced that transform to 101.08 seconds. Retain that exact batch variant.

Before further extraction, benchmark parallel CPU execution of the same fixed
independent one-million-sample chunks using three persistent spawned workers.
Preserve original chunk order, padding, per-chunk float32 conversion, frequency
grid, interpolation, and final dtype. Compare its full spectrogram to the saved
CPU batch-16 reference on the same I0006 recording. Require exact array equality;
if it fails, use the already verified serial batch-128 implementation. No new
outcome or score definition is evaluated. This changes execution scheduling,
not the transform. Record cold and warm times, remote memory and worker count.

All processes and large arrays remain on Colab; the local Mac keeps only the
CLI connection and small private checkpoint archives. The prior 484-source
record checkpoint and 68-hour active runtime budget are preserved. The rest of
the source queue and every submitted scientific rule remain unchanged.
