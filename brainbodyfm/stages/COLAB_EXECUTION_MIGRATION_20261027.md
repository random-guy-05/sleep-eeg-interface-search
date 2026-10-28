# Execution migration and hardware benchmark

October 27, 2026, after the user requested moving computation off the local Mac
because RAM use was excessive. All local model processes were stopped. The
saved HSP reconstruction has 484 records; the STAGES CARR extraction has 130.
Existing outputs, code hashes, outcome definitions and inference plans remain
unchanged. The original 68-hour active extraction budget is retained.

The official Google Colab CLI is used. H100 allocation was rejected by the
account backend. A T4 High-RAM runtime and an A100 High-RAM runtime were allocated
for an execution-only benchmark. No cognitive metrics are used to choose hardware.
The expensive runtime is retained only if the full extraction is materially
faster, including its CPU preprocessing. Record the GPU, RAM, vCPU count,
software versions, timing breakdown, warmup and matched input checks.

Choose the first already completed record from each source center in the saved
queue. Use these same three recordings on both runtimes. Time download,
EDF reading/filtering, exact bounded CWT preprocessing and forward inference
separately. Include warmup separately; do not count installation or checkpoint
download as steady-state throughput. Record full-precision GPU and CPU forward
timings on the same spectrograms. Compare GPU latents and the four projected
Stable4 heads with existing CPU-cache values. Do not inspect any STAGES rank
metric during this hardware evaluation.

Preserve the existing checkpoint SHA256 and all pinned submitted/upstream source
hashes. HSP queue order, membership and relative filenames are unchanged; only
the local filesystem prefix is mapped to `/content/nsrr/work/data`. Keep an
explicit original snapshot. Existing source latents are transferred as a small
archive; the 2.4-GB checkpoint is downloaded directly to Colab from the original
publisher and checked against the frozen hash, avoiding a large local RAM copy.

Use full-precision GPU inference, with autocast disabled, to match the numerical
precision of the prior CPU run. Disable TF32. A benchmark parity gate requires
latent relative L2 <=1e-4 and the maximum absolute Stable4 head discrepancy
<=1e-3 of its SD in the 484 cached source records. If a gate fails, stop for
numerical investigation rather than resume under a changed input pipeline.
Software changes from the Mac environment are recorded. These tolerances assess
numerical transfer; they are not evidence about cognitive prediction.

For hardware choice, compare median steady-state end-to-end time on matched
records. An A100 speedup of at least 20% over T4 is the fixed criterion for
retaining the A100; otherwise retain T4 and release A100. Report uncertainty from
this small benchmark. If remote CPU preprocessing dominates and a CPU forward
is competitive, report that rather than attributing throughput to the GPU.

Colab runtimes are ephemeral. Preserve source checkpoint bundles and execution
receipts outside the VM, without changing restricted-data sharing permissions.
Do not mark local saved artifacts as Colab results before remote execution and
the parity check actually pass. The original reconstruction remains a
reconstruction, not recovery of the organizer-trained artifact.
