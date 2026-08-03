# Legacy Tangram Evaluation: Reuse Only

This directory archives the evaluation material for the **previous Tangram
implementation**, which supported model/tensor memory reuse but did not yet
implement LayerPipe/LayerWeave pipeline loading.

It is retained for historical result inspection and reproduction only.  It is
not the entry point for the current Tensor-level simulator or the current
real-GPU LayerPipe/LayerWeave implementation.

## Contents

- `0*`: environment, trace generation, and model-format preparation notes.
- `1-overall*` and `1.0*`: legacy Tangram versus baseline loading experiments.
- `1.1*`: VMM, page-size, tensor grouping, reuse, and ODKV experiments from
  the pre-pipeline implementation.
- `2*` and `3*`: latency breakdown and allocation-policy analysis.
- `4*`: locality and model-mapping sensitivity experiments.
- `5*`: old Tangram/Aegaeon comparison artifacts.
- `6*`: the old request-level end-to-end simulator experiment.
- `7-microbenchs.md`: microbenchmark notes associated with that evaluation
  generation.
- `*.log`, `*.json`, `*.cdf`: preserved outputs from those experiments.

## Current implementations

- Real-GPU LayerPipe/LayerWeave:
  `tools/layerpipe/`
- Tensor-level performance simulator:
  `evaluation/tensor_simulator/`

The archived shell scripts still resolve the repository root and can be run
from their new location.  Their default outputs and documentation links have
been updated to remain inside this legacy directory.  They may nevertheless
depend on old runtime behavior and should not be used as evidence for the
current implementation without revalidation.
