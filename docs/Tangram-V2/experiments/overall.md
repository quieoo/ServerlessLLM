# Overall System Comparison

## Question

How much do pipeline loading, reuse, and Tangram's joint design reduce exposed
loading and end-to-end latency relative to the baseline systems?

The five systems are:

1. Baseline
2. Pipe-only
3. Reuse-only with cache-bytes routing and LRU
4. Aegaeon
5. Tangram with Prefix-MCKP and transition-aware routing

The comparison uses 4 GPUs, a 42 GiB pool per GPU, ServeGen model switches,
5000 requests, GPU busy probability 0.5, and availability seed 1234. Aegaeon
uses output length from the trace.

## Run

```bash
bash docs/Tangram-V2/1.overall.sh
```

Outputs:

```text
evaluation/tensor_simulator/results/end2end/corrected-busyp0p5-req5000/
```

The launcher executes the five systems concurrently and writes one JSON, CSV,
and log per system.

## Report

Report mean and p95 exposed loading stall, TTFT, H2D traffic, and SLO
attainment. Keep the system configuration and trace identical across systems.

