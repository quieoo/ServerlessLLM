
# Tangram

A serverless LLM framework that accelerates model loading through GPU memory reuse and affinity. 
This repo contains the source code of Tangram and ElasticKV (a Inference Engine that supports on-demand KV cache allocation) and evaluation scripts for the paper: **Tangram: Accelerating Serverless LLM Loading through GPU Memory Reuse and Affinity**

<p align="center">
  <img src="Tangram.svg" alt="Tangram" width=95%>
</p>

---

## System Requirements
- CUDA Version: 12.4
- Python Version: 3.10

NOTE: other versions of CUDA and Python may also work, but we have not tested them.

## Getting Started


- Create envs on Controller Node and Worker Nodes, respectively
````bash
conda create -n sllm-0.6 python=3.10 -y
conda create -n sllm-worker-0.6 python=3.10 -y
````

- Install
````bash
conda activate sllm-0.6
cd Tangram
pip install .

conda activate sllm-worker-0.6
cd Tangram
pip install .
cd sllm_store
./rebuild.sh

cd Tangram/ElasticKV
pip install .
````
Details of installation and errors can be found in [EnvironmentSetup.md](evaluation/EnvironmentSetup.md)

## Performance Benchmarks


The latest benchmark manuals and scripts are maintained in `docs/`, including:
- [Environment setup and build guide](docs/0-env_set.md)
- [Trace generation guide](docs/0.1-trace_gen.md)
- [Model format conversion guide](docs/0.2-model_format.md)
- Overall loading benchmark: [Tangram script](docs/1-overall.sh), [baseline script](docs/1.0-overall_baseline.sh)
- [Breakdown benchmark script](docs/2-breakdown.sh)
- [Allocation policy analysis script](docs/3-analysis_allocation.sh)
- Sensitivity benchmarks: [locality script](docs/4-sensitivity_locality.sh), [mapping script](docs/4-sensitivity_mapping.sh)
- Baseline comparison: [Aegaeon script](docs/5-Aegaeon.sh), [Tangram script](docs/5.1-Tangram.sh)
- End-to-end evaluation: [manual](docs/6-end2end.md), [simulation script](docs/6-end2end-sim.sh)

## Detailed Instructions
A detailed instruction for creating CRIU checkpoints and restoring models can be found in [quick_start.md](evaluation/quick_start.md)

## Acknowledgements
This project is a fork of [ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM), and I would like to thank the original author(s) for their amazing work. You can also cite the original paper: ServerlessLLM: Low-Latency Serverless Inference for Large Language Models.
