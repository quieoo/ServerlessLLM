code area: 
````bash
````
# Setup codes
- Get source codes
````bash
git clone https://github.com/quieoo/ServerlessLLM.git
or
git clone https://gitee.com/quieoo/ServerlessLLM.git
git checkout sllm-0.6-dev-1

git clone https://github.com/quieoo/vllm.git
git checkout vllm-0.5.1-dev
````

- Create conda envs
````bash
conda create -n sllm-0.6 python=3.10 -y
conda create -n sllm-worker-0.6 python=3.10 -y
````

- Install
````bash
conda activate sllm-0.6
cd ServerlessLLM
pip install .

conda activate sllm-worker-0.6
cd ServerlessLLM
pip install .
cd ServerlessLLM/sllm_store
./rebuild.sh

cd vllm
pip install .
````

# Prepare Models

````bash
cd ServerlessLLM/examples/sllm_store/
python save_vllm_model.py --model_name llama8b --local_model_path /mnt/n0/models/llama3.18b_intruc_chinese/ --storage_path /mnt/n0/models/vllm/
````