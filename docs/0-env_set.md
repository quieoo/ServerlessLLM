
````bash
mkdir /mnt/n0/Tangram
cd /mnt/n0/Tangram
git clone https://github.com/quieoo/Tangram.git
````

# Single-Node Test
创建运行mock_allocation，测试Tangram在模型加载上的效率。


## Prepare
````bash
sudo apt-get update
# minimumal dependencies
sudo apt-get install cmake g++ nlohmann-json3-dev libboost-dev

# assume CUDA is installed
nvidia-smi

# if nvcc not installed
    # assume CUDA Version: 12.4
sudo apt-get install -y cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
````

## Build
````bash
cd /mnt/n0/Tangram/Tangram/tools/mock_allocation/
cmake -S . -B build
cmake --build build --target Allocateion -j
````

# End-to-End Test

- Create conda envs
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
cd Tangram/sllm_store
./rebuild.sh
cd Tangram/ElasticKV
pip install .
````