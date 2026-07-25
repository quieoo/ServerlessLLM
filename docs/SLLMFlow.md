# Motivation
- 当前的计算-加载流水线不能总是掩盖加载时延
- 显存内存在可重用的参数
- ODKV可以进一步释放显存

# Design
- 基于VMM的参数重用
- 重用-计算-加载流水线
- 调度策略，缓存策略
    这应该是这个方案主要smart的地方
    调度策略：
        已知：
            1. 模型实例当前队列中存在的请求的总输入大小，一些离线Profile确定的信息，可以得到这批输入+这个模型+目标GPU上面执行Prefill的每一层所需的时间
            2. 目标模型在每个GPU上面缓存了多少层参数
        可以估计模型调度到目标GPU上是否能近乎无损地加载，或者不能的话，加载时延可能是多少；或者如果输入本身很长，模型本身就没有加载时延，那么它调度到哪个GPU上面对当前模型来说就没有区别
        但是即使确定了将当前模型的加载时延，模型调度对目标GPU上其它缓存的模型的影响也不同。
        因此调度策略应该如何设计。
    缓存策略：
        当模型实例被分配一个GPU之后，此时空间不足，需要evict其它参数，eviction策略应该是load-aware的，比如优先将不热的模型，后续层的参数驱逐，这里需要trade的是模型冷热以及层位置的价值的权重，是优先保留热模型的所有参数，还是优先保留模型的前几层参数以换取未来更好的加载掩盖；
        同时可能也需要考虑不同模型，不同模型普遍在输入长度上面拥有不同的特征和规律。可能会发现有些模型的输入普遍较长，此时它保留前面的几层就可以了。


# Baselines
    - 纯计算-加载流水线
        模型切换，清理上一个模型，没有任何重用；
        做当前层的Prefill的同时预先加载下一层模型的参数
    - 纯 chunk-level重用
        建立VMM-based显存池，缓存其它模型的参数chunk
        调度时候看当前可用GPU里面缓存参数量最多的
    - 预取（Aegaeon）
        如果当前模型运行时显存还有剩余空间，提前加载调度器队首的下一个模型


# 实验

## 准备模型

将当前本地的rank_0/tensor.data_*这种 vLLM 打包格式转换成所需要的Hugging Face safetensors

转换命令：
```bash
cd /mnt/n0/Tangram/Tangram
/home/sdu/.conda/envs/sllm-worker/bin/python \
  tools/layerpipe/convert_servegen_config.py \
  --config configs/servegen_8_models.json \
  --output-root /mnt/n0/models/hf/servegen_8_models \
  --output-config configs/servegen_8_models_layerpipe.json \
  --max-shard-size 2GiB
```

## baseline-naive
```bash
cd /mnt/n0/Tangram/Tangram

nohup env \
    CONFIG_PATH=configs/servegen_8_models_layerpipe.json \
    LOAD_MODE=native \
    MAX_REQUESTS=100 \
    MAX_BATCH_SIZE=1 \
    OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 \
    bash docs/1.2-layerpipe.sh \
    >> docs/1.2-layerpipe.log 2>&1 &
echo $!
```

## baseline-layerpipe
```bash
nohup env \
    CONFIG_PATH=configs/servegen_8_models_layerpipe.json \
    LOAD_MODE=layerpipe \
    MAX_REQUESTS=100 \
    MAX_BATCH_SIZE=1 \
    OUTPUT_TOKENS_OVERRIDE=1 \
    TRACE_TIME_SCALE=150 \
    bash docs/1.2-layerpipe.sh \
    >> docs/1.2-layerpipe.log 2>&1 &
echo $!
```
关键参数：
- MAX_BATCH_SIZE 限制一次从等待队列中抽取多少个“同模型请求”组成 batch。=0表示不限制

## baseline-vmm-reuse

```bash
cd /mnt/n0/Tangram/Tangram

CONFIG_PATH=configs/servegen_8_models_layerpipe.json \
LOAD_MODE=vmm \
VMM_POOL_GIB=40 \
VMM_PAGE_SIZE_MIB=0 \
MAX_REQUESTS=100 \
MAX_BATCH_SIZE=2 \
TRACE_TIME_SCALE=150 \
OUTPUT_TOKENS_OVERRIDE=1 \
bash docs/1.2-layerpipe.sh \
2>&1 | tee docs/1.2-vmm.log
```