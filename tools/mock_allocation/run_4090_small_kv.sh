./build/Allocateion -g 20 -m 100 -r guas -s 40 -p 4 --affinity --gpu 2 --regenerate --config configs/4090-small.json -kv 0.5 --block 8
./build/Allocateion -g 20 -m 100 -r guas -s 40 -p 4 --affinity --gpu 2 --config configs/4090-small.json -kv 0.5 --block 16
./build/Allocateion -g 20 -m 100 -r guas -s 40 -p 4 --affinity --gpu 2 --config configs/4090-small.json -kv 0.5 --block 32
