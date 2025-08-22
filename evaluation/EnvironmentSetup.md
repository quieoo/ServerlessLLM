code area: 
````bash
````
# Setup codes

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
python save_vllm_model.py --model_name llama8b --local_model_path /home/zhchen/zwb/models/llama3.18b_intruc_chinese/ --storage_path /home/zhchen/zwb/models/vllm/

python save_vllm_model.py --model_name qwen7b_tmp --local_model_path /mnt/n0/models/qwen7b/ --storage_path /mnt/n0/models/vllm/

````

# Start Clusters
make sure all commands running in the path with 'models' exists

````bash
# controller node
conda activate sllm-0.6
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --head --port=6379 --num-cpus=16 --num-gpus=0 --resources='{"control_node": 1}' --block

# worker nodes
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=0
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1, "store_port":8073}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_1": 1, "store_port":8074}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_2": 1, "store_port":8075}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=3
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_3": 1, "store_port":8076}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_4": 1, "store_port":8077}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=5
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_5": 1, "store_port":8078}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=6
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_6": 1, "store_port":8079}' --block

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=7
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
ray start --address=0.0.0.0:6379 --num-cpus=16 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_7": 1, "store_port":8080}' --block

# start the Stores
conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=0
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8073

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=1
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8074

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=2
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8075

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=3
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8076

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=4
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8077

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=5
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8078

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=6
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8079

conda activate sllm-worker-0.6
export CUDA_VISIBLE_DEVICES=7
sllm-store start  --mem-pool-size 2GB --chunk-size 0B --num-thread 0 --port 8080


# SLLM controller


conda activate sllm-0.6
export RAY_TMPDIR=/home/zhchen/zwb/ray_tmp
sllm-serve start --enable_storage_aware
````


# Collect insertions 
````bash
# sllm
git diff --shortstat 58992d130831f03fd3fe977195e312087a13145d HEAD -- . ':(exclude)evaluation/' ':(exclude)tools/'
# vllm
git diff --shortstat 50eed24d252965a81ce50b64fd387d60fb1f4f6e HEAD -- . ':(exclude)CMakeFiles/'
````

# Seen Installing Errors

Error-1

````
-- Could NOT find Unwind (missing: Unwind_INCLUDE_DIR Unwind_LIBRARY)
  -- Looking for C++ include dlfcn.h
  -- Looking for C++ include dlfcn.h - found
  -- Looking for C++ include elf.h
  -- Looking for C++ include elf.h - found
  -- Looking for C++ include glob.h
  -- Looking for C++ include glob.h - found
  -- Looking for C++ include link.h
  -- Looking for C++ include link.h - found
  -- Looking for C++ include pwd.h
  -- Looking for C++ include pwd.h - found
  -- Looking for C++ include sys/exec_elf.h
  -- Looking for C++ include sys/exec_elf.h - not found
  -- Looking for C++ include sys/syscall.h
  -- Looking for C++ include sys/syscall.h - found
  -- Looking for C++ include sys/time.h
  -- Looking for C++ include sys/time.h - found
  -- Looking for C++ include sys/types.h
  -- Looking for C++ include sys/types.h - found
  -- Looking for C++ include sys/utsname.h
  -- Looking for C++ include sys/utsname.h - found
  -- Looking for C++ include sys/wait.h
  -- Looking for C++ include sys/wait.h - found
  -- Looking for C++ include syscall.h
  -- Looking for C++ include syscall.h - found
  -- Looking for C++ include syslog.h
  -- Looking for C++ include syslog.h - found
  -- Looking for C++ include ucontext.h
  -- Looking for C++ include ucontext.h - found
  -- Looking for C++ include unistd.h
  -- Looking for C++ include unistd.h - found
  -- Looking for C++ include stdint.h
  -- Looking for C++ include stdint.h - found
  -- Looking for C++ include stddef.h
  -- Looking for C++ include stddef.h - found
  -- Check size of mode_t
  -- Check size of mode_t - done
  -- Check size of ssize_t
  -- Check size of ssize_t - done
  -- Looking for dladdr
  -- Looking for dladdr - found
  -- Looking for fcntl
  -- Looking for fcntl - found
  -- Looking for posix_fadvise
  -- Looking for posix_fadvise - found
  -- Looking for pread
  -- Looking for pread - found
  -- Looking for pwrite
  -- Looking for pwrite - found
  -- Looking for sigaction
  -- Looking for sigaction - found
  -- Looking for sigaltstack
  -- Looking for sigaltstack - found
  -- Looking for backtrace
  -- Looking for backtrace - found
  -- Looking for backtrace_symbols
  -- Looking for backtrace_symbols - found
  -- Looking for _chsize_s
  -- Looking for _chsize_s - not found
  -- Looking for UnDecorateSymbolName
  -- Looking for UnDecorateSymbolName - not found
  -- Looking for abi::__cxa_demangle
  -- Looking for abi::__cxa_demangle - found
  -- Looking for __argv
  -- Looking for __argv - not found
  -- Looking for getprogname
  -- Looking for getprogname - not found
  -- Looking for program_invocation_short_name
  -- Looking for program_invocation_short_name - found
  -- Performing Test HAVE___PROGNAME
  -- Performing Test HAVE___PROGNAME - Success
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_PC
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_PC - Failed
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_EIP
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_EIP - Failed
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_RIP
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_RIP - Success
  -- Looking for gmtime_r
  -- Looking for gmtime_r - found
  -- Looking for localtime_r
  -- Looking for localtime_r - found
  -- Performing Test COMPILER_HAS_HIDDEN_VISIBILITY
  -- Performing Test COMPILER_HAS_HIDDEN_VISIBILITY - Success
  -- Performing Test COMPILER_HAS_HIDDEN_INLINE_VISIBILITY
  -- Performing Test COMPILER_HAS_HIDDEN_INLINE_VISIBILITY - Success
  -- Performing Test COMPILER_HAS_DEPRECATED_ATTR
  -- Performing Test COMPILER_HAS_DEPRECATED_ATTR - Success
  -- CUDA found
  -- The CUDA compiler identification is NVIDIA 11.5.119 with host compiler GNU 11.4.0
  -- Detecting CUDA compiler ABI info
  -- Detecting CUDA compiler ABI info - done
  -- Check for working CUDA compiler: /usr/bin/nvcc - skipped
  -- Detecting CUDA compile features
  -- Detecting CUDA compile features - done
  -- Found Python: /home/zhchen/miniconda3/envs/sllm-worker-0.6/bin/python3.10 (found version "3.10.18") found components: Interpreter Development.Module
  -- Found python matching: /home/zhchen/miniconda3/envs/sllm-worker-0.6/bin/python3.10.
  -- Found CUDA: /usr (found version "11.5")
  -- Found CUDAToolkit: /usr/include (found version "11.5.119")
  -- Caffe2: CUDA detected: 11.5
  -- Caffe2: CUDA nvcc is: /usr/bin/nvcc
  -- Caffe2: CUDA toolkit directory: /usr
  -- Caffe2: Header version is: 11.5
  -- /usr/lib/x86_64-linux-gnu/libnvrtc.so shorthash is 65f2c18b
  -- USE_CUDNN is set to 0. Compiling without cuDNN support
  -- USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support
  -- Autodetected CUDA architecture(s):  8.6+PTX 8.6+PTX 8.6+PTX 8.6+PTX 8.6+PTX 8.6+PTX 8.6+PTX 8.6+PTX
  -- Added CUDA NVCC flags for: -gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_86,code=compute_86
  CMake Warning at /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
    static library kineto_LIBRARY-NOTFOUND not found.
  Call Stack (most recent call first):
    /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
    CMakeLists.txt:87 (find_package)


  -- Found Torch: /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/lib/libtorch.so
  -- CUDA supported arches: 7.0;7.5;8.0;8.6;8.9;9.0
  -- CUDA target arches: 86-real;86-virtual
  -- Configuring done (22.5s)
  -- Generating done (0.1s)
  -- Build files have been written to: /home/zhchen/zwb/ServerlessLLM/sllm_store/build/temp.linux-x86_64-cpython-310
  Change Dir: '/home/zhchen/zwb/ServerlessLLM/sllm_store/build/temp.linux-x86_64-cpython-310'

  Run Build Command(s): /tmp/pip-build-env-9n6335f5/overlay/bin/ninja -v -j 128 _C
  [1/5] /usr/bin/g++-9 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc -isystem /home/zhchen/miniconda3/envs/sllm-worker-0.6/include/python3.10 -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -O3 -DNDEBUG -std=gnu++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_C.dir/csrc/checkpoint/aligned_buffer.cpp.o -MF CMakeFiles/_C.dir/csrc/checkpoint/aligned_buffer.cpp.o.d -o CMakeFiles/_C.dir/csrc/checkpoint/aligned_buffer.cpp.o -c /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp: In destructor ‘AlignedBuffer::~AlignedBuffer()’:
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp:38:11: warning: ignoring return value of ‘ssize_t pwrite(int, const void*, size_t, __off_t)’, declared with attribute warn_unused_result [-Wunused-result]
     38 |     pwrite(fd_, buffer_, buf_pos_, file_offset_);
        |     ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp: In member function ‘size_t AlignedBuffer::writeData(const void*, size_t)’:
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp:59:19: warning: ignoring return value of ‘char* strerror_r(int, char*, size_t)’, declared with attribute warn_unused_result [-Wunused-result]
     59 |         strerror_r(errno, err_msg, sizeof(err_msg));
        |         ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp: In member function ‘size_t AlignedBuffer::writePadding(size_t)’:
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/aligned_buffer.cpp:108:11: warning: ignoring return value of ‘ssize_t pwrite(int, const void*, size_t, __off_t)’, declared with attribute warn_unused_result [-Wunused-result]
    108 |     pwrite(fd_, buffer_, buf_pos_, file_offset_);
        |     ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  [2/5] /usr/bin/g++-9 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc -isystem /home/zhchen/miniconda3/envs/sllm-worker-0.6/include/python3.10 -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -O3 -DNDEBUG -std=gnu++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_C.dir/csrc/checkpoint/tensor_writer.cpp.o -MF CMakeFiles/_C.dir/csrc/checkpoint/tensor_writer.cpp.o.d -o CMakeFiles/_C.dir/csrc/checkpoint/tensor_writer.cpp.o -c /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/tensor_writer.cpp
  [3/5] /usr/bin/g++-9 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc -isystem /home/zhchen/miniconda3/envs/sllm-worker-0.6/include/python3.10 -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -O3 -DNDEBUG -std=gnu++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_C.dir/csrc/checkpoint/checkpoint_py.cpp.o -MF CMakeFiles/_C.dir/csrc/checkpoint/checkpoint_py.cpp.o.d -o CMakeFiles/_C.dir/csrc/checkpoint/checkpoint_py.cpp.o -c /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/checkpoint_py.cpp
  [4/5] /usr/bin/nvcc -forward-unknown-to-host-compiler -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc -isystem /home/zhchen/miniconda3/envs/sllm-worker-0.6/include/python3.10 -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -DONNX_NAMESPACE=onnx_c2 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_86,code=[sm_86]" "--generate-code=arch=compute_86,code=[compute_86]" -Xcompiler=-fPIC -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o -MF CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o.d -x cu -c /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/checkpoint.cu -o CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o
  FAILED: CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o
  /usr/bin/nvcc -forward-unknown-to-host-compiler -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc -isystem /home/zhchen/miniconda3/envs/sllm-worker-0.6/include/python3.10 -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include -isystem /tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -DONNX_NAMESPACE=onnx_c2 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_86,code=[sm_86]" "--generate-code=arch=compute_86,code=[compute_86]" -Xcompiler=-fPIC -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o -MF CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o.d -x cu -c /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/checkpoint.cu -o CMakeFiles/_C.dir/csrc/checkpoint/checkpoint.cu.o
  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/checkpoint/checkpoint.cu(341): warning #177-D: variable "pos" was declared but never referenced

  /usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’:
    435 |         function(_Functor&& __f)
        |                                                                                                                                                 ^
  /usr/include/c++/11/bits/std_function.h:435:145: note:         ‘_ArgTypes’
  /usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
    530 |         operator=(_Functor&& __f)
        |                                                                                                                                                  ^
  /usr/include/c++/11/bits/std_function.h:530:146: note:         ‘_ArgTypes’
  ninja: build stopped: subcommand failed.

  Traceback (most recent call last):
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
      main()
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
      json_out["return_val"] = hook(**hook_input["kwargs"])
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 280, in build_wheel
      return _build_backend().build_wheel(
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 435, in build_wheel
      return _build(['bdist_wheel', '--dist-info-dir', str(metadata_directory)])
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 423, in _build
      return self._build_with_temp_dir(
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 404, in _build_with_temp_dir
      self.run_setup()
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
      exec(code, locals())
    File "<string>", line 223, in <module>
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/__init__.py", line 115, in setup
      return distutils.core.setup(**attrs)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 186, in setup
      return run_commands(dist)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
      dist.run_commands()
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
      self.run_command(cmd)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
      super().run_command(command)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
      cmd_obj.run()
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/command/bdist_wheel.py", line 370, in run
      self.run_command("build")
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
      self.distribution.run_command(command)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
      super().run_command(command)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
      cmd_obj.run()
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/command/build.py", line 135, in run
      self.run_command(cmd_name)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
      self.distribution.run_command(command)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
      super().run_command(command)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
      cmd_obj.run()
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 96, in run
      _build_ext.run(self)
    File "/tmp/pip-build-env-9n6335f5/overlay/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 368, in run
      self.build_extensions()
    File "<string>", line 214, in build_extensions
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/subprocess.py", line 369, in check_call
      raise CalledProcessError(retcode, cmd)
  subprocess.CalledProcessError: Command '['cmake', '--build', '.', '--target', '_C', '-j', '128']' returned non-zero exit status 1.
  error: subprocess-exited-with-error
  
  × Building wheel for serverless-llm-store (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  full command: /home/zhchen/miniconda3/envs/sllm-worker-0.6/bin/python3.10 /home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py build_wheel /tmp/tmpf8ccska6
  cwd: /home/zhchen/zwb/ServerlessLLM/sllm_store
  Building wheel for serverless-llm-store (pyproject.toml) ... error
  ERROR: Failed building wheel for serverless-llm-store
Failed to build serverless-llm-store
ERROR: Failed to build installable wheels for some pyproject.toml based projects (serverless-llm-store)
````

````
CMake Warning at build/temp.linux-x86_64-cpython-310/_deps/glog-src/CMakeLists.txt:77 (find_package):
    By not providing "Findgflags.cmake" in CMAKE_MODULE_PATH this project has
    asked CMake to find a package configuration file provided by "gflags", but
    CMake did not find one.

    Could not find a package configuration file provided by "gflags" (requested
    version 2.2.2) with any of the following names:

      gflagsConfig.cmake
      gflags-config.cmake

    Add the installation prefix of "gflags" to CMAKE_PREFIX_PATH or set
    "gflags_DIR" to a directory containing one of the above files.  If "gflags"
    provides a separate development package or SDK, be sure it has been
    installed.


  -- Could NOT find Unwind (missing: Unwind_INCLUDE_DIR Unwind_LIBRARY)
  -- Looking for C++ include dlfcn.h
  -- Looking for C++ include dlfcn.h - found
  -- Looking for C++ include elf.h
  -- Looking for C++ include elf.h - found
  -- Looking for C++ include glob.h
  -- Looking for C++ include glob.h - found
  -- Looking for C++ include link.h
  -- Looking for C++ include link.h - found
  -- Looking for C++ include pwd.h
  -- Looking for C++ include pwd.h - found
  -- Looking for C++ include sys/exec_elf.h
  -- Looking for C++ include sys/exec_elf.h - not found
  -- Looking for C++ include sys/syscall.h
  -- Looking for C++ include sys/syscall.h - found
  -- Looking for C++ include sys/time.h
  -- Looking for C++ include sys/time.h - found
  -- Looking for C++ include sys/types.h
  -- Looking for C++ include sys/types.h - found
  -- Looking for C++ include sys/utsname.h
  -- Looking for C++ include sys/utsname.h - found
  -- Looking for C++ include sys/wait.h
  -- Looking for C++ include sys/wait.h - found
  -- Looking for C++ include syscall.h
  -- Looking for C++ include syscall.h - found
  -- Looking for C++ include syslog.h
  -- Looking for C++ include syslog.h - found
  -- Looking for C++ include ucontext.h
  -- Looking for C++ include ucontext.h - found
  -- Looking for C++ include unistd.h
  -- Looking for C++ include unistd.h - found
  -- Looking for C++ include stdint.h
  -- Looking for C++ include stdint.h - found
  -- Looking for C++ include stddef.h
  -- Looking for C++ include stddef.h - found
  -- Check size of mode_t
  -- Check size of mode_t - done
  -- Check size of ssize_t
  -- Check size of ssize_t - done
  -- Looking for dladdr
  -- Looking for dladdr - not found
  -- Looking for fcntl
  -- Looking for fcntl - found
  -- Looking for posix_fadvise
  -- Looking for posix_fadvise - found
  -- Looking for pread
  -- Looking for pread - found
  -- Looking for pwrite
  -- Looking for pwrite - found
  -- Looking for sigaction
  -- Looking for sigaction - found
  -- Looking for sigaltstack
  -- Looking for sigaltstack - found
  -- Looking for backtrace
  -- Looking for backtrace - found
  -- Looking for backtrace_symbols
  -- Looking for backtrace_symbols - found
  -- Looking for _chsize_s
  -- Looking for _chsize_s - not found
  -- Looking for UnDecorateSymbolName
  -- Looking for UnDecorateSymbolName - not found
  -- Looking for abi::__cxa_demangle
  -- Looking for abi::__cxa_demangle - found
  -- Looking for __argv
  -- Looking for __argv - not found
  -- Looking for getprogname
  -- Looking for getprogname - not found
  -- Looking for program_invocation_short_name
  -- Looking for program_invocation_short_name - found
  -- Performing Test HAVE___PROGNAME
  -- Performing Test HAVE___PROGNAME - Success
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_PC
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_PC - Failed
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_EIP
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_EIP - Failed
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_RIP
  -- Performing Test HAVE_PC_FROM_UCONTEXT_uc_mcontext_gregs_REG_RIP - Success
  -- Looking for gmtime_r
  -- Looking for gmtime_r - found
  -- Looking for localtime_r
  -- Looking for localtime_r - found
  -- Performing Test COMPILER_HAS_HIDDEN_VISIBILITY
  -- Performing Test COMPILER_HAS_HIDDEN_VISIBILITY - Success
  -- Performing Test COMPILER_HAS_HIDDEN_INLINE_VISIBILITY
  -- Performing Test COMPILER_HAS_HIDDEN_INLINE_VISIBILITY - Success
  -- Performing Test COMPILER_HAS_DEPRECATED_ATTR
  -- Performing Test COMPILER_HAS_DEPRECATED_ATTR - Success
  -- CUDA found
  CMake Error at /tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/cmake/data/share/cmake-4.0/Modules/CMakeDetermineCompilerId.cmake:909 (message):
    Compiling the CUDA compiler identification source file
    "CMakeCUDACompilerId.cu" failed.

    Compiler: /usr/bin/nvcc

    Build flags:

    Id flags:
    --keep;--keep-dir;tmp;-ccbin=/home/zhchen/miniconda3/envs/sllm-worker-0.6/bin/x86_64-conda-linux-gnu-g++
    -v



    The output was:

    1

    #$ _NVVM_BRANCH_=nvvm

    #$ _SPACE_=

    #$ _CUDART_=cudart

    #$ _HERE_=/usr/lib/nvidia-cuda-toolkit/bin

    #$ _THERE_=/usr/lib/nvidia-cuda-toolkit/bin

    #$ _TARGET_SIZE_=

    #$ _TARGET_DIR_=

    #$ _TARGET_SIZE_=64

    #$ NVVMIR_LIBRARY_DIR=/usr/lib/nvidia-cuda-toolkit/libdevice

    #$
    PATH=/usr/lib/nvidia-cuda-toolkit/bin:/tmp/pip-build-env-ovt348h8/overlay/bin:/tmp/pip-build-env-ovt348h8/normal/bin:/home/zhchen/miniconda3/envs/sllm-worker-0.6/bin:/home/zhchen/miniconda3/condabin:/home/zhchen/.nix-profile/bin:/nix/var/nix/profiles/default/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

    #$ LIBRARIES= -L/usr/lib/x86_64-linux-gnu/stubs -L/usr/lib/x86_64-linux-gnu

    #$ rm tmp/a_dlink.reg.c

    #$
    "/home/zhchen/miniconda3/envs/sllm-worker-0.6/bin"/x86_64-conda-linux-gnu-g++
    -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -E -x c++
    -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__
    -D__CUDACC_VER_MAJOR__=11 -D__CUDACC_VER_MINOR__=5
    -D__CUDACC_VER_BUILD__=119 -D__CUDA_API_VER_MAJOR__=11
    -D__CUDA_API_VER_MINOR__=5 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include
    "cuda_runtime.h" -m64 "CMakeCUDACompilerId.cu" -o
    "tmp/CMakeCUDACompilerId.cpp1.ii"

    cc1plus: fatal error: cuda_runtime.h: No such file or directory

    compilation terminated.

    # --error 0x1 --





  Call Stack (most recent call first):
    /tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/cmake/data/share/cmake-4.0/Modules/CMakeDetermineCompilerId.cmake:8 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
    /tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/cmake/data/share/cmake-4.0/Modules/CMakeDetermineCompilerId.cmake:53 (__determine_compiler_id_test)
    /tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/cmake/data/share/cmake-4.0/Modules/CMakeDetermineCUDACompiler.cmake:139 (CMAKE_DETERMINE_COMPILER_ID)
    CMakeLists.txt:44 (enable_language)


  -- Configuring incomplete, errors occurred!
  Traceback (most recent call last):
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
      main()
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
      json_out["return_val"] = hook(**hook_input["kwargs"])
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 280, in build_wheel
      return _build_backend().build_wheel(
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 435, in build_wheel
      return _build(['bdist_wheel', '--dist-info-dir', str(metadata_directory)])
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 423, in _build
      return self._build_with_temp_dir(
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 404, in _build_with_temp_dir
      self.run_setup()
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
      exec(code, locals())
    File "<string>", line 223, in <module>
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/__init__.py", line 115, in setup
      return distutils.core.setup(**attrs)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 186, in setup
      return run_commands(dist)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
      dist.run_commands()
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
      self.run_command(cmd)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
      super().run_command(command)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
      cmd_obj.run()
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/command/bdist_wheel.py", line 370, in run
      self.run_command("build")
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
      self.distribution.run_command(command)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
      super().run_command(command)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
      cmd_obj.run()
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/command/build.py", line 135, in run
      self.run_command(cmd_name)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
      self.distribution.run_command(command)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
      super().run_command(command)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
      cmd_obj.run()
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 96, in run
      _build_ext.run(self)
    File "/tmp/pip-build-env-ovt348h8/overlay/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 368, in run
      self.build_extensions()
    File "<string>", line 200, in build_extensions
    File "<string>", line 182, in configure
    File "/home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/subprocess.py", line 369, in check_call
      raise CalledProcessError(retcode, cmd)
  subprocess.CalledProcessError: Command '['cmake', '/home/zhchen/zwb/ServerlessLLM/sllm_store', '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/zhchen/zwb/ServerlessLLM/sllm_store/build/lib.linux-x86_64-cpython-310/sllm_store', '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/home/zhchen/zwb/ServerlessLLM/sllm_store/build/lib.linux-x86_64-cpython-310/sllm_store', '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=build/temp.linux-x86_64-cpython-310', '-DCMAKE_VERBOSE_MAKEFILE=ON', '-DSLLM_STORE_PYTHON_EXECUTABLE=/home/zhchen/miniconda3/envs/sllm-worker-0.6/bin/python3.10', '-DCMAKE_JOB_POOL_COMPILE:STRING=compile', '-DCMAKE_JOB_POOLS:STRING=compile=16']' returned non-zero exit status 1.
  error: subprocess-exited-with-error
  
  × Building wheel for serverless-llm-store (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  full command: /home/zhchen/miniconda3/envs/sllm-worker-0.6/bin/python3.10 /home/zhchen/miniconda3/envs/sllm-worker-0.6/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py build_wheel /tmp/tmpjixgznlb
  cwd: /home/zhchen/zwb/ServerlessLLM/sllm_store
  Building wheel for serverless-llm-store (pyproject.toml) ... error
  ERROR: Failed building wheel for serverless-llm-store
Failed to build serverless-llm-store
ERROR: Failed to build installable wheels for some pyproject.toml based projects (serverless-llm-store)
````

````bash
# 激活conda环境
conda activate sllm-worker-0.6

# 
conda install -c conda-forge gflags glog
conda install -c conda-forge cudatoolkit=11.5 cudatoolkit-dev=11.5
conda install -c conda-forge gcc_linux-64=9.4.0 gxx_linux-64=9.4.0


# 设置正确的环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export CUDA_INCLUDE_PATH=$CONDA_PREFIX/include
export CUDA_LIBRARY_PATH=$CONDA_PREFIX/lib
export PATH=$CONDA_PREFIX/bin:$PATH

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# 验证CUDA头文件
ls $CUDA_INCLUDE_PATH/cuda_runtime.h

# 清理并重新编译
cd ServerlessLLM/sllm_store
rm -rf build/
./rebuild.sh
````



````
10: fatal error: boost/graph/adjacency_list.hpp: No such file or directory
    158 | #include <boost/graph/adjacency_list.hpp>
        |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(231): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(232): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(233): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(294): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(295): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(297): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(303): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(327): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(337): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(485): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(486): error: structured binding cannot be captured

  /home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu(487): error: structured binding cannot be captured

  12 errors detected in the compilation of "/home/zhchen/zwb/ServerlessLLM/sllm_store/csrc/sllm_store/model.cu".
````
````bash
conda install -c conda-forge boost
# change the source code of 'model.cu' to compatible with lower CXX compiler
````




运行报错：
````
(sllm-0.6) zhchen@super:~$ conda activate sllm-0.6
sllm-serve start --enable_storage_aware
Traceback (most recent call last):
  File "/home/zhchen/miniconda3/envs/sllm-0.6/bin/sllm-serve", line 5, in <module>
    from sllm.serve.commands.serve.sllm_serve import main
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/sllm/serve/commands/serve/sllm_serve.py", line 27, in <module>
    from sllm.serve.controller import SllmController
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/sllm/serve/controller.py", line 26, in <module>
    from sllm.serve.store_manager import StoreManager
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/sllm/serve/store_manager.py", line 28, in <module>
    from sllm.serve.model_downloader import (
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/sllm/serve/model_downloader.py", line 25, in <module>
    from transformers import AutoTokenizer
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/transformers/__init__.py", line 26, in <module>
    from . import dependency_versions_check
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/transformers/dependency_versions_check.py", line 57, in <module>
    require_version_core(deps[pkg])
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/transformers/utils/versions.py", line 117, in require_version_core
    return require_version(requirement, hint)
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/transformers/utils/versions.py", line 111, in require_version
    _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
  File "/home/zhchen/miniconda3/envs/sllm-0.6/lib/python3.10/site-packages/transformers/utils/versions.py", line 44, in _compare_versions
    raise ImportError(
ImportError: numpy>=1.17,<2.0 is required for a normal functioning of this module, but found numpy==2.2.6.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main
````

````bash
conda remove numpy numpy-base -y
conda clean --all
pip uninstall numpy -y

conda install "numpy>=1.17,<2.0"
````