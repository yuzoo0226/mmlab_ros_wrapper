# テスト時の環境

```bash
 > python3 include/mmaction2/mmaction/utils/collect_env.py 
sys.platform: linux
Python: 3.8.10 (default, May 26 2023, 14:05:08) [GCC 9.4.0]
CUDA available: True
numpy_random_seed: 2147483648
GPU 0: NVIDIA GeForce RTX 3060
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.7, V11.7.99
GCC: x86_64-linux-gnu-gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
PyTorch: 1.13.1+cu117
PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.5
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.13.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 

TorchVision: 0.14.1+cu117
OpenCV: 4.8.0
MMEngine: 0.8.4
MMAction2: 1.1.0+bbb40ec
MMCV: 2.0.1
MMDetection: 3.1.0
MMPose: 1.1.0

```


```bash
 > python3 include/mmdeploy/tools/check_env.py 
09/03 16:17:11 - mmengine - INFO - 

09/03 16:17:11 - mmengine - INFO - **********Environmental information**********
09/03 16:17:12 - mmengine - INFO - sys.platform: linux
09/03 16:17:12 - mmengine - INFO - Python: 3.8.10 (default, May 26 2023, 14:05:08) [GCC 9.4.0]
09/03 16:17:12 - mmengine - INFO - CUDA available: True
09/03 16:17:12 - mmengine - INFO - numpy_random_seed: 2147483648
09/03 16:17:12 - mmengine - INFO - GPU 0: NVIDIA GeForce RTX 3060
09/03 16:17:12 - mmengine - INFO - CUDA_HOME: /usr/local/cuda
09/03 16:17:12 - mmengine - INFO - NVCC: Cuda compilation tools, release 11.7, V11.7.99
09/03 16:17:12 - mmengine - INFO - GCC: x86_64-linux-gnu-gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
09/03 16:17:12 - mmengine - INFO - PyTorch: 1.13.1+cu117
09/03 16:17:12 - mmengine - INFO - PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.5
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.13.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 

09/03 16:17:12 - mmengine - INFO - TorchVision: 0.14.1+cu117
09/03 16:17:12 - mmengine - INFO - OpenCV: 4.8.0
09/03 16:17:12 - mmengine - INFO - MMEngine: 0.8.4
09/03 16:17:12 - mmengine - INFO - MMCV: 2.0.1
09/03 16:17:12 - mmengine - INFO - MMCV Compiler: GCC 9.4
09/03 16:17:12 - mmengine - INFO - MMCV CUDA Compiler: 11.7
09/03 16:17:12 - mmengine - INFO - MMDeploy: 1.2.0+bbb40ec
09/03 16:17:12 - mmengine - INFO - 

09/03 16:17:12 - mmengine - INFO - **********Backend information**********
09/03 16:17:12 - mmengine - INFO - tensorrt:	None
09/03 16:17:12 - mmengine - INFO - ONNXRuntime:	1.15.1
09/03 16:17:12 - mmengine - INFO - ONNXRuntime-gpu:	1.15.1
09/03 16:17:12 - mmengine - INFO - ONNXRuntime custom ops:	Available
09/03 16:17:12 - mmengine - INFO - pplnn:	None
09/03 16:17:12 - mmengine - INFO - ncnn:	None
09/03 16:17:12 - mmengine - INFO - snpe:	None
09/03 16:17:12 - mmengine - INFO - openvino:	None
09/03 16:17:12 - mmengine - INFO - torchscript:	1.13.1+cu117
09/03 16:17:12 - mmengine - INFO - torchscript custom ops:	NotAvailable
09/03 16:17:12 - mmengine - INFO - rknn-toolkit:	None
09/03 16:17:12 - mmengine - INFO - rknn-toolkit2:	None
09/03 16:17:12 - mmengine - INFO - ascend:	None
09/03 16:17:12 - mmengine - INFO - coreml:	None
09/03 16:17:12 - mmengine - INFO - tvm:	None
09/03 16:17:12 - mmengine - INFO - vacc:	None
09/03 16:17:12 - mmengine - INFO - 

09/03 16:17:12 - mmengine - INFO - **********Codebase information**********
09/03 16:17:12 - mmengine - INFO - mmdet:	3.1.0
09/03 16:17:12 - mmengine - INFO - mmseg:	None
09/03 16:17:12 - mmengine - INFO - mmpretrain:	None
09/03 16:17:12 - mmengine - INFO - mmocr:	None
09/03 16:17:12 - mmengine - INFO - mmagic:	None
09/03 16:17:12 - mmengine - INFO - mmdet3d:	None
09/03 16:17:12 - mmengine - INFO - mmpose:	1.1.0
09/03 16:17:12 - mmengine - INFO - mmrotate:	None
09/03 16:17:12 - mmengine - INFO - mmaction:	1.1.0
09/03 16:17:12 - mmengine - INFO - mmrazor:	None
09/03 16:17:12 - mmengine - INFO - mmyolo:	None

```