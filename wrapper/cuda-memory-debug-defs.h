#pragma once
#include "cuda-memory-debug.h"

#define cudaMalloc(a,b) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::Malloc(a,b))
#define cudaFree(a) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::Free(a))
#define cudaHostAlloc(a,b,c) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::HostAlloc(a,b,c))
#define cudaFreeHost(a) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::FreeHost(a))
#define cudaHostGetDevicePointer(a,b,c) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::HostGetDevicePointer(a,b,c))
#define cudaMemcpy(a,b,c,d) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::Memcpy(a,b,c,d))
#define cudaMemset(a,b,c) (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::Memset(a,b,c))
#define cudaCheckOverwrite() (CUDA_WRAPPER::MakeContext(__FILE__,__LINE__),CUDA_WRAPPER::CheckOverwrite())
