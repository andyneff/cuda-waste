#pragma once
#include "cuda-memory-debug.h"

#define cudaMalloc(a,b) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::Malloc(a,b))
#define cudaFree(a) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::Free(a))
#define cudaHostAlloc(a,b,c) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::HostAlloc(a,b,c))
#define cudaFreeHost(a) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::FreeHost(a))
#define cudaHostGetDevicePointer(a,b,c) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::HostGetDevicePointer(a,b,c))
#define cudaMemcpy(a,b,c,d) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::Memcpy(a,b,c,d))
#define cudaMemset(a,b,c) (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::Memset(a,b,c))
#define cudaCheckOverwrite() (CudaMemoryDebug::MakeContext(__FILE__,__LINE__),CudaMemoryDebug::CheckOverwrite())
