#pragma once
#pragma push_macro("cudaMalloc")
#undef cudaMalloc
#pragma push_macro("cudaFree")
#undef cudaFree
#pragma push_macro("cudaHostAlloc")
#undef cudaHostAlloc
#pragma push_macro("cudaFreeHost")
#undef cudaFreeHost
#pragma push_macro("cudaHostGetDevicePointer")
#undef cudaHostGetDevicePointer
#pragma push_macro("cudaMemcpy")
#undef cudaMemcpy
#pragma push_macro("cudaMemset")
#undef cudaMemset

