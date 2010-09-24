#include <stdio.h>
#include <cuda.h>
// Host code
int main()
{
	int N = 3;
	size_t size = N * sizeof(float);
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);

	// Set up vectors.
	for (int i = 0; i < N; ++i)
	{
		h_A[i] = i * 1.0;
		h_B[i] = i * 1.0 + 1;
		h_C[i] = 0;
		printf("i %d A %f B %f C %f\n", i, h_A[i], h_B[i], h_C[i]);
	}

	// Initialize
	if (cuInit(0) != CUDA_SUCCESS)
		exit (0);

	// Get number of devices supporting CUDA
	int deviceCount = 0;
	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0)
	{
		printf("There is no device supporting CUDA.\n");
		exit (0);
	}

	// Get handle for device 0
	CUdevice cuDevice = 0;
	CUresult r1 = cuDeviceGet(&cuDevice, 0);
	// Create context
	CUcontext cuContext;
	cuCtxCreate(&cuContext, 0, cuDevice);
	// Create module from binary file
	CUmodule cuModule;
	CUresult r2 = cuModuleLoad(&cuModule, "VecAdd.ptx");
	// Get function handle from module
	CUfunction vecAdd;
	CUresult r3 = cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");
	// Allocate vectors in device memory
	CUdeviceptr d_A;
	CUresult r4 = cuMemAlloc(&d_A, size);
	CUdeviceptr d_B;
	CUresult r5 = cuMemAlloc(&d_B, size);
	CUdeviceptr d_C;
	CUresult r6 = cuMemAlloc(&d_C, size);
	// Copy vectors from host memory to device memory
	// h_A and h_B are input vectors stored in host memory
	CUresult r7 = cuMemcpyHtoD(d_A, h_A, size);
	CUresult r8 = cuMemcpyHtoD(d_B, h_B, size);
	// Invoke kernel
#define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)
	int offset = 0;
	void* ptr;
	ptr = (void*)(size_t)d_A;
	ALIGN_UP(offset, __alignof(ptr));
	CUresult r9 = cuParamSetv(vecAdd, offset, &ptr, sizeof(ptr));
	offset += sizeof(ptr);
	ptr = (void*)(size_t)d_B;
	ALIGN_UP(offset, __alignof(ptr));
	CUresult r10 = cuParamSetv(vecAdd, offset, &ptr, sizeof(ptr));
	offset += sizeof(ptr);
	ptr = (void*)(size_t)d_C;
	ALIGN_UP(offset, __alignof(ptr));
	CUresult r11 = cuParamSetv(vecAdd, offset, &ptr, sizeof(ptr));
	offset += sizeof(ptr);
	ptr = (void*)(int)N;
	ALIGN_UP(offset, __alignof(ptr));
	CUresult r11a = cuParamSetv(vecAdd, offset, &ptr, sizeof(ptr));
	offset += sizeof(ptr);
	CUresult r12 = cuParamSetSize(vecAdd, offset);
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	CUresult r13 = cuFuncSetBlockShape(vecAdd, threadsPerBlock, 1, 1);
	CUresult r14 = cuLaunchGrid(vecAdd, blocksPerGrid, 1);
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	CUresult r15 = cuMemcpyDtoH(h_C, d_C, size);
	for (int i = 0; i < N; ++i)
	{
		printf("i %d A %f B %f C %f\n", i, h_A[i], h_B[i], h_C[i]);
	}

	// Free device memory
	cuMemFree(d_A);
	cuMemFree(d_B);
	cuMemFree(d_C);
}
