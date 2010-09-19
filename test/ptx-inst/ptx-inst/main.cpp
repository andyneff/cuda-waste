#include <stdio.h>
#include <iostream>
#include <cuda.h>
// Host code
int main()
{
	int N = 16;
	double * h_A = (double*)malloc(N * sizeof(double));
	float * h_B = (float*)malloc(N * sizeof(float));
	int * h_C = (int*)malloc(N * sizeof(int));
	char * h_D = (char*)malloc(N * sizeof(char));
	bool * h_E = (bool*)malloc(N * sizeof(bool));
	memset(h_E, 0, N * sizeof(bool));

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
	CUcontext cuContext;
	cuCtxCreate(&cuContext, 0, cuDevice);
	CUmodule cuModule;
	CUresult r2 = cuModuleLoad(&cuModule, "inst.ptx");
	CUfunction inst_basic;
	CUresult r3 = cuModuleGetFunction(&inst_basic, cuModule, "InstBasic");
	if (r3 != CUDA_SUCCESS)
	{
		std::cout << "File not found\n";
		exit(1);
	}
	CUdeviceptr d_E;
	CUresult r4 = cuMemAlloc(&d_E, N * sizeof(bool));
	CUresult r7 = cuMemcpyHtoD(d_E, h_E, N * sizeof(bool));

#define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)
	int offset = 0;
	void* ptr;
	
	ptr = (void*)(size_t)d_E;
	ALIGN_UP(offset, __alignof(ptr));
	CUresult r9 = cuParamSetv(inst_basic, offset, &ptr, sizeof(ptr));
	offset += sizeof(ptr);
	
	CUresult r12 = cuParamSetSize(inst_basic, offset);

	int threadsPerBlock = 1;
	int blocksPerGrid = 1;

	CUresult r13 = cuFuncSetBlockShape(inst_basic, threadsPerBlock, 1, 1);
	CUresult r14 = cuLaunchGrid(inst_basic, blocksPerGrid, 1);

	CUresult r15 = cuMemcpyDtoH(h_E, d_E, N * sizeof(bool));
	for (int i = 0; i < N; ++i)
	{
		std::cout << "i " << i << " = " << h_E[i] << "\n";
	}


	{
		memset(h_E, 0, N * sizeof(bool));
		CUresult r7 = cuMemcpyHtoD(d_E, h_E, N * sizeof(bool));

		CUfunction proc;
		CUresult r3 = cuModuleGetFunction(&proc, cuModule, "InstIntegerArithmetic");

		int offset = 0;
		void* ptr;
		
		ptr = (void*)(size_t)d_E;
		ALIGN_UP(offset, __alignof(ptr));
		CUresult r9 = cuParamSetv(proc, offset, &ptr, sizeof(ptr));
		offset += sizeof(ptr);
		
		CUresult r12 = cuParamSetSize(proc, offset);

		int threadsPerBlock = 1;
		int blocksPerGrid = 1;

		CUresult r13 = cuFuncSetBlockShape(proc, threadsPerBlock, 1, 1);
		CUresult r14 = cuLaunchGrid(proc, blocksPerGrid, 1);

		CUresult r15 = cuMemcpyDtoH(h_E, d_E, N * sizeof(bool));
		for (int i = 0; i < N; ++i)
		{
			std::cout << "i " << i << " = " << h_E[i] << "\n";
		}

	}

	// Free device memory
	cuMemFree(d_E);
}
