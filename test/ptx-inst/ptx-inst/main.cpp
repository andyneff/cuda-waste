#include <stdio.h>
#include <iostream>
#include <cuda.h>
// Host code
void doit(char * t)
{
	printf(t);
	exit(1);
}
#define Test(x, s) {int t = x; if (t != CUDA_SUCCESS) {printf("fail %d %s\n",t, s); exit(1);}}

int main()
{
	int N = 10000;
	double * h_A = (double*)malloc(N * sizeof(double));
	float * h_B = (float*)malloc(N * sizeof(float));
	int * h_C = (int*)malloc(N * sizeof(int));
	char * h_D = (char*)malloc(N * sizeof(char));
	bool * h_E = (bool*)malloc(N * sizeof(bool));
	memset(h_E, 0, N * sizeof(bool));

	// Initialize
	Test(cuInit(0), "cuInit");

	// Get number of devices supporting CUDA
	int deviceCount = 0;
	Test(cuDeviceGetCount(&deviceCount), "cuDeviceGetCount");
	if (deviceCount == 0)
	{
		printf("There is no device supporting CUDA.\n");
		exit (0);
	}

	// Get handle for device 0
	CUdevice cuDevice = 0;
	Test(cuDeviceGet(&cuDevice, 0), "cuDeviceGet");
	CUcontext cuContext;
	int xxx = cuCtxCreate(&cuContext, 0, cuDevice);
	CUmodule cuModule;
	Test(cuModuleLoad(&cuModule, "inst.ptx"), "cuModuleLoad");
	CUfunction inst_basic;
	Test(cuModuleGetFunction(&inst_basic, cuModule, "InstBasic"), "cuModuleGetFunction");
	CUdeviceptr d_E;
	Test(cuMemAlloc(&d_E, N * sizeof(bool)), "cuMemAlloc");
	Test(cuMemcpyHtoD(d_E, h_E, N * sizeof(bool)), "cuMemcpyHtoD");

#define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)
	int offset = 0;
	void* ptr;
	
	{
		CUdeviceptr d_C;
		memset(h_C, 0, N * sizeof(int));
		Test(cuMemAlloc(&d_C, sizeof(int)), "cuMemAlloc");
		Test(cuMemcpyHtoD(d_C, h_C, sizeof(int)), "cuMemcpyHtoD");

		CUfunction proc;
		Test(cuModuleGetFunction(&proc, cuModule, "InstIntegerArithmetic"), "cuModuleGetFunction");

		int offset = 0;
		void* ptr;
		
		ptr = (void*)(size_t)d_E;
		ALIGN_UP(offset, __alignof(ptr));
		Test(cuParamSetv(proc, offset, &ptr, sizeof(ptr)), "cuParamSetv");
		offset += sizeof(ptr);
		
		ptr = (void*)(size_t)d_C;
		ALIGN_UP(offset, __alignof(ptr));
		Test(cuParamSetv(proc, offset, &ptr, sizeof(ptr)), "cuParamSetv");
		offset += sizeof(ptr);
		
		Test(cuParamSetSize(proc, offset), "cuParamSetSize");

		int threadsPerBlock = 1;
		int blocksPerGrid = 1;

		Test(cuFuncSetBlockShape(proc, threadsPerBlock, 1, 1), "cuFuncSetBlockShape");
		Test(cuLaunchGrid(proc, blocksPerGrid, 1), "cuLaunchGrid");

		Test(cuMemcpyDtoH(h_E, d_E, N * sizeof(bool)), "cuMemcpyDtoH");
		Test(cuMemcpyDtoH(h_C, d_C, sizeof(int)), "cuMemcpyDtoH");

		N = *h_C;
		for (int i = 0; i < N; ++i)
		{
			std::cout << "i " << i << " = " << h_E[i] << "\n";
			if (h_E[i] == 0)
			{
				std::cout << "Test " << i << " failed.\n";
			}
		}
	}

	// Free device memory
	Test(cuMemFree(d_E), "cuMemFree");
}
