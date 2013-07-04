#include <iostream>
//3d_array.cu
#include <stdlib.h>
#include <stdio.h>

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,cudaExtent extent)
{
	char * start = (char*) devPitchedPtr.ptr;
	size_t x_dim = devPitchedPtr.pitch;
	size_t x_times_y_dim = x_dim * extent.height;

	int x = threadIdx.x;
	int y = threadIdx.y;
	int z = blockIdx.x;
	printf("%d %d %d\n", x, y, z);

	char * ptr = start + z * x_times_y_dim + y * x_dim + x * sizeof(float);
	*(float*)ptr = x + (y*10) + (z*100);
}

#define X 10
#define Y 5
#define Z 3

int main(void)
{
	cudaSetDevice(0);
	float array[X][Y][Z];
	float result[X][Y][Z];
	cudaError_t status = cudaSuccess;

	//initialise array
	for (int x = 0; x < X; x++)
	{
		for (int y = 0; y < Y; y++)
		{
			for (int z = 0; z < Z; z++)
			{
				array[x][y][z] = 1.0;
			}
		}
	}
	for (int x = 0; x < X; x++)
	{
		for (int y = 0; y < Y; y++)
		{
			for (int z = 0; z < Z; z++)
			{
				result[x][y][z] = 0.0;
			}
		}
	}

	//allocate memory on device for a 3D matrix
	cudaExtent extent;
	extent.width = X *sizeof(float);
	extent.height = Y;
	extent.depth = Z;

	cudaPitchedPtr mem_device;
	status = cudaMalloc3D(&mem_device,extent);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Malloc: %s\n", cudaGetErrorString(status));
	}

	//copy memory to device
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = make_cudaPitchedPtr((void*)array, X * sizeof(float), X, Y);
	p.dstPtr = mem_device;
	p.extent = extent;
	p.kind = cudaMemcpyHostToDevice;
	status = cudaMemcpy3D(&p);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "MemcpyHtD: %s\n", cudaGetErrorString(status));
	}

	//run 3d kernel!
	dim3 block_dim(X, Y);

	MyKernel<<<Z, block_dim>>>(mem_device, extent);

	//copy result array back to host
	cudaMemcpy3DParms q = {0};
	q.srcPtr = mem_device;
	q.dstPtr = make_cudaPitchedPtr((void*)result, X * sizeof(float), X, Y);
	q.extent = extent;
	q.kind = cudaMemcpyDeviceToHost;
	status = cudaMemcpy3D(&q);
	if(status != cudaSuccess)
	{
		fprintf(stderr, "MemcpyDtoH: %s\n", cudaGetErrorString(status));
	}

	for (int x = 0; x < X; x++)
	{
		for (int y = 0; y < Y; y++)
		{
			for (int z = 0; z < Z; z++)
			{
				printf("%f ", result[x][y][z]);
			}
			printf("\n");
		}
		printf("\n");
	}

	cudaFree(mem_device.ptr);
}
