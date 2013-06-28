#include <iostream>
#include <stdio.h>

void CUDA_CALL(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		std::cout << "Error " << result << "\n";
		exit(1);
	}
}

texture<float4, 2, cudaReadModeElementType> texRef2;


// Simple transformation kernel
__global__ void transformKernel(float * output_x, float4 * output_y, int width, int height, int output_width, int output_height, float theta)
{
	// Calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
     
	float u = x / (float)width;
	float v = y / (float)height;

	if (x == 10 && y == 15)
	{
	float tu = ((float)width * x) / output_width;
	float tv = ((float)height * y) / output_height;

	output_x[y * output_width + x] = tv;
	output_y[y * output_width + x] = tex2D(texRef2, tu, tv);
	}
}

// Host code
int main()
{
	int width = 10;
	int height = 10;
	int size = width * height;

	float4 * h_data = (float4*) malloc(width * height * sizeof(float4));
	for (int i = 0 ; i < height; i++)
	{
		for (int j = 0 ; j < width; j++)
			h_data[i * width + j].x = i;
	}

	printf ("\n Original array \n");
	for (int i = 0 ; i < height; i++)
	{
		for (int j = 0 ; j < width; j++)
			printf ("%f ", h_data[i * width + j].x);
		printf ("\n");
	}

	// create a CUDA array on the device
	cudaArray* cuArray;
	CUDA_CALL( cudaMallocArray(&cuArray, &texRef2.channelDesc, width, height) );
	CUDA_CALL( cudaMemcpyToArray(cuArray, 0, 0, h_data, sizeof(float4) * size, cudaMemcpyHostToDevice) );
   
	// bind a texture to the CUDA array
	cudaBindTextureToArray(texRef2, cuArray);
 
	// Set texture access
	texRef2.normalized = false;
	texRef2.filterMode = cudaFilterModeLinear;
        
	int output_width = 100;
	int output_height = 100;

	// Set up block dims.
	dim3 dimBlock(10, 10);
	dim3 dimGrid((output_width - 1)/dimBlock.x + 1, (output_height - 1)/dimBlock.y + 1);
     
	printf("blockDim = %d, %d\n", dimBlock.x, dimBlock.y);
	printf("gridDim = %d, %d\n", dimGrid.x, dimGrid.y);

	// Allocate result of transformation in device memory
	int output_size =  dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y;
	printf("output_size %d\n", output_size);
	float * d_output_x;
	float4* d_output_y;
	CUDA_CALL( cudaMalloc(&d_output_x, output_size * sizeof(float)) );
	CUDA_CALL( cudaMemset(d_output_x, 0, output_size * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&d_output_y, output_size * sizeof(float4)) );
	float * h_output_x = (float*)malloc(output_size * sizeof(float));
	float4 * h_output_y = (float4*)malloc(output_size * sizeof(float4));

	transformKernel<<<dimGrid, dimBlock>>>(d_output_x, d_output_y, width, height, output_width, output_height, 90);
    cudaDeviceSynchronize();

	CUDA_CALL( cudaMemcpy(h_output_x, d_output_x, output_size * sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMemcpy(h_output_y, d_output_y, output_size * sizeof(float4), cudaMemcpyDeviceToHost) );

	printf ("\nAfter operation\n");
	for (int i = 0 ; i < output_height; i++)
	{
		for (int j = 0 ; j < output_width; j++)
			printf ("%f ", h_output_x[i * output_width + j]);
		printf ("\n");
	}
	printf ("\n");

	system ("pause");
	// Free device memory
	cudaFree(d_output_x);
	cudaFree(d_output_y);
	return 0;
}

