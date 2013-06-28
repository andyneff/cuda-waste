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

texture<float4, 1, cudaReadModeElementType> texRef1;


// Simple transformation kernel
__global__ void transformKernel(float * output_x, float4 * output_y, int width, int height, int output_width, int output_height, float theta)
{
	// Calculate normalized texture coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
     
	float u = x / (float)width;
	float v = y / (float)height;

	float tu = ((float)width * x) / output_width;
	float tv = ((float)height * y) / output_height;

	output_x[y * output_width + x] = tu;
	output_y[y * output_width + x] = tex1D(texRef1, tu);
}

// Host code
int main()
{
	int width = 10;
	int height = 1;
	int size = width * height;

	float4 * h_data = (float4*) malloc(width * height * sizeof(float4));
	for (int j = 0; j < width; j++)
	{
			h_data[j].x = j + 1;
			h_data[j].y = j;
			h_data[j].z = j - 1;
			h_data[j].w = j + 2;
	}

	printf ("\n Original array \n");
	for (int j = 0 ; j < width; j++)
		printf ("%f ", h_data[j].x);
	printf ("\n");
          
	// create a CUDA array on the device
	cudaArray* cuArray;
	CUDA_CALL( cudaMallocArray(&cuArray, &texRef1.channelDesc, size, height) );
	CUDA_CALL( cudaMemcpyToArray(cuArray, 0, 0, h_data, sizeof(float4) * size, cudaMemcpyHostToDevice) );
   
	// bind a texture to the CUDA array
	cudaBindTextureToArray(texRef1, cuArray);
 
	// Set texture access
	texRef1.normalized = false;
	texRef1.filterMode = cudaFilterModeLinear;
        
	int output_width = 100;
	int output_height = 1;

	// Set up block dims.
	dim3 dimBlock(10, 1);
	dim3 dimGrid((output_width - 1)/dimBlock.x + 1, (output_height - 1)/dimBlock.y + 1);
     
	printf("blockDim = %d, %d\n", dimBlock.x, dimBlock.y);
	printf("gridDim = %d, %d\n", dimGrid.x, dimGrid.y);

	// Allocate result of transformation in device memory
	int output_size =  dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y;
	printf("output_size %d\n", output_size);
	float* d_output_x;
	float4* d_output_y;
	CUDA_CALL( cudaMalloc(&d_output_x, output_size * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&d_output_y, output_size * sizeof(float4)) );
	float * h_output_x = (float*)malloc(output_size * sizeof(float));
	float4 * h_output_y = (float4*)malloc(output_size * sizeof(float4));

	transformKernel<<<dimGrid, dimBlock>>>(d_output_x, d_output_y, width, height, output_width, output_height, 90);
    cudaDeviceSynchronize();

	CUDA_CALL( cudaMemcpy(h_output_x, d_output_x, output_size * sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMemcpy(h_output_y, d_output_y, output_size * sizeof(float4), cudaMemcpyDeviceToHost) );

	printf ("\nAfter operation\n");
	for (int j = 0; j < output_width; j++)
		printf ("%f %f %f %f %f\n", h_output_x[j], h_output_y[j].x, h_output_y[j].y, h_output_y[j].z, h_output_y[j].w);
	printf ("\n");

	system ("pause");
	// Free device memory
	cudaFree(d_output_x);
	cudaFree(d_output_y);
	return 0;
}

