#include <iostream>

__global__ void fun(int * mem)
{
	*mem = 1;
}

int main()
{
	int h = 0;
	int * d;
	cudaMalloc(&d, sizeof(int));
	cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
	fun<<<1,1>>>(d);
	cudaThreadSynchronize();
	int rv = cudaGetLastError();
	cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "Result = " << h << "\n";
	return 0;
}
