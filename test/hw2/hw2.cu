#include <iostream>

__global__ void fun(int * d, int * e)
{
	*d = 1;
	*e = 2;
}

int main()
{
	int h = 0;
	int * d;
	int * e;
	cudaMalloc(&d, sizeof(int));
	cudaMalloc(&e, sizeof(int));
	cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(e, &h, sizeof(int), cudaMemcpyHostToDevice);
	fun<<<1,1>>>(d, e);
	cudaThreadSynchronize();
	int rv = cudaGetLastError();
	cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "Result = " << h << "\n";
	cudaMemcpy(&h, e, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "Result = " << h << "\n";
	return 0;
}
