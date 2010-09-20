#include "emulator.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include "tree.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Cuda Driver API/Emulator implementation.
///
/////////////////////////////////////////////////////////////////////////////////////////////////////

CUresult CUDA_EMULATOR::_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    // Now, given the name of the kernel function being called, find
    // the entry for it.
	std::map<char*, TREE *, ltstr>::iterator j = this->entry.find((char*)name);
	if (j == this->entry.end())
		return CUDA_ERROR_NOT_FOUND;
	TREE * data = j->second;
	*hfunc = (CUfunction)data;
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuModuleLoad(CUmodule *module, const char *fname)
{
	int size = 1000000;
	char * buffer = (char *)malloc(size);
    // Open file, parse, and record AST.
	std::ifstream myfile(fname);
	int count = 0;
	if (myfile.is_open())
	{
		while (! myfile.eof())
		{
			if (count >= size)
			{
				size = size * 2;
				buffer = (char *)realloc(buffer, size);
			}
			int c = myfile.get();
			if (c != -1)
				buffer[count++] = c;
			else
				break;
		}
		myfile.close();
		buffer[count++] = 0;
	}
	if (count == 0)
		return CUDA_ERROR_FILE_NOT_FOUND;
	TREE * mod = this->Extract_From_Source(this->device, (char*)buffer);
	*module = (CUmodule) mod;
	if (mod != 0)
		return CUDA_SUCCESS;
	else
		return CUDA_ERROR_INVALID_CONTEXT;
}

CUresult CUDA_EMULATOR::_cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
    // Unknown what to do for param size.
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuLaunchGrid(CUfunction hfunc, int grid_width, int grid_height)
{
	dim3 gridDim(grid_width, grid_height, 1);
	this->ConfigureGrid(gridDim);
	this->Execute((TREE*)hfunc);
	return CUDA_SUCCESS;
}


CUresult CUDA_EMULATOR::_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    // record argument, size, offset.
	CUDA_EMULATOR::arg * a = new CUDA_EMULATOR::arg();
	a->argument = malloc(numbytes);
	memcpy(const_cast<void*>(a->argument), ptr, numbytes);
	a->size = numbytes;
	a->offset = offset;
	this->arguments.push_back(a);
	return CUDA_SUCCESS;
}

