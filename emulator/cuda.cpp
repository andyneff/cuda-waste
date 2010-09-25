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

CUresult CUDA_EMULATOR::_cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
	*pctx = (CUcontext)malloc(sizeof(CUcontext));
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
	*pctx = (CUcontext)malloc(sizeof(CUcontext));
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuCtxDestroy( CUcontext ctx )
{
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuCtxDetach(CUcontext ctx)
{
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuDeviceGet(CUdevice *device, int ordinal)
{
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
	switch (attrib)
	{
		case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
			*pi = 1024;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
			*pi = 1024;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
			*pi = 1024;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
			*pi = 64;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
			*pi = 65535;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
			*pi = 65535;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
			*pi = 1;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
			*pi = 49152;
			break;
		case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
			*pi = 65536;
			break;
		case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
			*pi = 32;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
			*pi = 2147483647;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
			*pi = 32768;
			break;
		//case CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK:  same as CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
		//	*pi = 32768;
		//	break;
		case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
			*pi = 810000;
			break;
		case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
			*pi = 512;
			break;
		case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
			*pi = 1;
			break;
		case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
			*pi = 14;
			break;
		case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_INTEGRATED:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
			*pi = 1;
			break;
		case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
			*pi = 65536;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
			*pi = 65536;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
			*pi = 65535;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
			*pi = 2048;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
			*pi = 2048;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:
			*pi = 2048;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH:
			*pi = 16384;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT:
			*pi = 16384;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES:
			*pi = 2048;
			break;
		case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
			*pi = 512;
			break;
		case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
			*pi = 1;
			break;
		case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
			*pi = 3;
			break;
		case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
			*pi = 0;
			break;
		default:
			assert(false);
			*pi = 0;
			break;
	}
	return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
	CUdevprop loc= {
		1024,
		{ 1024, 1024, 64},
		{ 65535, 65535, 1},
		49152,
		65536,
		32,
		2147483647,
		32768,
		810000,
		512
	};
	*prop = loc;
	return CUDA_SUCCESS;
}


CUresult CUDA_EMULATOR::_cuDriverGetVersion(int * driverVersion)
{
    *driverVersion = 3020;
    return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId )
{
    *ppExportTable = 0;
    return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    *major = 2;
    *minor = 0;
    return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuDeviceGetName(char *name, int len, CUdevice dev)
{
    strncpy(name, "emulator", len);
    return CUDA_SUCCESS;
}

CUresult CUDA_EMULATOR::_cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    *bytes = 1309081600;
    return CUDA_SUCCESS;
}
