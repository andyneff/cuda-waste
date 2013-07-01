/*
   Copyright 2010 Ken Domino

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <string.h>

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <string>
#include <sstream>
#include <crtdbg.h>
#include "emulated-device.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <map>
#include "tree.h"
#include <process.h>    /* _beginthread, _endthread */
#include "thread.h"
#include "symbol-table.h"
#include "string-table.h"
#include "symbol.h"
#include "constant.h"
#include "types.h"
#include "../wrapper/memdbg.h"
#include "../waste/version.h"
#include "../wrapper/cuda-wrapper.h"
#include <stdio.h>
#include <assert.h>
#define ZLIB_WINAPI 1
#include "zlib.h"
#include <__cudaFatFormat.h>
#include "../wrapper/call-stack-info.h"
#include "entry.h"
#include "texref.h"
#include "texture.h"
#include "array.h"
#include "texarr.h"
#include "module.h"
#include "regvar.h"
#include "errors.h"


#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)




//typedef struct __cudaFatCudaBinaryRec {
//    unsigned long            magic;
//    unsigned long            version;
//    unsigned long            gpuInfoVersion;
//    char*                   key;
//    char*                   ident;
//    char*                   usageMode;
//    __cudaFatPtxEntry             *ptx;
//    __cudaFatCubinEntry           *cubin;
//    __cudaFatDebugEntry           *debug;
//    void*                  debugInfo;
//    unsigned int                   flags;
//    __cudaFatSymbol               *exported;
//    __cudaFatSymbol               *imported;
//    struct __cudaFatCudaBinaryRec *dependends;
//    unsigned int                   characteristic;
//    __cudaFatElfEntry             *elf;
//} __cudaFatCudaBinary;

typedef struct __cudaFatCudaBinary2HeaderRec { 
    unsigned int            magic;
    unsigned int            version;
    unsigned long long int  length;
} __cudaFatCudaBinary2Header;

enum FatBin2EntryType {
    FATBIN_2_PTX = 0x1
};

typedef struct __cudaFatCudaBinary2EntryRec { 
    unsigned int           type;
    unsigned int           binary;
    unsigned long long int binarySize;
    unsigned int           unknown2;
    unsigned int           kindOffset;
    unsigned int           unknown3;
    unsigned int           unknown4;
    unsigned int           name;
    unsigned int           nameSize;
    unsigned long long int flags;
    unsigned long long int unknown7;
    unsigned long long int uncompressedBinarySize;
} __cudaFatCudaBinary2Entry;

#define COMPRESSED_PTX 0x0000000000001000LL

typedef struct __cudaFatCudaBinaryRec2 {
    int magic;
    int version;
    const unsigned long long* fatbinData;
    char* f;
} __cudaFatCudaBinary2;

/*
 * Magic numbers for fat bins, including previous versions.
 */
#define __cudaFatVERSION   0x00000004
#define __cudaFatMAGIC     0x1ee55a01
#define __cudaFatMAGIC2    0x466243b1
#define __cudaFatMAGIC3    0xba55ed50

inline int stoi(const char *  _Ptr, size_t *_Idx = 0,
    int _Base = 10)
    {   // convert string to int
    char *_Eptr;
    errno = 0;
    long _Ans = _CSTD strtol(_Ptr, &_Eptr, _Base);

    if (_Idx != 0)
        *_Idx = (size_t)(_Eptr - _Ptr);
    return ((int)_Ans);
    }

static char temp_buffer[50000];

char * file_name_tail(char * file_name)
{
    if (file_name == 0)
        return "";
    char * t = &file_name[strlen(file_name)];
    for (; t >= file_name; --t)
    {
        if (*t == '/' || *t == '\\')
        {
            sprintf(temp_buffer, "...%s", t);
            return temp_buffer;
        }
    }
    strncpy(temp_buffer, file_name, 49);
    return temp_buffer;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Cuda Driver API/Emulator implementation.
///
/////////////////////////////////////////////////////////////////////////////////////////////////////


CUresult EMULATED_DEVICE::_cuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArray3DCreate\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArray3DCreate_v2( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArray3DCreate_v2\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
	return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArray3DGetDescriptor\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArray3DGetDescriptor_v2( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArray3DGetDescriptor_v2\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArrayCreate\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArrayCreate_v2( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArrayCreate_v2\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArrayDestroy( CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArrayDestroy\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArrayGetDescriptor\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuArrayGetDescriptor_v2( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuArrayGetDescriptor_v2\n";
 	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    *pctx = (CUcontext)malloc(sizeof(CUcontext));
 	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxCreate\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxCreate_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxDestroy( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuCtxDetach(CUcontext ctx)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuCtxGetDevice(CUdevice * device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxGetDevice\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxGetLimit\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxPopCurrent( CUcontext *pctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxPopCurrent\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxPushCurrent( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxPushCurrent\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxSetLimit(CUlimit limit, size_t value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxSetLimit\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuCtxSynchronize(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuCtxSynchronize\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    *major = 2;
    *minor = 0;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceGet(CUdevice *device, int ordinal)
{
    *device = 0;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
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
        //  *pi = 32768;
        //  break;
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
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceGetCount(int *count)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    *count = 1;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceGetName(char *name, int len, CUdevice dev)
{
    strncpy(name, "emulator", len);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
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
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    *bytes = 1309081600;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuDeviceTotalMem_v2(unsigned int *bytes, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CUresult res = this->_cuDeviceTotalMem(bytes, dev);
	ERRS * error = new ERRS(res, __FILE__, " " + __LINE__);
	return res;
}

CUresult EMULATED_DEVICE::_cuDriverGetVersion(int * driverVersion)
{
    *driverVersion = 3020;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuEventCreate( CUevent *phEvent, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuEventCreate\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuEventDestroy( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuEventDestroy\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuEventElapsedTime\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuEventQuery( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuEventQuery\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuEventRecord( CUevent hEvent, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuEventRecord\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuEventSynchronize( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuEventSynchronize\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuFuncGetAttribute\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z)
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    dim3 blockDim(x,y,z);
    this->ConfigureBlock(blockDim);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuFuncSetCacheConfig\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuFuncSetSharedSize\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId )
{
    *ppExportTable = 0;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsMapResources\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsResourceGetMappedPointer\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGraphicsResourceGetMappedPointer_v2( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsResourceGetMappedPointer_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsResourceSetMapFlags\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsSubResourceGetMappedArray\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsUnmapResources\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuGraphicsUnregisterResource\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuInit(unsigned int Flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (Flags == 0)
	{
		ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
        return CUDA_SUCCESS;
	}
	else
    {
		ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult EMULATED_DEVICE::_cuLaunch ( CUfunction f )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuLaunch\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuLaunchGrid(CUfunction hfunc, int grid_width, int grid_height)
{
    dim3 gridDim(grid_width, grid_height, 1);
    this->ConfigureGrid(gridDim);
    this->Execute((ENTRY*)hfunc);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuLaunchGridAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize)
{
    // Basic, no frills, allocation.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    *dptr = (CUdeviceptr)malloc(bytesize);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuMemAlloc_v2( CUdeviceptr *dptr, unsigned int bytesize)
{
    // Basic, no frills, allocation.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    *dptr = (CUdeviceptr)malloc(bytesize);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuMemAllocHost(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemAllocHost\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemAllocHost_v2(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemAllocHost_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemAllocPitch\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemAllocPitch_v2( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemAllocPitch_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy2D( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy2D\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy2D_v2( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy2D_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy2DAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy2DAsync_v2( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy2DAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy2DUnaligned\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy2DUnaligned_v2( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy2DUnaligned_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy3D( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy3D\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy3D_v2( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy3D_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy3DAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpy3DAsync_v2( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpy3DAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoA\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoA_v2( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoA_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoD\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoD_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoH\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoH_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoH_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoHAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyAtoHAsync_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyAtoHAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoA\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoA_v2 ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoA_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoD\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoD_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoDAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoDAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    memcpy(dstHost, (void*)srcDevice, ByteCount);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoH_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    memcpy(dstHost, (void*)srcDevice, ByteCount);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoHAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyDtoHAsync_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyDtoHAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyHtoA\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoA_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyHtoA_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyHtoAAsync\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoAAsync_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyHtoAAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    memcpy((void*)dstDevice, srcHost, ByteCount);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    memcpy((void*)dstDevice, srcHost, ByteCount);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

CUresult EMULATED_DEVICE::_cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemcpyHtoDAsync_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemFree(CUdeviceptr dptr)
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    free((void*)dptr);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;     
}

CUresult EMULATED_DEVICE::_cuMemFree_v2(CUdeviceptr dptr)
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    free((void*)dptr);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;     
}

CUresult EMULATED_DEVICE::_cuMemFreeHost(void *p)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemFreeHost\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemGetAddressRange\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemGetAddressRange_v2( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemGetAddressRange_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemGetInfo(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemGetInfo\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemGetInfo_v2(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemGetInfo_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemHostAlloc\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemHostGetDevicePointer\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemHostGetDevicePointer_v2( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemHostGetDevicePointer_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemHostGetFlags( unsigned int *pFlags, void *p )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemHostGetFlags\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD16\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD16_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD2D16\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD2D16_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD2D16_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD2D32\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD2D32_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD2D32_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD2D8\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD2D8_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD2D8_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD32\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD32\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD8\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuMemsetD8_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    // Now, given the name of the kernel function being called, find
    // the entry for it.
    void * foo = hfunc;
    MODULE * module = dynamic_cast<MODULE *>((MODULE *)foo);
    std::map<char*, ENTRY *, ltstr>::iterator j = module->entry.find((char*)name);
    if (j == module->entry.end())
	{
		ERRS * error = new ERRS(CUDA_ERROR_NOT_FOUND, __FILE__, " " + __LINE__);
        return CUDA_ERROR_NOT_FOUND;
	}
	ENTRY * data = j->second;
    *hfunc = (CUfunction)data;
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleGetGlobal\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleGetGlobal_v2(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleGetGlobal_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleGetSurfRef\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleGetTexRef\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleLoad(CUmodule * hmod, const char *fname)
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
	{
		ERRS * error = new ERRS(CUDA_ERROR_FILE_NOT_FOUND, __FILE__, " " + __LINE__);
        return CUDA_ERROR_FILE_NOT_FOUND;
	}
	MODULE * module = this->Parse(this->device, (char*)buffer);
    *hmod = (CUmodule)module;
    if (module != 0)
    {
		ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
		return CUDA_SUCCESS;
	}
	else
    {
		ERRS * error = new ERRS(CUDA_ERROR_INVALID_CONTEXT, __FILE__, " " + __LINE__);
		return CUDA_ERROR_INVALID_CONTEXT;
	}
}

CUresult EMULATED_DEVICE::_cuModuleLoadData(CUmodule *module, const void *image)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleLoadData\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleLoadDataEx\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleLoadFatBinary\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuModuleUnload(CUmodule hmod)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuModuleUnload\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuParamSetf    (CUfunction hfunc, int offset, float value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuParamSetf\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuParamSeti    (CUfunction hfunc, int offset, unsigned int value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuParamSeti\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
    // Unknown what to do for param size.
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuParamSetTexRef\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    // record argument, size, offset.
    EMULATED_DEVICE::arg * a = new EMULATED_DEVICE::arg();
    a->argument = malloc(numbytes);
    memcpy(const_cast<void*>(a->argument), ptr, numbytes);
    a->size = numbytes;
    a->offset = offset;
    this->arguments.push_back(a);
	ERRS * error = new ERRS(CUDA_SUCCESS, __FILE__, " " + __LINE__);
    return CUDA_SUCCESS;
}

CUresult EMULATED_DEVICE::_cuStreamCreate( CUstream *phStream, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuStreamCreate\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuStreamDestroy( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuStreamDestroy\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuStreamQuery( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuStreamQuery\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuStreamSynchronize( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuStreamSynchronize\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuSurfRefGetArray\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuSurfRefSetArray\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefCreate( CUtexref *pTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefCreate\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefDestroy( CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefDestroy\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetAddress\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetAddress_v2( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetAddress_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetAddressMode\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetArray\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetFilterMode\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetFlags\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefGetFormat\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetAddress\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetAddress_v2( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetAddress_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetAddress2D\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetAddress2D_v2( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetAddress2D_v2\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetAddressMode\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetArray\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetFilterMode\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetFlags\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult EMULATED_DEVICE::_cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cuTexRefSetFormat\n";
	ERRS * error = new ERRS(CUDA_ERROR_INVALID_VALUE, __FILE__, " " + __LINE__);
    return CUDA_ERROR_INVALID_VALUE;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Cuda Runtime API/Emulator implementation.
///
/////////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t EMULATED_DEVICE::_cudaArrayGetInfo()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaArrayGetInfo\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaBindSurfaceToArray(const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Unimplemented function _cudaBindSurfaceToArray\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "_cudaBindTexture called, " << context << ".\n\n";
    }

    // Associate the "texref" with the rest of the info in this call.
    // When assigning or grabbing the values for texref, we'll need this information.
    TEXREF * tr = new TEXREF();
    tr->desc = (struct cudaChannelFormatDesc*)desc;
    tr->devPtr = (void*)devPtr;
    tr->offset = offset;
    tr->size = size;
    tr->texref = (struct textureReference*)texref;

    std::pair<void*, TEXREF*> i;
    i.first = (void*)texref;
    i.second = tr;
    this->texture_to_device_memory_binding.insert(i);

	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);

    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "_cudaBindTexture2D called, " << context << ".\n\n";
    }

    // Associate the "texref" with the rest of the info in this call.
    // When assigning or grabbing the values for texref, we'll need this information.
    TEXREF * tr = new TEXREF();
    tr->desc = (struct cudaChannelFormatDesc*)desc;
    tr->devPtr = (void*)devPtr;
    tr->offset = offset;
    tr->size = 0;
    tr->width = width;
    tr->height = height;
    tr->pitch = pitch;
    tr->texref = (struct textureReference*)texref;

    std::pair<void*, TEXREF*> i;
    i.first = (void*)texref;
    i.second = tr;
    this->texture_to_device_memory_binding.insert(i);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "_cudaBindTextureToArray called, " << context << ".\n\n";
    }

    // Associate the "texref" with the rest of the info in this call.
    // When assigning or grabbing the values for texref, we'll need this information.
    TEXARR * ta = new TEXARR();
    ta->texref = (struct textureReference*)texref;
    ta->array = (struct cudaArray *) array;
    ta->desc = (struct cudaChannelFormatDesc*)desc;

    std::pair<void*, TEXARR*> i;
    i.first = (void*)texref;
    i.second = ta;
    this->texture_to_array_binding.insert(i);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaBindTextureToMipmappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaBindTextureToMipmappedArray\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaChooseDevice called, " << context << ".\n\n";
    }
    *device = 0;
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    this->ConfigureBlock(blockDim);
    this->ConfigureGrid(gridDim);
    this->ConfigureSharedMemory(sharedMem);
    this->ConfigureStream(stream);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

struct cudaChannelFormatDesc EMULATED_DEVICE::_cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaCreateChannelDesc called, " << context << ".\n\n";
        (*cu->output_stream) << "cudaCreateChannelDesc parameters "
            << x
            << " "
            << y
            << " "
            << z
            << " "
            << w
            << " "
            << f
            << ".\n\n";
    }
    struct cudaChannelFormatDesc result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    result.f = f;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaCreateChannelDesc returns "
            << result.x
            << " "
            << result.y
            << " "
            << result.z
            << " "
            << result.w
            << " "
            << result.f
            << ".\n\n";
    }
    return result;
}

cudaError_t EMULATED_DEVICE::_cudaCreateSurfaceObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaCreateSurfaceObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaCreateTextureObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaCreateTextureObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10GetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10GetDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10GetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10GetDevices\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10GetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10GetDirect3DDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10MapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10MapResources\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10RegisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10ResourceGetMappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10ResourceGetMappedArray\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10ResourceGetMappedPitch()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10ResourceGetMappedPitch\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10ResourceGetMappedPointer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10ResourceGetMappedPointer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10ResourceGetMappedSize()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10ResourceGetMappedSize\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10ResourceGetSurfaceDimensions()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10ResourceGetSurfaceDimensions\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10ResourceSetMapFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10ResourceSetMapFlags\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10SetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10SetDirect3DDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10UnmapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10UnmapResources\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D10UnregisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D10UnregisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D11GetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D11GetDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D11GetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D11GetDevices\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D11GetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D11GetDirect3DDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D11SetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D11SetDirect3DDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9Begin()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9Begin\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9End()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9End\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9GetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9GetDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9GetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9GetDevices\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9GetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9GetDirect3DDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9MapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9MapResources\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9MapVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9MapVertexBuffer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9RegisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9RegisterVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9RegisterVertexBuffer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9ResourceGetMappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9ResourceGetMappedArray\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9ResourceGetMappedPitch()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9ResourceGetMappedPitch\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9ResourceGetMappedPointer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9ResourceGetMappedPointer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9ResourceGetMappedSize()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9ResourceGetMappedSize\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9ResourceGetSurfaceDimensions()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9ResourceGetSurfaceDimensions\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9ResourceSetMapFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9ResourceSetMapFlags\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9SetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9SetDirect3DDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9UnmapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9UnmapResources\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9UnmapVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9UnmapVertexBuffer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9UnregisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9UnregisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaD3D9UnregisterVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaD3D9UnregisterVertexBuffer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDestroySurfaceObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDestroySurfaceObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDestroyTextureObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDestroyTextureObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceCanAccessPeer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceCanAccessPeer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceDisablePeerAccess()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceDisablePeerAccess\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceEnablePeerAccess()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceEnablePeerAccess\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetAttribute()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetAttribute\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetByPCIBusId()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetByPCIBusId\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetCacheConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetLimit()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetLimit\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetPCIBusId()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetPCIBusId\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetSharedMemConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetSharedMemConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceGetStreamPriorityRange()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceGetStreamPriorityRange\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceReset()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceReset\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceSetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceSetCacheConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceSetLimit()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceSetLimit\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceSetSharedMemConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceSetSharedMemConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDeviceSynchronize()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaDeviceSynchronize\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaDriverGetVersion(int *driverVersion)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaDriverGetVersion is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventCreate(cudaEvent_t *event)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventCreate is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventCreateWithFlags(cudaEvent_t *event, int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventCreateWithFlags is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventDestroy(cudaEvent_t event)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventDestroy is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventElapsedTime is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventQuery(cudaEvent_t event)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventQuery is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventRecord is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaEventSynchronize(cudaEvent_t event)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaEventSynchronize is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaFree(void * ptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaFree called, ptr = " << ptr << ", "
             << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (ptr == 0)
    {
        (*cu->output_stream) << "Pointer in _cudaFree("
            << "ptr = " << ptr
            << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }

    int di = this->FindAllocatedBlock(ptr);
    if (di == -1)
    {
        (*cu->output_stream) << "Pointer to _cudaFree(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }
    EMULATED_DEVICE::data * d = &(*this->alloc_list)[di];
    void * local = ((char*)ptr) - cu->padding_size;
    if (d->ptr != local)
    {
        (*cu->output_stream) << "Pointer to _cudaFree(" << ptr << ") is invalid.\n\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }
    (*this->alloc_list).erase((*this->alloc_list).begin() + di);
    this->CheckSinglePtrOverwrite(d);
    free(local);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaFreeArray(struct cudaArray *array)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    struct cudaArray * ptr = array;
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaFreeArray called, ptr = " << ptr << ", "
                << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (ptr == 0)
    {
        (*cu->output_stream) << "Pointer in _cudaFreeArray("
                << "ptr = " << ptr
                << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
                << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }

    int di = this->FindAllocatedArray(ptr);
    if (di == -1)
    {
        (*cu->output_stream) << "Pointer to _cudaFreeArray(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
                << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }
    ARRAY * d = (ARRAY*) array;
    this->arrays.erase(this->arrays.begin() + di);
    free(d->Memory());
    delete d;
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaFreeHost(void * ptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;

    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaFreeHost called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (ptr == 0)
    {
        (*cu->output_stream) << "Pointer in FreeHost("
                << "ptr = " << ptr
                << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
		return cudaErrorInvalidDevicePointer;
    }

    int di = this->FindAllocatedBlock(ptr);
    if (di == -1)
    {
        (*cu->output_stream) << "Pointer to FreeHost(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
		return cudaErrorInvalidDevicePointer;
    }
    EMULATED_DEVICE::data * d = &(*this->alloc_list)[di];
    void * local = ((char*)ptr) - cu->padding_size;
    if (d->ptr != local)
    {
        (*cu->output_stream) << "Pointer to FreeHost(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
		return cudaErrorMemoryAllocation;
    }
    (*this->alloc_list).erase((*this->alloc_list).begin() + di);
    this->CheckSinglePtrOverwrite(d);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaFreeMipmappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaFreeMipmappedArray\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaFuncGetAttributes is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaFuncSetCacheConfig is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaFuncSetSharedMemConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaFuncSetSharedMemConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "_cudaGetChannelDesc called, " << context << ".\n\n";
    }
    ARRAY * arr = (ARRAY*)array;
    *desc = *arr->Desc();
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaGetDevice(int *device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetDevice is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetDeviceCount(int *count)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetDeviceCount called, " << context << ".\n\n";
    }
    *count = 1;
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    if (strcmp(this->device, "compute_20") == 0)
    {
        static cudaDeviceProp p = {
            "emulator",                     // char name[256];
            3000000000,                     // size_t totalGlobalMem;
            3000000,                        // size_t sharedMemPerBlock;
            500,                            // int regsPerBlock;
            500,                            // int warpSize;
            10,                             // size_t memPitch;
            16000,                          // int maxThreadsPerBlock;
            {1000,1,1},                     // int maxThreadsDim[3];
            {1000,1,1},                     // int maxGridSize[3];
            111,                            // int clockRate;
            100000000,                      // size_t totalConstMem;
            2,                              // int major;
            0,                              // int minor;
            11,                             // size_t textureAlignment;
            11,                             // int deviceOverlap;
            11,                             // int multiProcessorCount;
            1,                              // int kernelExecTimeoutEnabled;
            1,                              // int integrated;
            1,                              // int canMapHostMemory;
            1                               // int computeMode;
        };
        *prop = p;
    } else
    {
        static cudaDeviceProp p = {
            "emulator",                     // char name[256];
            3000000000,                     // size_t totalGlobalMem;
            3000000,                        // size_t sharedMemPerBlock;
            500,                            // int regsPerBlock;
            500,                            // int warpSize;
            10,                             // size_t memPitch;
            16000,                          // int maxThreadsPerBlock;
            {1000,1,1},                     // int maxThreadsDim[3];
            {1000,1,1},                     // int maxGridSize[3];
            111,                            // int clockRate;
            100000000,                      // size_t totalConstMem;
            1,                              // int major;
            4,                              // int minor;
            11,                             // size_t textureAlignment;
            11,                             // int deviceOverlap;
            11,                             // int multiProcessorCount;
            1,                              // int kernelExecTimeoutEnabled;
            1,                              // int integrated;
            1,                              // int canMapHostMemory;
            1                               // int computeMode;
        };
        *prop = p;
    }
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

char buff[5000];


const char* EMULATED_DEVICE::_cudaGetErrorString(cudaError_t error)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * errs = ERRS::LastError();
	std::string s = errs->String();
	s = s + " ";
	s = s + static_cast<std::ostringstream*>( &(std::ostringstream() << errs->Code()) )->str();
	s = s + " ";
	s = s + errs->Location();
	strcpy(buff, s.c_str());
	return buff;
}

cudaError_t EMULATED_DEVICE::_cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetExportTable is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetLastError(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetLastError called, " << context << ".\n\n";
    }
	ERRS * error = ERRS::LastError();
	if (error != 0)
	    return (cudaError_t)error->Code();
	else
		return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaGetMipmappedArrayLevel()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGetMipmappedArrayLevel\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetSurfaceAlignmentOffset(size_t *offset, const struct surfaceReference *surfref)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetSurfaceAlignmentOffset is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetSurfaceObjectResourceDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGetSurfaceObjectResourceDesc\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetSurfaceReference(const struct surfaceReference **surfref, const char *symbol)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetSurfaceReference is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetSymbolAddress(void **devPtr, const char *symbol)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetSymbolAddress is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetSymbolSize(size_t *size, const char *symbol)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetSymbolSize is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetTextureAlignmentOffset is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetTextureObjectResourceDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGetTextureObjectResourceDesc\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetTextureObjectResourceViewDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGetTextureObjectResourceViewDesc\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetTextureObjectTextureDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGetTextureObjectTextureDesc\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGetTextureReference(const struct textureReference **texref, const char *symbol)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGetTextureReference is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLGetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLGetDevices\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLMapBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLMapBufferObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLMapBufferObjectAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLMapBufferObjectAsync\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLRegisterBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLRegisterBufferObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLSetBufferObjectMapFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLSetBufferObjectMapFlags\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLSetGLDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLSetGLDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLUnmapBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLUnmapBufferObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLUnmapBufferObjectAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLUnmapBufferObjectAsync\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGLUnregisterBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGLUnregisterBufferObject\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsD3D10RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGraphicsD3D10RegisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsD3D11RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGraphicsD3D11RegisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsD3D9RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGraphicsD3D9RegisterResource\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsGLRegisterBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGraphicsGLRegisterBuffer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsGLRegisterImage()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaGraphicsGLRegisterImage\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsMapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGraphicsMapResources is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, struct cudaGraphicsResource *resource)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGraphicsResourceGetMappedPointer is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsResourceSetMapFlags(struct cudaGraphicsResource *resource, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGraphicsResourceSetMapFlags is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsSubResourceGetMappedArray(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGraphicsSubResourceGetMappedArray is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsUnmapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGraphicsUnmapResources is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaGraphicsUnregisterResource(struct cudaGraphicsResource *resource)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaGraphicsUnregisterResource is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaHostAlloc(void ** ptr, size_t size, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    void * local = 0;
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaHostAlloc called, " << context << ".\n\n";
    }

    if (this->IsBadPointer(ptr))
    {
        (*cu->output_stream) << "Bad pointer passed to HostAlloc("
            << ptr << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorMemoryAllocation, __FILE__, " " + __LINE__);
        return cudaErrorMemoryAllocation;
    }

    *ptr = 0;

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    char * hostbuffer = (char*)malloc(size + 2 * cu->padding_size);
    if (! hostbuffer)
    {
        (*cu->output_stream) << "Local memory allocation for setting buffer header and footer in cudaHostAlloc failed.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorMemoryAllocation, __FILE__, " " + __LINE__);
        return cudaErrorMemoryAllocation;
    }

    // Mark paddings with padding characters.
    for (char * init = (char*)hostbuffer; init < (((char*)hostbuffer) + cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    for (char * init = ((char*)hostbuffer) + size + cu->padding_size; init < (((char*)hostbuffer) + size + 2 * cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    // Init buffer with zeros.
    for (char * init = ((char*)hostbuffer) + cu->padding_size; init < (((char*)hostbuffer) + size + cu->padding_size); ++init)
    {
        *init = 0;
    }
    memcpy(local, hostbuffer, size + 2 * cu->padding_size);
    free(hostbuffer);
    EMULATED_DEVICE::data d;
    d.ptr = local;
    d.size = size + 2 * cu->padding_size;
    d.context = strdup(cu->Context());
    d.is_host = true;
    (*this->alloc_list).push_back(d);
    *ptr = ((char*)local) + cu->padding_size;
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaHostGetDevicePointer called, " << context << ".\n\n";
    }

    if (this->IsBadPointer(pDevice))
    {
        (*cu->output_stream) << "pDevice passed to HostGetDevicePointer(" << pDevice << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
    }
    if (flags != 0)
    {
        (*cu->output_stream) << "flags passed to HostGetDevicePointer(..., ..., " << flags << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
    }

    void * local = ((char*)pHost) - cu->padding_size;

    unsigned int i;
    for (i = 0; i < (*this->alloc_list).size(); ++i)
    {
        if ((*this->alloc_list)[i].ptr == local)
            break;
    }
    if (i == (*this->alloc_list).size())
    {
        (*cu->output_stream) << "pHost pointer passed to HostGetDevicePointer(..., " << pHost << ", ...) is invalid.\n";
    }
    EMULATED_DEVICE::data d = (*this->alloc_list)[i];
    if (! d.is_host)
    {
        (*cu->output_stream) << "pHost pointer passed to HostGetDevicePointer(..., " << pHost << ", ...) is invalid.\n";
        (*cu->output_stream) << " The pointer is not a pointer created by cudaHostAlloc\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
    }
    // Map pDevice correctly.
    *pDevice = (void*) (((char*)*pDevice) + cu->padding_size);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaHostGetFlags is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaHostRegister()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaHostRegister\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaHostUnregister()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaHostUnregister\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaIpcCloseMemHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaIpcCloseMemHandle\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaIpcGetEventHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaIpcGetEventHandle\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaIpcGetMemHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaIpcGetMemHandle\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaIpcOpenEventHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaIpcOpenEventHandle\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaIpcOpenMemHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaIpcOpenMemHandle\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaLaunch(const char *hostfun)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaLaunch called, " << context << ".\n\n";
    }

    // Given the address of the kernel function in the host, determine the name of the kernel
    // it is calling in PTX, using information provided by RegisterFatBinary and _cudaRegisterFunction.
    std::map<void*, char*>::iterator i = this->fun_to_name.find((void*)hostfun);
    assert(i != this->fun_to_name.end());
    char * name = i->second;

	std::vector<ENTRY*> possible_entries;

    // Go through all modules, look entry.
    for (std::vector<MODULE *>::iterator it = this->modules.begin(); it != this->modules.end(); ++it)
    {
        MODULE * module = *it;
        std::map<char*, ENTRY *, ltstr>::iterator j = module->entry.find(name);
		if (j != module->entry.end())
		{
			possible_entries.push_back(j->second);
		}
	}

	if (possible_entries.size() == 0)
	{
		ERRS * error = new ERRS(cudaErrorInvalidDeviceFunction, __FILE__, " " + __LINE__);
		return cudaErrorInvalidDeviceFunction;
	}
	
	if (possible_entries.size() == 1)
	{
		ENTRY * entry = possible_entries[0];
        this->Execute(entry);
		ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
        return cudaSuccess; // seems dubious.
	}

	ERRS * error = new ERRS(cudaErrorInvalidDeviceFunction, __FILE__, " " + __LINE__);
    return cudaErrorInvalidDeviceFunction;
}

cudaError_t EMULATED_DEVICE::_cudaMalloc(void ** ptr, size_t size)
{
    void * local = 0;
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMalloc called, " << context << ".\n\n";
        (*cu->output_stream).flush();
    }
            
    if (this->IsBadPointer(ptr))
    {
        (*cu->output_stream) << "Bad pointer passed to cudaMalloc("
            << ptr << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorMemoryAllocation, __FILE__, " " + __LINE__);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t resetErrKernel = _cudaGetLastError();

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    {
        local = malloc(size+2*cu->padding_size);
    }
    // Mark paddings with padding characters.
    for (char * init = (char*)local; init < (((char*)local) + cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    for (char * init = ((char*)local) + size + cu->padding_size; init < (((char*)local) + size + 2 * cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    // Init buffer with zeros.
    for (char * init = ((char*)local) + cu->padding_size; init < (((char*)local) + size + cu->padding_size); ++init)
    {
        *init = 0;
    }
    EMULATED_DEVICE::data d;
    d.ptr = local;
	std::cout << "malloc " << local << " size " << size << "\n";
    d.size = size + 2 * cu->padding_size;
    d.is_host = false;
    d.context = strdup(cu->Context());
    this->alloc_list->push_back(d);
    *ptr = ((char*)local) + cu->padding_size;
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaMalloc3D()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMalloc3D\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMalloc3DArray(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMalloc3DArray is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0))
{
    void * local = 0;
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMallocArray called, " << context << ".\n\n";
        (*cu->output_stream).flush();
    }

    if (this->IsBadPointer(desc))
    {
        (*cu->output_stream) << "Bad pointer passed to cudaMallocArray("
                << desc << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorMemoryAllocation, __FILE__, " " + __LINE__);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t resetErrKernel = _cudaGetLastError();

    // calculate size.
    // NOTE: a cudaChannelFormatDesc is a structure of the sizes of components of a linear piece of memory.  It is not a 4D cube.
    unsigned int size = width * height * (desc->x + desc->y + desc->z + desc->w)/8;  // bytes.
    ARRAY * arr = new ARRAY((struct cudaChannelFormatDesc *)desc, width, height, flags);

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    local = (void*) malloc(size+2*cu->padding_size);
    if (! local)
    {
        (*cu->output_stream) << "Host memory allocation failed in cudaMallocArray.  The buffer is used to initialize the device buffer.\n";
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorMemoryAllocation, __FILE__, " " + __LINE__);
        return cudaErrorMemoryAllocation;
    }
    // Mark paddings with padding characters.
    for (char * init = (char*)local; init < (((char*)local) + cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    for (char * init = ((char*)local) + size + cu->padding_size; init < (((char*)local) + size + 2 * cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    // Init buffer with zeros.
    for (char * init = ((char*)local) + cu->padding_size; init < (((char*)local) + size + cu->padding_size); ++init)
    {
        *init = 0;
    }
    
    arr->Memory((unsigned char*)(((char*)local)));

    *array = (struct cudaArray*) arr;
    this->arrays.push_back(arr);

	std::cout << "malloc " << local << " size " << size << "\n";

	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;     
}

cudaError_t EMULATED_DEVICE::_cudaMallocHost(void **ptr, size_t size)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMallocHost is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMallocMipmappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMallocMipmappedArray\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

int roundUp(size_t numToRound, size_t multiple)
{
    if(multiple == 0)
    {
        return numToRound;
    }

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;
    return numToRound + multiple - remainder;
}

cudaError_t EMULATED_DEVICE::_cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    void * local = 0;
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    // Round up width to be multiple of 128.
    size_t size = roundUp(width, 128 * 4) * height;

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaMallocPitch called, " << context << ".\n\n";
        (*cu->output_stream).flush();
    }
            
    if (this->IsBadPointer(devPtr))
    {
        (*cu->output_stream) << "Bad pointer passed to _cudaMallocPitch("
            << devPtr << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorMemoryAllocation, __FILE__, " " + __LINE__);
        return cudaErrorMemoryAllocation;
    }

    cudaError_t resetErrKernel = _cudaGetLastError();

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    {
        local = malloc(size+2*cu->padding_size);
    }
    // Mark paddings with padding characters.
    for (char * init = (char*)local; init < (((char*)local) + cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    for (char * init = ((char*)local) + size + cu->padding_size; init < (((char*)local) + size + 2 * cu->padding_size); ++init)
    {
        *init = cu->padding_byte;
    }
    // Init buffer with zeros.
    for (char * init = ((char*)local) + cu->padding_size; init < (((char*)local) + size + cu->padding_size); ++init)
    {
        *init = 0;
    }
    EMULATED_DEVICE::data d;
    d.ptr = local;
    d.size = size + 2 * cu->padding_size;
    d.is_host = false;
    d.context = strdup(cu->Context());
    this->alloc_list->push_back(d);
    *devPtr = ((char*)local) + cu->padding_size;
    *pitch = size / height;
	std::cout << "malloc " << local << " size " << size << "\n";
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMemcpy called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (dst == 0)
    {
        (*cu->output_stream) << "Destination pointer in _cudaMemcpy("
            << "dst = " << dst
            << ", ..., ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
        //memcpy(dst, src, count);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }
    if (src == 0)
    {
        (*cu->output_stream) << "Source pointer passed to _cudaMemcpy(..., "
            << "src = " << src
            << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
        //memcpy(dst, src, count);
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }

    // Four cases:
    if (kind == cudaMemcpyHostToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds != -1 && dd == -1)
        {
            (*cu->output_stream) << "Source and destination pointers in _cudaMemcpy("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " are reversed in directionality.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (dd == -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy("
                << "dst = " << dst
                << ", ..., ..., ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (ds != -1 && ! (*this->alloc_list)[ds].is_host)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy(..., src = " << src
                << ", ..., ...) looks invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (ds != -1 && (*this->alloc_list)[ds].is_host)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy(..., src = " << src
                << ", ..., ...) is a pointer to a host block that could be device addressible.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        EMULATED_DEVICE::data * ddst = 0;
        EMULATED_DEVICE::data * dsrc = 0;
        if (dd != -1)
            ddst = &(*this->alloc_list)[dd];
        if (ds != -1)
            dsrc = &(*this->alloc_list)[ds];
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        memcpy(dst, src, count);
        err = cudaSuccess;
        // Perform overwrite check again.
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
		return err;
    }
    else if (kind == cudaMemcpyDeviceToHost)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        if (ds == -1 && dd != -1)
        {
            (*cu->output_stream) << "Source and destination pointers in _cudaMemcpy("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " are reversed in directionality.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy(..., "
                << "src = " << src
                << ", ..., ..., ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (dd != -1 && ! (*this->alloc_list)[dd].is_host)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpy(..., "
                << "src = " << src
                << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (dd != -1 && (*this->alloc_list)[dd].is_host)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpy("
                << "dst = " << dst
                << ", ..., ..., ...) is a pointer to a host block that could be device addressible.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (this->IsBadPointer(dst))
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpy("
                << "dst = " << dst
                << ", ..., ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        // Check before copy if block boundaries are intact.
        EMULATED_DEVICE::data * ddst = 0;
        EMULATED_DEVICE::data * dsrc = 0;
        if (dd != -1)
            ddst = &(*this->alloc_list)[dd];
        if (ds != -1)
            dsrc = &(*this->alloc_list)[ds];
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        memcpy(dst, src, count);
        err = cudaSuccess;
        // Perform overwrite check again.
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
		return err;
    }
    else
	{
		ERRS * error = new ERRS(cudaErrorInvalidMemcpyDirection, __FILE__, " " + __LINE__);
        return cudaErrorInvalidMemcpyDirection;
	}
}

void copy_aux(char * dst, size_t dpitch, char * src, size_t spitch, size_t width, size_t height)
{
    for (int i = 0, j = 0; ;)
    {
        if (i % spitch >= width)
        {
            int inc = spitch - width;
            i += inc;
        }
        if (j % dpitch >= width)
        {
            int inc = dpitch - width;
            j += inc;
        }
        dst[j++] = src[i++];
        if (i >= spitch * height)
            break;
    }
}


cudaError_t EMULATED_DEVICE::_cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = cu->Context();
    int count = spitch * height;

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaMemcpy2D called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (dst == 0)
    {
        (*cu->output_stream) << "Destination pointer in _cudaMemcpy2D("
            << "dst = " << dst
            << ", ..., ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }
    if (src == 0)
    {
        (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2D(..., "
            << "src = " << src
            << ", ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }

    // Four cases:
    if (kind == cudaMemcpyHostToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds != -1 && dd == -1)
        {
            (*cu->output_stream) << "Source and destination pointers in _cudaMemcpy2D("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " are reversed in directionality.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (dd == -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2D("
                << "dst = " << dst
                << ", ..., ..., ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (ds != -1 && ! (*this->alloc_list)[ds].is_host)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2D(..., src = " << src
                << ", ..., ...) looks invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (ds != -1 && (*this->alloc_list)[ds].is_host)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2D(..., src = " << src
                << ", ..., ...) is a pointer to a host block that could be device addressible.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2D(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        EMULATED_DEVICE::data * ddst = 0;
        EMULATED_DEVICE::data * dsrc = 0;
        if (dd != -1)
            ddst = &(*this->alloc_list)[dd];
        if (ds != -1)
            dsrc = &(*this->alloc_list)[ds];
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        copy_aux((char*)dst, dpitch, (char*)src, spitch, width, height);
        err = cudaSuccess;
        // Perform overwrite check again.
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToHost)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        if (ds == -1 && dd != -1)
        {
            (*cu->output_stream) << "Source and destination pointers in _cudaMemcpy2D("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " are reversed in directionality.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2D(..., "
                << "src = " << src
                << ", ..., ..., ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (dd != -1 && ! (*this->alloc_list)[dd].is_host)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpy2D(..., "
                << "src = " << src
                << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (dd != -1 && (*this->alloc_list)[dd].is_host)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpy2D("
                << "dst = " << dst
                << ", ..., ..., ...) is a pointer to a host block that could be device addressible.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        else if (this->IsBadPointer(dst))
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpy2D("
                << "dst = " << dst
                << ", ..., ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        // Check before copy if block boundaries are intact.
        EMULATED_DEVICE::data * ddst = 0;
        EMULATED_DEVICE::data * dsrc = 0;
        if (dd != -1)
            ddst = &(*this->alloc_list)[dd];
        if (ds != -1)
            dsrc = &(*this->alloc_list)[ds];
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        copy_aux((char*)dst, dpitch, (char*)src, spitch, width, height);
        err = cudaSuccess;
        // Perform overwrite check again.
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2D("
                << "dst = " << dst
                << "src = " << src
                << ",  ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (dd == -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2D("
                << "dst = " << dst
                << "src = " << src
                << ", ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        
        if (this->IsBadPointer(dst))
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2D("
                << "dst = " << dst
                << "src = " << src
                << ", ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2D("
                << "dst = " << dst
                << "src = " << src
                << ", ...) "
                << " is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        // Check before copy if block boundaries are intact.
        EMULATED_DEVICE::data * ddst = 0;
        EMULATED_DEVICE::data * dsrc = 0;
        if (dd != -1)
            ddst = &(*this->alloc_list)[dd];
        if (ds != -1)
            dsrc = &(*this->alloc_list)[ds];
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        copy_aux((char*)dst, dpitch, (char*)src, spitch, width, height);
        err = cudaSuccess;
        // Perform overwrite check again.
        if (ddst)
            this->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            this->CheckSinglePtrOverwrite(dsrc);
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else
        return cudaErrorMemoryAllocation;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpy2DArrayToArray is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpy2DAsync is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpy2DFromArray is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpy2DFromArrayAsync is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = cu->Context();
    int count = spitch * height;

    ARRAY * arr = (ARRAY*) dst;

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaMemcpy2D called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (dst == 0)
    {
        (*cu->output_stream) << "Destination pointer in _cudaMemcpy2DToArray("
            << "dst = " << dst
            << ", ..., ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }
    if (src == 0)
    {
        (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2DToArray(..., "
            << "src = " << src
            << ", ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
        return cudaErrorInvalidDevicePointer;
    }

    // Four cases:
    if (kind == cudaMemcpyHostToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        int ad = this->FindAllocatedArray(dst);
        int as = this->FindAllocatedArray(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds != -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2DToArray("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " is on the device, but cudaMemcpyHostToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2DToArray("
                << "dst = " << dst
                << ", ..., ..., ...) "
                << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2DToArray(..., src = " << src
                << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (ad == -1)
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy(..., src = " << src
                << ", ..., ...) is not an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }

        // Perform copy.
        cudaError_t err;
        copy_aux((char*)arr->Memory() + cu->padding_size, width, (char*)src, spitch, width, height);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        int ad = this->FindAllocatedArray(dst);
        int as = this->FindAllocatedArray(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2DToArray("
                    << "dst = " << dst
                    << ", src = " << src << ", ..., ...) "
                    << " is a host pointer, but cudaMemcpyDeviceToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2DToArray("
                    << "dst = " << dst
                    << ", ..., ..., ...) "
                    << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2DToArray(..., src = " << src
                    << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (ad == -1)
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy(..., src = " << src
                    << ", ..., ...) is not an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidDevicePointer, __FILE__, " " + __LINE__);
			return cudaErrorInvalidDevicePointer;
        }

        // Perform copy.
        cudaError_t err;
        copy_aux((char*)arr->Memory() + cu->padding_size, width, (char*)src, spitch, width, height);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else {
        (*cu->output_stream) << "Direction copy to _cudaMemcpy2DToArray(..., "
                << "src = " << src
                << ", ..., ...) is invalid. You can only copy from host/device to an array.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy3D(const struct cudaMemcpy3DParms *p)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpy3D is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpy3D(p);
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy3DPeer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMemcpy3DPeer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpy3DPeerAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMemcpy3DPeerAsync\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpyArrayToArray is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpy(dst, src, count, kind);
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemcpyFromArray is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaMemcpyFromSymbol called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (symbol == 0)
    {
        (*cu->output_stream) << "Destination pointer in _cudaMemcpyFromSymbol("
                << "dst = " << dst
                << "symbol = " << symbol
                << ", ..., ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
		return cudaErrorInvalidValue;
    }

    if (dst == 0)
    {
        (*cu->output_stream) << "Source pointer passed to _cudaMemcpyFromSymbol("
                << "dst = " << dst
                << "symbol = " << symbol
                << ", ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }

    if (kind == cudaMemcpyDeviceToHost)
    {
        int ds = this->FindAllocatedBlock(symbol);
        int dd = this->FindAllocatedBlock(dst);
        int as = this->FindAllocatedArray(symbol);
        int ad = this->FindAllocatedArray(dst);

        if (ds != -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ") where symbol is memory allocated by cudaMalloc, not a symbol.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ..., ...) "
                    << " where dst is device memory.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ...) where symbol is device array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (ad != -1)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ...) is dst is device array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (this->IsBadPointer(dst))
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        std::map<void*, REGVAR*>::iterator j = this->variables.find((void*)symbol);
        if (j == this->variables.end())
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}

        REGVAR * var = j->second;

        SYMBOL * sym = this->global_symbol_table->FindSymbol(var->deviceAddress);
        if (sym == 0)
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}

        // Perform copy.
        cudaError_t err;
        memcpy((char*)dst , sym->pvalue, count);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        int ds = this->FindAllocatedBlock(symbol);
        int dd = this->FindAllocatedBlock(dst);
        int as = this->FindAllocatedArray(symbol);
        int ad = this->FindAllocatedArray(dst);

        if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << " is not on the device, but cudaMemcpyDeviceToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ..., ...) "
                    << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (ad != -1)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                    << ", ..., ...) is not an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        if (this->IsBadPointer(dst))
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyFromSymbol("
                    << "dst = " << dst
                    << "symbol = " << symbol
                << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        std::map<void*, REGVAR*>::iterator j = this->variables.find((void*)symbol);
        if (j == this->variables.end())
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}
        
        REGVAR * var = j->second;

        SYMBOL * sym = this->global_symbol_table->FindSymbol(var->deviceAddress);
        if (sym == 0)
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}

        // Perform copy.
        cudaError_t err;
        memcpy((char*)dst , sym->pvalue, count);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else {
        (*cu->output_stream) << "Direction copy to _cudaMemcpyFromSymbol("
                << "dst = " << dst
                << "symbol = " << symbol
                << ", ..., ...) is invalid. You can only copy from host/device to an array.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyPeer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMemcpyPeer\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyPeerAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMemcpyPeerAsync\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = cu->Context();

    assert(wOffset == 0 && hOffset == 0);

    ARRAY * arr = (ARRAY*) dst;

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaMemcpy2D called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (dst == 0)
    {
        (*cu->output_stream) << "Destination pointer in _cudaMemcpy2DToArray("
                << "dst = " << dst
                << ", ..., ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        //memcpy(dst, src, count);
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
    if (src == 0)
    {
        (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2DToArray(..., "
                << "src = " << src
                << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        //memcpy(dst, src, count);
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }

    if (kind == cudaMemcpyHostToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        int ad = this->FindAllocatedArray(dst);
        int as = this->FindAllocatedArray(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds != -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2DToArray("
                    << "dst = " << dst
                    << ", src = " << src << ", ..., ...) "
                    << " is on the device, but cudaMemcpyHostToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2DToArray("
                    << "dst = " << dst
                    << ", ..., ..., ...) "
                    << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2DToArray(..., src = " << src
                    << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (ad == -1)
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy(..., src = " << src
                    << ", ..., ...) is not an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        // Perform copy.
        cudaError_t err;
        memcpy((char*)arr->Memory() + cu->padding_size, src, count);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        int dd = this->FindAllocatedBlock(dst);
        int ds = this->FindAllocatedBlock(src);
        int ad = this->FindAllocatedArray(dst);
        int as = this->FindAllocatedArray(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpy2DToArray("
                    << "dst = " << dst
                    << ", src = " << src << ", ..., ...) "
                    << " is a host pointer, but cudaMemcpyDeviceToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpy2DToArray("
                    << "dst = " << dst
                    << ", ..., ..., ...) "
                    << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpy2DToArray(..., src = " << src
                    << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (ad == -1)
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy(..., src = " << src
                    << ", ..., ...) is not an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        // Perform copy.
        cudaError_t err;
        memcpy((char*)arr->Memory() + cu->padding_size, src, count);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else {
        (*cu->output_stream) << "Direction copy to _cudaMemcpy2DToArray(..., "
                << "src = " << src
                << ", ..., ...) is invalid. You can only copy from host/device to an array.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "_cudaMemcpyToSymbol called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (symbol == 0)
    {
        (*cu->output_stream) << "Destination pointer in _cudaMemcpyToSymbol("
                << "symbol = " << symbol
                << ", ..., ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }

    if (src == 0)
    {
        (*cu->output_stream) << "Source pointer passed to _cudaMemcpyToSymbol(..., "
                << "src = " << src
                << ", ..., ...) is invalid.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }

    if (kind == cudaMemcpyHostToDevice)
    {
        int dd = this->FindAllocatedBlock(symbol);
        int ds = this->FindAllocatedBlock(src);
        int ad = this->FindAllocatedArray(symbol);
        int as = this->FindAllocatedArray(src);

        if (ds != -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpyToSymbol("
                    << "symbol = " << symbol
                    << ", src = " << src << ", ...) "
                    << " is on the device, but cudaMemcpyHostToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpyToSymbol("
                    << "symbol = " << symbol
                    << ", ..., ..., ...) "
                    << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyToSymbol(..., src = " << src
                    << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (ad != -1)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpyToSymbol(..., src = " << src
                    << ", ..., ...) is an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyToSymbol(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        std::map<void*, REGVAR*>::iterator j = this->variables.find((void*)symbol);
        if (j == this->variables.end())
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}

        REGVAR * var = j->second;

        SYMBOL * sym = this->global_symbol_table->FindSymbol(var->deviceAddress);
        if (sym == 0)
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}

        // Perform copy.
        cudaError_t err;
        memcpy((char*)sym->pvalue, src, count);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToDevice)
    {
        int dd = this->FindAllocatedBlock(symbol);
        int ds = this->FindAllocatedBlock(src);
        int ad = this->FindAllocatedArray(symbol);
        int as = this->FindAllocatedArray(src);

        if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in _cudaMemcpyToSymbol("
                    << "symbol = " << symbol
                    << ", src = " << src << ", ...) "
                    << " is not on the device, but cudaMemcpyDeviceToDevice specified.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (dd != -1)
        {
            (*cu->output_stream) << "Destination pointer in _cudaMemcpyToSymbol("
                    << "symbol = " << symbol
                    << ", ..., ..., ...) "
                    << " should be array, but was allocated using cudaMalloc.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (as != -1)
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyToSymbol(..., src = " << src
                    << ", ..., ...) is device pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (ad != -1)
        {
            (*cu->output_stream) << "Destination pointer passed to _cudaMemcpyToSymbol(..., src = " << src
                    << ", ..., ...) is an array pointer.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }
        if (this->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to _cudaMemcpyToSymbol(..., src = " << src << ", ..., ...) is invalid.\n";
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
        }

        std::map<void*, REGVAR*>::iterator j = this->variables.find((void*)symbol);
        if (j == this->variables.end())
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}
        
        REGVAR * var = j->second;

        SYMBOL * sym = this->global_symbol_table->FindSymbol(var->deviceAddress);
        if (sym == 0)
		{
			ERRS * error = new ERRS(cudaErrorInvalidSymbol, __FILE__, " " + __LINE__);
            return cudaErrorInvalidSymbol;
		}

        // Perform copy.
        cudaError_t err;
        memcpy((char*)sym->pvalue , src, count);
        err = cudaSuccess;
        // Perform overwrite check again.
		ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
        return err;
    }
    else {
        (*cu->output_stream) << "Direction copy to _cudaMemcpyToSymbol(..., "
                << "src = " << src
                << ", ..., ...) is invalid. You can only copy from host/device to an array.\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
}

cudaError_t EMULATED_DEVICE::_cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t EMULATED_DEVICE::_cudaMemGetInfo(size_t *free, size_t *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    *free = 250000000;
    *total = 3000000000;
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaMemset(void * dst, int value, size_t count)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMemset called, " << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (dst == 0)
    {
        (*cu->output_stream) << "Destination pointer in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }

    int dd = this->FindAllocatedBlock(dst);
    if (dd == -1)
    {
        (*cu->output_stream) << "Destination pointer in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
    EMULATED_DEVICE::data * ddst = 0;
    ddst = &(*this->alloc_list)[dd];
    if (cu->device_pointer_to_first_byte_in_block)
    {
        void * local = ((char*)dst) - cu->padding_size;
        if (ddst->ptr != local)
        {
            (*cu->output_stream) << "Destination pointer in Memset("
                << "dst = " << dst
                << ", ..., ...) is invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in "
                << context << ".\n\n";
            if (cu->quit_on_error)
                exit(1);
            if (cu->do_not_call_cuda_after_sanity_check_fail)
			{
				ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
                return cudaErrorInvalidValue;
			}
        }
    }

    if (this->CheckSinglePtrOverwrite(ddst) != CUDA_WRAPPER::OK)
    {
        (*cu->output_stream) << "Destination block in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid -- overwritten.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
		{
			ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
            return cudaErrorInvalidValue;
		}
    }
    // Perform copy.
    cudaError_t err;
    memset(dst, value, count);
    err = cudaSuccess;

    // Perform overwrite check again.
    if (this->CheckSinglePtrOverwrite(ddst) != CUDA_WRAPPER::OK)
    {
        (*cu->output_stream) << "Destination block in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid -- overwritten.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
		ERRS * error = new ERRS(cudaErrorInvalidValue, __FILE__, " " + __LINE__);
        return cudaErrorInvalidValue;
    }
	ERRS * error = new ERRS(err, __FILE__, " " + __LINE__);
    return err;
}

cudaError_t EMULATED_DEVICE::_cudaMemsetAsync(void * dst, int value, size_t count, cudaStream_t stream = 0)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemset(dst, value, count);
}

cudaError_t EMULATED_DEVICE::_cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemset2D is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return this->_cudaMemset2D(devPtr, pitch, value, width, height);
}

cudaError_t EMULATED_DEVICE::_cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaMemset3D is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaMemset3DAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaMemset3DAsync\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaPeekAtLastError(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaPeekAtLastError is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaPointerGetAttributes()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaPointerGetAttributes\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaProfilerStart()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaProfilerStart\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaProfilerStop()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaProfilerStop\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaRegisterDeviceFunction()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaRegisterDeviceFunction\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

void** EMULATED_DEVICE::_cudaRegisterFatBinary(void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * profile = 0;
    char * ptx = 0;

    if(*(int*)fatCubin == __cudaFatMAGIC)
    {
        (*cu->output_stream) << "Found old fat binary format!" << "\n";
        __cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)fatCubin;
        profile = binary->ident;

        unsigned int ptxVersion = 0;

        // Get the highest PTX version

        for (unsigned int i = 0; ; ++i)
        {
            if((binary->ptx[i].ptx) == 0)
                break;

            std::string computeCapability = binary->ptx[i].gpuProfileName;
            std::string versionString(computeCapability.begin() + 8,
                computeCapability.end());

            unsigned int thisVersion = 0;
            
            thisVersion = ::stoi(versionString.c_str());
            if(thisVersion > ptxVersion)
            {
                ptxVersion = thisVersion;
                ptx = binary->ptx[i].ptx;
            }
        }
        (*cu->output_stream) << " Selected version " << ptxVersion << "\n";

        this->Parse(profile, ptx);
    }
    else if (*(int*)fatCubin == __cudaFatMAGIC2) {
        (*cu->output_stream) << "Found new fat binary format!" << "\n";
        __cudaFatCudaBinary2* binary = (__cudaFatCudaBinary2*) fatCubin;
        __cudaFatCudaBinary2Header* header =
            (__cudaFatCudaBinary2Header*) binary->fatbinData;

        (*cu->output_stream) << " binary size is: " << header->length << " bytes\n";

        char* base = (char*)(header + 1);
        long long unsigned int offset = 0;
        __cudaFatCudaBinary2EntryRec* entry = (__cudaFatCudaBinary2EntryRec*)(base);

        while (!(entry->type & FATBIN_2_PTX) && offset < header->length) {
            entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
            offset += entry->binary + entry->binarySize;
        }
        profile = (char*)entry + entry->name;       
        if (entry->type & FATBIN_2_PTX) {
            ptx  = (char*)entry + entry->binary;
        }
        else {
            ptx = 0;
        }

        if(entry->flags & COMPRESSED_PTX)
        {
            int ret, flush;
            unsigned have;
            z_stream strm;
            int size = entry->binarySize;
            unsigned char * in = (unsigned char*)ptx;
            int out_size = entry->uncompressedBinarySize * 2;
            unsigned char * out = (unsigned char*)malloc(out_size);
            /* allocate inflate state */
            strm.zalloc = Z_NULL;
            strm.zfree = Z_NULL;
            strm.opaque = Z_NULL;
            strm.avail_in = 0;
            strm.next_in = Z_NULL;
            ret = inflateInit(&strm);
            if (ret != Z_OK)
                return 0;       
            /* decompress until deflate stream ends or end of file */
            char * in_ptr = (char*)in;
            char * out_ptr = (char*)out;
            do {
                // For now, take whole thing.
                int in_size = size;
                size -= in_size;
                strm.avail_in = in_size;
                if (strm.avail_in == 0)
                    break;
                strm.next_in = (Bytef*) in_ptr;
                /* run inflate() on input until output buffer not full */
                do {
                    strm.avail_out = out_size;
                    strm.next_out = (unsigned char*)out_ptr;
                    ret = inflate(&strm, Z_NO_FLUSH);
                    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
                    switch (ret) {
                        case Z_NEED_DICT:
                            ret = Z_DATA_ERROR;     /* and fall through */
                        case Z_DATA_ERROR:
                        case Z_MEM_ERROR:
                            (void)inflateEnd(&strm);
                            return 0;
                    }
                    have = in_size - strm.avail_out;
                    out_ptr += have;
                } while (strm.avail_out == 0);
                /* done when inflate() says it's done */
            } while (ret != Z_STREAM_END);
            /* clean up and return */
            (void)inflateEnd(&strm);
            ptx = (char*)out;
            *(ptx + entry->uncompressedBinarySize) = 0;
        }

        this->Parse(profile, ptx);

    }
    return 0;
}

void EMULATED_DEVICE::_cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::pair<void*, char*> i;
    i.first = (void*)hostFun;
    i.second = deviceFun;
    this->fun_to_name.insert(i);
}

cudaError_t EMULATED_DEVICE::_cudaRegisterPrelinkedFatBinary()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaRegisterPrelinkedFatBinary\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaRegisterShared()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaRegisterShared\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaRegisterSharedVar()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaRegisterSharedVar\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

void EMULATED_DEVICE::_cudaRegisterSurface(void **fatCubinHandle, const struct surfaceReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaRegisterSurface is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
}

void EMULATED_DEVICE::_cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *textureName, int dim, int norm, int ext)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "__cudaRegisterTexture called, " << context << ".\n\n";
    }
    // Associate "textureName" with "hostVar".  When using textures, we'll need to get the address 
    // of the texture in memory from the name of the texture.
    TEXTURE * texture = new TEXTURE();
    texture->hostVar = (struct textureReference*)hostVar;
    texture->deviceAddress = (void**) deviceAddress;
    texture->textureName = (char*) textureName;
    texture->dim = dim;
    texture->norm = norm;
    texture->ext = ext;
    std::pair<char*, TEXTURE*> i;
    i.first = (char*)textureName;
    i.second = (TEXTURE*)texture;
    this->texturename_to_texture.insert(i);
}

void EMULATED_DEVICE::_cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "_cudaRegisterVar called, " << context << ".\n\n";
    }

    REGVAR * var = new REGVAR();
    var->constant = constant;
    var->deviceAddress = deviceAddress;
    var->deviceName = const_cast<char*>(deviceName);
    var->ext = ext;
    var->global = global;
    var->hostVar = hostVar;
    var->size = size;
    std::pair<void*, REGVAR*> i;
    i.first = (void*)hostVar;
    i.second = var;
    this->variables.insert(i);
}

cudaError_t EMULATED_DEVICE::_cudaRuntimeGetVersion(int *runtimeVersion)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaRuntimeGetVersion is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaSetDevice(int device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaSetDevice called, " << context << ".\n\n";
    }
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaSetDeviceFlags(unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "_cudaSetDeviceFlags called, " << context << ".\n\n";
    }
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
   return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaSetDoubleForDevice(double *d)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaSetDoubleForDevice is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaSetDoubleForHost(double *d)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaSetDoubleForHost is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    // record argument, size, offset.
    EMULATED_DEVICE::arg * a = new EMULATED_DEVICE::arg();
    //assert(size == 4);
    a->argument = malloc(size);
    memcpy(const_cast<void*>(a->argument), arg, size);
    a->size = size;
    a->offset = offset;
    this->arguments.push_back(a);
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaSetValidDevices(int *device_arr, int len)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaSetValidDevices is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaStreamAddCallback()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaStreamAddCallback\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaStreamCreate(cudaStream_t *pStream)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaStreamCreateWithFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaStreamCreateWithFlags\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaStreamCreateWithPriority()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaStreamCreateWithPriority\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaStreamDestroy(cudaStream_t stream)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaStreamGetFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaStreamGetFlags\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaStreamGetPriority()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaStreamGetPriority\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaStreamQuery(cudaStream_t stream)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaStreamSynchronize(cudaStream_t stream)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaStreamWaitEvent()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaStreamWaitEvent\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaThreadExit()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaThreadExit called, " << context << ".\n\n";
    }

    if ((*this->alloc_list).size() != 0)
    {
        (*cu->output_stream) << " Warning, cudaThreadExit reinitializes the entire runtime.  All pointers to old cuda memory are stale.\n";
    }

    (*this->alloc_list).clear();
    cudaError_t result = cudaSuccess;
    result = cudaSuccess;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaThreadExit returns " << result << ".\n\n";
    }
	ERRS * error = new ERRS(result, __FILE__, " " + __LINE__);
    return result;
}

cudaError_t EMULATED_DEVICE::_cudaThreadGetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaThreadGetCacheConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaThreadGetLimit(enum cudaLimit limit, size_t value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaThreadGetLimit is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaThreadSetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaThreadSetCacheConfig\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::cout << "Function _cudaThreadSetLimit is not implemented.\n";
    _CUDA_RUNTIME::Unimplemented();
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
    return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_cudaThreadSynchronize(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

cudaError_t EMULATED_DEVICE::_cudaUnbindTexture(const struct textureReference *texref)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaUnbindTexture called, " << context << ".\n\n";
    }
	ERRS * error = new ERRS(cudaSuccess, __FILE__, " " + __LINE__);
    return cudaSuccess;
}

void EMULATED_DEVICE::_cudaUnregisterFatBinary(void **fatCubinHandle)
{
    // Should probably do something like free the ast...
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaUnregisterFatBinary called, " << context << ".\n\n";
    }
}

cudaError_t EMULATED_DEVICE::_cudaWGLGetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _cudaWGLGetDevice\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}

cudaError_t EMULATED_DEVICE::_NvOptimusEnablementCuda()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	std::cout << "Unimplemented function _NvOptimusEnablementCuda\n";
	ERRS * error = new ERRS(cudaErrorNotYetImplemented, __FILE__, " " + __LINE__);
	return cudaErrorNotYetImplemented;
}














CRIT_SECTION EMULATED_DEVICE::sm_CritSec;

EMULATED_DEVICE::EMULATED_DEVICE()
{
    this->device = "compute_20";
    this->string_table = new STRING_TABLE();
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    this->extern_memory_buffer = 0;
    this->num_threads = 1;
    this->max_instruction_thread = 100;
    this->alloc_list = new std::vector<data>();
    this->global_symbol_table = 0;
}

void EMULATED_DEVICE::SetTrace(int level)
{
}

// In ptxp/driver.cpp.
extern TREE * parse(char * source);

MODULE * EMULATED_DEVICE::Parse(char * module_name, char * source)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

    // parse all modules, regardless of module name selected.
    if (cu->Level() > 1 || cu->trace_all_calls)
    {
        std::cout << "====================================================\n";
        std::cout << "PROFILE = " << module_name << std::endl;
        std::cout << "CODE:\n";
        std::cout << source << std::endl;
        std::cout << "====================================================\n\n\n";
    }

    TREE * tree = parse(source);
    if (! tree)
    {
        std::cout << "Error: cannot parse PTX!\n";
        return false;
    }
    if (cu->Level() > 1 || cu->trace_all_calls)
    {
        std::cout << "====================================================\n";
        std::cout << "TREE: " << std::endl;
        this->Print(tree, 0);
        std::cout << "====================================================\n\n\n";
    }
    MODULE * module = new MODULE();
    module->name = this->StringTableEntry(module_name);
    module->tree = tree;

    // Extract entry points, functions, etc. from tree. Note, tree is passed into function because it works recursively.
    Extract_From_Tree(module, tree);
    this->modules.push_back(module);

    // Extract globals, constants, etc.
    // Create symbol table for outer blocks.
    this->global_symbol_table = this->PushSymbolTable(this->global_symbol_table);
    int sc[] = { K_GLOBAL, K_TEX, K_CONST, 0};
    this->SetupVariables(this->global_symbol_table, tree, sc);

    return module;
}

void EMULATED_DEVICE::Extract_From_Tree(MODULE * module, TREE * node)
{
    // Traverse the tree and look for key features like entry, func, variable declarations, etc.
    if (node->GetType() == TREE_ENTRY)
    {
        // First child will be name node.
        ENTRY * entry = new ENTRY();
        TREE * word = node->GetChild(0);
        char * name = word->GetText();
        entry->name = name;
        entry->module = module;
        entry->tree = node;
        std::pair<char*, ENTRY *> i;
        i.first = (char*)name;
        i.second = entry;
        module->entry.insert(i);
    }
    else if (node->GetType() == TREE_FUNC)
    {
        ENTRY * entry = new ENTRY();
        TREE * word = node->GetChild(0);
        char * name = word->GetText();
        entry->name = name;
        entry->module = module;
        entry->tree = node;
        std::pair<char*, ENTRY *> i;
        i.first = (char*)name;
        i.second = entry;
        module->func.insert(i);
    }
    for (int i = 0; i < node->GetChildCount(); ++i)
    {
        TREE * child = node->GetChild(i);
        Extract_From_Tree(module, child);
    }
} 

void EMULATED_DEVICE::SetupParams(SYMBOL_TABLE * symbol_table, TREE * e)
{
    // Create a new symbol table block for the parameters.
    //SYMBOL_TABLE * symbol_table = this->root;
    // For each parameter, create a symbol table entry, bind the arguments.
    // To do this, walk down both the AST and the setup argument list, and
    // associate each entry with the other in a symbol table entry.
    {
        int argc = this->arguments.size();
        TREE * param_list = (TREE *)e->GetChild(1);
        std::vector<arg*>::iterator ia = this->arguments.begin();
        for (int i = 0; i < argc; ++i, ++ia)
        {
            // Get to the parameter in the AST.
            TREE * param = param_list->GetChild(i);
            TREE * name = param->GetChild(0);
            char * n = this->StringTableEntry(name->GetText());
            TREE * type = param->GetChild(1)->GetChild(0);
            char * t = type->GetText();
            // Get to the argument in the set up list.
            arg * a = *ia;
            // Create a symbol table entry.
            SYMBOL * s = new SYMBOL();
            s->emulator = this;
            s->pvalue = (void*)a->argument;
            s->name = n;
            s->size = a->size;
            s->typestring = t;
            s->array = false;
            s->total_size = 0;
            s->type = type->GetType();
            s->storage_class = K_PARAM;
            symbol_table->EnterSymbol(s);
        }
    }
}

size_t EMULATED_DEVICE::Sizeof(int type)
{
    switch (type)
    {
    case K_S8: return sizeof(signed __int8);
    case K_S16: return sizeof(signed __int16);
    case K_S32: return sizeof(signed __int32);
    case K_S64: return sizeof(signed __int64);
    case K_U8: return sizeof(unsigned __int8);
    case K_U16: return sizeof(unsigned __int16);
    case K_U32: return sizeof(unsigned __int32);
    case K_U64: return sizeof(unsigned __int64);
    case K_F16: throw new EMU_ERROR("F16 unimplemented.\n"); return sizeof(float);
    case K_F32: return sizeof(float);
    case K_F64: return sizeof(double);
    case K_B8: return sizeof(unsigned __int8);
    case K_B16: return sizeof(signed __int16);
    case K_B32: return sizeof(signed __int32);
    case K_B64: return sizeof(signed __int64);
    case K_PRED: return sizeof(bool);
    // Texref is a dynamic type, so size is unknown. .tex in ptx is now deprecated in favor of this POS.
    case K_TEXREF: return 0;
    }
    assert(false);
    return 0;
}

int EMULATED_DEVICE::GetSize(TREE * tree_par_register)
{
    TREE * c = (TREE *)tree_par_register->GetChild(0);
    return (int)atoi(c->GetText());
}

SYMBOL_TABLE * EMULATED_DEVICE::PushSymbolTable(SYMBOL_TABLE * parent)
{
    SYMBOL_TABLE * symbol_table = new SYMBOL_TABLE();
    symbol_table->parent_block_symbol_table = parent;
    return symbol_table;
}

void EMULATED_DEVICE::SetupVariables(SYMBOL_TABLE * symbol_table, TREE * code, int * desired_storage_classes)
{
    // Go through the block and create entries in the symbol table for each variable declared.
    for (int i = 0; i < code->GetChildCount(); ++i)
    {
        TREE * var = code->GetChild(i);
        if (var->GetType() == TREE_VAR)
        {
            SetupSingleVar(symbol_table, var, desired_storage_classes, false, 0);
        }
    }
}

void EMULATED_DEVICE::SetupSingleVar(SYMBOL_TABLE * symbol_table, TREE * var, int * desired_storage_classes, bool externed, size_t total_size)
{
    // Got variable declaration.
    // Now extract info out of variable declaration.
    char * name = 0;
    int nreg = 0;
    TREE * ttype = 0;
    char * type = 0;
    int size = 0;
    int storage_class = 0;
    bool wrong_class = true;
    TREE * tarray = 0;
    TREE * tinitializer_values = 0;
    int vec = 0;
    int total = 0;
    for (int j = 0; j < (int)var->GetChildCount(); ++j)
    {
        TREE * c = var->GetChild(j);
        int ct = c->GetType();
        if (ct == TREE_SPACE)
        {
            TREE * chi = c->GetChild(0);
            storage_class = chi->GetType();
            // no need to continue if wrong storage class.
            for (int k = 0; desired_storage_classes[k] != 0; ++k)
            {
                if (storage_class == desired_storage_classes[k])
                {
                    wrong_class = false;
                }
            }
        } else if (ct == TREE_ALIGN)
        {
            // Nothing to do.
        } else if (ct == TREE_TYPE)
        {
            ttype = c->GetChild(0);
            type = ttype->GetText();
            int t = ttype->GetType();
            size = Sizeof(t);
        } else if (ct == T_WORD)
        {
            name = c->GetText();
        } else if (ct == TREE_PAR_REGISTER)
        {
            nreg = GetSize(c);
        } else if (ct == TREE_ARRAY)
        {
            // declare var as an array.
            tarray = c;
            // Using the symbol in ptx is essentially a pointer.
            // But, mov and cvta loads a pointer to the pointer when
            // addressing symbols in memory.
            total = 1;
            for (int a = 0; ; ++a)
            {
                TREE * t = tarray->GetChild(a);
                if (t == 0)
                    break;
                int gt = t->GetType();
                // Look at size information if not external.
                if (externed == false && gt == T_OB)
                {
                    ++a;
                    TREE * n = tarray->GetChild(a);
                    assert(n != 0);
                    if (n->GetType() == T_DEC_LITERAL)
                    {
                        int sz = atoi(n->GetText());
                        total = total * sz;
                    }
                    ++a;
                    TREE * t2 = tarray->GetChild(a);
                    assert(t2 != 0);
                    assert(t2->GetType() == T_CB);
                    ++a;
                }
                else if (externed != 0)
                    ;
                else assert(false);
            }
        } else if (ct == T_EQ)
        {
            tinitializer_values = c;
        } else if (ct == TREE_VECTOR_TYPE)
        {
            tarray = c;
            TREE * c2 = c->GetChild(0);
            ct = c2->GetType();
            vec = ct;
            if (ct == K_V2)
                total = 2;
            else
                total = 4;
        } else assert(false);
    }
    if (wrong_class)
        return;
    // Convert raw info into symbol declarations and sizes.
    if (nreg > 0)
    {
        for (int k = 0; k < nreg; ++k)
        {
            char full_name[1000];
            sprintf(full_name, "%s%d", name, k);
            // Create a symbol table entry.
            SYMBOL * s = new SYMBOL();
            s->emulator = this;
            s->name = this->StringTableEntry(full_name);
            s->size = size;
            s->pvalue = (void*)malloc(size);
            memset(s->pvalue, 0, size);
            s->typestring = this->StringTableEntry(type);
            s->type = ttype->GetType();
            s->storage_class = storage_class;
            s->array = false;
            s->total_size = total_size;
            symbol_table->EnterSymbol(s);
        }
    } else {
        // Create a symbol table entry.
        SYMBOL * s = new SYMBOL();
        s->emulator = this;
        s->name = this->StringTableEntry(name);
        s->size = size;
        // array flag helps in printing, but it works like any other
        // storage.
        s->array = false;
        void * ptr = 0;
        if (tarray != 0)
        {
            s->array = true;
            if (! externed)
            {
                s->total_size = size * total;
                ptr = (void*)malloc(size * total);
                memset(ptr, 0, size * total);
            }
            else
            {
                // Each extern points to the same allocated array.
                s->total_size = total_size;
                ptr = this->extern_memory_buffer;
            }
            s->pvalue = ptr;
        }
        else
        {
            s->total_size = size;
            s->pvalue = (void*)malloc(size);
            memset(s->pvalue, 0, size);
            ptr = s->pvalue;
        }

        // Now work on optional initializer...
        if (tinitializer_values != 0)
        {
            unsigned char * mptr = (unsigned char *)ptr;
            for (int a = 0; ; ++a)
            {
                TREE * t = tinitializer_values->GetChild(a);
                if (t == 0)
                break;
                int gt = t->GetType();
                if (gt == TREE_CONSTANT_EXPR)
                {
                    TREE * n = t->GetChild(0);
                    int type = ttype->GetType();
                    CONSTANT c = Eval(type, n);
                    TYPES::Types * s1 = (TYPES::Types*)mptr;
                    switch (type)
                    {
                        case K_B8:
                            s1->b8 = c.value.b8;
                            break;
                        case K_U8:
                            s1->u8 = c.value.u8;
                            break;
                        case K_S8:
                            s1->s8 = c.value.s8;
                            break;
                        case K_B16:
                            s1->b16 = c.value.b16;
                            break;
                        case K_U16:
                            s1->u16 = c.value.u16;
                            break;
                        case K_S16:
                            s1->s16 = c.value.s16;
                            break;
                        case K_B32:
                            s1->b32 = c.value.b32;
                            break;
                        case K_U32:
                            s1->u32 = c.value.u32;
                            break;
                        case K_S32:
                            s1->s32 = c.value.s32;
                            break;
                        case K_B64:
                            s1->b64 = c.value.b64;
                            break;
                        case K_U64:
                            s1->u64 = c.value.u64;
                            break;
                        case K_S64:
                            s1->s64 = c.value.s64;
                            break;
                        case K_F32:
                            s1->f32 = c.value.f32;
                            break;
                        case K_F64:
                            s1->f64 = c.value.f64;
                            break;
                        default:
                        assert(false);
                    }
                }
                else assert(false);
                mptr += size;
            }
        }
        s->typestring = this->StringTableEntry(type);
        s->type = ttype->GetType();
        s->storage_class = storage_class;
        // Add the entry into the symbol table.
        symbol_table->EnterSymbol(s);
    }
}


void EMULATED_DEVICE::SetupGotos(SYMBOL_TABLE * symbol_table, TREE * code)
{
    // Scan ahead and find all labels.  Enter them into the symbol
    // table.
    for (int i = 0; i < code->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)code->GetChild(i);
        if (child->GetType() == TREE_LABEL)
        {
            TREE * label = child->GetChild(0);
            char * name = label->GetText();
            SYMBOL * s = new SYMBOL();
            s->emulator = this;
            s->name = this->StringTableEntry(name);
            s->typestring = this->StringTableEntry("label");
            s->type = label->GetType();
            s->size = 0;
            s->pvalue = (void*)i;
            s->storage_class = 0;
            s->array = false;
            s->total_size = 0;
            // Add the entry into the symbol table.
            symbol_table->EnterSymbol(s);
        }
    }
}

void EMULATED_DEVICE::SetupExternShared(SYMBOL_TABLE * symbol_table, TREE * code)
{
    // No need to resolve anything if no shared memory to set up.
    if (this->conf.sharedMem == 0)
        return;
    this->extern_memory_buffer = (void*)malloc(conf.sharedMem);
    memset(this->extern_memory_buffer, 0, conf.sharedMem);
    for (TREE * p = code; p != 0; p = p->GetParent())
    {
        // Scan ahead and find all extern nodes.
        // Enter them into the symbol table if they are shared
        // memory.
        for (int i = 0; i < p->GetChildCount(); ++i)
        {
            TREE * child = (TREE *)p->GetChild(i);
            if (child->GetType() == TREE_EXTERN)
            {
                TREE * cc = child->GetChild(0);
                if (cc)
                {
                    int t = child->GetChild(0)->GetType();
                    if (t != TREE_VAR)
                        continue;
                }
                TREE * var = child->GetChild(0);
                int sc[] = { K_SHARED, 0};
                SetupSingleVar(symbol_table, var, sc, true, conf.sharedMem);
            }
        }
    }
}

void EMULATED_DEVICE::ConfigureGrid(dim3 dim)
{
    this->conf.gridDim = dim;
}

void EMULATED_DEVICE::ConfigureBlock(dim3 dim)
{
    this->conf.blockDim = dim;
}

void EMULATED_DEVICE::ConfigureSharedMemory(size_t sharedMem)
{
    conf.sharedMem = sharedMem;
}

void EMULATED_DEVICE::ConfigureStream(cudaStream_t stream)
{
    conf.stream = stream;
}

void EMULATED_DEVICE::ResetArgs()
{
    for (std::vector<arg*>::iterator ia = this->arguments.begin();
        ia != this->arguments.end(); ++ia)
    {
        delete *ia;
    }
    this->arguments.clear();
}


void EMULATED_DEVICE::Execute(ENTRY * entry)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

//    _CrtMemState state_begin;
//    _CrtMemCheckpoint(&state_begin);

    //this->trace_level = 3;



    //// Get function block.
    TREE * code = FindBlock(entry->tree);

    // Create symbol table for outer blocks, specifically shared memory.
    SYMBOL_TABLE * obst = PushSymbolTable(this->global_symbol_table);

    for (TREE * p = code->GetParent()->GetParent(); p != 0; p = p->GetParent())
    {
        int sc[] = { K_SHARED, 0};
        SetupVariables(obst, p, sc);
    }

    // Create symbol table for this block.
    SYMBOL_TABLE * block_symbol_table = PushSymbolTable(obst);
    SetupGotos(block_symbol_table, code);
    SetupParams(block_symbol_table, entry->tree);
    CreateSymbol(block_symbol_table, "%nctaid", "dim3", K_V4, &conf.gridDim, sizeof(conf.gridDim), K_LOCAL);
    CreateSymbol(block_symbol_table, "%ntid", "dim3", K_V4, &conf.blockDim, sizeof(conf.blockDim), K_LOCAL);

    // do_thread_synch is a flag for an optimization.  If false, then the symbol tables for different threads
    // can be shared, thus avoiding extra symbol table creation.  However, sharing the same symbol table for
    // registers is not good for multithreading, so we set this to true for now--until I can figure out a
    // way to get this optimization working again.
    bool do_thread_synch = true;
    // = CodeRequiresThreadSynchronization(code);
    if (cu->Level() > 1)
        std::cout << "Thread synchronization " << (do_thread_synch ? "is" : "is not") << " required.\n";
    for (int bidx = 0; bidx < conf.gridDim.x; ++bidx)
    {
        for (int bidy = 0; bidy < conf.gridDim.y; ++bidy)
        {
            for (int bidz = 0; bidz < conf.gridDim.z; ++bidz)
            {
                ExecuteSingleBlock(block_symbol_table, do_thread_synch, code, bidx, bidy, bidz);
            }
        }
    }
    delete block_symbol_table;
    delete obst;

    this->ResetArgs();
}

bool EMULATED_DEVICE::CodeRequiresThreadSynchronization(TREE * code)
{
    bool result = false;
    for (int i = 0; i < (int)code->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)code->GetChild(i);
        if (child->GetType() == TREE_INST)
        {
            TREE * inst = child;
            TREE * i = inst->GetChild(0);
            int inst_type = i->GetType();
            if (inst_type == TREE_PRED)
            {
                // Predicate preceeds the instruction.
                TREE * pred = i;
                i = (TREE *)inst->GetChild(1);
                inst_type = i->GetType();
            }
            switch (inst_type)
            {
                case KI_BAR:
                case KI_MEMBAR:
                    return true;
                default:
                    break;
            }
        }
    }
    return false;
}

void EMULATED_DEVICE::ExecuteSingleBlock(SYMBOL_TABLE * symbol_table, bool do_thread_synch, TREE * code, int bidx, int bidy, int bidz)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    std::queue<THREAD *> wait_queue;
    std::queue<THREAD *> active_queue;

    // Keep track of symbol table root to restore later.  This is because of the awful
    // use of root on a per-thread basis.
    SYMBOL_TABLE * save = symbol_table;

    // Two ways to do this.  If there is no thread synchronization,
    // then threads can run serially, one after another.  In this case,
    // a symbol table of registers/locals/aligns/params does not have
    // to be created for every thread.  Instead, create this at a more
    // global level.  If the code contains thread synchronization,
    // then create the local symbols for each thread.
    // This test is just for performance enhancement.
    // Create a new symbol table and add the block index variables.
    SYMBOL_TABLE * block_symbol_table = PushSymbolTable(symbol_table);
    dim3 bid(bidx, bidy, bidz);
    CreateSymbol(block_symbol_table, "%ctaid", "dim3", K_V4, &bid, sizeof(bid), K_LOCAL);

    if (do_thread_synch)
    {
        // Add to this symbol table any explicit shared memory
        // variables.
        int sc[] = { K_SHARED, 0 };
        SetupVariables(block_symbol_table, code, sc);
    } else
    {
        int sc[] = { K_SHARED, K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
        SetupVariables(block_symbol_table, code, sc);
    }

    // Add to this symbol table any extern declared shared memory
    // variables.
    SetupExternShared(block_symbol_table, code);

    for (int tidx = 0; tidx < conf.blockDim.x; ++tidx)
    {
        for (int tidy = 0; tidy < conf.blockDim.y; ++tidy)
        {
            for (int tidz = 0; tidz < conf.blockDim.z; ++tidz)
            {
                SYMBOL_TABLE * root = PushSymbolTable(block_symbol_table);
                dim3 tid(tidx, tidy, tidz);
                CreateSymbol(root, "%tid", "dim3", K_V4, &tid, sizeof(tid), K_LOCAL);
                if (do_thread_synch)
                {
                    int sc[] = { K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
                    SetupVariables(root, code, sc);
                }
                THREAD * thread = new THREAD(this, code, 0, root);
                wait_queue.push(thread);
            }
        }
    }

    bool spawn = true;
    int max_threads = this->num_threads;
    int num_waiting_threads = 0;
    while (! wait_queue.empty())
    {
        while (! wait_queue.empty())
        {
            if (active_queue.size() >= max_threads)
                break;
            THREAD * thread = wait_queue.front();
            wait_queue.pop();
            if (! thread->Waiting())
            {
                if (spawn)
                {
                    //MEMDBG * mem = MEMDBG::Singleton();
                    //mem->WrapModules();
                    HANDLE hThread = (HANDLE) _beginthreadex(0, 0, THREAD::WinThreadExecute, (void*)thread, CREATE_SUSPENDED, 0);
                    if (hThread)
                    {
                        thread->SetHandle(hThread);
                        ResumeThread(hThread);
                        active_queue.push(thread);
                    }
                    else printf("error in thread spawn\n");
                }
                else
                {
                    thread->Execute();
                    if (! thread->Finished())
                    {
                        wait_queue.push(thread);
                        num_waiting_threads++;
                    }
                    else
                        delete thread;
                }
            }
            else if (! thread->Finished())
            {
                wait_queue.push(thread);
                num_waiting_threads++;
            }
            else
                delete thread;
        }
        // Wait for all active threads to stop.
        if (! active_queue.empty())
        {
            HANDLE arr[20];
            for (int i = 0; i < active_queue.size(); ++i)
            {
                THREAD * thread = active_queue.front();
                active_queue.pop();
                arr[i] = thread->GetHandle();
                active_queue.push(thread);
            }
            ::WaitForMultipleObjects(
                active_queue.size(),
                arr,
                TRUE,
                INFINITE);
        }
        while (! active_queue.empty())
        {
            THREAD * thread = active_queue.front();
            active_queue.pop();
//            WaitForSingleObject(thread->GetHandle(), INFINITE );
            CloseHandle(thread->GetHandle()); // _endthreadex(0); does not free resources.  Call Closehandle to free.
            //MEMDBG * mem = MEMDBG::Singleton();
            //mem->UnwrapModules();
            //mem->Output();
            thread->SetHandle(0);
            // Check the status of the threads.
            if (! thread->Finished())
            {
                wait_queue.push(thread);
                num_waiting_threads++;
            }
            else
                delete thread;
        }

        // thread synchronization assumes all threads wait.
        if (num_waiting_threads != 0 && num_waiting_threads == wait_queue.size())
        {
            if (cu->Level() > 1)
                std::cout << "All " << num_waiting_threads << " threads synchronized!\n";
            // all threads waiting.  Reset all threads to not wait.
            for (int i = 0; i < num_waiting_threads; ++i)
            {
                THREAD * t = wait_queue.front();
                wait_queue.pop();
                t->Reset();
                wait_queue.push(t);
            }
            num_waiting_threads = 0;
        }
    }
    // Delete block symbol table.
    delete block_symbol_table;
    if (this->extern_memory_buffer)
        delete this->extern_memory_buffer;
    this->extern_memory_buffer = 0;
}

void EMULATED_DEVICE::PrintName(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    std::cout << inst->GetChild(start)->GetText() << "\n";
} 

void EMULATED_DEVICE::Print(TREE * node, int level)
{
    for (int i = 0; i < level; ++i)
        std::cout << "   ";
    std::cout << node->GetText() << "\n";
    for (int i = 0; i < (int)node->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)node->GetChild(i);
        Print(child, level+1);
    }
} 


void EMULATED_DEVICE::CreateSymbol(SYMBOL_TABLE * symbol_table, char * name, char * typestring, int type, void * value, size_t size, int storage_class)
{
    // First find it.
    SYMBOL * s = symbol_table->FindSymbol(name);
    if (s)
    {
        assert(false);
        return;
    }
    // Create a symbol table entry.
    s = new SYMBOL();
    s->emulator = this;
    s->name = this->StringTableEntry(name);
    s->typestring = this->StringTableEntry(typestring);
    s->type = type;
    s->size = size;
    s->pvalue = (void*)malloc(size);
    s->storage_class = storage_class;
    s->array = false;
    s->total_size = 0;
    memcpy(s->pvalue, value, size);
    // Add the entry into the symbol table.
    symbol_table->EnterSymbol(s);
}

TREE * EMULATED_DEVICE::FindBlock(TREE * node)
{
    for (int i = 0; i < (int)node->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)node->GetChild(i);
        if (child->GetType() == TREE_BLOCK)
            return child;
    }
    return 0;
}

int EMULATED_DEVICE::FindFirstInst(TREE * block, int first)
{
    for (int i = first; i < (int)block->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)block->GetChild(i);
        if (child->GetType() == TREE_INST || child->GetType() == TREE_BLOCK)
            return i;
    }
    return -1;
}

TREE * EMULATED_DEVICE::GetInst(TREE * block, int pc)
{
    assert(block->GetType() == TREE_BLOCK);
    TREE * inst = (TREE *)block->GetChild(pc);
    return inst;
}


char * EMULATED_DEVICE::StringTableEntry(char * text)
{
    return this->string_table->Entry(text);
}

CONSTANT EMULATED_DEVICE::Eval(int expected_type, TREE * const_expr)
{
    // Perform bottom-up evaluation of a constant expression.
    CONSTANT result;
    result = result.Eval(expected_type, const_expr);
    return result;
}

void EMULATED_DEVICE::unimplemented(bool condition, char * text)
{
    if (condition)
    {
        throw new EMU_ERROR(text);
    }
}

void EMULATED_DEVICE::unimplemented(char * text)
{
    throw new EMU_ERROR(text);
}


void EMULATED_DEVICE::RunDevice(char * device)
{
    this->device = this->string_table->Entry(device);
}

void EMULATED_DEVICE::SetEmulationThreads(int i)
{
    this->num_threads = i;
}


EMULATED_DEVICE::return_type EMULATED_DEVICE::CheckSinglePtrOverwrite(const data * d)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CALL_STACK_INFO * csi = CALL_STACK_INFO::Singleton();
    if (! d->is_host)
    {
        unsigned char * hostbuffer = (unsigned char *)malloc(d->size);
        if (! hostbuffer)
            return NOT_OK;

        cudaError_t e1;
        memcpy(hostbuffer, d->ptr, d->size);
        e1 = cudaSuccess;
        if (e1 != 0)
        {
            free(hostbuffer);
            return NOT_OK;
        }
        bool other = true;
        for (unsigned char * c = (unsigned char *)hostbuffer; c < (((unsigned char *)hostbuffer) + cu->padding_size); ++c)
        {
            if (*c != cu->padding_byte)
            {
                (*cu->output_stream) << "Overwrite of cuda memory block header.\n";
                (*cu->output_stream) << " Pointer " << d->ptr << " was allocated in "
                    << d->context << ".\n";
                (*cu->output_stream) << " This check was performed during a CUDA call in "
                        << cu->Context() << ".\n\n";
                other = false;
                break;
            }
        }
        if (other)
            for (unsigned char * c = ((unsigned char *)hostbuffer) + d->size - cu->padding_size; c < (((unsigned char *)hostbuffer) + d->size); ++c)
            {
                if (*c != cu->padding_byte)
                {
                    (*cu->output_stream) << "Overwrite of cuda memory block footer.\n";
                    (*cu->output_stream) << " Pointer " << d->ptr << " was allocated in file "
                        << d->context << ".\n";
                    (*cu->output_stream) << " This check was performed during a CUDA call in "
                        << cu->Context() << ".\n\n";
                    other = false;
                    break;
                }
            }

        free(hostbuffer);
    }
    else
    {
        bool other = true;
        for (unsigned char * c = (unsigned char *)d->ptr; c < (((unsigned char *)d->ptr) + cu->padding_size); ++c)
        {
            if (*c != cu->padding_byte)
            {
                (*cu->output_stream) << "Memory overwrite for cuda memory block header.\n";
                (*cu->output_stream) << " Pointer " << d->ptr << " was allocated in file "
                    << d->context << ".\n";
                (*cu->output_stream) << " This check was performed during a CUDA call in "
                        << cu->Context() << ".\n\n";
                other = false;
                break;
            }
        }
        if (other)
            for (unsigned char * c = ((unsigned char *)d->ptr) + d->size - cu->padding_size; c < (((unsigned char *)d->ptr) + d->size); ++c)
            {
                if (*c != cu->padding_byte)
                {
                    (*cu->output_stream) << "Overwrite of cuda memory block footer.\n";
                    (*cu->output_stream) << " Pointer " << d->ptr << " was allocated in file "
                        << d->context << ".\n";
                    (*cu->output_stream) << " This check was performed during a CUDA call in "
                        << cu->Context() << ".\n\n";
                    other = false;
                    break;
                }
            }
    }
    return OK;      
}

EMULATED_DEVICE::return_type EMULATED_DEVICE::CheckOverwrite()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    // Check if there are overwrites.
    for (unsigned int i = 0; i < (*this->alloc_list).size(); ++i)
    {
        data d = (*this->alloc_list)[i];
        this->CheckSinglePtrOverwrite(&d);
    }
    return OK;      
}



void EMULATED_DEVICE::ExitHandler()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    // Check if there are no unfreed blocks.
    for (unsigned int i = 0; i < (*this->alloc_list).size(); ++i)
    {
        data d = (*this->alloc_list)[i];
        (*cu->output_stream) << "Unfreed CUDA memory block.\n";
        (*cu->output_stream) << " Pointer " << d.ptr << " was allocated in "
            << d.context << ".\n";
        (*cu->output_stream) << " Block size is " << d.size << " bytes.\n";
        (*cu->output_stream) << " This check was called during program exit, "
            << cu->Context() << " (exit handler).\n\n";
        this->CheckSinglePtrOverwrite(&d);
    }
}

int EMULATED_DEVICE::FindAllocatedBlock(const void * pointer)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    unsigned int i;
    for (i = 0; i < (*this->alloc_list).size(); ++i)
    {
        data * d = &(*this->alloc_list)[i];
        if (pointer >= ((unsigned char *)d->ptr) + cu->padding_size
            && pointer < (d->size - cu->padding_size + (unsigned char *)d->ptr))
            break;
    }
    if (i == (*this->alloc_list).size())
    {
        return -1;
    }
    return i;
}

int EMULATED_DEVICE::FindAllocatedArray(const void * pointer)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    unsigned int i;
    for (i = 0; i < this->arrays.size(); ++i)
    {
        ARRAY * d = this->arrays[i];
        if (pointer == (void*) d)
            break;
    }
    if (i == this->arrays.size())
        return -1;
    else
        return i;
}

bool EMULATED_DEVICE::IsBadPointer(const void * ptr)
{
    bool bad = false;

#if WIN32
    __try
    {
        // read
        unsigned char value = *(unsigned char*)ptr;
    }
    __except(1)
    {
        bad = true;
    }
#endif
    return bad;
}



int EMULATED_DEVICE::TraceLevel()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->Level();
}
