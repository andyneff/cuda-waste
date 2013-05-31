#include "stdafx.h"
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

   Code based on source code by
   Erik Rydgren, erik@rydgrens.net, in tracealloc.cpp
   http://www.codeproject.com/KB/debug/Memory_leak_finder.aspx?msg=3408698

   (Erik's code was based on code       
    for ExtendedTrace written by
    Zoltan Csizmadia, zoltan_csizmadia@yahoo.com.)

*/

// TODO = get stack information from DIA SDK on Windows.

#include "targetver.h"
#include <iostream>
#include <vector>
#include <assert.h>
#include "cuda-wrapper.h"
#include "cuda-memory-debug-undefs.h"
//#include <atlcomcli.h> // For CComPtr
//#include <dia2.h>
#include "call-stack-info.h"
#include "hook-mgr.h"
#include <__cudaFatFormat.h>
#include "../devices/emulated-device.h"
#include "_cuda.h"
#include "_cuda_runtime.h"


#include "../devices/emulated-device.h"


extern char * file_name_tail(char * file_name);

_CUDA_RUNTIME::_CUDA_RUNTIME()
{
    this->did_wrap = false;
}

void _CUDA_RUNTIME::WrapModule(char * cuda_module_name)
{
    //std::cout <<  "WRAPPING " << cuda_module_name << "\n";
    // Add Driver API hooking.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    HOOK_MANAGER * hook_manager = cu->hook_manager;
    bool complain = false;
    if (hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)_CUDA_RUNTIME::_cudaMalloc, complain))
    {
        if (this->did_wrap)
            return;
        this->did_wrap = true;
        this->hModule = hook_manager->GetModule(cuda_module_name);
        bool complain = false;
        // Hook cudart library.
        // Any unknown after this should be flagged.
		hook_manager->HookImport(cuda_module_name, "cudaBindSurfaceToArray", (PROC)_CUDA_RUNTIME::_cudaBindSurfaceToArray, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTexture", (PROC)_CUDA_RUNTIME::_cudaBindTexture, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTexture2D", (PROC)_CUDA_RUNTIME::_cudaBindTexture2D, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTextureToArray", (PROC)_CUDA_RUNTIME::_cudaBindTextureToArray, complain);


        hook_manager->HookImport(cuda_module_name, "__cudaRegisterFatBinary", (PROC)_CUDA_RUNTIME::_cudaRegisterFatBinary, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterFunction", (PROC)_CUDA_RUNTIME::_cudaRegisterFunction, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterSurface", (PROC)_CUDA_RUNTIME::_cudaRegisterSurface, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterTexture", (PROC)_CUDA_RUNTIME::_cudaRegisterTexture, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterVar", (PROC)_CUDA_RUNTIME::_cudaRegisterVar, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaUnregisterFatBinary", (PROC)_CUDA_RUNTIME::_cudaUnregisterFatBinary, complain);


        hook_manager->HookImport(cuda_module_name, "cudaChooseDevice", (PROC)_CUDA_RUNTIME::_cudaChooseDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)_CUDA_RUNTIME::_cudaConfigureCall, complain);
        hook_manager->HookImport(cuda_module_name, "cudaCreateChannelDesc", (PROC)_CUDA_RUNTIME::_cudaCreateChannelDesc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaDriverGetVersion", (PROC)_CUDA_RUNTIME::_cudaDriverGetVersion, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventCreate", (PROC)_CUDA_RUNTIME::_cudaEventCreate, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventCreateWithFlags", (PROC)_CUDA_RUNTIME::_cudaEventCreateWithFlags, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventDestroy", (PROC)_CUDA_RUNTIME::_cudaEventDestroy, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventElapsedTime", (PROC)_CUDA_RUNTIME::_cudaEventElapsedTime, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventQuery", (PROC)_CUDA_RUNTIME::_cudaEventQuery, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventRecord", (PROC)_CUDA_RUNTIME::_cudaEventRecord, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventSynchronize", (PROC)_CUDA_RUNTIME::_cudaEventSynchronize, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)_CUDA_RUNTIME::_cudaFree, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFreeArray", (PROC)_CUDA_RUNTIME::_cudaFreeArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)_CUDA_RUNTIME::_cudaFreeHost, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFuncGetAttributes", (PROC)_CUDA_RUNTIME::_cudaFuncGetAttributes, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFuncSetCacheConfig", (PROC)_CUDA_RUNTIME::_cudaFuncSetCacheConfig, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetChannelDesc", (PROC)_CUDA_RUNTIME::_cudaGetChannelDesc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDevice", (PROC)_CUDA_RUNTIME::_cudaGetDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceCount", (PROC)_CUDA_RUNTIME::_cudaGetDeviceCount, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceProperties", (PROC)_CUDA_RUNTIME::_cudaGetDeviceProperties, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetErrorString", (PROC)_CUDA_RUNTIME::_cudaGetErrorString, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetExportTable", (PROC)_CUDA_RUNTIME::_cudaGetExportTable, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)_CUDA_RUNTIME::_cudaGetLastError, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceReference", (PROC)_CUDA_RUNTIME::_cudaGetSurfaceReference, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolAddress", (PROC)_CUDA_RUNTIME::_cudaGetSymbolAddress, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolSize", (PROC)_CUDA_RUNTIME::_cudaGetSymbolSize, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetTextureAlignmentOffset", (PROC)_CUDA_RUNTIME::_cudaGetTextureAlignmentOffset, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetTextureReference", (PROC)_CUDA_RUNTIME::_cudaGetTextureReference, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsMapResources", (PROC)_CUDA_RUNTIME::_cudaGraphicsMapResources, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceGetMappedPointer", (PROC)_CUDA_RUNTIME::_cudaGraphicsResourceGetMappedPointer, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceSetMapFlags", (PROC)_CUDA_RUNTIME::_cudaGraphicsResourceSetMapFlags, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsSubResourceGetMappedArray", (PROC)_CUDA_RUNTIME::_cudaGraphicsSubResourceGetMappedArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnmapResources", (PROC)_CUDA_RUNTIME::_cudaGraphicsUnmapResources, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnregisterResource", (PROC)_CUDA_RUNTIME::_cudaGraphicsUnregisterResource, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)_CUDA_RUNTIME::_cudaHostAlloc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)_CUDA_RUNTIME::_cudaHostGetDevicePointer, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetFlags", (PROC)_CUDA_RUNTIME::_cudaHostGetFlags, complain);
        hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)_CUDA_RUNTIME::_cudaLaunch, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)_CUDA_RUNTIME::_cudaMalloc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMalloc3DArray", (PROC)_CUDA_RUNTIME::_cudaMalloc3DArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocArray", (PROC)_CUDA_RUNTIME::_cudaMallocArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocHost", (PROC)_CUDA_RUNTIME::_cudaMallocHost, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocPitch", (PROC)_CUDA_RUNTIME::_cudaMallocPitch, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)_CUDA_RUNTIME::_cudaMemcpy, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2D", (PROC)_CUDA_RUNTIME::_cudaMemcpy2D, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DArrayToArray", (PROC)_CUDA_RUNTIME::_cudaMemcpy2DArrayToArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpy2DAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArray", (PROC)_CUDA_RUNTIME::_cudaMemcpy2DFromArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArrayAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpy2DFromArrayAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArray", (PROC)_CUDA_RUNTIME::_cudaMemcpy2DToArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArrayAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpy2DToArrayAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy3D", (PROC)_CUDA_RUNTIME::_cudaMemcpy3D, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpy3DAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyArrayToArray", (PROC)_CUDA_RUNTIME::_cudaMemcpyArrayToArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArray", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArrayAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromArrayAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbol", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromSymbol, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbolAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromSymbolAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArray", (PROC)_CUDA_RUNTIME::_cudaMemcpyToArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArrayAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyToArrayAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbol", (PROC)_CUDA_RUNTIME::_cudaMemcpyToSymbol, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbolAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyToSymbolAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemGetInfo", (PROC)_CUDA_RUNTIME::_cudaMemGetInfo, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)_CUDA_RUNTIME::_cudaMemset, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset2D", (PROC)_CUDA_RUNTIME::_cudaMemset2D, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset3D", (PROC)_CUDA_RUNTIME::_cudaMemset3D, complain);
        hook_manager->HookImport(cuda_module_name, "cudaPeekAtLastError", (PROC)_CUDA_RUNTIME::_cudaPeekAtLastError, complain);
        hook_manager->HookImport(cuda_module_name, "cudaRuntimeGetVersion", (PROC)_CUDA_RUNTIME::_cudaRuntimeGetVersion, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDevice", (PROC)_CUDA_RUNTIME::_cudaSetDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDeviceFlags", (PROC)_CUDA_RUNTIME::_cudaSetDeviceFlags, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForDevice", (PROC)_CUDA_RUNTIME::_cudaSetDoubleForDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForHost", (PROC)_CUDA_RUNTIME::_cudaSetDoubleForHost, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)_CUDA_RUNTIME::_cudaSetupArgument, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetValidDevices", (PROC)_CUDA_RUNTIME::_cudaSetValidDevices, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamCreate", (PROC)_CUDA_RUNTIME::_cudaStreamCreate, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamDestroy", (PROC)_CUDA_RUNTIME::_cudaStreamDestroy, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamQuery", (PROC)_CUDA_RUNTIME::_cudaStreamQuery, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamSynchronize", (PROC)_CUDA_RUNTIME::_cudaStreamSynchronize, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)_CUDA_RUNTIME::_cudaThreadExit, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadGetLimit", (PROC)_CUDA_RUNTIME::_cudaThreadGetLimit, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSetLimit", (PROC)_CUDA_RUNTIME::_cudaThreadSetLimit, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSynchronize", (PROC)_CUDA_RUNTIME::_cudaThreadSynchronize, complain);
        hook_manager->HookImport(cuda_module_name, "cudaUnbindTexture", (PROC)_CUDA_RUNTIME::_cudaUnbindTexture, complain);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaBindSurfaceToArray(const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaBindSurfaceToArray(surfref, array, desc);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaBindTexture(offset, texref, devPtr, desc, size);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaBindTextureToArray(texref, array, desc);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaChooseDevice(device, prop);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

struct cudaChannelFormatDesc CUDARTAPI _CUDA_RUNTIME::_cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaCreateChannelDesc(x, y, z, w, f);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDriverGetVersion(int *driverVersion)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDriverGetVersion(driverVersion);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventCreate(cudaEvent_t *event)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventCreate(event);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventCreateWithFlags(cudaEvent_t *event, int flags)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventCreateWithFlags(event, flags);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventDestroy(cudaEvent_t event)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventDestroy(event);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventElapsedTime(ms, start, end);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventQuery(cudaEvent_t event)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventQuery(event);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventRecord(event, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaEventSynchronize(cudaEvent_t event)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaEventSynchronize(event);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFree(void * ptr)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFree(ptr);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFreeArray(struct cudaArray *array)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFreeArray(array);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFreeHost(void * ptr)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFreeHost(ptr);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFuncGetAttributes(attr, func);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFuncSetCacheConfig(func, cacheConfig);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetChannelDesc(desc, array);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetDevice(int *device)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetDevice(device);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetDeviceCount(int *count)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetDeviceCount(count);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetDeviceProperties(prop, device);
}

const char* CUDARTAPI _CUDA_RUNTIME::_cudaGetErrorString(cudaError_t error)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetErrorString(error);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetExportTable(ppExportTable, pExportTableId);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetLastError(void)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetLastError();
}

cudaError_t CUDARTAPI _cudaGetSurfaceAlignmentOffset(size_t *offset, const struct surfaceReference *surfref)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetSurfaceAlignmentOffset(offset, surfref);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetSurfaceReference(const struct surfaceReference **surfref, const char *symbol)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetSurfaceReference(surfref, symbol);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetSymbolAddress(void **devPtr, const char *symbol)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetSymbolAddress(devPtr, symbol);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetSymbolSize(size_t *size, const char *symbol)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetSymbolSize(size, symbol);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetTextureAlignmentOffset(offset, texref);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetTextureReference(const struct textureReference **texref, const char *symbol)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetTextureReference(texref, symbol);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsMapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsMapResources(count, resources, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, struct cudaGraphicsResource *resource)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsResourceSetMapFlags(struct cudaGraphicsResource *resource, unsigned int flags)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsResourceSetMapFlags(resource, flags);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsSubResourceGetMappedArray(struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsSubResourceGetMappedArray(arrayPtr, resource, arrayIndex, mipLevel);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsUnmapResources(int count, struct cudaGraphicsResource **resources, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsUnmapResources(count, resources, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsUnregisterResource(struct cudaGraphicsResource *resource)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsUnregisterResource(resource);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaHostAlloc(void ** ptr, size_t size, unsigned int flags)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaHostAlloc(ptr, size, flags);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaHostGetFlags(pFlags, pHost);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaLaunch(const char *entry)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaLaunch(entry);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMalloc(void ** ptr, size_t size)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMalloc(ptr, size);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMalloc3DArray(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMalloc3DArray(pitchedDevPtr, extent);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMallocArray(array, desc, width, height, flags);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMallocHost(void **ptr, size_t size)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMallocHost(ptr, size);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy(dst, src, count, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy3D(const struct cudaMemcpy3DParms *p)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy3D(p);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy3DAsync(p, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemGetInfo(size_t *free, size_t *total)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemGetInfo(free, total);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset(void * devPtr, int value, size_t count)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset(devPtr, value, count);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset2D(devPtr, pitch, value, width, height);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaPeekAtLastError(void)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaPeekAtLastError();
}

void** CUDARTAPI _CUDA_RUNTIME::_cudaRegisterFatBinary(void *fatCubin)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterFatBinary(fatCubin);
}

void CUDARTAPI _CUDA_RUNTIME::_cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	cu->CurrentDevice()->_cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void CUDARTAPI _CUDA_RUNTIME::_cudaRegisterSurface(void **fatCubinHandle, const struct surfaceReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterSurface(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
}

void CUDARTAPI _CUDA_RUNTIME::_cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterTexture(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
}

void CUDARTAPI _CUDA_RUNTIME::_cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaRuntimeGetVersion(int *runtimeVersion)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRuntimeGetVersion(runtimeVersion);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetDevice(int device)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaSetDevice(device);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetDeviceFlags(unsigned int flags)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaSetDeviceFlags(flags);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetDoubleForDevice(double *d)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaSetDoubleForDevice(d);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetDoubleForHost(double *d)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaSetDoubleForHost(d);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaSetupArgument(arg, size, offset);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetValidDevices(int *device_arr, int len)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaSetValidDevices(device_arr, len);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamCreate(cudaStream_t *pStream)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamCreate(pStream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamDestroy(cudaStream_t stream)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamDestroy(stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamQuery(cudaStream_t stream)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamQuery(stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamSynchronize(cudaStream_t stream)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamSynchronize(stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadExit()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadExit();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadGetLimit(enum cudaLimit limit, size_t value)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadGetLimit(limit, value);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadSetLimit(limit, value);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadSynchronize(void)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadSynchronize();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaUnbindTexture(const struct textureReference *texref)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaUnbindTexture(texref);
}

void CUDARTAPI _CUDA_RUNTIME::_cudaUnregisterFatBinary(void **fatCubinHandle)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaUnregisterFatBinary(fatCubinHandle);
}








void _CUDA_RUNTIME::Unimplemented()
{
    std::cout << "FATAL ERROR: Unimplemented function!!!!\n";
    // get call back and get hooked name to know what function is
    // unimplemented.

    // Figure out the name of the function that is bound to CUDA_WRAPPER::Unimplemented.
    // Not especially portable, but it is general.
    // 1) Go up call stack to caller of this function.  Call stack includes
    // CallTree, Unimplemented, then the caller.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CALL_STACK_INFO * csi = CALL_STACK_INFO::Singleton();
    std::list<void*> * call_stack = csi->CallTree();
    std::list<void*>::iterator it = call_stack->begin();
    if (it != call_stack->end())
        ++it;
    if (it != call_stack->end())
        ++it;
    if (it != call_stack->end())
        ++it;
    // 2) The instruction at *it will be the instruction just after the call.
    // Assuming that the call function is relative, step back 4 bytes,
    // get the value to add to *it.  This is a jmp instruction.  Step forward
    // 2 bytes to get the address of the iat, which contains another jump
    // instruction.
    unsigned char * p = 0;
    if (it != call_stack->end())
    {
        p = (unsigned char *)*it;
        // step back f bytes and get opcode.
        unsigned char * pm5 = p - 5;
        void * iat = 0;
        if (*pm5 == 0xe8)
        {
            // Relative call...
            int del = *(int*)(p-4);
            // Get call to.
            unsigned char * jmp = p + del;
            if (*jmp == 0xe9)
            {
                // Relative jmp...
                void ** piat = 0;
                unsigned char * jmp2 = (unsigned char *)(jmp + *(int*)(jmp + 1) + 5);
                if (*jmp2 == 0xff && *(jmp2+1) == 0x25)
                {
                    // Abs jmp.
                    iat = (void*)( *(int*)(jmp2 + 2));
                }
            }
            else if (*jmp == 0xff && *(jmp+1) == 0x25)
            {
                // Abs jmp.
                iat = (void*)( *(int*)(jmp + 2));
            }
            if (iat != 0)
            {
                HOOK_MANAGER * hm = cu->hook_manager;
                HookedFunction * hf = hm->FindHook(iat);
                if (hf)
                {
                    std::cout << "Function " << hf->Get_FuncName() << " needs to be implemented to debug this program.\n";
                }
            }
        }
    }
    delete call_stack;
    exit(1);
}
