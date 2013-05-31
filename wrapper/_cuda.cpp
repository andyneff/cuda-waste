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


void _CUDA::WrapModule()
{
    // Add Driver API hooking.
    if (did_wrap)
        return;
    did_wrap = true;
    char * cuda_module_name = "nvcuda.dll";
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    HOOK_MANAGER * hook_manager = cu->hook_manager;
    bool complain = false;
    PROC proc = hook_manager->HookImport(cuda_module_name, "cuArray3DCreate", (PROC)_CUDA::_cuArray3DCreate, complain);
    if (proc)
    {
        this->hModule = hook_manager->GetModule(cuda_module_name);
    }
    hook_manager->HookImport(cuda_module_name, "cuArray3DCreate_v2", (PROC)_CUDA::_cuArray3DCreate_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuArray3DGetDescriptor", (PROC)_CUDA::_cuArray3DGetDescriptor, complain);
    hook_manager->HookImport(cuda_module_name, "cuArray3DGetDescriptor_v2", (PROC)_CUDA::_cuArray3DGetDescriptor_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuArrayCreate", (PROC)_CUDA::_cuArrayCreate, complain);
    hook_manager->HookImport(cuda_module_name, "cuArrayCreate_v2", (PROC)_CUDA::_cuArrayCreate_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuArrayDestroy", (PROC)_CUDA::_cuArrayDestroy, complain);
    hook_manager->HookImport(cuda_module_name, "cuArrayGetDescriptor", (PROC)_CUDA::_cuArrayGetDescriptor, complain);
    hook_manager->HookImport(cuda_module_name, "cuArrayGetDescriptor_v2", (PROC)_CUDA::_cuArrayGetDescriptor_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxAttach", (PROC)_CUDA::_cuCtxAttach, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxCreate", (PROC)_CUDA::_cuCtxCreate, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxCreate_v2", (PROC)_CUDA::_cuCtxCreate, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxDestroy", (PROC)_CUDA::_cuCtxDestroy, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxDetach", (PROC)_CUDA::_cuCtxDetach, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxGetDevice", (PROC)_CUDA::_cuCtxGetDevice, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxGetLimit", (PROC)_CUDA::_cuCtxGetLimit, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxPopCurrent", (PROC)_CUDA::_cuCtxPopCurrent, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxPushCurrent", (PROC)_CUDA::_cuCtxPushCurrent, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxSetLimit", (PROC)_CUDA::_cuCtxSetLimit, complain);
    hook_manager->HookImport(cuda_module_name, "cuCtxSynchronize", (PROC)_CUDA::_cuCtxSynchronize, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceComputeCapability", (PROC)_CUDA::_cuDeviceComputeCapability, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGet", (PROC)_CUDA::_cuDeviceGet, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetAttribute", (PROC)_CUDA::_cuDeviceGetAttribute, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetCount", (PROC)_CUDA::_cuDeviceGetCount, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetName", (PROC)_CUDA::_cuDeviceGetName, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetProperties", (PROC)_CUDA::_cuDeviceGetProperties, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceTotalMem", (PROC)_CUDA::_cuDeviceTotalMem, complain);
    hook_manager->HookImport(cuda_module_name, "cuDeviceTotalMem_v2", (PROC)_CUDA::_cuDeviceTotalMem_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuDriverGetVersion", (PROC)_CUDA::_cuDriverGetVersion, complain);
    hook_manager->HookImport(cuda_module_name, "cuEventCreate", (PROC)_CUDA::_cuEventCreate, complain);
    hook_manager->HookImport(cuda_module_name, "cuEventDestroy", (PROC)_CUDA::_cuEventDestroy, complain);
    hook_manager->HookImport(cuda_module_name, "cuEventElapsedTime", (PROC)_CUDA::_cuEventElapsedTime, complain);
    hook_manager->HookImport(cuda_module_name, "cuEventQuery", (PROC)_CUDA::_cuEventQuery, complain);
    hook_manager->HookImport(cuda_module_name, "cuEventRecord", (PROC)_CUDA::_cuEventRecord, complain);
    hook_manager->HookImport(cuda_module_name, "cuEventSynchronize", (PROC)_CUDA::_cuEventSynchronize, complain);
    hook_manager->HookImport(cuda_module_name, "cuFuncGetAttribute", (PROC)_CUDA::_cuFuncGetAttribute, complain);
    hook_manager->HookImport(cuda_module_name, "cuFuncSetBlockShape", (PROC)_CUDA::_cuFuncSetBlockShape, complain);
    hook_manager->HookImport(cuda_module_name, "cuFuncSetCacheConfig", (PROC)_CUDA::_cuFuncSetCacheConfig, complain);
    hook_manager->HookImport(cuda_module_name, "cuFuncSetSharedSize", (PROC)_CUDA::_cuFuncSetSharedSize, complain);
    hook_manager->HookImport(cuda_module_name, "cuGetExportTable", (PROC)_CUDA::_cuGetExportTable, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsMapResources", (PROC)_CUDA::_cuGraphicsMapResources, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsResourceGetMappedPointer", (PROC)_CUDA::_cuGraphicsResourceGetMappedPointer, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsResourceGetMappedPointer_v2", (PROC)_CUDA::_cuGraphicsResourceGetMappedPointer_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsResourceSetMapFlags", (PROC)_CUDA::_cuGraphicsResourceSetMapFlags, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsSubResourceGetMappedArray", (PROC)_CUDA::_cuGraphicsSubResourceGetMappedArray, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsUnmapResources", (PROC)_CUDA::_cuGraphicsUnmapResources, complain);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsUnregisterResource", (PROC)_CUDA::_cuGraphicsUnregisterResource, complain);
    hook_manager->HookImport(cuda_module_name, "cuInit", (PROC)_CUDA::_cuInit, complain);
    hook_manager->HookImport(cuda_module_name, "cuLaunch", (PROC)_CUDA::_cuLaunch, complain);
    hook_manager->HookImport(cuda_module_name, "cuLaunchGrid", (PROC)_CUDA::_cuLaunchGrid, complain);
    hook_manager->HookImport(cuda_module_name, "cuLaunchGridAsync", (PROC)_CUDA::_cuLaunchGridAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemAlloc", (PROC)_CUDA::_cuMemAlloc, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemAlloc_v2", (PROC)_CUDA::_cuMemAlloc_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocHost", (PROC)_CUDA::_cuMemAllocHost, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocHost_v2", (PROC)_CUDA::_cuMemAllocHost_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocPitch", (PROC)_CUDA::_cuMemAllocPitch, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocPitch_v2", (PROC)_CUDA::_cuMemAllocPitch_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2D", (PROC)_CUDA::_cuMemcpy2D, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2D_v2", (PROC)_CUDA::_cuMemcpy2D_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DAsync", (PROC)_CUDA::_cuMemcpy2DAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DAsync_v2", (PROC)_CUDA::_cuMemcpy2DAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DUnaligned", (PROC)_CUDA::_cuMemcpy2DUnaligned, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DUnaligned_v2", (PROC)_CUDA::_cuMemcpy2DUnaligned_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3D", (PROC)_CUDA::_cuMemcpy3D, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3D_v2", (PROC)_CUDA::_cuMemcpy3D_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3DAsync", (PROC)_CUDA::_cuMemcpy3DAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3DAsync_v2", (PROC)_CUDA::_cuMemcpy3DAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoA", (PROC)_CUDA::_cuMemcpyAtoA, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoA_v2", (PROC)_CUDA::_cuMemcpyAtoA_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoD", (PROC)_CUDA::_cuMemcpyAtoD, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoD_v2", (PROC)_CUDA::_cuMemcpyAtoD_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoH", (PROC)_CUDA::_cuMemcpyAtoH, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoH_v2", (PROC)_CUDA::_cuMemcpyAtoH_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoHAsync", (PROC)_CUDA::_cuMemcpyAtoHAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoHAsync_v2", (PROC)_CUDA::_cuMemcpyAtoHAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoA", (PROC)_CUDA::_cuMemcpyDtoA, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoA_v2", (PROC)_CUDA::_cuMemcpyDtoA_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoD", (PROC)_CUDA::_cuMemcpyDtoD, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoD_v2", (PROC)_CUDA::_cuMemcpyDtoD_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoDAsync", (PROC)_CUDA::_cuMemcpyDtoDAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoDAsync_v2", (PROC)_CUDA::_cuMemcpyDtoDAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoH", (PROC)_CUDA::_cuMemcpyDtoH, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoH_v2", (PROC)_CUDA::_cuMemcpyDtoH_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoHAsync", (PROC)_CUDA::_cuMemcpyDtoHAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoHAsync_v2", (PROC)_CUDA::_cuMemcpyDtoHAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoA", (PROC)_CUDA::_cuMemcpyHtoA, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoA_v2", (PROC)_CUDA::_cuMemcpyHtoA_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoAAsync", (PROC)_CUDA::_cuMemcpyHtoAAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoAAsync_v2", (PROC)_CUDA::_cuMemcpyHtoAAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoD", (PROC)_CUDA::_cuMemcpyHtoD, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoD_v2", (PROC)_CUDA::_cuMemcpyHtoD_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoDAsync", (PROC)_CUDA::_cuMemcpyHtoDAsync, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoDAsync_v2", (PROC)_CUDA::_cuMemcpyHtoDAsync_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemFree", (PROC)_CUDA::_cuMemFree, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemFree_v2", (PROC)_CUDA::_cuMemFree_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemFreeHost", (PROC)_CUDA::_cuMemFreeHost, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemGetAddressRange", (PROC)_CUDA::_cuMemGetAddressRange, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemGetAddressRange_v2", (PROC)_CUDA::_cuMemGetAddressRange_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemGetInfo", (PROC)_CUDA::_cuMemGetInfo, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemGetInfo_v2", (PROC)_CUDA::_cuMemGetInfo_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemHostAlloc", (PROC)_CUDA::_cuMemHostAlloc, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemHostGetDevicePointer", (PROC)_CUDA::_cuMemHostGetDevicePointer, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemHostGetDevicePointer_v2", (PROC)_CUDA::_cuMemHostGetDevicePointer_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemHostGetFlags", (PROC)_CUDA::_cuMemHostGetFlags, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD16", (PROC)_CUDA::_cuMemsetD16, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD16_v2", (PROC)_CUDA::_cuMemsetD16_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D16", (PROC)_CUDA::_cuMemsetD2D16, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D16_v2", (PROC)_CUDA::_cuMemsetD2D16_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D32", (PROC)_CUDA::_cuMemsetD2D32, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D32_v2", (PROC)_CUDA::_cuMemsetD2D32_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D8", (PROC)_CUDA::_cuMemsetD2D8, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D8_v2", (PROC)_CUDA::_cuMemsetD2D8_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD32", (PROC)_CUDA::_cuMemsetD32, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD32_v2", (PROC)_CUDA::_cuMemsetD32_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD8", (PROC)_CUDA::_cuMemsetD8, complain);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD8_v2", (PROC)_CUDA::_cuMemsetD8_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetFunction", (PROC)_CUDA::_cuModuleGetFunction, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetGlobal", (PROC)_CUDA::_cuModuleGetGlobal, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetGlobal_v2", (PROC)_CUDA::_cuModuleGetGlobal_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetSurfRef", (PROC)_CUDA::_cuModuleGetSurfRef, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetTexRef", (PROC)_CUDA::_cuModuleGetTexRef, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoad", (PROC)_CUDA::_cuModuleLoad, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoadData", (PROC)_CUDA::_cuModuleLoadData, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoadDataEx", (PROC)_CUDA::_cuModuleLoadDataEx, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoadFatBinary", (PROC)_CUDA::_cuModuleLoadFatBinary, complain);
    hook_manager->HookImport(cuda_module_name, "cuModuleUnload", (PROC)_CUDA::_cuModuleUnload, complain);
    hook_manager->HookImport(cuda_module_name, "cuParamSetf", (PROC)_CUDA::_cuParamSetf, complain);
    hook_manager->HookImport(cuda_module_name, "cuParamSeti", (PROC)_CUDA::_cuParamSeti, complain);
    hook_manager->HookImport(cuda_module_name, "cuParamSetSize", (PROC)_CUDA::_cuParamSetSize, complain);
    hook_manager->HookImport(cuda_module_name, "cuParamSetTexRef", (PROC)_CUDA::_cuParamSetTexRef, complain);
    hook_manager->HookImport(cuda_module_name, "cuParamSetv", (PROC)_CUDA::_cuParamSetv, complain);
    hook_manager->HookImport(cuda_module_name, "cuStreamCreate", (PROC)_CUDA::_cuStreamCreate, complain);
    hook_manager->HookImport(cuda_module_name, "cuStreamDestroy", (PROC)_CUDA::_cuStreamDestroy, complain);
    hook_manager->HookImport(cuda_module_name, "cuStreamQuery", (PROC)_CUDA::_cuStreamQuery, complain);
    hook_manager->HookImport(cuda_module_name, "cuStreamSynchronize", (PROC)_CUDA::_cuStreamSynchronize, complain);
    hook_manager->HookImport(cuda_module_name, "cuSurfRefGetArray", (PROC)_CUDA::_cuSurfRefGetArray, complain);
    hook_manager->HookImport(cuda_module_name, "cuSurfRefSetArray", (PROC)_CUDA::_cuSurfRefSetArray, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefCreate", (PROC)_CUDA::_cuTexRefCreate, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefDestroy", (PROC)_CUDA::_cuTexRefDestroy, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetAddress", (PROC)_CUDA::_cuTexRefGetAddress, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetAddress_v2", (PROC)_CUDA::_cuTexRefGetAddress_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetAddressMode", (PROC)_CUDA::_cuTexRefGetAddressMode, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetArray", (PROC)_CUDA::_cuTexRefGetArray, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetFilterMode", (PROC)_CUDA::_cuTexRefGetFilterMode, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetFlags", (PROC)_CUDA::_cuTexRefGetFlags, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetFormat", (PROC)_CUDA::_cuTexRefGetFormat, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress", (PROC)_CUDA::_cuTexRefSetAddress, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress2D", (PROC)_CUDA::_cuTexRefSetAddress2D, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress2D_v2", (PROC)_CUDA::_cuTexRefSetAddress2D_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress_v2", (PROC)_CUDA::_cuTexRefSetAddress_v2, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddressMode", (PROC)_CUDA::_cuTexRefSetAddressMode, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetArray", (PROC)_CUDA::_cuTexRefSetArray, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetFilterMode", (PROC)_CUDA::_cuTexRefSetFilterMode, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetFlags", (PROC)_CUDA::_cuTexRefSetFlags, complain);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetFormat", (PROC)_CUDA::_cuTexRefSetFormat, complain);
}

CUresult CUDAAPI _CUDA::_cuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArray3DCreate(pHandle, pAllocateArray);
}

CUresult CUDAAPI _CUDA::_cuArray3DCreate_v2( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArray3DCreate_v2(pHandle, pAllocateArray );
}

CUresult CUDAAPI _CUDA::_cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArray3DGetDescriptor( pArrayDescriptor, hArray );
}

CUresult CUDAAPI _CUDA::_cuArray3DGetDescriptor_v2( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArray3DGetDescriptor_v2( pArrayDescriptor, hArray );
}

CUresult CUDAAPI _CUDA::_cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArrayCreate( pHandle, pAllocateArray );
}

CUresult CUDAAPI _CUDA::_cuArrayCreate_v2( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArrayCreate_v2( pHandle, pAllocateArray );
}

CUresult CUDAAPI _CUDA::_cuArrayDestroy( CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArrayDestroy( hArray );
}

CUresult CUDAAPI _CUDA::_cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArrayGetDescriptor( pArrayDescriptor, hArray );
}

CUresult CUDAAPI _CUDA::_cuArrayGetDescriptor_v2( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuArrayGetDescriptor_v2( pArrayDescriptor, hArray );
}

CUresult CUDAAPI _CUDA::_cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxAttach(pctx, flags);
}

CUresult CUDAAPI _CUDA::_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxCreate(pctx, flags, dev );
}

CUresult CUDAAPI _CUDA::_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxCreate_v2(pctx, flags, dev );
}

CUresult CUDAAPI _CUDA::_cuCtxDestroy( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxDestroy( ctx );
}

CUresult CUDAAPI _CUDA::_cuCtxDetach(CUcontext ctx)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxDetach( ctx);
}

CUresult CUDAAPI _CUDA::_cuCtxGetDevice(CUdevice *device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxGetDevice(device);
}

CUresult CUDAAPI _CUDA::_cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxGetLimit(pvalue, limit);
}

CUresult CUDAAPI _CUDA::_cuCtxPopCurrent( CUcontext *pctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxPopCurrent( pctx );
}

CUresult CUDAAPI _CUDA::_cuCtxPushCurrent( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxPushCurrent( ctx );
}

CUresult CUDAAPI _CUDA::_cuCtxSetLimit(CUlimit limit, size_t value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxSetLimit( limit, value);
}

CUresult CUDAAPI _CUDA::_cuCtxSynchronize(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuCtxSynchronize();
}

CUresult CUDAAPI _CUDA::_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceComputeCapability(major, minor, dev);
}

CUresult CUDAAPI _CUDA::_cuDeviceGet(CUdevice *device, int ordinal)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceGet(device, ordinal);
}

CUresult CUDAAPI _CUDA::_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceGetAttribute(pi, attrib, dev);
}

CUresult CUDAAPI _CUDA::_cuDeviceGetCount(int *count)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceGetCount(count);
}

CUresult CUDAAPI _CUDA::_cuDeviceGetName(char *name, int len, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceGetName(name, len, dev);
}

CUresult CUDAAPI _CUDA::_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceGetProperties(prop, dev);
}

CUresult CUDAAPI _CUDA::_cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceTotalMem(bytes, dev);
}

CUresult CUDAAPI _CUDA::_cuDeviceTotalMem_v2(unsigned int *bytes, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDeviceTotalMem_v2(bytes, dev);
}

CUresult CUDAAPI _CUDA::_cuDriverGetVersion(int *driverVersion)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuDriverGetVersion(driverVersion);
}

CUresult CUDAAPI _CUDA::_cuEventCreate( CUevent *phEvent, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuEventCreate( phEvent, Flags );
}

CUresult CUDAAPI _CUDA::_cuEventDestroy( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuEventDestroy( hEvent );
}

CUresult CUDAAPI _CUDA::_cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuEventElapsedTime( pMilliseconds, hStart, hEnd );
}

CUresult CUDAAPI _CUDA::_cuEventQuery( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuEventQuery( hEvent );
}

CUresult CUDAAPI _CUDA::_cuEventRecord( CUevent hEvent, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuEventRecord( hEvent, hStream );
}

CUresult CUDAAPI _CUDA::_cuEventSynchronize( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuEventSynchronize( hEvent );
}

CUresult CUDAAPI _CUDA::_cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuFuncGetAttribute (pi, attrib, hfunc);
}

CUresult CUDAAPI _CUDA::_cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuFuncSetBlockShape (hfunc, x, y, z);
}

CUresult CUDAAPI _CUDA::_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuFuncSetCacheConfig(hfunc, config);
}

CUresult CUDAAPI _CUDA::_cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuFuncSetSharedSize (hfunc, bytes);
}

CUresult CUDAAPI _CUDA::_cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGetExportTable( ppExportTable, pExportTableId );
}

CUresult CUDAAPI _CUDA::_cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsMapResources( count, resources, hStream );
}

CUresult CUDAAPI _CUDA::_cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsResourceGetMappedPointer( pDevPtr, pSize, resource );
}

CUresult CUDAAPI _CUDA::_cuGraphicsResourceGetMappedPointer_v2( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsResourceGetMappedPointer_v2( pDevPtr, pSize, resource );
}

CUresult CUDAAPI _CUDA::_cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsResourceSetMapFlags( resource, flags );
}

CUresult CUDAAPI _CUDA::_cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsSubResourceGetMappedArray( pArray, resource, arrayIndex, mipLevel );
}

CUresult CUDAAPI _CUDA::_cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsUnmapResources( count, resources, hStream );
}

CUresult CUDAAPI _CUDA::_cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuGraphicsUnregisterResource(resource);
}

CUresult CUDAAPI _CUDA::_cuInit(unsigned int Flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuInit( Flags);
}

CUresult CUDAAPI _CUDA::_cuLaunch ( CUfunction f )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuLaunch ( f );
}

CUresult CUDAAPI _CUDA::_cuLaunchGrid (CUfunction f, int grid_width, int grid_height)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuLaunchGrid (f, grid_width, grid_height);
}

CUresult CUDAAPI _CUDA::_cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuLaunchGridAsync( f, grid_width, grid_height, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemAlloc( dptr, bytesize);
}

CUresult CUDAAPI _CUDA::_cuMemAlloc_v2( CUdeviceptr *dptr, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemAlloc_v2( dptr, bytesize);
}

CUresult CUDAAPI _CUDA::_cuMemAllocHost(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemAllocHost(pp, bytesize);
}

CUresult CUDAAPI _CUDA::_cuMemAllocHost_v2(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemAllocHost_v2(pp, bytesize);
}

CUresult CUDAAPI _CUDA::_cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemAllocPitch( dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult CUDAAPI _CUDA::_cuMemAllocPitch_v2( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemAllocPitch_v2( dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult CUDAAPI _CUDA::_cuMemcpy2D( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy2D( pCopy );
}

CUresult CUDAAPI _CUDA::_cuMemcpy2D_v2( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy2D_v2( pCopy );
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy2DAsync( pCopy, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DAsync_v2( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy2DAsync_v2( pCopy, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy2DUnaligned( pCopy );
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DUnaligned_v2( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy2DUnaligned_v2( pCopy );
}

CUresult CUDAAPI _CUDA::_cuMemcpy3D( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy3D( pCopy );
}

CUresult CUDAAPI _CUDA::_cuMemcpy3D_v2( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy3D_v2( pCopy );
}

CUresult CUDAAPI _CUDA::_cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy3DAsync( pCopy, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpy3DAsync_v2( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpy3DAsync_v2( pCopy, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoA( dstArray, dstOffset, srcArray, srcOffset, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoA_v2( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoA_v2( dstArray, dstOffset, srcArray, srcOffset, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoD ( dstDevice, srcArray, srcOffset, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoD_v2 ( dstDevice, srcArray, srcOffset, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoH( dstHost, srcArray, srcOffset, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoH_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoH_v2( dstHost, srcArray, srcOffset, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoHAsync( dstHost, srcArray, srcOffset, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoHAsync_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyAtoHAsync_v2( dstHost, srcArray, srcOffset, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoA ( dstArray, dstOffset, srcDevice, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoA_v2 ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoA_v2 ( dstArray, dstOffset, srcDevice, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoD ( dstDevice, srcDevice, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoD_v2 ( dstDevice,  srcDevice, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoDAsync ( dstDevice, srcDevice, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoDAsync_v2 ( dstDevice, srcDevice, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoH (dstHost, srcDevice, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoH_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoH_v2 (dstHost, srcDevice, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoHAsync (dstHost, srcDevice, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoHAsync_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyDtoHAsync_v2 (dstHost, srcDevice, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoA( dstArray, dstOffset, srcHost, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoA_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoA_v2( dstArray, dstOffset, srcHost, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoAAsync( dstArray, dstOffset, srcHost, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoAAsync_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoAAsync_v2( dstArray, dstOffset, srcHost, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoD ( dstDevice, srcHost, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoD_v2 ( dstDevice, srcHost, ByteCount );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoDAsync ( dstDevice, srcHost, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemcpyHtoDAsync_v2 ( dstDevice, srcHost, ByteCount, hStream );
}

CUresult CUDAAPI _CUDA::_cuMemFree(CUdeviceptr dptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemFree(dptr);
}

CUresult CUDAAPI _CUDA::_cuMemFree_v2(CUdeviceptr dptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemFree_v2(dptr);
}

CUresult CUDAAPI _CUDA::_cuMemFreeHost(void *p)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemFreeHost(p);
}

CUresult CUDAAPI _CUDA::_cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemGetAddressRange( pbase, psize, dptr );
}

CUresult CUDAAPI _CUDA::_cuMemGetAddressRange_v2( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemGetAddressRange_v2( pbase, psize, dptr );
}

CUresult CUDAAPI _CUDA::_cuMemGetInfo(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemGetInfo(free, total);
}

CUresult CUDAAPI _CUDA::_cuMemGetInfo_v2(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemGetInfo_v2(free, total);
}

CUresult CUDAAPI _CUDA::_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemHostAlloc(pp, bytesize, Flags );
}

CUresult CUDAAPI _CUDA::_cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemHostGetDevicePointer( pdptr, p, Flags );
}

CUresult CUDAAPI _CUDA::_cuMemHostGetDevicePointer_v2( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemHostGetDevicePointer_v2( pdptr, p, Flags );
}

CUresult CUDAAPI _CUDA::_cuMemHostGetFlags( unsigned int *pFlags, void *p )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemHostGetFlags( pFlags, p );
}

CUresult CUDAAPI _CUDA::_cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD16( dstDevice, us, N );
}

CUresult CUDAAPI _CUDA::_cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD16_v2( dstDevice, us, N );
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD2D16( dstDevice, dstPitch, us, Width, Height );
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D16_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD2D16_v2( dstDevice, dstPitch, us, Width, Height );
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD2D32( dstDevice, dstPitch, ui, Width, Height );
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D32_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD2D32_v2( dstDevice, dstPitch, ui, Width, Height );
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD2D8( dstDevice, dstPitch, uc, Width, Height );
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D8_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD2D8_v2( dstDevice,  dstPitch, uc, Width, Height );
}

CUresult CUDAAPI _CUDA::_cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD32( dstDevice, ui, N );
}

CUresult CUDAAPI _CUDA::_cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD32_v2( dstDevice, ui, N );
}

CUresult CUDAAPI _CUDA::_cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD8( dstDevice, uc, N );
}

CUresult CUDAAPI _CUDA::_cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuMemsetD8_v2( dstDevice, uc, N );
}

CUresult CUDAAPI _CUDA::_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleGetFunction(hfunc, hmod, name);
}

CUresult CUDAAPI _CUDA::_cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult CUDAAPI _CUDA::_cuModuleGetGlobal_v2(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

CUresult CUDAAPI _CUDA::_cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleGetSurfRef(pSurfRef, hmod, name);
}

CUresult CUDAAPI _CUDA::_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleGetTexRef(pTexRef, hmod, name);
}

CUresult CUDAAPI _CUDA::_cuModuleLoad(CUmodule *module, const char *fname)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleLoad(module, fname);
}

CUresult CUDAAPI _CUDA::_cuModuleLoadData(CUmodule *module, const void *image)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleLoadData(module, image);
}

CUresult CUDAAPI _CUDA::_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
}

CUresult CUDAAPI _CUDA::_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleLoadFatBinary(module, fatCubin);
}

CUresult CUDAAPI _CUDA::_cuModuleUnload(CUmodule hmod)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuModuleUnload(hmod);
}

CUresult CUDAAPI _CUDA::_cuParamSetf    (CUfunction hfunc, int offset, float value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuParamSetf    (hfunc, offset, value);
}

CUresult CUDAAPI _CUDA::_cuParamSeti    (CUfunction hfunc, int offset, unsigned int value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuParamSeti    (hfunc, offset, value);
}

CUresult CUDAAPI _CUDA::_cuParamSetSize (CUfunction hfunc, unsigned int numbytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuParamSetSize (hfunc, numbytes);
}

CUresult CUDAAPI _CUDA::_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuParamSetTexRef(hfunc, texunit, hTexRef);
}

CUresult CUDAAPI _CUDA::_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuParamSetv( hfunc, offset, ptr, numbytes);
}

CUresult CUDAAPI _CUDA::_cuStreamCreate( CUstream *phStream, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuStreamCreate( phStream, Flags );
}

CUresult CUDAAPI _CUDA::_cuStreamDestroy( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuStreamDestroy( hStream );
}

CUresult CUDAAPI _CUDA::_cuStreamQuery( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuStreamQuery( hStream );
}

CUresult CUDAAPI _CUDA::_cuStreamSynchronize( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuStreamSynchronize( hStream );
}

CUresult CUDAAPI _CUDA::_cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuSurfRefGetArray( phArray, hSurfRef );
}

CUresult CUDAAPI _CUDA::_cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuSurfRefSetArray( hSurfRef, hArray, Flags );
}

CUresult CUDAAPI _CUDA::_cuTexRefCreate( CUtexref *pTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefCreate( pTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefDestroy( CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefDestroy( hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetAddress( pdptr, hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetAddress_v2( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetAddress_v2( pdptr, hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetAddressMode( pam, hTexRef, dim );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetArray( phArray, hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetFilterMode( pfm, hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetFlags( pFlags, hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefGetFormat( pFormat, pNumChannels, hTexRef );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetAddress( ByteOffset, hTexRef, dptr, bytes );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress_v2( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetAddress_v2( ByteOffset, hTexRef, dptr, bytes );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetAddress2D( hTexRef, desc, dptr, Pitch);
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress2D_v2( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetAddress2D_v2( hTexRef, desc, dptr, Pitch);
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetAddressMode( hTexRef, dim, am );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetArray( hTexRef, hArray, Flags );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetFilterMode( hTexRef, fm );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetFlags( hTexRef, Flags );
}

CUresult CUDAAPI _CUDA::_cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    return cu->CurrentDevice()->_cuTexRefSetFormat( hTexRef, fmt, NumPackedComponents );
}
