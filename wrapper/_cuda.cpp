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
#include "../emulator/emulator.h"
#include "_cuda.h"


void _CUDA::WrapModule()
{
    // Add Driver API hooking.
    char * cuda_module_name = "nvcuda.dll";
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    HookManager * hook_manager = cu->hook_manager;
    hook_manager->HookImport(cuda_module_name, "cuArray3DCreate", (PROC)_CUDA::_cuArray3DCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuArray3DCreate_v2", (PROC)_CUDA::_cuArray3DCreate_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuArray3DGetDescriptor", (PROC)_CUDA::_cuArray3DGetDescriptor, true);
    hook_manager->HookImport(cuda_module_name, "cuArray3DGetDescriptor_v2", (PROC)_CUDA::_cuArray3DGetDescriptor_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuArrayCreate", (PROC)_CUDA::_cuArrayCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuArrayCreate_v2", (PROC)_CUDA::_cuArrayCreate_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuArrayDestroy", (PROC)_CUDA::_cuArrayDestroy, true);
    hook_manager->HookImport(cuda_module_name, "cuArrayGetDescriptor", (PROC)_CUDA::_cuArrayGetDescriptor, true);
    hook_manager->HookImport(cuda_module_name, "cuArrayGetDescriptor_v2", (PROC)_CUDA::_cuArrayGetDescriptor_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxAttach", (PROC)_CUDA::_cuCtxAttach, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxCreate", (PROC)_CUDA::_cuCtxCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxCreate_v2", (PROC)_CUDA::_cuCtxCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxDestroy", (PROC)_CUDA::_cuCtxDestroy, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxDetach", (PROC)_CUDA::_cuCtxDetach, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxGetDevice", (PROC)_CUDA::_cuCtxGetDevice, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxGetLimit", (PROC)_CUDA::_cuCtxGetLimit, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxPopCurrent", (PROC)_CUDA::_cuCtxPopCurrent, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxPushCurrent", (PROC)_CUDA::_cuCtxPushCurrent, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxSetLimit", (PROC)_CUDA::_cuCtxSetLimit, true);
    hook_manager->HookImport(cuda_module_name, "cuCtxSynchronize", (PROC)_CUDA::_cuCtxSynchronize, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceComputeCapability", (PROC)_CUDA::_cuDeviceComputeCapability, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGet", (PROC)_CUDA::_cuDeviceGet, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetAttribute", (PROC)_CUDA::_cuDeviceGetAttribute, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetCount", (PROC)_CUDA::_cuDeviceGetCount, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetName", (PROC)_CUDA::_cuDeviceGetName, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceGetProperties", (PROC)_CUDA::_cuDeviceGetProperties, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceTotalMem", (PROC)_CUDA::_cuDeviceTotalMem, true);
    hook_manager->HookImport(cuda_module_name, "cuDeviceTotalMem_v2", (PROC)_CUDA::_cuDeviceTotalMem_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuDriverGetVersion", (PROC)_CUDA::_cuDriverGetVersion, true);
    hook_manager->HookImport(cuda_module_name, "cuEventCreate", (PROC)_CUDA::_cuEventCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuEventDestroy", (PROC)_CUDA::_cuEventDestroy, true);
    hook_manager->HookImport(cuda_module_name, "cuEventElapsedTime", (PROC)_CUDA::_cuEventElapsedTime, true);
    hook_manager->HookImport(cuda_module_name, "cuEventQuery", (PROC)_CUDA::_cuEventQuery, true);
    hook_manager->HookImport(cuda_module_name, "cuEventRecord", (PROC)_CUDA::_cuEventRecord, true);
    hook_manager->HookImport(cuda_module_name, "cuEventSynchronize", (PROC)_CUDA::_cuEventSynchronize, true);
    hook_manager->HookImport(cuda_module_name, "cuFuncGetAttribute", (PROC)_CUDA::_cuFuncGetAttribute, true);
    hook_manager->HookImport(cuda_module_name, "cuFuncSetBlockShape", (PROC)_CUDA::_cuFuncSetBlockShape, true);
    hook_manager->HookImport(cuda_module_name, "cuFuncSetCacheConfig", (PROC)_CUDA::_cuFuncSetCacheConfig, true);
    hook_manager->HookImport(cuda_module_name, "cuFuncSetSharedSize", (PROC)_CUDA::_cuFuncSetSharedSize, true);
    hook_manager->HookImport(cuda_module_name, "cuGetExportTable", (PROC)_CUDA::_cuGetExportTable, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsMapResources", (PROC)_CUDA::_cuGraphicsMapResources, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsResourceGetMappedPointer", (PROC)_CUDA::_cuGraphicsResourceGetMappedPointer, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsResourceGetMappedPointer_v2", (PROC)_CUDA::_cuGraphicsResourceGetMappedPointer_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsResourceSetMapFlags", (PROC)_CUDA::_cuGraphicsResourceSetMapFlags, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsSubResourceGetMappedArray", (PROC)_CUDA::_cuGraphicsSubResourceGetMappedArray, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsUnmapResources", (PROC)_CUDA::_cuGraphicsUnmapResources, true);
    hook_manager->HookImport(cuda_module_name, "cuGraphicsUnregisterResource", (PROC)_CUDA::_cuGraphicsUnregisterResource, true);
    hook_manager->HookImport(cuda_module_name, "cuInit", (PROC)_CUDA::_cuInit, true);
    hook_manager->HookImport(cuda_module_name, "cuLaunch", (PROC)_CUDA::_cuLaunch, true);
    hook_manager->HookImport(cuda_module_name, "cuLaunchGrid", (PROC)_CUDA::_cuLaunchGrid, true);
    hook_manager->HookImport(cuda_module_name, "cuLaunchGridAsync", (PROC)_CUDA::_cuLaunchGridAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemAlloc", (PROC)_CUDA::_cuMemAlloc, true);
    hook_manager->HookImport(cuda_module_name, "cuMemAlloc_v2", (PROC)_CUDA::_cuMemAlloc_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocHost", (PROC)_CUDA::_cuMemAllocHost, true);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocHost_v2", (PROC)_CUDA::_cuMemAllocHost_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocPitch", (PROC)_CUDA::_cuMemAllocPitch, true);
    hook_manager->HookImport(cuda_module_name, "cuMemAllocPitch_v2", (PROC)_CUDA::_cuMemAllocPitch_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2D", (PROC)_CUDA::_cuMemcpy2D, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2D_v2", (PROC)_CUDA::_cuMemcpy2D_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DAsync", (PROC)_CUDA::_cuMemcpy2DAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DAsync_v2", (PROC)_CUDA::_cuMemcpy2DAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DUnaligned", (PROC)_CUDA::_cuMemcpy2DUnaligned, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy2DUnaligned_v2", (PROC)_CUDA::_cuMemcpy2DUnaligned_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3D", (PROC)_CUDA::_cuMemcpy3D, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3D_v2", (PROC)_CUDA::_cuMemcpy3D_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3DAsync", (PROC)_CUDA::_cuMemcpy3DAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpy3DAsync_v2", (PROC)_CUDA::_cuMemcpy3DAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoA", (PROC)_CUDA::_cuMemcpyAtoA, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoA_v2", (PROC)_CUDA::_cuMemcpyAtoA_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoD", (PROC)_CUDA::_cuMemcpyAtoD, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoD_v2", (PROC)_CUDA::_cuMemcpyAtoD_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoH", (PROC)_CUDA::_cuMemcpyAtoH, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoH_v2", (PROC)_CUDA::_cuMemcpyAtoH_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoHAsync", (PROC)_CUDA::_cuMemcpyAtoHAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyAtoHAsync_v2", (PROC)_CUDA::_cuMemcpyAtoHAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoA", (PROC)_CUDA::_cuMemcpyDtoA, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoA_v2", (PROC)_CUDA::_cuMemcpyDtoA_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoD", (PROC)_CUDA::_cuMemcpyDtoD, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoD_v2", (PROC)_CUDA::_cuMemcpyDtoD_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoDAsync", (PROC)_CUDA::_cuMemcpyDtoDAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoDAsync_v2", (PROC)_CUDA::_cuMemcpyDtoDAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoH", (PROC)_CUDA::_cuMemcpyDtoH, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoH_v2", (PROC)_CUDA::_cuMemcpyDtoH_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoHAsync", (PROC)_CUDA::_cuMemcpyDtoHAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyDtoHAsync_v2", (PROC)_CUDA::_cuMemcpyDtoHAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoA", (PROC)_CUDA::_cuMemcpyHtoA, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoA_v2", (PROC)_CUDA::_cuMemcpyHtoA_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoAAsync", (PROC)_CUDA::_cuMemcpyHtoAAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoAAsync_v2", (PROC)_CUDA::_cuMemcpyHtoAAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoD", (PROC)_CUDA::_cuMemcpyHtoD, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoD_v2", (PROC)_CUDA::_cuMemcpyHtoD_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoDAsync", (PROC)_CUDA::_cuMemcpyHtoDAsync, true);
    hook_manager->HookImport(cuda_module_name, "cuMemcpyHtoDAsync_v2", (PROC)_CUDA::_cuMemcpyHtoDAsync_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemFree", (PROC)_CUDA::_cuMemFree, true);
    hook_manager->HookImport(cuda_module_name, "cuMemFree_v2", (PROC)_CUDA::_cuMemFree_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemFreeHost", (PROC)_CUDA::_cuMemFreeHost, true);
    hook_manager->HookImport(cuda_module_name, "cuMemGetAddressRange", (PROC)_CUDA::_cuMemGetAddressRange, true);
    hook_manager->HookImport(cuda_module_name, "cuMemGetAddressRange_v2", (PROC)_CUDA::_cuMemGetAddressRange_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemGetInfo", (PROC)_CUDA::_cuMemGetInfo, true);
    hook_manager->HookImport(cuda_module_name, "cuMemGetInfo_v2", (PROC)_CUDA::_cuMemGetInfo_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemHostAlloc", (PROC)_CUDA::_cuMemHostAlloc, true);
    hook_manager->HookImport(cuda_module_name, "cuMemHostGetDevicePointer", (PROC)_CUDA::_cuMemHostGetDevicePointer, true);
    hook_manager->HookImport(cuda_module_name, "cuMemHostGetDevicePointer_v2", (PROC)_CUDA::_cuMemHostGetDevicePointer_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemHostGetFlags", (PROC)_CUDA::_cuMemHostGetFlags, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD16", (PROC)_CUDA::_cuMemsetD16, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD16_v2", (PROC)_CUDA::_cuMemsetD16_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D16", (PROC)_CUDA::_cuMemsetD2D16, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D16_v2", (PROC)_CUDA::_cuMemsetD2D16_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D32", (PROC)_CUDA::_cuMemsetD2D32, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D32_v2", (PROC)_CUDA::_cuMemsetD2D32_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D8", (PROC)_CUDA::_cuMemsetD2D8, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD2D8_v2", (PROC)_CUDA::_cuMemsetD2D8_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD32", (PROC)_CUDA::_cuMemsetD32, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD32_v2", (PROC)_CUDA::_cuMemsetD32_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD8", (PROC)_CUDA::_cuMemsetD8, true);
    hook_manager->HookImport(cuda_module_name, "cuMemsetD8_v2", (PROC)_CUDA::_cuMemsetD8_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetFunction", (PROC)_CUDA::_cuModuleGetFunction, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetGlobal", (PROC)_CUDA::_cuModuleGetGlobal, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetGlobal_v2", (PROC)_CUDA::_cuModuleGetGlobal_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetSurfRef", (PROC)_CUDA::_cuModuleGetSurfRef, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleGetTexRef", (PROC)_CUDA::_cuModuleGetTexRef, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoad", (PROC)_CUDA::_cuModuleLoad, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoadData", (PROC)_CUDA::_cuModuleLoadData, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoadDataEx", (PROC)_CUDA::_cuModuleLoadDataEx, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleLoadFatBinary", (PROC)_CUDA::_cuModuleLoadFatBinary, true);
    hook_manager->HookImport(cuda_module_name, "cuModuleUnload", (PROC)_CUDA::_cuModuleUnload, true);
    hook_manager->HookImport(cuda_module_name, "cuParamSetf", (PROC)_CUDA::_cuParamSetf, true);
    hook_manager->HookImport(cuda_module_name, "cuParamSeti", (PROC)_CUDA::_cuParamSeti, true);
    hook_manager->HookImport(cuda_module_name, "cuParamSetSize", (PROC)_CUDA::_cuParamSetSize, true);
    hook_manager->HookImport(cuda_module_name, "cuParamSetTexRef", (PROC)_CUDA::_cuParamSetTexRef, true);
    hook_manager->HookImport(cuda_module_name, "cuParamSetv", (PROC)_CUDA::_cuParamSetv, true);
    hook_manager->HookImport(cuda_module_name, "cuStreamCreate", (PROC)_CUDA::_cuStreamCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuStreamDestroy", (PROC)_CUDA::_cuStreamDestroy, true);
    hook_manager->HookImport(cuda_module_name, "cuStreamQuery", (PROC)_CUDA::_cuStreamQuery, true);
    hook_manager->HookImport(cuda_module_name, "cuStreamSynchronize", (PROC)_CUDA::_cuStreamSynchronize, true);
    hook_manager->HookImport(cuda_module_name, "cuSurfRefGetArray", (PROC)_CUDA::_cuSurfRefGetArray, true);
    hook_manager->HookImport(cuda_module_name, "cuSurfRefSetArray", (PROC)_CUDA::_cuSurfRefSetArray, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefCreate", (PROC)_CUDA::_cuTexRefCreate, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefDestroy", (PROC)_CUDA::_cuTexRefDestroy, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetAddress", (PROC)_CUDA::_cuTexRefGetAddress, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetAddress_v2", (PROC)_CUDA::_cuTexRefGetAddress_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetAddressMode", (PROC)_CUDA::_cuTexRefGetAddressMode, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetArray", (PROC)_CUDA::_cuTexRefGetArray, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetFilterMode", (PROC)_CUDA::_cuTexRefGetFilterMode, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetFlags", (PROC)_CUDA::_cuTexRefGetFlags, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefGetFormat", (PROC)_CUDA::_cuTexRefGetFormat, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress", (PROC)_CUDA::_cuTexRefSetAddress, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress2D", (PROC)_CUDA::_cuTexRefSetAddress2D, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress2D_v2", (PROC)_CUDA::_cuTexRefSetAddress2D_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddress_v2", (PROC)_CUDA::_cuTexRefSetAddress_v2, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetAddressMode", (PROC)_CUDA::_cuTexRefSetAddressMode, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetArray", (PROC)_CUDA::_cuTexRefSetArray, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetFilterMode", (PROC)_CUDA::_cuTexRefSetFilterMode, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetFlags", (PROC)_CUDA::_cuTexRefSetFlags, true);
    hook_manager->HookImport(cuda_module_name, "cuTexRefSetFormat", (PROC)_CUDA::_cuTexRefSetFormat, true);
}

CUresult CUDAAPI _CUDA::_cuArray3DCreate( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuArray3DCreate proc = (ptr_cuArray3DCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArray3DCreate);
        return (*proc)(pHandle, pAllocateArray);
    } else
    {
        std::cout << "Unimplemented function _cuArray3DCreate\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuArray3DCreate_v2( CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuArray3DCreate_v2 proc = (ptr_cuArray3DCreate_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArray3DCreate_v2);
		return (*proc)(pHandle, pAllocateArray);
	} else
	{
		std::cout << "Unimplemented function _cuArray3DCreate_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuArray3DGetDescriptor( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuArray3DGetDescriptor proc = (ptr_cuArray3DGetDescriptor)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArray3DGetDescriptor);
        return (*proc)(pArrayDescriptor, hArray);
    } else
    {
        std::cout << "Unimplemented function _cuArray3DGetDescriptor\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuArray3DGetDescriptor_v2( CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuArray3DGetDescriptor_v2 proc = (ptr_cuArray3DGetDescriptor_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArray3DGetDescriptor_v2);
		return (*proc)(pArrayDescriptor, hArray);
	} else
	{
		std::cout << "Unimplemented function _cuArray3DGetDescriptor_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuArrayCreate( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuArrayCreate proc = (ptr_cuArrayCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArrayCreate);
        return (*proc)(pHandle, pAllocateArray );
    } else
    {
        std::cout << "Unimplemented function _cuArrayCreate\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuArrayCreate_v2( CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuArrayCreate_v2 proc = (ptr_cuArrayCreate_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArrayCreate_v2);
		return (*proc)(pHandle, pAllocateArray );
	} else
	{
		std::cout << "Unimplemented function _cuArrayCreate_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuArrayDestroy( CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuArrayDestroy proc = (ptr_cuArrayDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArrayDestroy);
        return (*proc)( hArray );
    } else
    {
        std::cout << "Unimplemented function _cuArrayDestroy\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuArrayGetDescriptor( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuArrayGetDescriptor proc = (ptr_cuArrayGetDescriptor)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArrayGetDescriptor);
        return (*proc)( pArrayDescriptor, hArray );
    } else
    {
        std::cout << "Unimplemented function _cuArrayGetDescriptor\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuArrayGetDescriptor_v2( CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuArrayGetDescriptor_v2 proc = (ptr_cuArrayGetDescriptor_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuArrayGetDescriptor_v2);
		return (*proc)( pArrayDescriptor, hArray );
	} else
	{
		std::cout << "Unimplemented function _cuArrayGetDescriptor_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxAttach proc = (ptr_cuCtxAttach)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxAttach);
        return (*proc)(pctx, flags);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuCtxAttach(pctx, flags);
    }
}

CUresult CUDAAPI _CUDA::_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxCreate proc = (ptr_cuCtxCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxCreate);
        return (*proc)(pctx, flags, dev);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuCtxCreate(pctx, flags, dev);
    }
}

CUresult CUDAAPI _CUDA::_cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuCtxCreate_v2 proc = (ptr_cuCtxCreate_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxCreate_v2);
		return (*proc)(pctx, flags, dev);
	} else
	{
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuCtxCreate(pctx, flags, dev);
	}
}

CUresult CUDAAPI _CUDA::_cuCtxDestroy( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxDestroy proc = (ptr_cuCtxDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxDestroy);
        return (*proc)( ctx );
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuCtxDestroy(ctx);
    }
}

CUresult CUDAAPI _CUDA::_cuCtxDetach(CUcontext ctx)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxDetach proc = (ptr_cuCtxDetach)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxDetach);
        return (*proc)(ctx);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuCtxDetach(ctx);
    }
}

CUresult CUDAAPI _CUDA::_cuCtxGetDevice(CUdevice *device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxGetDevice proc = (ptr_cuCtxGetDevice)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxGetDevice);
        return (*proc)(device);
    } else
    {
        std::cout << "Unimplemented function _cuCtxGetDevice\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxGetLimit proc = (ptr_cuCtxGetLimit)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxGetLimit);
        return (*proc)(pvalue, limit);
    } else
    {
        std::cout << "Unimplemented function _cuCtxGetLimit\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuCtxPopCurrent( CUcontext *pctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxPopCurrent proc = (ptr_cuCtxPopCurrent)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxPopCurrent);
        return (*proc)( pctx );
    } else
    {
        std::cout << "Unimplemented function _cuCtxPopCurrent\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuCtxPushCurrent( CUcontext ctx )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxPushCurrent proc = (ptr_cuCtxPushCurrent)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxPushCurrent);
        return (*proc)( ctx );
    } else
    {
        std::cout << "Unimplemented function _cuCtxPushCurrent\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuCtxSetLimit(CUlimit limit, size_t value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxSetLimit proc = (ptr_cuCtxSetLimit)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxSetLimit);
        return (*proc)(limit, value);
    } else
    {
        std::cout << "Unimplemented function _cuCtxSetLimit\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuCtxSynchronize(void)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuCtxSynchronize proc = (ptr_cuCtxSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuCtxSynchronize);
        return (*proc)();
    } else
    {
        std::cout << "Unimplemented function _cuCtxSynchronize\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceComputeCapability proc = (ptr_cuDeviceComputeCapability)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceComputeCapability);
        return (*proc)(major, minor, dev);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceComputeCapability(major, minor, dev);
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceGet(CUdevice *device, int ordinal)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceGet proc = (ptr_cuDeviceGet)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGet);
        return (*proc)(device, ordinal);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceGet(device, ordinal);
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceGetAttribute proc = (ptr_cuDeviceGetAttribute)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetAttribute);
        return (*proc)(pi, attrib, dev);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceGetAttribute(pi, attrib, dev);
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceGetCount(int *count)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceGetCount proc = (ptr_cuDeviceGetCount)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetCount);
        return (*proc)(count);
    } else
    {
        *count = 1;
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceGetName(char *name, int len, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceGetName proc = (ptr_cuDeviceGetName)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetName);
        return (*proc)(name, len, dev);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceGetName(name, len, dev);
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceGetProperties proc = (ptr_cuDeviceGetProperties)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceGetProperties);
        return (*proc)(prop, dev);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceGetProperties(prop, dev);
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceTotalMem(unsigned int *bytes, CUdevice dev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDeviceTotalMem proc = (ptr_cuDeviceTotalMem)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceTotalMem);
        return (*proc)(bytes, dev);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceTotalMem(bytes, dev);
    }
}

CUresult CUDAAPI _CUDA::_cuDeviceTotalMem_v2(unsigned int *bytes, CUdevice dev)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuDeviceTotalMem_v2 proc = (ptr_cuDeviceTotalMem_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDeviceTotalMem_v2);
		return (*proc)(bytes, dev);
	} else
	{
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDeviceTotalMem(bytes, dev);
	}
}

CUresult CUDAAPI _CUDA::_cuDriverGetVersion(int *driverVersion)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuDriverGetVersion proc = (ptr_cuDriverGetVersion)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuDriverGetVersion);
        return (*proc)(driverVersion);
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuDriverGetVersion(driverVersion);
    }
}

CUresult CUDAAPI _CUDA::_cuEventCreate( CUevent *phEvent, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuEventCreate proc = (ptr_cuEventCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventCreate);
        return (*proc)( phEvent, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuEventCreate\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuEventDestroy( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuEventDestroy proc = (ptr_cuEventDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventDestroy);
        return (*proc)( hEvent );
    } else
    {
        std::cout << "Unimplemented function _cuEventDestroy\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuEventElapsedTime( float *pMilliseconds, CUevent hStart, CUevent hEnd )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuEventElapsedTime proc = (ptr_cuEventElapsedTime)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventElapsedTime);
        return (*proc)( pMilliseconds, hStart, hEnd );
    } else
    {
        std::cout << "Unimplemented function _cuEventElapsedTime\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuEventQuery( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuEventQuery proc = (ptr_cuEventQuery)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventQuery);
        return (*proc)( hEvent );
    } else
    {
        std::cout << "Unimplemented function _cuEventQuery\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuEventRecord( CUevent hEvent, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuEventRecord proc = (ptr_cuEventRecord)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventRecord);
        return (*proc)( hEvent, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuEventRecord\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuEventSynchronize( CUevent hEvent )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuEventSynchronize proc = (ptr_cuEventSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuEventSynchronize);
        return (*proc)( hEvent );
    } else
    {
        std::cout << "Unimplemented function _cuEventSynchronize\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuFuncGetAttribute proc = (ptr_cuFuncGetAttribute)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncGetAttribute);
        return (*proc) (pi, attrib, hfunc);
    } else
    {
        std::cout << "Unimplemented function _cuFuncGetAttribute\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z)
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuFuncSetBlockShape proc = (ptr_cuFuncSetBlockShape)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncSetBlockShape);
        return (*proc)(hfunc, x, y, z);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        dim3 blockDim(x,y,z);
        emulator->ConfigureBlock(blockDim);
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI _CUDA::_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuFuncSetCacheConfig proc = (ptr_cuFuncSetCacheConfig)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncSetCacheConfig);
        return (*proc)(hfunc, config);
    } else
    {
        std::cout << "Unimplemented function _cuFuncSetCacheConfig\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuFuncSetSharedSize proc = (ptr_cuFuncSetSharedSize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuFuncSetSharedSize);
        return (*proc) (hfunc, bytes);
    } else
    {
        std::cout << "Unimplemented function _cuFuncSetSharedSize\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuGetExportTable( const void **ppExportTable, const CUuuid *pExportTableId )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGetExportTable proc = (ptr_cuGetExportTable)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGetExportTable);
        return (*proc)( ppExportTable, pExportTableId );
    } else
    {
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->_cuGetExportTable(ppExportTable, pExportTableId);
    }
}

CUresult CUDAAPI _CUDA::_cuGraphicsMapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGraphicsMapResources proc = (ptr_cuGraphicsMapResources)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsMapResources);
        return (*proc)( count, resources, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuGraphicsMapResources\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuGraphicsResourceGetMappedPointer( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGraphicsResourceGetMappedPointer proc = (ptr_cuGraphicsResourceGetMappedPointer)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsResourceGetMappedPointer);
        return (*proc)( pDevPtr, pSize, resource );
    } else
    {
        std::cout << "Unimplemented function _cuGraphicsResourceGetMappedPointer\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuGraphicsResourceGetMappedPointer_v2( CUdeviceptr *pDevPtr, unsigned int *pSize, CUgraphicsResource resource )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuGraphicsResourceGetMappedPointer_v2 proc = (ptr_cuGraphicsResourceGetMappedPointer_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsResourceGetMappedPointer);
		return (*proc)( pDevPtr, pSize, resource );
	} else
	{
		std::cout << "Unimplemented function _cuGraphicsResourceGetMappedPointer_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuGraphicsResourceSetMapFlags( CUgraphicsResource resource, unsigned int flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGraphicsResourceSetMapFlags proc = (ptr_cuGraphicsResourceSetMapFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsResourceSetMapFlags);
        return (*proc)( resource, flags );;
    } else
    {
        std::cout << "Unimplemented function _cuGraphicsResourceSetMapFlags\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuGraphicsSubResourceGetMappedArray( CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGraphicsSubResourceGetMappedArray proc = (ptr_cuGraphicsSubResourceGetMappedArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsSubResourceGetMappedArray);
        return (*proc)( pArray, resource, arrayIndex, mipLevel );
    } else
    {
        std::cout << "Unimplemented function _cuGraphicsSubResourceGetMappedArray\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuGraphicsUnmapResources( unsigned int count, CUgraphicsResource *resources, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGraphicsUnmapResources proc = (ptr_cuGraphicsUnmapResources)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsUnmapResources);
        return (*proc)( count, resources, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuGraphicsUnmapResources\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuGraphicsUnregisterResource proc = (ptr_cuGraphicsUnregisterResource)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuGraphicsUnregisterResource);
        return (*proc)(resource);
    } else
    {
        std::cout << "Unimplemented function _cuGraphicsUnregisterResource\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuInit(unsigned int Flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuInit proc = (ptr_cuInit)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuInit);
        return (*proc)(Flags);
    } else
    {
        if (Flags == 0)
            return CUDA_SUCCESS;
        else
            return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuLaunch ( CUfunction f )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuLaunch proc = (ptr_cuLaunch)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuLaunch);
        return (*proc) ( f );
    } else
    {
        std::cout << "Unimplemented function _cuLaunch\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuLaunchGrid (CUfunction f, int grid_width, int grid_height)
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuLaunchGrid proc = (ptr_cuLaunchGrid)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuLaunchGrid);
        return (*proc)(f, grid_width, grid_height);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cuLaunchGrid(f, grid_width, grid_height);
    }
}

CUresult CUDAAPI _CUDA::_cuLaunchGridAsync( CUfunction f, int grid_width, int grid_height, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuLaunchGridAsync proc = (ptr_cuLaunchGridAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuLaunchGridAsync);
        return (*proc)( f, grid_width, grid_height, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuLaunchGridAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize)
{
    // Basic, no frills, allocation.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemAlloc proc = (ptr_cuMemAlloc)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAlloc);
        CUresult e1 = (*proc)(dptr, bytesize);
        return e1;
    } else
    {
        *dptr = (CUdeviceptr)malloc(bytesize);
        return CUDA_SUCCESS;     
    }
}

CUresult CUDAAPI _CUDA::_cuMemAlloc_v2( CUdeviceptr *dptr, unsigned int bytesize)
{
	// Basic, no frills, allocation.
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemAlloc_v2 proc = (ptr_cuMemAlloc_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAlloc_v2);
		CUresult e1 = (*proc)(dptr, bytesize);
		return e1;
	} else
	{
		*dptr = (CUdeviceptr)malloc(bytesize);
		return CUDA_SUCCESS;     
	}
}

CUresult CUDAAPI _CUDA::_cuMemAllocHost(void **pp, unsigned int bytesize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemAllocHost proc = (ptr_cuMemAllocHost)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocHost);
        return (*proc)(pp, bytesize);
    } else
    {
        std::cout << "Unimplemented function _cuMemAllocHost\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemAllocHost_v2(void **pp, unsigned int bytesize)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemAllocHost_v2 proc = (ptr_cuMemAllocHost_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocHost_v2);
		return (*proc)(pp, bytesize);
	} else
	{
		std::cout << "Unimplemented function _cuMemAllocHost_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemAllocPitch( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemAllocPitch proc = (ptr_cuMemAllocPitch)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocPitch);
        return (*proc)( dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    } else
    {
        std::cout << "Unimplemented function _cuMemAllocPitch\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemAllocPitch_v2( CUdeviceptr *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemAllocPitch_v2 proc = (ptr_cuMemAllocPitch_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemAllocPitch_v2);
		return (*proc)( dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
	} else
	{
		std::cout << "Unimplemented function _cuMemAllocPitch_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpy2D( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpy2D proc = (ptr_cuMemcpy2D)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2D);
        return (*proc)( pCopy );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpy2D\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpy2D_v2( const CUDA_MEMCPY2D *pCopy )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpy2D_v2 proc = (ptr_cuMemcpy2D_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2D_v2);
		return (*proc)( pCopy );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpy2D_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpy2DAsync proc = (ptr_cuMemcpy2DAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DAsync);
        return (*proc)( pCopy, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpy2DAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DAsync_v2( const CUDA_MEMCPY2D *pCopy, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpy2DAsync_v2 proc = (ptr_cuMemcpy2DAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DAsync_v2);
		return (*proc)( pCopy, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpy2DAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpy2DUnaligned proc = (ptr_cuMemcpy2DUnaligned)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DUnaligned);
        return (*proc)( pCopy );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpy2DUnaligned\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpy2DUnaligned_v2( const CUDA_MEMCPY2D *pCopy )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpy2DUnaligned_v2 proc = (ptr_cuMemcpy2DUnaligned_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy2DUnaligned_v2);
		return (*proc)( pCopy );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpy2DUnaligned_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpy3D( const CUDA_MEMCPY3D *pCopy )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpy3D proc = (ptr_cuMemcpy3D)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3D);
        return (*proc)( pCopy );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpy3D\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpy3D_v2( const CUDA_MEMCPY3D *pCopy )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpy3D_v2 proc = (ptr_cuMemcpy3D_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3D_v2);
		return (*proc)( pCopy );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpy3D_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpy3DAsync proc = (ptr_cuMemcpy3DAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3DAsync);
        return (*proc)( pCopy, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpy3DAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpy3DAsync_v2( const CUDA_MEMCPY3D *pCopy, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpy3DAsync_v2 proc = (ptr_cuMemcpy3DAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpy3DAsync_v2);
		return (*proc)( pCopy, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpy3DAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoA( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyAtoA proc = (ptr_cuMemcpyAtoA)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoA);
        return (*proc)( dstArray, dstOffset, srcArray, srcOffset, ByteCount );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyAtoA\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoA_v2( CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyAtoA_v2 proc = (ptr_cuMemcpyAtoA_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoA_v2);
		return (*proc)( dstArray, dstOffset, srcArray, srcOffset, ByteCount );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyAtoA_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyAtoD proc = (ptr_cuMemcpyAtoD)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoD);
        return (*proc) ( dstDevice, srcArray, srcOffset, ByteCount );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyAtoD\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyAtoD_v2 proc = (ptr_cuMemcpyAtoD_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoD_v2);
		return (*proc) ( dstDevice, srcArray, srcOffset, ByteCount );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyAtoD_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoH( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyAtoH proc = (ptr_cuMemcpyAtoH)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoH);
        return (*proc)( dstHost, srcArray, srcOffset, ByteCount );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyAtoH\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoH_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyAtoH_v2 proc = (ptr_cuMemcpyAtoH_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoH_v2);
		return (*proc)( dstHost, srcArray, srcOffset, ByteCount );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyAtoH_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyAtoHAsync proc = (ptr_cuMemcpyAtoHAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoHAsync);
        return (*proc)( dstHost, srcArray, srcOffset, ByteCount, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyAtoHAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyAtoHAsync_v2( void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyAtoHAsync_v2 proc = (ptr_cuMemcpyAtoHAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyAtoHAsync_v2);
		return (*proc)( dstHost, srcArray, srcOffset, ByteCount, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyAtoHAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoA ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyDtoA proc = (ptr_cuMemcpyDtoA)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoA);
        return (*proc) ( dstArray, dstOffset, srcDevice, ByteCount );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyDtoA\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoA_v2 ( CUarray dstArray, unsigned int dstOffset, CUdeviceptr srcDevice, unsigned int ByteCount )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyDtoA_v2 proc = (ptr_cuMemcpyDtoA_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoA_v2);
		return (*proc) ( dstArray, dstOffset, srcDevice, ByteCount );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyDtoA_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoD (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyDtoD proc = (ptr_cuMemcpyDtoD)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoD);
        return (*proc) (dstDevice, srcDevice, ByteCount );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyDtoD\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyDtoD_v2 proc = (ptr_cuMemcpyDtoD_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoD_v2);
		return (*proc) (dstDevice, srcDevice, ByteCount );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyDtoD_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoDAsync (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyDtoDAsync proc = (ptr_cuMemcpyDtoDAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoDAsync);
        return (*proc) (dstDevice, srcDevice, ByteCount, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyDtoDAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyDtoDAsync_v2 proc = (ptr_cuMemcpyDtoDAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoDAsync_v2);
		return (*proc) (dstDevice, srcDevice, ByteCount, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyDtoDAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyDtoH proc = (ptr_cuMemcpyDtoH)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoH);
        CUresult e1 = (*proc)(dstHost, srcDevice, ByteCount);
        return e1;
    } else
    {
        memcpy(dstHost, (void*)srcDevice, ByteCount);
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoH_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount )
{
	// Basic, no frills.
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyDtoH_v2 proc = (ptr_cuMemcpyDtoH_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoH_v2);
		CUresult e1 = (*proc)(dstHost, srcDevice, ByteCount);
		return e1;
	} else
	{
		memcpy(dstHost, (void*)srcDevice, ByteCount);
		return CUDA_SUCCESS;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoHAsync (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyDtoHAsync proc = (ptr_cuMemcpyDtoHAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoHAsync);
        return (*proc) (dstHost, srcDevice, ByteCount, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyDtoHAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyDtoHAsync_v2 (void *dstHost, CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyDtoHAsync_v2 proc = (ptr_cuMemcpyDtoHAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyDtoHAsync_v2);
		return (*proc) (dstHost, srcDevice, ByteCount, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyDtoHAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoA( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyHtoA proc = (ptr_cuMemcpyHtoA)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoA);
        return (*proc)( dstArray, dstOffset, srcHost, ByteCount );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyHtoA\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoA_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyHtoA_v2 proc = (ptr_cuMemcpyHtoA_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoA_v2);
		return (*proc)( dstArray, dstOffset, srcHost, ByteCount );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyHtoA_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoAAsync( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyHtoAAsync proc = (ptr_cuMemcpyHtoAAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoAAsync);
        return (*proc)( dstArray, dstOffset, srcHost, ByteCount, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyHtoAAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoAAsync_v2( CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyHtoAAsync_v2 proc = (ptr_cuMemcpyHtoAAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoAAsync_v2);
		return (*proc)( dstArray, dstOffset, srcHost, ByteCount, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyHtoAAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoD (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyHtoD proc = (ptr_cuMemcpyHtoD)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoD);
        CUresult e1 = (*proc)(dstDevice, srcHost, ByteCount);
        return e1;
    } else
    {
        memcpy((void*)dstDevice, srcHost, ByteCount);
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount )
{
	// Basic, no frills.
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyHtoD_v2 proc = (ptr_cuMemcpyHtoD_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoD_v2);
		CUresult e1 = (*proc)(dstDevice, srcHost, ByteCount);
		return e1;
	} else
	{
		memcpy((void*)dstDevice, srcHost, ByteCount);
		return CUDA_SUCCESS;
	}
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoDAsync (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemcpyHtoDAsync proc = (ptr_cuMemcpyHtoDAsync)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoDAsync);
        return (*proc) (dstDevice, srcHost, ByteCount, hStream );
    } else
    {
        std::cout << "Unimplemented function _cuMemcpyHtoDAsync\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemcpyHtoDAsync_v2 proc = (ptr_cuMemcpyHtoDAsync_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemcpyHtoDAsync_v2);
		return (*proc) (dstDevice, srcHost, ByteCount, hStream );
	} else
	{
		std::cout << "Unimplemented function _cuMemcpyHtoDAsync_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemFree(CUdeviceptr dptr)
{
    // Basic, no frills.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemFree proc = (ptr_cuMemFree)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemFree);
        CUresult e1 = (*proc)(dptr);
        return e1;
    } else
    {
        free((void*)dptr);
        return CUDA_SUCCESS;     
    }
}

CUresult CUDAAPI _CUDA::_cuMemFree_v2(CUdeviceptr dptr)
{
	// Basic, no frills.
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemFree_v2 proc = (ptr_cuMemFree_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemFree_v2);
		CUresult e1 = (*proc)(dptr);
		return e1;
	} else
	{
		free((void*)dptr);
		return CUDA_SUCCESS;     
	}
}

CUresult CUDAAPI _CUDA::_cuMemFreeHost(void *p)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemFreeHost proc = (ptr_cuMemFreeHost)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemFreeHost);
        return (*proc)(p);
    } else
    {
        std::cout << "Unimplemented function _cuMemFreeHost\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemGetAddressRange( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemGetAddressRange proc = (ptr_cuMemGetAddressRange)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetAddressRange);
        return (*proc)( pbase, psize, dptr );
    } else
    {
        std::cout << "Unimplemented function _cuMemGetAddressRange\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemGetAddressRange_v2( CUdeviceptr *pbase, unsigned int *psize, CUdeviceptr dptr )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemGetAddressRange_v2 proc = (ptr_cuMemGetAddressRange_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetAddressRange_v2);
		return (*proc)( pbase, psize, dptr );
	} else
	{
		std::cout << "Unimplemented function _cuMemGetAddressRange_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemGetInfo(unsigned int *free, unsigned int *total)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemGetInfo proc = (ptr_cuMemGetInfo)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetInfo);
        return (*proc)(free, total);
    } else
    {
        std::cout << "Unimplemented function _cuMemGetInfo\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemGetInfo_v2(unsigned int *free, unsigned int *total)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemGetInfo_v2 proc = (ptr_cuMemGetInfo_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemGetInfo_v2);
		return (*proc)(free, total);
	} else
	{
		std::cout << "Unimplemented function _cuMemGetInfo_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemHostAlloc proc = (ptr_cuMemHostAlloc)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostAlloc);
        return (*proc)(pp, bytesize, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuMemHostAlloc\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemHostGetDevicePointer( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemHostGetDevicePointer proc = (ptr_cuMemHostGetDevicePointer)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostGetDevicePointer);
        return (*proc)( pdptr, p, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuMemHostGetDevicePointer\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemHostGetDevicePointer_v2( CUdeviceptr *pdptr, void *p, unsigned int Flags )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemHostGetDevicePointer_v2 proc = (ptr_cuMemHostGetDevicePointer_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostGetDevicePointer_v2);
		return (*proc)( pdptr, p, Flags );
	} else
	{
		std::cout << "Unimplemented function _cuMemHostGetDevicePointer_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemHostGetFlags( unsigned int *pFlags, void *p )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemHostGetFlags proc = (ptr_cuMemHostGetFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemHostGetFlags);
        return (*proc)( pFlags, p );
    } else
    {
        std::cout << "Unimplemented function _cuMemHostGetFlags\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD16( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemsetD16 proc = (ptr_cuMemsetD16)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD16);
        return (*proc)( dstDevice, us, N );
    } else
    {
        std::cout << "Unimplemented function _cuMemsetD16\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, unsigned int N )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemsetD16_v2 proc = (ptr_cuMemsetD16_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD16_v2);
		return (*proc)( dstDevice, us, N );
	} else
	{
		std::cout << "Unimplemented function _cuMemsetD16_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D16( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemsetD2D16 proc = (ptr_cuMemsetD2D16)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D16);
        return (*proc)( dstDevice, dstPitch, us, Width, Height );
    } else
    {
        std::cout << "Unimplemented function _cuMemsetD2D16\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D16_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemsetD2D16_v2 proc = (ptr_cuMemsetD2D16_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D16_v2);
		return (*proc)( dstDevice, dstPitch, us, Width, Height );
	} else
	{
		std::cout << "Unimplemented function _cuMemsetD2D16_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D32( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemsetD2D32 proc = (ptr_cuMemsetD2D32)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D32);
        return (*proc)( dstDevice, dstPitch, ui, Width, Height );
    } else
    {
        std::cout << "Unimplemented function _cuMemsetD2D32\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D32_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemsetD2D32_v2 proc = (ptr_cuMemsetD2D32_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D32_v2);
		return (*proc)( dstDevice, dstPitch, ui, Width, Height );
	} else
	{
		std::cout << "Unimplemented function _cuMemsetD2D32_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D8( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemsetD2D8 proc = (ptr_cuMemsetD2D8)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D8);
        return (*proc)( dstDevice, dstPitch, uc, Width, Height );
    } else
    {
        std::cout << "Unimplemented function _cuMemsetD2D8\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD2D8_v2( CUdeviceptr dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemsetD2D8_v2 proc = (ptr_cuMemsetD2D8_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD2D8_v2);
		return (*proc)( dstDevice, dstPitch, uc, Width, Height );
	} else
	{
		std::cout << "Unimplemented function _cuMemsetD2D8_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemsetD32 proc = (ptr_cuMemsetD32)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD32);
        return (*proc)( dstDevice, ui, N );
    } else
    {
        std::cout << "Unimplemented function _cuMemsetD32\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int ui, unsigned int N )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemsetD32 proc = (ptr_cuMemsetD32)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD32);
		return (*proc)( dstDevice, ui, N );
	} else
	{
		std::cout << "Unimplemented function _cuMemsetD32\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuMemsetD8 proc = (ptr_cuMemsetD8)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD8);
        return (*proc)( dstDevice, uc, N );
    } else
    {
        std::cout << "Unimplemented function _cuMemsetD8\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char uc, unsigned int N )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuMemsetD8_v2 proc = (ptr_cuMemsetD8_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuMemsetD8_v2);
		return (*proc)( dstDevice, uc, N );
	} else
	{
		std::cout << "Unimplemented function _cuMemsetD8_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleGetFunction proc = (ptr_cuModuleGetFunction)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetFunction);
        return (*proc)(hfunc, hmod, name);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cuModuleGetFunction(hfunc, hmod, name);
    }
}

CUresult CUDAAPI _CUDA::_cuModuleGetGlobal(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleGetGlobal proc = (ptr_cuModuleGetGlobal)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetGlobal);
        return (*proc)(dptr, bytes, hmod, name);
    } else
    {
        std::cout << "Unimplemented function _cuModuleGetGlobal\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuModuleGetGlobal_v2(CUdeviceptr *dptr, unsigned int *bytes, CUmodule hmod, const char *name)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuModuleGetGlobal_v2 proc = (ptr_cuModuleGetGlobal_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetGlobal_v2);
		return (*proc)(dptr, bytes, hmod, name);
	} else
	{
		std::cout << "Unimplemented function _cuModuleGetGlobal_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleGetSurfRef proc = (ptr_cuModuleGetSurfRef)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetSurfRef);
        return (*proc)(pSurfRef, hmod, name);   ;
    } else
    {
        std::cout << "Unimplemented function _cuModuleGetSurfRef\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleGetTexRef proc = (ptr_cuModuleGetTexRef)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleGetTexRef);
        return (*proc)(pTexRef, hmod, name);
    } else
    {
        std::cout << "Unimplemented function _cuModuleGetTexRef\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuModuleLoad(CUmodule *module, const char *fname)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleLoad proc = (ptr_cuModuleLoad)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoad);
        return (*proc)(module, fname);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cuModuleLoad(module, fname);
    }
}

CUresult CUDAAPI _CUDA::_cuModuleLoadData(CUmodule *module, const void *image)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleLoadData proc = (ptr_cuModuleLoadData)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoadData);
        return (*proc)(module, image);
    } else
    {
        std::cout << "Unimplemented function _cuModuleLoadData\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleLoadDataEx proc = (ptr_cuModuleLoadDataEx)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoadDataEx);
        return (*proc)(module, image, numOptions, options, optionValues);
    } else
    {
        std::cout << "Unimplemented function _cuModuleLoadDataEx\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleLoadFatBinary proc = (ptr_cuModuleLoadFatBinary)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleLoadFatBinary);
        return (*proc)(module, fatCubin);
    } else
    {
        std::cout << "Unimplemented function _cuModuleLoadFatBinary\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuModuleUnload(CUmodule hmod)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuModuleUnload proc = (ptr_cuModuleUnload)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuModuleUnload);
        return (*proc)(hmod);
    } else
    {
        std::cout << "Unimplemented function _cuModuleUnload\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuParamSetf    (CUfunction hfunc, int offset, float value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuParamSetf proc = (ptr_cuParamSetf)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetf);
        return (*proc)    (hfunc, offset, value);
    } else
    {
        std::cout << "Unimplemented function _cuParamSetf\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuParamSeti    (CUfunction hfunc, int offset, unsigned int value)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuParamSeti proc = (ptr_cuParamSeti)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSeti);
        return (*proc)    (hfunc, offset, value);
    } else
    {
        std::cout << "Unimplemented function _cuParamSeti\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuParamSetSize (CUfunction hfunc, unsigned int numbytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuParamSetSize proc = (ptr_cuParamSetSize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetSize);
        return (*proc)(hfunc, numbytes);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        emulator->_cuParamSetSize(hfunc, numbytes);
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI _CUDA::_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuParamSetTexRef proc = (ptr_cuParamSetTexRef)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetTexRef);
        return (*proc)(hfunc, texunit, hTexRef);
    } else
    {
        std::cout << "Unimplemented function _cuParamSetTexRef\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuParamSetv proc = (ptr_cuParamSetv)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuParamSetv);
        return (*proc)(hfunc, offset, ptr, numbytes);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        emulator->_cuParamSetv(hfunc, offset, ptr, numbytes);
        return CUDA_SUCCESS;
    }
}

CUresult CUDAAPI _CUDA::_cuStreamCreate( CUstream *phStream, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuStreamCreate proc = (ptr_cuStreamCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamCreate);
        return (*proc)( phStream, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuStreamCreate\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuStreamDestroy( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuStreamDestroy proc = (ptr_cuStreamDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamDestroy);
        return (*proc)( hStream );
    } else
    {
        std::cout << "Unimplemented function _cuStreamDestroy\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuStreamQuery( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuStreamQuery proc = (ptr_cuStreamQuery)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamQuery);
        return (*proc)( hStream );
    } else
    {
        std::cout << "Unimplemented function _cuStreamQuery\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuStreamSynchronize( CUstream hStream )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuStreamSynchronize proc = (ptr_cuStreamSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuStreamSynchronize);
        return (*proc)( hStream );
    } else
    {
        std::cout << "Unimplemented function _cuStreamSynchronize\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuSurfRefGetArray( CUarray *phArray, CUsurfref hSurfRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuSurfRefGetArray proc = (ptr_cuSurfRefGetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuSurfRefGetArray);
        return (*proc)( phArray, hSurfRef );
    } else
    {
        std::cout << "Unimplemented function _cuSurfRefGetArray\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuSurfRefSetArray( CUsurfref hSurfRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuSurfRefSetArray proc = (ptr_cuSurfRefSetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuSurfRefSetArray);
        return (*proc)( hSurfRef, hArray, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuSurfRefSetArray\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefCreate( CUtexref *pTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefCreate proc = (ptr_cuTexRefCreate)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefCreate);
        return (*proc)( pTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefCreate\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefDestroy( CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefDestroy proc = (ptr_cuTexRefDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefDestroy);
        return (*proc)( hTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefDestroy\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefGetAddress( CUdeviceptr *pdptr, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefGetAddress proc = (ptr_cuTexRefGetAddress)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetAddress);
        return (*proc)( pdptr, hTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefGetAddress\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefGetAddress_v2( CUdeviceptr *pdptr, CUtexref hTexRef )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuTexRefGetAddress_v2 proc = (ptr_cuTexRefGetAddress_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetAddress_v2);
		return (*proc)( pdptr, hTexRef );
	} else
	{
		std::cout << "Unimplemented function _cuTexRefGetAddress_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuTexRefGetAddressMode( CUaddress_mode *pam, CUtexref hTexRef, int dim )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefGetAddressMode proc = (ptr_cuTexRefGetAddressMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetAddressMode);
        return (*proc)( pam, hTexRef, dim );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefGetAddressMode\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefGetArray proc = (ptr_cuTexRefGetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetArray);
        return (*proc)( phArray, hTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefGetArray\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefGetFilterMode( CUfilter_mode *pfm, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefGetFilterMode proc = (ptr_cuTexRefGetFilterMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetFilterMode);
        return (*proc)( pfm, hTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefGetFilterMode\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefGetFlags( unsigned int *pFlags, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefGetFlags proc = (ptr_cuTexRefGetFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetFlags);
        return (*proc)( pFlags, hTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefGetFlags\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefGetFormat( CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefGetFormat proc = (ptr_cuTexRefGetFormat)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefGetFormat);
        return (*proc)( pFormat, pNumChannels, hTexRef );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefGetFormat\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetAddress proc = (ptr_cuTexRefSetAddress)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress);
        return (*proc)( ByteOffset, hTexRef, dptr, bytes );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetAddress\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress_v2( unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes )
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuTexRefSetAddress_v2 proc = (ptr_cuTexRefSetAddress_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress);
		return (*proc)( ByteOffset, hTexRef, dptr, bytes );
	} else
	{
		std::cout << "Unimplemented function _cuTexRefSetAddress_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress2D( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetAddress2D proc = (ptr_cuTexRefSetAddress2D)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress2D);
        return (*proc)( hTexRef, desc, dptr, Pitch);
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetAddress2D\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddress2D_v2( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, unsigned int Pitch)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	if (! cu->do_emulation)
	{
		ptr_cuTexRefSetAddress2D_v2 proc = (ptr_cuTexRefSetAddress2D_v2)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddress2D_v2);
		return (*proc)( hTexRef, desc, dptr, Pitch);
	} else
	{
		std::cout << "Unimplemented function _cuTexRefSetAddress2D_v2\n";
		return CUDA_ERROR_INVALID_VALUE;
	}
}

CUresult CUDAAPI _CUDA::_cuTexRefSetAddressMode( CUtexref hTexRef, int dim, CUaddress_mode am )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetAddressMode proc = (ptr_cuTexRefSetAddressMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetAddressMode);
        return (*proc)( hTexRef, dim, am );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetAddressMode\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetArray proc = (ptr_cuTexRefSetArray)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetArray);
        return (*proc)( hTexRef, hArray, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetArray\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetFilterMode( CUtexref hTexRef, CUfilter_mode fm )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetFilterMode proc = (ptr_cuTexRefSetFilterMode)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetFilterMode);
        return (*proc)( hTexRef, fm );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetFilterMode\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetFlags( CUtexref hTexRef, unsigned int Flags )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetFlags proc = (ptr_cuTexRefSetFlags)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetFlags);
        return (*proc)( hTexRef, Flags );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetFlags\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult CUDAAPI _CUDA::_cuTexRefSetFormat( CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents )
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptr_cuTexRefSetFormat proc = (ptr_cuTexRefSetFormat)cu->hook_manager->FindOriginal((PROC)_CUDA::_cuTexRefSetFormat);
        return (*proc)( hTexRef, fmt, NumPackedComponents );
    } else
    {
        std::cout << "Unimplemented function _cuTexRefSetFormat\n";
        return CUDA_ERROR_INVALID_VALUE;
    }
}
