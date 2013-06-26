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

		hook_manager->HookImport(cuda_module_name, "__cudaRegisterDeviceFunction", (PROC)_CUDA_RUNTIME::_cudaRegisterDeviceFunction, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterFatBinary", (PROC)_CUDA_RUNTIME::_cudaRegisterFatBinary, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterFunction", (PROC)_CUDA_RUNTIME::_cudaRegisterFunction, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterPrelinkedFatBinary", (PROC)_CUDA_RUNTIME::_cudaRegisterPrelinkedFatBinary, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterShared", (PROC)_CUDA_RUNTIME::_cudaRegisterShared, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterSharedVar", (PROC)_CUDA_RUNTIME::_cudaRegisterSharedVar, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterSurface", (PROC)_CUDA_RUNTIME::_cudaRegisterSurface, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterTexture", (PROC)_CUDA_RUNTIME::_cudaRegisterTexture, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaRegisterVar", (PROC)_CUDA_RUNTIME::_cudaRegisterVar, complain);
		hook_manager->HookImport(cuda_module_name, "__cudaUnregisterFatBinary", (PROC)_CUDA_RUNTIME::_cudaUnregisterFatBinary, complain);

		hook_manager->HookImport(cuda_module_name, "cudaArrayGetInfo", (PROC)_CUDA_RUNTIME::_cudaArrayGetInfo, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindSurfaceToArray", (PROC)_CUDA_RUNTIME::_cudaBindSurfaceToArray, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTexture", (PROC)_CUDA_RUNTIME::_cudaBindTexture, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTexture2D", (PROC)_CUDA_RUNTIME::_cudaBindTexture2D, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTextureToArray", (PROC)_CUDA_RUNTIME::_cudaBindTextureToArray, complain);
		hook_manager->HookImport(cuda_module_name, "cudaBindTextureToMipmappedArray", (PROC)_CUDA_RUNTIME::_cudaBindTextureToMipmappedArray, complain);

        hook_manager->HookImport(cuda_module_name, "cudaChooseDevice", (PROC)_CUDA_RUNTIME::_cudaChooseDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)_CUDA_RUNTIME::_cudaConfigureCall, complain);
		hook_manager->HookImport(cuda_module_name, "cudaCreateChannelDesc", (PROC)_CUDA_RUNTIME::_cudaCreateChannelDesc, complain);

		hook_manager->HookImport(cuda_module_name, "cudaCreateSurfaceObject", (PROC)_CUDA_RUNTIME::_cudaCreateSurfaceObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaCreateTextureObject", (PROC)_CUDA_RUNTIME::_cudaCreateTextureObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10GetDevice", (PROC)_CUDA_RUNTIME::_cudaD3D10GetDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10GetDevices", (PROC)_CUDA_RUNTIME::_cudaD3D10GetDevices, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10GetDirect3DDevice", (PROC)_CUDA_RUNTIME::_cudaD3D10GetDirect3DDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10MapResources", (PROC)_CUDA_RUNTIME::_cudaD3D10MapResources, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10RegisterResource", (PROC)_CUDA_RUNTIME::_cudaD3D10RegisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10ResourceGetMappedArray", (PROC)_CUDA_RUNTIME::_cudaD3D10ResourceGetMappedArray, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10ResourceGetMappedPitch", (PROC)_CUDA_RUNTIME::_cudaD3D10ResourceGetMappedPitch, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10ResourceGetMappedPointer", (PROC)_CUDA_RUNTIME::_cudaD3D10ResourceGetMappedPointer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10ResourceGetMappedSize", (PROC)_CUDA_RUNTIME::_cudaD3D10ResourceGetMappedSize, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10ResourceGetSurfaceDimensions", (PROC)_CUDA_RUNTIME::_cudaD3D10ResourceGetSurfaceDimensions, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10ResourceSetMapFlags", (PROC)_CUDA_RUNTIME::_cudaD3D10ResourceSetMapFlags, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10SetDirect3DDevice", (PROC)_CUDA_RUNTIME::_cudaD3D10SetDirect3DDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10UnmapResources", (PROC)_CUDA_RUNTIME::_cudaD3D10UnmapResources, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D10UnregisterResource", (PROC)_CUDA_RUNTIME::_cudaD3D10UnregisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D11GetDevice", (PROC)_CUDA_RUNTIME::_cudaD3D11GetDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D11GetDevices", (PROC)_CUDA_RUNTIME::_cudaD3D11GetDevices, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D11GetDirect3DDevice", (PROC)_CUDA_RUNTIME::_cudaD3D11GetDirect3DDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D11SetDirect3DDevice", (PROC)_CUDA_RUNTIME::_cudaD3D11SetDirect3DDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9Begin", (PROC)_CUDA_RUNTIME::_cudaD3D9Begin, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9End", (PROC)_CUDA_RUNTIME::_cudaD3D9End, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9GetDevice", (PROC)_CUDA_RUNTIME::_cudaD3D9GetDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9GetDevices", (PROC)_CUDA_RUNTIME::_cudaD3D9GetDevices, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9GetDirect3DDevice", (PROC)_CUDA_RUNTIME::_cudaD3D9GetDirect3DDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9MapResources", (PROC)_CUDA_RUNTIME::_cudaD3D9MapResources, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9MapVertexBuffer", (PROC)_CUDA_RUNTIME::_cudaD3D9MapVertexBuffer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9RegisterResource", (PROC)_CUDA_RUNTIME::_cudaD3D9RegisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9RegisterVertexBuffer", (PROC)_CUDA_RUNTIME::_cudaD3D9RegisterVertexBuffer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9ResourceGetMappedArray", (PROC)_CUDA_RUNTIME::_cudaD3D9ResourceGetMappedArray, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9ResourceGetMappedPitch", (PROC)_CUDA_RUNTIME::_cudaD3D9ResourceGetMappedPitch, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9ResourceGetMappedPointer", (PROC)_CUDA_RUNTIME::_cudaD3D9ResourceGetMappedPointer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9ResourceGetMappedSize", (PROC)_CUDA_RUNTIME::_cudaD3D9ResourceGetMappedSize, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9ResourceGetSurfaceDimensions", (PROC)_CUDA_RUNTIME::_cudaD3D9ResourceGetSurfaceDimensions, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9ResourceSetMapFlags", (PROC)_CUDA_RUNTIME::_cudaD3D9ResourceSetMapFlags, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9SetDirect3DDevice", (PROC)_CUDA_RUNTIME::_cudaD3D9SetDirect3DDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9UnmapResources", (PROC)_CUDA_RUNTIME::_cudaD3D9UnmapResources, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9UnmapVertexBuffer", (PROC)_CUDA_RUNTIME::_cudaD3D9UnmapVertexBuffer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9UnregisterResource", (PROC)_CUDA_RUNTIME::_cudaD3D9UnregisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaD3D9UnregisterVertexBuffer", (PROC)_CUDA_RUNTIME::_cudaD3D9UnregisterVertexBuffer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDestroySurfaceObject", (PROC)_CUDA_RUNTIME::_cudaDestroySurfaceObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDestroyTextureObject", (PROC)_CUDA_RUNTIME::_cudaDestroyTextureObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceCanAccessPeer", (PROC)_CUDA_RUNTIME::_cudaDeviceCanAccessPeer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceDisablePeerAccess", (PROC)_CUDA_RUNTIME::_cudaDeviceDisablePeerAccess, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceEnablePeerAccess", (PROC)_CUDA_RUNTIME::_cudaDeviceEnablePeerAccess, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetAttribute", (PROC)_CUDA_RUNTIME::_cudaDeviceGetAttribute, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetByPCIBusId", (PROC)_CUDA_RUNTIME::_cudaDeviceGetByPCIBusId, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetCacheConfig", (PROC)_CUDA_RUNTIME::_cudaDeviceGetCacheConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetLimit", (PROC)_CUDA_RUNTIME::_cudaDeviceGetLimit, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetPCIBusId", (PROC)_CUDA_RUNTIME::_cudaDeviceGetPCIBusId, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetSharedMemConfig", (PROC)_CUDA_RUNTIME::_cudaDeviceGetSharedMemConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceGetStreamPriorityRange", (PROC)_CUDA_RUNTIME::_cudaDeviceGetStreamPriorityRange, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceReset", (PROC)_CUDA_RUNTIME::_cudaDeviceReset, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceSetCacheConfig", (PROC)_CUDA_RUNTIME::_cudaDeviceSetCacheConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceSetLimit", (PROC)_CUDA_RUNTIME::_cudaDeviceSetLimit, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceSetSharedMemConfig", (PROC)_CUDA_RUNTIME::_cudaDeviceSetSharedMemConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaDeviceSynchronize", (PROC)_CUDA_RUNTIME::_cudaDeviceSynchronize, complain);

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
		hook_manager->HookImport(cuda_module_name, "cudaFreeMipmappedArray", (PROC)_CUDA_RUNTIME::_cudaFreeMipmappedArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFuncGetAttributes", (PROC)_CUDA_RUNTIME::_cudaFuncGetAttributes, complain);
		hook_manager->HookImport(cuda_module_name, "cudaFuncSetCacheConfig", (PROC)_CUDA_RUNTIME::_cudaFuncSetCacheConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaFuncSetSharedMemConfig", (PROC)_CUDA_RUNTIME::_cudaFuncSetSharedMemConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetChannelDesc", (PROC)_CUDA_RUNTIME::_cudaGetChannelDesc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDevice", (PROC)_CUDA_RUNTIME::_cudaGetDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceCount", (PROC)_CUDA_RUNTIME::_cudaGetDeviceCount, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceProperties", (PROC)_CUDA_RUNTIME::_cudaGetDeviceProperties, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetErrorString", (PROC)_CUDA_RUNTIME::_cudaGetErrorString, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetExportTable", (PROC)_CUDA_RUNTIME::_cudaGetExportTable, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)_CUDA_RUNTIME::_cudaGetLastError, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetMipmappedArrayLevel", (PROC)_CUDA_RUNTIME::_cudaGetMipmappedArrayLevel, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceObjectResourceDesc", (PROC)_CUDA_RUNTIME::_cudaGetSurfaceObjectResourceDesc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceReference", (PROC)_CUDA_RUNTIME::_cudaGetSurfaceReference, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolAddress", (PROC)_CUDA_RUNTIME::_cudaGetSymbolAddress, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolSize", (PROC)_CUDA_RUNTIME::_cudaGetSymbolSize, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetTextureAlignmentOffset", (PROC)_CUDA_RUNTIME::_cudaGetTextureAlignmentOffset, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetTextureObjectResourceDesc", (PROC)_CUDA_RUNTIME::_cudaGetTextureObjectResourceDesc, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetTextureObjectResourceViewDesc", (PROC)_CUDA_RUNTIME::_cudaGetTextureObjectResourceViewDesc, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetTextureObjectTextureDesc", (PROC)_CUDA_RUNTIME::_cudaGetTextureObjectTextureDesc, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGetTextureReference", (PROC)_CUDA_RUNTIME::_cudaGetTextureReference, complain);

		hook_manager->HookImport(cuda_module_name, "cudaGLGetDevices", (PROC)_CUDA_RUNTIME::_cudaGLGetDevices, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLMapBufferObject", (PROC)_CUDA_RUNTIME::_cudaGLMapBufferObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLMapBufferObjectAsync", (PROC)_CUDA_RUNTIME::_cudaGLMapBufferObjectAsync, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLRegisterBufferObject", (PROC)_CUDA_RUNTIME::_cudaGLRegisterBufferObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLSetBufferObjectMapFlags", (PROC)_CUDA_RUNTIME::_cudaGLSetBufferObjectMapFlags, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLSetGLDevice", (PROC)_CUDA_RUNTIME::_cudaGLSetGLDevice, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLUnmapBufferObject", (PROC)_CUDA_RUNTIME::_cudaGLUnmapBufferObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLUnmapBufferObjectAsync", (PROC)_CUDA_RUNTIME::_cudaGLUnmapBufferObjectAsync, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGLUnregisterBufferObject", (PROC)_CUDA_RUNTIME::_cudaGLUnregisterBufferObject, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGraphicsD3D10RegisterResource", (PROC)_CUDA_RUNTIME::_cudaGraphicsD3D10RegisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGraphicsD3D11RegisterResource", (PROC)_CUDA_RUNTIME::_cudaGraphicsD3D11RegisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGraphicsD3D9RegisterResource", (PROC)_CUDA_RUNTIME::_cudaGraphicsD3D9RegisterResource, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGraphicsGLRegisterBuffer", (PROC)_CUDA_RUNTIME::_cudaGraphicsGLRegisterBuffer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaGraphicsGLRegisterImage", (PROC)_CUDA_RUNTIME::_cudaGraphicsGLRegisterImage, complain);

        hook_manager->HookImport(cuda_module_name, "cudaGraphicsMapResources", (PROC)_CUDA_RUNTIME::_cudaGraphicsMapResources, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceGetMappedPointer", (PROC)_CUDA_RUNTIME::_cudaGraphicsResourceGetMappedPointer, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceSetMapFlags", (PROC)_CUDA_RUNTIME::_cudaGraphicsResourceSetMapFlags, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsSubResourceGetMappedArray", (PROC)_CUDA_RUNTIME::_cudaGraphicsSubResourceGetMappedArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnmapResources", (PROC)_CUDA_RUNTIME::_cudaGraphicsUnmapResources, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnregisterResource", (PROC)_CUDA_RUNTIME::_cudaGraphicsUnregisterResource, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)_CUDA_RUNTIME::_cudaHostAlloc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)_CUDA_RUNTIME::_cudaHostGetDevicePointer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaHostGetFlags", (PROC)_CUDA_RUNTIME::_cudaHostGetFlags, complain);

		hook_manager->HookImport(cuda_module_name, "cudaHostRegister", (PROC)_CUDA_RUNTIME::_cudaHostRegister, complain);
		hook_manager->HookImport(cuda_module_name, "cudaHostUnregister", (PROC)_CUDA_RUNTIME::_cudaHostUnregister, complain);
		hook_manager->HookImport(cuda_module_name, "cudaIpcCloseMemHandle", (PROC)_CUDA_RUNTIME::_cudaIpcCloseMemHandle, complain);
		hook_manager->HookImport(cuda_module_name, "cudaIpcGetEventHandle", (PROC)_CUDA_RUNTIME::_cudaIpcGetEventHandle, complain);
		hook_manager->HookImport(cuda_module_name, "cudaIpcGetMemHandle", (PROC)_CUDA_RUNTIME::_cudaIpcGetMemHandle, complain);
		hook_manager->HookImport(cuda_module_name, "cudaIpcOpenEventHandle", (PROC)_CUDA_RUNTIME::_cudaIpcOpenEventHandle, complain);
		hook_manager->HookImport(cuda_module_name, "cudaIpcOpenMemHandle", (PROC)_CUDA_RUNTIME::_cudaIpcOpenMemHandle, complain);

        hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)_CUDA_RUNTIME::_cudaLaunch, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)_CUDA_RUNTIME::_cudaMalloc, complain);

		hook_manager->HookImport(cuda_module_name, "cudaMalloc3D", (PROC)_CUDA_RUNTIME::_cudaMalloc3D, complain);

        hook_manager->HookImport(cuda_module_name, "cudaMalloc3DArray", (PROC)_CUDA_RUNTIME::_cudaMalloc3DArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocArray", (PROC)_CUDA_RUNTIME::_cudaMallocArray, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMallocHost", (PROC)_CUDA_RUNTIME::_cudaMallocHost, complain);

		hook_manager->HookImport(cuda_module_name, "cudaMallocMipmappedArray", (PROC)_CUDA_RUNTIME::_cudaMallocMipmappedArray, complain);

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

		hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DPeer", (PROC)_CUDA_RUNTIME::_cudaMemcpy3DPeer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DPeerAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpy3DPeerAsync, complain);

        hook_manager->HookImport(cuda_module_name, "cudaMemcpyArrayToArray", (PROC)_CUDA_RUNTIME::_cudaMemcpyArrayToArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArray", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArrayAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromArrayAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbol", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromSymbol, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbolAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyFromSymbolAsync, complain);

		hook_manager->HookImport(cuda_module_name, "cudaMemcpyPeer", (PROC)_CUDA_RUNTIME::_cudaMemcpyPeer, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMemcpyPeerAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyPeerAsync, complain);

        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArray", (PROC)_CUDA_RUNTIME::_cudaMemcpyToArray, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArrayAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyToArrayAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbol", (PROC)_CUDA_RUNTIME::_cudaMemcpyToSymbol, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbolAsync", (PROC)_CUDA_RUNTIME::_cudaMemcpyToSymbolAsync, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemGetInfo", (PROC)_CUDA_RUNTIME::_cudaMemGetInfo, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)_CUDA_RUNTIME::_cudaMemset, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMemset2D", (PROC)_CUDA_RUNTIME::_cudaMemset2D, complain);

		hook_manager->HookImport(cuda_module_name, "cudaMemset2DAsync", (PROC)_CUDA_RUNTIME::_cudaMemset2DAsync, complain);

		hook_manager->HookImport(cuda_module_name, "cudaMemset3D", (PROC)_CUDA_RUNTIME::_cudaMemset3D, complain);

		hook_manager->HookImport(cuda_module_name, "cudaMemset3DAsync", (PROC)_CUDA_RUNTIME::_cudaMemset3DAsync, complain);
		hook_manager->HookImport(cuda_module_name, "cudaMemsetAsync", (PROC)_CUDA_RUNTIME::_cudaMemsetAsync, complain);

		hook_manager->HookImport(cuda_module_name, "cudaPeekAtLastError", (PROC)_CUDA_RUNTIME::_cudaPeekAtLastError, complain);

		hook_manager->HookImport(cuda_module_name, "cudaPointerGetAttributes", (PROC)_CUDA_RUNTIME::_cudaPointerGetAttributes, complain);
		hook_manager->HookImport(cuda_module_name, "cudaProfilerStart", (PROC)_CUDA_RUNTIME::_cudaProfilerStart, complain);
		hook_manager->HookImport(cuda_module_name, "cudaProfilerStop", (PROC)_CUDA_RUNTIME::_cudaProfilerStop, complain);
		hook_manager->HookImport(cuda_module_name, "cudaRuntimeGetVersion", (PROC)_CUDA_RUNTIME::_cudaRuntimeGetVersion, complain);

        hook_manager->HookImport(cuda_module_name, "cudaSetDevice", (PROC)_CUDA_RUNTIME::_cudaSetDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDeviceFlags", (PROC)_CUDA_RUNTIME::_cudaSetDeviceFlags, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForDevice", (PROC)_CUDA_RUNTIME::_cudaSetDoubleForDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForHost", (PROC)_CUDA_RUNTIME::_cudaSetDoubleForHost, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)_CUDA_RUNTIME::_cudaSetupArgument, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetValidDevices", (PROC)_CUDA_RUNTIME::_cudaSetValidDevices, complain);

		hook_manager->HookImport(cuda_module_name, "cudaStreamAddCallback", (PROC)_CUDA_RUNTIME::_cudaStreamAddCallback, complain);

		hook_manager->HookImport(cuda_module_name, "cudaStreamCreate", (PROC)_CUDA_RUNTIME::_cudaStreamCreate, complain);

		hook_manager->HookImport(cuda_module_name, "cudaStreamCreateWithFlags", (PROC)_CUDA_RUNTIME::_cudaStreamCreateWithFlags, complain);
		hook_manager->HookImport(cuda_module_name, "cudaStreamCreateWithPriority", (PROC)_CUDA_RUNTIME::_cudaStreamCreateWithPriority, complain);

		hook_manager->HookImport(cuda_module_name, "cudaStreamDestroy", (PROC)_CUDA_RUNTIME::_cudaStreamDestroy, complain);
		hook_manager->HookImport(cuda_module_name, "cudaStreamGetFlags", (PROC)_CUDA_RUNTIME::_cudaStreamGetFlags, complain);
		hook_manager->HookImport(cuda_module_name, "cudaStreamGetPriority", (PROC)_CUDA_RUNTIME::_cudaStreamGetPriority, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamQuery", (PROC)_CUDA_RUNTIME::_cudaStreamQuery, complain);
		hook_manager->HookImport(cuda_module_name, "cudaStreamSynchronize", (PROC)_CUDA_RUNTIME::_cudaStreamSynchronize, complain);

		hook_manager->HookImport(cuda_module_name, "cudaStreamWaitEvent", (PROC)_CUDA_RUNTIME::_cudaStreamWaitEvent, complain);

		hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)_CUDA_RUNTIME::_cudaThreadExit, complain);

		hook_manager->HookImport(cuda_module_name, "cudaThreadGetCacheConfig", (PROC)_CUDA_RUNTIME::_cudaThreadGetCacheConfig, complain);

		hook_manager->HookImport(cuda_module_name, "cudaThreadGetLimit", (PROC)_CUDA_RUNTIME::_cudaThreadGetLimit, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSetCacheConfig", (PROC)_CUDA_RUNTIME::_cudaThreadSetCacheConfig, complain);
		hook_manager->HookImport(cuda_module_name, "cudaThreadSetLimit", (PROC)_CUDA_RUNTIME::_cudaThreadSetLimit, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSynchronize", (PROC)_CUDA_RUNTIME::_cudaThreadSynchronize, complain);
		hook_manager->HookImport(cuda_module_name, "cudaUnbindTexture", (PROC)_CUDA_RUNTIME::_cudaUnbindTexture, complain);

		hook_manager->HookImport(cuda_module_name, "cudaWGLGetDevice", (PROC)_CUDA_RUNTIME::_cudaWGLGetDevice, complain);
		hook_manager->HookImport(cuda_module_name, "NvOptimusEnablementCuda", (PROC)_CUDA_RUNTIME::_NvOptimusEnablementCuda, complain);

    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaArrayGetInfo()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaArrayGetInfo();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaBindTextureToMipmappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaBindTextureToMipmappedArray();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaCreateSurfaceObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaCreateSurfaceObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaCreateTextureObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaCreateTextureObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10GetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10GetDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10GetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10GetDevices();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10GetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10GetDirect3DDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10MapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10MapResources();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10RegisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10ResourceGetMappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10ResourceGetMappedArray();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10ResourceGetMappedPitch()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10ResourceGetMappedPitch();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10ResourceGetMappedPointer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10ResourceGetMappedPointer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10ResourceGetMappedSize()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10ResourceGetMappedSize();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10ResourceGetSurfaceDimensions()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10ResourceGetSurfaceDimensions();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10ResourceSetMapFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10ResourceSetMapFlags();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10SetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10SetDirect3DDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10UnmapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10UnmapResources();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D10UnregisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D10UnregisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D11GetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D11GetDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D11GetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D11GetDevices();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D11GetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D11GetDirect3DDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D11SetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D11SetDirect3DDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9Begin()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9Begin();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9End()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9End();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9GetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9GetDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9GetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9GetDevices();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9GetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9GetDirect3DDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9MapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9MapResources();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9MapVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9MapVertexBuffer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9RegisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9RegisterVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9RegisterVertexBuffer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9ResourceGetMappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9ResourceGetMappedArray();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9ResourceGetMappedPitch()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9ResourceGetMappedPitch();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9ResourceGetMappedPointer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9ResourceGetMappedPointer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9ResourceGetMappedSize()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9ResourceGetMappedSize();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9ResourceGetSurfaceDimensions()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9ResourceGetSurfaceDimensions();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9ResourceSetMapFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9ResourceSetMapFlags();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9SetDirect3DDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9SetDirect3DDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9UnmapResources()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9UnmapResources();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9UnmapVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9UnmapVertexBuffer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9UnregisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9UnregisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaD3D9UnregisterVertexBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaD3D9UnregisterVertexBuffer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDestroySurfaceObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDestroySurfaceObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDestroyTextureObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDestroyTextureObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceCanAccessPeer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceCanAccessPeer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceDisablePeerAccess()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceDisablePeerAccess();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceEnablePeerAccess()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceEnablePeerAccess();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetAttribute()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetAttribute();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetByPCIBusId()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetByPCIBusId();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetCacheConfig();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetLimit()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetLimit();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetPCIBusId()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetPCIBusId();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetSharedMemConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetSharedMemConfig();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceGetStreamPriorityRange()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceGetStreamPriorityRange();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceReset()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceReset();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceSetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceSetCacheConfig();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceSetLimit()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceSetLimit();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceSetSharedMemConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceSetSharedMemConfig();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaDeviceSynchronize()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaDeviceSynchronize();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFreeMipmappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFreeMipmappedArray();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaFuncSetSharedMemConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaFuncSetSharedMemConfig();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetMipmappedArrayLevel()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetMipmappedArrayLevel();
}

cudaError_t CUDARTAPI _cudaGetSurfaceAlignmentOffset(size_t *offset, const struct surfaceReference *surfref)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetSurfaceAlignmentOffset(offset, surfref);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetSurfaceObjectResourceDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetSurfaceObjectResourceDesc();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetTextureObjectResourceDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetTextureObjectResourceDesc();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetTextureObjectResourceViewDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetTextureObjectResourceViewDesc();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetTextureObjectTextureDesc()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetTextureObjectTextureDesc();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetTextureReference(const struct textureReference **texref, const char *symbol)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGetTextureReference(texref, symbol);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLGetDevices()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLGetDevices();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLMapBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLMapBufferObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLMapBufferObjectAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLMapBufferObjectAsync();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLRegisterBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLRegisterBufferObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLSetBufferObjectMapFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLSetBufferObjectMapFlags();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLSetGLDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLSetGLDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLUnmapBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLUnmapBufferObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLUnmapBufferObjectAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLUnmapBufferObjectAsync();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGLUnregisterBufferObject()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGLUnregisterBufferObject();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsD3D10RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsD3D10RegisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsD3D11RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsD3D11RegisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsD3D9RegisterResource()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsD3D9RegisterResource();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsGLRegisterBuffer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsGLRegisterBuffer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGraphicsGLRegisterImage()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaGraphicsGLRegisterImage();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaHostRegister()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaHostRegister();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaHostUnregister()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaHostUnregister();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaIpcCloseMemHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaIpcCloseMemHandle();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaIpcGetEventHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaIpcGetEventHandle();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaIpcGetMemHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaIpcGetMemHandle();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaIpcOpenEventHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaIpcOpenEventHandle();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaIpcOpenMemHandle()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaIpcOpenMemHandle();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMalloc3D()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMalloc3D();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMallocMipmappedArray()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMallocMipmappedArray();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy3DPeer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy3DPeer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpy3DPeerAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpy3DPeerAsync();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyPeer()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyPeer();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemcpyPeerAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemcpyPeerAsync();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemsetAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemsetAsync();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset2D(devPtr, pitch, value, width, height);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset2DAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset2DAsync();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaMemset3DAsync()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaMemset3DAsync();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaPeekAtLastError(void)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaPeekAtLastError();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaPointerGetAttributes()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaPointerGetAttributes();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaProfilerStart()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaProfilerStart();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaProfilerStop()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaProfilerStop();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaRegisterDeviceFunction()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterDeviceFunction();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaRegisterPrelinkedFatBinary()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterPrelinkedFatBinary();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaRegisterShared()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterShared();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaRegisterSharedVar()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaRegisterSharedVar();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamAddCallback()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamAddCallback();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamCreate(cudaStream_t *pStream)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamCreate(pStream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamCreateWithFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamCreateWithFlags();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamCreateWithPriority()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamCreateWithPriority();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamDestroy(cudaStream_t stream)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamDestroy(stream);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamGetFlags()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamGetFlags();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamGetPriority()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamGetPriority();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamWaitEvent()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaStreamWaitEvent();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadExit()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadExit();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadGetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadGetCacheConfig();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadGetLimit(enum cudaLimit limit, size_t value)
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadGetLimit(limit, value);
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadSetCacheConfig()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaThreadSetCacheConfig();
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

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaWGLGetDevice()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_cudaWGLGetDevice();
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_NvOptimusEnablementCuda()
{
	CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
	return cu->CurrentDevice()->_NvOptimusEnablementCuda();
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
