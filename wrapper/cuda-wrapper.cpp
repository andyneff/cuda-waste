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
#include <atlcomcli.h> // For CComPtr
#include <dia2.h>
#include "call-stack-info.h"
#include "hook-mgr.h"
#include <__cudaFatFormat.h>
#include "../emulator/emulator.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pointers to functions in the CUDA library, unaltered.
//////////////////////////////////////////////////////////////////////////////////////////////////////////
CUDA_WRAPPER::typePtrCudaMalloc3D CUDA_WRAPPER::ptrCudaMalloc3D;
CUDA_WRAPPER::typePtrCudaMalloc3DArray CUDA_WRAPPER::ptrCudaMalloc3DArray;
CUDA_WRAPPER::typePtrCudaMemset3D CUDA_WRAPPER::ptrCudaMemset3D;
CUDA_WRAPPER::typePtrCudaMemcpy3D CUDA_WRAPPER::ptrCudaMemcpy3D;
CUDA_WRAPPER::typePtrCudaMemcpy3DAsync CUDA_WRAPPER::ptrCudaMemcpy3DAsync;
CUDA_WRAPPER::typePtrCudaMalloc CUDA_WRAPPER::ptrCudaMalloc;
CUDA_WRAPPER::typePtrCudaMallocHost CUDA_WRAPPER::ptrCudaMallocHost;
CUDA_WRAPPER::typePtrCudaMallocPitch CUDA_WRAPPER::ptrCudaMallocPitch;
CUDA_WRAPPER::typePtrCudaMallocArray CUDA_WRAPPER::ptrCudaMallocArray;
CUDA_WRAPPER::typePtrCudaFree CUDA_WRAPPER::ptrCudaFree;
CUDA_WRAPPER::typePtrCudaFreeHost CUDA_WRAPPER::ptrCudaFreeHost;
CUDA_WRAPPER::typePtrCudaFreeArray CUDA_WRAPPER::ptrCudaFreeArray;
CUDA_WRAPPER::typePtrCudaHostAlloc CUDA_WRAPPER::ptrCudaHostAlloc;
CUDA_WRAPPER::typePtrCudaHostGetDevicePointer CUDA_WRAPPER::ptrCudaHostGetDevicePointer;
CUDA_WRAPPER::typePtrCudaHostGetFlags CUDA_WRAPPER::ptrCudaHostGetFlags;
CUDA_WRAPPER::typePtrCudaMemGetInfo CUDA_WRAPPER::ptrCudaMemGetInfo;
CUDA_WRAPPER::typePtrCudaMemcpy CUDA_WRAPPER::ptrCudaMemcpy;
CUDA_WRAPPER::typePtrCudaMemcpyToArray CUDA_WRAPPER::ptrCudaMemcpyToArray;
CUDA_WRAPPER::typePtrCudaMemcpyFromArray CUDA_WRAPPER::ptrCudaMemcpyFromArray;
CUDA_WRAPPER::typePtrCudaMemcpyArrayToArray CUDA_WRAPPER::ptrCudaMemcpyArrayToArray;
CUDA_WRAPPER::typePtrCudaMemcpy2D CUDA_WRAPPER::ptrCudaMemcpy2D;
CUDA_WRAPPER::typePtrCudaMemcpy2DToArray CUDA_WRAPPER::ptrCudaMemcpy2DToArray;
CUDA_WRAPPER::typePtrCudaMemcpy2DFromArray CUDA_WRAPPER::ptrCudaMemcpy2DFromArray;
CUDA_WRAPPER::typePtrCudaMemcpy2DArrayToArray CUDA_WRAPPER::ptrCudaMemcpy2DArrayToArray;
CUDA_WRAPPER::typePtrCudaMemcpyToSymbol CUDA_WRAPPER::ptrCudaMemcpyToSymbol;
CUDA_WRAPPER::typePtrCudaMemcpyFromSymbol CUDA_WRAPPER::ptrCudaMemcpyFromSymbol;
CUDA_WRAPPER::typePtrCudaMemcpyAsync CUDA_WRAPPER::ptrCudaMemcpyAsync;
CUDA_WRAPPER::typePtrCudaMemcpyToArrayAsync CUDA_WRAPPER::ptrCudaMemcpyToArrayAsync;
CUDA_WRAPPER::typePtrCudaMemcpyFromArrayAsync CUDA_WRAPPER::ptrCudaMemcpyFromArrayAsync;
CUDA_WRAPPER::typePtrCudaMemcpy2DAsync CUDA_WRAPPER::ptrCudaMemcpy2DAsync;
CUDA_WRAPPER::typePtrCudaMemcpy2DToArrayAsync CUDA_WRAPPER::ptrCudaMemcpy2DToArrayAsync;
CUDA_WRAPPER::typePtrCudaMemcpy2DFromArrayAsync CUDA_WRAPPER::ptrCudaMemcpy2DFromArrayAsync;
CUDA_WRAPPER::typePtrCudaMemcpyToSymbolAsync CUDA_WRAPPER::ptrCudaMemcpyToSymbolAsync;
CUDA_WRAPPER::typePtrCudaMemcpyFromSymbolAsync CUDA_WRAPPER::ptrCudaMemcpyFromSymbolAsync;
CUDA_WRAPPER::typePtrCudaMemset CUDA_WRAPPER::ptrCudaMemset;
CUDA_WRAPPER::typePtrCudaMemset2D CUDA_WRAPPER::ptrCudaMemset2D;
CUDA_WRAPPER::typePtrCudaGetSymbolAddress CUDA_WRAPPER::ptrCudaGetSymbolAddress;
CUDA_WRAPPER::typePtrCudaGetSymbolSize CUDA_WRAPPER::ptrCudaGetSymbolSize;
CUDA_WRAPPER::typePtrCudaGetDeviceCount CUDA_WRAPPER::ptrCudaGetDeviceCount;
CUDA_WRAPPER::typePtrCudaGetDeviceProperties CUDA_WRAPPER::ptrCudaGetDeviceProperties;
CUDA_WRAPPER::typePtrCudaChooseDevice CUDA_WRAPPER::ptrCudaChooseDevice;
CUDA_WRAPPER::typePtrCudaSetDevice CUDA_WRAPPER::ptrCudaSetDevice;
CUDA_WRAPPER::typePtrCudaGetDevice CUDA_WRAPPER::ptrCudaGetDevice;
CUDA_WRAPPER::typePtrCudaSetValidDevices CUDA_WRAPPER::ptrCudaSetValidDevices;
CUDA_WRAPPER::typePtrCudaSetDeviceFlags CUDA_WRAPPER::ptrCudaSetDeviceFlags;
CUDA_WRAPPER::typePtrCudaBindTexture CUDA_WRAPPER::ptrCudaBindTexture;
CUDA_WRAPPER::typePtrCudaBindTexture2D CUDA_WRAPPER::ptrCudaBindTexture2D;
CUDA_WRAPPER::typePtrCudaBindTextureToArray CUDA_WRAPPER::ptrCudaBindTextureToArray;
CUDA_WRAPPER::typePtrCudaUnbindTexture CUDA_WRAPPER::ptrCudaUnbindTexture;
CUDA_WRAPPER::typePtrCudaGetTextureAlignmentOffset CUDA_WRAPPER::ptrCudaGetTextureAlignmentOffset;
CUDA_WRAPPER::typePtrCudaGetTextureReference CUDA_WRAPPER::ptrCudaGetTextureReference;
CUDA_WRAPPER::typePtrCudaBindSurfaceToArray CUDA_WRAPPER::ptrCudaBindSurfaceToArray;
CUDA_WRAPPER::typePtrCudaGetSurfaceAlignmentOffset CUDA_WRAPPER::ptrCudaGetSurfaceAlignmentOffset;
CUDA_WRAPPER::typePtrCudaGetSurfaceReference CUDA_WRAPPER::ptrCudaGetSurfaceReference;
CUDA_WRAPPER::typePtrCudaGetChannelDesc CUDA_WRAPPER::ptrCudaGetChannelDesc;
CUDA_WRAPPER::typePtrCudaCreateChannelDesc CUDA_WRAPPER::ptrCudaCreateChannelDesc;
CUDA_WRAPPER::typePtrCudaGetLastError CUDA_WRAPPER::ptrCudaGetLastError;
CUDA_WRAPPER::typePtrCudaPeekAtLastError CUDA_WRAPPER::ptrCudaPeekAtLastError;
CUDA_WRAPPER::typePtrCudaGetErrorString CUDA_WRAPPER::ptrCudaGetErrorString;
CUDA_WRAPPER::typePtrCudaConfigureCall CUDA_WRAPPER::ptrCudaConfigureCall;
CUDA_WRAPPER::typePtrCudaSetupArgument CUDA_WRAPPER::ptrCudaSetupArgument;
CUDA_WRAPPER::typePtrCudaFuncSetCacheConfig CUDA_WRAPPER::ptrCudaFuncSetCacheConfig;
CUDA_WRAPPER::typePtrCudaLaunch CUDA_WRAPPER::ptrCudaLaunch;
CUDA_WRAPPER::typePtrCudaFuncGetAttributes CUDA_WRAPPER::ptrCudaFuncGetAttributes;
CUDA_WRAPPER::typePtrCudaStreamCreate CUDA_WRAPPER::ptrCudaStreamCreate;
CUDA_WRAPPER::typePtrCudaStreamDestroy CUDA_WRAPPER::ptrCudaStreamDestroy;
CUDA_WRAPPER::typePtrCudaStreamSynchronize CUDA_WRAPPER::ptrCudaStreamSynchronize;
CUDA_WRAPPER::typePtrCudaStreamQuery CUDA_WRAPPER::ptrCudaStreamQuery;
CUDA_WRAPPER::typePtrCudaEventCreate CUDA_WRAPPER::ptrCudaEventCreate;
CUDA_WRAPPER::typePtrCudaEventCreateWithFlags CUDA_WRAPPER::ptrCudaEventCreateWithFlags;
CUDA_WRAPPER::typePtrCudaEventRecord CUDA_WRAPPER::ptrCudaEventRecord;
CUDA_WRAPPER::typePtrCudaEventQuery CUDA_WRAPPER::ptrCudaEventQuery;
CUDA_WRAPPER::typePtrCudaEventSynchronize CUDA_WRAPPER::ptrCudaEventSynchronize;
CUDA_WRAPPER::typePtrCudaEventDestroy CUDA_WRAPPER::ptrCudaEventDestroy;
CUDA_WRAPPER::typePtrCudaEventElapsedTime CUDA_WRAPPER::ptrCudaEventElapsedTime;
CUDA_WRAPPER::typePtrCudaSetDoubleForDevice CUDA_WRAPPER::ptrCudaSetDoubleForDevice;
CUDA_WRAPPER::typePtrCudaSetDoubleForHost CUDA_WRAPPER::ptrCudaSetDoubleForHost;
CUDA_WRAPPER::typePtrCudaThreadExit CUDA_WRAPPER::ptrCudaThreadExit;
CUDA_WRAPPER::typePtrCudaThreadSynchronize CUDA_WRAPPER::ptrCudaThreadSynchronize;
CUDA_WRAPPER::typePtrCudaThreadSetLimit CUDA_WRAPPER::ptrCudaThreadSetLimit;
CUDA_WRAPPER::typePtrCudaThreadGetLimit CUDA_WRAPPER::ptrCudaThreadGetLimit;
CUDA_WRAPPER::typePtrCudaDriverGetVersion CUDA_WRAPPER::ptrCudaDriverGetVersion;
CUDA_WRAPPER::typePtrCudaRuntimeGetVersion CUDA_WRAPPER::ptrCudaRuntimeGetVersion;
CUDA_WRAPPER::typePtrCudaGetExportTable CUDA_WRAPPER::ptrCudaGetExportTable;
CUDA_WRAPPER::typePtrCudaGraphicsUnregisterResource CUDA_WRAPPER::ptrCudaGraphicsUnregisterResource;
CUDA_WRAPPER::typePtrCudaGraphicsResourceSetMapFlags CUDA_WRAPPER::ptrCudaGraphicsResourceSetMapFlags;
CUDA_WRAPPER::typePtrCudaGraphicsMapResources CUDA_WRAPPER::ptrCudaGraphicsMapResources;
CUDA_WRAPPER::typePtrCudaGraphicsUnmapResources CUDA_WRAPPER::ptrCudaGraphicsUnmapResources;
CUDA_WRAPPER::typePtrCudaGraphicsResourceGetMappedPointer CUDA_WRAPPER::ptrCudaGraphicsResourceGetMappedPointer;
CUDA_WRAPPER::typePtrCudaGraphicsSubResourceGetMappedArray CUDA_WRAPPER::ptrCudaGraphicsSubResourceGetMappedArray;
CUDA_WRAPPER::typePtrCudaRegisterFatBinary CUDA_WRAPPER::ptrCudaRegisterFatBinary;
CUDA_WRAPPER::typePtrCudaUnregisterFatBinary CUDA_WRAPPER::ptrCudaUnregisterFatBinary;
CUDA_WRAPPER::typePtrCudaRegisterVar CUDA_WRAPPER::ptrCudaRegisterVar;
CUDA_WRAPPER::typePtrCudaRegisterTexture CUDA_WRAPPER::ptrCudaRegisterTexture;
CUDA_WRAPPER::typePtrCudaRegisterSurface CUDA_WRAPPER::ptrCudaRegisterSurface;
CUDA_WRAPPER::typePtrCudaRegisterFunction CUDA_WRAPPER::ptrCudaRegisterFunction;


static char temp_buffer[50000];

CUDA_WRAPPER * CUDA_WRAPPER::singleton;
typedef cudaError_t (*cfunc)(void **devPtr, size_t size);
cudaError_t (*ptrCudaMalloc)(void **devPtr, size_t size);


CUDA_WRAPPER * CUDA_WRAPPER::Singleton()
{
    if (singleton)
        return singleton;
    singleton = new CUDA_WRAPPER();
    return singleton;
}

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


CUDA_WRAPPER::CUDA_WRAPPER()
{
    CUDA_WRAPPER * cu = this;
    cu->padding_size = 32;
    cu->padding_byte = 0xde;
    cu->device_pointer_to_first_byte_in_block = false;
    cu->do_not_call_cuda_after_sanity_check_fail = false;
    cu->output_stream = & std::cout;
    cu->trace_all_calls = false;
    cu->quit_on_error = false;
    cu->do_emulation = true;
    cu->global_context = 0;
    cu->hook_manager = 0;
    cu->do_crash = false;
}

void CUDA_WRAPPER::Unimplemented()
{
    std::cout << "FATAL ERROR: Unimplemented function!!!!\n";
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    exit(1);
}

bool CUDA_WRAPPER::DoInit(char * cuda_module_name, HookManager * hm)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CallStackInfo * csi = CallStackInfo::Singleton();
    csi->ClassifyAsPrefix("call-stack-info.cpp");
    csi->ClassifyAsPrefix("cuda-memory-debug.cpp");
    csi->ClassifyAsPrefix("ptr-cuda-runtime.cpp");

    if (atexit(CUDA_WRAPPER::ExitHandler))
    {
        char * context = cu->Context(3);
        (*cu->output_stream) << "Cannot register CUDA_WRAPPER exit handler during initialization.  Memory leaks will not be detected.\n";
        (*cu->output_stream) << " Call stack is:\n"
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
    }

    cu->hook_manager = hm;
    cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)CUDA_WRAPPER::Malloc);
    cu->hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)CUDA_WRAPPER::HostAlloc);
    cu->hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)CUDA_WRAPPER::Free);
    cu->hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)CUDA_WRAPPER::FreeHost);
    cu->hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)CUDA_WRAPPER::HostGetDevicePointer);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)CUDA_WRAPPER::Memcpy);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)CUDA_WRAPPER::Memset);
    cu->hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)CUDA_WRAPPER::ThreadExit);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)CUDA_WRAPPER::GetLastError);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterFatBinary", (PROC)CUDA_WRAPPER::RegisterFatBinary);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterFunction", (PROC)CUDA_WRAPPER::RegisterFunction);
    cu->hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)CUDA_WRAPPER::Launch);
    cu->hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)CUDA_WRAPPER::SetupArgument);
    cu->hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)CUDA_WRAPPER::ConfigureCall);
    

    // Unimplemented functions.
    cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc3D", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc3DArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemset3D", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy3D", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DAsync", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMallocHost", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMallocPitch", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMallocArray", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaFreeArray", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaHostGetFlags", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemGetInfo", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyArrayToArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2D", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DArrayToArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbol", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbol", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbolAsync", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbolAsync", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemset2D", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetSymbolAddress", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetSymbolSize", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetDeviceCount", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetDeviceProperties", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaChooseDevice", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaSetDevice", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetDevice", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaSetValidDevices", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaSetDeviceFlags", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaBindTexture", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaBindTexture2D", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaBindTextureToArray", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaUnbindTexture", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetTextureAlignmentOffset", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetTextureReference", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaBindSurfaceToArray", (PROC)CUDA_WRAPPER::Unimplemented);
// not in library?????????? cu->hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceAlignmentOffset", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceReference", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetChannelDesc", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaCreateChannelDesc", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaPeekAtLastError", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetErrorString", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaFuncSetCacheConfig", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaFuncGetAttributes", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaStreamCreate", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaStreamDestroy", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaStreamSynchronize", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaStreamQuery", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventCreate", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventCreateWithFlags", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventRecord", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventQuery", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventSynchronize", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventDestroy", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaEventElapsedTime", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForDevice", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForHost", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaThreadSynchronize", (PROC)CUDA_WRAPPER::ThreadSynchronize);
    cu->hook_manager->HookImport(cuda_module_name, "cudaThreadSetLimit", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaThreadGetLimit", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaDriverGetVersion", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaRuntimeGetVersion", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetExportTable", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnregisterResource", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceSetMapFlags", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsMapResources", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnmapResources", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceGetMappedPointer", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsSubResourceGetMappedArray", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaRegisterFatBinary", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaUnregisterFatBinary", (PROC)CUDA_WRAPPER::UnregisterFatBinary);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterVar", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterTexture", (PROC)CUDA_WRAPPER::Unimplemented);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterSurface", (PROC)CUDA_WRAPPER::Unimplemented);
//  cu->hook_manager->HookImport(cuda_module_name, "cudaRegisterFunction", (PROC)CUDA_WRAPPER::Unimplemented);

    return true;
}

void CUDA_WRAPPER::ExitHandler()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    // Check if there are no unfreed blocks.
    for (unsigned int i = 0; i < cu->alloc_list.size(); ++i)
    {
        data d = cu->alloc_list[i];
        (*cu->output_stream) << "Unfreed CUDA memory block.\n";
        (*cu->output_stream) << " Pointer " << d.ptr << " was allocated in "
            << d.context << ".\n";
        (*cu->output_stream) << " Block size is " << d.size << " bytes.\n";
        (*cu->output_stream) << " This check was called during program exit, "
            << cu->Context() << " (exit handler).\n\n";
        cu->CheckSinglePtrOverwrite(&d);
    }
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::CheckSinglePtrOverwrite(const data * d)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CallStackInfo * csi = CallStackInfo::Singleton();
    if (! d->is_host)
    {
        unsigned char * hostbuffer = (unsigned char *)malloc(d->size);
        if (! hostbuffer)
            return NOT_OK;

        cudaError_t e1;
        if (! cu->do_emulation)
        {
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
            e1 = (*proc)(hostbuffer, d->ptr, d->size, cudaMemcpyDeviceToHost);
        } else
        {
            memcpy(hostbuffer, d->ptr, d->size);
            e1 = cudaSuccess;
        }
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
                    (*cu->output_stream) << "Overwrite of cuda memory block header.\n";
                    (*cu->output_stream) << " Pointer " << d->ptr << " was allocated in file "
                        << d->context << ".\n";
                    (*cu->output_stream) << " This check was performed during a CUDA call in "
                        << Context() << ".\n\n";
                    other = false;
                    break;
                }
            }
    }
    return OK;      
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::CheckOverwrite()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    // Check if there are overwrites.
    for (unsigned int i = 0; i < cu->alloc_list.size(); ++i)
    {
        data d = cu->alloc_list[i];
        cu->CheckSinglePtrOverwrite(&d);
    }
    return OK;      
}

int CUDA_WRAPPER::FindAllocatedBlock(const void * pointer)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    unsigned int i;
    for (i = 0; i < cu->alloc_list.size(); ++i)
    {
        data * d = &cu->alloc_list[i];
        if (pointer >= ((unsigned char *)d->ptr) + cu->padding_size
            && pointer < (d->size - cu->padding_size + (unsigned char *)d->ptr))
            break;
    }
    if (i == cu->alloc_list.size())
    {
        return -1;
    }
    return i;
}

bool CUDA_WRAPPER::IsBadPointer(const void * ptr)
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


char * CUDA_WRAPPER::Context(int lines)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->global_context && *cu->global_context)
        return cu->global_context;

    CallStackInfo * csi = CallStackInfo::Singleton();
    char * context = csi->Context(lines);
    return context;
}

void CUDA_WRAPPER::MakeContext(char * file_name, int line)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->global_context)
        free(cu->global_context);
#define BUFFERSIZE 50000
    char buffer[BUFFERSIZE];
    strcpy(buffer, "file ");
    strncat(buffer, file_name_tail(file_name), BUFFERSIZE);
    strncat(buffer, ", line ", BUFFERSIZE);
    char buffer2[200];
    sprintf(buffer2, "%d", line);
    strncat(buffer, buffer2, BUFFERSIZE);
    cu->global_context = strdup(buffer);
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  Wrappers for CUDA memory management API.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


cudaError_t CUDARTAPI CUDA_WRAPPER::Malloc(void ** ptr, size_t size)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

    void * local = 0;

    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMalloc called, " << context << ".\n\n";
    }
            
    if (IsBadPointer(ptr))
    {
        (*cu->output_stream) << "Bad pointer passed to cudaMalloc("
            << ptr << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }

    cudaError_t resetErrKernel = GetLastError();

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    typePtrCudaMalloc proc;
    if (! cu->do_emulation)
    {
        cudaError_t e1 = (*ptrCudaMalloc)((void**)&local, size + 2 * cu->padding_size);
        if (e1 != 0)
        {
            (*cu->output_stream) << "cudaMalloc failed."
                << " Return value = "
                << e1 << ".\n";
            (*cu->output_stream) << " This check was performed in " << Context() << ".\n\n";
            if (cu->quit_on_error)
                exit(1);
            if (cu->do_not_call_cuda_after_sanity_check_fail)
                return cudaErrorMemoryAllocation;
            return e1;
        }
    } else
    {
        local = malloc(size+2*cu->padding_size);
    }
    char * hostbuffer = (char*)malloc(size + 2 * cu->padding_size);
    if (! hostbuffer)
    {
        (*cu->output_stream) << "Host memory allocation failed in cudaMalloc.  The buffer is used to initialize the device buffer.\n";
        (*cu->output_stream) << " This check was performed in " << Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
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
    if (! cu->do_emulation)
    {
        cudaError_t e2 = (*ptrCudaMemcpy)(local, hostbuffer, size + 2 * cu->padding_size, cudaMemcpyHostToDevice);
        if (e2 != 0)
        {
            (*cu->output_stream) << "cudaMemcpy failed, rc = " << e2 << ", in cudaMalloc. This call is used to initialize the device buffer.\n";
            (*cu->output_stream) << " This check was performed in " << Context() << ".\n\n";
            if (cu->quit_on_error)
                exit(1);
            if (cu->do_not_call_cuda_after_sanity_check_fail)
                return cudaErrorMemoryAllocation;
            return e2;
        }
    } else
    {
        memcpy(local, hostbuffer, size + 2 * cu->padding_size);
    }
    free(hostbuffer);
    data d;
    d.ptr = local;
    d.size = size + 2 * cu->padding_size;
    d.is_host = false;
    d.context = strdup(cu->Context());
    cu->alloc_list.push_back(d);
    *ptr = ((char*)local) + cu->padding_size;
    return cudaSuccess;     
}

cudaError_t CUDARTAPI CUDA_WRAPPER::Free(void * ptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaFree called, ptr = " << ptr << ", "
             << context << ".\n\n";
    }

    // Null pointer sanity check.
    if (ptr == 0)
    {
        (*cu->output_stream) << "Pointer in HostFree("
            << "ptr = " << ptr
            << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        if (! cu->do_emulation)
        {
            // not sure what it does, send off to cuda api.
            cudaError_t err = (*ptrCudaFree)(ptr);
            return err;
        }
        return cudaErrorMemoryAllocation;
    }

    int di = cu->FindAllocatedBlock(ptr);
    if (di == -1)
    {
        (*cu->output_stream) << "Pointer to cudaFree(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        if (! cu->do_emulation)
        {
            // not sure what it does, send off to cuda api.
            cudaError_t err = (*ptrCudaFree)(ptr);
            return err;
        }
        return cudaErrorMemoryAllocation;
    }
    data * d = &cu->alloc_list[di];
    void * local = ((char*)ptr) - cu->padding_size;
    if (d->ptr != local)
    {
        (*cu->output_stream) << "Pointer to cudaFree(" << ptr << ") is invalid.\n\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        if (! cu->do_emulation)
        {
            // not sure what it does, send off to cuda api.
            cudaError_t err = (*ptrCudaFree)(ptr);
            return err;
        }
        return cudaErrorMemoryAllocation;
    }
    cu->alloc_list.erase(cu->alloc_list.begin() + di);
    cu->CheckSinglePtrOverwrite(d);
    if (! cu->do_emulation)
    {
        cudaError_t err = (*ptrCudaFree)(local);
        return err;
    } else
    {
        free(local);
        return cudaSuccess;
    }
}


cudaError_t CUDARTAPI CUDA_WRAPPER::HostAlloc(void ** ptr, size_t size, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    void * local = 0;
    char * context = Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaHostAlloc called, " << context << ".\n\n";
    }

    if (cu->IsBadPointer(ptr))
    {
        (*cu->output_stream) << "Bad pointer passed to HostAlloc("
            << ptr << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }

    *ptr = 0;

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    cudaError_t e1 = (*ptrCudaHostAlloc)((void**)&local, size + 2 * cu->padding_size, flags);
    if (e1 != 0)
    {
        (*cu->output_stream) << "cudaHostAlloc failed."
            << " Return value = "
            << e1 << ".\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
        return e1;
    }
    char * hostbuffer = (char*)malloc(size + 2 * cu->padding_size);
    if (! hostbuffer)
    {
        (*cu->output_stream) << "Local memory allocation for setting buffer header and footer in cudaHostAlloc failed.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
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
    data d;
    d.ptr = local;
    d.size = size + 2 * cu->padding_size;
    d.context = strdup(cu->Context());
    d.is_host = true;
    cu->alloc_list.push_back(d);
    *ptr = ((char*)local) + cu->padding_size;
    return cudaSuccess;     
}

cudaError_t CUDARTAPI CUDA_WRAPPER::FreeHost(void * ptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;

    char * context = Context();

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
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        cudaError_t err = (*ptrCudaFree)(ptr);
        return err;
    }

    int di = cu->FindAllocatedBlock(ptr);
    if (di == -1)
    {
        (*cu->output_stream) << "Pointer to FreeHost(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        cudaError_t err = (*ptrCudaFree)(ptr);
        return err;
    }
    data * d = &cu->alloc_list[di];
    void * local = ((char*)ptr) - cu->padding_size;
    if (d->ptr != local)
    {
        (*cu->output_stream) << "Pointer to FreeHost(" << ptr << ") is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        cudaError_t err = (*ptrCudaFree)(ptr);
        return err;
    }
    cu->alloc_list.erase(cu->alloc_list.begin() + di);
    cu->CheckSinglePtrOverwrite(d);
    cudaError_t err = (*ptrCudaFreeHost)(ptr);
    if (err)
    {
        if (cu->quit_on_error)
            exit(1);
    }
    return err;
}

cudaError_t CUDARTAPI CUDA_WRAPPER::HostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaHostGetDevicePointer called, " << context << ".\n\n";
    }

    if (cu->IsBadPointer(pDevice))
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
    for (i = 0; i < cu->alloc_list.size(); ++i)
    {
        if (cu->alloc_list[i].ptr == local)
            break;
    }
    if (i == cu->alloc_list.size())
    {
        (*cu->output_stream) << "pHost pointer passed to HostGetDevicePointer(..., " << pHost << ", ...) is invalid.\n";
    }
    data d = cu->alloc_list[i];
    if (! d.is_host)
    {
        (*cu->output_stream) << "pHost pointer passed to HostGetDevicePointer(..., " << pHost << ", ...) is invalid.\n";
        (*cu->output_stream) << " The pointer is not a pointer created by cudaHostAlloc\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
    }
    cudaError_t err = (*ptrCudaHostGetDevicePointer)(pDevice, local, flags);
    if (err)
    {
        return err;
    }
    // Map pDevice correctly.
    *pDevice = (void*) (((char*)*pDevice) + cu->padding_size);
    return err;
}

cudaError_t CUDARTAPI CUDA_WRAPPER::Memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
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
        (*cu->output_stream) << "Destination pointer in Memcpy("
            << "dst = " << dst
            << ", ..., ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (! cu->do_emulation)
        {
            cudaError_t err = (*ptrCudaMemcpy)(dst, src, count, kind);
            return err;
        } else
        {
            //memcpy(dst, src, count);
            return cudaErrorMemoryAllocation;
        }
    }
    if (src == 0)
    {
        (*cu->output_stream) << "Source pointer passed to Memcpy(..., "
            << "src = " << src
            << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
        if (! cu->do_emulation)
        {
            cudaError_t err = (*ptrCudaMemcpy)(dst, src, count, kind);
            return err;
        } else
        {
            //memcpy(dst, src, count);
            return cudaErrorMemoryAllocation;
        }
    }

    // Four cases:
    if (kind == cudaMemcpyHostToDevice)
    {
        int dd = cu->FindAllocatedBlock(dst);
        int ds = cu->FindAllocatedBlock(src);

        // Users can pass a pointer to a pointer in the middle of a block.
        // Also, the source pointer can look like a device pointer if the address
        // ranges of the source and target overlap.  This did happen for me using
        // a Geforce 9800 on Windows.  So, FindAllocatedBlock may return a block
        // even though it really is a host pointer!

        if (ds != -1 && dd == -1)
        {
            (*cu->output_stream) << "Source and destination pointers in Memcpy("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " are reversed in directionality.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (dd == -1)
        {
            (*cu->output_stream) << "Destination pointer in Memcpy("
                << "dst = " << dst
                << ", ..., ..., ...) "
                << " is invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (ds != -1 && ! cu->alloc_list[ds].is_host)
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src
                << ", ..., ...) looks invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (ds != -1 && cu->alloc_list[ds].is_host)
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src
                << ", ..., ...) is a pointer to a host block that could be device addressible.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (cu->IsBadPointer(src))
        {
            (*cu->output_stream) << "Source pointer passed to Memcpy(..., src = " << src << ", ..., ...) is invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        data * ddst = 0;
        data * dsrc = 0;
        if (dd != -1)
            ddst = &cu->alloc_list[dd];
        if (ds != -1)
            dsrc = &cu->alloc_list[ds];
        if (ddst)
            cu->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            cu->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        if (! cu->do_emulation)
        {
            err = (*ptrCudaMemcpy)(dst, src, count, kind);
        } else
        {
            memcpy(dst, src, count);
            err = cudaSuccess;
        }
        // Perform overwrite check again.
        if (ddst)
            cu->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            cu->CheckSinglePtrOverwrite(dsrc);
        return err;
    }
    else if (kind == cudaMemcpyDeviceToHost)
    {
        int dd = cu->FindAllocatedBlock(dst);
        int ds = cu->FindAllocatedBlock(src);
        if (ds == -1 && dd != -1)
        {
            (*cu->output_stream) << "Source and destination pointers in Memcpy("
                << "dst = " << dst
                << ", src = " << src << ", ..., ...) "
                << " are reversed in directionality.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (ds == -1)
        {
            (*cu->output_stream) << "Source pointer in Memcpy(..., "
                << "src = " << src
                << ", ..., ..., ...) "
                << " is invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (dd != -1 && ! cu->alloc_list[dd].is_host)
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy(..., "
                << "src = " << src
                << ", ..., ...) is invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (dd != -1 && cu->alloc_list[dd].is_host)
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy("
                << "dst = " << dst
                << ", ..., ..., ...) is a pointer to a host block that could be device addressible.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        else if (cu->IsBadPointer(dst))
        {
            (*cu->output_stream) << "Destination pointer passed to Memcpy("
                << "dst = " << dst
                << ", ..., ..., ...) is invalid.\n";
            (*cu->output_stream) << " This check was performed during a CUDA call in file "
                << file_name_tail(file_name) << ", line " << line << ".\n\n";
        }
        // Check before copy if block boundaries are intact.
        data * ddst = 0;
        data * dsrc = 0;
        if (dd != -1)
            ddst = &cu->alloc_list[dd];
        if (ds != -1)
            dsrc = &cu->alloc_list[ds];
        if (ddst)
            cu->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            cu->CheckSinglePtrOverwrite(dsrc);
        // Perform copy.
        cudaError_t err;
        if (! cu->do_emulation)
        {
            err = (*ptrCudaMemcpy)(dst, src, count, kind);
        } else
        {
            memcpy(dst, src, count);
            err = cudaSuccess;
        }
        // Perform overwrite check again.
        if (ddst)
            cu->CheckSinglePtrOverwrite(ddst);
        if (dsrc)
            CheckSinglePtrOverwrite(dsrc);
        return err;
    }
    else
    {
        cudaError_t err = (*ptrCudaMemcpy)(dst, src, count, kind);
        return err;
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::Memset(void * dst, int value, size_t count)
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
        cudaError_t err = (*ptrCudaMemset)(dst, value, count);
        return err;
    }

    int dd = cu->FindAllocatedBlock(dst);
    if (dd == -1)
    {
        (*cu->output_stream) << "Destination pointer in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        cudaError_t err = (*ptrCudaMemset)(dst, value, count);
        return err;
    }
    data * ddst = 0;
    ddst = &cu->alloc_list[dd];
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
                return cudaErrorMemoryAllocation;
        }
    }

    if (cu->CheckSinglePtrOverwrite(ddst) != OK)
    {
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }
    // Perform copy.
    cudaError_t err = (*ptrCudaMemset)(dst, value, count);
    // Perform overwrite check again.
    if (cu->CheckSinglePtrOverwrite(ddst) != OK)
    {
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }
    return err;
}


cudaError_t CUDARTAPI CUDA_WRAPPER::ThreadExit()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaThreadExit called, " << context << ".\n\n";
    }

    if (cu->alloc_list.size() != 0)
    {
        (*cu->output_stream) << " Warning, cudaThreadExit reinitializes the entire runtime.  All pointers to old cuda memory are stale.\n";
    }

    cu->alloc_list.clear();
    if (! cu->do_emulation)
        return (*ptrCudaThreadExit)();
    else
        return cudaSuccess;
}

cudaError_t CUDA_WRAPPER::GetLastError()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
        return (*ptrCudaGetLastError)();
    else
        return cudaSuccess;
}

void** CUDA_WRAPPER::RegisterFatBinary(void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->do_crash)
    {
        printf("here\n");
        //assert(false);
        // if this fails, try another...
        int x = 0;
        int y = 0;
        int z = x/y;
    }

    if (true) // for now, ignore (fatCubin)
    {
        std::cout << "NEW FATBIN\n";
        // Get PTX code from the record.
        __cudaFatCudaBinary * fcb = (__cudaFatCudaBinary *)fatCubin;
        if (fcb)
        {
            __cudaFatPtxEntry * ptx = fcb->ptx;
            for ( ; ptx && ptx->gpuProfileName; ptx++)
            {
                char * profile = ptx->gpuProfileName;
                char * code = ptx->ptx;
                std::cout << "====================================================\n";
                std::cout << "PROFILE = " << profile << std::endl;
                std::cout << "CODE:\n";
                std::cout << code << std::endl;
                std::cout << "====================================================\n\n\n";
                CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
                emulator->Extract_From_Source(profile, code);
           }

            // ELF contains just in time code for every PTX.
            // Execution will depend on picking which one for the device.
            __cudaFatElfEntry * elf = fcb->elf;
            for ( ; elf; elf = elf->next)
            {
                char * code = elf->elf;
            }
        }
    }
    if (! cu->do_emulation)
    {
        typePtrCudaRegisterFatBinary proc = (typePtrCudaRegisterFatBinary)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::RegisterFatBinary);
        return (*proc)(fatCubin);
    } else
        return 0;
}

void CUDARTAPI CUDA_WRAPPER::UnregisterFatBinary(void **fatCubinHandle)
{
    // Should probably do something like free the ast...
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaUnregisterFatBinary proc = (typePtrCudaUnregisterFatBinary)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::UnregisterFatBinary);
        (*proc)(fatCubinHandle);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::Launch(const char *entry)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaLaunch proc = (typePtrCudaLaunch)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Launch);
        return (*proc)(entry);
    } else
    {
        CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
        emulator->Execute((void*)entry);
        return cudaSuccess;
    }
}

void CUDARTAPI CUDA_WRAPPER::RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaRegisterFunction proc = (typePtrCudaRegisterFunction)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::RegisterFunction);
        (*proc)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    } else
    {
        CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
        emulator->RegisterFunction((void*)hostFun, deviceFun);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaConfigureCall proc = (typePtrCudaConfigureCall)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::ConfigureCall);
        return (*proc)(gridDim, blockDim, sharedMem, stream);
    } else
    {
        CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
        return emulator->ConfigureCall(gridDim, blockDim, sharedMem, stream);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::SetupArgument(const void *arg, size_t size, size_t offset)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaSetupArgument proc = (typePtrCudaSetupArgument)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::SetupArgument);
        return (*proc)(arg, size, offset);
    } else
    {
        CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
        return emulator->SetupArgument(arg, size, offset);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::ThreadSynchronize(void)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaThreadSynchronize proc = (typePtrCudaThreadSynchronize)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::ThreadSynchronize);
        return (*proc)();
    } else
    {
        CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
        return emulator->ThreadSynchronize();
    }
}



///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  Options for behavior of this debugging wrapper.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetPaddingSize(size_t s)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->padding_size = s;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetPaddingSize called, " << context << ".\n";
        (*cu->output_stream) << " Padding size now " << s << "\n\n";
    }
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetPaddingByte(unsigned char b)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->padding_byte = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetPaddingByte called, " << context << ".\n";
        (*cu->output_stream) << " Padding byte now " << b << "\n\n";
    }
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetDevicePointerToFirstByteInBlock(bool b)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->device_pointer_to_first_byte_in_block = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDevicePointerToFirstByteInBlock called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetOutputStream(std::ostream * fp)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    cu->output_stream = fp;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetOutputStream called, " << context << ".\n\n";
    }
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetTraceAllCalls(bool b)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->trace_all_calls = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetTraceAllCalls called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetQuitOnError(bool b)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->quit_on_error = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetQuitOnError called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}


CUDA_WRAPPER::return_type CUDA_WRAPPER::SetDoNotCallCudaAfterSanityCheckFail(bool b)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->do_not_call_cuda_after_sanity_check_fail = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDoNotCallCudaAfterSanityCheckFail called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::CopyOptions(CUDA_WRAPPER * ptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    (*cu->output_stream) << "CopyOptions called, " << context << ".\n\n";
    cu->padding_size = ptr->padding_size;
    cu->padding_byte = ptr->padding_byte;
    cu->device_pointer_to_first_byte_in_block = ptr->device_pointer_to_first_byte_in_block;
    cu->do_not_call_cuda_after_sanity_check_fail = ptr->do_not_call_cuda_after_sanity_check_fail;
    cu->trace_all_calls = ptr->trace_all_calls;
    cu->quit_on_error = ptr->quit_on_error;
    return OK;
}

CUDA_WRAPPER::return_type CUDA_WRAPPER::SetDevice(char * device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->device = device;
    CUDA_EMULATOR * emulator = CUDA_EMULATOR::Singleton();
    emulator->SetDevice(device);

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDevice called, " << context << ".\n";
        (*cu->output_stream) << " Device now " << device << "\n\n";
    }
    return OK;
}

