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
CudaMemoryDebug::typePtrCudaMalloc3D CudaMemoryDebug::ptrCudaMalloc3D;
CudaMemoryDebug::typePtrCudaMalloc3DArray CudaMemoryDebug::ptrCudaMalloc3DArray;
CudaMemoryDebug::typePtrCudaMemset3D CudaMemoryDebug::ptrCudaMemset3D;
CudaMemoryDebug::typePtrCudaMemcpy3D CudaMemoryDebug::ptrCudaMemcpy3D;
CudaMemoryDebug::typePtrCudaMemcpy3DAsync CudaMemoryDebug::ptrCudaMemcpy3DAsync;
CudaMemoryDebug::typePtrCudaMalloc CudaMemoryDebug::ptrCudaMalloc;
CudaMemoryDebug::typePtrCudaMallocHost CudaMemoryDebug::ptrCudaMallocHost;
CudaMemoryDebug::typePtrCudaMallocPitch CudaMemoryDebug::ptrCudaMallocPitch;
CudaMemoryDebug::typePtrCudaMallocArray CudaMemoryDebug::ptrCudaMallocArray;
CudaMemoryDebug::typePtrCudaFree CudaMemoryDebug::ptrCudaFree;
CudaMemoryDebug::typePtrCudaFreeHost CudaMemoryDebug::ptrCudaFreeHost;
CudaMemoryDebug::typePtrCudaFreeArray CudaMemoryDebug::ptrCudaFreeArray;
CudaMemoryDebug::typePtrCudaHostAlloc CudaMemoryDebug::ptrCudaHostAlloc;
CudaMemoryDebug::typePtrCudaHostGetDevicePointer CudaMemoryDebug::ptrCudaHostGetDevicePointer;
CudaMemoryDebug::typePtrCudaHostGetFlags CudaMemoryDebug::ptrCudaHostGetFlags;
CudaMemoryDebug::typePtrCudaMemGetInfo CudaMemoryDebug::ptrCudaMemGetInfo;
CudaMemoryDebug::typePtrCudaMemcpy CudaMemoryDebug::ptrCudaMemcpy;
CudaMemoryDebug::typePtrCudaMemcpyToArray CudaMemoryDebug::ptrCudaMemcpyToArray;
CudaMemoryDebug::typePtrCudaMemcpyFromArray CudaMemoryDebug::ptrCudaMemcpyFromArray;
CudaMemoryDebug::typePtrCudaMemcpyArrayToArray CudaMemoryDebug::ptrCudaMemcpyArrayToArray;
CudaMemoryDebug::typePtrCudaMemcpy2D CudaMemoryDebug::ptrCudaMemcpy2D;
CudaMemoryDebug::typePtrCudaMemcpy2DToArray CudaMemoryDebug::ptrCudaMemcpy2DToArray;
CudaMemoryDebug::typePtrCudaMemcpy2DFromArray CudaMemoryDebug::ptrCudaMemcpy2DFromArray;
CudaMemoryDebug::typePtrCudaMemcpy2DArrayToArray CudaMemoryDebug::ptrCudaMemcpy2DArrayToArray;
CudaMemoryDebug::typePtrCudaMemcpyToSymbol CudaMemoryDebug::ptrCudaMemcpyToSymbol;
CudaMemoryDebug::typePtrCudaMemcpyFromSymbol CudaMemoryDebug::ptrCudaMemcpyFromSymbol;
CudaMemoryDebug::typePtrCudaMemcpyAsync CudaMemoryDebug::ptrCudaMemcpyAsync;
CudaMemoryDebug::typePtrCudaMemcpyToArrayAsync CudaMemoryDebug::ptrCudaMemcpyToArrayAsync;
CudaMemoryDebug::typePtrCudaMemcpyFromArrayAsync CudaMemoryDebug::ptrCudaMemcpyFromArrayAsync;
CudaMemoryDebug::typePtrCudaMemcpy2DAsync CudaMemoryDebug::ptrCudaMemcpy2DAsync;
CudaMemoryDebug::typePtrCudaMemcpy2DToArrayAsync CudaMemoryDebug::ptrCudaMemcpy2DToArrayAsync;
CudaMemoryDebug::typePtrCudaMemcpy2DFromArrayAsync CudaMemoryDebug::ptrCudaMemcpy2DFromArrayAsync;
CudaMemoryDebug::typePtrCudaMemcpyToSymbolAsync CudaMemoryDebug::ptrCudaMemcpyToSymbolAsync;
CudaMemoryDebug::typePtrCudaMemcpyFromSymbolAsync CudaMemoryDebug::ptrCudaMemcpyFromSymbolAsync;
CudaMemoryDebug::typePtrCudaMemset CudaMemoryDebug::ptrCudaMemset;
CudaMemoryDebug::typePtrCudaMemset2D CudaMemoryDebug::ptrCudaMemset2D;
CudaMemoryDebug::typePtrCudaGetSymbolAddress CudaMemoryDebug::ptrCudaGetSymbolAddress;
CudaMemoryDebug::typePtrCudaGetSymbolSize CudaMemoryDebug::ptrCudaGetSymbolSize;
CudaMemoryDebug::typePtrCudaGetDeviceCount CudaMemoryDebug::ptrCudaGetDeviceCount;
CudaMemoryDebug::typePtrCudaGetDeviceProperties CudaMemoryDebug::ptrCudaGetDeviceProperties;
CudaMemoryDebug::typePtrCudaChooseDevice CudaMemoryDebug::ptrCudaChooseDevice;
CudaMemoryDebug::typePtrCudaSetDevice CudaMemoryDebug::ptrCudaSetDevice;
CudaMemoryDebug::typePtrCudaGetDevice CudaMemoryDebug::ptrCudaGetDevice;
CudaMemoryDebug::typePtrCudaSetValidDevices CudaMemoryDebug::ptrCudaSetValidDevices;
CudaMemoryDebug::typePtrCudaSetDeviceFlags CudaMemoryDebug::ptrCudaSetDeviceFlags;
CudaMemoryDebug::typePtrCudaBindTexture CudaMemoryDebug::ptrCudaBindTexture;
CudaMemoryDebug::typePtrCudaBindTexture2D CudaMemoryDebug::ptrCudaBindTexture2D;
CudaMemoryDebug::typePtrCudaBindTextureToArray CudaMemoryDebug::ptrCudaBindTextureToArray;
CudaMemoryDebug::typePtrCudaUnbindTexture CudaMemoryDebug::ptrCudaUnbindTexture;
CudaMemoryDebug::typePtrCudaGetTextureAlignmentOffset CudaMemoryDebug::ptrCudaGetTextureAlignmentOffset;
CudaMemoryDebug::typePtrCudaGetTextureReference CudaMemoryDebug::ptrCudaGetTextureReference;
CudaMemoryDebug::typePtrCudaBindSurfaceToArray CudaMemoryDebug::ptrCudaBindSurfaceToArray;
CudaMemoryDebug::typePtrCudaGetSurfaceAlignmentOffset CudaMemoryDebug::ptrCudaGetSurfaceAlignmentOffset;
CudaMemoryDebug::typePtrCudaGetSurfaceReference CudaMemoryDebug::ptrCudaGetSurfaceReference;
CudaMemoryDebug::typePtrCudaGetChannelDesc CudaMemoryDebug::ptrCudaGetChannelDesc;
CudaMemoryDebug::typePtrCudaCreateChannelDesc CudaMemoryDebug::ptrCudaCreateChannelDesc;
CudaMemoryDebug::typePtrCudaGetLastError CudaMemoryDebug::ptrCudaGetLastError;
CudaMemoryDebug::typePtrCudaPeekAtLastError CudaMemoryDebug::ptrCudaPeekAtLastError;
CudaMemoryDebug::typePtrCudaGetErrorString CudaMemoryDebug::ptrCudaGetErrorString;
CudaMemoryDebug::typePtrCudaConfigureCall CudaMemoryDebug::ptrCudaConfigureCall;
CudaMemoryDebug::typePtrCudaSetupArgument CudaMemoryDebug::ptrCudaSetupArgument;
CudaMemoryDebug::typePtrCudaFuncSetCacheConfig CudaMemoryDebug::ptrCudaFuncSetCacheConfig;
CudaMemoryDebug::typePtrCudaLaunch CudaMemoryDebug::ptrCudaLaunch;
CudaMemoryDebug::typePtrCudaFuncGetAttributes CudaMemoryDebug::ptrCudaFuncGetAttributes;
CudaMemoryDebug::typePtrCudaStreamCreate CudaMemoryDebug::ptrCudaStreamCreate;
CudaMemoryDebug::typePtrCudaStreamDestroy CudaMemoryDebug::ptrCudaStreamDestroy;
CudaMemoryDebug::typePtrCudaStreamSynchronize CudaMemoryDebug::ptrCudaStreamSynchronize;
CudaMemoryDebug::typePtrCudaStreamQuery CudaMemoryDebug::ptrCudaStreamQuery;
CudaMemoryDebug::typePtrCudaEventCreate CudaMemoryDebug::ptrCudaEventCreate;
CudaMemoryDebug::typePtrCudaEventCreateWithFlags CudaMemoryDebug::ptrCudaEventCreateWithFlags;
CudaMemoryDebug::typePtrCudaEventRecord CudaMemoryDebug::ptrCudaEventRecord;
CudaMemoryDebug::typePtrCudaEventQuery CudaMemoryDebug::ptrCudaEventQuery;
CudaMemoryDebug::typePtrCudaEventSynchronize CudaMemoryDebug::ptrCudaEventSynchronize;
CudaMemoryDebug::typePtrCudaEventDestroy CudaMemoryDebug::ptrCudaEventDestroy;
CudaMemoryDebug::typePtrCudaEventElapsedTime CudaMemoryDebug::ptrCudaEventElapsedTime;
CudaMemoryDebug::typePtrCudaSetDoubleForDevice CudaMemoryDebug::ptrCudaSetDoubleForDevice;
CudaMemoryDebug::typePtrCudaSetDoubleForHost CudaMemoryDebug::ptrCudaSetDoubleForHost;
CudaMemoryDebug::typePtrCudaThreadExit CudaMemoryDebug::ptrCudaThreadExit;
CudaMemoryDebug::typePtrCudaThreadSynchronize CudaMemoryDebug::ptrCudaThreadSynchronize;
CudaMemoryDebug::typePtrCudaThreadSetLimit CudaMemoryDebug::ptrCudaThreadSetLimit;
CudaMemoryDebug::typePtrCudaThreadGetLimit CudaMemoryDebug::ptrCudaThreadGetLimit;
CudaMemoryDebug::typePtrCudaDriverGetVersion CudaMemoryDebug::ptrCudaDriverGetVersion;
CudaMemoryDebug::typePtrCudaRuntimeGetVersion CudaMemoryDebug::ptrCudaRuntimeGetVersion;
CudaMemoryDebug::typePtrCudaGetExportTable CudaMemoryDebug::ptrCudaGetExportTable;
CudaMemoryDebug::typePtrCudaGraphicsUnregisterResource CudaMemoryDebug::ptrCudaGraphicsUnregisterResource;
CudaMemoryDebug::typePtrCudaGraphicsResourceSetMapFlags CudaMemoryDebug::ptrCudaGraphicsResourceSetMapFlags;
CudaMemoryDebug::typePtrCudaGraphicsMapResources CudaMemoryDebug::ptrCudaGraphicsMapResources;
CudaMemoryDebug::typePtrCudaGraphicsUnmapResources CudaMemoryDebug::ptrCudaGraphicsUnmapResources;
CudaMemoryDebug::typePtrCudaGraphicsResourceGetMappedPointer CudaMemoryDebug::ptrCudaGraphicsResourceGetMappedPointer;
CudaMemoryDebug::typePtrCudaGraphicsSubResourceGetMappedArray CudaMemoryDebug::ptrCudaGraphicsSubResourceGetMappedArray;
CudaMemoryDebug::typePtrCudaRegisterFatBinary CudaMemoryDebug::ptrCudaRegisterFatBinary;
CudaMemoryDebug::typePtrCudaUnregisterFatBinary CudaMemoryDebug::ptrCudaUnregisterFatBinary;
CudaMemoryDebug::typePtrCudaRegisterVar CudaMemoryDebug::ptrCudaRegisterVar;
CudaMemoryDebug::typePtrCudaRegisterTexture CudaMemoryDebug::ptrCudaRegisterTexture;
CudaMemoryDebug::typePtrCudaRegisterSurface CudaMemoryDebug::ptrCudaRegisterSurface;
CudaMemoryDebug::typePtrCudaRegisterFunction CudaMemoryDebug::ptrCudaRegisterFunction;


static char temp_buffer[50000];

CudaMemoryDebug * CudaMemoryDebug::singleton;
typedef cudaError_t (*cfunc)(void **devPtr, size_t size);
cudaError_t (*ptrCudaMalloc)(void **devPtr, size_t size);


CudaMemoryDebug * CudaMemoryDebug::Singleton()
{
    if (singleton)
        return singleton;
    singleton = new CudaMemoryDebug();
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


CudaMemoryDebug::CudaMemoryDebug()
{
    CudaMemoryDebug * cu = this;
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
	cu->do_crash = true;
}


bool CudaMemoryDebug::DoInit(char * cuda_module_name, HookManager * hm)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    CallStackInfo * csi = CallStackInfo::Singleton();
    csi->ClassifyAsPrefix("call-stack-info.cpp");
    csi->ClassifyAsPrefix("cuda-memory-debug.cpp");
    csi->ClassifyAsPrefix("ptr-cuda-runtime.cpp");

    if (atexit(CudaMemoryDebug::ExitHandler))
    {
        char * context = cu->Context(3);
        (*cu->output_stream) << "Cannot register CudaMemoryDebug exit handler during initialization.  Memory leaks will not be detected.\n";
        (*cu->output_stream) << " Call stack is:\n"
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
    }

    cu->hook_manager = hm;
    cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)CudaMemoryDebug::Malloc);
    cu->hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)CudaMemoryDebug::HostAlloc);
    cu->hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)CudaMemoryDebug::Free);
	cu->hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)CudaMemoryDebug::FreeHost);
    cu->hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)CudaMemoryDebug::HostGetDevicePointer);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)CudaMemoryDebug::Memcpy);
    cu->hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)CudaMemoryDebug::Memset);
    cu->hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)CudaMemoryDebug::ThreadExit);
    cu->hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)CudaMemoryDebug::GetLastError);
    cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterFatBinary", (PROC)CudaMemoryDebug::RegisterFatBinary);
	cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterFunction", (PROC)CudaMemoryDebug::RegisterFunction);
	cu->hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)CudaMemoryDebug::Launch);
	cu->hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)CudaMemoryDebug::SetupArgument);
	cu->hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)CudaMemoryDebug::ConfigureCall);
	

	// Unimplemented functions.
	cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc3D", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc3DArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemset3D", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy3D", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DAsync", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMallocHost", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMallocPitch", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMallocArray", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaFreeArray", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaHostGetFlags", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemGetInfo", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyArrayToArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2D", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DArrayToArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbol", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbol", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArrayAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArrayAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArrayAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArrayAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbolAsync", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbolAsync", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaMemset2D", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetSymbolAddress", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetSymbolSize", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetDeviceCount", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetDeviceProperties", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaChooseDevice", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaSetDevice", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetDevice", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaSetValidDevices", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaSetDeviceFlags", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaBindTexture", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaBindTexture2D", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaBindTextureToArray", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaUnbindTexture", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetTextureAlignmentOffset", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetTextureReference", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaBindSurfaceToArray", (PROC)CudaMemoryDebug::Unimplemented);
// not in library??????????	cu->hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceAlignmentOffset", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceReference", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetChannelDesc", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaCreateChannelDesc", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaPeekAtLastError", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetErrorString", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaFuncSetCacheConfig", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaFuncGetAttributes", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaStreamCreate", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaStreamDestroy", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaStreamSynchronize", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaStreamQuery", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventCreate", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventCreateWithFlags", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventRecord", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventQuery", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventSynchronize", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventDestroy", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaEventElapsedTime", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForDevice", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForHost", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaThreadSynchronize", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaThreadSetLimit", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaThreadGetLimit", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaDriverGetVersion", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaRuntimeGetVersion", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGetExportTable", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnregisterResource", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceSetMapFlags", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsMapResources", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnmapResources", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceGetMappedPointer", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "cudaGraphicsSubResourceGetMappedArray", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaRegisterFatBinary", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "__cudaUnregisterFatBinary", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterVar", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterTexture", (PROC)CudaMemoryDebug::Unimplemented);
	cu->hook_manager->HookImport(cuda_module_name, "__cudaRegisterSurface", (PROC)CudaMemoryDebug::Unimplemented);
//	cu->hook_manager->HookImport(cuda_module_name, "cudaRegisterFunction", (PROC)CudaMemoryDebug::Unimplemented);

    return true;
}

void CudaMemoryDebug::ExitHandler()
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    // Check if there are no unfreed blocks.
    for (int i = 0; i < cu->alloc_list.size(); ++i)
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

CudaMemoryDebug::return_type CudaMemoryDebug::CheckSinglePtrOverwrite(const data * d)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    CallStackInfo * csi = CallStackInfo::Singleton();
    if (! d->is_host)
    {
        unsigned char * hostbuffer = (unsigned char *)malloc(d->size);
        if (! hostbuffer)
            return NOT_OK;

		cudaError_t e1;
		if (! cu->do_emulation)
		{
			typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CudaMemoryDebug::Memcpy);
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

CudaMemoryDebug::return_type CudaMemoryDebug::CheckOverwrite()
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    // Check if there are overwrites.
    for (int i = 0; i < cu->alloc_list.size(); ++i)
    {
        data d = cu->alloc_list[i];
        cu->CheckSinglePtrOverwrite(&d);
    }
    return OK;      
}

int CudaMemoryDebug::FindAllocatedBlock(const void * pointer)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    int i;
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

bool CudaMemoryDebug::IsBadPointer(const void * ptr)
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


char * CudaMemoryDebug::Context(int lines)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    if (cu->global_context && *cu->global_context)
        return cu->global_context;

    CallStackInfo * csi = CallStackInfo::Singleton();
    char * context = csi->Context(lines);
    return context;
}

void CudaMemoryDebug::MakeContext(char * file_name, int line)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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

void CudaMemoryDebug::Unimplemented()
{
	std::cout << "FATAL ERROR: Unimplemented function!!!!\n";
	exit(1);
}

cudaError_t CUDARTAPI CudaMemoryDebug::Malloc(void ** ptr, size_t size)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();

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

cudaError_t CUDARTAPI CudaMemoryDebug::Free(void * ptr)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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


cudaError_t CUDARTAPI CudaMemoryDebug::HostAlloc(void ** ptr, size_t size, unsigned int flags)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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

cudaError_t CUDARTAPI CudaMemoryDebug::FreeHost(void * ptr)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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

cudaError_t CUDARTAPI CudaMemoryDebug::HostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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

    int i;
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

cudaError_t CUDARTAPI CudaMemoryDebug::Memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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

cudaError_t CUDARTAPI CudaMemoryDebug::Memset(void * dst, int value, size_t count)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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


cudaError_t CUDARTAPI CudaMemoryDebug::ThreadExit()
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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

cudaError_t CudaMemoryDebug::GetLastError()
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
	if (! cu->do_emulation)
		return (*ptrCudaGetLastError)();
	else
		return cudaSuccess;
}

void** CudaMemoryDebug::RegisterFatBinary(void *fatCubin)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();

		printf("cu emul %s\n", cu->do_emulation? "yes":"no");
	printf("HI HI HI\n");
	if (cu->do_crash)
	{
		printf("here\n");
		//assert(false);
		// if this fails, try another...
		int x = 0;
		int y = 0;
		cu = (CudaMemoryDebug*)(x/y);
		int z = x/y;
		printf ("WHAT THE F!\n");
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
				EMULATOR * emulator = EMULATOR::Singleton();
				emulator->Extract_From_Source(code);
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
	    return (*ptrCudaRegisterFatBinary)(fatCubin);
	else
		return 0;
}

cudaError_t CUDARTAPI CudaMemoryDebug::Launch(const char *entry)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
	if (! cu->do_emulation)
	    return (*ptrCudaLaunch)(entry);
	else
	{
		EMULATOR * emulator = EMULATOR::Singleton();
		emulator->Execute((void*)entry);
		return cudaSuccess;
	}
}

void CUDARTAPI CudaMemoryDebug::RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
	if (! cu->do_emulation)
	    (*ptrCudaRegisterFunction)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
	else
	{
		EMULATOR * emulator = EMULATOR::Singleton();
		emulator->RegisterFunction((void*)hostFun, deviceFun);
	}
}

cudaError_t CUDARTAPI CudaMemoryDebug::ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
	// set up dimensions, shared memory, and stream for the kernel launch.
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
	if (! cu->do_emulation)
	    return (*ptrCudaConfigureCall)(gridDim, blockDim, sharedMem, stream);
	else
	{
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->ConfigureCall(gridDim, blockDim, sharedMem, stream);
	}
}

cudaError_t CUDARTAPI CudaMemoryDebug::SetupArgument(const void *arg, size_t size, size_t offset)
{
	// arg contains pointer to the argument for the function call.
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
	if (! cu->do_emulation)
	    return (*ptrCudaSetupArgument)(arg, size, offset);
	else
	{
		EMULATOR * emulator = EMULATOR::Singleton();
		return emulator->SetupArgument(arg, size, offset);
	}
}


///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  Options for behavior of this debugging wrapper.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

CudaMemoryDebug::return_type CudaMemoryDebug::SetPaddingSize(size_t s)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();

    cu->padding_size = s;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetPaddingSize called, " << context << ".\n";
        (*cu->output_stream) << " Padding size now " << s << "\n\n";
    }
    return OK;
}

CudaMemoryDebug::return_type CudaMemoryDebug::SetPaddingByte(unsigned char b)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();

    cu->padding_byte = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetPaddingByte called, " << context << ".\n";
        (*cu->output_stream) << " Padding byte now " << b << "\n\n";
    }
    return OK;
}

CudaMemoryDebug::return_type CudaMemoryDebug::SetDevicePointerToFirstByteInBlock(bool b)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();

    cu->device_pointer_to_first_byte_in_block = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDevicePointerToFirstByteInBlock called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}

CudaMemoryDebug::return_type CudaMemoryDebug::SetOutputStream(std::ostream * fp)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();
    cu->output_stream = fp;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetOutputStream called, " << context << ".\n\n";
    }
    return OK;
}

CudaMemoryDebug::return_type CudaMemoryDebug::SetTraceAllCalls(bool b)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();

    cu->trace_all_calls = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetTraceAllCalls called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}

CudaMemoryDebug::return_type CudaMemoryDebug::SetQuitOnError(bool b)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();

    cu->quit_on_error = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetQuitOnError called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}


CudaMemoryDebug::return_type CudaMemoryDebug::SetDoNotCallCudaAfterSanityCheckFail(bool b)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
    char * context = cu->Context();

    cu->do_not_call_cuda_after_sanity_check_fail = b;
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDoNotCallCudaAfterSanityCheckFail called, " << context << ".\n";
        (*cu->output_stream) << " value now " << (b?"true":"false") << "\n\n";
    }
    return OK;
}

CudaMemoryDebug::return_type CudaMemoryDebug::CopyOptions(CudaMemoryDebug * ptr)
{
    CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
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
