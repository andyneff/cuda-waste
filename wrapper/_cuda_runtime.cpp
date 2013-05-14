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
#include "_cuda_runtime.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#define ZLIB_WINAPI 1
#include "zlib.h"

extern char * file_name_tail(char * file_name);


void _CUDA_RUNTIME::WrapModule(char * cuda_module_name)
{
    // Add Driver API hooking.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    HOOK_MANAGER * hook_manager = cu->hook_manager;
    bool complain = false;
    if (hook_manager->HookImport(cuda_module_name, "cudaMalloc3D", (PROC)_CUDA_RUNTIME::Unimplemented, false))
    {
        if (this->did_wrap)
            return;
        this->did_wrap = true;
        this->hModule = hook_manager->GetModule(cuda_module_name);
        bool complain = false;
        // Hook cudart library.
        // Any unknown after this should be flagged.
        hook_manager->HookImport(cuda_module_name, "cudaMalloc3DArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset3D", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy3D", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)_CUDA_RUNTIME::Malloc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocHost", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocPitch", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMallocArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)_CUDA_RUNTIME::Free, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)_CUDA_RUNTIME::FreeHost, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFreeArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)_CUDA_RUNTIME::HostAlloc, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)_CUDA_RUNTIME::HostGetDevicePointer, complain);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetFlags", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemGetInfo", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)_CUDA_RUNTIME::Memcpy, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyArrayToArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2D", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DArrayToArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbol", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbol", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArrayAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArrayAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArrayAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArrayAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbolAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbolAsync", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)_CUDA_RUNTIME::Memset, complain);
        hook_manager->HookImport(cuda_module_name, "cudaMemset2D", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolAddress", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolSize", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceCount", (PROC)_CUDA_RUNTIME::_cudaGetDeviceCount, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceProperties", (PROC)_CUDA_RUNTIME::_cudaGetDeviceProperties, complain);
        hook_manager->HookImport(cuda_module_name, "cudaChooseDevice", (PROC)_CUDA_RUNTIME::_cudaChooseDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDevice", (PROC)_CUDA_RUNTIME::_cudaSetDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetDevice", (PROC)_CUDA_RUNTIME::_cudaGetDevice, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetValidDevices", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDeviceFlags", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaBindTexture", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaBindTexture2D", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaBindTextureToArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaUnbindTexture", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetTextureAlignmentOffset", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetTextureReference", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaBindSurfaceToArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
    // not in library??????????
    // hook_manager->HookImport(cuda_module_name,
    // "cudaGetSurfaceAlignmentOffset",
    // (PROC)CUDA_WRAPPER::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceReference", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetChannelDesc", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaCreateChannelDesc", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)_CUDA_RUNTIME::_cudaGetLastError, complain);
        hook_manager->HookImport(cuda_module_name, "cudaPeekAtLastError", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetErrorString", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)_CUDA_RUNTIME::_cudaConfigureCall, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)_CUDA_RUNTIME::_cudaSetupArgument, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFuncSetCacheConfig", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)_CUDA_RUNTIME::_cudaLaunch, complain);
        hook_manager->HookImport(cuda_module_name, "cudaFuncGetAttributes", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamCreate", (PROC)_CUDA_RUNTIME::_cudaStreamCreate, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamDestroy", (PROC)_CUDA_RUNTIME::_cudaStreamDestroy, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamSynchronize", (PROC)_CUDA_RUNTIME::_cudaStreamSynchronize, complain);
        hook_manager->HookImport(cuda_module_name, "cudaStreamQuery", (PROC)_CUDA_RUNTIME::_cudaStreamQuery, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventCreate", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventCreateWithFlags", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventRecord", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventQuery", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventSynchronize", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventDestroy", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaEventElapsedTime", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForDevice", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForHost", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)_CUDA_RUNTIME::ThreadExit, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSynchronize", (PROC)_CUDA_RUNTIME::_cudaThreadSynchronize, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSetLimit", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaThreadGetLimit", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaDriverGetVersion", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaRuntimeGetVersion", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGetExportTable", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnregisterResource", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceSetMapFlags", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsMapResources", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnmapResources", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceGetMappedPointer", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsSubResourceGetMappedArray", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterFatBinary", (PROC)_CUDA_RUNTIME::_cudaRegisterFatBinary, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaUnregisterFatBinary", (PROC)_CUDA_RUNTIME::_cudaUnregisterFatBinary, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterVar", (PROC)_CUDA_RUNTIME::_cudaRegisterVar, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterTexture", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterSurface", (PROC)_CUDA_RUNTIME::Unimplemented, complain);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterFunction", (PROC)_CUDA_RUNTIME::_cudaRegisterFunction, complain);
    }
}
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
//
//  Wrappers for CUDA memory management API.
//
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////


cudaError_t CUDARTAPI _CUDA_RUNTIME::Malloc(void ** ptr, size_t size)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

    void * local = 0;

    char * context = cu->Context();

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "cudaMalloc called, " << context << ".\n\n";
    }
            
    if (cu->IsBadPointer(ptr))
    {
        (*cu->output_stream) << "Bad pointer passed to cudaMalloc("
            << ptr << ", ..., ...).\n";
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }

    cudaError_t resetErrKernel = _cudaGetLastError();

    // Allocate a cuda memory buffer that is "bytes" long plus padding on either side.
    typePtrCudaMalloc proc;
    if (! cu->do_emulation)
    {
        typePtrCudaMalloc proc = (typePtrCudaMalloc)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Malloc);
        cudaError_t e1 = (*proc)((void**)&local, size + 2 * cu->padding_size);
        if (e1 != 0)
        {
            (*cu->output_stream) << "cudaMalloc failed."
                << " Return value = "
                << e1 << ".\n";
            (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
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
        (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
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
        typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
        cudaError_t e2 = (*proc)(local, hostbuffer, size + 2 * cu->padding_size, cudaMemcpyHostToDevice);
        if (e2 != 0)
        {
            (*cu->output_stream) << "cudaMemcpy failed, rc = " << e2 << ", in cudaMalloc. This call is used to initialize the device buffer.\n";
            (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
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
    CUDA_WRAPPER::data d;
    d.ptr = local;
    d.size = size + 2 * cu->padding_size;
    d.is_host = false;
    d.context = strdup(cu->Context());
    cu->alloc_list.push_back(d);
    *ptr = ((char*)local) + cu->padding_size;
    return cudaSuccess;     
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::Free(void * ptr)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

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
            typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
            cudaError_t err = (*proc)(ptr);
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
            typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
            cudaError_t err = (*proc)(ptr);
            return err;
        }
        return cudaErrorMemoryAllocation;
    }
    CUDA_WRAPPER::data * d = &cu->alloc_list[di];
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
            typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
            cudaError_t err = (*proc)(ptr);
            return err;
        }
        return cudaErrorMemoryAllocation;
    }
    cu->alloc_list.erase(cu->alloc_list.begin() + di);
    cu->CheckSinglePtrOverwrite(d);
    if (! cu->do_emulation)
    {
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
        cudaError_t err = (*proc)(local);
        return err;
    } else
    {
        free(local);
        return cudaSuccess;
    }
}


cudaError_t CUDARTAPI _CUDA_RUNTIME::HostAlloc(void ** ptr, size_t size, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    void * local = 0;
    char * context = cu->Context();

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
    typePtrCudaHostAlloc proc = (typePtrCudaHostAlloc)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::HostAlloc);
    cudaError_t e1 = (*proc)((void**)&local, size + 2 * cu->padding_size, flags);
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
    CUDA_WRAPPER::data d;
    d.ptr = local;
    d.size = size + 2 * cu->padding_size;
    d.context = strdup(cu->Context());
    d.is_host = true;
    cu->alloc_list.push_back(d);
    *ptr = ((char*)local) + cu->padding_size;
    return cudaSuccess;     
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::FreeHost(void * ptr)
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
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorInvalidDevicePointer;
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
        cudaError_t err = (*proc)(ptr);
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
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
        cudaError_t err = (*proc)(ptr);
        return err;
    }
    CUDA_WRAPPER::data * d = &cu->alloc_list[di];
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
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Free);
        cudaError_t err = (*proc)(ptr);
        return err;
    }
    cu->alloc_list.erase(cu->alloc_list.begin() + di);
    cu->CheckSinglePtrOverwrite(d);
    typePtrCudaFreeHost proc = (typePtrCudaFreeHost)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::FreeHost);
    cudaError_t err = (*proc)(ptr);
    if (err)
    {
        if (cu->quit_on_error)
            exit(1);
    }
    return err;
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::HostGetDevicePointer(void ** pDevice, void * pHost, unsigned int flags)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * file_name = 0;
    int line = 0;
    char * context = cu->Context();

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
    CUDA_WRAPPER::data d = cu->alloc_list[i];
    if (! d.is_host)
    {
        (*cu->output_stream) << "pHost pointer passed to HostGetDevicePointer(..., " << pHost << ", ...) is invalid.\n";
        (*cu->output_stream) << " The pointer is not a pointer created by cudaHostAlloc\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in file "
            << file_name_tail(file_name) << ", line " << line << ".\n\n";
    }
    typePtrCudaHostGetDevicePointer proc = (typePtrCudaHostGetDevicePointer)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::HostGetDevicePointer);
    cudaError_t err = (*proc)(pDevice, local, flags);
    if (err)
    {
        return err;
    }
    // Map pDevice correctly.
    *pDevice = (void*) (((char*)*pDevice) + cu->padding_size);
    return err;
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::Memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
            cudaError_t err = (*proc)(dst, src, count, kind);
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
            cudaError_t err = (*proc)(dst, src, count, kind);
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
        CUDA_WRAPPER::data * ddst = 0;
        CUDA_WRAPPER::data * dsrc = 0;
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
            err = (*proc)(dst, src, count, kind);
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
        CUDA_WRAPPER::data * ddst = 0;
        CUDA_WRAPPER::data * dsrc = 0;
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
            err = (*proc)(dst, src, count, kind);
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
    else
    {
        typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
        cudaError_t err = (*proc)(dst, src, count, kind);
        return err;
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::Memset(void * dst, int value, size_t count)
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
        typePtrCudaMemset proc = (typePtrCudaMemset)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memset);
        cudaError_t err = (*proc)(dst, value, count);
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
        typePtrCudaMemset proc = (typePtrCudaMemset)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memset);
        cudaError_t err = (*proc)(dst, value, count);
        return err;
    }
    CUDA_WRAPPER::data * ddst = 0;
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

    if (cu->CheckSinglePtrOverwrite(ddst) != CUDA_WRAPPER::OK)
    {
        (*cu->output_stream) << "Destination block in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid -- overwritten.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }
    // Perform copy.
	cudaError_t err;
    if (! cu->do_emulation)
    {
	    typePtrCudaMemset proc = (typePtrCudaMemset)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memset);
		err = (*proc)(dst, value, count);
        if (err != 0)
        {
            (*cu->output_stream) << "cudaMemset failed."
                << " Return value = "
                << err << ".\n";
            (*cu->output_stream) << " This check was performed in " << cu->Context() << ".\n\n";
            if (cu->quit_on_error)
                exit(1);
            if (cu->do_not_call_cuda_after_sanity_check_fail)
                return err;
            return err;
        }
    } else
    {
        memset(dst, value, count);
        err = cudaSuccess;
    }

    // Perform overwrite check again.
    if (cu->CheckSinglePtrOverwrite(ddst) != CUDA_WRAPPER::OK)
    {
        (*cu->output_stream) << "Destination block in Memset("
            << "dst = " << dst
            << ", ..., ...) is invalid -- overwritten.\n";
        (*cu->output_stream) << " This check was performed during a CUDA call in "
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
        if (cu->do_not_call_cuda_after_sanity_check_fail)
            return cudaErrorMemoryAllocation;
    }
    return err;
}


cudaError_t CUDARTAPI _CUDA_RUNTIME::ThreadExit()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

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
    {
        typePtrCudaThreadExit proc = (typePtrCudaThreadExit)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::ThreadExit);
        return (*proc)();
    }
    else
        return cudaSuccess;
}

cudaError_t _CUDA_RUNTIME::_cudaGetLastError()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetLastError called, " << context << ".\n\n";
    }

    if (! cu->do_emulation)
    {
        typePtrCudaGetLastError proc = (typePtrCudaGetLastError)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaGetLastError);
        return (*proc)();
    }
    else
        return cudaSuccess;
}


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


void** _CUDA_RUNTIME::_cudaRegisterFatBinary(void *fatCubin)
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

		    thisVersion = stoi(versionString);
		    if(thisVersion > ptxVersion)
		    {
			    ptxVersion = thisVersion;
			    ptx = binary->ptx[i].ptx;
		    }
	    }
	    (*cu->output_stream) << " Selected version " << ptxVersion << "\n";

	    EMULATOR * emulator = EMULATOR::Singleton();
	    emulator->Parse(profile, ptx);
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

	    EMULATOR * emulator = EMULATOR::Singleton();
	    emulator->Parse(profile, ptx);

    }

    if (! cu->do_emulation)
    {
	    typePtrCudaRegisterFatBinary proc = (typePtrCudaRegisterFatBinary)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaRegisterFatBinary);
	    return (*proc)(fatCubin);
    } else
	    return 0;
}

void CUDARTAPI _CUDA_RUNTIME::_cudaUnregisterFatBinary(void **fatCubinHandle)
{
    // Should probably do something like free the ast...
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaUnregisterFatBinary called, " << context << ".\n\n";
    }
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaUnregisterFatBinary called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaUnregisterFatBinary proc = (typePtrCudaUnregisterFatBinary)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaUnregisterFatBinary);
        (*proc)(fatCubinHandle);
    }
}

void CUDARTAPI _CUDA_RUNTIME::_cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaUnregisterFatBinary called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaRegisterVar proc = (typePtrCudaRegisterVar)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaRegisterVar);
        (*proc)(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    }
	else
	{
        // no op for now.
	}
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaLaunch(const char *entry)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaLaunch called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaLaunch proc = (typePtrCudaLaunch)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaLaunch);
        return (*proc)(entry);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        emulator->_cudaLaunch(entry);
        return cudaSuccess;
    }
}

void CUDARTAPI _CUDA_RUNTIME::_cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaRegisterFunction called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaRegisterFunction proc = (typePtrCudaRegisterFunction)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaRegisterFunction);
        (*proc)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        emulator->_cudaRegisterFunction((void*)hostFun, deviceFun);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaConfigureCall called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaConfigureCall proc = (typePtrCudaConfigureCall)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaConfigureCall);
        return (*proc)(gridDim, blockDim, sharedMem, stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaSetupArgument called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaSetupArgument proc = (typePtrCudaSetupArgument)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaSetupArgument);
        return (*proc)(arg, size, offset);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaSetupArgument(arg, size, offset);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaThreadSynchronize(void)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaThreadSynchronize called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaThreadSynchronize proc = (typePtrCudaThreadSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaThreadSynchronize);
        return (*proc)();
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaThreadSynchronize();
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetDevice(int *device)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetDevice called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaGetDevice proc = (typePtrCudaGetDevice)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaGetDevice);
        return (*proc)(device);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaGetDevice(device);
    }
}


cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetDeviceProperties called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaGetDeviceProperties proc = (typePtrCudaGetDeviceProperties)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaGetDeviceProperties);
        return (*proc)(prop, device);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaGetDeviceProperties(prop, device);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaGetDeviceCount(int *count)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaGetDeviceCount called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaGetDeviceCount proc = (typePtrCudaGetDeviceCount)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaGetDeviceCount);
        return (*proc)(count);
    } else
    {
        *count = 1;
        return cudaSuccess;
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaChooseDevice called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        ptrCudaChooseDevice proc = (ptrCudaChooseDevice)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaChooseDevice);
        return (*proc)(device, prop);
    } else
    {
        *device = 0;
        return cudaSuccess;
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaSetDevice(int device)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaSetDevice called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        typePtrCudaSetDevice proc = (typePtrCudaSetDevice)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaSetDevice);
        return (*proc)(device);
    } else
    {
        return cudaSuccess;
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamCreate(cudaStream_t *pStream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaStreamCreate called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        ptrCudaStreamCreate proc = (ptrCudaStreamCreate)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaStreamCreate);
        return (*proc)(pStream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamCreate(pStream);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamDestroy(cudaStream_t stream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaStreamDestroy called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        ptrCudaStreamDestroy proc = (ptrCudaStreamDestroy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaStreamDestroy);
        return (*proc)(stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamDestroy(stream);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamSynchronize(cudaStream_t stream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaStreamSynchronize called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        ptrCudaStreamSynchronize proc = (ptrCudaStreamSynchronize)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaStreamSynchronize);
        return (*proc)(stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamSynchronize(stream);
    }
}

cudaError_t CUDARTAPI _CUDA_RUNTIME::_cudaStreamQuery(cudaStream_t stream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (cu->trace_all_calls)
    {
        char * context = cu->Context();
        (*cu->output_stream) << "cudaStreamQuery called, " << context << ".\n\n";
    }
    if (! cu->do_emulation)
    {
        ptrCudaStreamQuery proc = (ptrCudaStreamQuery)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::_cudaStreamQuery);
        return (*proc)(stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamQuery(stream);
    }
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
    ++it;
    ++it;
    // 2) The instruction at *it will be the instruction just after the call.
    // Assuming that the call function is relative, step back 4 bytes,
    // get the value to add to *it.  This is a jmp instruction.  Step forward
    // 2 bytes to get the address of the iat, which contains another jump
    // instruction.
    unsigned char * p = (unsigned char *)*it;
    // step back f bytes and get opcode.
    unsigned char * pm5 = p - 5;
    void * iat;
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
        HOOK_MANAGER * hm = cu->hook_manager;
        HookedFunction * hf = hm->FindHook(iat);
        if (hf)
        {
            std::cout << "Function " << hf->Get_FuncName() << " needs to be implemented to debug this program.\n";
        }
    }
    delete call_stack;
    exit(1);
}

