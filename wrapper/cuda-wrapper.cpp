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
    cu->do_debug_halt = false;
    cu->_cuda = new _CUDA();
}

void CUDA_WRAPPER::Unimplemented()
{
    std::cout << "FATAL ERROR: Unimplemented function!!!!\n";
    // get call back and get hooked name to know what function is
    // unimplemented.

    // Figure out the name of the function that is bound to CUDA_WRAPPER::Unimplemented.
    // Not especially portable, but it is general.
    // 1) Go up call stack to caller of this function.  Call stack includes
    // CallTree, Unimplemented, then the caller.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CallStackInfo * csi = CallStackInfo::Singleton();
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
        HookManager * hm = cu->hook_manager;
        HookedFunction * hf = hm->FindHook(iat);
        if (hf)
        {
            std::cout << "Function " << hf->Get_FuncName() << " needs to be implemented to debug this program.\n";
        }
    }
    delete call_stack;
    exit(1);
}

void CUDA_WRAPPER::DoInit()
{
    // This routine is static, so get the singleton for the cuda
    // wrapper.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CallStackInfo * csi = CallStackInfo::Singleton();
    csi->ClassifyAsPrefix("call-stack-info.cpp");
    csi->ClassifyAsPrefix("cuda-wrapper.cpp");
    if (atexit(CUDA_WRAPPER::ExitHandler))
    {
        char * context = cu->Context(3);
        (*cu->output_stream) << "Cannot register CUDA_WRAPPER exit handler during initialization.  Memory leaks will not be detected.\n";
        (*cu->output_stream) << " Call stack is:\n"
            << context << ".\n\n";
        if (cu->quit_on_error)
            exit(1);
    }
    HookManager * hm = new HookManager();
    cu->hook_manager = hm;
    // Force load of CUDA driver API, so it can be hooked.
    LoadLibraryA("nvcuda.dll");
    // Make sure no sneaky way to access nvcuda.
    hm->HookSystemFuncs();
}

bool CUDA_WRAPPER::WrapModule(char * cuda_module_name)
{
    // There are two API sets that could be substituted, one for
    // the cuda runtime library (cudart*.dll), and the other for cuda
    // driver (nvcuda.dll).  If we recognize any, all hooks should be
    // defined.  Otherwise, there will be odd behavior.
    this->_cuda->WrapModule();

    if (hook_manager->HookImport(cuda_module_name, "cudaMalloc3D", (PROC)CUDA_WRAPPER::Unimplemented, false))
    {
        // Hook cudart library.
        // Any unknown after this should be flagged.
        hook_manager->HookImport(cuda_module_name, "cudaMalloc3DArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemset3D", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy3D", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy3DAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMalloc", (PROC)CUDA_WRAPPER::Malloc, true);
        hook_manager->HookImport(cuda_module_name, "cudaMallocHost", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMallocPitch", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMallocArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaFree", (PROC)CUDA_WRAPPER::Free, true);
        hook_manager->HookImport(cuda_module_name, "cudaFreeHost", (PROC)CUDA_WRAPPER::FreeHost, true);
        hook_manager->HookImport(cuda_module_name, "cudaFreeArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaHostAlloc", (PROC)CUDA_WRAPPER::HostAlloc, true);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetDevicePointer", (PROC)CUDA_WRAPPER::HostGetDevicePointer, true);
        hook_manager->HookImport(cuda_module_name, "cudaHostGetFlags", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemGetInfo", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy", (PROC)CUDA_WRAPPER::Memcpy, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyArrayToArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2D", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DArrayToArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbol", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbol", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DToArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpy2DFromArrayAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyToSymbolAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemcpyFromSymbolAsync", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemset", (PROC)CUDA_WRAPPER::Memset, true);
        hook_manager->HookImport(cuda_module_name, "cudaMemset2D", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolAddress", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetSymbolSize", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceCount", (PROC)CUDA_WRAPPER::_cudaGetDeviceCount, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetDeviceProperties", (PROC)CUDA_WRAPPER::_cudaGetDeviceProperties, true);
        hook_manager->HookImport(cuda_module_name, "cudaChooseDevice", (PROC)CUDA_WRAPPER::_cudaChooseDevice, true);
        hook_manager->HookImport(cuda_module_name, "cudaSetDevice", (PROC)CUDA_WRAPPER::_cudaSetDevice, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetDevice", (PROC)CUDA_WRAPPER::_cudaGetDevice, true);
        hook_manager->HookImport(cuda_module_name, "cudaSetValidDevices", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaSetDeviceFlags", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaBindTexture", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaBindTexture2D", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaBindTextureToArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaUnbindTexture", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetTextureAlignmentOffset", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetTextureReference", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaBindSurfaceToArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
    // not in library??????????
    // hook_manager->HookImport(cuda_module_name,
    // "cudaGetSurfaceAlignmentOffset",
    // (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetSurfaceReference", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetChannelDesc", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaCreateChannelDesc", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetLastError", (PROC)CUDA_WRAPPER::GetLastError, true);
        hook_manager->HookImport(cuda_module_name, "cudaPeekAtLastError", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetErrorString", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaConfigureCall", (PROC)CUDA_WRAPPER::_cudaConfigureCall, true);
        hook_manager->HookImport(cuda_module_name, "cudaSetupArgument", (PROC)CUDA_WRAPPER::_cudaSetupArgument, true);
        hook_manager->HookImport(cuda_module_name, "cudaFuncSetCacheConfig", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaLaunch", (PROC)CUDA_WRAPPER::_cudaLaunch, true);
        hook_manager->HookImport(cuda_module_name, "cudaFuncGetAttributes", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaStreamCreate", (PROC)CUDA_WRAPPER::_cudaStreamCreate, true);
        hook_manager->HookImport(cuda_module_name, "cudaStreamDestroy", (PROC)CUDA_WRAPPER::_cudaStreamDestroy, true);
        hook_manager->HookImport(cuda_module_name, "cudaStreamSynchronize", (PROC)CUDA_WRAPPER::_cudaStreamSynchronize, true);
        hook_manager->HookImport(cuda_module_name, "cudaStreamQuery", (PROC)CUDA_WRAPPER::_cudaStreamQuery, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventCreate", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventCreateWithFlags", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventRecord", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventQuery", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventSynchronize", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventDestroy", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaEventElapsedTime", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForDevice", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaSetDoubleForHost", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaThreadExit", (PROC)CUDA_WRAPPER::ThreadExit, true);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSynchronize", (PROC)CUDA_WRAPPER::_cudaThreadSynchronize, true);
        hook_manager->HookImport(cuda_module_name, "cudaThreadSetLimit", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaThreadGetLimit", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaDriverGetVersion", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaRuntimeGetVersion", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGetExportTable", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnregisterResource", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceSetMapFlags", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsMapResources", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsUnmapResources", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsResourceGetMappedPointer", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "cudaGraphicsSubResourceGetMappedArray", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterFatBinary", (PROC)CUDA_WRAPPER::_cudaRegisterFatBinary, true);
        hook_manager->HookImport(cuda_module_name, "__cudaUnregisterFatBinary", (PROC)CUDA_WRAPPER::_cudaUnregisterFatBinary, true);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterVar", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterTexture", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterSurface", (PROC)CUDA_WRAPPER::Unimplemented, true);
        hook_manager->HookImport(cuda_module_name, "__cudaRegisterFunction", (PROC)CUDA_WRAPPER::_cudaRegisterFunction, true);
    }

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
                    (*cu->output_stream) << "Overwrite of cuda memory block footer.\n";
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

void CUDA_WRAPPER::SetEmulationMode(int yes_no)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (yes_no)
        cu->do_emulation = true;
    else
        cu->do_emulation = false;
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

CUDA_WRAPPER::return_type CUDA_WRAPPER::RunDevice(char * device)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();

    cu->device = device;
    EMULATOR * emulator = EMULATOR::Singleton();
    emulator->RunDevice(device);

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDevice called, " << context << ".\n";
        (*cu->output_stream) << " Device now " << device << "\n\n";
    }
    return OK;
}

void CUDA_WRAPPER::SetTrace(int level)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (level >= 10)
    {
        cu->do_debug_halt = true;
        level -= 10;
    }

    if (cu->do_debug_halt)
    {
        printf("Halting to invoke debugger...\n");
        _asm {
            int 3;
        }
    }

    EMULATOR * emulator = EMULATOR::Singleton();
    emulator->SetTrace(level);
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
        typePtrCudaMalloc proc = (typePtrCudaMalloc)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Malloc);
        cudaError_t e1 = (*proc)((void**)&local, size + 2 * cu->padding_size);
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
        typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
        cudaError_t e2 = (*proc)(local, hostbuffer, size + 2 * cu->padding_size, cudaMemcpyHostToDevice);
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
            typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
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
            typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
            cudaError_t err = (*proc)(ptr);
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
            typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
            cudaError_t err = (*proc)(ptr);
            return err;
        }
        return cudaErrorMemoryAllocation;
    }
    cu->alloc_list.erase(cu->alloc_list.begin() + di);
    cu->CheckSinglePtrOverwrite(d);
    if (! cu->do_emulation)
    {
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
        cudaError_t err = (*proc)(local);
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
    typePtrCudaHostAlloc proc = (typePtrCudaHostAlloc)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::HostAlloc);
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
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
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
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
        cudaError_t err = (*proc)(ptr);
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
        typePtrCudaFree proc = (typePtrCudaFree)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Free);
        cudaError_t err = (*proc)(ptr);
        return err;
    }
    cu->alloc_list.erase(cu->alloc_list.begin() + di);
    cu->CheckSinglePtrOverwrite(d);
    typePtrCudaFreeHost proc = (typePtrCudaFreeHost)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::FreeHost);
    cudaError_t err = (*proc)(ptr);
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
    typePtrCudaHostGetDevicePointer proc = (typePtrCudaHostGetDevicePointer)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::HostGetDevicePointer);
    cudaError_t err = (*proc)(pDevice, local, flags);
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
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
            typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
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
            CheckSinglePtrOverwrite(dsrc);
        return err;
    }
    else
    {
        typePtrCudaMemcpy proc = (typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memcpy);
        cudaError_t err = (*proc)(dst, src, count, kind);
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
        typePtrCudaMemset proc = (typePtrCudaMemset)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memset);
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
        typePtrCudaMemset proc = (typePtrCudaMemset)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memset);
        cudaError_t err = (*proc)(dst, value, count);
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
    typePtrCudaMemset proc = (typePtrCudaMemset)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::Memset);
    cudaError_t err = (*proc)(dst, value, count);
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
    {
        typePtrCudaThreadExit proc = (typePtrCudaThreadExit)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::ThreadExit);
        return (*proc)();
    }
    else
        return cudaSuccess;
}

cudaError_t CUDA_WRAPPER::GetLastError()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaGetLastError proc = (typePtrCudaGetLastError)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::GetLastError);
        return (*proc)();
    }
    else
        return cudaSuccess;
}

void** CUDA_WRAPPER::_cudaRegisterFatBinary(void *fatCubin)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

    // std::cout << "NEW FATBIN\n";
    // Get PTX code from the record.
    __cudaFatCudaBinary * fcb = (__cudaFatCudaBinary *)fatCubin;
    if (fcb)
    {
        __cudaFatPtxEntry * ptx = fcb->ptx;
        for ( ; ptx && ptx->gpuProfileName; ptx++)
        {
            char * profile = ptx->gpuProfileName;
            char * code = ptx->ptx;
            EMULATOR * emulator = EMULATOR::Singleton();
            emulator->Parse(profile, code);
        }

        // ELF contains just in time code for every PTX.
        // Execution will depend on picking which one for the device.
        //__cudaFatElfEntry * elf = fcb->elf;
        //for ( ; elf; elf = elf->next)
        //{
        //char * code = elf->elf;
        //}
    }
    if (! cu->do_emulation)
    {
        typePtrCudaRegisterFatBinary proc = (typePtrCudaRegisterFatBinary)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaRegisterFatBinary);
        return (*proc)(fatCubin);
    } else
        return 0;
}

void CUDARTAPI CUDA_WRAPPER::_cudaUnregisterFatBinary(void **fatCubinHandle)
{
    // Should probably do something like free the ast...
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaUnregisterFatBinary proc = (typePtrCudaUnregisterFatBinary)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaUnregisterFatBinary);
        (*proc)(fatCubinHandle);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaLaunch(const char *entry)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaLaunch proc = (typePtrCudaLaunch)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaLaunch);
        return (*proc)(entry);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        emulator->_cudaLaunch(entry);
        return cudaSuccess;
    }
}

void CUDARTAPI CUDA_WRAPPER::_cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaRegisterFunction proc = (typePtrCudaRegisterFunction)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaRegisterFunction);
        (*proc)(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        emulator->_cudaRegisterFunction((void*)hostFun, deviceFun);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0))
{
    // set up dimensions, shared memory, and stream for the kernel launch.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaConfigureCall proc = (typePtrCudaConfigureCall)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaConfigureCall);
        return (*proc)(gridDim, blockDim, sharedMem, stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaSetupArgument proc = (typePtrCudaSetupArgument)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaSetupArgument);
        return (*proc)(arg, size, offset);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaSetupArgument(arg, size, offset);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaThreadSynchronize(void)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaThreadSynchronize proc = (typePtrCudaThreadSynchronize)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaThreadSynchronize);
        return (*proc)();
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaThreadSynchronize();
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaGetDevice(int *device)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaGetDevice proc = (typePtrCudaGetDevice)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaGetDevice);
        return (*proc)(device);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaGetDevice(device);
    }
}


cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaGetDeviceProperties proc = (typePtrCudaGetDeviceProperties)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaGetDeviceProperties);
        return (*proc)(prop, device);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaGetDeviceProperties(prop, device);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaGetDeviceCount(int *count)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaGetDeviceCount proc = (typePtrCudaGetDeviceCount)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaGetDeviceCount);
        return (*proc)(count);
    } else
    {
        *count = 1;
        return cudaSuccess;
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptrCudaChooseDevice proc = (ptrCudaChooseDevice)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaChooseDevice);
        return (*proc)(device, prop);
    } else
    {
        *device = 0;
        return cudaSuccess;
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaSetDevice(int device)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        typePtrCudaSetDevice proc = (typePtrCudaSetDevice)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaSetDevice);
        return (*proc)(device);
    } else
    {
        return cudaSuccess;
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaStreamCreate(cudaStream_t *pStream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptrCudaStreamCreate proc = (ptrCudaStreamCreate)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaStreamCreate);
        return (*proc)(pStream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamCreate(pStream);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaStreamDestroy(cudaStream_t stream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptrCudaStreamDestroy proc = (ptrCudaStreamDestroy)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaStreamDestroy);
        return (*proc)(stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamDestroy(stream);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaStreamSynchronize(cudaStream_t stream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptrCudaStreamSynchronize proc = (ptrCudaStreamSynchronize)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaStreamSynchronize);
        return (*proc)(stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamSynchronize(stream);
    }
}

cudaError_t CUDARTAPI CUDA_WRAPPER::_cudaStreamQuery(cudaStream_t stream)
{
    // arg contains pointer to the argument for the function call.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    if (! cu->do_emulation)
    {
        ptrCudaStreamQuery proc = (ptrCudaStreamQuery)cu->hook_manager->FindOriginal((PROC)CUDA_WRAPPER::_cudaStreamQuery);
        return (*proc)(stream);
    } else
    {
        EMULATOR * emulator = EMULATOR::Singleton();
        return emulator->_cudaStreamQuery(stream);
    }
}

