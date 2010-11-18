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
    cu->_cuda_runtime = new _CUDA_RUNTIME();
}



void CUDA_WRAPPER::DoInit()
{
    // This routine is static, so get the singleton for the cuda
    // wrapper.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CALL_STACK_INFO * csi = CALL_STACK_INFO::Singleton();
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
    HOOK_MANAGER * hm = HOOK_MANAGER::Singleton();
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
    this->_cuda_runtime->WrapModule(cuda_module_name);
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
    CALL_STACK_INFO * csi = CALL_STACK_INFO::Singleton();
    if (! d->is_host)
    {
        unsigned char * hostbuffer = (unsigned char *)malloc(d->size);
        if (! hostbuffer)
            return NOT_OK;

        cudaError_t e1;
        if (! cu->do_emulation)
        {
            _CUDA_RUNTIME::typePtrCudaMemcpy proc = (_CUDA_RUNTIME::typePtrCudaMemcpy)cu->hook_manager->FindOriginal((PROC)_CUDA_RUNTIME::Memcpy);
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

    CALL_STACK_INFO * csi = CALL_STACK_INFO::Singleton();
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
// With the cuda runtime library loaded, I can't seem to get this thing
// to stop chewing up memory.  There is a memory leak that occurs with
// thread spawning.  On a hunch, if emulation is desired, unload the
// cuda runtime and cuda driver api's.
//    if (cu->do_emulation)
//    {
//      HMODULE hModule_runtime = cu->_cuda_runtime->hModule;
//      for (int i = 0; i < 10000; ++i)
//      {
//          BOOL b = ::FreeLibrary(hModule_runtime);
//          printf ("%d ", b);
//      }
//      printf ("\n");
//      HMODULE hModule_driver = cu->_cuda->hModule;
//      for (int i = 0; i < 1000; ++i)
//      {
//          BOOL b = ::FreeLibrary(hModule_driver);
//          printf ("%d ", b);
//      }
//      printf ("\n");
//   }
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
    EMULATOR * emulator = EMULATOR::Singleton();
    emulator->SetTrace(level);
}

void CUDA_WRAPPER::StartDebugger()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    printf("Halting to invoke debugger...\n");
    _asm {
        int 3;
    }
}

void CUDA_WRAPPER::SetEmulationThreads(int i)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    EMULATOR * emulator = EMULATOR::Singleton();
    emulator->SetEmulationThreads(i);
}
