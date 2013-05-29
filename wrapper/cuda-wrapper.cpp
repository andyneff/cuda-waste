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
#include "../waste/version.h"

static char temp_buffer[50000];

CUDA_WRAPPER * CUDA_WRAPPER::singleton;
typedef cudaError_t (*cfunc)(void **devPtr, size_t size);
cudaError_t (*ptrCudaMalloc)(void **devPtr, size_t size);

int mgetpid()
{
    return 0;
}



CUDA_WRAPPER * CUDA_WRAPPER::Singleton()
{
    if (CUDA_WRAPPER::singleton)
        return CUDA_WRAPPER::singleton;
    CUDA_WRAPPER::singleton = new CUDA_WRAPPER();
    return CUDA_WRAPPER::singleton;
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
    cu->do_debugger = false;
    cu->global_context = 0;
    cu->hook_manager = 0;
    cu->do_debug_halt = false;
    cu->level = 0;
    cu->device = 0; // nothing specific.
    cu->_cuda = new _CUDA();
    cu->_cuda_runtime = new _CUDA_RUNTIME();
    cu->stack_size = 10 * 1024 * 1024;
}



void CUDA_WRAPPER::DoInit()
{
    // This routine is static, so get the singleton for the cuda
    // wrapper.
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    CALL_STACK_INFO * csi = CALL_STACK_INFO::Singleton();
    csi->ClassifyAsPrefix("call-stack-info.cpp");
    csi->ClassifyAsPrefix("cuda-wrapper.cpp");
    csi->ClassifyAsPrefix("_cuda_runtime.cpp");
    csi->ClassifyAsPrefix("_cuda_runtime.h");
    csi->ClassifyAsPrefix("cuda_runtime.h");  // This is the include file for CUDA, but we want what calls it.
    csi->ClassifyAsPrefix("_cuda.cpp");
    csi->ClassifyAsPrefix("_cuda.h");
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
    //cu->StartListener();
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
    EMULATED_DEVICE * emulator = EMULATED_DEVICE::Singleton();
    emulator->RunDevice(device);

    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetDevice called, " << context << ".\n";
        (*cu->output_stream) << " Device now " << device << "\n\n";
    }
    return OK;
}


void CUDA_WRAPPER::SetTrace(int lev)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "SetTrace called, " << context << ".\n";
    }
    EMULATED_DEVICE * emulator = EMULATED_DEVICE::Singleton();
    cu->level = lev;
    emulator->SetTrace(lev);
}

void CUDA_WRAPPER::SetStackSize(int size)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    cu->stack_size = size;
}

void CUDA_WRAPPER::StartDebugger()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    //printf("Halting to invoke debugger...\n");
#if !defined(_WIN64)
    _asm {
        int 3;
    }
#endif
}

void CUDA_WRAPPER::SetStartDebugger()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    cu->do_debugger = true;
}

void CUDA_WRAPPER::SetEmulationThreads(int i)
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    EMULATED_DEVICE * emulator = EMULATED_DEVICE::Singleton();
    emulator->SetEmulationThreads(i);
}





///////////////////////////////////////////////////////////////
// DLL Injection
///////////////////////////////////////////////////////////////
byte code[4000];
int size = 0;

void AddBytes(byte * code, byte a)
{
    code[size++] = a;
}

void AddBytes(byte * code, byte a, byte b)
{
    code[size++] = a;
    code[size++] = b;
}

void AddBytes(byte * code, byte a, byte b, byte c)
{
    code[size++] = a;
    code[size++] = b;
    code[size++] = c;
}

void AddBytes(byte * code, byte a, byte b, byte c, byte d)
{
    code[size++] = a;
    code[size++] = b;
    code[size++] = c;
    code[size++] = d;
}

void AddBytes(byte * code, byte a, byte b, byte c, byte d, byte e)
{
    code[size++] = a;
    code[size++] = b;
    code[size++] = c;
    code[size++] = d;
    code[size++] = e;
}

void AddBytes(byte * code, byte a, byte b, byte c, byte d,
              byte e, byte f)
{
    code[size++] = a;
    code[size++] = b;
    code[size++] = c;
    code[size++] = d;
    code[size++] = e;
    code[size++] = f;
}

void AddBytes(byte * code, byte a, byte b, byte c, byte d,
              byte e, byte f, byte g)
{
    code[size++] = a;
    code[size++] = b;
    code[size++] = c;
    code[size++] = d;
    code[size++] = e;
    code[size++] = f;
    code[size++] = g;
}

void AddBytes(byte * code, byte a, byte b, byte c, byte d,
              byte e, byte f, byte g, byte h)
{
    code[size++] = a;
    code[size++] = b;
    code[size++] = c;
    code[size++] = d;
    code[size++] = e;
    code[size++] = f;
    code[size++] = g;
    code[size++] = h;
}

typedef CUDA_WRAPPER::return_type (CUDARTAPI *pSetTraceAllCalls)(bool b);
char * str_cuda_wrapper = "wrapper.dll";
char * str_trace_all_calls = "?SetTraceAllCalls@CUDA_WRAPPER@@SG?AW4return_type@1@_N@Z";
char * str_pading_byte = "?SetPaddingByte@CUDA_WRAPPER@@SG?AW4return_type@1@E@Z";
char * str_padding_size = "?SetPaddingSize@CUDA_WRAPPER@@SG?AW4return_type@1@I@Z";
char * str_quit_on_error = "?SetQuitOnError@CUDA_WRAPPER@@SG?AW4return_type@1@_N@Z";
char * str_do_not_call_cuda_after_sanity_check_fail = "?SetDoNotCallCudaAfterSanityCheckFail@CUDA_WRAPPER@@SG?AW4return_type@1@_N@Z";
char * str_device_pointer_to_first_byte_in_block = "?SetDevicePointerToFirstByteInBlock@CUDA_WRAPPER@@SG?AW4return_type@1@_N@Z";
char * str_set_device = "?RunDevice@CUDA_WRAPPER@@SG?AW4return_type@1@PAD@Z";
char * str_set_trace = "?SetTrace@CUDA_WRAPPER@@SGXH@Z";
char * str_wrap_cuda = "?WrapCuda@CUDA_WRAPPER@@SGHXZ";
char * str_set_emulator_mode = "?SetEmulationMode@CUDA_WRAPPER@@SGXH@Z";
char * str_start_debugger = "?StartDebugger@CUDA_WRAPPER@@SGXXZ";
char * str_set_num_threads = "?SetEmulationThreads@CUDA_WRAPPER@@SGXH@Z";


void ErrorExit(char * lpszFunction) 
{ 
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError(); 

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );

    // Display the error message and exit the process

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
        (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR)); 

    std::cerr << lpszFunction << " failed " << dw << " " << lpMsgBuf << "\n"; 
    wprintf(L"Formatted message: %s\n", lpMsgBuf);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
}

LPVOID ll = 0;
LPVOID gpa = 0;

HANDLE __stdcall CUDA_WRAPPER::StartProcess(char * command)
{
    ll = (LPVOID)GetProcAddress(GetModuleHandleA("kernel32.dll"), "LoadLibraryA");;
    gpa = (LPVOID)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetProcAddress");

    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

    int trace_all_calls = cu->trace_all_calls;
    int padding_size = cu->padding_size;
    int padding_byte = cu->padding_byte;
    int quit_on_error = cu->quit_on_error;
    int do_not_call_cuda_after_sanity_check_fail = cu->do_not_call_cuda_after_sanity_check_fail;
    int device_pointer_to_first_byte_in_block = cu->device_pointer_to_first_byte_in_block;
    bool set_emulator_mode = cu->do_emulation;
    bool do_debugger = cu->do_debugger;
    bool set_num_threads = false;
    int num_threads = 0;
    char * device = cu->device;
    int level = cu->level;

    ///////////////// Dll injection step 1 //////////////////////////////////////////////////
    // Create client process in suspended mode.
    STARTUPINFOA startupInfo;
    PROCESS_INFORMATION processInformation;
    memset(&startupInfo, 0, sizeof(STARTUPINFOA));
    memset(&processInformation, 0, sizeof(PROCESS_INFORMATION));
    startupInfo.cb = sizeof(STARTUPINFOA);
    BOOL rv_cp = CreateProcessA(0, command, 0, 0, 0, CREATE_DEFAULT_ERROR_MODE | CREATE_SUSPENDED, 0, 0, &startupInfo, &processInformation);
    if (! rv_cp)
    {
        std::cerr << "Program '" << command << "' could not be found.\n"
            << "Check the spelling and path for your program.\n";
        return 0;
    }
    DWORD process = processInformation.dwProcessId;
    HANDLE mainthread = processInformation.hThread;
    
    for (;;) {

        ///////////////// Dll injection step 2 //////////////////////////////////////////////////
        // Adjust privileges of host process.
        HANDLE hToken;
        LUID luid;
        TOKEN_PRIVILEGES tkprivs;
        BOOL bRet;
        ZeroMemory(&tkprivs, sizeof(TOKEN_PRIVILEGES));
        BOOL rv_op = OpenProcessToken(GetCurrentProcess(), (TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY), &hToken);
        if (rv_op == 0)
        {
            std::cerr << "Program '" << command << "' could not be found.\n"
                << "Check the spelling and path for your program.\n";
            break;
        }
        BOOL rv_lp = LookupPrivilegeValue(0, SE_DEBUG_NAME, &luid);
        if (rv_lp == 0)
        {
            ErrorExit("LookupPrivilegeValue");
            break;
        }
        tkprivs.PrivilegeCount = 1;
        tkprivs.Privileges[0].Luid = luid;
        tkprivs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        BOOL rv_atp = AdjustTokenPrivileges(hToken, 0, &tkprivs, sizeof(TOKEN_PRIVILEGES), 0, 0);
        if (rv_atp == 0)
        {
            ErrorExit("AdjustTokenPrivileges");
            break;
        }
        CloseHandle(hToken);

        ///////////////// Dll injection step 3 //////////////////////////////////////////////////
        // Get Eip register.  This is the address of the first byte of code
        // for the main program.  We'll jump to this in code we'll set up
        // later.
        CONTEXT context = { 0 };
        DWORD originalEip;

        context.ContextFlags = CONTEXT_FULL | CONTEXT_DEBUG_REGISTERS;
        //| CONTEXT_CONTROL | CONTEXT_DEBUG_REGISTERS;
        BOOL rv_gtc1 = GetThreadContext(mainthread, &context);
        if (rv_gtc1 == 0)
        {
            ErrorExit("GetThreadContext");
            break;
        }
#if defined(_WIN64)
        originalEip = context.Rip;
#elif defined(_WIN32)
        originalEip = context.Eip;
#endif
        ///////////////// Dll injection step 4 //////////////////////////////////////////////////
        // Allocate some chunks of memory in client and write strings to that.
        HANDLE hProcess;
        SIZE_T written;
        hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, process);

        LPVOID pszCMD;
        pszCMD = VirtualAllocEx(hProcess, NULL, strlen(str_cuda_wrapper) + 1, MEM_COMMIT, PAGE_READWRITE);
        BOOL rv_wpw1 = WriteProcessMemory(hProcess, pszCMD, (LPVOID) str_cuda_wrapper, strlen(str_cuda_wrapper) + 1, &written);
        if (rv_wpw1 == 0)
        {
            std::cerr << "WriteProcessMemory failed.\n";
            break;
        }

        LPVOID stack_page;
        int stack_page_size = 10 * 1024 * 1024;
        stack_page = VirtualAllocEx(hProcess, NULL, stack_page_size, MEM_COMMIT, PAGE_READWRITE);
        if (stack_page == 0)
        {
            std::cerr << "VirtualAllocEx failed.\n";
            break;
        }

        ///////////////// Dll injection step 4 //////////////////////////////////////////////////
        // Set up memory containing instructions to load the Cuda memory debug dll
        // then continue execution of the debugged program.  In addition,
        // set up instructions to set options for the dll.
    #define MAXCODESIZE 5000
        LPVOID codePtr = VirtualAllocEx(hProcess, NULL, MAXCODESIZE, MEM_COMMIT, PAGE_READWRITE);
        if (codePtr == 0)
        {
            std::cerr << "VirtualAllocEx failed.\n";
            break;
        }

#define JmpRelativeAddress( ptr, offset, addr ) *((LPVOID *) &ptr[offset]) = (LPVOID)( (DWORD_PTR)(addr) - (DWORD_PTR)(ptr) - (DWORD_PTR) (offset + 4) )
#define JmpAbsoluteAddress( ptr, offset, addr ) *((LPVOID *) &ptr[offset]) = (LPVOID)( (DWORD_PTR)(addr) )
#define JmpRelativeAddressBased( ptr, offset, addr, base, add_subtract ) *((LPVOID *) &ptr[offset]) = (LPVOID)( (DWORD_PTR)(addr) - (DWORD_PTR)(base) - (DWORD_PTR) (offset + 4 + add_subtract) )

#define xxx(offset, addr, base, add_subtract ) (LPVOID)( (DWORD_PTR)(addr) - (DWORD_PTR)(base) - (DWORD_PTR) (offset + 4 + add_subtract) )

        // NOTE: ALL THESE MACHINE CODE INSTRUCTIONS COME STRAIGHT OUT OF THE INTEL AND/OR AMD X86 X64 INSTRUCTION MANUALS.
        // http://www.intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html

        /* UNREAL! NO INLINE ASM IN VISUAL STUDIO FOR X64 TARGETS!
        __asm {
        }
        */

        // Let's pad the memory with NOP's since the Visual Studio disassembler freaks out very easily
        // with invalid opcodes in previous memory.
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);
        AddBytes(code, 0x90, 0x90, 0x90, 0x90);

        // Add code to load wrapper.dll library.
        {
#if defined(_WIN64)
            // int 3
            //AddBytes(code, 0xcc);

            BOOL rv_wpw_stack = WriteProcessMemory(hProcess, stack_page, (LPVOID) context.Rsp,
                sizeof(context.Rsp), &written);

            // Set new stack.
            // mov rsp, 0x000000000 (immediate, with value of wrapper.dll string address)
            AddBytes(code, 0x48, 0xbc);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, (DWORD)stack_page + stack_page_size - 1024); // -1024 for slack.

            /*
            0000000000090021 50                   push        rax  
            0000000000090022 51                   push        rcx  
            0000000000090023 52                   push        rdx  
            0000000000090024 53                   push        rbx  
            0000000000090025 54                   push        rsp  
            0000000000090026 55                   push        rbp  
            0000000000090027 56                   push        rsi  
            0000000000090028 57                   push        rdi  
            */
            AddBytes(code, 0x50);
            AddBytes(code, 0x51);
            AddBytes(code, 0x52);
            AddBytes(code, 0x53);
            AddBytes(code, 0x54);
            AddBytes(code, 0x55);
            AddBytes(code, 0x56);
            AddBytes(code, 0x57);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // mov rac, 0x000000000 (immediate, with value of wrapper.dll string address)
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Check top word, see if zero.
            LPVOID bb = xxx(size-4, ll_page, codePtr, 0);
#elif defined(_WIN32)
            // int 3
            AddBytes(code, 0xcc);

            AddBytes(code, 0x9c);   // pushfd
            AddBytes(code, 0x60);   // pushad

            // Inject cuda-memory-debug wrapper library target (load the library into this program).
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);   // patch with actual string address.
    
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0); // patch with actual function address.
#endif
        }

        // Add code to set trace level.
        {
#if defined(_WIN64)
            // Make some temporary space.
            //  sub  rsp,0xC0h  
            AddBytes(code, 0x48, 0x81, 0xEC, 0xC0, 0x00, 0x00, 0x00);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // Load address of "wrapper.dll" string into rac.
            // mov rac, 0x000000000
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Store the returned HMODULE on the stack.
            //  mov   qword ptr [rsp+8h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x08);

            char * str = "?SetTrace@CUDA_WRAPPER@@SAXH@Z";
            LPVOID pstr = VirtualAllocEx(hProcess, NULL, strlen(str) + 1, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, pstr, (LPVOID) str, strlen(str) + 1, &written);

            // Load address of SetTrace string into rad.
            // mov rad, 0x000000000
            AddBytes(code, 0x48, 0xba);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pstr);

            // Get the returned HMODULE on the stack.
            //  mov  rcx,qword ptr [rsp+8h]  
            AddBytes(code, 0x48, 0x8b, 0x4c, 0x24, 0x08);

            // Allocate memory for the address of GetProcAddress.
            LPVOID gpa_page;
            int gpa_page_size = sizeof(gpa_page);
            gpa_page = VirtualAllocEx(hProcess, NULL, gpa_page_size, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, gpa_page, (LPVOID) &gpa, sizeof(LPVOID), &written);

            // Call GetProcAddress.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, gpa_page, codePtr, 0);

            // Store the address of SetTrace on the stack.
            //  mov   qword ptr [rsp+10h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x10);

            // Load level for SetTrace()
            //  mov         cl,level  
            AddBytes(code, 0xB1, level);
            
            // Call SetTrace
            // call        qword ptr [rsp+10h] 
            AddBytes(code, 0xFF, 0x94, 0x24, 0x10, 0x00, 0x00, 0x00); 

            // Remove temporary space.
            //  add rsp,0C0h
            AddBytes(code, 0x48, 0x81, 0xC4, 0xC0, 0x00, 0x00, 0x00);

#elif defined(_WIN32)
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_trace_all_calls) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_set_trace, strlen(str_set_trace) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "?SetTrace@CUDA_WRAPPER@@SAXH@Z"
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x68, 0, 0, 0, 0); // push level
            JmpAbsoluteAddress(code, size-4, level);
            AddBytes(code, 0xff, 0xd0); // call eax
#endif
        }

        // Add code to invoke StartDebugger.
        if (do_debugger)
        {
#if defined(_WIN64)
            // Make some temporary space.
            //  sub  rsp,0xC0h  
            AddBytes(code, 0x48, 0x81, 0xEC, 0xC0, 0x00, 0x00, 0x00);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // Load address of "wrapper.dll" string into rac.
            // mov rac, 0x000000000
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Store the returned HMODULE on the stack.
            //  mov   qword ptr [rsp+8h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x08);

            char * str = "?StartDebugger@CUDA_WRAPPER@@SAXXZ";
            LPVOID pstr = VirtualAllocEx(hProcess, NULL, strlen(str) + 1, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, pstr, (LPVOID) str, strlen(str) + 1, &written);

            // Load address of StartDebugger string into rad.
            // mov rad, 0x000000000
            AddBytes(code, 0x48, 0xba);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pstr);

            // Get the returned HMODULE on the stack.
            //  mov  rcx,qword ptr [rsp+8h]  
            AddBytes(code, 0x48, 0x8b, 0x4c, 0x24, 0x08);

            // Allocate memory for the address of GetProcAddress.
            LPVOID gpa_page;
            int gpa_page_size = sizeof(gpa_page);
            gpa_page = VirtualAllocEx(hProcess, NULL, gpa_page_size, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, gpa_page, (LPVOID) &gpa, sizeof(LPVOID), &written);

            // Call GetProcAddress.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, gpa_page, codePtr, 0);

            // Store the address of StartDebugger on the stack.
            //  mov   qword ptr [rsp+10h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x10);

            // Call StartDebugger
            // call        qword ptr [rsp+10h] 
            AddBytes(code, 0xFF, 0x94, 0x24, 0x10, 0x00, 0x00, 0x00); 

            // Remove temporary space.
            //  add rsp,0C0h
            AddBytes(code, 0x48, 0x81, 0xC4, 0xC0, 0x00, 0x00, 0x00);

#elif defined(_WIN32)
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_padding_size) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_start_debugger, strlen(str_start_debugger) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_start_debugger
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0xff, 0xd0); // call eax
#endif
        }

        // Add code to call WrapCuda
        {
#if defined(_WIN64)
            // Make some temporary space.
            //  sub  rsp,0xC0h  
            AddBytes(code, 0x48, 0x81, 0xEC, 0xC0, 0x00, 0x00, 0x00);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // Load address of "wrapper.dll" string into rac.
            // mov rac, 0x000000000
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Store the returned HMODULE on the stack.
            //  mov   qword ptr [rsp+8h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x08);

            char * str = "?WrapCuda@CUDA_WRAPPER@@SAHXZ";
            LPVOID pstr = VirtualAllocEx(hProcess, NULL, strlen(str) + 1, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, pstr, (LPVOID) str, strlen(str) + 1, &written);

            // Load address of WrapCuda string into rad.
            // mov rad, 0x000000000
            AddBytes(code, 0x48, 0xba);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pstr);

            // Get the returned HMODULE on the stack.
            //  mov  rcx,qword ptr [rsp+8h]  
            AddBytes(code, 0x48, 0x8b, 0x4c, 0x24, 0x08);

            // Allocate memory for the address of GetProcAddress.
            LPVOID gpa_page;
            int gpa_page_size = sizeof(gpa_page);
            gpa_page = VirtualAllocEx(hProcess, NULL, gpa_page_size, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, gpa_page, (LPVOID) &gpa, sizeof(LPVOID), &written);

            // Call GetProcAddress.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, gpa_page, codePtr, 0);

            // Store the address of WrapCuda on the stack.
            //  mov   qword ptr [rsp+10h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x10);

            // Call WrapCuda
            // call        qword ptr [rsp+10h] 
            AddBytes(code, 0xFF, 0x94, 0x24, 0x10, 0x00, 0x00, 0x00); 

            // Remove temporary space.
            //  add rsp,0C0h
            AddBytes(code, 0x48, 0x81, 0xC4, 0xC0, 0x00, 0x00, 0x00);

#elif defined(_WIN32)
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_padding_size) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_wrap_cuda, strlen(str_wrap_cuda) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_wrap_cuda
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0xff, 0xd0); // call eax
#endif
        }

        if (trace_all_calls)
        {
#if defined(_WIN64)
            // Make some temporary space.
            //  sub  rsp,0xC0h  
            AddBytes(code, 0x48, 0x81, 0xEC, 0xC0, 0x00, 0x00, 0x00);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // Load address of "wrapper.dll" string into rac.
            // mov rac, 0x000000000
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Store the returned HMODULE on the stack.
            //  mov   qword ptr [rsp+8h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x08);

            char * str = "?SetTraceAllCalls@CUDA_WRAPPER@@SA?AW4return_type@1@_N@Z";
            LPVOID pstr = VirtualAllocEx(hProcess, NULL, strlen(str) + 1, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, pstr, (LPVOID) str, strlen(str) + 1, &written);

            // Load address of "?SetTraceAllCalls@CUDA_WRAPPER@@SA?AW4return_type@1@_N@Z" string into rad.
            // mov rad, 0x000000000
            AddBytes(code, 0x48, 0xba);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pstr);

            // Get the returned HMODULE on the stack.
            //  mov  rcx,qword ptr [rsp+8h]  
            AddBytes(code, 0x48, 0x8b, 0x4c, 0x24, 0x08);

            // Allocate memory for the address of GetProcAddress.
            LPVOID gpa_page;
            int gpa_page_size = sizeof(gpa_page);
            gpa_page = VirtualAllocEx(hProcess, NULL, gpa_page_size, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, gpa_page, (LPVOID) &gpa, sizeof(LPVOID), &written);

            // Call GetProcAddress.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, gpa_page, codePtr, 0);

            // Store the address of SetTraceAllCalls on the stack.
            //  mov   qword ptr [rsp+10h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x10);

            // Load 1 for SetTraceAllCalls()
            //  mov         cl,1  
            AddBytes(code, 0xB1, trace_all_calls);
            
            // Call SetTraceAllCalls
            // call        qword ptr [rsp+10h] 
            AddBytes(code, 0xFF, 0x94, 0x24, 0x10, 0x00, 0x00, 0x00); 

            // Remove temporary space.
            //  add rsp,0C0h
            AddBytes(code, 0x48, 0x81, 0xC4, 0xC0, 0x00, 0x00, 0x00);

#elif defined(_WIN32)
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_trace_all_calls) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_trace_all_calls, strlen(str_trace_all_calls) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "?SetTraceAllCalls@CUDA_WRAPPER@@SG?AW4return_type@1@_N@Z"
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x6a, trace_all_calls); // push 1 or 0
            AddBytes(code, 0xff, 0xd0); // call eax
#endif
        }

        if (quit_on_error)
        {
#if defined(_WIN64)
            // Make some temporary space.
            //  sub  rsp,0xC0h  
            AddBytes(code, 0x48, 0x81, 0xEC, 0xC0, 0x00, 0x00, 0x00);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // Load address of "wrapper.dll" string into rac.
            // mov rac, 0x000000000
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Store the returned HMODULE on the stack.
            //  mov   qword ptr [rsp+8h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x08);

            char * str = "?SetQuitOnError@CUDA_WRAPPER@@SA?AW4return_type@1@_N@Z";
            LPVOID pstr = VirtualAllocEx(hProcess, NULL, strlen(str) + 1, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, pstr, (LPVOID) str, strlen(str) + 1, &written);

            // Load address of SetQuitOnError string into rad.
            // mov rad, 0x000000000
            AddBytes(code, 0x48, 0xba);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pstr);

            // Get the returned HMODULE on the stack.
            //  mov  rcx,qword ptr [rsp+8h]  
            AddBytes(code, 0x48, 0x8b, 0x4c, 0x24, 0x08);

            // Allocate memory for the address of GetProcAddress.
            LPVOID gpa_page;
            int gpa_page_size = sizeof(gpa_page);
            gpa_page = VirtualAllocEx(hProcess, NULL, gpa_page_size, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, gpa_page, (LPVOID) &gpa, sizeof(LPVOID), &written);

            // Call GetProcAddress.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, gpa_page, codePtr, 0);

            // Store the address of SetQuitOnError on the stack.
            //  mov   qword ptr [rsp+10h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x10);

            // Load quit_on_error for SetQuitOnError()
            //  mov         cl,quit_on_error  
            AddBytes(code, 0xB1, quit_on_error);
            
            // Call SetTraceAllCalls
            // call        qword ptr [rsp+10h] 
            AddBytes(code, 0xFF, 0x94, 0x24, 0x10, 0x00, 0x00, 0x00); 

            // Remove temporary space.
            //  add rsp,0C0h
            AddBytes(code, 0x48, 0x81, 0xC4, 0xC0, 0x00, 0x00, 0x00);

#elif defined(_WIN32)
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_quit_on_error) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_quit_on_error, strlen(str_quit_on_error) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_quit_on_error
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x6a, quit_on_error); // push 1 or 0
            AddBytes(code, 0xff, 0xd0); // call eax
#endif
        }
        if (0)
        {
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_do_not_call_cuda_after_sanity_check_fail) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_do_not_call_cuda_after_sanity_check_fail, strlen(str_do_not_call_cuda_after_sanity_check_fail) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_do_not_call_cuda_after_sanity_check_fail
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x6a, do_not_call_cuda_after_sanity_check_fail); // push 1 or 0
            AddBytes(code, 0xff, 0xd0); // call eax
        }
        if (0)
        {
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_device_pointer_to_first_byte_in_block) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_device_pointer_to_first_byte_in_block, strlen(str_device_pointer_to_first_byte_in_block) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_device_pointer_to_first_byte_in_block
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x6a, device_pointer_to_first_byte_in_block); // push 1 or 0
            AddBytes(code, 0xff, 0xd0); // call eax
        }
        if (0)
        {
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_pading_byte) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_pading_byte, strlen(str_pading_byte) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_pading_byte
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x68, 0, 0, 0, 0); // push padding_byte
            JmpAbsoluteAddress(code, size-4, padding_byte);
            AddBytes(code, 0xff, 0xd0); // call eax
        }
        if (device)
        {
            LPVOID pszDevice = VirtualAllocEx(hProcess, NULL, strlen(device) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw1 = WriteProcessMemory(hProcess, pszDevice, (LPVOID) device, strlen(device) + 1, &written);
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_set_device) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_set_device, strlen(str_set_device) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "RunDevice"
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x68, 0, 0, 0, 0); // push device
            JmpAbsoluteAddress(code, size-4, pszDevice);
            AddBytes(code, 0xff, 0xd0); // call eax
        }
        if (0)
        {
            LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_set_num_threads) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_set_num_threads, strlen(str_set_num_threads) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push SetEmulationThreads
            JmpAbsoluteAddress(code, size-4, pszSetFunc);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x68, 0, 0, 0, 0); // push level
            JmpAbsoluteAddress(code, size-4, num_threads);
            AddBytes(code, 0xff, 0xd0); // call eax
        }
        // Force emulation or no emulation.
        if (set_emulator_mode)
        {
#if defined(_WIN64)
            // Make some temporary space.
            //  sub  rsp,0xC0h  
            AddBytes(code, 0x48, 0x81, 0xEC, 0xC0, 0x00, 0x00, 0x00);

            // Allocate memory for the address of LoadLibraryA.
            LPVOID ll_page;
            int ll_page_size = sizeof(ll_page);
            ll_page = VirtualAllocEx(hProcess, NULL, ll_page_size, MEM_COMMIT, PAGE_READWRITE);
            if (ll_page == 0)
            {
                std::cerr << "VirtualAllocEx failed.\n";
                break;
            }
            BOOL rv_wpw2 = WriteProcessMemory(hProcess, ll_page, (LPVOID) &ll, sizeof(LPVOID), &written);
            if (rv_wpw2 == 0)
            {
                std::cerr << "WriteProcessMemory failed.\n";
                break;
            }

            // Load address of "wrapper.dll" string into rac.
            // mov rac, 0x000000000
            AddBytes(code, 0x48, 0xb9);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pszCMD);

            // Call loadlibrarya.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, ll_page, codePtr, 0);

            // Store the returned HMODULE on the stack.
            //  mov   qword ptr [rsp+8h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x08);

            char * str = "?SetEmulationMode@CUDA_WRAPPER@@SAXH@Z";
            LPVOID pstr = VirtualAllocEx(hProcess, NULL, strlen(str) + 1, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, pstr, (LPVOID) str, strlen(str) + 1, &written);

            // Load address of SetEmulationMode string into rad.
            // mov rad, 0x000000000
            AddBytes(code, 0x48, 0xba);
            AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
            JmpAbsoluteAddress(code, size-8, pstr);

            // Get the returned HMODULE on the stack.
            //  mov  rcx,qword ptr [rsp+8h]  
            AddBytes(code, 0x48, 0x8b, 0x4c, 0x24, 0x08);

            // Allocate memory for the address of GetProcAddress.
            LPVOID gpa_page;
            int gpa_page_size = sizeof(gpa_page);
            gpa_page = VirtualAllocEx(hProcess, NULL, gpa_page_size, MEM_COMMIT, PAGE_READWRITE);
            WriteProcessMemory(hProcess, gpa_page, (LPVOID) &gpa, sizeof(LPVOID), &written);

            // Call GetProcAddress.
            AddBytes(code, 0xff, 0x15);
            AddBytes(code, 0, 0, 0, 0);
            JmpRelativeAddressBased(code, size-4, gpa_page, codePtr, 0);

            // Store the address of SetEmulationMode on the stack.
            //  mov   qword ptr [rsp+10h],rax
            AddBytes(code, 0x48, 0x89, 0x44, 0x24, 0x10);

            // Load set_emulator_mode for SetEmulationMode()
            //  mov         cl,set_emulator_mode  
            AddBytes(code, 0xB1, set_emulator_mode);
            
            // Call SetEmulationMode
            // call        qword ptr [rsp+10h] 
            AddBytes(code, 0xFF, 0x94, 0x24, 0x10, 0x00, 0x00, 0x00); 

            // Remove temporary space.
            //  add rsp,0C0h
            AddBytes(code, 0x48, 0x81, 0xC4, 0xC0, 0x00, 0x00, 0x00);

#elif defined(_WIN32)
            LPVOID psz = VirtualAllocEx(hProcess, NULL, strlen(str_set_emulator_mode) + 1, MEM_COMMIT, PAGE_READWRITE);
            BOOL rv_wpw = WriteProcessMemory(hProcess, psz, (LPVOID) str_set_emulator_mode, strlen(str_set_emulator_mode) + 1, &written);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
            JmpAbsoluteAddress(code, size-4, pszCMD);
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
            JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
            AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "SetEmulatorMode"
            JmpAbsoluteAddress(code, size-4, psz);
            AddBytes(code, 0x50); // push eax
            AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
            JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
            AddBytes(code, 0x68, 0, 0, 0, 0); // push true/false (as 1 or 0 int).
            JmpAbsoluteAddress(code, size-1, set_emulator_mode);
            AddBytes(code, 0xff, 0xd0); // call eax
#endif
        }

#if defined(_WIN64)

        AddBytes(code, 0x5f, 0x5e, 0x5d, 0x5c, 0x5b, 0x5a, 0x59, 0x58);

        // Restore old stack.
        // mov rsp, 0x000000000 (immediate, with value of wrapper.dll string address)
        AddBytes(code, 0x48, 0xbc);
        AddBytes(code, 0, 0, 0, 0, 0, 0, 0, 0);
        JmpAbsoluteAddress(code, size-8, (DWORD)context.Rsp);

#elif defined(_WIN32)
        // Restore registers and jump to original program entry.  Address
        // patched below.
        AddBytes(code, 0x61); // popad
        AddBytes(code, 0x9D); // popfd
#endif
    
        AddBytes(code, 0xE9, 0x00, 0x00, 0x00, 0x00); // jmp originalEip
        // original EIP address
        JmpRelativeAddressBased(code, size-4, originalEip, codePtr, 0);
        // Write assembly code

        SIZE_T count;
        if (!WriteProcessMemory(hProcess, codePtr, (LPVOID) code,
            size, &count)) return FALSE;

        DWORD oldProtect;
        if(!VirtualProtectEx(hProcess, codePtr, size, PAGE_EXECUTE, &oldProtect)) {
            printf("Error: %d\n", GetLastError());
        }

        ///////////////// Dll injection step 5 //////////////////////////////////////////////////
        // Start up thread at the code we created.
#if defined(_WIN64)
        context.Rip = (DWORD) codePtr + 32;
#elif defined(_WIN32)
        context.Eip = (DWORD) codePtr + 32;
#endif
        SetThreadContext(mainthread, &context);
        ResumeThread(mainthread);

        return hProcess;

    }

    return 0;
}
