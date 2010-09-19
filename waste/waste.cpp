#include "stdafx.h"
#include <windows.h>
#include <iostream>
#include "../wrapper/cuda-wrapper.h"

void Help()
{
    std::cerr << "\
USAGE: waste [OPTION] ... FILE\n\
Emulation of CUDA program FILE.\n\
\n\
Options:\n\
  -t=NUMBER                         Trace CUDA memory API calls, emulator, etc, at a certain level of noisiness.\n\
  -s=NUMBER, --padding-size=NUMBER  Set size of padding for buffer allocations.\n\
  -b=CHAR, --padding-byte=CHAR      Set byte of padding.\n\
  -q, --quit-on-error               Quit program on error detection.\n\
  -k, --skip-on-error               Skip over CUDA call when the usage is invalid.\n\
  -n, --non-standard-ptr            Allow computed pointers.\n\
";
    exit(1);
}

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

int main(int argc, char * argv[])
{
    // Perform debugging of CUDA memory calls.
    argc--;
    argv++;

    if (argc == 0)
    {
        Help();
    }

    bool set_trace_all_calls = false;
    int trace_all_calls;
    bool set_padding_size = false;
    int padding_size;
    bool set_padding_byte = false;
    int padding_byte;
    bool set_quit_on_error = false;
    int quit_on_error;
    bool set_do_not_call_cuda_after_sanity_check_fail = false;
    int do_not_call_cuda_after_sanity_check_fail;
    bool set_device_pointer_to_first_byte_in_block = false;
    int device_pointer_to_first_byte_in_block;
    bool set_device = false;
    char * device;
    int level = 0;

    // Create a structure containing options for debug.
    while (argc > 0)
    {
        if (**argv == '-')
        {
            if (strncmp("-t=", *argv, 3) == 0)
            {
                set_trace_all_calls = true;
                trace_all_calls = true;
                level = atoi(3+*argv);
            }
            else if (strncmp("-s=", *argv, 3) == 0)
            {
                set_padding_size = true;
                padding_size = atoi(3+*argv);
            }
            else if (strncmp("--padding-size=", *argv, 15) == 0)
            {
                set_padding_size = true;
                padding_size = atoi(15+*argv);
            }
            else if (strncmp("-b=", *argv, 3) == 0)
            {
                set_padding_byte = true;
                padding_byte = atoi(3+*argv);
            }
            else if (strncmp("--padding-byte=", *argv, 15) == 0)
            {
                set_padding_byte = true;
                padding_byte = atoi(15+*argv);
            }
            else if (strcmp("-q", *argv) == 0 || strcmp("--quit-on-error", *argv) == 0)
            {
                set_quit_on_error = true;
                quit_on_error = true;
            }
            else if (strcmp("-k", *argv) == 0 || strcmp("--skip-on-error", *argv) == 0)
            {
                set_do_not_call_cuda_after_sanity_check_fail = true;
                do_not_call_cuda_after_sanity_check_fail = true;
            }
            else if (strcmp("-n", *argv) == 0 || strcmp("--non-standard-ptr", *argv) == 0)
            {
                set_device_pointer_to_first_byte_in_block = true;
                device_pointer_to_first_byte_in_block = true;
            }
            else if (strcmp("-d", *argv) == 0 || strcmp("--device", *argv) == 0)
            {
                set_device = true;
                argc--; argv++;
                device = *argv;
            }
            else
                Help();
            argc--; argv++;
        }
        else
            break;
    }

    ///////////////// Dll injection step 1 //////////////////////////////////////////////////
    // Create client process in suspended mode.
    STARTUPINFOA startupInfo;
    PROCESS_INFORMATION processInformation;
    memset(&startupInfo, 0, sizeof(STARTUPINFOA));
    memset(&processInformation, 0, sizeof(PROCESS_INFORMATION));
    startupInfo.cb = sizeof(STARTUPINFOA);
    BOOL rv_cp = CreateProcessA(*argv, 0, 0, 0, 0, CREATE_DEFAULT_ERROR_MODE | CREATE_SUSPENDED, 0, 0, &startupInfo, &processInformation);
    if (! rv_cp)
    {
        std::cerr << "Program '" << *argv << "' could not be found.\n"
            << "Check the spelling and path for your program.\n";
        exit(1);
    }
    DWORD process = processInformation.dwProcessId;
    HANDLE mainthread = processInformation.hThread;
    
    ///////////////// Dll injection step 2 //////////////////////////////////////////////////
    // Adjust privileges of host process.
    HANDLE hToken;
    LUID luid;
    TOKEN_PRIVILEGES tkprivs;
    BOOL bRet;
    ZeroMemory(&tkprivs, sizeof(TOKEN_PRIVILEGES));
    BOOL rv_op = OpenProcessToken(GetCurrentProcess(), (TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY), &hToken);
    BOOL rv_lp = LookupPrivilegeValue(0, SE_DEBUG_NAME, &luid);
    tkprivs.PrivilegeCount = 1;
    tkprivs.Privileges[0].Luid = luid;
    tkprivs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    BOOL rv_atp = AdjustTokenPrivileges(hToken, 0, &tkprivs, sizeof(TOKEN_PRIVILEGES), 0, 0);
    CloseHandle(hToken);

    ///////////////// Dll injection step 3 //////////////////////////////////////////////////
    // Get Eip register.  This is the address of the first byte of code
    // for the main program.  We'll jump to this in code we'll set up
    // later.
    CONTEXT context;
    DWORD originalEip;
    context.ContextFlags = CONTEXT_FULL | CONTEXT_DEBUG_REGISTERS;
    GetThreadContext(mainthread, &context);
    originalEip = context.Eip;
    
    ///////////////// Dll injection step 4 //////////////////////////////////////////////////
    // Allocate some chunks of memory in client and write strings to that.
    HANDLE hProcess;
    LPVOID pszCMD;
    SIZE_T written;
    hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, process);
    pszCMD = VirtualAllocEx(hProcess, NULL, strlen(str_cuda_wrapper) + 1, MEM_COMMIT, PAGE_READWRITE);
    BOOL rv_wpw1 = WriteProcessMemory(hProcess, pszCMD, (LPVOID) str_cuda_wrapper, strlen(str_cuda_wrapper) + 1, &written);

    ///////////////// Dll injection step 4 //////////////////////////////////////////////////
    // Set up memory containing instructions to load the Cuda memory debug dll
    // then continue execution of the debugged program.  In addition,
    // set up instructions to set options for the dll.
#define MAXCODESIZE 5000
    LPVOID codePtr = VirtualAllocEx(hProcess, NULL, MAXCODESIZE, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if (!codePtr)
        return 1;

#define JmpRelativeAddress( ptr, offset, addr ) *((LPVOID *) &ptr[offset]) = (LPVOID)( (DWORD_PTR)(addr) - (DWORD_PTR)(ptr) - (DWORD_PTR) (offset + 4) )
#define JmpAbsoluteAddress( ptr, offset, addr ) *((LPVOID *) &ptr[offset]) = (LPVOID)( (DWORD_PTR)(addr) )
#define JmpRelativeAddressBased( ptr, offset, addr, base, add_subtract ) *((LPVOID *) &ptr[offset]) = (LPVOID)( (DWORD_PTR)(addr) - (DWORD_PTR)(base) - (DWORD_PTR) (offset + 4 + add_subtract) )

    // Save ALL registers.
    AddBytes(code, 0x9c);   // pushfd
    AddBytes(code, 0x60);   // pushad

    // Inject cuda-memory-debug wrapper library target (load the library into this program).
    AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
    JmpAbsoluteAddress(code, size-4, pszCMD);   // patch with actual string address.
    
    AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
    JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0); // patch with actual function address.

    // No way to set an option directly until the dll is loaded.
    // Instead, load pointer to function for modifying an option, then
    // call it.
    if (set_trace_all_calls)
    {
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
        AddBytes(code, 0x6a, 0x01); // push 1
        AddBytes(code, 0xff, 0xd0); // call eax
    }
    if (level > 0)
    {
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
    }

    if (set_quit_on_error)
    {
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
        AddBytes(code, 0x6a, 0x01); // push 1
        AddBytes(code, 0xff, 0xd0); // call eax
    }
    if (set_do_not_call_cuda_after_sanity_check_fail)
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
        AddBytes(code, 0x6a, 0x01); // push 1
        AddBytes(code, 0xff, 0xd0); // call eax
    }
    if (set_device_pointer_to_first_byte_in_block)
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
        AddBytes(code, 0x6a, 0x01); // push 1
        AddBytes(code, 0xff, 0xd0); // call eax
    }

    if (set_padding_size)
    {
        LPVOID pszSetFunc = VirtualAllocEx(hProcess, NULL, strlen(str_padding_size) + 1, MEM_COMMIT, PAGE_READWRITE);
        BOOL rv_wpw2 = WriteProcessMemory(hProcess, pszSetFunc, (LPVOID) str_padding_size, strlen(str_padding_size) + 1, &written);
        AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push "wrapper.dll"
        JmpAbsoluteAddress(code, size-4, pszCMD);
        AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call LoadLibraryA
        JmpRelativeAddressBased(code, size-4, &LoadLibraryA, codePtr, 0);
        AddBytes(code, 0x68, 0x00, 0x00, 0x00, 0x00); // push str_padding_size
        JmpAbsoluteAddress(code, size-4, pszSetFunc);
        AddBytes(code, 0x50); // push eax
        AddBytes(code, 0xE8, 0x00, 0x00, 0x00, 0x00); // call GetProcAddress
        JmpRelativeAddressBased(code, size-4, &GetProcAddress, codePtr, 0);
        AddBytes(code, 0x68, 0, 0, 0, 0); // push padding_size
        JmpAbsoluteAddress(code, size-4, padding_size);
        AddBytes(code, 0xff, 0xd0); // call eax
    }
    if (set_padding_byte)
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
    if (set_device)
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

    // Add hooks.
    {
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
    }

    // Restore registers and jump to original program entry.  Address
    // patched below.
    AddBytes(code, 0x61); // popad
    AddBytes(code, 0x9D); // popfd
    AddBytes(code, 0xE9, 0x00, 0x00, 0x00, 0x00); // jmp originalEip
    // original EIP address
    JmpRelativeAddressBased(code, size-4, originalEip, codePtr, 0);
    // Write assembly code
    SIZE_T count;
    if (!WriteProcessMemory(hProcess, codePtr, (LPVOID) code,
        size, &count)) return FALSE;

    ///////////////// Dll injection step 5 //////////////////////////////////////////////////
    // Start up thread at the code we created.
    context.Eip = (DWORD) codePtr;
    SetThreadContext(mainthread, &context);
    ResumeThread(mainthread);

    ///////////////// Dll injection step 6 //////////////////////////////////////////////////
    // Wait for the program we spawned to finish.
    WaitForSingleObject( hProcess, INFINITE );

    return 0;
}

