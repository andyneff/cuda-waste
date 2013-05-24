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
*/
#pragma once

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.
#include "_cuda.h"
#include "_cuda_runtime.h"

class HOOK_MANAGER;

#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

/// Wrapper for CUDA's dynamic link library.
#ifndef __dv
#define __dv(x)
#endif


class DLL_API CUDA_WRAPPER
{
public:
    struct data
    {
        void * ptr;
        bool is_host;
        int size;
        char * context;
    };
    friend std::vector<data>;
private:
    CUDA_WRAPPER();
    static CUDA_WRAPPER * singleton;
public:
    HOOK_MANAGER * hook_manager;

        size_t padding_size;
    unsigned char padding_byte;
    bool device_pointer_to_first_byte_in_block;
    std::ostream * output_stream;
    bool trace_all_calls;
    bool quit_on_error;
    bool do_not_call_cuda_after_sanity_check_fail;
    bool do_emulation;
	bool do_hw_devices;
    bool do_debug_halt;
    bool init;
    static void ExitHandler();
    std::vector<data> alloc_list;
    static char * Context(int lines = 1);
    char * global_context;
    char * device; // device to run.
	int level;
	bool do_debugger;
    _CUDA * _cuda;
	_CUDA_RUNTIME * _cuda_runtime;

public:
    static CUDA_WRAPPER * Singleton();
    void DoInit();
    bool WrapModule(char * modname);
    static BOOL CUDARTAPI WrapCuda();
    static void MakeContext(char * file_name, int line);

    enum return_type {
        NOT_OK = 0,
        OK = 1
    };
    static return_type CheckOverwrite();
    static return_type __stdcall SetPaddingSize(size_t s);
    static return_type __stdcall SetPaddingByte(unsigned char b);
    static return_type __stdcall SetDevicePointerToFirstByteInBlock(bool b);
    static return_type __stdcall SetOutputStream(std::ostream * fp);
    static return_type __stdcall SetTraceAllCalls(bool b);
    static return_type __stdcall SetQuitOnError(bool b);
    static return_type __stdcall SetDoNotCallCudaAfterSanityCheckFail(bool b);
	static void __stdcall SetEmulationThreads(int i);
    static void __stdcall SetEmulationMode(int yes_no);
	static void __stdcall SetStartDebugger();
    static return_type CopyOptions(CUDA_WRAPPER * ptr);
    static return_type __stdcall RunDevice(char * device);
    static void __stdcall SetTrace(int level);
	static void __stdcall StartDebugger();
    static return_type CheckSinglePtrOverwrite(const data * d);
    static bool IsBadPointer(const void * ptr);
    static int FindAllocatedBlock(const void * pointer);
	static unsigned int __stdcall WinThreadListener(void * wrapper);
	void StartListener();
	static HANDLE __stdcall StartProcess(char * command);

};

