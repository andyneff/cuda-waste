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
#include "stdafx.h"
#include <Psapi.h> // EnumProcessModules, GetModuleFileNameExA
#pragma comment(lib,"Psapi")
#include "cuda-wrapper.h"
#include "hook-mgr.h"

BOOL CUDA_WRAPPER::WrapCuda()
{
    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();
    char * context = cu->Context();
    if (cu->trace_all_calls)
    {
        (*cu->output_stream) << "WrapCuda called " << context << ".\n";
    }
    cu->DoInit();
    HANDLE  hProcess = GetCurrentProcess();
    HMODULE hModuleArray[1024];
    DWORD   nModules;
    DWORD   cbNeeded;
    BOOL rv_epm = ::EnumProcessModules(hProcess, hModuleArray, sizeof(hModuleArray), &cbNeeded);
    if (! rv_epm)
    {
        CloseHandle(hProcess);
    std::cerr << "WrapCuda failed\n";
    std::cerr.flush();
        return FALSE;
    }
    nModules = cbNeeded / sizeof(hModuleArray[0]);
    bool found = false;
    for (DWORD j = 0; j < nModules; j++)
    {
        HMODULE hModule = hModuleArray[j];
        char    szModuleName[MAX_PATH];
        BOOL rv_gmfn = ::GetModuleFileNameExA(hProcess, hModule, szModuleName, sizeof(szModuleName));
        if (! rv_gmfn)
        {
            CloseHandle(hProcess);
    std::cerr << "WrapCuda failed\n";
    std::cerr.flush();
            return FALSE;
        }
        // Regardless of which module, add wrapper API.
        cu->WrapModule(szModuleName);
    }
    return TRUE;
}

// NOTE: "It is not safe to call LoadLibrary from DllMain."  Don't wrap
// the API here because it will want to load the CUDA API's in order to
// do the hooking.
BOOL APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    BOOL bResult = TRUE;
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        {
            break;
        }
    case DLL_PROCESS_DETACH:
        {
            break;
        }
    }
    return TRUE;
}
