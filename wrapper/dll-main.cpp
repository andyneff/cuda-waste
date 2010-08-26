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



void AddWrapper(char * cuda_module_name, HookManager * p)
{
	CudaMemoryDebug * cu = CudaMemoryDebug::Singleton();
	cu->DoInit(cuda_module_name, p);
}

BOOL LoadCuda()
{
	HookManager * p = new HookManager();
	HANDLE  hProcess = GetCurrentProcess();
	HMODULE hModuleArray[1024];
	DWORD   nModules;
	DWORD   cbNeeded;
	BOOL rv_epm = ::EnumProcessModules(hProcess, hModuleArray, sizeof(hModuleArray), &cbNeeded);
	if (! rv_epm)
	{
		CloseHandle(hProcess);
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
			return FALSE;
		}
		if (strstr(szModuleName, "cudart32_30_14") != 0 || strstr(szModuleName, "cudart32_31_9") != 0)
		{
			AddWrapper(szModuleName, p);
			found = true;
			break;
		}
	}
	return TRUE;
}

BOOL APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
	BOOL bResult = TRUE;
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		{
			return LoadCuda();
			break;
		}
	case DLL_PROCESS_DETACH:
		{
			break;
		}
	}
	return TRUE;
}
