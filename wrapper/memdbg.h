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

class HookManager;

#ifdef DLL_EXPORTS
#define DLL_API __declspec(dllexport)
#else
#define DLL_API __declspec(dllimport)
#endif

class DLL_API MEMDBG
{
public:
#define MAX_CONTEXT 30
    struct data
    {
        void * ptr;
        bool is_alloc;
        int size;
		unsigned int allocation_number;
        void ** context;
		enum { is_RtlAllocateHeap = 1,
			is_RtlFreeHeap,
			is_HeapAlloc,
			is_HeapFree } type;
    };
    HookManager * hook_manager;
	size_t max_allocs;
	unsigned int current_alloc;
	unsigned int allocation_number;
	size_t max_context;
	data * allocs;
	static MEMDBG * singleton;
	MEMDBG();
	static PVOID WINAPI _RtlAllocateHeap(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      SIZE_T Size);
	static BOOLEAN WINAPI _RtlFreeHeap(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      PVOID HeapBase
	);
	static LPVOID WINAPI _HeapAlloc(
	  __in  HANDLE hHeap,
	  __in  DWORD dwFlags,
	  __in  SIZE_T dwBytes
	);
	static BOOL WINAPI _HeapFree(
	  __in  HANDLE hHeap,
	  __in  DWORD dwFlags,
	  __in  LPVOID lpMem
	);
	bool is_wrapped;
	bool record;
public:
	static MEMDBG * Singleton();
	void WrapModules();
	void UnwrapModules();
	void SetMaxAllocs(int size);
	void SetMaxContext(int size);
	void OnRecord();
	void OffRecord();
	void Output();
};

