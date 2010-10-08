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
#include "targetver.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "call-stack-info.h"
#include "hook-mgr.h"
#include "memdbg.h"


MEMDBG * MEMDBG::singleton;

MEMDBG::MEMDBG()
{
	this->hook_manager = HookManager::Singleton();
	this->is_wrapped = false;
	this->current_alloc = 0;
	this->allocation_number = 1;
	this->max_allocs = 100000;
	this->max_context = 20;
	this->allocs = 0;
	this->record = true;
}

MEMDBG * MEMDBG::Singleton()
{
	if (singleton == 0)
		singleton = new MEMDBG();
	return singleton;
}

void MEMDBG::SetMaxAllocs(int size)
{
	// Cannot change the size of the context recording midway.
	if (this->is_wrapped)
		return;
	this->max_allocs = size;
}

void MEMDBG::SetMaxContext(int size)
{
	// Cannot change the size of the context recording midway.
	if (this->is_wrapped)
		return;
	this->max_context = size;
}

void MEMDBG::WrapModules()
{
	this->is_wrapped = true;
	bool save_record = this->record;
	this->record = false;

	// At this point, create space for data.
	this->allocs = (data*)malloc(sizeof(data) * this->max_allocs);
	memset(this->allocs, 0, sizeof(data) * this->max_allocs);
	for (int i = 0; i < this->max_allocs; ++i)
	{
		this->allocs[i].context = (void**)malloc(sizeof(void*) * this->max_context);
		for (int j = 0; j < this->max_context; ++j)
			this->allocs[i].context[j] = 0;
	}

	// Wrap...
	PROC proc1 = hook_manager->HookImport("ntdll.dll", "RtlAllocateHeap", (PROC)MEMDBG::_RtlAllocateHeap, true);
	PROC proc2 = hook_manager->HookImport("ntdll.dll", "RtlFreeHeap", (PROC)MEMDBG::_RtlFreeHeap, true);
	PROC proc3 = hook_manager->HookImport("kernel32.dll", "HeapAlloc", (PROC)MEMDBG::_HeapAlloc, true);
	PROC proc4 = hook_manager->HookImport("kernel32.dll", "HeapFree", (PROC)MEMDBG::_HeapFree, true);

	this->record = save_record;
}

void MEMDBG::UnwrapModules()
{
	bool save_record = this->record;
	this->record = false;

	// Unwrap...
	hook_manager->UnHookImport("ntdll.dll", "RtlAllocateHeap");
	hook_manager->UnHookImport("ntdll.dll", "RtlFreeHeap");
	hook_manager->UnHookImport("kernel32.dll", "HeapAlloc");
	hook_manager->UnHookImport("kernel32.dll", "HeapFree");

	this->is_wrapped = false;
	this->record = save_record;
}

PVOID WINAPI MEMDBG::_RtlAllocateHeap(
  __in      PVOID HeapHandle,
  __in_opt  ULONG Flags,
  __in      SIZE_T Size)
{
	typedef PVOID (__stdcall * ptr_RtlAllocateHeap)(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      SIZE_T Size);
	MEMDBG * memdbg = MEMDBG::Singleton();
    ptr_RtlAllocateHeap proc = (ptr_RtlAllocateHeap)memdbg->hook_manager->FindOriginal((PROC)MEMDBG::_RtlAllocateHeap);
	static bool reentrant;
	if (!reentrant)
	{
		reentrant = true;
		// Call the damn thing.
        PVOID result = (*proc)(HeapHandle, Flags, Size);
	    if (memdbg->record)
		{
			data * d = & memdbg->allocs[memdbg->current_alloc++];
			if (memdbg->current_alloc >= memdbg->max_allocs)
				memdbg->current_alloc = 0;
			d->is_alloc = true;
			d->ptr = result;
			d->size = Size;
			d->allocation_number = memdbg->allocation_number;
			d->type = data::is_RtlAllocateHeap;
			CallStackInfo * csi = CallStackInfo::Singleton();
			void ** addresses = csi->AddressContext(memdbg->max_context);
			for (int i = 0; i < memdbg->max_context; ++i)
				d->context[i] = addresses[i];
		}
		memdbg->allocation_number++;
		reentrant = false;
		return result;
	}
	else
	{
		// Call the damn thing.
		memdbg->allocation_number++;
        PVOID result = (*proc)(HeapHandle, Flags, Size);
		return result;
	}
}

BOOLEAN WINAPI MEMDBG::_RtlFreeHeap(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      PVOID HeapBase
	)
{
	typedef BOOLEAN (__stdcall * ptr_RtlFreeHeap)(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      PVOID HeapBase);
	MEMDBG * memdbg = MEMDBG::Singleton();
    ptr_RtlFreeHeap proc = (ptr_RtlFreeHeap)memdbg->hook_manager->FindOriginal((PROC)MEMDBG::_RtlFreeHeap);
	static bool reentrant;
	if (!reentrant)
	{
		reentrant = true;
		// Call the damn thing.
        BOOLEAN result = (*proc)(HeapHandle, Flags, HeapBase);
	    if (memdbg->record)
		{
			data * d = & memdbg->allocs[memdbg->current_alloc++];
			if (memdbg->current_alloc >= memdbg->max_allocs)
				memdbg->current_alloc = 0;
			d->is_alloc = false;
			d->ptr = HeapBase;
			d->size = 0;
			d->allocation_number = memdbg->allocation_number;
			d->type = data::is_RtlFreeHeap;
			CallStackInfo * csi = CallStackInfo::Singleton();
			void ** addresses = csi->AddressContext(memdbg->max_context);
			for (int i = 0; i < memdbg->max_context; ++i)
				d->context[i] = addresses[i];
		}
		reentrant = false;
		memdbg->allocation_number++;
		return result;
	}
	else
	{
		// Call the damn thing.
		memdbg->allocation_number++;
        BOOLEAN result = (*proc)(HeapHandle, Flags, HeapBase);
		return result;
	}
}


PVOID WINAPI MEMDBG::_HeapAlloc(
  __in      PVOID HeapHandle,
  __in_opt  ULONG Flags,
  __in      SIZE_T Size)
{
	typedef PVOID (WINAPI * ptr_HeapAlloc)(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      SIZE_T Size);
	MEMDBG * memdbg = MEMDBG::Singleton();
    ptr_HeapAlloc proc = (ptr_HeapAlloc)memdbg->hook_manager->FindOriginal((PROC)MEMDBG::_HeapAlloc);
	static bool reentrant;
	if (!reentrant)
	{
		reentrant = true;
		// Call the damn thing.
        PVOID result = (*proc)(HeapHandle, Flags, Size);
	    if (memdbg->record)
		{
			data * d = & memdbg->allocs[memdbg->current_alloc++];
			if (memdbg->current_alloc >= memdbg->max_allocs)
				memdbg->current_alloc = 0;
			d->is_alloc = true;
			d->ptr = result;
			d->size = Size;
			d->allocation_number = memdbg->allocation_number;
			d->type = data::is_HeapAlloc;
//			CallStackInfo * csi = CallStackInfo::Singleton();
//			void ** addresses = csi->AddressContext(memdbg->max_context);
//			for (int i = 0; i < memdbg->max_context; ++i)
//				d->context[i] = addresses[i];
		}
		memdbg->allocation_number++;
		reentrant = false;
		return result;
	}
	else
	{
		// Call the damn thing.
		memdbg->allocation_number++;
        PVOID result = (*proc)(HeapHandle, Flags, Size);
		return result;
	}
}

BOOL WINAPI MEMDBG::_HeapFree(
	  __in  HANDLE hHeap,
	  __in  DWORD dwFlags,
	  __in  LPVOID lpMem
	)
{
	typedef BOOLEAN (__stdcall * ptr_HeapFree)(
		__in      PVOID HeapHandle,
		__in_opt  ULONG Flags,
		__in      PVOID HeapBase);
	MEMDBG * memdbg = MEMDBG::Singleton();
    ptr_HeapFree proc = (ptr_HeapFree)memdbg->hook_manager->FindOriginal((PROC)MEMDBG::_RtlFreeHeap);
	static bool reentrant;
	if (!reentrant)
	{
		reentrant = true;
		// Call the damn thing.
        BOOL result = (*proc)(hHeap, dwFlags, lpMem);
	    if (memdbg->record)
		{
			data * d = & memdbg->allocs[memdbg->current_alloc++];
			if (memdbg->current_alloc >= memdbg->max_allocs)
				memdbg->current_alloc = 0;
			d->is_alloc = false;
			d->ptr = lpMem;
			d->size = 0;
			d->allocation_number = memdbg->allocation_number;
			d->type = data::is_HeapFree;
//			CallStackInfo * csi = CallStackInfo::Singleton();
//			void ** addresses = csi->AddressContext(memdbg->max_context);
//			for (int i = 0; i < memdbg->max_context; ++i)
//				d->context[i] = addresses[i];
		}
		reentrant = false;
		memdbg->allocation_number++;
		return result;
	}
	else
	{
		// Call the damn thing.
		memdbg->allocation_number++;
        BOOL result = (*proc)(hHeap, dwFlags, lpMem);
		return result;
	}
}


void MEMDBG::OnRecord()
{
	this->record = true;
}

void MEMDBG::OffRecord()
{
	this->record = false;
}

void MEMDBG::Output()
{
	bool save_record = this->record;
	this->record = false;

	unsigned int min_allocation_number = 0xffffffff;
	unsigned int index_min_allocation_number = 0;
	for (int i = 0; i < this->max_allocs; ++i)
	{
		// Find min alloc/dealloc event number.
		if (this->allocs[i].allocation_number < min_allocation_number && this->allocs[i].allocation_number != 0)
		{
			min_allocation_number = this->allocs[i].allocation_number;
			index_min_allocation_number = i;
		}
	}

	for (int i = index_min_allocation_number; i < this->max_allocs; ++i)
	{
		data * d = &this->allocs[i];
		if (d->allocation_number != 0)
		{
			std::cout << "ptr = " << std::hex << d->ptr << " " << std::dec << d->size
				<< " " << d->allocation_number;
			switch (d->type)
			{
			case data::is_RtlAllocateHeap:
				std::cout << " RtlAllocateHeap   ";
				break;
			case data::is_RtlFreeHeap:
				std::cout << " RtlFreeHeap       ";
				break;
			case data::is_HeapAlloc:
				std::cout << " HeapAlloc         ";
				break;
			case data::is_HeapFree:
				std::cout << " HeapFree          ";
				break;
			}
			std::cout << "\n";
		}
	}
	for (int i = 0; i < index_min_allocation_number; ++i)
	{
		data * d = &this->allocs[i];
		if (d->allocation_number != 0)
		{
			std::cout << "ptr = " << std::hex << d->ptr << " " << std::dec << d->size
				<< " " << d->allocation_number;
			switch (d->type)
			{
			case data::is_RtlAllocateHeap:
				std::cout << " RtlAllocateHeap   ";
				break;
			case data::is_RtlFreeHeap:
				std::cout << " RtlFreeHeap       ";
				break;
			case data::is_HeapAlloc:
				std::cout << " HeapAlloc         ";
				break;
			case data::is_HeapFree:
				std::cout << " HeapFree          ";
				break;
			}
			std::cout << "\n";
		}
	}
	this->record = save_record;
}

