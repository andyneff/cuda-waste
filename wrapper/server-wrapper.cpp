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
#include <stdlib.h>
#include <process.h>    /* _beginthread, _endthread */
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
#include "../client-wrapper/DoRPC_h.h"


void CUDA_WRAPPER::StartListener()
{
	HANDLE hThread = (HANDLE) _beginthreadex(0, 0, CUDA_WRAPPER::WinThreadListener, (void*)this, CREATE_SUSPENDED, 0);
	if (hThread)
	{
		ResumeThread(hThread);
	}
}

unsigned int __stdcall CUDA_WRAPPER::WinThreadListener(void * wrapper)
{
	CUDA_WRAPPER * cu = (CUDA_WRAPPER *) wrapper;

	for (;;)
	{
		RPC_STATUS status;

		// Uses the protocol combined with the endpoint for receiving
		// remote procedure calls.
		status = RpcServerUseProtseqEp(
			(unsigned char*)("ncacn_ip_tcp"),// Use TCP/IP protocol
			RPC_C_PROTSEQ_MAX_REQS_DEFAULT,    // Backlog q length for TCP/IP.
			(unsigned char*)("9191"),    // TCP/IP port to use.
			NULL);       // No security.

		if(status)
		{
			_endthreadex(status);
			return status;
		}

		// Registers the DoRPC interface.
		status = RpcServerRegisterIf(
			DoRPC_v1_0_s_ifspec, // Interface to register.
			NULL,   // Use the MIDL generated entry-point vector.
			NULL);   // Use the MIDL generated entry-point vector.

		if(status)
		{
			_endthreadex(status);
			return status;
		}

		// Start to listen for remote procedure calls for all registered interfaces.
		// This call will not return until RpcMgmtStopServerListening is called.
		status = RpcServerListen(
			1,       // Recommended minimum number of threads.
			RPC_C_LISTEN_MAX_CALLS_DEFAULT,  // Recommended maximum number of threads.
			FALSE);              // Start listening now.

		if(status)
		{
			_endthreadex(status);
			return status;
		}
	}

	return 0;
}

// Now we implement our server function.
int Here()
{
   printf("Here!!!!\n");
   return 1;
}


void __RPC_FAR * __RPC_API midl_user_allocate(size_t cBytes) 
{ 
    return(malloc(cBytes)); 
}

void __RPC_API midl_user_free(void __RPC_FAR * p) 
{ 
    free(p); 
}
