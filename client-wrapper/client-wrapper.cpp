// This is the main DLL file.

#include "stdafx.h"
#include <stdio.h>
#include "DoRPC_h.h"
#include "client-wrapper.h"


int xxx()
{
	RPC_STATUS status;
	RPC_WSTR szStringBinding = NULL;

	// Creates a string binding handle.
	// This function formats the passed values in a 
	// predefined format for use by RPC. Just like printf
	// Connection is not done here.
	status = RpcStringBindingCompose(
		NULL, // UUID to bind to.
		(RPC_WSTR)("ncacn_ip_tcp"), 
		// Use TCP/IP protocol.
		(RPC_WSTR)("localhost"), // TCP/IP network // the same machine as server
		(RPC_WSTR)("9191"), 
		// TCP/IP port to use.
		NULL,     // Protocol dependent network options to use.
		&szStringBinding);   // String binding output.

	if (status)
		return status;

	// Validates the format of the string binding handle and converts
	// it to a binding handle.
	// Connection is not done here either.
	status = RpcBindingFromStringBinding(
		szStringBinding, // The string binding to validate.
		&hDoRPCBinding); // Put the result in the implicit binding

	// handle defined in the IDL file.

	if(status)
	{
		return status;
	}

	RpcTryExcept
	{
		// Calls the RPC function. The hDoRPCBinding binding handle
		// is used implicitly.
		// Connection is done here.
		const unsigned char szMsg[] = "Client: I Can RPC Now!";
		Here();
	}
	RpcExcept(1)
	{
		printf("Runtime exception occured: %d\n",RpcExceptionCode());
	}
	RpcEndExcept

		// Free the memory allocated by a string.
		status = RpcStringFree(&szStringBinding);
	// String to be freed.

	if(status)
	{
		return status;
	}

	// Releases binding handle resources and disconnects from the server.
	status = RpcBindingFree(
		&hDoRPCBinding); // Frees the implicit binding handle defined in
	// the IDL file.

	if (status)
	{
		return status;
	}

	return 1234;
}


_declspec(dllexport) int __cdecl Here2()
{
	return xxx();
}


_declspec(dllexport) BOOL __cdecl StartCudaEmulationProcess(
  __in_opt     LPCTSTR lpApplicationName,
  __inout_opt  LPTSTR lpCommandLine,
  __in_opt     LPSECURITY_ATTRIBUTES lpProcessAttributes,
  __in_opt     LPSECURITY_ATTRIBUTES lpThreadAttributes,
  __in         BOOL bInheritHandles,
  __in         DWORD dwCreationFlags,
  __in_opt     LPVOID lpEnvironment,
  __in_opt     LPCTSTR lpCurrentDirectory,
  __in         LPSTARTUPINFO lpStartupInfo,
  __out        LPPROCESS_INFORMATION lpProcessInformation
);