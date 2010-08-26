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
#include "call-stack-info.h"
#include <windows.h> // required by stupid dbghelp.h -- it doesn't know its own dependencies...
#include <dbghelp.h>
#include <stdio.h>

extern char * file_name_tail(char * file_name);

CallStackInfo::CallStackInfo()
{
#define BUFFERSIZE 50000
    char lpszPath[BUFFERSIZE];
    DWORD symOptions = SymGetOptions();

    symOptions |= SYMOPT_LOAD_LINES; 
    symOptions &= ~SYMOPT_UNDNAME;
    SymSetOptions( symOptions );

    if (! GetEnvironmentVariableA( "PATH", lpszPath, BUFFERSIZE))
    {
        strcpy(lpszPath, ".");
    }
    SymInitialize(GetCurrentProcess(), lpszPath, TRUE);
}

CallStackInfo * CallStackInfo::Singleton()
{
    static CallStackInfo * singleton = 0;
    if (singleton)
        return singleton;
    singleton = new CallStackInfo();
    return singleton;
}

// Find the module name from the ip
bool CallStackInfo::GetModuleNameFromAddress(void * address, char * lpszModule)
{
    BOOL              ret = FALSE;
    IMAGEHLP_MODULE   moduleInfo;

    ::ZeroMemory( &moduleInfo, sizeof(moduleInfo) );
    moduleInfo.SizeOfStruct = sizeof(moduleInfo);

    if (SymGetModuleInfo( GetCurrentProcess(), (DWORD)address, &moduleInfo ) )
    {
        strncpy(lpszModule, moduleInfo.ModuleName, sizeof(moduleInfo.ModuleName) );
        ret = true;
    }
    else
        // Not found :(
        strcpy( lpszModule, "?");
    
    return ret;
}

// Get function prototype and parameter info from ip address and stack address
bool CallStackInfo::GetFunctionInfoFromAddresses( void * fnAddress, void * stackAddress, char * lpszSymbol )
{
    return FALSE;
#ifdef XXX
    BOOL              ret = FALSE;
    DWORD             dwDisp = 0;
    DWORD             dwSymSize = 10000;
   TCHAR             lpszUnDSymbol[BUFFERSIZE]=_T("?");
    CHAR              lpszNonUnicodeUnDSymbol[BUFFERSIZE]="?";
    LPTSTR            lpszParamSep = NULL;
    LPCTSTR           lpszParsed = lpszUnDSymbol;
    PIMAGEHLP_SYMBOL  pSym = (PIMAGEHLP_SYMBOL)GlobalAlloc( GMEM_FIXED, dwSymSize );

    ::ZeroMemory( pSym, dwSymSize );
    pSym->SizeOfStruct = dwSymSize;
    pSym->MaxNameLength = dwSymSize - sizeof(IMAGEHLP_SYMBOL);

   // Set the default to unknown
    _tcscpy( lpszSymbol, _T("?") );

    // Get symbol info for IP
    if ( SymGetSymFromAddr( GetCurrentProcess(), (ULONG)fnAddress, &dwDisp, pSym ) )
    {
       // Make the symbol readable for humans
        UnDecorateSymbolName( pSym->Name, lpszNonUnicodeUnDSymbol, BUFFERSIZE, 
            UNDNAME_COMPLETE | 
            UNDNAME_NO_THISTYPE |
            UNDNAME_NO_SPECIAL_SYMS |
            UNDNAME_NO_MEMBER_TYPE |
            UNDNAME_NO_MS_KEYWORDS |
            UNDNAME_NO_ACCESS_SPECIFIERS );

      // Symbol information is ANSI string
        PCSTR2LPTSTR( lpszNonUnicodeUnDSymbol, lpszUnDSymbol );

      // I am just smarter than the symbol file :)
        if ( _tcscmp(lpszUnDSymbol, _T("_WinMain@16")) == 0 )
            _tcscpy(lpszUnDSymbol, _T("WinMain(HINSTANCE,HINSTANCE,LPCTSTR,int)"));
        else
        if ( _tcscmp(lpszUnDSymbol, _T("_main")) == 0 )
            _tcscpy(lpszUnDSymbol, _T("main(int,TCHAR * *)"));
        else
        if ( _tcscmp(lpszUnDSymbol, _T("_mainCRTStartup")) == 0 )
            _tcscpy(lpszUnDSymbol, _T("mainCRTStartup()"));
        else
        if ( _tcscmp(lpszUnDSymbol, _T("_wmain")) == 0 )
            _tcscpy(lpszUnDSymbol, _T("wmain(int,TCHAR * *,TCHAR * *)"));
        else
        if ( _tcscmp(lpszUnDSymbol, _T("_wmainCRTStartup")) == 0 )
            _tcscpy(lpszUnDSymbol, _T("wmainCRTStartup()"));

        lpszSymbol[0] = _T('\0');

      // Let's go through the stack, and modify the function prototype, and insert the actual
      // parameter values from the stack
        if ( _tcsstr( lpszUnDSymbol, _T("(void)") ) == NULL && _tcsstr( lpszUnDSymbol, _T("()") ) == NULL)
        {
            ULONG index = 0;
            for( ; ; index++ )
            {
                lpszParamSep = _tcschr( lpszParsed, _T(',') );
                if ( lpszParamSep == NULL )
                    break;

                *lpszParamSep = _T('\0');

                _tcscat( lpszSymbol, lpszParsed );
                _stprintf( lpszSymbol + _tcslen(lpszSymbol), _T("=0x%08X,"), *((ULONG*)(stackAddress) + 2 + index) );

                lpszParsed = lpszParamSep + 1;
            }

            lpszParamSep = _tcschr( lpszParsed, _T(')') );
            if ( lpszParamSep != NULL )
            {
                *lpszParamSep = _T('\0');

                _tcscat( lpszSymbol, lpszParsed );
                _stprintf( lpszSymbol + _tcslen(lpszSymbol), _T("=0x%08X)"), *((ULONG*)(stackAddress) + 2 + index) );

                lpszParsed = lpszParamSep + 1;
            }
        }

        _tcscat( lpszSymbol, lpszParsed );
   
        ret = TRUE;
    }

    GlobalFree( pSym );

    return ret;
#endif
}

// Get source file name and line number from IP address
// The output format is: "sourcefile(linenumber)" or
//                       "modulename!address" or
//                       "address"
bool CallStackInfo::GetSourceInfoFromAddress( void * address, char * lpszSourceInfo, char * full_file_name )
{
    bool           ret = FALSE;
    IMAGEHLP_LINE  lineInfo;
    DWORD          dwDisp;
    char          lpszFileName[BUFFERSIZE];
    char          lpModuleInfo[BUFFERSIZE];

    strcpy(lpszFileName, "");
    strcpy(lpModuleInfo, "");
    strcpy(lpszSourceInfo, "?(?)");

    ::ZeroMemory( &lineInfo, sizeof( lineInfo ) );
    lineInfo.SizeOfStruct = sizeof( lineInfo );

    if ( SymGetLineFromAddr( GetCurrentProcess(), (DWORD)address, &dwDisp, &lineInfo ) )
    {
        char * fn = file_name_tail(lineInfo.FileName);
        strncpy(full_file_name, lineInfo.FileName, BUFFERSIZE);
        sprintf(lpszSourceInfo, "file %s, line %d", fn, lineInfo.LineNumber);
        ret = true;
    }
    else
    {
        strncpy(full_file_name, "", BUFFERSIZE);
        GetModuleNameFromAddress( address, lpModuleInfo );
        if ( lpModuleInfo[0] == '?' || lpModuleInfo[0] == '\0')
            sprintf(lpszSourceInfo, "Address 0x%08X", address);
        else
            sprintf(lpszSourceInfo, "Module %s, Address 0x%08X", lpModuleInfo, address);
        ret = false;
    }
    return ret;
}

char * CallStackInfo::Context(int lines)
{
    char buffer[BUFFERSIZE];
    strcpy(buffer, "");

    typedef USHORT (WINAPI *CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
    CaptureStackBackTraceType func = (CaptureStackBackTraceType)(GetProcAddress(LoadLibraryA("kernel32.dll"), "RtlCaptureStackBackTrace"));
    const int kMaxCallers = 62; 
    void* callers[kMaxCallers];
    int count = (func)(0, kMaxCallers, callers, NULL);
    bool seen_prefix = false;
    int times = 0;
    for(int i = 0; i < count; i++)
    {
        char context_string[BUFFERSIZE];
        char full_file_name[BUFFERSIZE];
        GetSourceInfoFromAddress(callers[i], context_string, full_file_name);
        if (! seen_prefix)
        {
            bool present = false;
            for (std::list<char*>::iterator it = ignore_files.begin(); it != ignore_files.end(); it++)
            {
                // Remove prefix if the full_file_name matches a file on the list of files to ignore.
                if (strstr(full_file_name, *it))
                {
                    // present.
                    present = true;
                    break;
                }
            }
            if (! present)
            {
                seen_prefix = true;
            }
        }
        else
            seen_prefix = true;
        if (seen_prefix)
        {
            if (times)
                strcat(buffer, "\n");
            strcat(buffer, context_string);
            times++;
            if (times == lines)
                break;
        }
    }

	if (strcmp("", buffer) == 0)
	{
		strcat(buffer, "(no call stack available)");
	}

    return strdup(buffer);
}

void CallStackInfo::ClassifyAsPrefix(char * file)
{
    // Add file name to list of files that you want to remove from call stack context string.
    ignore_files.push_back(file);
}

