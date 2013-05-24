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
//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Based upon the code in API Hooking Revealed (http://www.codeproject.com/KB/system/hooksys.aspx)
// by Ivo Ivanov (ivopi@hotmail.com).
//                                                                         
//////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <imagehlp.h>
#include "hook-mgr.h"
#include "lock-mgr.h"
#include <Psapi.h>

#pragma comment(lib, "imagehlp.lib")


HMODULE ModuleFromAddress(PVOID pv) 
{
    MEMORY_BASIC_INFORMATION mbi;
    return ((::VirtualQuery(pv, &mbi, sizeof(mbi)) != 0) 
            ? (HMODULE) mbi.AllocationBase : NULL);
}

char * __cdecl mstrrchr (
        const char * string,
        int ch
        )
{
        char *start = (char *)string;

        while (*string++)                       /* find end of string */
                ;
                                                /* search towards front */
        while (--string != start && *string != (char)ch)
                ;

        if (*string == (char)ch)                /* char found ? */
                return( (char *)string );

        return(NULL);
}

bool ExtractModuleFileName(char* pszFullFileName)
{
    bool  bResult = false;
    if (TRUE != ::IsBadReadPtr(pszFullFileName, MAX_PATH))
    {
        char  *pdest;
        int   ch = '\\';
        pdest = mstrrchr(pszFullFileName, ch);
        if( pdest != NULL )
            strcpy(pszFullFileName, &pdest[1]);
        bResult = true;
    }
    return bResult;
}


HookedFunctions* HOOK_MANAGER::sm_pHookedFunctions = NULL;
CRIT_SECTION        HOOK_MANAGER::sm_CritSec;
HOOK_MANAGER * HOOK_MANAGER::singleton;

HOOK_MANAGER::HOOK_MANAGER()
{
    m_hmodThisInstance   = ModuleFromAddress(HOOK_MANAGER::MyGetProcAddress);
    m_bSystemFuncsHooked = false;
    sm_pHookedFunctions  = new HookedFunctions(this);
}

HOOK_MANAGER * HOOK_MANAGER::Singleton()
{
	if (HOOK_MANAGER::singleton == 0)
		singleton = new HOOK_MANAGER();
	return singleton;
}

HOOK_MANAGER::~HOOK_MANAGER()
{
    UnHookAllFuncs();
    delete sm_pHookedFunctions;
}

bool HOOK_MANAGER::HookSystemFuncs()
{
    if (! m_bSystemFuncsHooked)
    {
        HookImport("Kernel32.dll", "LoadLibraryW", (PROC) HOOK_MANAGER::MyLoadLibraryW, true);
        HookImport("Kernel32.dll", "LoadLibraryExA", (PROC) HOOK_MANAGER::MyLoadLibraryExA, true);
        HookImport("Kernel32.dll", "LoadLibraryExW", (PROC) HOOK_MANAGER::MyLoadLibraryExW, true);
        HookImport("Kernel32.dll", "LoadLibraryA", (PROC) HOOK_MANAGER::MyLoadLibraryA, true);
        HookImport("Kernel32.dll", "GetProcAddress", (PROC) HOOK_MANAGER::MyGetProcAddress, true);
        m_bSystemFuncsHooked = true;
    }
    return m_bSystemFuncsHooked;
}

void HOOK_MANAGER::UnHookAllFuncs()
{
    if (m_bSystemFuncsHooked)
    {
        HookedFunction* pHook;
        HookedFunctions::const_iterator itr;
        for (itr = sm_pHookedFunctions->begin(); itr != sm_pHookedFunctions->end(); ++itr)
        {
            pHook = itr->second;
            pHook->UnHookImport();
            delete pHook;
        }
        sm_pHookedFunctions->clear();
        m_bSystemFuncsHooked = false;
    }
}

BOOL HOOK_MANAGER::AreThereHookedFunctions()
{
    return (sm_pHookedFunctions->size() > 0);
}

PROC HOOK_MANAGER::HookImport(PCSTR pszCalleeModName, PCSTR pszFuncName, PROC pfnHook, bool flag)
{
    LOCK_MANAGER<CRIT_SECTION>  lockMgr(sm_CritSec, TRUE);

    BOOL                  bResult = FALSE;
    PROC                  pfnOrig = NULL;
    try
    {
        HMODULE hModule = ::GetModuleHandleA(pszCalleeModName);
        HANDLE  hProcess = GetCurrentProcess();
        char    szModuleName[MAX_PATH];
        BOOL rv_gmfn = ::GetModuleFileNameExA(hProcess, hModule, szModuleName, sizeof(szModuleName));
        if (! rv_gmfn)
        {
            CloseHandle(hProcess);
            return pfnOrig;
        }
        if (!sm_pHookedFunctions->GetHookedFunction(
                hModule, 
                pszFuncName
                ))
        {
            pfnOrig = GetProcAddressWindows(
                ::GetModuleHandleA(szModuleName),
                pszFuncName
                );
            if (NULL == pfnOrig)
            {
                HMODULE hmod = ::LoadLibraryA(szModuleName);
                if (NULL != hmod)
                    pfnOrig = GetProcAddressWindows(
                        ::GetModuleHandleA(szModuleName),
                        pszFuncName
                        );
            }
            if (NULL != pfnOrig)
                bResult = AddHook(
                    szModuleName, 
                    pszFuncName, 
                    pfnOrig,
                    pfnHook
                    );
            else
            {
                // Unknown function...
                if (flag)
                    std::cerr << "Unknown function " << pszFuncName << " of module " << pszCalleeModName << "\n Check name.\n";
                return 0;
            }
        }
    }
    catch(...) {}
    return pfnOrig;
}

HookedFunction* HOOK_MANAGER::FindHook(PCSTR pszCalleeModName, PCSTR pszFuncName)
{
    LOCK_MANAGER<CRIT_SECTION>  lockMgr(sm_CritSec, TRUE);
    HookedFunction * result = 0;
    try
    {
        result = sm_pHookedFunctions->GetHookedFunction(
                pszCalleeModName, 
                pszFuncName
                );
    }
    catch(...) {}
    return result;
}

HookedFunction * HOOK_MANAGER::FindHook(void * iat)
{
    LOCK_MANAGER<CRIT_SECTION>  lockMgr(sm_CritSec, TRUE);
    HookedFunction * result = 0;
    try
    {
        result = sm_pHookedFunctions->GetHookedFunction(iat);
    }
    catch(...) {}
    return result;
}



PROC HOOK_MANAGER::FindOriginal(PROC wrapper_function)
{
    HookedFunction* pHook;
    HookedFunctions::const_iterator itr;
    for (itr = sm_pHookedFunctions->begin(); itr != sm_pHookedFunctions->end(); ++itr)
    {
        pHook = itr->second;
        if (pHook->Get_pfnHook() == wrapper_function)
            return pHook->Get_pfnOrig();
    }
    return 0;
}

HMODULE HOOK_MANAGER::GetModule(char * mod_name)
{
	return ::LoadLibraryA(mod_name);
}

BOOL HOOK_MANAGER::UnHookImport(PCSTR pszCalleeModName, PCSTR pszFuncName)
{
    LOCK_MANAGER<CRIT_SECTION>  lockMgr(sm_CritSec, TRUE);

    BOOL bResult = TRUE;
    try
    {
        bResult = RemoveHook(pszCalleeModName, pszFuncName);
    }
    catch (...)
    {
    }
    return bResult;
}

BOOL HOOK_MANAGER::AddHook(PCSTR pszCalleeModName, PCSTR pszFuncName, PROC pfnOrig, PROC pfnHook)
{
    BOOL             bResult = FALSE;
    HookedFunction* pHook   = NULL;

    if (!sm_pHookedFunctions->GetHookedFunction(
            pszCalleeModName, 
            pszFuncName
            ))
    {
        pHook = new HookedFunction(
            sm_pHookedFunctions,
            pszCalleeModName, 
            pszFuncName, 
            pfnOrig,
            pfnHook
            );
        bResult = sm_pHookedFunctions->AddHook(pHook);
        BOOL bResult2 = pHook->HookImport();
		if (! bResult2)
			sm_pHookedFunctions->RemoveHook(pHook);
		bResult |= bResult2;
    }
    return bResult;
}

BOOL HOOK_MANAGER::RemoveHook(PCSTR pszCalleeModName, PCSTR pszFuncName)
{
    BOOL             bResult = FALSE;
    HookedFunction *pHook   = NULL;

    pHook = sm_pHookedFunctions->GetHookedFunction(
        pszCalleeModName, 
        pszFuncName
        );
    if ( NULL != pHook )
    {
        bResult = pHook->UnHookImport();
        if ( bResult )
        {
            bResult = sm_pHookedFunctions->RemoveHook( pHook );
            if ( bResult )
                delete pHook;
        }
    }
    return bResult;
}

void WINAPI HOOK_MANAGER::HackModuleOnLoad(HMODULE hmod, DWORD dwFlags)
{
    if ((hmod != NULL) && ((dwFlags & LOAD_LIBRARY_AS_DATAFILE) == 0)) 
    {
        LOCK_MANAGER<CRIT_SECTION>  lockMgr(sm_CritSec, TRUE);
        
        HookedFunction* pHook;
        HookedFunctions::const_iterator itr;
        for (itr = sm_pHookedFunctions->begin(); 
            itr != sm_pHookedFunctions->end(); 
            ++itr)
        {
            pHook = itr->second;
            pHook->ReplaceInOneModule(
                pHook->Get_CalleeModName(), 
                pHook->Get_pfnOrig(), 
                pHook->Get_pfnHook(), 
                hmod
                );
        }
    }
}

HMODULE WINAPI HOOK_MANAGER::MyLoadLibraryA(PCSTR pszModuleName)
{
    HMODULE hmod = ::LoadLibraryA(pszModuleName);
    HackModuleOnLoad(hmod, 0);

    return hmod;
}

HMODULE WINAPI HOOK_MANAGER::MyLoadLibraryW(PCWSTR pszModuleName)
{
    HMODULE hmod = ::LoadLibraryW(pszModuleName);
    HackModuleOnLoad(hmod, 0);
    return hmod;
}

HMODULE WINAPI HOOK_MANAGER::MyLoadLibraryExA(PCSTR pszModuleName, HANDLE hFile, DWORD dwFlags)
{
    HMODULE hmod = ::LoadLibraryExA(pszModuleName, hFile, dwFlags);
    HackModuleOnLoad(hmod, 0);
    return hmod;
}

HMODULE WINAPI HOOK_MANAGER::MyLoadLibraryExW(PCWSTR pszModuleName, HANDLE hFile, DWORD dwFlags)
{
    HMODULE hmod = ::LoadLibraryExW(pszModuleName, hFile, dwFlags);
    HackModuleOnLoad(hmod, 0);
    return hmod;
}

FARPROC WINAPI HOOK_MANAGER::MyGetProcAddress(HMODULE hmod, PCSTR pszProcName)
{
    LOCK_MANAGER<CRIT_SECTION>  lockMgr(sm_CritSec, TRUE);
    FARPROC pfn = GetProcAddressWindows(hmod, pszProcName);
    HookedFunction* pFuncHook = 
        sm_pHookedFunctions->GetHookedFunction(
            hmod, 
            pszProcName
            );
    if (NULL != pFuncHook)
    {
        pfn = pFuncHook->Get_pfnHook();
    }
    return pfn;
}

FARPROC WINAPI HOOK_MANAGER::GetProcAddressWindows(HMODULE hmod, PCSTR pszProcName)
{
    return ::GetProcAddress(hmod, pszProcName);
}

const BYTE cPushOpCode = 0x68;   

HookedFunction::HookedFunction(HookedFunctions* pHookedFunctions, PCSTR pszCalleeModName, PCSTR pszFuncName, 
    PROC pfnOrig, PROC pfnHook):
    m_pHookedFunctions(pHookedFunctions),
    m_bHooked(FALSE),
    m_pfnOrig(pfnOrig),
    m_pfnHook(pfnHook)
{
    strcpy(m_szCalleeModName, pszCalleeModName); 
    strcpy(m_szFuncName, pszFuncName);  
}

HookedFunction::~HookedFunction()
{
    UnHookImport();
}


PCSTR HookedFunction::Get_CalleeModName() const
{
    return const_cast<PCSTR>(m_szCalleeModName);
}

PCSTR HookedFunction::Get_FuncName() const
{
    return const_cast<PCSTR>(m_szFuncName);
}

PROC HookedFunction::Get_pfnHook() const
{
    return m_pfnHook;
}

PROC HookedFunction::Get_pfnOrig() const
{
    return m_pfnOrig;
}

BOOL HookedFunction::HookImport()
{
    m_bHooked = DoHook(TRUE, m_pfnOrig, m_pfnHook);
    return m_bHooked;
}

BOOL HookedFunction::UnHookImport()
{
    if (m_bHooked)
        m_bHooked = !DoHook(FALSE, m_pfnHook, m_pfnOrig);
    return !m_bHooked;
}

BOOL HookedFunction::ReplaceInAllModules(
    BOOL  bHookOrRestore,
    PCSTR pszCalleeModName, 
    PROC  pfnCurrent, 
    PROC  pfnNew
    ) 
{
    BOOL bResult = FALSE;

    if ((NULL != pfnCurrent) && (NULL != pfnNew))
    {
        BOOL                bReplace  = FALSE;
        ExecutableModule  *pProcess = NULL;
        PROCESS_MANAGER        process_mgr; 
        Module     *pModule;
        process_mgr.PopulateProcess(::GetCurrentProcessId(), TRUE);
        pProcess = process_mgr.GetProcessById(::GetCurrentProcessId());
        if (NULL != pProcess)
        {
            for (int i = 0; i < pProcess->GetModuleCount(); i++)
            {
                pModule = pProcess->GetModuleByIndex(i);
                bReplace = (pModule->Get_Module() != ModuleFromAddress(HOOK_MANAGER::MyLoadLibraryA)); 
                if (bReplace)
                    bResult = ReplaceInOneModule(
                        pszCalleeModName, 
                        pfnCurrent, 
                        pfnNew, 
                        pModule->Get_Module()
                        ) || bResult;
            }
            bResult = ReplaceInOneModule(
                pszCalleeModName, 
                pfnCurrent, 
                pfnNew, 
                pProcess->Get_Module()
                ) || bResult;
        }
    }
    return bResult;
}


char * __cdecl mstrstr (
        const char * str1,
        const char * str2
        )
{
        char *cp = (char *) str1;
        char *s1, *s2;

        if ( !*str2 )
            return((char *)str1);

        while (*cp)
        {
                s1 = cp;
                s2 = (char *) str2;

                while ( *s1 && *s2 && !(*s1-*s2) )
                        s1++, s2++;

                if (!*s2)
                        return(cp);

                cp++;
        }

        return(NULL);

}

BOOL HookedFunction::ReplaceInOneModule(PCSTR pszCalleeModName, PROC pfnCurrent, PROC pfnNew, HMODULE hmodCaller) 
{
    BOOL bResult = FALSE;
    __try
    {
        ULONG ulSize;
        PIMAGE_IMPORT_DESCRIPTOR pImportDesc = 
            (PIMAGE_IMPORT_DESCRIPTOR)ImageDirectoryEntryToData(
            hmodCaller,
            TRUE, 
            IMAGE_DIRECTORY_ENTRY_IMPORT, 
            &ulSize
            );
        if (pImportDesc == NULL)
            __leave;  
        while (pImportDesc->Name)
        {
            PSTR pszModName = (PSTR)((PBYTE) hmodCaller + pImportDesc->Name);
            char import_mod_name[4000];
            for (int i = 0; ; i++)
            {
                import_mod_name[i] = tolower(pszModName[i]);
                if (pszModName[i] == '\0')
                    break;
            }

            char import_callee_mod_name[4000];
            for (int i = 0; ; i++)
            {
                import_callee_mod_name[i] = tolower(pszCalleeModName[i]);
                if (pszCalleeModName[i] == '\0')
                    break;
            }

//          std::cout << "pszModName " << import_mod_name << " pszCalleeModName " << import_callee_mod_name << "\n";
            if (stricmp(import_mod_name, import_callee_mod_name) == 0) 
                break;   // Found
            if (stricmp(import_callee_mod_name, import_mod_name) == 0) 
                break;   // Found
            if (mstrstr(import_callee_mod_name, import_mod_name) != 0)
                break;   // Found
            if (mstrstr(import_mod_name, import_callee_mod_name) != 0)
                break;   // Found
            pImportDesc++;
        }
        if (pImportDesc->Name == 0)
            __leave;
        // Get caller's IAT 
        PIMAGE_THUNK_DATA pThunk = 
            (PIMAGE_THUNK_DATA)( (PBYTE) hmodCaller + pImportDesc->FirstThunk );
        while (pThunk->u1.Function)
        {
            PROC* ppfn = (PROC*) &pThunk->u1.Function;
            BOOL bFound = (*ppfn == pfnCurrent);
            if (bFound) 
            {
                MEMORY_BASIC_INFORMATION mbi;
        
                ::VirtualQuery(ppfn, &mbi, sizeof(MEMORY_BASIC_INFORMATION));

                //if (FALSE == ::VirtualProtect(
                //    mbi.BaseAddress,
                //    mbi.RegionSize,
                //    PAGE_READWRITE,
                //    &mbi.Protect)
                //    )
                //    __leave;
                DWORD dwOldProtect;
                if (FALSE == ::VirtualProtect(
                    ppfn,
                    4,
                    PAGE_READWRITE,
                    &dwOldProtect)
                    )
                    __leave;

                *ppfn = *pfnNew;
                bResult = TRUE;
                //::VirtualProtect(
                //    mbi.BaseAddress,
                //    mbi.RegionSize,
                //    mbi.Protect,
                //    &dwOldProtect
                //    );
                DWORD dwTemp;
                ::VirtualProtect(
                    ppfn,
                    4,
                    dwOldProtect,
                    &dwTemp
                    );
                this->m_iatList.push_back(ppfn);

                break;
            }
            pThunk++;
        }
    }
    __finally
    {
    }
    return bResult;
}

bool HookedFunction::Contains(void * iat)
{
    std::list<void*>::iterator it = this->m_iatList.begin();
    while (it != this->m_iatList.end())
    {
        if (*it == iat)
            return true;
        ++it;
    }
    return false;
}

BOOL HookedFunction::DoHook(BOOL bHookOrRestore, PROC pfnCurrent, PROC pfnNew)
{
    return ReplaceInAllModules(bHookOrRestore, m_szCalleeModName, pfnCurrent, pfnNew);
}

HookedFunction* HookedFunctions::GetHookedFunction(HMODULE hmodOriginal, PCSTR pszFuncName)
{
    char szFileName[MAX_PATH];
    // Hmm, does not seem to get full path name.
    ::GetModuleFileNameA(hmodOriginal, szFileName, MAX_PATH);
    ExtractModuleFileName(szFileName);
    // Get full path.
    HANDLE  hProcess = GetCurrentProcess();
    char    szModuleName[MAX_PATH];
    BOOL rv_gmfn = ::GetModuleFileNameExA(hProcess, hmodOriginal, szModuleName, sizeof(szModuleName));
    if (! rv_gmfn)
    {
        CloseHandle(hProcess);
        return FALSE;
    }
    return GetHookedFunction(szModuleName, pszFuncName);
}

HookedFunction* HookedFunctions::GetHookedFunction(void * iat)
{
    // For each hooked function find iat.
    HookedFunctions::iterator itr = begin();
    while (itr != end())
    {
        HookedFunction * hf = itr->second;
        if (hf->Contains(iat))
            return hf;
        itr++;
    }
    return 0;
}


BOOL HookedFunctions::GetFunctionNameFromExportSection(HMODULE hmodOriginal, DWORD dwFuncOrdinalNum, PSTR pszFuncName) 
{
    BOOL bResult = FALSE;
    strcpy(pszFuncName, "\0");
    __try
    {
        ULONG ulSize;
        PIMAGE_EXPORT_DIRECTORY pExportDir = 
            (PIMAGE_EXPORT_DIRECTORY)ImageDirectoryEntryToData(
            hmodOriginal, 
            TRUE, 
            IMAGE_DIRECTORY_ENTRY_EXPORT, 
            &ulSize
            );
        if (pExportDir == NULL)
            __leave;  
        PSTR pszDllName = reinterpret_cast<PSTR>( pExportDir->Name + (DWORD)hmodOriginal);
        DWORD dwFuncNumber = pExportDir->Base;
        DWORD dwNumberOfExported = pExportDir->NumberOfFunctions;
        PDWORD pdwFunctions = (PDWORD)( pExportDir->AddressOfFunctions + (DWORD)hmodOriginal);
        PWORD pwOrdinals = (PWORD)(pExportDir->AddressOfNameOrdinals + (DWORD)hmodOriginal);
        DWORD *pszFuncNames =   (DWORD *)(pExportDir->AddressOfNames + (DWORD)hmodOriginal);
        PSTR pszExpFunName;
        for (long i = 0; i < dwNumberOfExported; i++, pdwFunctions++)
        {
            DWORD entryPointRVA = *pdwFunctions;
            if ( entryPointRVA == 0 )
                continue;
            for ( unsigned j=0; j < pExportDir->NumberOfNames; j++ )
            {
                if ( pwOrdinals[j] == i  )
                {
                    pszExpFunName = (PSTR)(pszFuncNames[j] + (DWORD)hmodOriginal);
                    if (dwFuncOrdinalNum == pwOrdinals[j] + 1)
                    {
                        if ((pszExpFunName != NULL) && (strlen(pszExpFunName) > 0))
                            strcpy(pszFuncName, pszExpFunName);
                        __leave;
                    }
                }
            }
        }
    }
    __finally
    {
    }
    return bResult;
}

void HookedFunctions::GetFunctionNameByOrdinal(PCSTR pszCalleeModName, DWORD dwFuncOrdinalNum, PSTR pszFuncName)
{
    HMODULE hmodOriginal = ::GetModuleHandleA(pszCalleeModName);
    GetFunctionNameFromExportSection(hmodOriginal, dwFuncOrdinalNum, pszFuncName);
}

HookedFunction* HookedFunctions::GetHookedFunction(PCSTR pszCalleeModName, PCSTR pszFuncName)
{
    HookedFunction* pHook = NULL;
    char szFuncName[MAX_PATH];
    if ( (pszFuncName) && 
         ((DWORD)pszFuncName > 0xFFFF) && 
         strlen(pszFuncName) ) 
    {
        strcpy(szFuncName, pszFuncName);
    }
    else
    {
        GetFunctionNameByOrdinal(pszCalleeModName, (DWORD)pszFuncName, szFuncName);
    }
    if (strlen(szFuncName) > 0)
    {
        char szKey[MAX_PATH];
		// This routine could be called during an exit thread event.  Calling the C runtime causes the buffers
		// to be reallocated, and get possibly confused, because the C runtime should be killed on exit thread.
		// Just use normal concatenation instead of sprintf.
        //sprintf(
        //    szKey, 
        //    "<%s><%s>", 
        //    pszCalleeModName,
        //    szFuncName
        //    );
		strcpy(szKey, pszCalleeModName);
		strcat(szKey, "|");
		strcat(szKey, szFuncName);
        HookedFunctions::const_iterator citr = find( szKey );
        if ( citr != end() )
            pHook = citr->second;
    }
    return pHook;
}



BOOL HookedFunctions::AddHook(HookedFunction* pHook)
{
    BOOL bResult = FALSE;
    if (NULL != pHook)
    {
        char szKey[MAX_PATH];
        //sprintf(
        //    szKey, 
        //    "<%s><%s>", 
        //    pHook->Get_CalleeModName(),
        //    pHook->Get_FuncName()
        //    );
		strcpy(szKey, pHook->Get_CalleeModName());
		strcat(szKey, "|");
		strcat(szKey, pHook->Get_FuncName());

        HookedFunctions::iterator lb = lower_bound(szKey);
        insert( lb, value_type(szKey, pHook) );
        bResult = TRUE;
    }
    return bResult;
}

BOOL HookedFunctions::RemoveHook(HookedFunction* pHook)
{
    BOOL bResult = FALSE;
    try
    {
        if (NULL != pHook)
        {
            char szKey[MAX_PATH];
            //sprintf(
            //    szKey, 
            //    "<%s><%s>", 
            //    pHook->Get_CalleeModName(),
            //    pHook->Get_FuncName()
            //    );
			strcpy(szKey, pHook->Get_CalleeModName());
			strcat(szKey, "|");
			strcat(szKey, pHook->Get_FuncName());

            HookedFunctions::iterator itr = find(szKey);
            if (itr != end())
            {
                delete itr->second;
                erase(itr);
            }
            bResult = TRUE;
        }
    }
    catch (...)
    {
        bResult = FALSE;
    }
    return bResult;
}
