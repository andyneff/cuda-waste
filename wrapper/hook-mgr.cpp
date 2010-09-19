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

#pragma comment(lib, "imagehlp.lib")


HMODULE ModuleFromAddress(PVOID pv) 
{
    MEMORY_BASIC_INFORMATION mbi;
    return ((::VirtualQuery(pv, &mbi, sizeof(mbi)) != 0) 
            ? (HMODULE) mbi.AllocationBase : NULL);
}

bool ExtractModuleFileName(char* pszFullFileName)
{
    bool  bResult = false;
    if (TRUE != ::IsBadReadPtr(pszFullFileName, MAX_PATH))
    {
        char  *pdest;
        int   ch = '\\';
        pdest = strrchr(pszFullFileName, ch);
        if( pdest != NULL )
            strcpy(pszFullFileName, &pdest[1]);
        bResult = true;
    }
    return bResult;
}


HookedFunctions* HookManager::sm_pHookedFunctions = NULL;
CriticalSection        HookManager::sm_CritSec;

HookManager::HookManager()
{
    m_hmodThisInstance   = ModuleFromAddress(HookManager::MyGetProcAddress);
    m_bSystemFuncsHooked = false;
    sm_pHookedFunctions  = new HookedFunctions(this); 
}

HookManager::~HookManager()
{
    UnHookAllFuncs();
    delete sm_pHookedFunctions;
}

bool HookManager::HookSystemFuncs()
{
    if (m_bSystemFuncsHooked)
    {
        HookImport("Kernel32.dll", "LoadLibraryA", (PROC) HookManager::MyLoadLibraryA, true);
        HookImport("Kernel32.dll", "LoadLibraryW", (PROC) HookManager::MyLoadLibraryW, true);
        HookImport("Kernel32.dll", "LoadLibraryExA", (PROC) HookManager::MyLoadLibraryExA, true);
        HookImport("Kernel32.dll", "LoadLibraryExW", (PROC) HookManager::MyLoadLibraryExW, true);
        m_bSystemFuncsHooked = true;
    }
    return m_bSystemFuncsHooked;
}

void HookManager::UnHookAllFuncs()
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

BOOL HookManager::AreThereHookedFunctions()
{
    return (sm_pHookedFunctions->size() > 0);
}


PROC HookManager::HookImport(PCSTR pszCalleeModName, PCSTR pszFuncName, PROC pfnHook, bool flag)
{
    LockManager<CriticalSection>  lockMgr(sm_CritSec, TRUE);

    BOOL                  bResult = FALSE;
    PROC                  pfnOrig = NULL;
    try
    {
        if (!sm_pHookedFunctions->GetHookedFunction(
                pszCalleeModName, 
                pszFuncName
                ))
        {
            pfnOrig = GetProcAddressWindows(
                ::GetModuleHandleA(pszCalleeModName),
                pszFuncName
                );
            if (NULL == pfnOrig)
            {
                HMODULE hmod = ::LoadLibraryA(pszCalleeModName);
                if (NULL != hmod)
                    pfnOrig = GetProcAddressWindows(
                        ::GetModuleHandleA(pszCalleeModName),
                        pszFuncName
                        );
            }
            if (NULL != pfnOrig)
                bResult = AddHook(
                    pszCalleeModName, 
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

HookedFunction* HookManager::FindHook(PCSTR pszCalleeModName, PCSTR pszFuncName)
{
    LockManager<CriticalSection>  lockMgr(sm_CritSec, TRUE);
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

HookedFunction * HookManager::FindHook(void * iat)
{
    LockManager<CriticalSection>  lockMgr(sm_CritSec, TRUE);
    HookedFunction * result = 0;
    try
    {
        result = sm_pHookedFunctions->GetHookedFunction(iat);
    }
    catch(...) {}
    return result;
}



PROC HookManager::FindOriginal(PROC wrapper_function)
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

BOOL HookManager::UnHookImport(PCSTR pszCalleeModName, PCSTR pszFuncName)
{
    LockManager<CriticalSection>  lockMgr(sm_CritSec, TRUE);

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

BOOL HookManager::AddHook(PCSTR pszCalleeModName, PCSTR pszFuncName, PROC pfnOrig, PROC pfnHook)
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
        pHook->HookImport();
        bResult = sm_pHookedFunctions->AddHook(pHook);
    }
    return bResult;
}

BOOL HookManager::RemoveHook(PCSTR pszCalleeModName, PCSTR pszFuncName)
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

void WINAPI HookManager::HackModuleOnLoad(HMODULE hmod, DWORD dwFlags)
{
    if ((hmod != NULL) && ((dwFlags & LOAD_LIBRARY_AS_DATAFILE) == 0)) 
    {
        LockManager<CriticalSection>  lockMgr(sm_CritSec, TRUE);
        
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

HMODULE WINAPI HookManager::MyLoadLibraryA(PCSTR pszModuleName)
{
    HMODULE hmod = ::LoadLibraryA(pszModuleName);
    HackModuleOnLoad(hmod, 0);

    return hmod;
}

HMODULE WINAPI HookManager::MyLoadLibraryW(PCWSTR pszModuleName)
{
    HMODULE hmod = ::LoadLibraryW(pszModuleName);
    HackModuleOnLoad(hmod, 0);
    return hmod;
}

HMODULE WINAPI HookManager::MyLoadLibraryExA(PCSTR pszModuleName, HANDLE hFile, DWORD dwFlags)
{
    HMODULE hmod = ::LoadLibraryExA(pszModuleName, hFile, dwFlags);
    HackModuleOnLoad(hmod, 0);
    return hmod;
}

HMODULE WINAPI HookManager::MyLoadLibraryExW(PCWSTR pszModuleName, HANDLE hFile, DWORD dwFlags)
{
    HMODULE hmod = ::LoadLibraryExW(pszModuleName, hFile, dwFlags);
    HackModuleOnLoad(hmod, 0);
    return hmod;
}

FARPROC WINAPI HookManager::MyGetProcAddress(HMODULE hmod, PCSTR pszProcName)
{
    LockManager<CriticalSection>  lockMgr(sm_CritSec, TRUE);
    FARPROC pfn = GetProcAddressWindows(hmod, pszProcName);
    HookedFunction* pFuncHook = 
        sm_pHookedFunctions->GetHookedFunction(
            hmod, 
            pszProcName
            );
    if (NULL != pFuncHook)
        pfn = pFuncHook->Get_pfnHook();
    return pfn;
}

FARPROC WINAPI HookManager::GetProcAddressWindows(HMODULE hmod, PCSTR pszProcName)
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
        ProcessManager        process_mgr; 
        Module     *pModule;
        process_mgr.PopulateProcess(::GetCurrentProcessId(), TRUE);
        pProcess = process_mgr.GetProcessById(::GetCurrentProcessId());
        if (NULL != pProcess)
        {
            for (int i = 0; i < pProcess->GetModuleCount(); i++)
            {
                pModule = pProcess->GetModuleByIndex(i);
                bReplace = (pModule->Get_Module() != ModuleFromAddress(HookManager::MyLoadLibraryA)); 
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
            if (stricmp(pszModName, pszCalleeModName) == 0) 
                break;   // Found
            else if (strstr(pszCalleeModName, pszModName) != 0)
                break;
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
                if (FALSE == ::VirtualProtect(
                    mbi.BaseAddress,
                    mbi.RegionSize,
                    PAGE_READWRITE,
                    &mbi.Protect)
                    )
                    __leave;
                this->m_iatList.push_back(ppfn);
                *ppfn = *pfnNew;
                bResult = TRUE;
                DWORD dwOldProtect;
                ::VirtualProtect(
                    mbi.BaseAddress,
                    mbi.RegionSize,
                    mbi.Protect,
                    &dwOldProtect
                    );
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
    ::GetModuleFileNameA(hmodOriginal, szFileName, MAX_PATH);
    ExtractModuleFileName(szFileName);
    return GetHookedFunction(szFileName, pszFuncName);
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
        sprintf(
            szKey, 
            "<%s><%s>", 
            pszCalleeModName,
            szFuncName
            );
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
        sprintf(
            szKey, 
            "<%s><%s>", 
            pHook->Get_CalleeModName(),
            pHook->Get_FuncName()
            );
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
            sprintf(
                szKey, 
                "<%s><%s>", 
                pHook->Get_CalleeModName(),
                pHook->Get_FuncName()
                );
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
