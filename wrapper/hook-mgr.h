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
#pragma once


#include <map>
#include <string>
#include <list>
using namespace std;
#include "process-mgr.h"
#include "lock-mgr.h"

class HookedFunctions;
class HookedFunction;

class HookManager  
{
public:
    HookManager();
    virtual ~HookManager();
public:
    PROC HookImport(PCSTR pszCalleeModName, PCSTR pszFuncName, PROC pfnHook, bool flag);
    HookedFunction* FindHook(PCSTR pszCalleeModName, PCSTR pszFuncName);
    HookedFunction * FindHook(void * iat);
    PROC FindOriginal(PROC wrapper_function);
    BOOL UnHookImport(PCSTR pszCalleeModName, PCSTR pszFuncName);
    bool HookSystemFuncs();
    void UnHookAllFuncs();
    BOOL AreThereHookedFunctions();
private:
    friend class HookedFunction;
    static CriticalSection sm_CritSec;
    HMODULE m_hmodThisInstance;
    static HookedFunctions* sm_pHookedFunctions;
    bool m_bSystemFuncsHooked;
    static void WINAPI HackModuleOnLoad(HMODULE hmod, DWORD dwFlags);
    static HMODULE WINAPI MyLoadLibraryA(PCSTR  pszModuleName);
    static HMODULE WINAPI MyLoadLibraryW(PCWSTR pszModuleName);
    static HMODULE WINAPI MyLoadLibraryExA(PCSTR  pszModuleName, HANDLE hFile, DWORD dwFlags);
    static HMODULE WINAPI MyLoadLibraryExW(PCWSTR pszModuleName, HANDLE hFile, DWORD dwFlags);
    static FARPROC WINAPI MyGetProcAddress(HMODULE hmod, PCSTR pszProcName);
    static FARPROC WINAPI GetProcAddressWindows(HMODULE hmod, PCSTR pszProcName);
    BOOL AddHook(PCSTR  pszCalleeModName, PCSTR pszFuncName, PROC pfnOrig, PROC pfnHook);
    BOOL RemoveHook(PCSTR pszCalleeModName, PCSTR pszFuncName);
};

class HookedFunction  
{
public:
    HookedFunction(HookedFunctions* pHookedFunctions, PCSTR pszCalleeModName, PCSTR pszFuncName, 
        PROC pfnOrig, PROC pfnHook);
    virtual ~HookedFunction();
    PCSTR Get_CalleeModName() const;
    PCSTR Get_FuncName() const;
    PROC Get_pfnHook() const;
    PROC Get_pfnOrig() const;
    BOOL HookImport();
    BOOL UnHookImport();
    BOOL ReplaceInOneModule(PCSTR pszCalleeModName, PROC pfnCurrent, PROC pfnNew, HMODULE hmodCaller);
    bool Contains(void * iat);

private:
    HookedFunctions* m_pHookedFunctions;
    BOOL m_bHooked;
    char m_szCalleeModName[MAX_PATH];
    char m_szFuncName[MAX_PATH];
    PROC m_pfnOrig;
    PROC m_pfnHook;
    std::list<void*> m_iatList;
    BOOL DoHook(BOOL bHookOrRestore, PROC pfnCurrent, PROC pfnNew);
    BOOL ReplaceInAllModules(BOOL bHookOrRestore, PCSTR pszCalleeModName, PROC pfnCurrent, PROC pfnNew);
};


class StringCompare
{
public:
    bool operator()(const string& x, const string& y) const
    {
        return ( _stricmp(x.c_str(), y.c_str()) < 0 );
    }
};

class HookedFunctions: public map<string, HookedFunction*, StringCompare>
{
public:
    HookedFunctions(HookManager* pApiHookMgr) : m_pApiHookMgr(pApiHookMgr) {};
    virtual ~HookedFunctions() {};
public:
    HookedFunction* GetHookedFunction(PCSTR pszCalleeModName, PCSTR pszFuncName);
    HookedFunction* GetHookedFunction(HMODULE hmod, PCSTR pszFuncName);
    HookedFunction* GetHookedFunction(void * iat);
    BOOL AddHook(HookedFunction* pHook);
    BOOL RemoveHook(HookedFunction* pHook);
private:
    BOOL GetFunctionNameFromExportSection(HMODULE hmodOriginal, DWORD dwFuncOrdinalNum, PSTR pszFuncName); 
    void GetFunctionNameByOrdinal(PCSTR pszCalleeModName, DWORD dwFuncOrdinalNum, PSTR pszFuncName);
    HookManager* m_pApiHookMgr;
};
