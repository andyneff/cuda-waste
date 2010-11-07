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

#include <iostream>
#include <vector>
#include <windows.h>
#include <tlhelp32.h>

class Module;
class ModuleList;
class CRunningProcesses;
class ExecutableModule;
class CLibHandler;
class CRunningProcesses;

typedef HANDLE (WINAPI * PFNCREATETOOLHELP32SNAPSHOT)(DWORD dwFlags, DWORD th32ProcessID);
typedef BOOL (WINAPI * PFNPROCESS32FIRST)(HANDLE hSnapshot, LPPROCESSENTRY32 lppe);
typedef BOOL (WINAPI * PFNPROCESS32NEXT)(HANDLE hSnapshot, LPPROCESSENTRY32 lppe);
typedef BOOL (WINAPI * PFNMODULE32FIRST)(HANDLE hSnapshot, LPMODULEENTRY32 lpme);
typedef BOOL (WINAPI * PFNMODULE32NEXT)(HANDLE hSnapshot, LPMODULEENTRY32 lpme);
typedef BOOL (WINAPI * PFNENUMPROCESSES)(DWORD * lpidProcess, DWORD cb, DWORD * cbNeeded);
typedef BOOL (WINAPI * PFNENUMPROCESSMODULES)(HANDLE hProcess, HMODULE *lphModule, DWORD cb, LPDWORD lpcbNeeded);
typedef DWORD (WINAPI * PFNGETMODULEFILENAMEEXA)(HANDLE hProcess, HMODULE hModule, LPSTR lpFilename, DWORD nSize);


class PROCESS_MANAGER  
{
public:
    PROCESS_MANAGER();
    virtual ~PROCESS_MANAGER();
    BOOL Populate(BOOL bPopulateModules);
    BOOL PopulateProcess(DWORD dwProcessId, BOOL bPopulateModules);
    DWORD GetProcessCount() const;
    ExecutableModule* GetProcessById(DWORD dwProcessId);
private:
    CLibHandler       *m_pLibHandler;
    CRunningProcesses *m_pProcesses;
};




class ModuleList: private std::vector<Module*>  
{
public:
    ModuleList();
    virtual ~ModuleList();
    void Add(Module &moduleInstance);
    void ReleaseAll();
    Module* GetModule(DWORD dwIndex) const;
    DWORD GetCount() const;
};

class CElements: public ModuleList  
{
public:
    CElements() : ModuleList() {}
    virtual ~CElements() {};
};

class CLibHandler  
{
public:
    CLibHandler(CRunningProcesses* pProcesses) : m_pProcesses(pProcesses) {};
    virtual ~CLibHandler() {};
    virtual BOOL PopulateModules(Module* pProcess) = 0;
    virtual BOOL PopulateProcesses(BOOL bPopulateModules) = 0;
    virtual BOOL PopulateProcess(DWORD dwProcessId, BOOL bPopulateModules) = 0;
protected:
    CRunningProcesses* m_pProcesses;
};

class CLoadedModules: public CElements
{
public:
    CLoadedModules(DWORD dwProcessId);
    virtual ~CLoadedModules();
protected:
    DWORD m_dwProcessId;
};

class CRunningProcesses: public CElements
{
public:
    CRunningProcesses();
    virtual ~CRunningProcesses();
    ExecutableModule* GetProcessById(DWORD dwProcessId);
};

class CPsapiHandler: public CLibHandler
{
public:
    CPsapiHandler(CRunningProcesses* pProcesses);
    virtual ~CPsapiHandler();
    BOOL Initialize();
    void Finalize();
    virtual BOOL PopulateModules(Module* pProcess);
    virtual BOOL PopulateProcesses(BOOL bPopulateModules);
    virtual BOOL PopulateProcess(DWORD dwProcessId, BOOL bPopulateModules);
private:
    HMODULE                 m_hModPSAPI;
    PFNENUMPROCESSES        m_pfnEnumProcesses;
    PFNENUMPROCESSMODULES   m_pfnEnumProcessModules;
    PFNGETMODULEFILENAMEEXA m_pfnGetModuleFileNameExA;
};

class Module  
{
public:
    Module(char *pszName, HMODULE hModule);
    virtual ~Module();
    void AddModule(Module* pModuleInstance);
    void ReleaseModules();
    char*    Get_Name() const;
    void     Set_Name(char *pszName);
    HMODULE  Get_Module() const;
    void     Set_Module(HMODULE module);
    char*   GetBaseName() const;
private:
    char        *m_pszName;
    HMODULE      m_hModule;
protected:
    ModuleList *m_pInternalList;
};

class ExecutableModule: public Module
{
public:
    ExecutableModule(CLibHandler* pLibHandler, char* pszName, HMODULE hModule, DWORD dwProcessId);
    virtual ~ExecutableModule();
    DWORD Get_ProcessId() const;
    BOOL PopulateModules();
    DWORD GetModuleCount();
    Module* GetModuleByIndex(DWORD dwIndex);
private:
    DWORD        m_dwProcessId;
    CLibHandler* m_pLibHandler;
};
