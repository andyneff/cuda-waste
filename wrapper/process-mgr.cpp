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
#include "process-mgr.h"

PROCESS_MANAGER::PROCESS_MANAGER():
    m_pLibHandler(NULL)
{
    m_pProcesses = new CRunningProcesses();
    m_pLibHandler = new CPsapiHandler(m_pProcesses);
}

PROCESS_MANAGER::~PROCESS_MANAGER()
{
    delete m_pLibHandler;
    delete m_pProcesses;
}

BOOL PROCESS_MANAGER::Populate(BOOL bPopulateModules)
{
    m_pProcesses->ReleaseAll();
    return m_pLibHandler->PopulateProcesses(bPopulateModules); 
}

BOOL PROCESS_MANAGER::PopulateProcess(DWORD dwProcessId, BOOL bPopulateModules)
{
    m_pProcesses->ReleaseAll();
    return m_pLibHandler->PopulateProcess(dwProcessId, bPopulateModules); 
}

DWORD PROCESS_MANAGER::GetProcessCount() const
{
    return m_pProcesses->GetCount();
}

ExecutableModule* PROCESS_MANAGER::GetProcessById(DWORD dwProcessId)
{
    return static_cast<ExecutableModule*>
        (m_pProcesses->GetProcessById(dwProcessId));
}

CLoadedModules::CLoadedModules(DWORD dwProcessId):
    CElements(),
    m_dwProcessId(dwProcessId)
{
    
}

CLoadedModules::~CLoadedModules()
{
    
}

CRunningProcesses::CRunningProcesses():
    CElements()
{

}

CRunningProcesses::~CRunningProcesses()
{

}

ExecutableModule* CRunningProcesses::GetProcessById(DWORD dwProcessId)
{
    ExecutableModule* pResult = NULL;
    ExecutableModule* pProcess;
    for (long i = 0; i < GetCount(); i++)
    {
        pProcess = static_cast<ExecutableModule*>( GetModule(i) );
        if (pProcess->Get_ProcessId() == dwProcessId)
        {
            pResult = pProcess;
            break;
        }
    }
    return pResult;
}

CPsapiHandler::CPsapiHandler(CRunningProcesses* pProcesses):
    CLibHandler(pProcesses),
    m_hModPSAPI(NULL),
    m_pfnEnumProcesses(NULL),
    m_pfnEnumProcessModules(NULL),
    m_pfnGetModuleFileNameExA(NULL)
{
    
}

CPsapiHandler::~CPsapiHandler()
{
    Finalize();
}

BOOL CPsapiHandler::Initialize()
{
    BOOL bResult = FALSE;
    //
    // Get to the 3 functions in PSAPI.DLL dynamically.  We can't
    // be sure that PSAPI.DLL has been installed
    //
    if (NULL == m_hModPSAPI)
        m_hModPSAPI = ::LoadLibraryA("PSAPI.DLL");

    if (NULL != m_hModPSAPI)
    {
        m_pfnEnumProcesses = (PFNENUMPROCESSES)
            ::GetProcAddress(m_hModPSAPI,"EnumProcesses");

        m_pfnEnumProcessModules = (PFNENUMPROCESSMODULES)
            ::GetProcAddress(m_hModPSAPI, "EnumProcessModules");

        m_pfnGetModuleFileNameExA = (PFNGETMODULEFILENAMEEXA)
            ::GetProcAddress(m_hModPSAPI, "GetModuleFileNameExA");

    }
    bResult = 
        m_pfnEnumProcesses
        &&  m_pfnEnumProcessModules
        &&  m_pfnGetModuleFileNameExA;

    return bResult;
}

void CPsapiHandler::Finalize()
{
    if (NULL != m_hModPSAPI)
        ::FreeLibrary(m_hModPSAPI);
}

BOOL CPsapiHandler::PopulateModules(Module* pProcess)
{
    BOOL   bResult = TRUE;
    Module  *pDllModuleInstance = NULL;

    if (TRUE == Initialize())
    {
        DWORD pidArray[1024];
        DWORD cbNeeded;
        DWORD nProcesses;
        // EnumProcesses returns an array with process IDs
        if (m_pfnEnumProcesses(pidArray, sizeof(pidArray), &cbNeeded))
        {
            // Determine number of processes
            nProcesses = cbNeeded / sizeof(DWORD);  
            // Release the container
            pProcess->ReleaseModules();

            for (DWORD i = 0; i < nProcesses; i++)
            {
                HMODULE hModuleArray[1024];
                HANDLE  hProcess;
                DWORD   pid = pidArray[i];
                DWORD   nModules;
                // Let's open the process
                hProcess = ::OpenProcess(
                    PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                    FALSE, pid);

                if (!hProcess)
                    continue;

                if (static_cast<ExecutableModule*>(pProcess)->Get_ProcessId() != pid)
                {
                    ::CloseHandle(hProcess);
                    continue;
                }

                // EnumProcessModules function retrieves a handle for 
                // each module in the specified process. 
                if (!m_pfnEnumProcessModules(
                        hProcess, hModuleArray,
                        sizeof(hModuleArray), &cbNeeded))
                {
                    ::CloseHandle(hProcess);
                    continue;
                }

                // Calculate number of modules in the process                                   
                nModules = cbNeeded / sizeof(hModuleArray[0]);
                for (DWORD j = 0; j < nModules; j++)
                {
                    HMODULE hModule = hModuleArray[j];
                    char    szModuleName[MAX_PATH];
                    m_pfnGetModuleFileNameExA(
                        hProcess, 
                        hModule,
                        szModuleName, 
                        sizeof(szModuleName)
                        );

                    if (0 == j)   // First module is the EXE.  
                    {
                        // Do nothing.
                    } // if
                    else    // Not the first module.  It's a DLL
                    {
                        pDllModuleInstance = new Module(
                            szModuleName, 
                            hModule
                            );
                        pProcess->AddModule(pDllModuleInstance);
                    } // else
                } // for
                ::CloseHandle(hProcess);    // We're done with this process handle
            } // for
            bResult = TRUE;
        } // if
        else
        {
            bResult = FALSE;
        }
    } // if 
    else
    {
        bResult = FALSE;
    }
    return bResult;
}

BOOL CPsapiHandler::PopulateProcesses(BOOL bPopulateModules)
{
    BOOL   bResult = TRUE;
    ExecutableModule* pProcessInfo;
    char    szModuleName[MAX_PATH];
    
    if (TRUE == Initialize())
    {
        DWORD pidArray[1024];
        DWORD cbNeeded;
        DWORD nProcesses;
      
        if (m_pfnEnumProcesses(pidArray, sizeof(pidArray), &cbNeeded))
        {
            // Determine number of processes
            nProcesses = cbNeeded / sizeof(DWORD);  
            m_pProcesses->ReleaseAll();
            for (DWORD i = 0; i < nProcesses; i++)
            {
                HMODULE hModuleArray[1024];
                HANDLE hProcess;
                DWORD pid = pidArray[i];
                DWORD nModules;
                hProcess = ::OpenProcess(
                    PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                    FALSE, 
                    pid
                    );
                if (!hProcess)
                    continue;

                if (!m_pfnEnumProcessModules(hProcess, hModuleArray,
                                            sizeof(hModuleArray), &cbNeeded))
                {
                    ::CloseHandle(hProcess);
                    continue;
                }
                // Calculate number of modules in the process                                   
                nModules = cbNeeded / sizeof(hModuleArray[0]);

                for (DWORD j = 0; j < nModules; j++)
                {
                    HMODULE hModule = hModuleArray[j];

                    m_pfnGetModuleFileNameExA(
                        hProcess, 
                        hModule,
                        szModuleName, 
                        sizeof(szModuleName)
                        );

                    if (0 == j)   // First module is the EXE.  Just add it to the map
                    {
                        pProcessInfo = new ExecutableModule(
                            this,
                            szModuleName, 
                            hModule, 
                            pid
                            );
                        m_pProcesses->Add(*pProcessInfo);
                        if (bPopulateModules)
                            pProcessInfo->PopulateModules();
                        break;
                    } // if
                } // for
                ::CloseHandle(hProcess);    
            } // for
   
            bResult = TRUE;
        } // if
        else
        {
            bResult = FALSE;
        }
    } // if 
    else
    {
        bResult = FALSE;
    }
   
    return bResult;
}   

BOOL CPsapiHandler::PopulateProcess(DWORD dwProcessId, BOOL bPopulateModules)
{
    BOOL   bResult = TRUE;
    ExecutableModule* pProcessInfo;
    
    if (TRUE == Initialize())
    {
        m_pProcesses->ReleaseAll();
        HMODULE hModuleArray[1024];
        HANDLE  hProcess;
        DWORD   nModules;
        DWORD   cbNeeded;
        hProcess = ::OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
            FALSE, 
            dwProcessId
            );
        if (hProcess) 
        {
            if (!m_pfnEnumProcessModules(
                hProcess, 
                hModuleArray,
                sizeof(hModuleArray), 
                &cbNeeded
                ))
                ::CloseHandle(hProcess);
            else
            {
                // Calculate number of modules in the process                                   
                nModules = cbNeeded / sizeof(hModuleArray[0]);

                for (DWORD j = 0; j < nModules; j++)
                {
                    HMODULE hModule = hModuleArray[j];
                    char    szModuleName[MAX_PATH];

                    m_pfnGetModuleFileNameExA(
                        hProcess, 
                        hModule,
                        szModuleName, 
                        sizeof(szModuleName)
                        );

                    if (0 == j)   // First module is the EXE.  Just add it to the map
                    {
                        pProcessInfo = new ExecutableModule(
                            this,
                            szModuleName, 
                            hModule, 
                            dwProcessId
                            );
                        m_pProcesses->Add(*pProcessInfo);
                        if (bPopulateModules)
                            pProcessInfo->PopulateModules();
                        break;
                    } // if
                } // for
                ::CloseHandle(hProcess);    
            } // if
        } // if
    } // if 
    else
    {
        bResult = FALSE;
    }
    return bResult;
}


ModuleList::ModuleList()
{

}

ModuleList::~ModuleList()
{
    ReleaseAll();
}

void ModuleList::Add(Module &moduleInstance)
{
    push_back(&moduleInstance);
}

void ModuleList::ReleaseAll()
{
    ModuleList::iterator itr;
    Module *pModuleInstance;

    for (itr = begin(); itr != end(); ++itr)
    {
        pModuleInstance = *itr;
        delete pModuleInstance;
    } // for
    clear();
}

Module* ModuleList::GetModule(DWORD dwIndex) const
{
    return at(dwIndex); 
}

DWORD ModuleList::GetCount() const
{
    return size();
}

Module::Module(char *pszName, HMODULE hModule):
    m_pszName(NULL),
    m_hModule(NULL),
    m_pInternalList(NULL)
{
    Set_Name(pszName);
    Set_Module(hModule);
    
}

Module::~Module()
{
    delete m_pInternalList;

    if (NULL != m_pszName)
        delete [] m_pszName;    
}


void Module::AddModule(Module* pModuleInstance)
{
    if (NULL == m_pInternalList)
        m_pInternalList = new ModuleList();
    m_pInternalList->Add(*pModuleInstance);
}

void Module::ReleaseModules()
{
    if (NULL != m_pInternalList)
        m_pInternalList->ReleaseAll();
}

char* Module::Get_Name() const
{
    return m_pszName;
}

char* Module::GetBaseName() const
{
    char *pdest;
    int  ch = '\\';
    // Search backward 
    pdest = strrchr(m_pszName, ch);
    if(pdest != NULL)
        return &pdest[1];
    else
        return m_pszName;
}

void Module::Set_Name(char *pszName)
{
    if (NULL != m_pszName)
        delete [] m_pszName;    
    if ((NULL != pszName) && (strlen(pszName)))
    {
        m_pszName = new char[strlen(pszName) + 1];
        strcpy(m_pszName, pszName);
    }
    else
    {
        m_pszName = new char[strlen("\0") + 1];
        strcpy(m_pszName, "\0");
    }

}

HMODULE Module::Get_Module() const
{
    return m_hModule;
}

void Module::Set_Module(HMODULE module)
{
    m_hModule = module;
}

ExecutableModule::ExecutableModule(
    CLibHandler* pLibHandler,
    char*        pszName, 
    HMODULE      hModule, 
    DWORD        dwProcessId
    ):
    Module(pszName, hModule),
    m_pLibHandler(pLibHandler),
    m_dwProcessId(dwProcessId)
{
    
}

ExecutableModule::~ExecutableModule()
{
    
}

DWORD ExecutableModule::Get_ProcessId() const
{
    return m_dwProcessId;
}

BOOL ExecutableModule::PopulateModules()
{
    return m_pLibHandler->PopulateModules(this);
}


DWORD ExecutableModule::GetModuleCount()
{
    return m_pInternalList ? m_pInternalList->GetCount() : 0;
}

Module* ExecutableModule::GetModuleByIndex(DWORD dwIndex)
{
    return m_pInternalList ? m_pInternalList->GetModule(dwIndex) : NULL;
}
