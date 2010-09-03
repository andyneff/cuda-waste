#pragma once
#include <list>

class CallStackInfo
{
    CallStackInfo();
    std::list<char*> ignore_files;
    int context_size;
public:
    static CallStackInfo * Singleton();
    bool GetModuleNameFromAddress(/* in */ void * address, /* in-out */ char * lpszModule );
    bool GetFunctionInfoFromAddresses( void * fnAddress, void * stackAddress, char * lpszSymbol );
    bool GetSourceInfoFromAddress( void * address, char * lpszSourceInfo, char * full_file_name );
    char * Context(int lines = 1);
    void ClassifyAsPrefix(char * file);
    std::list<void*> * CallTree();
};
