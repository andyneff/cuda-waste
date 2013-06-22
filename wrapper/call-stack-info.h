#pragma once
#include <list>

class CALL_STACK_INFO
{
    CALL_STACK_INFO();
    std::list<char*> ignore_files;
    int context_size;
	static CALL_STACK_INFO * singleton;
public:
    static CALL_STACK_INFO * Singleton();
    bool GetModuleNameFromAddress(/* in */ void * address, /* in-out */ char * lpszModule );
    bool GetFunctionInfoFromAddresses( void * fnAddress, void * stackAddress, char * lpszSymbol );
    bool GetSourceInfoFromAddress( void * address, char * lpszSourceInfo, char * full_file_name );
    char * Context(int lines = 1);
	void * * AddressContext(size_t lines = 20);
    void ClassifyAsPrefix(char * file);
    std::list<void*> * CallTree();
};
