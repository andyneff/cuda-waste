#include <stdafx.h>
#include <string.h>
#include "DoRPC_h.h"

void __RPC_FAR * __RPC_API midl_user_allocate(size_t cBytes) 
{ 
    return(malloc(cBytes)); 
}

void __RPC_API midl_user_free(void __RPC_FAR * p) 
{ 
    free(p); 
}
