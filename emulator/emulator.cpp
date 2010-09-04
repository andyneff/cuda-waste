#include "emulator.h"
#include <assert.h>
#include <iostream>

CUDA_EMULATOR * CUDA_EMULATOR::singleton;

CUDA_EMULATOR * CUDA_EMULATOR::Singleton()
{
    if (singleton)
        return singleton;
    singleton = new CUDA_EMULATOR();
    return singleton;
}

CUDA_EMULATOR::CUDA_EMULATOR()
{
    this->root = 0;
	this->device = "compute_20";
	this->trace = false;
}

extern pANTLR3_BASE_TREE parse(char * source);

void CUDA_EMULATOR::Extract_From_Source(char * module_name, char * source)
{
    // Pick modules of only one type.
    if (strstr(module_name, this->device) == 0)
        return;
    pANTLR3_BASE_TREE mod = parse(source);
    if (! mod)
    {
        std::cout << "Error: cannot parse PTX!\n";
        assert(false);
    }
    modules.push_back(mod);
    Extract_From_Tree(mod);
}

void CUDA_EMULATOR::Extract_From_Tree(pANTLR3_BASE_TREE node)
{
    // Traverse the tree and look for key features like entry, func, variable declarations, etc.
    if (node->getType(node) == TREE_ENTRY)
    {
        // First child will be name node.
        pANTLR3_BASE_TREE word = (pANTLR3_BASE_TREE)node->getChild(node, 0);
        char * name = (char*)word->getText(word)->chars;
        printf("entry %s\n", name);
        std::pair<char*, pANTLR3_BASE_TREE> i;
        i.first = (char*)name;
        i.second = node;
        this->entry.insert(i);
    }
    else if (node->getType(node) == TREE_FUNC)
    {
        pANTLR3_BASE_TREE word = (pANTLR3_BASE_TREE)node->getChild(node, 0);
        char * name = (char*)word->getText(word)->chars;
        printf("func %s\n", name);
        std::pair<char*, pANTLR3_BASE_TREE> i;
        i.first = (char*)name;
        i.second = node;
        this->func.insert(i);
    }
    for (int i = 0; i < (int)node->getChildCount(node); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)node->getChild(node, i);
        Extract_From_Tree(child);
    }
} 

void ** CUDA_EMULATOR::RegisterFunction(void * fun, char * name)
{
    std::pair<void*, char*> i;
    i.first = fun;
    i.second = name;
    this->fun_to_name.insert(i);
    return 0;
}

cudaError_t CUDA_EMULATOR::SetupArgument(const void *arg, size_t size, size_t offset)
{
    // record argument, size, offset.
    CUDA_EMULATOR::arg * a = new CUDA_EMULATOR::arg();
    assert(size == 4);
    a->argument = malloc(size);
    memcpy(const_cast<void*>(a->argument), arg, size);
    a->size = size;
    a->offset = offset;
    this->arguments.push_back(a);
    return cudaSuccess;
}

cudaError_t CUDA_EMULATOR::GetDevice(int * device)
{
    *device = 0;
    return cudaSuccess;
}

cudaError_t CUDA_EMULATOR::GetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    static cudaDeviceProp p = {
        "emulator",                     // char name[256];
        3000000000,                     // size_t totalGlobalMem;
        3000000,                        // size_t sharedMemPerBlock;
        500,                            // int regsPerBlock;
        500,                            // int warpSize;
        10,                             // size_t memPitch;
        16000,                          // int maxThreadsPerBlock;
        {1000,1,1},                     // int maxThreadsDim[3];
        {1000,1,1},                     // int maxGridSize[3];
        100000000,                      // size_t totalConstMem;
        1,                              // int major;
        4,                              // int minor;
        111,                            // int clockRate;
        11,                             // size_t textureAlignment;
        11,                             // int deviceOverlap;
        11,                             // int multiProcessorCount;
        1,                              // int kernelExecTimeoutEnabled;
        1,                              // int integrated;
        1,                              // int canMapHostMemory;
        1                               // int computeMode;
    };
    *prop = p;
    return cudaSuccess;
}


void CUDA_EMULATOR::BindArguments(pANTLR3_BASE_TREE e)
{
    // Create a new symbol table block for the parameters.
    SymbolTable * symbol_table = new SymbolTable();
    symbol_table->parent_block_symbol_table = this->root;
    this->root = symbol_table;
    // For each parameter, create a symbol table entry, bind the arguments.
    // To do this, walk down both the AST and the setup argument list, and
    // associate each entry with the other in a symbol table entry.
    int argc = this->arguments.size();
    pANTLR3_BASE_TREE param_list = (pANTLR3_BASE_TREE)e->getChild(e, 1);
    std::list<arg*>::iterator ia = this->arguments.begin();
    for (int i = 0; i < argc; ++i, ++ia)
    {
        // Get to the parameter in the AST.
        pANTLR3_BASE_TREE param = GetChild(param_list, i);
        pANTLR3_BASE_TREE name = GetChild(param, 0);
        char * n = GetText(name);
        pANTLR3_BASE_TREE type = GetChild(GetChild(param, 1), 0);
        char * t = GetText(type);
        // Get to the argument in the set up list.
        arg * a = *ia;
        // Create a symbol table entry.
        Symbol * s = new Symbol();
        s->lvalue = (void*)a->argument;
        s->name = n;
        s->size = a->size;
        s->type = t;
        // Add the entry into the symbol table.
        std::pair<char*, Symbol*> sym;
        sym.first = n;
        sym.second = s;
        symbol_table->symbols.insert(sym);
    }
}

size_t CUDA_EMULATOR::Sizeof(int type)
{
    switch (type)
    {
    case K_S8: return sizeof(char);
    case K_S16: return sizeof(short);
    case K_S32: return sizeof(int);
    case K_S64: return sizeof(long);
    case K_U8: return sizeof(unsigned char);
    case K_U16: return sizeof(unsigned short);
    case K_U32: return sizeof(unsigned int);
    case K_U64: return sizeof(unsigned long);
    case K_F16: return sizeof(float);
    case K_F32: return sizeof(float);
    case K_F64: return sizeof(double);
    case K_B8: return sizeof(char);
    case K_B16: return sizeof(short);
    case K_B32: return sizeof(int);
    case K_B64: return sizeof(long);
    case K_PRED: return sizeof(bool);
    }
    assert(false);
    return 0;
}

int CUDA_EMULATOR::GetType(pANTLR3_BASE_TREE c)
{
    return (int)c->getType(c);
}

int CUDA_EMULATOR::GetSize(pANTLR3_BASE_TREE tree_par_register)
{
    pANTLR3_BASE_TREE c = (pANTLR3_BASE_TREE)tree_par_register->getChild(tree_par_register,0);
    return (int)atoi((char*)c->getText(c)->chars);
}

void CUDA_EMULATOR::SetupLocals(pANTLR3_BASE_TREE block)
{
    // Create a new symbol table block for the locals.
    SymbolTable * symbol_table = new SymbolTable();
    symbol_table->parent_block_symbol_table = this->root;
    this->root = symbol_table;
    // Go through the block and create entries in the symbol table for each variable declared.
    for (int i = 0; i < (int)block->getChildCount(block); ++i)
    {
        pANTLR3_BASE_TREE var = (pANTLR3_BASE_TREE)block->getChild(block, i);
        if (var->getType(var) == TREE_VAR)
        {
            // Got variable declaration.
            // Now extract info out of variable declaration.
            char * name = 0;
            int nreg = 0;
            char * type = 0;
            int size = 0;
            for (int j = 0; j < (int)var->getChildCount(var); ++j)
            {
                pANTLR3_BASE_TREE c = (pANTLR3_BASE_TREE)var->getChild(var, j);
                if (c->getType(c) == TREE_SPACE) {
                    // Nothing to do.
                } else if (c->getType(c) == TREE_ALIGN) {
                    // Nothing to do.
                } else if (c->getType(c) == TREE_TYPE) {
                    pANTLR3_BASE_TREE chi = GetChild(c, 0);
                    type = GetText(chi);
                    int t = GetType(chi);
                    size = Sizeof(t);
                } else if (c->getType(c) == T_WORD) {
                    name = GetText(c);
                } else if (c->getType(c) == TREE_PAR_REGISTER) {
                    nreg = GetSize(c);
                } else assert(false);
            }
            // Convert raw info into symbol declarations and sizes.
            if (nreg > 0)
            {
                for (int k = 0; k < nreg; ++k)
                {
                    char full_name[1000];
                    sprintf(full_name, "%s%d", name, k+1);
                    // Create a symbol table entry.
                    Symbol * s = new Symbol();
                    s->name = strdup(full_name);
                    s->size = size;
                    s->lvalue = (void*)malloc(size);
                    s->type = strdup(type);
                    // Add the entry into the symbol table.
                    std::pair<char*, Symbol*> sym;
                    sym.first = s->name;
                    sym.second = s;
                    symbol_table->symbols.insert(sym);
                }
            } else {
                // Create a symbol table entry.
                Symbol * s = new Symbol();
                s->name = strdup(name);
                s->size = size;
                s->lvalue = (void*)malloc(size);
                s->type = strdup(type);
                // Add the entry into the symbol table.
                std::pair<char*, Symbol*> sym;
                sym.first = s->name;
                sym.second = s;
                symbol_table->symbols.insert(sym);
            }
        }
    }
}

cudaError_t CUDA_EMULATOR::ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    conf.gridDim = gridDim;
    conf.blockDim = blockDim;
    conf.sharedMem = sharedMem;
    conf.stream = stream;
    return cudaSuccess;
}


cudaError_t CUDA_EMULATOR::ThreadSynchronize()
{
    return cudaSuccess;
}


void CUDA_EMULATOR::SetupGotos(pANTLR3_BASE_TREE block)
{
    // Scan ahead and find all labels.  Enter them into the symbol
    // table.
    for (int i = 0; i < (int)block->getChildCount(block); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)block->getChild(block, i);
        if (child->getType(child) == TREE_LABEL)
        {
            pANTLR3_BASE_TREE label = (pANTLR3_BASE_TREE)block->getChild(child, 0);
            char * name = GetText(label);
            Symbol * s = new Symbol();
            s->name = strdup(name);
            s->type = "label";
            s->size = 0;
            s->lvalue = (void*)i;
            SymbolTable * symbol_table = this->root;
            // Add the entry into the symbol table.
            std::pair<char*, Symbol*> sym;
            sym.first = s->name;
            sym.second = s;
            symbol_table->symbols.insert(sym);
        }
    }
}

void CUDA_EMULATOR::Execute(void* hostfun)
{
    // Given the address of the kernel function in the host, determine the name of the kernel
    // it is calling in PTX, using information provided by RegisterFatBinary and RegisterFunction.
    std::map<void*, char*>::iterator i = this->fun_to_name.find(hostfun);
    assert(i != this->fun_to_name.end());
    char * name = i->second;

    // Now, given the name of the kernel function being called, find
    // the entry for it.
    std::map<char*, pANTLR3_BASE_TREE, ltstr>::iterator j = this->entry.find(name);
    assert(j != this->entry.end());

    pANTLR3_BASE_TREE entry = j->second;

    // Get function block.
    pANTLR3_BASE_TREE block = FindBlock(entry);

    // Set up the symbol table for the function, and bind the arguments to the parameters declared.
    BindArguments(entry);

    // Set up local variables.
    SetupLocals(block);

    SetupGotos(block);

    // Set up dim3 bounds.
    SetupDimensionLocals();

    for (int bidx = 0; bidx < conf.gridDim.x; ++bidx)
    {
        for (int bidy = 0; bidy < conf.gridDim.y; ++bidy)
        {
            for (int bidz = 0; bidz < conf.gridDim.z; ++bidz)
            {
                for (int tidx = 0; tidx < conf.blockDim.x; ++tidx)
                {
                    for (int tidy = 0; tidy < conf.blockDim.y; ++tidy)
                    {
                        for (int tidz = 0; tidz < conf.blockDim.z; ++tidz)
                        {
                            dim3 tid(tidx, tidy, tidz);
                            dim3 bid(bidx, bidy, bidz);

                            // Set up dim3 indices.
                            SetupPredefined(tid, bid);

                            // Execute.
                            int pc = FindFirstInst(block, 0);
                            if (pc < 0)
                                return;
                            for (;;)
                            {
                                pANTLR3_BASE_TREE inst = GetInst(block, pc);

                                //Dump("before", pc, inst);

                                int next = Dispatch(inst);
                                if (next > 0)
                                    pc = next;
                                else if (next < 0)
                                    break;
                                else
                                    pc++;
                                pc = FindFirstInst(block, pc);

                                //Dump("after", pc, inst);
                            }
                        }
                    }
                }
            }
        }
    }
}

void CUDA_EMULATOR::Print(pANTLR3_BASE_TREE node, int level)
{
    for (int i = 0; i < level; ++i)
        std::cout << "   ";
    std::cout << GetText(node) << "\n";
    for (int i = 0; i < (int)node->getChildCount(node); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)node->getChild(node, i);
        Print(child, level+1);
    }
} 

void CUDA_EMULATOR::Dump(char * comment, int pc, pANTLR3_BASE_TREE inst)
{
    std::cout << "\n";
    std::cout << comment << "\n";
    std::cout << "PC = " << pc << "\n";
	Print(inst, 0);
    std::cout << "Symbol tables:\n";
    for (SymbolTable * st = root; st != 0; st = st->parent_block_symbol_table)
    {
        std::map<char*, Symbol*, ltstr>::iterator it;
        for (it = st->symbols.begin(); it != st->symbols.end(); ++it)
        {
            Symbol * s = (*it).second;
            std::cout << "name: " << s->name << " ";
            std::cout << "size: " << s->size << " ";
            std::cout << "type: " << s->type << " ";
            if (strcmp(s->type, "label") == 0)
                std::cout << "val:  " << (int)s->lvalue << "\n";
            else if (strcmp(s->type, "dim3") == 0)
                std::cout << "val:  " << ((dim3*)s->lvalue)->x
                << " " << ((dim3*)s->lvalue)->y
                << " " << ((dim3*)s->lvalue)->z
                << "\n";
            else if (strcmp(s->type, ".pred") == 0)
                std::cout << "val:  " << *((bool*)s->lvalue) << "\n";
            else if (strcmp(s->type, ".u8") == 0)
                std::cout << "val:  " << *(unsigned char*)s->lvalue << "\n";
            else if (strcmp(s->type, ".u16") == 0)
                std::cout << "val:  " << *(unsigned short*)s->lvalue << "\n";
            else if (strcmp(s->type, ".u32") == 0)
                std::cout << "val:  " << *(unsigned int*)s->lvalue << "\n";
            else if (strcmp(s->type, ".s8") == 0)
                std::cout << "val:  " << *(signed char*)s->lvalue << "\n";
            else if (strcmp(s->type, ".s16") == 0)
                std::cout << "val:  " << *(signed short*)s->lvalue << "\n";
            else if (strcmp(s->type, ".s32") == 0)
                std::cout << "val:  " << *(signed int*)s->lvalue << "\n";
            else if (strcmp(s->type, ".f32") == 0)
                std::cout << "val:  " << *(float*)s->lvalue << "\n";
            else if (strcmp(s->type, ".f64") == 0)
                std::cout << "val:  " << *(double*)s->lvalue << "\n";
            else assert(false);
        }
    }
}


CUDA_EMULATOR::Symbol * CUDA_EMULATOR::FindSymbol(char * name)
{
    SymbolTable * st = this->root;
    while (st)
    {
        std::map<char*, Symbol*, ltstr>::iterator it = st->symbols.find(name);
        if (it != st->symbols.end())
        {
            return it->second;
        }
        st = st->parent_block_symbol_table;
    }
    return 0;
}

void CUDA_EMULATOR::CreateSymbol(char * name, char * type, void * value, size_t size)
{
    // First find it.
    Symbol * s = FindSymbol(name);
    if (s)
    {
        // Update value.
        memcpy(s->lvalue, value, size);
        return;
    }
    // Create a symbol table entry.
    s = new Symbol();
    s->name = strdup(name);
    s->type = strdup(type);
    s->size = size;
    s->lvalue = (void*)malloc(size);
    memcpy(s->lvalue, value, size);
    // Add the entry into the symbol table.
    std::pair<char*, Symbol*> sym;
    sym.first = s->name;
    sym.second = s;
    SymbolTable * symbol_table = root;
    symbol_table->symbols.insert(sym);
}

void CUDA_EMULATOR::SetupDimensionLocals()
{
    // Create gridDim = %nctaid, and blockDim = %ntid
    CreateSymbol("%nctaid", "dim3", &conf.gridDim, sizeof(conf.gridDim));
    CreateSymbol("%ntid", "dim3", &conf.blockDim, sizeof(conf.blockDim));
}


void CUDA_EMULATOR::SetupPredefined(dim3 tid, dim3 bid)
{
    // Create threadIdx
    CreateSymbol("%tid", "dim3", &tid, sizeof(tid));
    CreateSymbol("%ctaid", "dim3", &bid, sizeof(tid));
}


pANTLR3_BASE_TREE CUDA_EMULATOR::FindBlock(pANTLR3_BASE_TREE node)
{
    for (int i = 0; i < (int)node->getChildCount(node); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)node->getChild(node, i);
        if (child->getType(child) == TREE_BLOCK)
            return child;
    }
    return 0;
}

int CUDA_EMULATOR::FindFirstInst(pANTLR3_BASE_TREE block, int first)
{
    for (int i = first; i < (int)block->getChildCount(block); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)block->getChild(block, i);
        if (child->getType(child) == TREE_INST)
            return i;
    }
    return -1;
}

pANTLR3_BASE_TREE CUDA_EMULATOR::GetInst(pANTLR3_BASE_TREE block, int pc)
{
    assert(block->getType(block) == TREE_BLOCK);
    pANTLR3_BASE_TREE inst = (pANTLR3_BASE_TREE)block->getChild(block, pc);
    return inst;
}


pANTLR3_BASE_TREE CUDA_EMULATOR::GetChild(pANTLR3_BASE_TREE node, int n)
{
    pANTLR3_BASE_TREE c = (pANTLR3_BASE_TREE)node->getChild(node, n);
    return c;
}

char * CUDA_EMULATOR::GetText(pANTLR3_BASE_TREE node)
{
    return (char*)node->getText(node)->chars;
}

int CUDA_EMULATOR::Dispatch(pANTLR3_BASE_TREE inst)
{
	if (this->trace)
	{
		Print(inst, 0);
	}
	
    pANTLR3_BASE_TREE i = (pANTLR3_BASE_TREE)inst->getChild(inst, 0);
    int inst_type = i->getType(i);
    if (inst_type == TREE_PRED)
    {
        // Predicate preceeds the instruction.
        pANTLR3_BASE_TREE pred = i;
        i = (pANTLR3_BASE_TREE)inst->getChild(inst, 1);
        inst_type = i->getType(i);
        // Check if pred is true.  If false, ignore instruction with this predicate.
        pANTLR3_BASE_TREE psym = (pANTLR3_BASE_TREE)inst->getChild(pred, 0);
        assert(GetType(psym) == T_WORD);
        Symbol * s = FindSymbol(GetText(psym));
        assert(s != 0);
        if (! *((bool*)s->lvalue))
        {
            std::cout << "Skipping " << GetText(i) << " because guard predicate is false\n";
            return 0; // continue.
        }
    }
    switch (inst_type)
    {
        case KI_ABS:
            break;
        case KI_ADD:
            DoAdd(inst);
            return 0; // continue.
        case KI_ADDC:
            break;
        case KI_AND:
            break;
        case KI_ATOM:
            break;
        case KI_BAR:
            break;
        case KI_BFE:
            break;
        case KI_BFI:
            break;
        case KI_BFIND:
            break;
        case KI_BRA:
            return DoBra(inst);
        case KI_BREV:
            break;
        case KI_BRKPT:
            break;
        case KI_CALL:
            break;
        case KI_CLZ:
            break;
        case KI_CNOT:
            break;
        case KI_COPYSIGN:
            break;
        case KI_COS:
            break;
        case KI_CVT:
            DoCvt(inst);
            return 0;
        case KI_CVTA:
            break;
        case KI_DIV:
            DoDiv(inst);
            return 0;
        case KI_EX2:
            break;
        case KI_EXIT:
            DoExit(inst);
            return -1; // end.
        case KI_FMA:
			DoFma(inst);
			return 0;
        case KI_ISSPACEP:
            break;
        case KI_LD:
            DoLd(inst);
            return 0; // continue.
        case KI_LDU:
            break;
        case KI_LG2:
            break;
        case KI_MAD24:
            break;
        case KI_MAD:
            break;
        case KI_MAX:
            break;
        case KI_MEMBAR:
            break;
        case KI_MIN:
            break;
        case KI_MOV:
            DoMov(inst);
            return 0; // continue.
        case KI_MUL24:
            break;
        case KI_MUL:
            DoMul(inst);
            return 0; // continue.
        case KI_NEG:
            break;
        case KI_NOT:
            break;
        case KI_OR:
            break;
        case KI_PMEVENT:
            break;
        case KI_POPC:
            break;
        case KI_PREFETCH:
            break;
        case KI_PREFETCHU:
            break;
        case KI_PRMT:
            break;
        case KI_RCP:
            break;
        case KI_RED:
            break;
        case KI_REM:
            break;
        case KI_RET:
            break;
        case KI_RSQRT:
            break;
        case KI_SAD:
            break;
        case KI_SELP:
            break;
        case KI_SET:
            break;
        case KI_SETP:
            DoSetp(inst);
            return 0; // continue.
        case KI_SHL:
            break;
        case KI_SHR:
            break;
        case KI_SIN:
            break;
        case KI_SLCT:
            break;
        case KI_SQRT:
            break;
        case KI_ST:
            DoSt(inst);
            return 0; // continue.
        case KI_SUB:
            break;
        case KI_SUBC:
            break;
        case KI_SULD:
            break;
        case KI_SUQ:
            break;
        case KI_SURED:
            break;
        case KI_SUST:
            break;
        case KI_TESTP:
            break;
        case KI_TEX:
            break;
        case KI_TRAP:
            break;
        case KI_TXQ:
            break;
        case KI_VABSDIFF:
            break;
        case KI_VADD:
            break;
        case KI_VMAD:
            break;
        case KI_VMAX:
            break;
        case KI_VMIN:
            break;
        case KI_VOTE:
            break;
        case KI_VSET:
            break;
        case KI_VSHL:
            break;
        case KI_VSHR:
            break;
        case KI_VSUB:
            break;
        case KI_XOR:
            break;
        default:
            break;
    }
    assert(false); // unimplemented instruction.
    return -1; // end.
}

void CUDA_EMULATOR::SetDevice(char * device)
{
    this->device = strdup(device);
}

CUDA_EMULATOR::Constant CUDA_EMULATOR::Eval(int expected_type, pANTLR3_BASE_TREE const_expr)
{
    Constant result;
	result.type = expected_type;
	char * dummy;
	char * text = GetText(const_expr);
    if (GetType(const_expr) == T_DEC_LITERAL)
    {
		switch (expected_type)
		{
			case K_U8:
				result.value.u8 = _strtoi64(text, &dummy, 10);
				break;
			case K_U16:
				result.value.u16 = _strtoi64(text, &dummy, 10);
				break;
			case K_U32:
				result.value.u32 = _strtoi64(text, &dummy, 10);
				break;
			case K_U64:
				result.value.u64 = _strtoi64(text, &dummy, 10);
				break;
			case K_S8:
				result.value.u8 = _strtoi64(text, &dummy, 10);
				break;
			case K_S16:
				result.value.s16 = _strtoi64(text, &dummy, 10);
				break;
			case K_S32:
				result.value.s32 = _strtoi64(text, &dummy, 10);
				break;
			case K_S64:
				result.value.s64 = _strtoi64(text, &dummy, 10);
				break;
			case K_B8:
				result.value.b8 = _strtoi64(text, &dummy, 10);
				break;
			case K_B16:
				result.value.b16 = _strtoi64(text, &dummy, 10);
				break;
			case K_B32:
				result.value.b32 = _strtoi64(text, &dummy, 10);
				break;
			case K_B64:
				result.value.b64 = _strtoi64(text, &dummy, 10);
				break;
			default:
				assert(false);
		}
    } else if (GetType(const_expr) == T_HEX_LITERAL)
    {
		text += 2;
		switch (expected_type)
		{
			case K_U8:
				result.value.u8 = _strtoi64(text, &dummy, 16);
				break;
			case K_U16:
				result.value.u16 = _strtoi64(text, &dummy, 16);
				break;
			case K_U32:
				result.value.u32 = _strtoi64(text, &dummy, 16);
				break;
			case K_U64:
				result.value.u64 = _strtoi64(text, &dummy, 16);
				break;
			case K_S8:
				result.value.u8 = _strtoi64(text, &dummy, 16);
				break;
			case K_S16:
				result.value.s16 = _strtoi64(text, &dummy, 16);
				break;
			case K_S32:
				result.value.s32 = _strtoi64(text, &dummy, 16);
				break;
			case K_S64:
				result.value.s64 = _strtoi64(text, &dummy, 16);
				break;
			case K_B8:
				result.value.b8 = _strtoi64(text, &dummy, 16);
				break;
			case K_B16:
				result.value.b16 = _strtoi64(text, &dummy, 16);
				break;
			case K_B32:
				result.value.b32 = _strtoi64(text, &dummy, 16);
				break;
			case K_B64:
				result.value.b64 = _strtoi64(text, &dummy, 16);
				break;
			default:
				assert(false);
		}
    } else if (GetType(const_expr) == T_FLT_LITERAL)
    {
		text += 2;
		switch (expected_type)
		{
			case K_F32:
				result.value.u32 = _strtoi64(text, &dummy, 16);
				break;
			case K_F64:
				result.value.u64 = _strtoi64(text, &dummy, 16);
				break;
			default:
				assert(false);
		}
    } else assert(false);
    return result;
}
