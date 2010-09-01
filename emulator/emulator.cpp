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
    this->device = "";
}

extern pANTLR3_BASE_TREE parse(char * source);

void CUDA_EMULATOR::Extract_From_Source(char * module_name, char * source)
{
    // Pick only one.
    if (modules.size() > 0)
        return;
    if (strcmp(this->device, "") != 0 && strcmp(module_name, this->device) != 0)
        return;
    pANTLR3_BASE_TREE mod = parse(source);
    if (! mod)
        return;
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
    if (i == this->fun_to_name.end())
        return;
    char * name = i->second;
    // Now, given the name of the kernel function being called, find the entry for it.
    std::map<char*, pANTLR3_BASE_TREE, ltstr>::iterator j = this->entry.find(name);
    if (j == this->entry.end())
        return;
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

                                Dump("after", pc, inst);
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
            return 0; // continue.
    }
    switch (inst_type)
    {
        case KI_ABS: ;
        case KI_ADD:
            DoAdd(inst);
            return 0; // continue.
        case KI_ADDC: ;
        case KI_AND: ;
        case KI_ATOM: ;
        case KI_BAR: ;
        case KI_BFE: ;
        case KI_BFI: ;
        case KI_BFIND: ;
        case KI_BRA:
            return DoBra(inst);
        case KI_BREV: ;
        case KI_BRKPT: ;
        case KI_CALL: ;
        case KI_CLZ: ;
        case KI_CNOT: ;
        case KI_COPYSIGN: ;
        case KI_COS: ;
        case KI_CVT:
            DoCvt(inst);
            return 0;
        case KI_CVTA: ;
        case KI_DIV: ;
        case KI_EX2: ;
        case KI_EXIT:
            DoExit(inst);
            return -1; // end.
        case KI_FMA: ;
        case KI_ISSPACEP: ;
        case KI_LD:
            DoLd(inst);
            return 0; // continue.
        case KI_LDU: ;
        case KI_LG2: ;
        case KI_MAD24: ;
        case KI_MAD: ;
        case KI_MAX: ;
        case KI_MEMBAR: ;
        case KI_MIN: ;
        case KI_MOV:
            DoMov(inst);
            return 0; // continue.
        case KI_MUL24: ;
        case KI_MUL:
            DoMul(inst);
            return 0; // continue.
        case KI_NEG: ;
        case KI_NOT: ;
        case KI_OR: ;
        case KI_PMEVENT: ;
        case KI_POPC: ;
        case KI_PREFETCH: ;
        case KI_PREFETCHU: ;
        case KI_PRMT: ;
        case KI_RCP: ;
        case KI_RED: ;
        case KI_REM: ;
        case KI_RET: ;
        case KI_RSQRT: ;
        case KI_SAD: ;
        case KI_SELP: ;
        case KI_SET: ;
        case KI_SETP:
            DoSetp(inst);
            return 0; // continue.
        case KI_SHL: ;
        case KI_SHR: ;
        case KI_SIN: ;
        case KI_SLCT: ;
        case KI_SQRT: ;
        case KI_ST:
            DoSt(inst);
            return 0; // continue.
        case KI_SUB: ;
        case KI_SUBC: ;
        case KI_SULD: ;
        case KI_SUQ: ;
        case KI_SURED: ;
        case KI_SUST: ;
        case KI_TESTP: ;
        case KI_TEX: ;
        case KI_TRAP: ;
        case KI_TXQ: ;
        case KI_VABSDIFF: ;
        case KI_VADD: ;
        case KI_VMAD: ;
        case KI_VMAX: ;
        case KI_VMIN: ;
        case KI_VOTE: ;
        case KI_VSET: ;
        case KI_VSHL: ;
        case KI_VSHR: ;
        case KI_VSUB: ;
        case KI_XOR: ;
        default: ;
    }
    return -1; // end.
}

void CUDA_EMULATOR::DoAdd(pANTLR3_BASE_TREE inst)
{
    std::cout << "ADD\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    pANTLR3_BASE_TREE ttype = GetChild(GetChild(inst, start+1), 0);
    pANTLR3_BASE_TREE ttype2 = 0;
    if (GetType(ttype) == K_SAT)
    {
        ttype2 = ttype;
        ttype = GetChild(GetChild(inst, start+1), 1);
    }
    int type = GetType(ttype);
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc1 = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE osrc2 = GetChild(inst, start+4);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);
    Symbol * sdst = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    union {
        long ls;
        int is;
        short ss;
        unsigned long lu;
        unsigned int iu;
        unsigned short su;
    } value1;
    union {
        long ls;
        int is;
        short ss;
        unsigned long lu;
        unsigned int iu;
        unsigned short su;
    } value2;
    if (GetType(src1) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            {
                value1.su = atoi(GetText(src1));
                break;
            }
            case K_S16:
            {
                value1.ss = atoi(GetText(src1));
                break;
            }
            case K_U32:
            {
                value1.iu = atoi(GetText(src1));
                break;
            }
            case K_S32:
            {
                value1.is = atoi(GetText(src1));
                break;
            }
            case K_S64:
            {
                value1.ls = atoi(GetText(src1));
                break;
            }
            case K_U64:
            {
                value1.lu = atoi(GetText(src1));
                break;
            }
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            {
                sscanf(GetText(src1), "%hx", &value1.su);
                break;
            }
            case K_S16:
            {
                sscanf(GetText(src1), "%hx", &value1.ss);
                break;
            }
            case K_U32:
            {
                sscanf(GetText(src1), "%x", &value1.iu);
                break;
            }
            case K_S32:
            {
                sscanf(GetText(src1), "%x", &value1.is);
                break;
            }
            case K_S64:
            {
                sscanf(GetText(src1), "%lx", &value1.ls);
                break;
            }
            case K_U64:
            {
                sscanf(GetText(src1), "%lx", &value1.lu);
                break;
            }
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_WORD)
    {
        Symbol * ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        switch (ssrc1->size)
        {
            case 2:
            {
                value1.ss = *(short int*)ssrc1->lvalue;
                break;
            }
            case 4:
            {
                value1.is = *(int*)ssrc1->lvalue;
                break;
            }
            case 8:
            {
                value1.ls = *(long*)ssrc1->lvalue;
                break;
            }
            default:
                assert(false);
        }
    } else assert(false);
    if (GetType(src2) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            {
                value2.su = atoi(GetText(src2));
                break;
            }
            case K_S16:
            {
                value2.ss = atoi(GetText(src2));
                break;
            }
            case K_U32:
            {
                value2.iu = atoi(GetText(src2));
                break;
            }
            case K_S32:
            {
                value2.is = atoi(GetText(src2));
                break;
            }
            case K_S64:
            {
                value2.ls = atoi(GetText(src2));
                break;
            }
            case K_U64:
            {
                value2.lu = atoi(GetText(src2));
                break;
            }
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            {
                sscanf(GetText(src2), "%hx", &value2.su);
                break;
            }
            case K_S16:
            {
                sscanf(GetText(src2), "%hx", &value2.ss);
                break;
            }
            case K_U32:
            {
                sscanf(GetText(src2), "%x", &value2.iu);
                break;
            }
            case K_S32:
            {
                sscanf(GetText(src2), "%x", &value2.is);
                break;
            }
            case K_S64:
            {
                sscanf(GetText(src2), "%lx", &value2.ls);
                break;
            }
            case K_U64:
            {
                sscanf(GetText(src2), "%lx", &value2.lu);
                break;
            }
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_WORD)
    {
        Symbol * ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        switch (ssrc2->size)
        {
            case 2:
            {
                value2.ss = *(short int*)ssrc2->lvalue;
                break;
            }
            case 4:
            {
                value2.is = *(int*)ssrc2->lvalue;
                break;
            }
            case 8:
            {
                value2.ls = *(long*)ssrc2->lvalue;
                break;
            }
            default:
                assert(false);
        }
    } else assert(false);

    switch (type)
    {
        case K_U16:
        {
            *(unsigned short*)sdst->lvalue = value1.su + value2.su;
            break;
        }
        case K_S16:
        {
            *(short*)sdst->lvalue = value1.ss + value2.ss;
            break;
        }
        case K_U32:
        {
            *(unsigned int*)sdst->lvalue = value1.iu + value2.iu;
            break;
        }
        case K_S32:
        {
            *(short*)sdst->lvalue = value1.is + value2.is;
            break;
        }
        case K_S64:
        {
            *(long*)sdst->lvalue = value1.ls + value2.ls;
            break;
        }
        case K_U64:
        {
            *(unsigned long*)sdst->lvalue = value1.lu + value2.lu;
            break;
        }
        default:
            assert(false);
    }
}

int CUDA_EMULATOR::DoBra(pANTLR3_BASE_TREE inst)
{
    std::cout << "BRA\n";
    int start = 0;
    for (;;)
    {
        pANTLR3_BASE_TREE t = GetChild(inst, start);
        assert(t != 0);
        if (GetType(t) == TREE_OPR)
            break;
        start++;
    }
    pANTLR3_BASE_TREE opr = GetChild(inst, start);
    pANTLR3_BASE_TREE dst = GetChild(opr, 0);
    assert(GetType(dst) == T_WORD);
    Symbol * sdst = FindSymbol(GetText(dst));
    assert (sdst != 0);
    return (int)sdst->lvalue;
}

void CUDA_EMULATOR::DoCvt(pANTLR3_BASE_TREE inst)
{
    std::cout << "CVT\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    assert(GetType(GetChild(inst, start)) == KI_CVT);
    start++;
    assert(GetType(GetChild(inst, start)) == TREE_TYPE);
    pANTLR3_BASE_TREE type = GetChild(inst, start);
    start++;
    bool ftz = false;
    bool sat = false;
    int src_type = 0;
    int dst_type = 0;
    int i = 0;
    for (;; ++i)
    {
        pANTLR3_BASE_TREE t = GetChild(type, i);
        if (t == 0)
            break;
        int gt = GetType(t);
        if (gt == TREE_IRND)
            assert(false);
        else if (gt == TREE_FRND)
            assert(false);
        else if (gt == K_FTZ)
            ftz = true;
        else if (gt == K_SAT)
            sat = true;
        else {
            if (dst_type == 0)
                dst_type = gt;
            else
                src_type = gt;
        }
    }
    assert(ftz == false); // unimplemented.
    assert(sat == false); // unimplemented.
    typedef union TYPES {
        long s64;
        int s32;
        short s16;
        signed char s8;
        unsigned long u64;
        unsigned int u32;
        unsigned short u16;
        unsigned char u8;
        float f32;
        double f64;
    } TYPES;
    TYPES * dst_value;
    TYPES * src_value;

    pANTLR3_BASE_TREE o1 = GetChild(inst, start++);
    assert(GetType(o1) == TREE_OPR);
    assert(GetType(GetChild(o1, 0)) == T_WORD);
    pANTLR3_BASE_TREE o2 = GetChild(inst, start++);
    assert(GetType(o2) == TREE_OPR);
    assert(GetType(GetChild(o2, 0)) == T_WORD);

    Symbol * s1 = FindSymbol(GetText(GetChild(o1, 0)));
    assert(s1 != 0);
    Symbol * s2 = FindSymbol(GetText(GetChild(o2, 0)));
    assert(s2 != 0);

    dst_value = (TYPES*)s1->lvalue;

    // handle .x, .y, .z stuff.
    if (strcmp(s2->type, "dim3") == 0)
    {
        // Get qualifier of the structure.
        pANTLR3_BASE_TREE tqual = GetChild(o2, 1);
        assert(tqual != 0);
        int qual = GetType(tqual);
        if (qual == K_X)
        {
            src_value = (TYPES*) &(((dim3*)s2->lvalue)->x);
        } else if (qual == K_Y)
        {
            src_value = (TYPES*) &(((dim3*)s2->lvalue)->y);
        } else if (qual == K_Z)
        {
            src_value = (TYPES*) &(((dim3*)s2->lvalue)->z);
        } else assert(false);
	} else
		src_value = (TYPES*)s2->lvalue;

    switch (src_type)
    {
        case K_U8:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u8;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u8;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u8;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u8;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u8;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u8;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u8;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u8;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_U16:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u16;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u16;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u16;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u16;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u16;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u16;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u16;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u16;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_U32:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u32;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u32;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u32;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u32;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u32;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u32;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u32;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u32;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_U64:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->u64;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->u64;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->u64;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->u64;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->u64;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->u64;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->u64;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->u64;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_S8:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s8;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s8;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s8;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s8;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s8;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s8;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s8;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s8;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_S16:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s16;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s16;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s16;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s16;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s16;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s16;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s16;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s16;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_S32:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s32;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s32;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s32;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s32;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s32;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s32;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s32;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s32;
                    break;
                default:
                    assert(false);
            }
            break;
        case K_S64:
            switch (dst_type)
            {
                case K_U8:
                    dst_value->u8 = src_value->s64;
                    break;
                case K_U16:
                    dst_value->u16 = src_value->s64;
                    break;
                case K_U32:
                    dst_value->u32 = src_value->s64;
                    break;
                case K_U64:
                    dst_value->u64 = src_value->s64;
                    break;
                case K_S8:
                    dst_value->s8 = src_value->s64;
                    break;
                case K_S16:
                    dst_value->s16 = src_value->s64;
                    break;
                case K_S32:
                    dst_value->s32 = src_value->s64;
                    break;
                case K_S64:
                    dst_value->s64 = src_value->s64;
                    break;
                default:
                    assert(false);
            }
            break;
        default:
            assert(false);
    }
}

void CUDA_EMULATOR::DoExit(pANTLR3_BASE_TREE inst)
{
    std::cout << "EXIT\n";
}

void CUDA_EMULATOR::DoLd(pANTLR3_BASE_TREE inst)
{
    std::cout << "LD\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    // Get two operands, assign source to destination.
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE osrc = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src = GetChild(osrc, 0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    if (src->getType(src) == T_WORD)
    {
        ssrc = FindSymbol(GetText(src));
    } else assert(false);
    *((int*)sdst->lvalue) = *(int*)ssrc->lvalue;
}

void CUDA_EMULATOR::DoMov(pANTLR3_BASE_TREE inst)
{
    std::cout << "MOV\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    // Get two operands, assign source to destination.
    int type = GetType(GetChild(GetChild(inst, start+1), 0));
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst, 0);
    pANTLR3_BASE_TREE osrc = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src = GetChild(osrc,0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);

    // Many complications based on the type of arguments.
    if (GetType(src) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            case K_S16:
            case K_B16:
                {
                    short int value = atoi(GetText(src));
                    *((short int*)sdst->lvalue) = value;
                    break;
                }
            case K_U32:
            case K_S32:
            case K_B32:
                {
                    int value = atoi(GetText(src));
                    *((int*)sdst->lvalue) = value;
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            case K_S16:
            case K_B16:
                {
                    short int value;
                    sscanf(GetText(src), "%hx", &value);
                    *((short int*)sdst->lvalue) = value;
                    break;
                }
            case K_U32:
            case K_S32:
            case K_B32:
                {
                    int value;
                    sscanf(GetText(src), "%hx", &value);
                    *((int*)sdst->lvalue) = value;
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src) == T_FLT_LITERAL)
    {
        switch (type)
        {
            case K_F32:
                {
                    float value;
                    sscanf(GetText(src), "%f", &value);
                    *((float*)sdst->lvalue) = value;
                    break;
                }
            case K_F64:
                {
                    double value;
                    sscanf(GetText(src), "%lf", &value);
                    *((double*)sdst->lvalue) = value;
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src) == T_WORD)
    {
        ssrc = FindSymbol(GetText(src));
        // Various types of id's to handle:
        assert(ssrc != 0);
        if (strcmp(ssrc->type, "dim3") == 0)
        {
            // Get qualifier of the structure.
            pANTLR3_BASE_TREE tqual = (pANTLR3_BASE_TREE)osrc->getChild(osrc, 1);
            assert(tqual != 0);
            int qual = GetType(tqual);
            if (qual == K_X)
            {
                if (Sizeof(type) == 2)
                    *((short int*)sdst->lvalue) = ((dim3*)ssrc->lvalue)->x;
                else if (Sizeof(type) == 4)
                    *((int*)sdst->lvalue) = ((dim3*)ssrc->lvalue)->x;
            } else if (qual == K_Y)
            {
                if (Sizeof(type) == 2)
                    *((short int*)sdst->lvalue) = ((dim3*)ssrc->lvalue)->y;
                else if (Sizeof(type) == 4)
                    *((int*)sdst->lvalue) = ((dim3*)ssrc->lvalue)->y;
            } else if (qual == K_Y)
            {
                if (Sizeof(type) == 2)
                    *((short int*)sdst->lvalue) = ((dim3*)ssrc->lvalue)->z;
                else if (Sizeof(type) == 4)
                    *((int*)sdst->lvalue) = ((dim3*)ssrc->lvalue)->z;
            }
        } else {
            memcpy(sdst->lvalue, ssrc->lvalue, Sizeof(type));
        }
    } else assert(false);
}

void CUDA_EMULATOR::DoMul(pANTLR3_BASE_TREE inst)
{
    std::cout << "MUL\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    // Multiply register and/or constants, and store in a register.
    pANTLR3_BASE_TREE ttype = GetChild(GetChild(inst, start+1), 0);
    pANTLR3_BASE_TREE ttype2 = 0;
    if (GetType(ttype) == K_LO ||
          GetType(ttype) == K_HI ||
          GetType(ttype) == K_WIDE)
    {
        ttype2 = ttype;
        ttype = GetChild(GetChild(inst, start+1), 1);
    }
    int type = GetType(ttype);
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc1 = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE osrc2 = GetChild(inst, start+4);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);
    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    long s1, s2;
    if (GetType(src1) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            case K_S16:
            case K_B16:
            case K_U32:
            case K_S32:
            case K_B32:
                {
                    s1 = atoi(GetText(src1));
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            case K_S16:
            case K_B16:
            case K_U32:
            case K_S32:
            case K_B32:
                {
                    sscanf(GetText(src1), "%hx", &s1);
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
        assert(ssrc1 != 0);
        switch (ssrc1->size)
        {
            case 2:
                {
                    s1 = *(short int*)ssrc1->lvalue;
                    break;
                }
            case 4:
                {
                    s1 = *(int*)ssrc1->lvalue;
                    break;
                }
            case 8:
                {
                    s1 = *(long*)ssrc1->lvalue;
                    break;
                }
            default:
                assert(false);
        }
    } else assert(false);
    if (GetType(src2) == T_DEC_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            case K_S16:
            case K_B16:
            case K_U32:
            case K_S32:
            case K_B32:
                {
                    s2 = atoi(GetText(src2));
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_HEX_LITERAL)
    {
        switch (type)
        {
            case K_U16:
            case K_S16:
            case K_B16:
            case K_U32:
            case K_S32:
            case K_B32:
                {
                    sscanf(GetText(src2), "%hx", &s2);
                    break;
                }
            default:
                assert(false);
        }
    } else if (GetType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
        assert(ssrc2 != 0);
        switch (ssrc2->size)
        {
            case 2:
                {
                    s2 = *(short int*)ssrc2->lvalue;
                    break;
                }
            case 4:
                {
                    s2 = *(int*)ssrc2->lvalue;
                    break;
                }
            case 8:
                {
                    s2 = *(long*)ssrc2->lvalue;
                    break;
                }
            default:
                assert(false);
        }
    } else assert(false);
    long result_signed = 0;
    unsigned long result_unsigned = 0;
    switch (type)
    {
        case K_U16:
        case K_B16:
        case K_U32:
        case K_B32:
            {
                result_unsigned = ((unsigned long)s1) * ((unsigned long)s2);
                break;
            }
        case K_S16:
        case K_S32:
            {
                result_signed = ((long)s1) * ((long)s2);
                break;
            }
        default:
            assert(false);
    }

    if (sdst->size == 2)
    {
        if (ttype2 && GetType(ttype2) == K_LO)
        {
            *(unsigned short*)sdst->lvalue = (result_signed | result_unsigned);
        }
        else if (ttype2 && GetType(ttype2) == K_HI)
        {
            *(unsigned short*)sdst->lvalue = (result_signed | result_unsigned) >> 16;
        } else assert(false);
    } else if (sdst->size == 4)
    {
        if (ttype2 && GetType(ttype2) == K_LO)
        {
            *(unsigned int*)sdst->lvalue = (result_signed | result_unsigned);
        }
        else if (ttype2 && GetType(ttype2) == K_HI)
        {
            *(unsigned int*)sdst->lvalue = (result_signed | result_unsigned) >> 16;
        }
        else if (ttype2 && GetType(ttype2) == K_WIDE)
        {
            *(unsigned int*)sdst->lvalue = (result_signed | result_unsigned);
        }
        else assert(false);
    } else assert(false);
}

void CUDA_EMULATOR::DoSetp(pANTLR3_BASE_TREE inst)
{
    std::cout << "SETP\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc1 = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src1 = GetChild(osrc1,0);
    pANTLR3_BASE_TREE osrc2 = GetChild(inst, start+4);
    pANTLR3_BASE_TREE src2 = GetChild(osrc2,0);
    Symbol * sdst = 0;
    Symbol * ssrc1 = 0;
    Symbol * ssrc2 = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    if (src1->getType(src1) == T_WORD)
    {
        ssrc1 = FindSymbol(GetText(src1));
    } else assert(false);
    if (src2->getType(src2) == T_WORD)
    {
        ssrc2 = FindSymbol(GetText(src2));
    } else assert(false);
    char * op = GetText(GetChild(GetChild(inst, start+1), 0));
    if (strcmp(op, ".eq") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue == *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".gt") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue > *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".lt") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue < *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".ge") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue >= *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".le") == 0)
        *((bool*)sdst->lvalue) = *(int*)ssrc1->lvalue <= *(int*)ssrc2->lvalue;
    else if (strcmp(op, ".hi") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue > *(unsigned int*)ssrc2->lvalue;
    else if (strcmp(op, ".lo") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue < *(unsigned int*)ssrc2->lvalue;
    else if (strcmp(op, ".hs") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue >= *(unsigned int*)ssrc2->lvalue;
    else if (strcmp(op, ".ls") == 0)
        *((bool*)sdst->lvalue) = *(unsigned int*)ssrc1->lvalue <= *(unsigned int*)ssrc2->lvalue;
}

void CUDA_EMULATOR::DoSt(pANTLR3_BASE_TREE inst)
{
    std::cout << "ST\n";
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    pANTLR3_BASE_TREE odst = GetChild(inst, start+2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc = GetChild(inst, start+3);
    pANTLR3_BASE_TREE src = GetChild(osrc,0);
    Symbol * sdst = 0;
    Symbol * ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    if (src->getType(src) == T_WORD)
    {
        ssrc = FindSymbol(GetText(src));
    } else assert(false);
    **((int**)sdst->lvalue) = *(int*)ssrc->lvalue;
}


void CUDA_EMULATOR::SetDevice(char * device)
{
    this->device = strdup(device);
}
