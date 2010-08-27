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
}

extern pANTLR3_BASE_TREE parse(char * source);

void CUDA_EMULATOR::Extract_From_Source(char * source)
{
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
        pANTLR3_BASE_TREE param = (pANTLR3_BASE_TREE)param_list->getChild(param_list, i);
        pANTLR3_BASE_TREE name = (pANTLR3_BASE_TREE)param->getChild(param, 0);
        char * n = (char*)name->getText(name)->chars;
        // Get to the argument in the set up list.
        arg * a = *ia;
        // Create a symbol table entry.
        Symbol * s = new Symbol();
        s->lvalue = (void*)a->argument;
        s->name = n;
        s->size = a->size;
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
            char * name;
            int nreg = 0;
            int type;
            int size;
            for (int j = 0; j < (int)var->getChildCount(var); ++j)
            {
                pANTLR3_BASE_TREE c = (pANTLR3_BASE_TREE)var->getChild(var, j);
                if (c->getType(c) == TREE_SPACE) {
                    // Nothing to do.
                } else if (c->getType(c) == TREE_ALIGN) {
                    // Nothing to do.
                } else if (c->getType(c) == TREE_TYPE) {
                    pANTLR3_BASE_TREE chi = GetChild(c, 0);
                    type = GetType(chi);
                    size = Sizeof(type);
                } else if (c->getType(c) == T_WORD) {
                    name = (char*)c->getText(c)->chars;
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

    // Execute.
    int pc = FindFirstInst(block, 0);
    if (pc < 0)
        return;
    for (;;)
    {
        pANTLR3_BASE_TREE inst = GetInst(block, pc);
        bool end = Dispatch(inst);
        if (end)
            break;
        pc++;
        pc = FindFirstInst(block, pc);
    }
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

bool CUDA_EMULATOR::Dispatch(pANTLR3_BASE_TREE inst)
{
    pANTLR3_BASE_TREE i = (pANTLR3_BASE_TREE)inst->getChild(inst,0);
    int inst_type = i->getType(i);
    switch (inst_type)
    {
        case KI_ABS: ;
        case KI_ADD: ;
        case KI_ADDC: ;
        case KI_AND: ;
        case KI_ATOM: ;
        case KI_BAR: ;
        case KI_BFE: ;
        case KI_BFI: ;
        case KI_BFIND: ;
        case KI_BRA: ;
        case KI_BREV: ;
        case KI_BRKPT: ;
        case KI_CALL: ;
        case KI_CLZ: ;
        case KI_CNOT: ;
        case KI_COPYSIGN: ;
        case KI_COS: ;
        case KI_CVT: ;
        case KI_CVTA: ;
        case KI_DIV: ;
        case KI_EX2: ;
        case KI_EXIT:
            DoExit(inst);
            return true;
        case KI_FMA: ;
        case KI_ISSPACEP: ;
        case KI_LD:
            DoLd(inst);
            return false;
        case KI_LDU: ;
        case KI_LG2: ;
        case KI_MAD24: ;
        case KI_MAD: ;
        case KI_MAX: ;
        case KI_MEMBAR: ;
        case KI_MIN: ;
        case KI_MOV:
            DoMov(inst);
            return false;
        case KI_MUL24: ;
        case KI_MUL: ;
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
        case KI_SETP: ;
        case KI_SHL: ;
        case KI_SHR: ;
        case KI_SIN: ;
        case KI_SLCT: ;
        case KI_SQRT: ;
        case KI_ST:
            DoSt(inst);
            return false;
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
    return true;
}


void CUDA_EMULATOR::DoExit(pANTLR3_BASE_TREE inst)
{
    std::cout << "EXIT\n";
}

void CUDA_EMULATOR::DoMov(pANTLR3_BASE_TREE inst)
{
    std::cout << "MOV\n";
    // Get two operands, assign source to destination.
    pANTLR3_BASE_TREE odst = GetChild(inst,2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc = GetChild(inst,3);
    pANTLR3_BASE_TREE src = GetChild(osrc,0);
    Symbol * sdst = 0;
    int ssrc = 0;
    if (dst->getType(dst) == T_WORD)
    {
        sdst = FindSymbol(GetText(dst));
    } else assert(false);
    if (src->getType(src) == T_DEC_LITERAL)
    {
        ssrc = atoi(GetText(src));
    } else assert(false);
    *((int*)sdst->lvalue) = ssrc;
}

void CUDA_EMULATOR::DoLd(pANTLR3_BASE_TREE inst)
{
    std::cout << "LD\n";
    // Get two operands, assign source to destination.
    pANTLR3_BASE_TREE odst = GetChild(inst,2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc = GetChild(inst,3);
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
    *((int*)sdst->lvalue) = *(int*)ssrc->lvalue;
}

void CUDA_EMULATOR::DoSt(pANTLR3_BASE_TREE inst)
{
    std::cout << "ST\n";
    pANTLR3_BASE_TREE odst = GetChild(inst,2);
    pANTLR3_BASE_TREE dst = GetChild(odst,0);
    pANTLR3_BASE_TREE osrc = GetChild(inst,3);
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