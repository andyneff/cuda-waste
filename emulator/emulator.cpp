#include "emulator.h"
#include <assert.h>

EMULATOR * EMULATOR::singleton;

EMULATOR * EMULATOR::Singleton()
{
    if (singleton)
        return singleton;
    singleton = new EMULATOR();
    return singleton;
}

EMULATOR::EMULATOR()
{
}

extern pANTLR3_BASE_TREE parse(char * source);

void EMULATOR::Extract_From_Source(char * source)
{
    pANTLR3_BASE_TREE mod = parse(source);
    if (! mod)
        return;
    modules.push_back(mod);
    Extract_From_Tree(mod);
}

void EMULATOR::Extract_From_Tree(pANTLR3_BASE_TREE node)
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
    for (int i = 0; i < node->getChildCount(node); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)node->getChild(node, i);
        Extract_From_Tree(child);
    }
} 

void ** EMULATOR::RegisterFunction(void * fun, char * name)
{
    std::pair<void*, char*> i;
    i.first = fun;
    i.second = name;
    this->fun_to_name.insert(i);
    return 0;
}

cudaError_t EMULATOR::SetupArgument(const void *arg, size_t size, size_t offset)
{
    // record argument, size, offset.
    EMULATOR::arg a;
    a.argument = arg;
    a.size = size;
    a.offset = offset;
    this->arguments.push_back(a);
    return cudaSuccess;
}

cudaError_t EMULATOR::ConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
    conf.gridDim = gridDim;
    conf.blockDim = blockDim;
    conf.sharedMem = sharedMem;
    conf.stream = stream;
    return cudaSuccess;
}

void EMULATOR::Execute(void* hostfun)
{
    std::map<void*, char*>::iterator i = this->fun_to_name.find(hostfun);
    if (i == this->fun_to_name.end())
        return;
    char * name = i->second;
    std::map<char*, pANTLR3_BASE_TREE, ltstr>::iterator j = this->entry.find(name);
    if (j == this->entry.end())
        return;
    pANTLR3_BASE_TREE tree = j->second;
    // Execute.
    pANTLR3_BASE_TREE block;
    block = FindBlock(tree);
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

pANTLR3_BASE_TREE EMULATOR::FindBlock(pANTLR3_BASE_TREE node)
{
    for (int i = 0; i < node->getChildCount(node); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)node->getChild(node, i);
        if (child->getType(child) == TREE_BLOCK)
            return child;
    }
    return 0;
}

int EMULATOR::FindFirstInst(pANTLR3_BASE_TREE block, int first)
{
    for (int i = first; i < block->getChildCount(block); ++i)
    {
        pANTLR3_BASE_TREE child = (pANTLR3_BASE_TREE)block->getChild(block, i);
        if (child->getType(child) == TREE_INST)
            return i;
    }
    return -1;
}

pANTLR3_BASE_TREE EMULATOR::GetInst(pANTLR3_BASE_TREE block, int pc)
{
	assert(block->getType(block) == TREE_BLOCK);
    pANTLR3_BASE_TREE inst = (pANTLR3_BASE_TREE)block->getChild(block, pc);
    return inst;
}

bool EMULATOR::Dispatch(pANTLR3_BASE_TREE inst)
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

void EMULATOR::DoMov(pANTLR3_BASE_TREE inst)
{
}

void EMULATOR::DoLd(pANTLR3_BASE_TREE inst)
{
}

void EMULATOR::DoSt(pANTLR3_BASE_TREE inst)
{
}
