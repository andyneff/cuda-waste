#include "emulator.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include "tree.h"

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
    this->string_table = new StringTable();
    this->trace_level = 0;
    this->extern_memory_buffer = 0;
    this->carry = 0;
}

void CUDA_EMULATOR::SetTrace(int level)
{
    this->trace_level = level;
}

// In ptxp/driver.cpp.
extern TREE * parse(char * source);

TREE * CUDA_EMULATOR::Extract_From_Source(char * module_name, char * source)
{
    // Pick modules of only one type.
    if (strstr(module_name, this->device) == 0)
        return 0;

    if (this->trace_level > 0)
    {
        std::cout << "====================================================\n";
        std::cout << "PROFILE = " << module_name << std::endl;
        std::cout << "CODE:\n";
        std::cout << source << std::endl;
        std::cout << "====================================================\n\n\n";
    }

    TREE * mod = parse(source);
    if (! mod)
    {
        std::cout << "Error: cannot parse PTX!\n";
        assert(false);
    }
    modules.push_back(mod);
    Extract_From_Tree(mod);
    return mod;
}

void CUDA_EMULATOR::Extract_From_Tree(TREE * node)
{
    // Traverse the tree and look for key features like entry, func, variable declarations, etc.
    if (node->GetType() == TREE_ENTRY)
    {
        // First child will be name node.
        TREE * word = GetChild(node, 0);
        char * name = word->GetText();
        std::pair<char*, TREE *> i;
        i.first = (char*)name;
        i.second = node;
        this->entry.insert(i);
    }
    else if (node->GetType() == TREE_FUNC)
    {
        TREE * word = GetChild(node, 0);
        char * name = word->GetText();
        std::pair<char*, TREE *> i;
        i.first = (char*)name;
        i.second = node;
        this->func.insert(i);
    }
    for (int i = 0; i < node->GetChildCount(); ++i)
    {
        TREE * child = node->GetChild(i);
        Extract_From_Tree(child);
    }
} 

void CUDA_EMULATOR::SetupParams(TREE * e)
{
    // Create a new symbol table block for the parameters.
    SymbolTable * symbol_table = this->root;
    // For each parameter, create a symbol table entry, bind the arguments.
    // To do this, walk down both the AST and the setup argument list, and
    // associate each entry with the other in a symbol table entry.
    {
        int argc = this->arguments.size();
        TREE * param_list = (TREE *)e->GetChild(1);
        std::list<arg*>::iterator ia = this->arguments.begin();
        for (int i = 0; i < argc; ++i, ++ia)
        {
            // Get to the parameter in the AST.
            TREE * param = GetChild(param_list, i);
            TREE * name = GetChild(param, 0);
            char * n = name->GetText();
            TREE * type = GetChild(GetChild(param, 1), 0);
            char * t = type->GetText();
            // Get to the argument in the set up list.
            arg * a = *ia;
            // Create a symbol table entry.
            Symbol * s = new Symbol();
            s->emulator = this;
            s->pvalue = (void*)a->argument;
            s->name = n;
            s->size = a->size;
            s->typestring = t;
            s->array = false;
            s->index_max = 0;
            s->type = type->GetType();
            s->storage_class = K_PARAM;
            // Add the entry into the symbol table.
            std::pair<char*, Symbol*> sym;
            sym.first = n;
            sym.second = s;
            symbol_table->symbols.insert(sym);
        }
    }
    // erase arg list for next launch.
    {
        std::list<arg*>::iterator ia = this->arguments.begin();
        arg * a = *ia;
        delete a;
    }
    this->arguments.clear();
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

int CUDA_EMULATOR::GetType(TREE * c)
{
    return c->GetType();
}

int CUDA_EMULATOR::GetSize(TREE * tree_par_register)
{
    TREE * c = (TREE *)tree_par_register->GetChild(0);
    return (int)atoi(c->GetText());
}

CUDA_EMULATOR::SymbolTable * CUDA_EMULATOR::PushSymbolTable()
{
    SymbolTable * symbol_table = new SymbolTable();
    symbol_table->parent_block_symbol_table = this->root;
    this->root = symbol_table;
    return symbol_table;
}

void CUDA_EMULATOR::PopSymbolTable()
{
    SymbolTable * symbol_table = this->root;
    this->root = symbol_table->parent_block_symbol_table;
}

void CUDA_EMULATOR::SetupVariables(TREE * code, int * desired_storage_classes)
{
    // Create a new symbol table block for the globals.
    SymbolTable * symbol_table = this->root;
    // Go through the block and create entries in the symbol table for each variable declared.
    for (int i = 0; i < code->GetChildCount(); ++i)
    {
        TREE * var = code->GetChild(i);
        if (var->GetType() == TREE_VAR)
        {
            SetupSingleVar(var, desired_storage_classes, false);
        }
    }
}

void CUDA_EMULATOR::SetupSingleVar(TREE * var, int * desired_storage_classes, bool externed)
{
    // Create a new symbol table block for the globals.
    SymbolTable * symbol_table = this->root;
    // Got variable declaration.
    // Now extract info out of variable declaration.
    char * name = 0;
    int nreg = 0;
    TREE * ttype = 0;
    char * type = 0;
    int size = 0;
    int storage_class = 0;
    bool wrong_class = true;
    TREE * tarray = 0;
    TREE * tinitializer_values = 0;
    for (int j = 0; j < (int)var->GetChildCount(); ++j)
    {
        TREE * c = var->GetChild(j);
        int ct = GetType(c);
        if (ct == TREE_SPACE)
        {
            TREE * chi = GetChild(c, 0);
            storage_class = GetType(chi);
            // no need to continue if wrong storage class.
            for (int k = 0; desired_storage_classes[k] != 0; ++k)
            {
                if (storage_class == desired_storage_classes[k])
                {
                    wrong_class = false;
                }
            }
        } else if (ct == TREE_ALIGN)
        {
            // Nothing to do.
        } else if (ct == TREE_TYPE)
        {
            ttype = GetChild(c, 0);
            type = ttype->GetText();
            int t = GetType(ttype);
            size = Sizeof(t);
        } else if (ct == T_WORD)
        {
            name = c->GetText();
        } else if (ct == TREE_PAR_REGISTER)
        {
            nreg = GetSize(c);
        } else if (ct == TREE_ARRAY)
        {
            // declare var as an array.
            tarray = c;
        } else if (ct == T_EQ)
        {
            tinitializer_values = c;
        } else assert(false);
    }
    if (wrong_class)
        return;
    // Convert raw info into symbol declarations and sizes.
    if (nreg > 0)
    {
        for (int k = 0; k < nreg; ++k)
        {
            char full_name[1000];
            sprintf(full_name, "%s%d", name, k+1);
            // Create a symbol table entry.
            Symbol * s = new Symbol();
            s->emulator = this;
            s->name = this->string_table->Entry(full_name);
            s->size = size;
            s->pvalue = (void*)malloc(size);
            s->typestring = this->string_table->Entry(type);
            s->type = ttype->GetType();
            s->storage_class = storage_class;
            s->array = false;
            s->index_max = 0;
            // Add the entry into the symbol table.
            std::pair<char*, Symbol*> sym;
            sym.first = s->name;
            sym.second = s;
            symbol_table->symbols.insert(sym);
        }
    } else {
        // Create a symbol table entry.
        Symbol * s = new Symbol();
        s->emulator = this;
        s->name = this->string_table->Entry(name);
        s->size = size;
        s->array = false;
        s->index_max = 0;
        // Allocate array if declared as one.
        if (tarray != 0)
        {
            s->array = true;
            // Using the symbol in ptx is essentially a pointer.
            // So, mov and cvta loads a pointer to a buffer.
            // So, there are two malloc's.
            int total = 1;
            for (int a = 0; ; ++a)
            {
                TREE * t = GetChild(tarray, a);
                if (t == 0)
                    break;
                int gt = GetType(t);
                // Look at size information if not external.
                if (externed == false && gt == T_OB)
                {
                    ++a;
                    TREE * n = GetChild(tarray, a);
                    assert(n != 0);
                    if (GetType(n) == T_DEC_LITERAL)
                    {
                        int sz = atoi(n->GetText());
                        total = total * sz;
                    }
                    ++a;
                    TREE * t2 = GetChild(tarray, a);
                    assert(t2 != 0);
                    assert(GetType(t2) == T_CB);
                    ++a;
                }
                else if (externed != 0)
                    ;
                else assert(false);
            }
            s->index_max = total;
            void * ptr = 0;
            if (! externed)
                ptr = (void*)malloc(size * total);
            else
                // Each extern points to the same allocated array.
                ptr = this->extern_memory_buffer;
            s->pvalue = (void*)malloc(sizeof(void*));
            ((TYPES*)s->pvalue)->pvoid = ptr;

            // Now work on optional initializer...
            if (tinitializer_values != 0)
            {
                unsigned char * mptr = (unsigned char *)ptr;
                for (int a = 0; ; ++a)
                {
                    TREE * t = GetChild(tinitializer_values, a);
                    if (t == 0)
                        break;
                    int gt = GetType(t);
                    if (gt == TREE_CONSTANT_EXPR)
                    {
                        TREE * n = GetChild(t, 0);
                        int type = ttype->GetType();
                        Constant c = Eval(type, n);
                        TYPES * s1 = (TYPES*)mptr;
                        switch (type)
                        {
                            case K_B8:
                                s1->b8 = c.value.b8;
                                break;
                            case K_U16:
                                s1->u16 = c.value.u16;
                                break;
                            case K_S16:
                                s1->s16 = c.value.s16;
                                break;
                            case K_U32:
                                s1->u32 = c.value.u32;
                                break;
                            case K_S32:
                                s1->s32 = c.value.s32;
                                break;
                            default:
                                assert(false);
                        }
                    }
                    else assert(false);
                    mptr += size;
                }
            }
        }
        else
        {
            s->pvalue = (void*)malloc(size);
        }
        s->typestring = this->string_table->Entry(type);
        s->type = ttype->GetType();
        s->storage_class = storage_class;
        // Add the entry into the symbol table.
        std::pair<char*, Symbol*> sym;
        sym.first = s->name;
        sym.second = s;
        symbol_table->symbols.insert(sym);
    }
}


void CUDA_EMULATOR::SetupGotos(TREE * code)
{
    SymbolTable * symbol_table = this->root;
    // Scan ahead and find all labels.  Enter them into the symbol
    // table.
    for (int i = 0; i < code->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)code->GetChild(i);
        if (child->GetType() == TREE_LABEL)
        {
            TREE * label = child->GetChild(0);
            char * name = label->GetText();
            Symbol * s = new Symbol();
            s->emulator = this;
            s->name = this->string_table->Entry(name);
            s->typestring = "label";
            s->type = label->GetType();
            s->size = 0;
            s->pvalue = (void*)i;
            s->storage_class = 0;
            s->array = false;
            s->index_max = 0;
            // Add the entry into the symbol table.
            std::pair<char*, Symbol*> sym;
            sym.first = s->name;
            sym.second = s;
            symbol_table->symbols.insert(sym);
        }
    }
}

void CUDA_EMULATOR::SetupExternShared(TREE * code)
{
    // No need to resolve anything if no shared memory to set up.
    if (this->conf.sharedMem == 0)
        return;
    this->extern_memory_buffer = (void*)malloc(conf.sharedMem);
    for (TREE * p = code; p != 0; p = p->GetParent())
    {
        SymbolTable * symbol_table = this->root;
        // Scan ahead and find all extern nodes.
        // Enter them into the symbol table if they are shared
        // memory.
        for (int i = 0; i < p->GetChildCount(); ++i)
        {
            TREE * child = (TREE *)p->GetChild(i);
            if (child->GetType() == TREE_EXTERN)
            {
                TREE * cc = child->GetChild(0);
                if (cc)
                {
                    int t = child->GetChild(0)->GetType();
                    if (t != TREE_VAR)
                        continue;
                }
                TREE * var = child->GetChild(0);
                int sc[] = { K_SHARED, 0};
                SetupSingleVar(var, sc, true);
            }
        }
    }
}

void CUDA_EMULATOR::ConfigureGrid(dim3 dim)
{
    this->conf.gridDim = dim;
}

void CUDA_EMULATOR::ConfigureBlock(dim3 dim)
{
    this->conf.blockDim = dim;
}

void CUDA_EMULATOR::ConfigureSharedMemory(size_t sharedMem)
{
    conf.sharedMem = sharedMem;
}

void CUDA_EMULATOR::ConfigureStream(cudaStream_t stream)
{
    conf.stream = stream;
}

void CUDA_EMULATOR::Execute(TREE * entry)
{
    // Get function block.
    TREE * code = FindBlock(entry);

    // Create symbol table for outer blocks.
    PushSymbolTable();
    for (TREE * p = code->GetParent()->GetParent(); p != 0; p = p->GetParent())
    {
        int sc[] = { K_GLOBAL, 0};
        SetupVariables(p, sc);
    }

    // Create symbol table for this block.
    PushSymbolTable();
    int sc[] = { K_GLOBAL, K_CONST, K_TEX, 0};
    SetupVariables(code, sc);
    SetupGotos(code);
    SetupParams(entry);
    CreateSymbol("%nctaid", "dim3", K_V4, &conf.gridDim, sizeof(conf.gridDim), K_LOCAL);
    CreateSymbol("%ntid", "dim3", K_V4, &conf.blockDim, sizeof(conf.blockDim), K_LOCAL);

    bool do_thread_synch = CodeRequiresThreadSynchronization(code);
    if (this->trace_level > 0)
        std::cout << "Thread synchronization " << (do_thread_synch ? "is" : "is not") << " required.\n";
    ExecuteBlocks(do_thread_synch, code);
}

void CUDA_EMULATOR::ExecuteBlocks(bool do_thread_synch, TREE * code)
{
    for (int bidx = 0; bidx < conf.gridDim.x; ++bidx)
    {
        for (int bidy = 0; bidy < conf.gridDim.y; ++bidy)
        {
            for (int bidz = 0; bidz < conf.gridDim.z; ++bidz)
            {
                ExecuteSingleBlock(do_thread_synch, code, bidx, bidy, bidz);
            }
        }
    }
}

bool CUDA_EMULATOR::CodeRequiresThreadSynchronization(TREE * code)
{
    bool result = false;
    for (int i = 0; i < (int)code->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)GetChild(code, i);
        if (GetType(child) == TREE_INST)
        {
            TREE * inst = child;
            TREE * i = (TREE *)GetChild(inst, 0);
            int inst_type = i->GetType();
            if (inst_type == TREE_PRED)
            {
                // Predicate preceeds the instruction.
                TREE * pred = i;
                i = (TREE *)inst->GetChild(1);
                inst_type = i->GetType();
            }
            switch (inst_type)
            {
                case KI_BAR:
                case KI_MEMBAR:
                    return true;
                default:
                    break;
            }
        }
    }
    return false;
}

void CUDA_EMULATOR::ExecuteSingleBlock(bool do_thread_synch, TREE * code, int bidx, int bidy, int bidz)
{
    //_CrtMemState state_begin;
    //_CrtMemCheckpoint(&state_begin);
    
    std::queue<Thread*> queue;

    // Keep track of symbol table root to restore later.  This is because of the awful
    // use of root on a per-thread basis.
    SymbolTable * save = this->root;

    // Two ways to do this.  If there is no thread synchronization,
    // then threads can run serially, one after another.  In this case,
    // a symbol table of registers/locals/aligns/params does not have
    // to be created for every thread.  Instead, create this at a more
    // global level.  If the code contains thread synchronization,
    // then create the local symbols for each thread.
    // This test is just for performance enhancement.
    // Create a new symbol table and add the block index variables.
    SymbolTable * block_symbol_table = PushSymbolTable();
    dim3 bid(bidx, bidy, bidz);
    CreateSymbol("%ctaid", "dim3", K_V4, &bid, sizeof(bid), K_LOCAL);

    if (do_thread_synch)
    {
        // Add to this symbol table any explicit shared memory
        // variables.
        int sc[] = { K_SHARED, 0 };
        SetupVariables(code, sc);
    } else
    {
        int sc[] = { K_SHARED, K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
        SetupVariables(code, sc);
    }

    // Add to this symbol table any extern declared shared memory
    // variables.
    SetupExternShared(code);

    for (int tidx = 0; tidx < conf.blockDim.x; ++tidx)
    {
        for (int tidy = 0; tidy < conf.blockDim.y; ++tidy)
        {
            for (int tidz = 0; tidz < conf.blockDim.z; ++tidz)
            {
                PushSymbolTable();
                dim3 tid(tidx, tidy, tidz);
                CreateSymbol("%tid", "dim3", K_V4, &tid, sizeof(tid), K_LOCAL);
                if (do_thread_synch)
                {
                    int sc[] = { K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
                    SetupVariables(code, sc);
                }
                Thread * thread = new Thread(this, code, 0, this->root);
                queue.push(thread);
                PopSymbolTable();
            }
        }
    }
    PopSymbolTable();

    int num_waiting_threads = 0;
    while (! queue.empty())
    {
        Thread * thread = queue.front();
        queue.pop();
        if (! thread->Waiting())
        {
            thread->Execute();
        }
        if (! thread->Finished())
        {
            queue.push(thread);
            num_waiting_threads++;
        }
        else
            delete thread;

        // thread synchronization assumes all threads wait.
        if (num_waiting_threads != 0 && num_waiting_threads == queue.size())
        {
            if (this->trace_level > 0)
                std::cout << "All " << num_waiting_threads << " threads synchronized!\n";
            // all threads waiting.  Reset all threads to not wait.
            for (int i = 0; i < num_waiting_threads; ++i)
            {
                Thread * t = queue.front();
                queue.pop();
                t->Reset();
                queue.push(t);
            }
            num_waiting_threads = 0;
        }
    }
    // Restore...
    this->root = save;
    // Delete block symbol table.
    delete block_symbol_table;
    if (this->extern_memory_buffer)
        delete this->extern_memory_buffer;
    this->extern_memory_buffer = 0;

    //_CrtMemState state_end;
    //_CrtMemCheckpoint(&state_end);

    //_CrtMemDumpAllObjectsSince(&state_begin);
}

CUDA_EMULATOR::Thread::Thread(CUDA_EMULATOR * emulator, TREE * block, int pc, CUDA_EMULATOR::SymbolTable * root)
{
    this->emulator = emulator;
    this->block = block;
    this->pc = pc;
    this->root = root;
    this->finished = false;
    this->wait = false;
    this->carry = 0;
}

CUDA_EMULATOR::Thread::~Thread()
{
    delete root;
}

bool CUDA_EMULATOR::Thread::Execute()
{
    // set up symbol table environment.
    this->emulator->root = this->root;
    int pc = this->pc;
    this->emulator->carry = this->carry;

    // Execute.
    pc = emulator->FindFirstInst(block, pc);
    if (pc < 0)
    {
        this->finished = true;
        return this->finished;
    }
    for (;;)
    {
        TREE * inst = this->emulator->GetInst(block, pc);
        if (this->emulator->trace_level > 3)
            this->emulator->Dump("before", pc, inst);

        int next = this->emulator->Dispatch(inst);
        if (next > 0)
            pc = next;
        else if (next == -KI_EXIT)
        {
            this->finished = true;
            return this->finished;
        }
        else if (next == -KI_BAR)
        {
            // Set state of this thread to wait, and pack up current program counter.
            this->wait = true;
        this->pc = pc + 1;
        this->carry = this->emulator->carry;
            return this->finished;
        }
        else
            pc++;

        pc = this->emulator->FindFirstInst(block, pc);

        if (this->emulator->trace_level > 2)
            this->emulator->Dump("after", pc, inst);
    }
}

bool CUDA_EMULATOR::Thread::Finished()
{
    return this->finished;
}

void CUDA_EMULATOR::Thread::Reset()
{
    this->wait = false;
}

bool CUDA_EMULATOR::Thread::Waiting()
{
    return this->wait;
}

void CUDA_EMULATOR::PrintName(TREE * inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    std::cout << GetChild(inst, start)->GetText() << "\n";
} 

void CUDA_EMULATOR::Print(TREE * node, int level)
{
    for (int i = 0; i < level; ++i)
        std::cout << "   ";
    std::cout << node->GetText() << "\n";
    if (this->trace_level > 1)
    {
        for (int i = 0; i < (int)node->GetChildCount(); ++i)
        {
            TREE * child = (TREE *)node->GetChild(i);
            Print(child, level+1);
        }
    }
} 

void CUDA_EMULATOR::Dump(char * comment, int pc, TREE * inst)
{
    std::cout << "\n";
    std::cout << comment << "\n";
    std::cout << "PC = " << pc << "\n";
    Print(inst, 0);
    std::cout << "Symbol tables:\n";
    int level = 0;
    for (SymbolTable * st = this->root; st != 0; st = st->parent_block_symbol_table, level++)
    {
        std::cout << "---- Level " << level << " ----\n";
        std::map<char*, Symbol*, ltstr>::iterator it;
        for (it = st->symbols.begin(); it != st->symbols.end(); ++it)
        {
            Symbol * s = (*it).second;
            std::cout << "name: " << s->name << " ";
            std::cout << "size: " << s->size << " ";
            std::cout << "stor: " << s->storage_class << " ";
            std::cout << "type: " << s->typestring << " ";
            if (strcmp(s->typestring, "label") == 0)
                std::cout << "val:  " << (int)s->pvalue << "\n";
            else if (strcmp(s->typestring, "dim3") == 0)
                std::cout << "val:  " << ((dim3*)s->pvalue)->x
                << " " << ((dim3*)s->pvalue)->y
                << " " << ((dim3*)s->pvalue)->z
                << "\n";
            else if (strcmp(s->typestring, ".pred") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->pred << "\n";
            else if (strcmp(s->typestring, ".u8") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u8 << "\n";
            else if (strcmp(s->typestring, ".u16") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u16 << "\n";
            else if (strcmp(s->typestring, ".u32") == 0)
            {
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u32 << " ";
                std::cout << std::hex << ((TYPES*)s->pvalue)->u32;
                std::cout << std::dec << "\n";
            }
            else if (strcmp(s->typestring, ".u64") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u64 << "\n";
            else if (strcmp(s->typestring, ".s8") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->s8 << "\n";
            else if (strcmp(s->typestring, ".s16") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->s16 << "\n";
            else if (strcmp(s->typestring, ".s32") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->s32 << "\n";
            else if (strcmp(s->typestring, ".s64") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->s64 << "\n";
            else if (strcmp(s->typestring, ".b8") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u8 << "\n";
            else if (strcmp(s->typestring, ".b16") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u16 << "\n";
            else if (strcmp(s->typestring, ".b32") == 0)
            {
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u32 << " ";
                std::cout << std::hex << ((TYPES*)s->pvalue)->u32;
                std::cout << std::dec << "\n";
            }
            else if (strcmp(s->typestring, ".b64") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->u64 << "\n";
            else if (strcmp(s->typestring, ".f32") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->f32 << "\n";
            else if (strcmp(s->typestring, ".f64") == 0)
                std::cout << "val:  " << ((TYPES*)s->pvalue)->f64 << "\n";
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

void CUDA_EMULATOR::CreateSymbol(char * name, char * typestring, int type, void * value, size_t size, int storage_class)
{
    // First find it.
    Symbol * s = FindSymbol(name);
    if (s)
    {
        assert(s->size == size);
        // Update value.
        memcpy(s->pvalue, value, size);
        return;
    }
    // Create a symbol table entry.
    s = new Symbol();
    s->emulator = this;
    s->name = this->string_table->Entry(name);
    s->typestring = this->string_table->Entry(typestring);
    s->type = type;
    s->size = size;
    s->pvalue = (void*)malloc(size);
    s->storage_class = storage_class;
    s->array = false;
    s->index_max = 0;
    memcpy(s->pvalue, value, size);
    // Add the entry into the symbol table.
    std::pair<char*, Symbol*> sym;
    sym.first = s->name;
    sym.second = s;
    SymbolTable * symbol_table = root;
    symbol_table->symbols.insert(sym);
}

TREE * CUDA_EMULATOR::FindBlock(TREE * node)
{
    for (int i = 0; i < (int)node->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)node->GetChild(i);
        if (child->GetType() == TREE_BLOCK)
            return child;
    }
    return 0;
}

int CUDA_EMULATOR::FindFirstInst(TREE * block, int first)
{
    for (int i = first; i < (int)block->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)block->GetChild(i);
        if (child->GetType() == TREE_INST)
            return i;
    }
    return -1;
}

TREE * CUDA_EMULATOR::GetInst(TREE * block, int pc)
{
    assert(block->GetType() == TREE_BLOCK);
    TREE * inst = (TREE *)block->GetChild(pc);
    return inst;
}


TREE * CUDA_EMULATOR::GetChild(TREE * node, int n)
{
    TREE * c = (TREE *)node->GetChild(n);
    return c;
}


char * CUDA_EMULATOR::StringTable::Entry(char * text)
{
    char * result = 0;
    std::map<char *, char*, ltstr>::iterator it = this->table.find(text);
    if (it == this->table.end())
    {
        std::pair<char *, char*> p;
        char * the_text = strdup(text);
        p.first = the_text;
        p.second = the_text;
        this->table.insert(p);
        result = the_text;
    }
    else
    {
        result = it->second;
    }
    return result;
}

char * CUDA_EMULATOR::StringTableEntry(char * text)
{
    return this->string_table->Entry(text);
}

int CUDA_EMULATOR::Dispatch(TREE * inst)
{
    if (this->trace_level > 0)
    {
        PrintName(inst);
        if (this->trace_level > 1)
            Print(inst, 0);
    }
    
    TREE * i = (TREE *)inst->GetChild(0);
    int inst_type = i->GetType();
    if (inst_type == TREE_PRED)
    {
        // Predicate preceeds the instruction.
        TREE * pred = i;
        i = (TREE *)inst->GetChild(1);
        inst_type = i->GetType();

        // Check if pred is true.  If false, ignore instruction with this predicate.
        int i = 0;
        bool not = false;
        TREE * tsym = 0;
        for (;; ++i)
        {
            TREE * t = GetChild(pred, i);
            if (t == 0)
                break;
            int gt = GetType(t);
            if (gt == T_NOT)
                not = true;
            else if (gt == T_WORD)
                tsym = t;
            else assert(false);
        }
        assert(tsym != 0);
        Symbol * sym = FindSymbol(tsym->GetText());
        assert(sym != 0);
        TYPES * s = (TYPES*)sym->pvalue;

        bool test = s->pred;
        if (not)
            test = ! test;
        if (! test)
        {
            if (this->trace_level > 1)
                std::cout << "Skipping instruction because guard predicate is false\n";
            return 0; // continue.
        }
    }
    try {
        switch (inst_type)
        {
            case KI_ABS:
                return DoAbs(inst);
            case KI_ADD:
                return DoAdd(inst);
            case KI_ADDC:
                return DoAddc(inst);
            case KI_AND:
                return DoAnd(inst);
            case KI_ATOM:
                return DoAtom(inst);
            case KI_BAR:
                return DoBar(inst);
            case KI_BFE:
                return DoBfe(inst);
            case KI_BFI:
                return DoBfi(inst);
            case KI_BFIND:
                return DoBfind(inst);
            case KI_BRA:
                return DoBra(inst);
            case KI_BREV:
                return DoBrev(inst);
            case KI_BRKPT:
                return DoBrkpt(inst);
            case KI_CALL:
                return DoCall(inst);
            case KI_CLZ:
                return DoClz(inst);
            case KI_CNOT:
                return DoCnot(inst);
            case KI_COPYSIGN:
                return DoCopysign(inst);
            case KI_COS:
                return DoCos(inst);
            case KI_CVT:
                return DoCvt(inst);
            case KI_CVTA:
                return DoCvta(inst);
            case KI_DIV:
                return DoDiv(inst);
            case KI_EX2:
                return DoEx2(inst);
            case KI_EXIT:
                return DoExit(inst);
            case KI_FMA:
                return DoFma(inst);
            case KI_ISSPACEP:
                return DoIsspacep(inst);
            case KI_LD:
                return DoLd(inst);
            case KI_LDU:
                return DoLdu(inst);
            case KI_LG2:
                return DoLg2(inst);
            case KI_MAD:
                return DoMad(inst);
            case KI_MAD24:
                return DoMad24(inst);
            case KI_MAX:
                return DoMax(inst);
            case KI_MEMBAR:
                return DoMembar(inst);
            case KI_MIN:
                return DoMin(inst);
            case KI_MOV:
                return DoMov(inst);
            case KI_MUL24:
                return DoMul24(inst);
            case KI_MUL:
                return DoMul(inst);
            case KI_NEG:
                return DoNeg(inst);
            case KI_NOT:
                return DoNot(inst);
            case KI_OR:
                return DoOr(inst);
            case KI_PMEVENT:
                return DoPmevent(inst);
            case KI_POPC:
                return DoPopc(inst);
            case KI_PREFETCH:
                return DoPrefetch(inst);
            case KI_PREFETCHU:
                return DoPrefetchu(inst);
            case KI_PRMT:
                return DoPrmt(inst);
            case KI_RCP:
                return DoRcp(inst);
            case KI_RED:
                return DoRed(inst);
            case KI_REM:
                return DoRem(inst);
            case KI_RET:
                return DoRet(inst);
            case KI_RSQRT:
                return DoRsqrt(inst);
            case KI_SAD:
                return DoSad(inst);
            case KI_SELP:
                return DoSelp(inst);
            case KI_SET:
                return DoSet(inst);
            case KI_SETP:
                return DoSetp(inst);
            case KI_SHL:
                return DoShl(inst);
            case KI_SHR:
                return DoShr(inst);
            case KI_SIN:
                return DoSin(inst);
            case KI_SLCT:
                return DoSlct(inst);
            case KI_SQRT:
                return DoSqrt(inst);
            case KI_ST:
                return DoSt(inst);
            case KI_SUB:
                return DoSub(inst);
            case KI_SUBC:
                return DoSubc(inst);
            case KI_SULD:
                return DoSuld(inst);
            case KI_SUQ:
                return DoSuq(inst);
            case KI_SURED:
                return DoSured(inst);
            case KI_SUST:
                return DoSust(inst);
            case KI_TESTP:
                return DoTestp(inst);
            case KI_TEX:
                return DoTex(inst);
            case KI_TRAP:
                return DoTrap(inst);
            case KI_TXQ:
                return DoTxq(inst);
            case KI_VABSDIFF:
                return DoVabsdiff(inst);
            case KI_VADD:
                return DoVadd(inst);
            case KI_VMAD:
                return DoVmad(inst);
            case KI_VMAX:
                return DoVmax(inst);
            case KI_VMIN:
                return DoVmin(inst);
            case KI_VOTE:
                return DoVote(inst);
            case KI_VSET:
                return DoVset(inst);
            case KI_VSHL:
                return DoVshl(inst);
            case KI_VSHR:
                return DoVshr(inst);
            case KI_VSUB:
                return DoVsub(inst);
            case KI_XOR:
                return DoXor(inst);
            default:
                assert(false);
        }
    } catch (Unimplemented * u)
    {
        std::cout << u->ShowReason() << "\n";
        delete u;
    }
    return -1; // end.
}

CUDA_EMULATOR::Constant CUDA_EMULATOR::Eval(int expected_type, TREE * const_expr)
{
    // Perform bottom-up evaluation of a constant expression.
    Constant result;
    result.type = expected_type;
    char * dummy;
    char * text = const_expr->GetText();
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
    } else if (GetType(const_expr) == T_QUESTION)
    {
        throw new Unimplemented("Question operator in constant expression not supported.\n");
    } else if (GetType(const_expr) == T_OROR)
    {
        // Perform boolean OR.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 || rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 || rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 || rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 || rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 || rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 || rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 || rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 || rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 || rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 || rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 || rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 || rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_ANDAND)
    {
        // Perform boolean AND.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 && rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 && rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 && rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 && rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 && rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 && rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 && rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 && rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 && rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 && rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 && rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 && rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_OR)
    {
        // Perform bit OR.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 | rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 | rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 | rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 | rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 | rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 | rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 | rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 | rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 | rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 | rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 | rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 | rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_XOR)
    {
        // Perform bit XOR.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 ^ rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 ^ rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 ^ rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 ^ rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 ^ rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 ^ rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 ^ rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 ^ rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 ^ rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 ^ rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 ^ rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 ^ rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_AND)
    {
        // Perform bit OR.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 & rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 & rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 & rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 & rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 & rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 & rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 & rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 & rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 & rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 & rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 & rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 & rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_EQEQ)
    {
        // Perform EQ.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 == rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 == rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 == rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 == rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 == rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 == rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 == rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 == rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 == rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 == rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 == rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 == rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_NOTEQ)
    {
        // Perform bit !=.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 != rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 != rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 != rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 != rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 != rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 != rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 != rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 != rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 != rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 != rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 != rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 != rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_LE)
    {
        // Perform bit LE.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 <= rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 <= rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 <= rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 <= rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 <= rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 <= rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 <= rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 <= rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 <= rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 <= rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 <= rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 <= rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_GE)
    {
        // Perform bit GE
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 >= rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 >= rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 >= rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 >= rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 >= rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 >= rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 >= rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 >= rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 >= rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 >= rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 >= rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 >= rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_LT)
    {
        // Perform bit <.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 < rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 < rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 < rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 < rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 < rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 < rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 < rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 < rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 < rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 < rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 < rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 < rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_GT)
    {
        // Perform bit OR.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 > rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 > rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 > rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 > rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 > rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 > rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 > rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 > rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 > rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 > rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 > rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 > rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_LTLT)
    {
        // Perform bit <<
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 << rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 << rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 << rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 << rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 << rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 << rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 << rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 << rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 << rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 << rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 << rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 << rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_GTGT)
    {
        // Perform bit >>
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 >> rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 >> rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 >> rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 >> rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 >> rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 >> rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 >> rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 >> rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 >> rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 >> rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 >> rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 >> rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_PLUS && const_expr->GetChild(1) != 0)
    {
        // Perform bit '+'.
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 + rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 + rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 + rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 + rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 + rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 + rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 + rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 + rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 + rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 + rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 + rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 + rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_MINUS && const_expr->GetChild(1) != 0)
    {
        // Perform bit '-'
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 - rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 - rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 - rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 - rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 - rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 - rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 - rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 - rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 - rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 - rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 - rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 - rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_STAR)
    {
        // Perform bit '*'
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 * rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 * rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 * rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 * rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 * rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 * rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 * rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 * rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 * rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 * rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 * rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 * rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_SLASH)
    {
        // Perform bit '/'
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 / rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 / rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 / rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 / rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 / rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 / rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 / rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 / rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 / rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 / rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 / rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 / rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_PERCENT)
    {
        // Perform bit '%'
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        Constant rhs = Eval(expected_type, const_expr->GetChild(1));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8 % rhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16 % rhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32 % rhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64 % rhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8 % rhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16 % rhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32 % rhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64 % rhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8 % rhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16 % rhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32 % rhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64 % rhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_PLUS)
    {
        // Perform bit >>
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = lhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_MINUS)
    {
        // Perform bit >>
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = - lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = - lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = - lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = - lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = - lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = - lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = - lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = - lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = - lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = - lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = - lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = - lhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_TILDE)
    {
        // Perform bit >>
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = ~ lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = ~ lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = ~ lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = ~ lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = ~ lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = ~ lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = ~ lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = ~ lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = ~ lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = ~ lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = ~ lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = ~ lhs.value.b64;
                break;
            default:
                assert(false);
        }
    } else if (GetType(const_expr) == T_NOT)
    {
        // Perform !
        Constant lhs = Eval(expected_type, const_expr->GetChild(0));
        switch (expected_type)
        {
            case K_U8:
                result.value.u8 = ! lhs.value.u8;
                break;
            case K_U16:
                result.value.u16 = ! lhs.value.u16;
                break;
            case K_U32:
                result.value.u32 = ! lhs.value.u32;
                break;
            case K_U64:
                result.value.u64 = ! lhs.value.u64;
                break;
            case K_S8:
                result.value.s8 = ! lhs.value.s8;
                break;
            case K_S16:
                result.value.s16 = ! lhs.value.s16;
                break;
            case K_S32:
                result.value.s32 = ! lhs.value.s32;
                break;
            case K_S64:
                result.value.s64 = ! lhs.value.s64;
                break;
            case K_B8:
                result.value.b8 = ! lhs.value.b8;
                break;
            case K_B16:
                result.value.b16 = ! lhs.value.b16;
                break;
            case K_B32:
                result.value.b32 = ! lhs.value.b32;
                break;
            case K_B64:
                result.value.b64 = ! lhs.value.b64;
                break;
            default:
                assert(false);
        }
    }
    else
        assert(false);
    return result;
}

void CUDA_EMULATOR::unimplemented(bool condition, char * text)
{
    if (condition)
    {
        throw new Unimplemented(text);
    }
}

void CUDA_EMULATOR::unimplemented(char * text)
{
    throw new Unimplemented(text);
}

CUDA_EMULATOR::SymbolTable::SymbolTable()
{
}

CUDA_EMULATOR::SymbolTable::~SymbolTable()
{
    std::map<char*, Symbol*, ltstr>::iterator it = this->symbols.begin();
    for ( ; it != this->symbols.end(); ++it)
    {
        delete it->second;
    }
}

