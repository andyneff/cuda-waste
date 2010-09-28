#include "emulator.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include "tree.h"
#include <process.h>    /* _beginthread, _endthread */
#include "thread.h"
#include "symbol-table.h"
#include "string-table.h"
#include "symbol.h"
#include "constant.h"
#include "types.h"

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
    this->device = "compute_20";
    this->string_table = new STRING_TABLE();
    this->trace_level = 0;
    this->extern_memory_buffer = 0;
}

void EMULATOR::SetTrace(int level)
{
    this->trace_level = level;
}

// In ptxp/driver.cpp.
extern TREE * parse(char * source);

TREE * EMULATOR::Extract_From_Source(char * module_name, char * source)
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

void EMULATOR::Extract_From_Tree(TREE * node)
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

void EMULATOR::SetupParams(SYMBOL_TABLE * symbol_table, TREE * e)
{
    // Create a new symbol table block for the parameters.
    //SYMBOL_TABLE * symbol_table = this->root;
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
            SYMBOL * s = new SYMBOL();
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
            std::pair<char*, SYMBOL*> sym;
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

size_t EMULATOR::Sizeof(int type)
{
    switch (type)
    {
    case K_S8: return sizeof(signed __int8);
    case K_S16: return sizeof(signed __int16);
    case K_S32: return sizeof(signed __int32);
    case K_S64: return sizeof(signed __int64);
    case K_U8: return sizeof(unsigned __int8);
    case K_U16: return sizeof(unsigned __int16);
    case K_U32: return sizeof(unsigned __int32);
    case K_U64: return sizeof(unsigned __int64);
    case K_F16: throw new Unimplemented("F16 unimplemented.\n"); return sizeof(float);
    case K_F32: return sizeof(float);
    case K_F64: return sizeof(double);
    case K_B8: return sizeof(unsigned __int8);
    case K_B16: return sizeof(signed __int16);
    case K_B32: return sizeof(signed __int32);
    case K_B64: return sizeof(signed __int64);
    case K_PRED: return sizeof(bool);
    }
    assert(false);
    return 0;
}

int EMULATOR::GetType(TREE * c)
{
    return c->GetType();
}

int EMULATOR::GetSize(TREE * tree_par_register)
{
    TREE * c = (TREE *)tree_par_register->GetChild(0);
    return (int)atoi(c->GetText());
}

SYMBOL_TABLE * EMULATOR::PushSymbolTable(SYMBOL_TABLE * parent)
{
    SYMBOL_TABLE * symbol_table = new SYMBOL_TABLE();
    symbol_table->parent_block_symbol_table = parent;
    return symbol_table;
}

void EMULATOR::SetupVariables(SYMBOL_TABLE * symbol_table, TREE * code, int * desired_storage_classes)
{
    // Go through the block and create entries in the symbol table for each variable declared.
    for (int i = 0; i < code->GetChildCount(); ++i)
    {
        TREE * var = code->GetChild(i);
        if (var->GetType() == TREE_VAR)
        {
            SetupSingleVar(symbol_table, var, desired_storage_classes, false);
        }
    }
}

void EMULATOR::SetupSingleVar(SYMBOL_TABLE * symbol_table, TREE * var, int * desired_storage_classes, bool externed)
{
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
    int vec = 0;
    int total = 0;
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
            // Using the symbol in ptx is essentially a pointer.
            // But, mov and cvta loads a pointer to the pointer when
            // addressing symbols in memory.
            total = 1;
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
        } else if (ct == T_EQ)
        {
            tinitializer_values = c;
        } else if (ct == TREE_VECTOR_TYPE)
        {
            tarray = c;
            TREE * c2 = c->GetChild(0);
            ct = c2->GetType();
            vec = ct;
            if (ct == K_V2)
                total = 2;
            else
                total = 4;
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
            sprintf(full_name, "%s%d", name, k);
            // Create a symbol table entry.
            SYMBOL * s = new SYMBOL();
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
            std::pair<char*, SYMBOL*> sym;
            sym.first = s->name;
            sym.second = s;
            symbol_table->symbols.insert(sym);
        }
    } else {
        // Create a symbol table entry.
        SYMBOL * s = new SYMBOL();
        s->emulator = this;
        s->name = this->string_table->Entry(name);
        s->size = size;
        // array flag helps in printing, but it works like any other
        // storage.
        s->array = false;
        s->index_max = 0;
        void * ptr = 0;
        // Allocate array if declared as one.
        if (tarray != 0)
        {
            s->array = true;
            s->index_max = total;
            if (! externed)
                ptr = (void*)malloc(size * total);
            else
                // Each extern points to the same allocated array.
                ptr = this->extern_memory_buffer;
            s->pvalue = ptr;
        }
        else
        {
            s->pvalue = (void*)malloc(size);
            ptr = s->pvalue;
        }

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
                    CONSTANT c = Eval(type, n);
                    TYPES::Types * s1 = (TYPES::Types*)mptr;
                    switch (type)
                    {
                        case K_B8:
                            s1->b8 = c.value.b8;
                            break;
                        case K_U8:
                            s1->u8 = c.value.u8;
                            break;
                        case K_S8:
                            s1->s8 = c.value.s8;
                            break;
                        case K_B16:
                            s1->b16 = c.value.b16;
                            break;
                        case K_U16:
                            s1->u16 = c.value.u16;
                            break;
                        case K_S16:
                            s1->s16 = c.value.s16;
                            break;
                        case K_B32:
                            s1->b32 = c.value.b32;
                            break;
                        case K_U32:
                            s1->u32 = c.value.u32;
                            break;
                        case K_S32:
                            s1->s32 = c.value.s32;
                            break;
                        case K_B64:
                            s1->b64 = c.value.b64;
                            break;
                        case K_U64:
                            s1->u64 = c.value.u64;
                            break;
                        case K_S64:
                            s1->s64 = c.value.s64;
                            break;
                        case K_F32:
                            s1->f32 = c.value.f32;
                            break;
                        case K_F64:
                            s1->f64 = c.value.f64;
                            break;
                        default:
                        assert(false);
                    }
                }
                else assert(false);
                mptr += size;
            }
        }
        s->typestring = this->string_table->Entry(type);
        s->type = ttype->GetType();
        s->storage_class = storage_class;
        // Add the entry into the symbol table.
        std::pair<char*, SYMBOL*> sym;
        sym.first = s->name;
        sym.second = s;
        symbol_table->symbols.insert(sym);
    }
}


void EMULATOR::SetupGotos(SYMBOL_TABLE * symbol_table, TREE * code)
{
    // Scan ahead and find all labels.  Enter them into the symbol
    // table.
    for (int i = 0; i < code->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)code->GetChild(i);
        if (child->GetType() == TREE_LABEL)
        {
            TREE * label = child->GetChild(0);
            char * name = label->GetText();
            SYMBOL * s = new SYMBOL();
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
            std::pair<char*, SYMBOL*> sym;
            sym.first = s->name;
            sym.second = s;
            symbol_table->symbols.insert(sym);
        }
    }
}

void EMULATOR::SetupExternShared(SYMBOL_TABLE * symbol_table, TREE * code)
{
    // No need to resolve anything if no shared memory to set up.
    if (this->conf.sharedMem == 0)
        return;
    this->extern_memory_buffer = (void*)malloc(conf.sharedMem);
    for (TREE * p = code; p != 0; p = p->GetParent())
    {
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
                SetupSingleVar(symbol_table, var, sc, true);
            }
        }
    }
}

void EMULATOR::ConfigureGrid(dim3 dim)
{
    this->conf.gridDim = dim;
}

void EMULATOR::ConfigureBlock(dim3 dim)
{
    this->conf.blockDim = dim;
}

void EMULATOR::ConfigureSharedMemory(size_t sharedMem)
{
    conf.sharedMem = sharedMem;
}

void EMULATOR::ConfigureStream(cudaStream_t stream)
{
    conf.stream = stream;
}

void EMULATOR::Execute(TREE * entry)
{
    // Get function block.
    TREE * code = FindBlock(entry);

    // Create symbol table for outer blocks.
    SYMBOL_TABLE * obst = PushSymbolTable(0);
    for (TREE * p = code->GetParent()->GetParent(); p != 0; p = p->GetParent())
    {
        int sc[] = { K_GLOBAL, 0};
        SetupVariables(obst, p, sc);
    }

    // Create symbol table for this block.
    SYMBOL_TABLE * block_symbol_table = PushSymbolTable(obst);
    int sc[] = { K_GLOBAL, K_CONST, K_TEX, 0};
    SetupVariables(block_symbol_table, code, sc);
    SetupGotos(block_symbol_table, code);
    SetupParams(block_symbol_table, entry);
    CreateSymbol(block_symbol_table, "%nctaid", "dim3", K_V4, &conf.gridDim, sizeof(conf.gridDim), K_LOCAL);
    CreateSymbol(block_symbol_table, "%ntid", "dim3", K_V4, &conf.blockDim, sizeof(conf.blockDim), K_LOCAL);

    // do_thread_synch is a flag for an optimization.  If false, then the symbol tables for different threads
    // can be shared, thus avoiding extra symbol table creation.  However, sharing the same symbol table for
    // registers is not good for multithreading, so we set this to true for now--until I can figure out a
    // way to get this optimization working again.
    bool do_thread_synch = true;
    // = CodeRequiresThreadSynchronization(code);
    if (this->trace_level > 0)
        std::cout << "Thread synchronization " << (do_thread_synch ? "is" : "is not") << " required.\n";
    for (int bidx = 0; bidx < conf.gridDim.x; ++bidx)
    {
        for (int bidy = 0; bidy < conf.gridDim.y; ++bidy)
        {
            for (int bidz = 0; bidz < conf.gridDim.z; ++bidz)
            {
                ExecuteSingleBlock(block_symbol_table, do_thread_synch, code, bidx, bidy, bidz);
            }
        }
    }
}

bool EMULATOR::CodeRequiresThreadSynchronization(TREE * code)
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

void EMULATOR::ExecuteSingleBlock(SYMBOL_TABLE * symbol_table, bool do_thread_synch, TREE * code, int bidx, int bidy, int bidz)
{
    //_CrtMemState state_begin;
    //_CrtMemCheckpoint(&state_begin);
    
    std::queue<THREAD *> wait_queue;
    std::queue<THREAD *> active_queue;

    // Keep track of symbol table root to restore later.  This is because of the awful
    // use of root on a per-thread basis.
    SYMBOL_TABLE * save = symbol_table;

    // Two ways to do this.  If there is no thread synchronization,
    // then threads can run serially, one after another.  In this case,
    // a symbol table of registers/locals/aligns/params does not have
    // to be created for every thread.  Instead, create this at a more
    // global level.  If the code contains thread synchronization,
    // then create the local symbols for each thread.
    // This test is just for performance enhancement.
    // Create a new symbol table and add the block index variables.
    SYMBOL_TABLE * block_symbol_table = PushSymbolTable(symbol_table);
    dim3 bid(bidx, bidy, bidz);
    CreateSymbol(symbol_table, "%ctaid", "dim3", K_V4, &bid, sizeof(bid), K_LOCAL);

    if (do_thread_synch)
    {
        // Add to this symbol table any explicit shared memory
        // variables.
        int sc[] = { K_SHARED, 0 };
        SetupVariables(symbol_table, code, sc);
    } else
    {
        int sc[] = { K_SHARED, K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
        SetupVariables(symbol_table, code, sc);
    }

    // Add to this symbol table any extern declared shared memory
    // variables.
    SetupExternShared(symbol_table, code);

    for (int tidx = 0; tidx < conf.blockDim.x; ++tidx)
    {
        for (int tidy = 0; tidy < conf.blockDim.y; ++tidy)
        {
            for (int tidz = 0; tidz < conf.blockDim.z; ++tidz)
            {
                SYMBOL_TABLE * root = PushSymbolTable(symbol_table);
                dim3 tid(tidx, tidy, tidz);
                CreateSymbol(root, "%tid", "dim3", K_V4, &tid, sizeof(tid), K_LOCAL);
                if (do_thread_synch)
                {
                    int sc[] = { K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
                    SetupVariables(root, code, sc);
                }
                THREAD * thread = new THREAD(this, code, 0, root);
                wait_queue.push(thread);
            }
        }
    }

    int max_threads = 2;
    int num_waiting_threads = 0;
    while (! wait_queue.empty())
    {
        while (! wait_queue.empty())
        {
            if (active_queue.size() >= max_threads)
                break;
            THREAD * thread = wait_queue.front();
            wait_queue.pop();
            if (! thread->Waiting())
            {
                HANDLE hThread = (HANDLE) _beginthreadex(0, 0, THREAD::WinThreadExecute, (void*)thread, CREATE_SUSPENDED, 0);
                if (hThread)
                {
                    thread->hThread = hThread;
                    ResumeThread(hThread);
                    active_queue.push(thread);
                }
                else printf("error in thread spawn\n");
            }
            else if (! thread->Finished())
            {
                wait_queue.push(thread);
                num_waiting_threads++;
            }
            else
                delete thread;
        }
        // Wait for all active threads to stop.
        while (! active_queue.empty())
        {
            THREAD * thread = active_queue.front();
            active_queue.pop();
            WaitForSingleObject( thread->hThread, INFINITE );
            thread->hThread = 0;
            // Check the status of the threads.
            if (! thread->Finished())
            {
                wait_queue.push(thread);
                num_waiting_threads++;
            }
            else
                delete thread;
        }

        // thread synchronization assumes all threads wait.
        if (num_waiting_threads != 0 && num_waiting_threads == wait_queue.size())
        {
            if (this->trace_level > 0)
                std::cout << "All " << num_waiting_threads << " threads synchronized!\n";
            // all threads waiting.  Reset all threads to not wait.
            for (int i = 0; i < num_waiting_threads; ++i)
            {
                THREAD * t = wait_queue.front();
                wait_queue.pop();
                t->Reset();
                wait_queue.push(t);
            }
            num_waiting_threads = 0;
        }
    }
    // Delete block symbol table.
    delete block_symbol_table;
    if (this->extern_memory_buffer)
        delete this->extern_memory_buffer;
    this->extern_memory_buffer = 0;

    //_CrtMemState state_end;
    //_CrtMemCheckpoint(&state_end);

    //_CrtMemDumpAllObjectsSince(&state_begin);
}

void EMULATOR::PrintName(TREE * inst)
{
    int start = 0;
    if (GetType(GetChild(inst, start)) == TREE_PRED)
        start++;
    std::cout << GetChild(inst, start)->GetText() << "\n";
} 

void EMULATOR::Print(TREE * node, int level)
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


void EMULATOR::CreateSymbol(SYMBOL_TABLE * symbol_table, char * name, char * typestring, int type, void * value, size_t size, int storage_class)
{
    // First find it.
    SYMBOL * s = symbol_table->FindSymbol(name);
    if (s)
    {
        assert(s->size == size);
        // Update value.
        memcpy(s->pvalue, value, size);
        return;
    }
    // Create a symbol table entry.
    s = new SYMBOL();
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
    std::pair<char*, SYMBOL*> sym;
    sym.first = s->name;
    sym.second = s;
    symbol_table->symbols.insert(sym);
}

TREE * EMULATOR::FindBlock(TREE * node)
{
    for (int i = 0; i < (int)node->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)node->GetChild(i);
        if (child->GetType() == TREE_BLOCK)
            return child;
    }
    return 0;
}

int EMULATOR::FindFirstInst(TREE * block, int first)
{
    for (int i = first; i < (int)block->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)block->GetChild(i);
        if (child->GetType() == TREE_INST)
            return i;
    }
    return -1;
}

TREE * EMULATOR::GetInst(TREE * block, int pc)
{
    assert(block->GetType() == TREE_BLOCK);
    TREE * inst = (TREE *)block->GetChild(pc);
    return inst;
}


TREE * EMULATOR::GetChild(TREE * node, int n)
{
    TREE * c = (TREE *)node->GetChild(n);
    return c;
}


char * EMULATOR::StringTableEntry(char * text)
{
    return this->string_table->Entry(text);
}

CONSTANT EMULATOR::Eval(int expected_type, TREE * const_expr)
{
    // Perform bottom-up evaluation of a constant expression.
    CONSTANT result;
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
        // Three cases:
        // "0F...", or "0f..." (hex 32-bit float)
        // "0D...", or "0f..." (hex 64-bit float)
        // "3.14159..." (float with decimal point)
        int len = strlen(text);
        if (len >= 2 && (text[1] == 'f' || text[1] == 'F'))
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
        } else if (len >= 2 && (text[1] == 'd' || text[1] == 'D'))
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
        } else
        {
            switch (expected_type)
            {
                case K_F32:
                    result.value.f32 = strtod(text, &dummy);
                    break;
                case K_F64:
                    result.value.f64 = strtod(text, &dummy);
                    break;
                default:
                    assert(false);
            }
        }
    } else if (GetType(const_expr) == T_QUESTION)
    {
        throw new Unimplemented("Question operator in constant expression not supported.\n");
    } else if (GetType(const_expr) == T_OROR)
    {
        // Perform boolean OR.
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
        CONSTANT rhs = Eval(expected_type, const_expr->GetChild(1));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
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
        CONSTANT lhs = Eval(expected_type, const_expr->GetChild(0));
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

void EMULATOR::unimplemented(bool condition, char * text)
{
    if (condition)
    {
        throw new Unimplemented(text);
    }
}

void EMULATOR::unimplemented(char * text)
{
    throw new Unimplemented(text);
}


void EMULATOR::RunDevice(char * device)
{
    this->device = this->string_table->Entry(device);
}



