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
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
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
#include "../wrapper/memdbg.h"

#define new new(_CLIENT_BLOCK,__FILE__, __LINE__)

EMULATOR * EMULATOR::singleton;

CRIT_SECTION EMULATOR::sm_CritSec;

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
    this->num_threads = 2;
}

void EMULATOR::SetTrace(int level)
{
    this->trace_level = level;
}

// In ptxp/driver.cpp.
extern TREE * parse(char * source);

EMULATOR::MOD * EMULATOR::Parse(char * module_name, char * source)
{
    // parse all modules, regardless of module name selected.
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
        return false;
    }
    MOD * module = new MOD();
    module->module_name = this->StringTableEntry(module_name);
    module->tree = mod;
    Extract_From_Tree(module, mod);
    this->modules.push_back(module);
    return module;
}

void EMULATOR::Extract_From_Tree(EMULATOR::MOD * module, TREE * node)
{
    // Traverse the tree and look for key features like entry, func, variable declarations, etc.
    if (node->GetType() == TREE_ENTRY)
    {
        // First child will be name node.
        TREE * word = node->GetChild(0);
        char * name = word->GetText();
        std::pair<char*, TREE *> i;
        i.first = (char*)name;
        i.second = node;
        module->entry.insert(i);
    }
    else if (node->GetType() == TREE_FUNC)
    {
        TREE * word = node->GetChild(0);
        char * name = word->GetText();
        std::pair<char*, TREE *> i;
        i.first = (char*)name;
        i.second = node;
        module->func.insert(i);
    }
    for (int i = 0; i < node->GetChildCount(); ++i)
    {
        TREE * child = node->GetChild(i);
        Extract_From_Tree(module, child);
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
            TREE * param = param_list->GetChild(i);
            TREE * name = param->GetChild(0);
            char * n = this->StringTableEntry(name->GetText());
            TREE * type = param->GetChild(1)->GetChild(0);
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
            s->total_size = 0;
            s->type = type->GetType();
            s->storage_class = K_PARAM;
            symbol_table->EnterSymbol(s);
        }
    }
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
            SetupSingleVar(symbol_table, var, desired_storage_classes, false, 0);
        }
    }
}

void EMULATOR::SetupSingleVar(SYMBOL_TABLE * symbol_table, TREE * var, int * desired_storage_classes, bool externed, size_t total_size)
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
        int ct = c->GetType();
        if (ct == TREE_SPACE)
        {
            TREE * chi = c->GetChild(0);
            storage_class = chi->GetType();
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
            ttype = c->GetChild(0);
            type = ttype->GetText();
            int t = ttype->GetType();
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
                TREE * t = tarray->GetChild(a);
                if (t == 0)
                    break;
                int gt = t->GetType();
                // Look at size information if not external.
                if (externed == false && gt == T_OB)
                {
                    ++a;
                    TREE * n = tarray->GetChild(a);
                    assert(n != 0);
                    if (n->GetType() == T_DEC_LITERAL)
                    {
                        int sz = atoi(n->GetText());
                        total = total * sz;
                    }
                    ++a;
                    TREE * t2 = tarray->GetChild(a);
                    assert(t2 != 0);
                    assert(t2->GetType() == T_CB);
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
            s->name = this->StringTableEntry(full_name);
            s->size = size;
            s->pvalue = (void*)malloc(size);
            s->typestring = this->StringTableEntry(type);
            s->type = ttype->GetType();
            s->storage_class = storage_class;
            s->array = false;
            s->total_size = total_size;
            symbol_table->EnterSymbol(s);
        }
    } else {
        // Create a symbol table entry.
        SYMBOL * s = new SYMBOL();
        s->emulator = this;
        s->name = this->StringTableEntry(name);
        s->size = size;
        // array flag helps in printing, but it works like any other
        // storage.
        s->array = false;
        s->total_size = 0;
        void * ptr = 0;
        // Allocate array if declared as one.
        if (tarray != 0)
        {
            s->array = true;
            s->total_size = size * total;
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
                TREE * t = tinitializer_values->GetChild(a);
                if (t == 0)
                break;
                int gt = t->GetType();
                if (gt == TREE_CONSTANT_EXPR)
                {
                    TREE * n = t->GetChild(0);
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
        s->typestring = this->StringTableEntry(type);
        s->type = ttype->GetType();
		s->storage_class = storage_class;
		s->total_size = total_size;
        // Add the entry into the symbol table.
        symbol_table->EnterSymbol(s);
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
            s->name = this->StringTableEntry(name);
            s->typestring = this->StringTableEntry("label");
            s->type = label->GetType();
            s->size = 0;
            s->pvalue = (void*)i;
            s->storage_class = 0;
            s->array = false;
            s->total_size = 0;
            // Add the entry into the symbol table.
            symbol_table->EnterSymbol(s);
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
				SetupSingleVar(symbol_table, var, sc, true, conf.sharedMem);
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

void EMULATOR::ResetArgs()
{
    for (std::list<arg*>::iterator ia = this->arguments.begin();
        ia != this->arguments.end(); ++ia)
    {
        delete *ia;
    }
    this->arguments.clear();
}


void EMULATOR::Execute(TREE * entry)
{
//    _CrtMemState state_begin;
//    _CrtMemCheckpoint(&state_begin);

    //// Get function block.
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
    delete block_symbol_table;
    delete obst;
    delete this->string_table;

//  _CrtCheckMemory();
//  _CrtMemState state_end;
//    _CrtMemCheckpoint(&state_end);
//  _CrtMemState diff;
//    _CrtMemDumpAllObjectsSince(&state_end);
//  _CrtMemDumpAllObjectsSince(&state_begin);
    this->string_table = new STRING_TABLE();
    this->ResetArgs();
}

bool EMULATOR::CodeRequiresThreadSynchronization(TREE * code)
{
    bool result = false;
    for (int i = 0; i < (int)code->GetChildCount(); ++i)
    {
        TREE * child = (TREE *)code->GetChild(i);
        if (child->GetType() == TREE_INST)
        {
            TREE * inst = child;
            TREE * i = inst->GetChild(0);
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
    CreateSymbol(block_symbol_table, "%ctaid", "dim3", K_V4, &bid, sizeof(bid), K_LOCAL);

    if (do_thread_synch)
    {
        // Add to this symbol table any explicit shared memory
        // variables.
        int sc[] = { K_SHARED, 0 };
        SetupVariables(block_symbol_table, code, sc);
    } else
    {
        int sc[] = { K_SHARED, K_REG, K_LOCAL, K_ALIGN, K_PARAM, 0};
        SetupVariables(block_symbol_table, code, sc);
    }

    // Add to this symbol table any extern declared shared memory
    // variables.
    SetupExternShared(block_symbol_table, code);

    for (int tidx = 0; tidx < conf.blockDim.x; ++tidx)
    {
        for (int tidy = 0; tidy < conf.blockDim.y; ++tidy)
        {
            for (int tidz = 0; tidz < conf.blockDim.z; ++tidz)
            {
                SYMBOL_TABLE * root = PushSymbolTable(block_symbol_table);
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

    bool spawn = true;
    int max_threads = this->num_threads;
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
                if (spawn)
                {
                    //MEMDBG * mem = MEMDBG::Singleton();
                    //mem->WrapModules();
                    HANDLE hThread = (HANDLE) _beginthreadex(0, 0, THREAD::WinThreadExecute, (void*)thread, CREATE_SUSPENDED, 0);
                    if (hThread)
                    {
                        thread->SetHandle(hThread);
                        ResumeThread(hThread);
                        active_queue.push(thread);
                    }
                    else printf("error in thread spawn\n");
                }
                else
                {
                    thread->Execute();
                    if (! thread->Finished())
                    {
                        wait_queue.push(thread);
                        num_waiting_threads++;
                    }
                    else
                        delete thread;
                }
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
        if (! active_queue.empty())
        {
            HANDLE arr[20];
            for (int i = 0; i < active_queue.size(); ++i)
            {
                THREAD * thread = active_queue.front();
                active_queue.pop();
                arr[i] = thread->GetHandle();
                active_queue.push(thread);
            }
            ::WaitForMultipleObjects(
                active_queue.size(),
                arr,
                TRUE,
                INFINITE);
        }
        while (! active_queue.empty())
        {
            THREAD * thread = active_queue.front();
            active_queue.pop();
//            WaitForSingleObject(thread->GetHandle(), INFINITE );
            CloseHandle(thread->GetHandle()); // _endthreadex(0); does not free resources.  Call Closehandle to free.
            //MEMDBG * mem = MEMDBG::Singleton();
            //mem->UnwrapModules();
            //mem->Output();
            thread->SetHandle(0);
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
}

void EMULATOR::PrintName(TREE * inst)
{
    int start = 0;
    if (inst->GetChild(start)->GetType() == TREE_PRED)
        start++;
    std::cout << inst->GetChild(start)->GetText() << "\n";
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
        assert(false);
        return;
    }
    // Create a symbol table entry.
    s = new SYMBOL();
    s->emulator = this;
    s->name = this->StringTableEntry(name);
    s->typestring = this->StringTableEntry(typestring);
    s->type = type;
    s->size = size;
    s->pvalue = (void*)malloc(size);
    s->storage_class = storage_class;
    s->array = false;
    s->total_size = 0;
    memcpy(s->pvalue, value, size);
    // Add the entry into the symbol table.
    symbol_table->EnterSymbol(s);
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


char * EMULATOR::StringTableEntry(char * text)
{
    return this->string_table->Entry(text);
}

CONSTANT EMULATOR::Eval(int expected_type, TREE * const_expr)
{
    // Perform bottom-up evaluation of a constant expression.
    CONSTANT result;
    result = result.Eval(expected_type, const_expr);
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

void EMULATOR::SetEmulationThreads(int i)
{
    this->num_threads = i;
}


