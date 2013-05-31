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
#pragma once
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <list>
#include <map>
#include "../ptxp/PtxLexer.h"
//#include "../ptxp/PtxParser.h"
#include "tree.h"
#include <cuda.h>
#include <cuda_runtime.h> // cudaError_t, CUDARTAPI, etc.

class EMULATED_DEVICE;
class SYMBOL_TABLE;

class THREAD
{
    private:
        // Root of AST that thread is executing.
        TREE * block;

        // Current program counter. If "n", then the current
        // instruction is just block->GetChild(n).
        int pc;

        // Boolean to denote if the thread has executed the EXIT
        // instruction.
        bool finished;

        // Boolean to dentoe if teh thread is at a BAR instruction and
        // is back in the wait state.
        bool wait;

        // Points to the topmost symbol table, e.g., registers, that
        // the thread uses.
        SYMBOL_TABLE * root;

        // The "carry" bit.  Generally inaccessible, execept for some
        // ADD and SUB instructions.
        int carry;

        // The owner of the thread.
        EMULATED_DEVICE * emulator;

        // The Windows HANDLE for the thread.  Threads are executed
        // concurrently.
        HANDLE hThread;
        
    public:
        THREAD(EMULATED_DEVICE * emulator, TREE * block, int pc, SYMBOL_TABLE * root);
        ~THREAD();
        static unsigned int __stdcall WinThreadExecute(void * thr); // THREAD * thread
        void Execute();
        bool Finished();
        void Reset();
        bool Waiting();
        void Dump(char * comment, int pc, TREE * inst);
        void SetHandle(HANDLE handle);
        HANDLE GetHandle();
        int Dispatch(TREE * inst);
        int DoAbs(TREE * inst);
        int DoAdd(TREE * inst);
        int DoAddc(TREE * inst);
        int DoAnd(TREE * inst);
        int DoAtom(TREE * inst);
        int DoBar(TREE * inst);
        int DoBfe(TREE * inst);
        int DoBfi(TREE * inst);
        int DoBfind(TREE * inst);
        int DoBra(TREE * inst);
        int DoBrev(TREE * inst);
        int DoBrkpt(TREE * inst);
        int DoCall(TREE * inst);
        int DoClz(TREE * inst);
        int DoCnot(TREE * inst);
        int DoCopysign(TREE * inst);
        int DoCos(TREE * inst);
        int DoCvt(TREE * inst);
        int DoCvta(TREE * inst);
        int DoDiv(TREE * inst);
        int DoEx2(TREE * inst);
        int DoExit(TREE * inst);
        int DoFma(TREE * inst);
        int DoIsspacep(TREE * inst);
        int DoLd(TREE * inst);
        int DoLdu(TREE * inst);
        int DoLg2(TREE * inst);
        int DoMad(TREE * inst);
        int DoMad24(TREE * inst);
        int DoMax(TREE * inst);
        int DoMembar(TREE * inst);
        int DoMin(TREE * inst);
        int DoMov(TREE * inst);
        int DoMul(TREE * inst);
        int DoMul24(TREE * inst);
        int DoNeg(TREE * inst);
        int DoNot(TREE * inst);
        int DoOr(TREE * inst);
        int DoPmevent(TREE * inst);
        int DoPopc(TREE * inst);
        int DoPrefetch(TREE * inst);
        int DoPrefetchu(TREE * inst);
        int DoPrmt(TREE * inst);
        int DoRcp(TREE * inst);
        int DoRed(TREE * inst);
        int DoRem(TREE * inst);
        int DoRet(TREE * inst);
        int DoRsqrt(TREE * inst);
        int DoSad(TREE * inst);
        int DoSelp(TREE * inst);
        int DoSet(TREE * inst);
        int DoSetp(TREE * inst);
        int DoShl(TREE * inst);
        int DoShr(TREE * inst);
        int DoSin(TREE * inst);
        int DoSlct(TREE * inst);
        int DoSqrt(TREE * inst);
        int DoSt(TREE * inst);
        int DoSub(TREE * inst);
        int DoSubc(TREE * inst);
        int DoSuld(TREE * inst);
        int DoSuq(TREE * inst);
        int DoSured(TREE * inst);
        int DoSust(TREE * inst);
        int DoTestp(TREE * inst);
        int DoTex(TREE * inst);
        int DoTrap(TREE * inst);
        int DoTxq(TREE * inst);
        int DoVabsdiff(TREE * inst);
        int DoVadd(TREE * inst);
        int DoVmad(TREE * inst);
        int DoVmax(TREE * inst);
        int DoVmin(TREE * inst);
        int DoVote(TREE * inst);
        int DoVset(TREE * inst);
        int DoVshl(TREE * inst);
        int DoVshr(TREE * inst);
        int DoVsub(TREE * inst);
        int DoXor(TREE * inst);

};

