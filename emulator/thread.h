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

class CUDA_EMULATOR;

class THREAD
{
	public:
		THREAD(CUDA_EMULATOR * emulator, TREE * block, int pc, SymbolTable * root);
		~THREAD();
		static unsigned int __stdcall WinThreadExecute(void * thr); // THREAD * thread
		void Execute();
		bool Finished();
		void Reset();
		bool Waiting();
		HANDLE hThread;
	private:
		TREE * block;
		int pc;
		bool finished;
		bool wait;
		SymbolTable * root;
		int carry;
		CUDA_EMULATOR * emulator;
	public:
		void Dump(char * comment, int pc, TREE * inst);
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

