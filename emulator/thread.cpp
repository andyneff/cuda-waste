#include "emulator.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <queue>
#include "tree.h"
#include <process.h>    /* _beginthread, _endthread */
#include "thread.h"
#include "symbol.h"
#include "symbol-table.h"


THREAD::THREAD(EMULATOR * emulator, TREE * block, int pc, SYMBOL_TABLE * root)
{
	this->emulator = emulator;
	this->block = block;
	this->pc = pc;
	this->root = root;
	this->finished = false;
	this->wait = false;
	this->carry = 0;
}

THREAD::~THREAD()
{
	delete root;
}

unsigned int __stdcall THREAD::WinThreadExecute(void * thr)
{
	THREAD * thread = (THREAD*) thr;
	thread->Execute();
	_endthreadex(0);
	return 0;
}

void THREAD::Execute()
{
    // set up symbol table environment.
	int pc = this->pc;

    // Execute.
	pc = emulator->FindFirstInst(block, pc);
	if (pc < 0)
	{
		this->finished = true;
		return;
	}
	for (;;)
	{
		TREE * inst = this->emulator->GetInst(block, pc);
		if (this->emulator->trace_level > 3)
			this->Dump("before", pc, inst);

		int next = this->Dispatch(inst);
		if (next > 0)
			pc = next;
		else if (next == -KI_EXIT)
		{
			this->finished = true;
			return;
		}
		else if (next == -KI_BAR)
		{
	    // Set state of this thread to wait, and pack up current program counter.
			this->wait = true;
			this->pc = pc + 1;
			return;
		}
		else
			pc++;

		pc = this->emulator->FindFirstInst(block, pc);

		if (this->emulator->trace_level > 2)
			this->Dump("after", pc, inst);
	}
}

bool THREAD::Finished()
{
	return this->finished;
}

void THREAD::Reset()
{
	this->wait = false;
}

bool THREAD::Waiting()
{
	return this->wait;
}

int THREAD::Dispatch(TREE * inst)
{
	if (this->emulator->trace_level > 0)
	{
		this->emulator->PrintName(inst);
		if (this->emulator->trace_level > 1)
			this->emulator->Print(inst, 0);
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
			TREE * t = pred->GetChild(i);
			if (t == 0)
				break;
			int gt = t->GetType();
			if (gt == T_NOT)
				not = true;
			else if (gt == T_WORD)
				tsym = t;
			else assert(false);
		}
		assert(tsym != 0);
		SYMBOL * sym = this->root->FindSymbol(tsym->GetText());
		assert(sym != 0);
		TYPES::Types * s = (TYPES::Types*)sym->pvalue;

		bool test = s->pred;
		if (not)
			test = ! test;
		if (! test)
		{
			if (this->emulator->trace_level > 1)
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
	} catch (EMULATOR::Unimplemented * u)
	{
		std::cout << u->ShowReason() << "\n";
		delete u;
	}
	return -1; // end.
}

void THREAD::Dump(char * comment, int pc, TREE * inst)
{
	std::cout << "\n";
	std::cout << comment << "\n";
	std::cout << "PC = " << pc << "\n";
	this->emulator->Print(inst, 0);
	std::cout << "Symbol tables:\n";
	this->root->Dump();
	std::cout.flush();
}

