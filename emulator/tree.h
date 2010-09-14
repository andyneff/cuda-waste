#pragma once

#include <vector>

class Symbol;

class TREE
{
public:
	TREE();
	int GetType();
	char * GetText();
	TREE * GetChild(int index);
	int GetChildCount();
	void SetText(char * text);
	void SetType(int type);
	void AddChild(TREE * child);
	Symbol * GetSymbol();
	void SetSymbol(Symbol * symbol);
	TREE * GetParent();
	void SetParent(TREE * parent);
private:
	std::vector<TREE*> children;
	int type;
	char * text;
	Symbol * symbol;
	TREE * parent;
};
