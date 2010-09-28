#pragma once

#include <vector>

class SYMBOL;

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
	SYMBOL * GetSymbol();
	void SetSymbol(SYMBOL * symbol);
	TREE * GetParent();
	void SetParent(TREE * parent);
private:
	std::vector<TREE*> children;
	int type;
	char * text;
	SYMBOL * symbol;
	TREE * parent;
};
