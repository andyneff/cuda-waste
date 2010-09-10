
#include "tree.h"

int TREE::GetType()
{
	return type;
}

char * TREE::GetText()
{
	return text;
}

TREE * TREE::GetChild(int index)
{
	if (index >= this->children.size())
		return 0;
	else
		return this->children[index];
}

int TREE::GetChildCount()
{
	return this->children.size();
}

void TREE::AddChild(TREE * child)
{
	this->children.push_back(child);
}

void TREE::SetText(char * text)
{
	this->text = text;
}

void TREE::SetType(int type)
{
	this->type = type;
}
