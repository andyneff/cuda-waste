
#include "tree.h"

TREE::TREE()
{
    this->parent = 0;
    this->text = 0;
    this->symbol = 0;
    this->type = 0;
}

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

TREE * TREE::GetParent()
{
    return this->parent;
}

void TREE::SetParent(TREE * parent)
{
    this->parent = parent;
}
