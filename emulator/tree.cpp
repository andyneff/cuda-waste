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
