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
