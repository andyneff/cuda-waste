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

// The TREE class is used to represent the abstract syntax tree from
// the parse of PTX assembly code.  Antlr is used to do the parsing.
// But, for a number of reasons, I do not want Antlr data structures
// used in the rest of the CUDA emulator.  Instead, the AST is
// converted from the Antlr data structure into this data structure.
class TREE
{
    private:
        // A node in the tree has children.
        std::vector<TREE*> children;

        // Each node has a type.
        int type;

        // Every node has a text representation.  The text
        // representation is just for the node itself, not for the
        // forest.
        char * text;

        // A node can represent a symbol, e.g., a register.  This field is an
        // optimization, which points to the symbol data structure.
        SYMBOL * symbol;

        // The TREE data structure is a general tree.  Every node has a
        // parent.
        TREE * parent;
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
};
