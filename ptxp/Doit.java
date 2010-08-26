import java.util.*;
import org.antlr.runtime.*;
import org.antlr.runtime.tree.*;
import java.io.*;

public class Doit {

    static void print(Tree node, int level)
    {
	for (int i = 0; i < level; ++i)
	    System.out.print("   ");
	System.out.println(node.toString());
	for (int i = 0; i < node.getChildCount(); ++i)
	{
	    Tree child = node.getChild(i);
	    print(child, level+1);
	}
    }
    
    public static void main(String[] args) throws Exception {
	boolean trace = false;
        if (args.length == 0)
            System.out.println("Missing PTX file name.");
        else
	    for (String s: args) {
		if (s.compareTo("-t") == 0)
		{
		    trace = true;
		}
		else {
		    try {
			System.out.println("Input file is " + s);
			File inFile = new File(s);
			String inFileName = inFile.getName();
			System.setIn(new FileInputStream(s));
			ANTLRInputStream input = new ANTLRInputStream(System.in);
			PtxLexer lexer = new PtxLexer(input);
			CommonTokenStream tokens = new CommonTokenStream(lexer);
			List list = tokens.getTokens();
			if (trace)
			    for (int i = 0; i < list.size(); ++i)
			    {
				Token t = (Token)list.get(i);
				if (t == null)
				    break;
				System.out.println("Tok # " + i + ", line " + t.getLine() + ", pos " + t.getCharPositionInLine()
					       + ": " + t.getType() + " '" + t.getText() + "'");
			    }
			PtxParser parser = new PtxParser(tokens);
			PtxParser.prog_return result = parser.prog();
			Tree tree = (Tree)result.getTree();
			if (trace)
			    print(tree, 0);
		    } catch (IOException e) {
			e.printStackTrace();
		    }
		}
            }
    }
}
