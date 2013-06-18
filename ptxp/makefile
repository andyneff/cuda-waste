java:
	cat Ptx.g | sed 's/=C/=Java/' > newPtx.g
	mv newPtx.g Ptx.g
	java -jar antlr-3.2.jar -version
	java -jar antlr-3.2.jar Ptx.g
	javac -classpath antlr-3.2.jar Doit.java PtxLexer.java PtxParser.java

c:
	cat Ptx.g | sed 's/=Java/=C/' > newPtx.g
	mv newPtx.g Ptx.g
	java -jar antlr-3.2.jar -version
	java -jar antlr-3.2.jar Ptx.g
	echo "Continue building in Visual Studio..."

test:
	java Doit testcases/*.ptx > out 2>&1
	diff out okout

tree:
	java Doit -t testcases/chw.compute_10.ptx > out 2>&1
	
clean:
	rm -f *.class PtxParser.java PtxLexer.java PtxParser.c PtxLexer.c PtxParser.h PtxLexer.h *.tokens elf.o out diffs

