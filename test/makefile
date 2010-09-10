test:
	cp ../Debug/{waste.exe,waste.pdb,wrapper.dll,wrapper.pdb} .
	./waste -d compute_10 hw.exe > out 2>&1
	./waste -d compute_20 hw.exe >> out  2>&1
	./waste -d compute_10 hw2.exe >> out 2>&1
	./waste -d compute_20 hw2.exe >> out 2>&1
	./waste -d compute_10 hw3.exe >> out 2>&1
	./waste -d compute_20 hw3.exe >> out 2>&1
	./waste -d compute_10 hw4.exe >> out 2>&1
	./waste -d compute_20 hw4.exe >> out 2>&1
	rm {waste.exe,waste.pdb,wrapper.dll,wrapper.pdb}
	diff okout out

clean:
	rm {waste.exe,waste.pdb,wrapper.dll,wrapper.pdb}
