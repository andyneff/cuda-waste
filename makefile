clean:
	rm -rf emulator/Debug emulator/Release
	rm -rf hw/Debug hw/Release
	rm -rf ptxp/Debug ptxp/Release
	rm -rf waste/Debug waste/Release
	rm -rf waste-setup/Debug waste-setup/Release
	rm -rf wrapper/Debug wrapper/Release
	rm -f emulator/emulator.vcxproj.user
	rm -f ptxp/ptxp.vcxproj.user
	rm -f waste/waste.vcxproj.user
	rm -f wrapper/wrapper.vcxproj.user
	rm -f waste.suo waste.sdf
	rm -rf Debug Release

build:
	MSBuild waste.sln /p:Configuration=Debug

testd:
	cd Debug; ../test/hw/Debug/hw.exe; ./waste -d compute_20 ../test/hw/Debug/hw.exe; ./waste -d compute_10 ../test/hw/Debug/hw.exe
	cd Debug; ../test/hw/Debug/hw2.exe; ./waste -d compute_20 ../test/hw/Debug/hw2.exe; ./waste -d compute_10 ../test/hw/Debug/hw2.exe
	cd Debug; ../test/hw/Debug/hw3.exe; ./waste -d compute_20 ../test/hw/Debug/hw3.exe; ./waste -d compute_10 ../test/hw/Debug/hw3.exe
	cd Debug; ../test/hw/Debug/hw4.exe; ./waste -d compute_20 ../test/hw/Debug/hw4.exe; ./waste -d compute_10 ../test/hw/Debug/hw4.exe
	cd Debug; ../test/hw/Debug/hw5.exe; ./waste -d compute_20 ../test/hw/Debug/hw5.exe; ./waste -d compute_10 ../test/hw/Debug/hw5.exe
	cd Debug; cp ../test/hw6/VecAdd.ptx .; ../test/hw/Debug/hw6.exe; ./waste -d ompute_20 ../test/hw/Debug/hw6.exe; ./waste -d compute_10 ../test/hw/Debug/hw6.exe
	cd Debug; cp ../test/ptx-inst/ptx-inst/inst.ptx .; ../test/ptx-inst/Debug/ptx-inst.exe; ./waste ../test/ptx-inst/Debug/ptx-inst.exe
	cd Debug; echo "" | ../test/BlackScholes.exe; echo "" | ./waste ../test/BlackScholes.exe
	cd Debug; echo "" | ../test/matrixMul.exe; echo "" | ./waste ../test/matrixMul.exe

testr:
	cd Release; ../test/hw/Debug/hw.exe; ./waste -d compute_20 ../test/hw/Debug/hw.exe; ./waste -d compute_10 ../test/hw/Debug/hw.exe
	cd Release; ../test/hw/Debug/hw2.exe; ./waste -d compute_20 ../test/hw/Debug/hw2.exe; ./waste -d compute_10 ../test/hw/Debug/hw2.exe
	cd Release; ../test/hw/Debug/hw3.exe; ./waste -d compute_20 ../test/hw/Debug/hw3.exe; ./waste -d compute_10 ../test/hw/Debug/hw3.exe
	cd Release; ../test/hw/Debug/hw4.exe; ./waste -d compute_20 ../test/hw/Debug/hw4.exe; ./waste -d compute_10 ../test/hw/Debug/hw4.exe
	cd Release; ../test/hw/Debug/hw5.exe; ./waste -d compute_20 ../test/hw/Debug/hw5.exe; ./waste -d compute_10 ../test/hw/Debug/hw5.exe
	cd Release; cp ../test/hw6/VecAdd.ptx .; ../test/hw/Debug/hw6.exe; ./waste -d ompute_20 ../test/hw/Debug/hw6.exe; ./waste -d compute_10 ../test/hw/Debug/hw6.exe
	cd Release; cp ../test/ptx-inst/ptx-inst/inst.ptx .; ../test/ptx-inst/Debug/ptx-inst.exe; ./waste ../test/ptx-inst/Debug/ptx-inst.exe
	cd Release; echo "" | ../test/BlackScholes.exe; echo "" | ./waste ../test/BlackScholes.exe
	cd Release; echo "" | ../test/matrixMul.exe; echo "" | ./waste ../test/matrixMul.exe

