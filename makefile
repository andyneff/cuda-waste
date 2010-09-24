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

	
tests:
	cd Debug; ../test/hw/Debug/hw.exe; ./waste -d compute_20 ../test/hw/Debug/hw.exe; ./waste -d compute_10 ../test/hw/Debug/hw.exe
	cd Debug; ../test/hw/Debug/hw2.exe; ./waste -d compute_20 ../test/hw/Debug/hw2.exe; ./waste -d compute_10 ../test/hw/Debug/hw2.exe
	cd Debug; ../test/hw/Debug/hw3.exe; ./waste -d compute_20 ../test/hw/Debug/hw3.exe; ./waste -d compute_10 ../test/hw/Debug/hw3.exe
	cd Debug; ../test/hw/Debug/hw4.exe; ./waste -d compute_20 ../test/hw/Debug/hw4.exe; ./waste -d compute_10 ../test/hw/Debug/hw4.exe
	cd Debug; ../test/hw/Debug/hw5.exe; ./waste -d compute_20 ../test/hw/Debug/hw5.exe; ./waste -d compute_10 ../test/hw/Debug/hw5.exe
	cd Debug; cp ../test/hw6/VecAdd.ptx .; ../test/hw/Debug/hw6.exe; ./waste -d ompute_20 ../test/hw/Debug/hw6.exe; ./waste -d compute_10 ../test/hw/Debug/hw6.exe
