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

	
	