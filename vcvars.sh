#

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")	

#############################################################
#
# Set up variables for CUDA Toolkit base, if not already set.
#
#############################################################

if [ "$CUDA_PATH" == "" ]
then
	echo CUDA_PATH not set.
	# well, let's try to find it.
	p=`find "/cygdrive/c/" -name 'NVIDIA GPU Computing Toolkit'`
	# now look for directory below that "v...", picking last one listed.
	x=""
	for p2 in $p/CUDA/v*
	do
		echo p2 = $p2
		if [ -d "$p2" ]
		then
			x="$p2"
		fi
	done
	if [ "$x" == "" ]
	then
		echo No CUDA Toolkit found.
		exit 1
	fi
	export CUDA_PATH="$x"
fi

echo CUDA_PATH is "$CUDA_PATH"

# check CUDA_PATH.
if [ -e "$CUDA_PATH" -a -d "$CUDA_PATH" ]
then
	echo CUDA_PATH is set and exists.
else
	echo CUDA_PATH is set, but does not exist.
	exit 1
fi

export CUDA_PATH=`cygpath --dos $CUDA_PATH`


#############################################################
# 
# Set up variable for zlib compression library.
#
#############################################################

if [ "$ZLIB_PATH" == "" ]
then
	echo ZLIB_PATH not set.
	# well, let's try to find it in downloads.
	p=`find "$HOMEDRIVE/$HOMEPATH/Downloads" -name 'zlib.h'`
	# now pick the last one.
	x=""
	for p2 in $p
	do
		if [ -f "$p2" ]
		then
			x="$p2"
		fi
	done
	if [ "$x" == "" ]
	then
		echo Cannot find zlib.
		exit 1
	fi
	export ZLIB_PATH="${x%%zlib.h}"
fi

echo ZLIB_PATH is "$ZLIB_PATH"

# check ZLIB_PATH.
if [ -e "$ZLIB_PATH" -a -d "$ZLIB_PATH" ]
then
	echo ZLIB_PATH is set and exists.
else
	echo ZLIB_PATH is set, but does not exist.
	exit 1
fi

export ZLIB_PATH=`cygpath --dos $ZLIB_PATH`


#############################################################
# 
# Set up path for Antlr runtime libraries and include files.
#
#############################################################

if [ "$ANTLR_PATH" == "" ]
then
	echo ANTLR_PATH not set.
	# well, let's try to find it.
	p=`find . -name antlr3.h`
	# now pick the last one.
	x=""
	for j in $p
	do
		j2=${j%%/include/antlr3.h}
		if [ -d $j2 ]
		then
			x="$j2"
			pushd "$x"
			x="`pwd`"
			popd
		fi
	done
	if [ "$x" == "" ]
	then
		echo Cannot find Antlr3 C libraries.
		exit 1
	fi
	export ANTLR_PATH="$x"
fi

if [ -d $ANTLR_PATH ]
then
	echo Antlr3 is in $ANTLR_PATH.
else
	echo Antlr3 does not exist.
	exit 1
fi

export ANTLR_PATH=`cygpath --dos $ANTLR_PATH`


#############################################################
# 
# Set up path for MSBuild.exe if people want that.
#
#############################################################

# get list of .NET directories.
p="$SYSTEMROOT/Microsoft.NET/Framework/v*"
l=`cygpath --unix $p`

# pick last in list that contains MSBuild.exe
for d in $l
do
	p2=$d/MSBuild.exe
	if [ -d $d -a -f $p2 ]
	then
		msbuild_dir=$d
	fi
done

msbuild_dir=`cygpath --dos $msbuild_dir`

if [ -d $msbuild_dir ]
then
	echo MSBuild.exe is in $msbuild_dir, and will be added to path.
	export PATH="$msbuild_dir:$PATH"
else
	echo MSBuild.exe does not exist.
	exit 1
fi

#############################################################
# 
# Set up a variable for Visual Studio.  The whole environment
# isn't set up, just one in order to boot strap using the
# file vcvars32.bat.
#
#############################################################

if [ "$VS_PATH" == "" ]
then
	echo VS_PATH not set.
	# well, let's try to find it.
	p=`find "/cygdrive/c/Program Files (x86)" -name 'Microsoft Visual Studio*'`
	# now look for directory below that "v...", picking last one listed.
	x=""
	for p2 in $p
	do
		if [ -d $p2/VC ]
		then
			x=$p2
		fi
	done
	if [ "$x" == "" ]
	then
		echo Visual Studio not found.
		exit 1
	fi
	export VS_PATH="$x"
fi

echo VS_PATH is "$VS_PATH"

# check VS_PATH.
if [ -e "$VS_PATH" -a -d "$VS_PATH" ]
then
	echo VS_PATH is set and exists.
else
	echo VS_PATH is set, but does not exist.
	exit 1
fi

vsp=`cygpath --dos $VS_PATH`

#############################################################
#
# Setup an shell with environment.
#
#############################################################

# The following lines are reset, and bash.exe *NOT* invoked with -l
# option.  This is because if you try to do a build from the command
# line, or from devenv.exe, you get an message saying illegal option
# to CL.EXE with TMP.

export TMP="`cygpath --dos $tmp`"
export TEMP="`cygpath --dos $tmp`"

cmd /k "$vsp\vc\bin\vcvars32.bat" "&&" set CHERE_INVOKING=y "&&" "c:\cygwin\bin\bash.exe" "-i"

