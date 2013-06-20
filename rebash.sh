#
echo "Setting and checking the environment for building CUDA Waste."
echo "You can make this faster by setting up your environment manually."
echo

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")	

#############################################################
#
# Set up variables for JAVA/JAVAC.
#
#############################################################

echo "Looking for javac.exe"
javac &> /dev/null
if [ "$?" -gt 1 ]
then
	echo javac.exe not found in path.
	echo -n "Searching ... "
	# well, let's try to find it.
	p=`find "/cygdrive/c/Program Files/Java" -iname javac.exe 2> /dev/null`
	x=""
	for p2 in $p
	do
		if [ -e "$p2" ]
		then
			x="$p2"
		fi
	done
	if [ "$x" == "" ]
	then
		echo No javac.exe found. Please install JDK SE from Oracle.com.
		exit 1
	fi
	# move up the tree to export the path.
	y=${x%%/javac.exe}
	echo "found in "$y
	export PATH="$y:$PATH"
	export CLASSPATH=".;"`cygpath --dos $x`
else
	echo javac.exe found.
fi
echo

#############################################################
#
# Set up SVN on path.
#
#############################################################

echo "Looking for svn.exe"
svn &> /dev/null
if [ "$?" -gt 1 ]
then
	echo svn.exe not found in path.
	echo -n "Searching ... "
	# well, let's try to find it.
	p=`find "/cygdrive/c/" -iname svn.exe 2> /dev/null`
	x=""
	for p2 in $p
	do
		if [ -e "$p2" ]
		then
			x="$p2"
		fi
	done
	if [ "$x" == "" ]
	then
		echo 'No svn.exe found. Please install SVN from TortoiseSVN.net (preferred) or via Cygwin setup.'
		exit 1
	fi
	# move up the tree to export the path.
	y=${x%%/svn.exe}
	echo "found in "$y
	export PATH="$y:$PATH"
else
	echo svn.exe found.
fi
echo

#############################################################
#
# Set up variables for CUDA Toolkit base, if not already set.
#
#############################################################

echo "Checking whether CUDA_PATH is set."
if [ "$CUDA_PATH" == "" ]
then
	echo CUDA_PATH not set.
	echo -n "Searching for NVIDIA GPU Computing Toolkit (CUDA) ..."
	# well, let's try to find it.
	p=`find "/cygdrive/c/" -name 'NVIDIA GPU Computing Toolkit' 2> /dev/null`
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
		echo "No NVIDIA CUDA GPU Toolkit found. Please install the toolkit (http://www.nvidia.com/object/cuda_home_new.html)."
		exit 1
	fi
	echo " found in "$x
	export CUDA_PATH="$x"
else
	echo CUDA_PATH is set to "$CUDA_PATH"
fi


export CUDA_PATH=`cygpath --dos $CUDA_PATH`

# I noticed that CUDA 5.5 doesn't seem to set the CUDA environmental
# variables correctly.  They're assumed to contain a trailing slash.
# check that CUDA_PATH ends in a slash.
y=${CUDA_PATH##*\\}
if [ "$y" != "" ]
then
	echo CUDA_PATH does not end in a slash. Fixing.
	export CUDA_PATH="$CUDA_PATH\\"
fi

# check CUDA_PATH.
if [ -e "$CUDA_PATH" -a -d "$CUDA_PATH" ]
then
	echo CUDA_PATH is valid.
else
	echo CUDA_PATH is set, but does not look to be a directory that exists.
	exit 1
fi

export CUDA_LIB_PATH="${CUDA_PATH%\\}\\lib\\"
export CUDA_INC_PATH="${CUDA_PATH%\\}include\\"
export CUDA_BIN_PATH="${CUDA_PATH%\\}bin\\"

echo

#############################################################
# 
# Set up variable for zlib compression library.
#
#############################################################

if [ "$ZLIB_PATH" == "" ]
then
	echo ZLIB_PATH not set.
	echo -n "Looking ..."
	# well, let's try to find it in downloads.
	p=`find "$HOMEDRIVE/$HOMEPATH/Downloads" -name 'zlib.h' 2> /dev/null`
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
	y="${x%%zlib.h}"
	echo "Found in "$y
	export ZLIB_PATH="$y"
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

echo

#############################################################
# 
# Set up path for Antlr runtime libraries and include files.
#
#############################################################

if [ "$ANTLR_PATH" == "" ]
then
	echo ANTLR_PATH not set.
	echo -n "Looking ..."
	# well, let's try to find it.
	p=`find . -name antlr3.h 2> /dev/null`
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
	echo "Found in "$x
	export ANTLR_PATH="$x"
	echo $ANTLR_PATH
fi

if [ -d $ANTLR_PATH ]
then
	echo Antlr3 is in $ANTLR_PATH.
else
	echo Antlr3 does not exist.
	exit 1
fi

export ANTLR_PATH=`cygpath --dos $ANTLR_PATH`

pwd=`pwd`
export ANTLR_JAR=`cygpath --dos $pwd/ptxp/antlr-3.2.jar`
export CLASSPATH="$ANTLR_JAR;$CLASSPATH"

echo

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
	echo
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
	echo -n "Looking ..."
	# well, let's try to find it.
	p=`find "/cygdrive/c/Program Files (x86)" -name 'Microsoft Visual Studio*' 2> /dev/null`
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
	echo "Found in "$x
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

export VS_PATH=`cygpath --dos $VS_PATH`

echo

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

cmd /k "$VS_PATH\vc\bin\vcvars32.bat" "&&" set CHERE_INVOKING=y "&&" "c:\cygwin\bin\bash.exe" "-i"

