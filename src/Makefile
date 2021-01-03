# Stockfish, a UCI chess playing engine derived from Glaurung 2.1
# Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
# Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
# Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
#
# Stockfish is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Stockfish is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


### ==========================================================================
### Section 1. General Configuration
### ==========================================================================

### Establish the operating system name
KERNEL = $(shell uname -s)
ifeq ($(KERNEL),Linux)
	OS = $(shell uname -o)
endif

### Executable name
EXE = cfish

### Installation dir definitions
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin

### Built-in benchmark for pgo-builds
PGOBENCH = ./$(EXE) bench 16 1 15 default depth nnue

### Object files
OBJS = benchmark.o bitbase.o bitboard.o endgame.o evaluate.o main.o \
	material.o misc.o movegen.o movepick.o pawns.o position.o psqt.o \
	search.o tbprobe.o thread.o timeman.o tt.o uci.o ucioption.o \
        numa.o settings.o polybook.o

### ==========================================================================
### Section 2. High-level Configuration
### ==========================================================================
#
# flag                --- Comp switch --- Description
# ----------------------------------------------------------------------------
#
# nnue = yes/no       --- -DNNUE           --- Enable/Disable NNUE
# pure = yes/no       --- -DNNUE_PURE      --- Enable/Disable NNUE pure only
# debug = yes/no      --- -DNDEBUG         --- Enable/Disable debug mode
# optimize = yes/no   --- (-O3/-fast etc.) --- Enable/Disable optimizations
# arch = (name)       --- (-arch)          --- Target architecture
# numa = yes/no       --- -DNUMA           --- Enable NUMA support
# lto = yes/no        --- -flto            --- Enable link-time optimization
# bits = 64/32        --- -DIS_64BIT       --- 64-/32-bit operating system
# prefetch = yes/no   --- -DUSE_PREFETCH   --- Use prefetch asm-instruction
# native = yes/no     --- -march=native    --- Optimize for local CPU
# popcnt = yes/no     --- -DUSE_POPCNT     --- Use popcnt asm-instruction
# pext = yes/no       --- -DUSE_PEXT       --- Use pext x86_64 asm-instruction
# sse = yes/no        --- -msse            --- Use Intel Streaming SIMD Extensions
# mmx = yes/no        --- -mmmx            --- Use Intel MMX instructions
# sse2 = yes/no       --- -msse2           --- Use Intel Streaming SIMD Extensions 2
# ssse3 = yes/no      --- -mssse3          --- Use Intel Supplemental Streaming SIMD Extensions 3
# sse41 = yes/no      --- -msse4.1         --- Use Intel Streaming SIMD Extensions 4.1
# avx = yes/no        --- -mavx            --- Use Intel Advanced Vector Extensions
# avx2 = yes/no       --- -mavx2           --- Use Intel Advanced Vector Extensions 2
# avx512 = yes/no     --- -mavx512bw       --- Use Intel Advanced Vector Extensions 512
# vnni = yes/no       --- -mavx512vnni     --- Use Intel Vector Neural Network Instructions 512
# neon = yes/no       --- -DUSE_NEON       --- Use ARM SIMD architecture
#
# Note that Makefile is space sensitive, so when adding new architectures
# or modifying existing flags, you have to make sure there are no extra spaces
# at the end of the line for flag values.

### 2.1. General and architecture defaults

ifeq ($(ARCH),)
    empty_arch = yes
endif

nnue = yes
pure = no
sparse = yes
optimize = yes
lto = no
debug = no
sanitize = no
numa = yes
bits = 64
prefetch = no
popcnt = no
pext = no
sse = no
mmx = no
sse2 = no
ssse3 = no
sse41 = no
avx = no
avx2 = no
avx512 = no
vnni = no
neon = no
ARCH = auto
native = no
embed = no
STRIP = strip

### 2.2 Architecture specific

ifeq ($(ARCH),auto)

native = yes

ifeq ($(COMP),)
	CC=gcc
else ifeq ($(COMP),gcc)
	CC=gcc
else ifeq ($(COMP),mingw)
	CC=gcc
else ifeq ($(COMP),icc)
	CC=icc
else ifeq ($(COMP),clang)
	CC=clang
endif

ifdef COMPCC
	CC=$(COMPCC)
endif

props = $(shell echo | $(CC) -m64 -march=native -E -dM -)

x86 = no
ifneq ($(findstring __i386__,$(props)),)
	x86 = yes
else ifneq ($(findstring __x86_64__,$(props)),)
	x86 = yes
endif
ifeq ($(x86),yes)

	ifneq ($(findstring __i386__,$(props)),)
		arch = i386
		bits = 32
	else
		arch = x86_64
		bits = 64
	endif
	ifneq ($(findstring __AVX512__,$(props)),)
		avx512 = yes
	endif
	ifneq ($(findstring __VNNI__,$(props)),)
		vnni = yes
	endif
	ifeq ($(bits),64)
	ifneq ($(findstring __BMI2__,$(props)),)
	# Let's not use pext on Zen 1 and Zen 2
	ifeq ($(findstring __znver1,$(props)),)
	ifeq ($(findstring __znver2,$(props)),)
		pext = yes
	endif
	endif
	endif
	endif
	ifneq ($(findstring __AVX2__,$(props)),)
		avx2 = yes
	endif
	ifneq ($(findstring __AVX__,$(props)),)
		avx = yes
	endif
	ifneq ($(findstring __SSE4_1__,$(props)),)
		sse41 = yes
	endif
	ifneq ($(findstring __POPCNT__,$(props)),)
		popcnt = yes
	endif
	ifneq ($(findstring __SSSE3__,$(props)),)
		ssse3 = yes
	endif
	ifneq ($(findstring __SSE2__,$(props)),)
		sse2 = yes
	endif
	ifneq ($(findstring __SSE__,$(props)),)
		sse = yes
		prefetch = yes
	endif
	ifeq ($(sse2),no)
	ifneq ($(findstring __MMX__,$(props)),)
		mmx = yes
	endif
	endif

else ifneq ($(findstring __arm__,$(props)),)
# make ARCH=native not yet supported on ARM
#	ifneq ($(findstring __ARM_NEON__,$(props)),)
#		neon = yes
#	endif
else ifneq ($(findstring __ppc__,$(props)),)
	ifneq ($(findstring __ppc64__,$(props)),)
		arch = ppc64
		popcnt = yes
		prefetch = yes
	else
		arch = ppc
		bits = 32
	endif
endif

endif

ifneq ($(findstring x86,$(ARCH)),)

# x86-32/64

ifneq ($(findstring x86-32,$(ARCH)),)
	arch = i386
	bits = 32
else
	arch = x86_64
	sse = yes
	sse2 = yes
endif

ifneq ($(findstring -sse,$(ARCH)),)
	sse = yes
endif

ifneq ($(findstring -popcnt,$(ARCH)),)
	popcnt = yes
endif

ifneq ($(findstring -mmx,$(ARCH)),)
	mmx = yes
endif

ifneq ($(findstring -sse2,$(ARCH)),)
	sse = yes
	sse2 = yes
endif

ifneq ($(findstring -ssse3,$(ARCH)),)
	sse = yes
	ssse3 = yes
endif

ifneq ($(findstring -sse41,$(ARCH)),)
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
endif

ifneq ($(findstring -modern,$(ARCH)),)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
endif

ifneq ($(findstring -avx,$(ARCH)),)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx = yes
endif

ifneq ($(findstring -avx2,$(ARCH)),)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
endif

ifneq ($(findstring -bmi2,$(ARCH)),)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
endif

ifneq ($(findstring -avx512,$(ARCH)),)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	avx512 = yes
endif

ifneq ($(findstring -vnni,$(ARCH)),)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	vnni = yes
endif

ifeq ($(sse),yes)
	prefetch = yes
endif

# 64-bit pext is not available on x86-32
ifeq ($(bits),32)
	pext = no
endif

# Make sure the ARCH string contains no invalid components
#ifneq ($(shell echo $(ARCH) | sed -E 's/x86-(32|64)(-sse|-popcnt|-mmx|-sse2|-sse3|-ssse3|-sse41|-modern|-avx2|-bmi2|-avx512|-vnni)*/OK/'),OK)
#	arch =
#endif

else

# all other architectures

ifeq ($(ARCH),general-32)
	arch = any
	bits = 32
endif

ifeq ($(ARCH),general-64)
	arch = any
endif

ifeq ($(ARCH),armv7)
	arch = armv7
	prefetch = yes
	bits = 32
	lto = yes
endif

ifeq ($(ARCH),armv7-neon)
	arch = armv7
	prefetch = yes
	popcnt = yes
	neon = yes
	bits = 32
	lto = yes
endif

ifeq ($(ARCH),armv8)
	arch = armv8
	prefetch = yes
	popcnt = yes
	neon = yes
	lto = yes
endif

ifeq ($(ARCH),apple-silicon)
	arch = arm64
	prefetch = yes
	popcnt = yes
	neon = yes
	lto = yes
endif

ifeq ($(ARCH),ppc-32)
	arch = ppc
	bits = 32
endif

ifeq ($(ARCH),ppc-64)
	arch = ppc64
	popcnt = yes
	prefetch = yes
endif

endif

ifeq ($(avx2),yes)
	sparse=no
endif

### ==========================================================================
### Section 3. Low-level configuration
### ==========================================================================

### 3.1 Selecting compiler (default = gcc)

CFLAGS += -Wall -std=c11 $(EXTRACFLAGS)
DEPENDFLAGS += -std=c11
LDFLAGS += $(EXTRALDFLAGS) -lm

ifeq ($(COMP),)
	COMP=gcc
endif

ifeq ($(COMP),gcc)
	comp=gcc
	CC=gcc
	CFLAGS += -pedantic -Wextra -Wshadow
	ifeq ($(ARCH),$(filter $(ARCH),armv7 armv8))
		ifeq ($(OS),Android)
			CFLAGS += -m$(bits)
			LDFLAGS += -m$(bits)
		endif
	else
		CFLAGS += -m$(bits)
		LDFLAGS += -m$(bits)
	endif

	ifeq ($(arch),$(filter $(arch),armv7))
		LDFLAGS += -latomic
	endif

	ifneq ($(KERNEL),Darwin)
		LDFLAGS += -Wl,--no-as-needed
	endif

	ifneq ($(KERNEL),$(filter $(KERNEL),Linux Darwin Haiku))
		CFLAGS += -Wno-pedantic-ms-format
	endif
endif

ifeq ($(COMP),mingw)
	comp=mingw
	CFLAGS += -pedantic -Wextra -Wshadow -Wno-pedantic-ms-format -m$(bits)
	LDFLAGS += -m$(bits)
	ifeq ($(KERNEL),Linux)
		ifeq ($(bits),64)
			ifeq ($(shell which x86_64-w64-mingw32-cc-posix),)
				CC=x86_64-w64-mingw32-gcc
			else
				CC=x86_64-w64-mingw32-cc-posix
			endif
		else
			ifeq ($(shell which i686-w64-mingw32-cc-posix),)
				CC=i686-w64-mingw32-gcc
			else
				CC=i686-w64-mingw32-cc-posix
			endif
		endif
	else
		CC=gcc
	endif
endif

ifeq ($(COMP),icc)
	comp=icc
	CC=icc
	CFLAGS += -diag-disable 344,711,2259,2330 -Wcheck -Wabi -Wdeprecated -strict-ansi
endif

ifeq ($(COMP),clang)
	comp=clang
	CC=clang
	CFLAGS += -pedantic -Wextra -Wshadow

	ifneq ($(KERNEL),Darwin)
	ifneq ($(KERNEL),OpenBSD)
		LDFLAGS += -latomic
	endif
	endif

	ifeq ($(arch),$(filter $(arch),armv7 armv8))
		ifeq ($(OS),Android)
			CFLAGS += -m$(bits)
			LDFLAGS += -m$(bits)
		endif
	else
		CFLAGS += -m$(bits)
		LDFLAGS += -m$(bits)
	endif
endif

ifneq ($(COMP),mingw)
ifeq ($(KERNEL),$(filter $(KERNEL),Linux Darwin Haiku))
	CFLAGS += -D_DEFAULT_SOURCE -D_BSD_SOURCE
endif
endif

ifeq ($(KERNEL),Darwin)
	CFLAGS += -arch $(arch) -mmacosx-version-min=10.14
	LDFLAGS += -arch $(arch) -mmacosx-version-min=10.14
	XCRUN = xcrun
endif

# To cross-compile for Android, NDK version r21 or later is recommended.
# In earlier NDK versions, you'll need to pass -fno-addrsig if using GNU
# binutils. Currently we don't know how to make PGO builds with the NDK yet.
ifeq ($(COMP),ndk)
	CFLAGS += -fPIE
	comp=clang
	ifeq ($(arch),armv7)
		CC=armv7a-linux-androideabi16-clang
		CFLAGS += -mthumb -march=armv7-a -mfloat-abi=softfp -mfpu=neon
		STRIP=arm-linux-androideabi-strip
	endif
	ifeq ($(arch),armv8)
		CC=aarch64-linux-android21-clang
		STRIP=aarch64-linux-android-strip
	endif
	LDFLAGS += -static -latomic -z muldefs
endif

### Allow overwriting CC from command line
ifdef COMPCC
	CC=$(COMPCC)
endif

ifeq ($(comp),icc)
	profile_make = icc-profile-make
	profile_use = icc-profile-use
else ifeq ($(comp),clang)
	profile_make = clang-profile-make
	profile_use = clang-profile-use
else
	profile_make = gcc-profile-make
	profile_use = gcc-profile-use
endif

### Sometimes gcc is really clang
ifeq ($(COMP),gcc)
	gccversion = $(shell $(CC) --version)
	gccisclang = $(findstring clang,$(gccversion))
	ifneq ($(gccisclang),)
		profile_make = clang-profile-make
		profile_use = clang-profile-use
	endif
endif

### On mingw use Windows threads, otherwise POSIX
ifneq ($(comp),mingw)
	# On Android Bionic's C library comes with its own pthread implementation bundled in
	ifneq ($(OS),Android)
		# Haiku has pthreads in its libroot, so only link it in on other platforms
		ifneq ($(KERNEL),Haiku)
			ifneq ($(COMP),ndk)
				LDFLAGS += -lpthread
			endif
		endif
	endif
endif

### 3.2 Debugging
ifeq ($(debug),no)
	CFLAGS += -DNDEBUG
else
	CFLAGS += -g
endif

ifneq ($(sanitize),no)
	CFLAGS += -g3 -fsanitize=$(sanitize)
	LDFLAGS += -fsanitize=$(sanitize)
endif

### 3.3 Optimization
ifeq ($(optimize),yes)

	CFLAGS += -O3

	ifeq ($(comp),$(filter $(comp),gcc mingw))
		ifeq ($(extra),yes)
			CFLAGS += -fira-loop-pressure -fconserve-stack -fmodulo-sched -fmodulo-sched-allow-regmoves -fsched-pressure -flimit-function-alignment -fno-tree-pre
		endif

		ifeq ($(OS),Android)
			CFLAGS += -fno-gcse -mthumb -march=armv7-a -mfloat-abi=softfp
		endif
	endif

	ifeq ($(comp),$(filter $(comp),gcc clang icc))
		ifeq ($(KERNEL),Darwin)
			CFLAGS += -mdynamic-no-pic
		endif
	endif
endif

### 3.4 Bits
ifeq ($(bits),64)
	CFLAGS += -DIS_64BIT
endif

### 3.5 prefetch and sse
ifeq ($(prefetch),no)
	CFLAGS += -DNO_PREFETCH
endif

### 3.6 popcnt
ifeq ($(popcnt),yes)
	ifeq ($(arch),$(filter $(arch),ppc64 armv7 armv8 arm64))
		CFLAGS += -DUSE_POPCNT
	else
	ifeq ($(comp),icc)
		CFLAGS += -msse3 -DUSE_POPCNT
	else
		CFLAGS += -msse3 -mpopcnt -DUSE_POPCNT
	endif
	endif
endif

### 3.7 pext
ifeq ($(pext),yes)
	CFLAGS += -DUSE_PEXT
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -msse4 -mbmi2
	endif
endif

### 3.8 vector instructions
ifeq ($(avx2),yes)
	CFLAGS += -DUSE_AVX2
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mavx2
	endif
else ifeq ($(avx),yes)
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mavx
	endif
endif

ifeq ($(avx512),yes)
	CFLAGS += -DUSE_AVX512
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mavx512f -mavx512bw
	endif
endif

ifeq ($(vnni),yes)
	CFLAGS += -DUSE_VNNI
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mavx512f -mavx512bw -mavx512vnni -mavx512dq -mavx512vl
	endif
endif

ifeq ($(sse41),yes)
	CFLAGS += -DUSE_SSE41
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -msse4.1
	endif
endif

ifeq ($(ssse3),yes)
	CFLAGS += -DUSE_SSSE3
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mssse3
	endif
endif

ifeq ($(sse2),yes)
	CFLAGS += -DUSE_SSE2
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -msse2
	endif
endif

ifeq ($(sse),yes)
	CFLAGS += -msse -DUSE_SSE
endif

ifeq ($(mmx),yes)
	CFLAGS += -DUSE_MMX
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mmmx
	endif
endif

ifeq ($(neon),yes)
	CFLAGS += -DUSE_NEON
	ifeq ($(KERNEL),Linux)
	ifneq ($(COMP),ndk)
	ifneq ($(arch),armv8)
		CFLAGS += -mfpu=neon
	endif
	endif
	endif
endif

ifeq ($(native),yes)
	ifneq ($(findstring ppc,$(arch)),)
		CFLAGS += -mcpu=native -mtune=native
	else ifneq ($(findstring arm,$(arch)),)
		CFLAGS += -march=native -mtune=native
	else
		CFLAGS += -march=native
	endif
endif

### numa
ifeq ($(numa),yes)
	CFLAGS += -DNUMA
        ifeq ($(KERNEL),Linux)
        ifneq ($(comp),mingw)
		LDFLAGS += -lnuma
        endif
        endif
endif

### NNUE
ifeq ($(nnue),yes)
	CFLAGS += -DNNUE
	OBJS += nnue.o
	ifeq ($(embed),yes)
		CFLAGS += -DNNUE_EMBEDDED
	endif
	ifeq ($(pure),yes)
		CFLAGS += -DNNUE_PURE
	endif
	ifeq ($(sparse),yes)
		CFLAGS += -DNNUE_SPARSE
	endif
endif

### 3.9 Link Time Optimization
### This is a mix of compile and link time options because the lto link phase
### needs access to the optimization flags.
ifeq ($(lto),yes)
ifeq ($(optimize),yes)
ifeq ($(debug),no)
	ifeq ($(comp),clang)
		CFLAGS += -flto
		ifneq ($(findstring MINGW,$(KERNEL)),)
			CFLAGS += -fuse-ld=lld
		else ifneq ($(findstring MSYS,$(KERNEL)),)
			CFLAGS += -fuse-ld=lld
		endif
		LDFLAGS += $(CFLAGS)

# GCC and CLANG  user different methods for parallelizing LTO, and CLANG
# pretends to be GCC on some systems.
	else ifeq ($(comp),gcc)
	ifeq ($(gccisclang),)
		CFLAGS += -flto
		LDFLAGS += $(CFLAGS) -flto=jobserver
		ifneq ($(findstring MINGW,$(KERNEL)),)
			LDFLAGS += -save-temps
		else ifneq ($(findstring MSYS,$(KERNEL)),)
			LDFLAGS += -save-temps
		endif
	else
		CFLAGS += -flto
		LDFLAGS += $(CFLAGS)
	endif

# To use LTO and static linking on windows, the tool chain requires a recent
# gcc: gcc version 10.1 in msys2 or TDM-GCC version 9.2 are known to work,
# older might not.
	else ifeq ($(comp),mingw)
		CFLAGS += -flto
		LDFLAGS += $(CFLAGS) -flto=jobserver
	endif
endif
endif
endif

### 3.9 Android 5 can only run position independent executables. Note that this
### breaks Android 4.0 and earlier.
ifeq ($(arch),armv7)
	CFLAGS += -fPIE
	LDFLAGS += -fPIE
endif


### ==========================================================================
### Section 4. Public targets
### ==========================================================================

help:
	@echo ""
	@echo "To compile Cfish, type: "
	@echo ""
	@echo "make target [ARCH=arch] [COMP=compiler] [COMPCC=cc]"
	@echo ""
	@echo "Supported targets:"
	@echo ""
	@echo "help                    > Display architecture details"
	@echo "build                   > Standard build"
	@echo "net                     > Download the default nnue net"
	@echo "profile-build (or pgo)  > PGO build"
	@echo "strip                   > Strip executable"
	@echo "install                 > Install executable"
	@echo "clean                   > Clean up"
	@echo ""
	@echo "Supported archs:"
	@echo ""
	@echo "auto                    > Auto-detect optimal architecture (default)"
	@echo "x86-64-avx512-vnni      > x86 64-bit with 512-bit AVX512/VNNI support"
	@echo "x86-64-avx512           > x86 64-bit with 512-bit AVX512 support"
	@echo "x86-64-vnni             > x86 64-bit with 256-bit AVX2/VNNI support"
	@echo "x86-64-bmi2             > x86 64-bit with AVX2 and BMI2 support"
	@echo "x86-64-avx2             > x86 64-bit with 256-bit AVX2 support"
	@echo "x86-64-avx              > x86 64-bit with AVX support (VEX encoding)"
	@echo "x86-64-sse41-popcnt     > x86 64-bit with SSE4.1 and popcount support"
	@echo "x86-64-modern           > common modern CPU, currently x86-64-sse41-popcnt"
	@echo "x86-64-ssse3            > x86 64-bit with SSSE3 support"
	@echo "x86-64-sse3-popcnt      > x86 64-bit with SSE3 and popcount support"
	@echo "x86-64                  > x86 64-bit generic (with SSE2 support)"
	@echo "x86-32-avx512-vnni      |"
	@echo "x86-32-...              | same as for x86-64"
	@echo "x86-32-sse3-popcnt      |"
	@echo "x86-32-sse2             > x86 32-bit with SSE2 support"
	@echo "x86-32-mmx-sse          > x86 32-bit with MMX and SSE support"
	@echo "x86-32-mmx              > x86 32-bit with MMX support"
	@echo "x86-32                  > x86 32-bit generic"
	@echo "ppc-64                  > PPC 64-bit"
	@echo "ppc-32                  > PPC 32-bit"
	@echo "armv8                   > ARMv8 64-bit with popcnt and neon"
	@echo "armv7-neon              > ARMv7 32-bit with popcnt and neon"
	@echo "armv7                   > ARMv7 32-bit"
	@echo "apple-silicon           > Apple silicon ARM64"
	@echo "general-64              > unspecified 64-bit"
	@echo "general-32              > unspecified 32-bit"
	@echo ""
	@echo "When using many threads, 256-bit AVX2 may perform better than 512-bit AVX512."
	@echo ""
	@echo "Supported compilers:"
	@echo ""
	@echo "gcc                     > Gnu compiler (default)"
	@echo "mingw                   > Gnu compiler with MinGW under Windows"
	@echo "clang                   > LLVM Clang compiler"
	@echo "icc                     > Intel compiler"
	@echo ""
	@echo "Advanced examples:"
	@echo "make profile-build ARCH=x86-64-bmi2 COMP=gcc COMPCC=gcc-9.0"
	@echo "make build ARCH=x86-64-ssse3 COMP=clang"
	@echo ""
	@echo "For best performance, try one or both of:"
	@echo "make pgo"
	@echo "make pgo COMP=clang"
	@echo ""
ifneq ($(empty_arch),yes)
	@echo "-------------------------------"
	@echo "The selected architecture $(ARCH) will enable the following configuration: "
	@$(MAKE) ARCH=$(ARCH) COMP=$(COMP) config-sanity
endif


.PHONY: help build profile-build strip install clean net objclean profileclean \
        config-sanity icc-profile-use icc-profile-make gcc-profile-use \
        gcc-profile-make clang-profile-use clang-profile-make pgo

build: net config-sanity
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) all

profile-build: net config-sanity objclean profileclean
	@echo ""
	@echo "Step 1/4. Building instrumented executable ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) $(profile_make)
	@echo ""
	@echo "Step 2/4. Running benchmark for pgo-build ..."
	$(PGOBENCH) > /dev/null
	@echo ""
	@echo "Step 3/4. Building optimized executable ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) objclean
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) $(profile_use)
	@echo ""
	@echo "Step 4/4. NOT deleting profile data ..."
#	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) profileclean

pgo: profile-build

strip:
	$(STRIP) $(EXE)

install:
	-mkdir -p -m 755 $(BINDIR)
	-cp $(EXE) $(BINDIR)
	-strip $(BINDIR)/$(EXE)

clean: objclean profileclean
	@rm -f .depend core

net:
	$(eval nnuenet := $(shell grep DefaultEvalFile evaluate.h | sed 's/.*\(nn-[a-z0-9]\{12\}.nnue\).*/\1/'))
	@echo "Default net: $(nnuenet)"
	$(eval nnuedownloadurl := https://tests.stockfishchess.org/api/nn/$(nnuenet))
	$(eval curl_or_wget := $(shell if hash curl 2>/dev/null; then echo "curl -skL"; elif hash wget 2>/dev/null; then echo "wget -qO-"; fi))
	@if test -f "$(nnuenet)"; then echo "Already available."; else echo "Downloading $(nnuedownloadurl)"; $(curl_or_wget) $(nnuedownloadurl) > $(nnuenet); fi
	$(eval shasum_command := $(shell if hash shasum 2>/dev/null; then echo "shasum -a 256 "; elif hash sha256sum 2>/dev/null; then echo "sha256sum "; fi))
	@if [ "$(nnuenet)" != "nn-"`$(shasum_command) $(nnuenet) | cut -c1-12`".nnue" ]; then echo "Failed download or $(nnuenet) corrupted, please delete!"; exit 1; fi

# clean binaries and objects
objclean:
	@rm -f $(EXE) $(EXE).exe *.o

# clean auxiliary profiling files
profileclean:
	@rm -rf profdir
	@rm -f bench.txt *.gcda *.gcno
	@rm -f cfish.profdata *.profraw

default:
	help

### ==========================================================================
### Section 5. Private targets
### ==========================================================================

all: $(EXE) .depend

config-sanity: net
	@echo ""
	@echo "Config:"
	@echo "debug: '$(debug)'"
	@echo "sanitize: '$(sanitize)'"
	@echo "optimize: '$(optimize)'"
	@echo "arch: '$(arch)'"
	@echo "bits: '$(bits)'"
	@echo "kernel: '$(KERNEL)'"
	@echo "os: '$(OS)'"
	@echo "prefetch: '$(prefetch)'"
	@echo "popcnt: '$(popcnt)'"
	@echo "pext: '$(pext)'"
	@echo "sse: '$(sse)'"
	@echo "mmx: '$(mmx)'"
	@echo "sse2: '$(sse2)'"
	@echo "ssse3: '$(ssse3)'"
	@echo "sse41: '$(sse41)'"
	@echo "avx: '$(avx)'"
	@echo "avx2: '$(avx2)'"
	@echo "avx512: '$(avx512)'"
	@echo "vnni: '$(vnni)'"
	@echo "neon: '$(neon)'"
	@echo "native: '$(native)'"
	@echo "embed: '$(embed)'"
	@echo ""
	@echo "Flags:"
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo ""
	@echo "Testing config sanity. If this fails, try 'make help' ..."
	@echo ""
	@test "$(debug)" = "yes" || test "$(debug)" = "no"
	@test "$(sanitize)" = "undefined" || test "$(sanitize)" = "thread" || test "$(sanitize)" = "address" || test "$(sanitize)" = "no"
	@test "$(optimize)" = "yes" || test "$(optimize)" = "no"
	@test "$(arch)" = "any" || test "$(arch)" = "x86_64" || test "$(arch)" = "i386" || \
	 test "$(arch)" = "ppc64" || test "$(arch)" = "ppc" || \
	 test "$(arch)" = "armv7" || test "$(arch)" = "armv8" || test "$(arch)" = "arm64"
	@test "$(bits)" = "32" || test "$(bits)" = "64"
	@test "$(prefetch)" = "yes" || test "$(prefetch)" = "no"
	@test "$(popcnt)" = "yes" || test "$(popcnt)" = "no"
	@test "$(sse)" = "yes" || test "$(sse)" = "no"
	@test "$(mmx)" = "yes" || test "$(mmx)" = "no"
	@test "$(sse2)" = "yes" || test "$(sse2)" = "no"
	@test "$(ssse3)" = "yes" || test "$(ssse3)" = "no"
	@test "$(sse41)" = "yes" || test "$(sse41)" = "no"
	@test "$(avx)" = "yes" || test "$(avx)" = "no"
	@test "$(avx2)" = "yes" || test "$(avx2)" = "no"
	@test "$(pext)" = "yes" || test "$(pext)" = "no"
	@test "$(avx512)" = "yes" || test "$(avx512)" = "no"
	@test "$(vnni)" = "yes" || test "$(vnni)" = "no"
	@test "$(neon)" = "yes" || test "$(neon)" = "no"
	@test "$(native)" = "yes" || test "$(native)" = "no"
	@test "$(embed)" = "yes" || test "$(embed)" = "no"
	@test "$(comp)" = "gcc" || test "$(comp)" = "icc" || test "$(comp)" = "mingw" || test "$(comp)" = "clang" \
	  || test "$(comp)" = "armv7a-linux-androideabi16-clang"  || test "$(comp)" = "aarch64-linux-android21-clang"

$(EXE): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

clang-profile-make:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-fprofile-instr-generate ' \
	EXTRALDFLAGS=' -fprofile-instr-generate' \
	all

clang-profile-use:
	$(XCRUN) llvm-profdata merge -output=cfish.profdata *.profraw
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-fprofile-instr-use=cfish.profdata' \
	EXTRALDFLAGS='-fprofile-use ' \
	all

gcc-profile-make:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-fprofile-generate' \
	EXTRALDFLAGS='-lgcov' \
	all

gcc-profile-use:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-fprofile-use -fno-peel-loops -fno-tracer' \
	EXTRALDFLAGS='-lgcov' \
	all

icc-profile-make:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-prof-gen=srcpos -prof_dir ./profdir' \
	all

icc-profile-use:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-prof_use -prof_dir ./profdir' \
	all

.depend:
	-@$(CC) $(DEPENDFLAGS) -MM $(OBJS:.o=.c) > $@ 2> /dev/null

-include .depend
