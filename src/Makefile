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
PGOBENCH = ./$(EXE) bench 16 1 15

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
# debug = yes/no      --- -DNDEBUG         --- Enable/Disable debug mode
# optimize = yes/no   --- (-O3/-fast etc.) --- Enable/Disable optimizations
# arch = (name)       --- (-arch)          --- Target architecture
# bits = 64/32        --- -DIS_64BIT       --- 64-/32-bit operating system
# prefetch = yes/no   --- -DUSE_PREFETCH   --- Use prefetch asm-instruction
# popcnt = yes/no     --- -DUSE_POPCNT     --- Use popcnt asm-instruction
# sse = yes/no        --- -msse            --- Use Intel Streaming SIMD Extensions
# pext = yes/no       --- -DUSE_PEXT       --- Use pext x86_64 asm-instruction
# native = yes/no     --- -march=native    --- Optimize for local CPU
# numa = yes/no       --- -DNUMA           --- Enable NUMA support
# lto = yes/no        --- -flto            --- Enable link-time optimization
#
# Note that Makefile is space sensitive, so when adding new architectures
# or modifying existing flags, you have to make sure there are no extra spaces
# at the end of the line for flag values.

### 2.1. General and architecture defaults
debug = no
optimize = yes
arch = x86_64
bits = 64
prefetch = yes
popcnt = yes
sse = yes
pext = no
native = yes
numa = yes
lto = no
extra = no

### 2.2 Architecture specific

ifeq ($(ARCH),general-32)
	arch = any
	bits = 32
	prefetch = no
	sse = no
	popcnt = no
endif

ifeq ($(ARCH),x86-32-old)
	arch = i386
	bits = 32
	prefetch = no
	sse = no
	popcnt = no
endif

ifeq ($(ARCH),x86-32)
	arch = i386
	bits = 32
	popcnt = no
endif

ifeq ($(ARCH),general-64)
	arch = any
	prefetch = no
	sse = no
	popcnt = no
endif

ifeq ($(ARCH),x86-64)
	arch = x86_64
	popcnt = no
endif

ifeq ($(ARCH),x86-64-modern)
	arch = x86_64
endif

ifeq ($(ARCH),x86-64-bmi2)
	arch = x86_64
	pext = yes
endif

ifeq ($(ARCH),armv7)
	arch = armv7
	bits = 32
	sse = no
	popcnt = no
endif

ifeq ($(ARCH),ppc-32)
	arch = ppc
	bits = 32
	prefetch = no
	sse = no
	popcnt = no
endif

ifeq ($(ARCH),ppc-64)
	arch = ppc64
	prefetch = no
	sse = no
	popcnt = no
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
	ifeq ($(ARCH),armv7)
		ifeq ($(OS),Android)
			CFLAGS += -m$(bits)
			LDFLAGS += -m$(bits)
		endif
	else
		CFLAGS += -m$(bits)
		LDFLAGS += -m$(bits)
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
#			ifeq ($(shell which x86_64-w64-mingw32-cc-posix),)
				CC=x86_64-w64-mingw32-gcc
#			else
#				CC=x86_64-w64-mingw32-cc-posix
#			endif
		else
			ifeq ($(shell which i686-w64-mingw32-cc-posix),)
				CC=i686-w64-mingw32-cc
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
	CC=icpc
	CFLAGS += -diag-disable 1476,10120 -Wcheck -Wabi -Wdeprecated -strict-ansi
endif

ifeq ($(COMP),clang)
	comp=clang
	CC=clang
	CFLAGS += -pedantic -Wextra -Wshadow -Wno-missing-braces -Wno-missing-field-initializers -Wno-unknown-attributes
	ifeq ($(ARCH),armv7)
		ifeq ($(OS),Android)
			CFLAGS += -m$(bits)
			LDFLAGS += -m$(bits)
		endif
	else
		CFLAGS += -m$(bits)
		LDFLAGS += -m$(bits)
	endif

	ifeq ($(KERNEL),Darwin)
		CFLAGS +=
		DEPENDFLAGS +=
	endif
endif

ifneq ($(COMP),mingw)
ifeq ($(KERNEL),$(filter $(KERNEL),Linux Darwin Haiku))
	CFLAGS += -D_POSIX_C_SOURCE=199506L
endif
endif

ifeq ($(comp),icc)
	profile_make = icc-profile-make
	profile_use = icc-profile-use
else
ifeq ($(comp),clang)
	profile_make = clang-profile-make
	profile_use = clang-profile-use
else
	profile_make = gcc-profile-make
	profile_use = gcc-profile-use
endif
endif

ifeq ($(KERNEL),Darwin)
	CFLAGS += -arch $(arch) -mmacosx-version-min=10.9
	LDFLAGS += -arch $(arch) -mmacosx-version-min=10.9
endif

### Travis CI script uses COMPILER to overwrite CC
ifdef COMPILER
	COMPCC=$(COMPILER)
endif

### Allow overwriting CC from command line
ifdef COMPCC
	CC=$(COMPCC)
endif

### On mingw use Windows threads, otherwise POSIX
ifneq ($(comp),mingw)
	# On Android Bionic's C library comes with its own pthread implementation bundled in
	ifneq ($(arch),armv7)
		# Haiku has pthreads in its libroot, so only link it in on other platforms
		ifneq ($(KERNEL),Haiku)
			LDFLAGS += -lpthread
		endif
	endif
endif

### 3.2 Debugging
ifeq ($(debug),no)
	CFLAGS += -DNDEBUG
else
	CFLAGS += -g
endif

### 3.3 Optimization
ifeq ($(optimize),yes)

	CFLAGS += -O3

	ifeq ($(comp),$(filter $(comp),gcc mingw))
		ifeq ($(extra),yes)
			CFLAGS += -fira-loop-pressure -fconserve-stack -fmodulo-sched -fmodulo-sched-allow-regmoves -fsched-pressure -flimit-function-alignment
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

### 3.5 prefetch
ifeq ($(prefetch),yes)
	ifeq ($(sse),yes)
		CFLAGS += -msse
		DEPENDFLAGS += -msse
	endif
else
	CFLAGS += -DNO_PREFETCH
endif

### 3.6 popcnt
ifeq ($(popcnt),yes)
	ifeq ($(comp),icc)
		CFLAGS += -msse3 -DUSE_POPCNT
	else
		CFLAGS += -msse3 -mpopcnt -DUSE_POPCNT
	endif
endif

### 3.7 pext
ifeq ($(pext),yes)
	CFLAGS += -DUSE_PEXT
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -mbmi2
	endif
endif

### native
ifeq ($(native),yes)
	CFLAGS += -march=native
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

### 3.8 Link Time Optimization, it works since gcc 4.5 but not on mingw under Windows.
### This is a mix of compile and link time options because the lto link phase
### needs access to the optimization flags.
ifeq ($(lto),yes)
ifeq ($(optimize),yes)
ifeq ($(debug),no)
	ifeq ($(comp),$(filter $(comp),gcc clang mingw))
		CFLAGS += -flto
		LDFLAGS += $(CFLAGS)
	endif
endif
endif
endif

### 3.9 Android 5 can only run position independent executables. Note that this
### breaks Android 4.0 and earlier.
ifeq ($(arch),armv7)
	CFLAGS += -fPIE
	LDFLAGS += -fPIE -pie
endif


### ==========================================================================
### Section 4. Public targets
### ==========================================================================

help:
	@echo ""
	@echo "To compile Cfish, type: "
	@echo ""
	@echo "make target ARCH=arch [COMP=compiler] [COMPCC=cc]"
	@echo ""
	@echo "Supported targets:"
	@echo ""
	@echo "build                   > Standard build"
	@echo "profile-build           > PGO build"
	@echo "strip                   > Strip executable"
	@echo "install                 > Install executable"
	@echo "clean                   > Clean up"
	@echo ""
	@echo "Supported archs:"
	@echo ""
	@echo "x86-64                  > x86 64-bit"
	@echo "x86-64-modern           > x86 64-bit with popcnt support"
	@echo "x86-64-bmi2             > x86 64-bit with pext support"
	@echo "x86-32                  > x86 32-bit with SSE support"
	@echo "x86-32-old              > x86 32-bit fall back for old hardware"
	@echo "ppc-64                  > PPC 64-bit"
	@echo "ppc-32                  > PPC 32-bit"
	@echo "armv7                   > ARMv7 32-bit"
	@echo "general-64              > unspecified 64-bit"
	@echo "general-32              > unspecified 32-bit"
	@echo ""
	@echo "Supported compilers:"
	@echo ""
	@echo "gcc                     > Gnu compiler (default)"
	@echo "mingw                   > Gnu compiler with MinGW under Windows"
	@echo "clang                   > LLVM Clang compiler"
	@echo "icc                     > Intel compiler"
	@echo ""
	@echo "Simple examples. If you don't know what to do, you likely want to run: "
	@echo ""
	@echo "make build ARCH=x86-64    (This is for 64-bit systems)"
	@echo "make build ARCH=x86-32    (This is for 32-bit systems)"
	@echo ""
	@echo "Advanced examples, for experienced users: "
	@echo ""
	@echo "make build ARCH=x86-64 COMP=clang"
	@echo "make profile-build ARCH=x86-64-modern COMP=gcc COMPCC=gcc-4.8"
	@echo ""

.PHONY: help build profile-build strip install clean objclean profileclean \
        config-sanity icc-profile-use icc-profile-make gcc-profile-use \
        gcc-profile-make clang-profile-use clang-profile-make

build: config-sanity
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) all

profile-build: config-sanity objclean profileclean
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

strip:
	strip $(EXE)

install:
	-mkdir -p -m 755 $(BINDIR)
	-cp $(EXE) $(BINDIR)
	-strip $(BINDIR)/$(EXE)

clean: objclean profileclean
	@rm -f .depend core

objclean:
	@rm -f $(EXE) $(EXE).exe *.o

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

config-sanity:
	@echo ""
	@echo "Config:"
	@echo "debug: '$(debug)'"
	@echo "optimize: '$(optimize)'"
	@echo "arch: '$(arch)'"
	@echo "bits: '$(bits)'"
	@echo "prefetch: '$(prefetch)'"
	@echo "popcnt: '$(popcnt)'"
	@echo "sse: '$(sse)'"
	@echo "pext: '$(pext)'"
	@echo ""
	@echo "Flags:"
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo ""
	@echo "Testing config sanity. If this fails, try 'make help' ..."
	@echo ""
	@test "$(debug)" = "yes" || test "$(debug)" = "no"
	@test "$(optimize)" = "yes" || test "$(optimize)" = "no"
	@test "$(arch)" = "any" || test "$(arch)" = "x86_64" || test "$(arch)" = "i386" || \
	 test "$(arch)" = "ppc64" || test "$(arch)" = "ppc" || test "$(arch)" = "armv7"
	@test "$(bits)" = "32" || test "$(bits)" = "64"
	@test "$(prefetch)" = "yes" || test "$(prefetch)" = "no"
	@test "$(popcnt)" = "yes" || test "$(popcnt)" = "no"
	@test "$(sse)" = "yes" || test "$(sse)" = "no"
	@test "$(pext)" = "yes" || test "$(pext)" = "no"
	@test "$(lto)" = "yes" || test "$(lto)" = "no"
	@test "$(comp)" = "gcc" || test "$(comp)" = "icc" || test "$(comp)" = "mingw" || test "$(comp)" = "clang"

$(EXE): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

clang-profile-make:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACFLAGS='-fprofile-instr-generate ' \
	EXTRALDFLAGS=' -fprofile-instr-generate' \
	all

clang-profile-use:
	llvm-profdata merge -output=cfish.profdata *.profraw
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

