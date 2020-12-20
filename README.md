# Cfish
This is a C port of Stockfish.

## Compiling Cfish
Compiling Cfish requires a working gcc or clang environment. The [MSYS2](https://www.msys2.org/) environment is recommended for compiling Cfish on Windows (see below on how to set up MSYS2).

To compile, type:

    make target [ARCH=arch] [COMP=compiler] [COMPCC=gcc-4.8] [further options]

from the `src` directory. Lists of supported targets, archs and compilers can be viewed by typing `make` or `make help`.

If the `ARCH` variable is not set or is set to `auto`, the Makefile will attempt to determine and use the optimal settings for your system. If this fails with an error or gives unsatisfactory results, you should set the desired architecture manually.

Be aware that a Cfish binary compiled specifically for your machine may not work on other (older) machines. If the binary has to work on multiple machines, set `ARCH` to the architecture that corresponds to the oldest/least capable machine.

Further options:

<table>
<tr><td><code>nnue=no</code></td><td>Do not include NNUE code</td></tr>
<tr><td><code>pure=yes</code></td><td>NNUE pure only (no hybrid or classical mode)</td></tr>
<tr><td><code>sparse=yes/no</code></td><td>Enable/disable NNUE sparse multiplication</td></tr>
<tr><td><code>numa=no</code></td><td>Disable NUMA support</td></tr>
<tr><td><code>lto=yes</code></td><td>Compile with link-time optimization</td></tr>
<tr><td><code>extra=yes</code></td><td>Compile with extra optimization options (gcc-7.x and higher)</td></tr>
</table>

The `sparse` option selects between two different NNUE implementations.
The option `sparse=yes` is likely superior (i.e. higher nps) for ARM-based CPUs, for Intel CPUs that do not support AVX2, and for AMD CPUs before Zen 3 (i.e. Ryzen 5000).

Add `numa=no` if compilation fails with`numa.h: No such file or directory` or `cannot find -lnuma`.

The optimization options currently enabled with `extra=yes` appear to be less effective now that the NNUE code has been added.

## UCI settings

#### Contempt
A positive contempt value lets Cfish evaluate a position more favourably the more material is left on the board.

#### Analysis Contempt
By default, contempt is set to zero during analysis to ensure unbiased analysis. Set this option to White or Black to analyse with contempt for that side.

#### Threads
The number of CPU threads used for searching a position.

#### Hash
The size of the hash table in MB.

#### Clear Hash
Clear the hash table.

#### Ponder
Let Cfish ponder its next move while the opponent is thinking.

#### MultiPV
Output the N best lines when searching. Leave at 1 for best performance.

#### Move Overhead
Compensation for network and GUI delay (in ms).

#### Slow Mover
Increase to make Cfish use more time, decrease to make Cfish use less time.

#### SyzygyPath
Path to the folders/directories storing the Syzygy tablebase files. Multiple directories are to be separated by ";" on Windows and by ":" on Unix-based operating systems. Do not use spaces around the ";" or ":".

Example: `C:\tablebases\wdl345;C:\tablebases\wdl6;D:\tablebases\dtz345;D:\tablebases\dtz6`

#### SyzygyProbeDepth
Minimum remaining search depth for which a position is probed. Increase this value to probe less aggressively.

#### Syzygy50MoveRule
Disable to let fifty-move rule draws detected by Syzygy tablebase probes count as wins or losses. This is useful for ICCF correspondence games.

#### SyzygyProbeLimit
Limit Syzygy tablebase probing to positions with at most this many pieces left (including kings and pawns).

#### SyzygyUseDTM
Use Syzygy DTM tablebases (not yet released).

#### BookFile/BestBookMove/BookDepth
Control PolyGlot book usage.

#### EvalFile
Name of NNUE network file.

#### Use NNUE
By default, Cfish uses NNUE in Stockfish's Hybrid mode, where certain positions are evaluated with the old handcrafted evaluation. Other modes are Pure (NNUE only) and Classical (handcrafted evaluation only).

#### LargePages
Control allocation of the hash table as Large Pages (LP). On Windows this option does not appear if the operating system lacks LP support or if LP has not properly been set up.

#### NUMA
This option only appears on NUMA machines, i.e. machines with two or more CPUs. If this option is set to "on" or "all", Cfish will spread its search threads over all nodes. If the option is set to "off", Cfish will ignore the NUMA architecture of the machine. On Linux, a subset of nodes may be specified on which to run the search threads (e.g. "0-1" or "0,1" to limit the search threads to nodes 0 and 1 out of nodes 0-3).

## How to set up MSYS2
1. Download and install MSYS2 from the [MSYS2](https://www.msys2.org/) website.
2. Open an MSYS2 MinGW 64-bit terminal (e.g. via the Windows Start menu).
3. Install the MinGW 64-bit toolchain by entering `pacman -S mingw-w64-x86_64-toolchain`.
4. Close the MSYS2 MinGW 64-bit terminal and open another.
