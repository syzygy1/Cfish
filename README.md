# Cfish
This is a C port of Stockfish.

## Compiling Cfish
Compiling Cfish requires a working gcc or clang environment. To compile, type:

    make target [ARCH=arch] [COMP=compiler] [COMPCC=gcc-4.8] [further options]

Supported targets:

<table>
<tr><td><code>build (or cfish)</code></td><td>Standard build</td></tr>
<tr><td><code>profile-build</code></td><td>PGO build</td></tr>
<tr><td><code>strip</code></td><td>Strip executable</td></tr>
<tr><td><code>install</code></td><td>Install executable</td></tr>
<tr><td><code>clean</code></td><td>Install executable</td></tr>
</table>

Supported archs (default is `x86-64-modern`):

<table>
<tr><td><code>x86-64</code></td><td>x86 64-bit</td></tr>
<tr><td><code>x86-64-modern</code></td><td>x86 64-bit with popcnt support</td></tr>
<tr><td><code>x86-64-bmi2</code></td><td>x86 64-bit with pext support</td></tr>
<tr><td><code>x86-32</code></td><td>x86 32-bit with SSE support</td></tr>
<tr><td><code>x86-32-old</code></td><td>x86 32-bit fall back for old hardware</td></tr>
<tr><td><code>ppc-64</code></td><td>PPC 64-bit</td></tr>
<tr><td><code>ppc-32</code></td><td>PPC 32-bit</td></tr>
<tr><td><code>armv7</code></td><td>ARMv7 32-bit</td></tr>
<tr><td><code>general-64</code></td><td>unspecified 64-bit</td></tr>
<tr><td><code>general-32</code></td><td>unspecified 32-bit</td></tr>
</table>

Supported compilers:

<table>
<tr><td><code>gcc</code></td><td>Gnu compiler (default)</td></tr>
<tr><td><code>mingw</code></td><td>MinGW Gnu compiler</td></tr>
<tr><td><code>clang</code></td><td>LLVM Clang compiler</td></tr>
<tr><td><code>icc</code></td><td>Intel compiler (untested)</td></tr>
</table>

Further options:

<table>
<tr><td><code>numa=no</code></td><td>Disable NUMA support</td></tr>
<tr><td><code>native=no</code></td><td>Disable -march=native compiler setting</td></tr>
<tr><td><code>lto=yes</code></td><td>Compile with link-time optimization</td></tr>
<tr><td><code>extra=yes</code></td><td>Compile with extra optimization options (gcc-7.x)</td></tr>
</table>

Add `numa=no` to fix the error `cannot find -lnuma`.

Add `native=no` to prevent the executable from being tied to the specific type of your CPU.

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

#### LargePages
Control allocation of the hash table as Large Pages (LP). On Windows this option does not appear if the operating system lacks LP support or if LP has not properly been set up.

#### NUMA
This option only appears on NUMA machines, i.e. machines with two or more CPUS. If this option is set to "on" or "all", Cfish will spread its search threads over all nodes. If the option is set to "off", Cfish will ignore the NUMA architecture of the machine. On Linux, a subset of nodes may be specified on which to run the search threads (e.g. "0-1" or "0,1" to limit the search threads to nodes 0 and 1 out of nodes 0-3).
