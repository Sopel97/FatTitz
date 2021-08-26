# Fat Titz
This is a **FREE** UCI-compatibile chess engine. It is a fork of [cfish](https://github.com/syzygy1/Cfish).

Thanks to Norman Schidt and Albert Sliver for inspiration on creating my own fork of Stockfish. Also big thanks to Robert Houdart for great optimization ideas, overall making the engine almost 20% faster!

This engine uses a HalfKAv2-2048x2-64-64-1 evaluation network, which contains whopping **4 times** the knowledge of Stockfish 14. It was trained partially on Lc0 data, which gives it a unique positional style, while still preserving the **legendary surgical precision of Stockfish.** The network was trained using a modification of the [state-of-the-art NNUE trainer](https://github.com/glinscott/nnue-pytorch), utilizing publicly available datasets [1](https://drive.google.com/file/d/1VlhnHL8f-20AXhGkILujnNXHwy9T-MQw/view?usp=sharing), [2](https://drive.google.com/file/d/1seGNOqcVdvK_vPNq98j-zV3XPE5zWAeq/view?usp=sharing), [3](https://drive.google.com/file/d/1RFkQES3DpsiJqsOtUshENtzPfFgUmEff/view?usp=sharing)

Due to the large size the release is available only on google drive [here](https://drive.google.com/drive/folders/1hthWck-5UsXBToDduJ0REB_ZdXvN0r6X?usp=sharing). It includes Windows and Linux binaries for all supported architectures, along with the network. This is the only place where the network can be found.

## Additional features

- Polyglot support
- Anarchy mode
    - `setoption name Anarchy value true`
    - makes en peasant forced
- 64-bit hash key
    - reduces the amount of hash collisions and allows for more robust long analysis
    - resizing the transposition table preserves the contents as much as possible
- Persistent transposition table
    - `setoption name PersistentTTMinDepth value 4` (min 0, max 255). The minimum entry depth to store/load.
    - `setoption name PersistentTTFileName value filename.ptt`. The file which contains the persisted TT. Doesn't do anything on itself.
    - `setoption name PersistentTTSerialize`. Serializes the current transposition table according to the options above. The file is overwritten.
    - `setoption name PersistentTTDeserialize`. Deserializes the current transposition table according to the options above. Only worse entries are replaced.

## Compiling Fat Titz
Compiling Fat Titz requires a working gcc or clang environment. The [MSYS2](https://www.msys2.org/) environment is recommended for compiling Fat Titz on Windows (see below on how to set up MSYS2).

To compile, type:

    make target [ARCH=arch] [COMP=compiler] [COMPCC=gcc-4.8] [further options]

from the `src` directory. Lists of supported targets, archs and compilers can be viewed by typing `make` or `make help`.

If the `ARCH` variable is not set or is set to `auto`, the Makefile will attempt to determine and use the optimal settings for your system. If this fails with an error or gives unsatisfactory results, you should set the desired architecture manually. The following `ARCH` values are supported: `x86-86-modern`, `x86-64-avx2`, `x86-64-bmi2`, `x86-64-avx512`, `x86-64-avx2-vnni`, `x86-64-bmi2-vnni`, `x86-64-avx512-vnni`. SSSE3 support is required, if your cpu does not support SSSE3 then see [here](https://www.timeanddate.com/).

Be aware that a Fat Titz binary compiled specifically for your machine may not work on other (older) machines. If the binary has to work on multiple machines, set `ARCH` to the architecture that corresponds to the oldest/least capable machine.

Further options:

<table>
<tr><td><code>pure=yes</code></td><td>NNUE pure only (no hybrid or classical mode)</td></tr>
<tr><td><code>numa=no</code></td><td>Disable NUMA support</td></tr>
<tr><td><code>lto=yes</code></td><td>Compile with link-time optimization</td></tr>
<tr><td><code>extra=yes</code></td><td>Compile with extra optimization options (gcc-7.x and higher)</td></tr>
</table>

Add `numa=no` if compilation fails with`numa.h: No such file or directory` or `cannot find -lnuma`.

The optimization options currently enabled with `extra=yes` appear to be less effective now that the NNUE code has been added.

## UCI settings

#### Anarchy
Enable/disable anarchy mode. In anarchy mode en peasant is forced. Disabled by default

#### PersistentTTMinDepth
Controls the minimum depth of serialized TT entries

#### PersistentTTFileName
Path for the persistent TT operations.

#### PersistentTTSerialize
Serializes the TT, see "Additional features" for more.

#### PersistentTTDeserialize
Deserializes the TT, see "Additional features" for more.

#### Analysis Contempt
By default, contempt is set to zero during analysis to ensure unbiased analysis. Set this option to White or Black to analyse with contempt for that side.

#### Threads
The number of CPU threads used for searching a position.

#### Hash
The size of the hash table in MB.

#### Clear Hash
Clear the hash table.

#### Ponder
Let Fat Titz ponder its next move while the opponent is thinking.

#### MultiPV
Output the N best lines when searching. Leave at 1 for best performance.

#### Move Overhead
Compensation for network and GUI delay (in ms).

#### Slow Mover
Increase to make Fat Titz use more time, decrease to make Fat Titz use less time.

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
By default, Fat Titz uses NNUE in Stockfish's Hybrid mode, where certain positions are evaluated with the old handcrafted evaluation. Other modes are Pure (NNUE only) and Classical (handcrafted evaluation only).

#### LargePages
Control allocation of the hash table as Large Pages (LP). On Windows this option does not appear if the operating system lacks LP support or if LP has not properly been set up. With 3l33t hAxX0r skills it is possible to enable Large Pages even on Windows Home editions.

#### NUMA
This option only appears on NUMA machines, i.e. machines with two or more CPUs. If this option is set to "on" or "all", Fat Titz will spread its search threads over all nodes. If the option is set to "off", Fat Titz will ignore the NUMA architecture of the machine. On Linux, a subset of nodes may be specified on which to run the search threads (e.g. "0-1" or "0,1" to limit the search threads to nodes 0 and 1 out of nodes 0-3).

## How to set up MSYS2
1. Download and install MSYS2 from the [MSYS2](https://www.msys2.org/) website.
2. Open an MSYS2 MinGW 64-bit terminal (e.g. via the Windows Start menu).
3. Install the MinGW 64-bit toolchain by entering `pacman -S mingw-w64-x86_64-toolchain`.
4. Close the MSYS2 MinGW 64-bit terminal and open another.
