# 🦠 Epidemic Spread Simulation — Parallel Programming (SIR Model)

> **ISET Sousse · Département Informatique · CCDAD1.1**
> Module: Parallel Programming — Project 4
> Student: Rayen Marzouk · Year: 2025–2026

---

## 📌 Table of Contents

1. [Project Objective & Importance](#-project-objective--importance)
2. [The SIR Model — COVID-19 Connection](#-the-sir-model--covid-19-connection)
3. [Mathematical Functions Used](#-mathematical-functions-used)
4. [Project Structure](#-project-structure)
5. [Why 512×512 Grid?](#-why-512512-grid)
6. [Sequential Version](#-sequential-version)
7. [OpenMP Version](#-openmp-version-shared-memory)
8. [MPI Version](#-mpi-version-distributed-memory)
9. [CUDA Version](#-cuda-version-gpu)
10. [Random Number Generators: LCG vs rand() vs cuRAND](#-random-number-generators-lcg-vs-rand-vs-curand)
11. [Shared Memory vs Distributed Memory](#-shared-memory-vs-distributed-memory)
12. [Role of Rank 0 (MPI Master)](#-role-of-rank-0-mpi-master)
13. [Why CPU↔GPU Communication?](#-why-cpugpu-communication)
14. [Why Use a Makefile?](#-why-use-a-makefile)
15. [Why Test with 4 Threads/Processes?](#-why-test-with-4-threadsprocesses)
16. [Performance Results & Speedup](#-performance-results--speedup)
17. [How to Run](#-how-to-run)

---

## 🎯 Project Objective & Importance

### What is this project?

This project simulates the **spread of an infectious disease** across a 2D grid of 512×512 = **262,144 cells**, using the **SIR epidemiological model**. Each cell represents an individual who can be:

- **S** — Susceptible (healthy, can get infected)
- **I** — Infected (sick, can spread the disease)
- **R** — Recovered (immune, permanently)

The simulation runs for **200 time steps**, and at each step, every cell checks its 8 neighbors and updates its state based on probabilistic rules.

### Why is this important?

```
┌──────────────────────────────────────────────────────────────────┐
│  512 × 512 = 262,144 cells                                       │
│  × 200 time steps                                                │
│  × 8 neighbors per cell                                          │
│  = over 400 MILLION neighbor checks                              │
│                                                                  │
│  ➜ Sequential computation is too slow — we NEED parallelism!    │
└──────────────────────────────────────────────────────────────────┘
```

This project explores **three parallel programming paradigms**:

| Paradigm | Technology | Memory Model | Target |
|----------|-----------|--------------|--------|
| Shared Memory | OpenMP | Shared RAM | Multi-core CPU |
| Distributed Memory | MPI | Each process has its own | Multi-node / Cluster |
| Massively Parallel | CUDA | GPU VRAM | GPU (thousands of threads) |

Real-world applications of this type of simulation include:
- Modeling COVID-19, influenza, or Ebola spread
- Planning vaccination campaigns
- Predicting hospital saturation
- Testing lockdown policies in silico

---

## 🧬 The SIR Model — COVID-19 Connection

### What is SIR?

SIR is a **compartmental epidemiological model** that divides a population into three groups:

```
     β (infection rate)        γ (recovery rate)
  S ─────────────────────► I ──────────────────► R
  Susceptible            Infected             Recovered
```

### Transition Rules (Cellular Automaton)

Each cell on the grid updates according to:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   S → I :  P(infection) = 1 − (1 − β)^k                       │
│            where k = number of infected neighbors (Moore)      │
│                                                                 │
│   I → R :  P(recovery)  = γ  (each time step)                 │
│                                                                 │
│   R → R :  Permanent immunity (no reinfection)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Moore Neighborhood (8 neighbors)

```
┌───┬───┬───┐
│ ↖ │ ↑ │ ↗ │
├───┼───┼───┤
│ ← │ C │ → │   C = current cell
├───┼───┼───┤
│ ↙ │ ↓ │ ↘ │
└───┴───┴───┘
Every cell always has exactly 8 neighbors
(toroidal/periodic boundary conditions)
```

### Periodic Boundary Conditions (Toroidal Grid)

The grid wraps around like a donut (torus). A corner cell still has 8 neighbors:

```
  Normal plane:              Toroidal grid (our model):
  ┌─────────────┐            ┌─────────────┐
  │ C ? ? ? ? ? │            │ C ──────► C │  Right edge wraps to left
  │ ? . . . . . │     →      │ ↕           ↕ │  Bottom edge wraps to top
  │ ? . . . . . │            │ C ──────► C │
  └─────────────┘            └─────────────┘

  Implementation: ni = (row + di + N) % N
                  nj = (col + dj + N) % N
```

### Parameters Used

| Parameter | Value | Meaning |
|-----------|-------|---------|
| β (beta)  | 0.30  | 30% transmission probability per infected neighbor |
| γ (gamma) | 0.05  | 5% recovery probability per time step |
| N         | 512   | Grid size (N × N) |
| T_max     | 200   | Number of simulation steps |
| Initial focus | 5×5 center | Starting infected zone |

### COVID-19 Connection

The SIR model was used extensively during the COVID-19 pandemic to:
- Estimate the **basic reproduction number R₀** (how many people one person infects)
- Model the effect of **mask mandates** (reducing β)
- Predict the impact of **vaccination** (moving people directly to R)
- Simulate **lockdowns** (reducing neighborhood connections)

In our model: with β=0.3 and γ=0.05, the effective R₀ ≈ β/γ = **6** — meaning one infected cell can theoretically infect 6 others, which is comparable to highly contagious COVID variants.

---

## 📐 Mathematical Functions Used

### 1. Infection Probability

```
P(S → I) = 1 − (1 − β)^k

Where:
  β = 0.3  (transmission rate per neighbor)
  k = number of infected neighbors (0 to 8)

Example:
  k=0 → P = 0        (no infected neighbors, no risk)
  k=1 → P = 0.30     (one infected neighbor)
  k=2 → P = 0.51     (two infected neighbors)
  k=8 → P = 0.9997   (surrounded by infected)
```

### 2. Recovery Probability

```
P(I → R) = γ = 0.05  (per time step)

Average time to recovery = 1/γ = 20 time steps
```

### 3. Speedup (Amdahl's Law)

```
        T_sequential
Sp =  ───────────────
        T_parallel(p)

Results:
  OpenMP (4 threads) : Sp = 0.8354 / 0.2040 ≈ 4.09×
  CUDA               : Sp = 0.8354 / 0.1292 ≈ 6.46×
```

### 4. Parallel Efficiency

```
        Speedup        Sp
Ep =  ──────────── = ──────
       p (workers)     p

OpenMP (4T): E = 4.09 / 4 = 102.3%  ← super-linear (cache effects)
MPI (4P):   E = 2.74 / 4 = 68.6%   ← communication overhead
```

### 5. LCG Random Number Generator

```
X_{n+1} = (a · X_n + c) mod m

Where:
  a = 1,664,525   (multiplier)
  c = 1,013,904,223  (increment)
  m = 2^32        (modulus, implicit via unsigned int overflow)

Implementation:
  *seed = (*seed) * 1664525u + 1013904223u;
  return *seed / (double)0xFFFFFFFFu;
```

---

## 📁 Project Structure

```
project4/
├── sequential/
│   ├── epidemic_seq.c       # Single-threaded baseline
│   └── Makefile
├── openmp/
│   ├── epidemic_omp.c       # OpenMP shared-memory parallelism
│   └── Makefile
├── mpi/
│   ├── epidemic_mpi.c       # MPI distributed-memory parallelism
│   └── Makefile
├── cuda/
│   ├── epidemic_cuda.cu     # CUDA GPU parallelism
│   └── Makefile
└── README.md
```

---

## ❓ Why 512×512 Grid?

The 512×512 grid is not chosen arbitrarily:

```
┌─────────────────────────────────────────────────────────────┐
│  Reason 1 — Sufficient scale for parallel benefit           │
│  262,144 cells = enough work to keep all threads busy.      │
│  A 64×64 grid would finish so fast that overhead would      │
│  dominate and parallelism would be SLOWER.                  │
│                                                             │
│  Reason 2 — Cache-friendly size                             │
│  512 = 2^9 → aligns perfectly with CUDA block sizes        │
│  (multiples of 16 and 32), maximizing GPU efficiency.       │
│                                                             │
│  Reason 3 — Memory budget                                   │
│  512×512×4 bytes (int) = 1 MB per grid                     │
│  Two grids (current + new) = 2 MB → fits in L2/L3 cache   │
│                                                             │
│  Reason 4 — MPI decomposition                               │
│  512 / 4 processes = 128 rows each → perfect integer split  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔢 Sequential Version

### Architecture

```
main()
  │
  ├─► srand(42)              ← Fixed seed → reproducible results
  ├─► init_grid()            ← 5×5 infected focus at center
  ├─► print_initial_grid()   ← Visual ASCII display
  ├─► print_stats(0)         ← t=0 counts
  │
  └─► for t = 1..200
        │
        ├─► step()           ← Core computation (read grid → write new_grid)
        │     │
        │     ├─► for each cell (i,j):
        │     │     ├─ count_infected_neighbors(i,j)   [8 neighbors]
        │     │     ├─ compute P(S→I) = 1-(1-β)^k
        │     │     ├─ draw random r = rand()/RAND_MAX
        │     │     └─ write new_grid[i][j]
        │     │
        │     └─► memcpy(grid, new_grid)    ← Atomic update (why 2 grids?)
        │
        └─► if (t % 20 == 0) print_stats(t)
```

### Why Two Grids?

```
  Without 2 grids (WRONG):         With 2 grids (CORRECT):
  ┌────┬────┬────┐                 ┌────┬────┬────┐
  │ S  │ I  │ S  │  Cell (0,1)     │ S  │ I  │ S  │  Read from grid[t]
  └────┴────┴────┘  gets infected  └────┴────┴────┘
  ↓ cell (0,0) reads (0,1)=I ✓    ↓ writes to new_grid[t+1]
  ↓ but cell (0,2) reads           ┌────┬────┬────┐
    ALREADY-MODIFIED (0,1)=I ✗     │ I  │ I  │ I  │  Write to new_grid[t+1]
                                   └────┴────┴────┘
  → Cascade effect, wrong physics  → All cells see state at time t ✓
```

### Key Functions

| Function | Role |
|----------|------|
| `init_grid()` | Sets all cells to S, places 5×5 I focus at center |
| `count_infected_neighbors(i,j)` | Counts I cells in Moore neighborhood (with periodic BC) |
| `step()` | Iterates all cells, applies transition rules, swaps grids |
| `print_stats(t)` | Counts and prints S/I/R totals at time t |

---

## 🔀 OpenMP Version (Shared Memory)

### Parallelization Strategy

```
  Grid 512×512 (shared memory — one process)
  ┌─────────────────────────────────────────┐
  │  Thread 0   │  rows   0 → 127           │
  ├─────────────────────────────────────────┤
  │  Thread 1   │  rows 128 → 255           │
  ├─────────────────────────────────────────┤
  │  Thread 2   │  rows 256 → 383           │
  ├─────────────────────────────────────────┤
  │  Thread 3   │  rows 384 → 511           │
  └─────────────────────────────────────────┘
  All threads share the same grid[] and new_grid[] arrays.
  No communication needed — just synchronization at memcpy.
```

### Key OpenMP Directives

```c
// Parallelize the double loop (collapse merges i and j into one range)
#pragma omp parallel for collapse(2) schedule(dynamic, 32)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) { ... }

// Parallel reduction for counting S/I/R (no race condition)
#pragma omp parallel for reduction(+:s,i,r) collapse(2)
```

### Thread-Safe Random Numbers (LCG)

Since `rand()` uses a **global shared state**, calling it from multiple threads simultaneously causes **race conditions**. The solution: each thread has its own local LCG seed.

```c
// Each cell gets a UNIQUE seed based on thread ID + position
unsigned int seed = (unsigned int)(omp_get_thread_num() * 100003u + i * N + j);

// LCG: fast, portable, no shared state
*seed = (*seed) * 1664525u + 1013904223u;
return *seed / (double)0xFFFFFFFFu;
```

```
  Thread 0: seed = 0*100003 + i*N + j  →  LCG₀ → LCG₀ → LCG₀
  Thread 1: seed = 1*100003 + i*N + j  →  LCG₁ → LCG₁ → LCG₁
  Thread 2: seed = 2*100003 + i*N + j  →  LCG₂ → LCG₂ → LCG₂
  Thread 3: seed = 3*100003 + i*N + j  →  LCG₃ → LCG₃ → LCG₃
                   │
                   └─ No sharing, no contention, fully independent sequences
```

---

## 🌐 MPI Version (Distributed Memory)

### Domain Decomposition

Each MPI process owns a **horizontal band** of rows, plus two **halo rows** (ghost rows) borrowed from neighboring processes:

```
  Global 512×512 grid                Local grid of each process
  ┌──────────────────┐               ┌──────────────────────┐
  │  P0: rows 0-127  │               │  halo top  (from P-1)│  ← ghost row
  ├──────────────────┤               ├──────────────────────┤
  │  P1: rows 128-255│    each  →    │  real rows           │
  ├──────────────────┤    process    │  (128 rows)          │
  │  P2: rows 256-383│    stores:    ├──────────────────────┤
  ├──────────────────┤               │  halo bottom(from P+1)│ ← ghost row
  │  P3: rows 384-511│               └──────────────────────┘
  └──────────────────┘
```

### Halo Exchange (Ghost Row Swap)

Before each simulation step, processes exchange their border rows:

```
  P0 sends its last real row  →  received as halo top    of P1
  P1 sends its first real row →  received as halo bottom of P0

  MPI_Sendrecv(
      &local_grid[1*N],           N, MPI_INT, up_rank,   0,   // send top
      &local_grid[(local_rows+1)*N], N, MPI_INT, down_rank, 0, // recv bottom
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  ┌───────┐    MPI_Sendrecv    ┌───────┐
  │  P0   │ ←────────────────► │  P1   │
  │row 127│                    │row 128│
  └───────┘                    └───────┘
     halo bottom of P0 = row 128 of P1
     halo top of P1    = row 127 of P0
```

### Global Statistics with MPI_Reduce

```c
MPI_Reduce(&local_S, &global_S, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//          each process           all summed    rank 0 collects result
```

```
  P0: ls=65000  ──┐
  P1: ls=65500  ──┼── MPI_Reduce(SUM) ──► P0 has gs = 262119
  P2: ls=66000  ──┤
  P3: ls=65619  ──┘
```

### Key Functions

| Function | Role |
|----------|------|
| `MPI_Init` / `MPI_Finalize` | Initialize and close MPI environment |
| `MPI_Comm_rank` | Get this process's ID (0 to size-1) |
| `MPI_Comm_size` | Get total number of processes |
| `MPI_Sendrecv` | Exchange halo rows with neighbors (non-deadlocking) |
| `MPI_Reduce` | Aggregate statistics to rank 0 |
| `MPI_Barrier` | Synchronize all processes at a point |
| `MPI_Wtime` | High-precision wall-clock timer |

---

## 🚀 CUDA Version (GPU)

### GPU Architecture

```
  CPU (Host)                        GPU (Device)
  ┌──────────────────┐              ┌──────────────────────────────────────┐
  │  Control logic   │              │           GRID of Blocks             │
  │  malloc/free     │              │  ┌────┬────┬────┬─ ─ ─┬────┐       │
  │  cudaMemcpy H→D  │──────────►   │  │B00 │B01 │B02 │ ... │B0n │       │
  │  Launch kernels  │              │  ├────┼────┼────┼─ ─ ─┼────┤       │
  │  cudaMemcpy D→H  │◄──────────   │  │B10 │B11 │B12 │ ... │B1n │       │
  └──────────────────┘              │  ├────┼────┼────┼─ ─ ─┼────┤       │
                                    │  │... │    │    │      │... │       │
                                    │  └────┴────┴────┴─ ─ ─┴────┘       │
                                    │                                      │
                                    │  Each block = 16×16 = 256 threads   │
                                    │  Grid = 32×32 = 1,024 blocks        │
                                    │  Total = 262,144 threads             │
                                    │  One thread per cell!                │
                                    └──────────────────────────────────────┘
```

### Thread → Cell Mapping

```
  dim3 threads(16, 16);                    // 256 threads per block
  dim3 blocks((N+15)/16, (N+15)/16);       // 32×32 = 1024 blocks

  Inside kernel:
  int x = blockIdx.x * blockDim.x + threadIdx.x;   // row
  int y = blockIdx.y * blockDim.y + threadIdx.y;   // column

  Block (2,3), Thread (5,7):
  x = 2*16 + 5 = 37
  y = 3*16 + 7 = 55
  → handles cell (37, 55)
```

### Kernel Logic Flow

```
  epidemic_step<<<blocks, threads>>>(d_grid, d_new_grid, d_states)
       │
       ├─ Each thread computes (x, y) from blockIdx + threadIdx
       ├─ Guard: if (x >= N || y >= N) return;    ← boundary check
       ├─ Read state = grid[x*N + y]
       │
       ├─ state == S:
       │    count infected neighbors (with periodic BC)
       │    prob = 1 - (1-BETA)^infected
       │    r = curand_uniform(&localState)
       │    new_grid[idx] = (r < prob) ? I : S
       │
       ├─ state == I:
       │    r = curand_uniform(&localState)
       │    new_grid[idx] = (r < GAMMA) ? R : I
       │
       └─ state == R:
            new_grid[idx] = R                    ← permanent immunity
```

### Memory Flow

```
  CPU RAM                            GPU VRAM
  ┌────────────┐   cudaMemcpy H→D   ┌──────────────┐
  │ h_grid     │──────────────────► │ d_grid       │
  │ (1 MB)     │                    │ d_new_grid   │
  └────────────┘                    │ d_states     │
         ▲                          └──────┬───────┘
         │       cudaMemcpy D→H            │ kernel runs
         └────────────────────────────────┘ 200 times
```

### cuRAND Initialization

```c
__global__ void init_rand(curandState *states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);  // unique sequence per thread
}
// Each of 262,144 threads gets its own independent RNG state
```

---

## 🎲 Random Number Generators: LCG vs rand() vs cuRAND

### Why does it matter?

In a stochastic simulation, each cell draws a random number every step. With 262,144 cells × 200 steps = **52 million random draws**. The generator must be:
- **Fast** — not a bottleneck
- **Thread-safe** — no race conditions in parallel code
- **Independent** — each thread gets different sequences

### Comparison

```
  rand()                     LCG (local)               cuRAND
  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
  │ Global state    │        │ Local per-thread│        │ GPU-native      │
  │ shared by all   │        │ state variable  │        │ per-thread state│
  │ threads         │        │                 │        │                 │
  │                 │        │ Xₙ₊₁ = aXₙ+c  │        │ XORWOW / MRG32 │
  │ ✗ NOT safe for  │        │                 │        │                 │
  │   multi-thread  │        │ ✓ Thread-safe   │        │ ✓ GPU-safe      │
  │ ✓ OK sequential │        │ ✓ Portable      │        │ ✓ High quality  │
  │ ✓ OK MPI        │        │ ✓ Ultra-fast    │        │ ✗ GPU only      │
  │   (own memory)  │        │                 │        │                 │
  └─────────────────┘        └─────────────────┘        └─────────────────┘
  Sequential, MPI only       OpenMP                     CUDA
```

| Generator | Thread-safe | Portable | Used in | Note |
|-----------|------------|----------|---------|------|
| `rand()` | ❌ No | ✅ Yes | Sequential, MPI | Global state — dangerous in OpenMP |
| `rand_r(seed)` | ✅ Yes | POSIX | OpenMP option | Moderate quality |
| LCG local | ✅ Yes | ✅ Yes | **OpenMP** | Fast, portable, no contention |
| `cuRAND` | ✅ Yes | GPU only | **CUDA** | Best quality, GPU-native |

---

## 💾 Shared Memory vs Distributed Memory

```
  SHARED MEMORY (OpenMP)              DISTRIBUTED MEMORY (MPI)
  ┌────────────────────────┐          ┌──────┐  ┌──────┐  ┌──────┐
  │     One Process        │          │  P0  │  │  P1  │  │  P2  │
  │  ┌─────────────────┐   │          │      │  │      │  │      │
  │  │   Shared RAM    │   │          │ RAM₀ │  │ RAM₁ │  │ RAM₂ │
  │  │  ┌──┬──┬──┬──┐  │   │          └──┬───┘  └──┬───┘  └──┬───┘
  │  │  │T0│T1│T2│T3│  │   │             │          │          │
  │  │  └──┴──┴──┴──┘  │   │             └──────────┴──────────┘
  │  └─────────────────┘   │                    Network / MPI
  └────────────────────────┘
  All threads read/write               Each process owns its own
  the SAME grid[] array.               memory partition.
  → No data copy needed                → Must SEND data to neighbors
  → Risk: race conditions              → No risk: nothing shared
  → Limited to 1 machine               → Scales to 1000s of machines
```

| Feature | OpenMP (Shared) | MPI (Distributed) |
|---------|----------------|-------------------|
| Memory | One shared pool | Each process has its own |
| Communication | None (direct read) | Explicit message passing |
| Scalability | Limited to 1 node | Unlimited (HPC clusters) |
| Complexity | Simple (pragma) | Higher (halos, reduce) |
| Best for | Single workstation | Supercomputers, clusters |

---

## 👑 Role of Rank 0 (MPI Master)

In MPI, processes don't know about each other — they must coordinate. **Rank 0** acts as the **master** (coordinator):

```
  ALL PROCESSES compute in parallel:
  ┌──────────────────────────────────────────────┐
  │  P0, P1, P2, P3 all run epidemic_step()      │
  │  on their own grid band simultaneously        │
  └──────────────────────────────────────────────┘
                          │
                    MPI_Reduce (sum S/I/R)
                          │
                          ▼
  ONLY RANK 0 does:
  ┌──────────────────────────────────────────────┐
  │  • Receives aggregated statistics (S, I, R)  │
  │  • Prints results to console                 │
  │  • Prints initial infected zone display      │
  │  • Measures and prints total elapsed time    │
  │  • Prints final simulation summary           │
  └──────────────────────────────────────────────┘
```

### How Rank 0 is used in the code:

```c
// Check if current process is rank 0
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Only rank 0 prints
if (rank == 0) {
    printf("t=%3d | S=%6d | I=%6d | R=%6d\n", t, gs, gi, gr);
}

// All processes contribute to the reduction
MPI_Reduce(&local_S, &global_S, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//                                                       ↑
//                                               root = rank 0 receives result
```

Without rank 0 coordination, all processes would print simultaneously → garbled output!

---

## 🔄 Why CPU↔GPU Communication?

### The GPU Cannot Work Alone

```
  GPU CANNOT:                          CPU MUST HANDLE:
  ┌─────────────────────────┐          ┌─────────────────────────┐
  │ ✗ Read files            │          │ ✓ Load initial grid data │
  │ ✗ Print to terminal     │          │ ✓ Print statistics       │
  │ ✗ Access CPU RAM        │          │ ✓ Manage loop control    │
  │ ✗ Manage the OS         │          │ ✓ Transfer data to GPU   │
  │ ✗ Run host code         │          │ ✓ Retrieve results       │
  └─────────────────────────┘          └─────────────────────────┘
```

### Communication Flow

```
  STEP 1: CPU initializes grid (S, I, R values)
          h_grid[i*N+j] = S/I/R ...

  STEP 2: CPU sends grid to GPU
          cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice)
          ──────────────────────────────────►
          CPU RAM                            GPU VRAM

  STEP 3: GPU runs 200 simulation steps
          epidemic_step<<<blocks, threads>>>(...)  ×200
          (CPU waits with cudaDeviceSynchronize())

  STEP 4: CPU retrieves final results
          cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost)
          ◄──────────────────────────────────
          CPU RAM                            GPU VRAM

  STEP 5: CPU counts S/I/R and prints summary
```

The GPU is like a **co-processor**: blazing fast at parallel math, but needs the CPU to set up the problem and collect the output.

---

## 🛠 Why Use a Makefile?

```
  WITHOUT Makefile:                    WITH Makefile:
  ┌─────────────────────────────┐      ┌─────────────────────────────┐
  │ $ gcc -O2 -fopenmp           │      │ $ make                       │
  │   epidemic_omp.c -o          │      │  → compiles automatically    │
  │   epidemic_omp               │      │                              │
  │                              │      │ $ make run T=8               │
  │ $ "/c/Program Files/...      │      │  → runs with 8 threads       │
  │   mpiexec.exe" -n 4 ...      │      │                              │
  │   epidemic_mpi.exe           │      │ $ make bench                 │
  │                              │      │  → runs 1,2,4,6 processes    │
  │  (error-prone, repetitive)   │      │    automatically             │
  └─────────────────────────────┘      └─────────────────────────────┘
```

A Makefile provides:
- **Reproducibility** — same flags every time
- **Convenience** — `make`, `make run`, `make bench`, `make clean`
- **Dependency tracking** — only recompiles what changed
- **Documentation** — self-documenting build system

---

## ⚙️ Why Test with 4 Threads/Processes?

### The Rule: Match Your CPU Core Count

```
  Your CPU has N physical cores
  ┌─────────────────────────────────────────────────────────┐
  │  Optimal thread count = number of physical CPU cores    │
  └─────────────────────────────────────────────────────────┘

  Example (quad-core CPU):
  ┌──────┬──────┬──────┬──────┐
  │Core 0│Core 1│Core 2│Core 3│  ← 4 cores
  │  T0  │  T1  │  T2  │  T3  │  ← 4 threads (perfect fit)
  └──────┴──────┴──────┴──────┘
```

### What happens if you exceed your core count?

```
  8 threads on a 4-core CPU:
  ┌──────┬──────┬──────┬──────┐
  │Core 0│Core 1│Core 2│Core 3│
  │ T0+T4│ T1+T5│ T2+T6│ T3+T7│  ← Context switching overhead!
  └──────┴──────┴──────┴──────┘
  
  The OS must time-share: T0 runs, pauses, T4 runs, pauses...
  → MORE threads does NOT mean MORE speed beyond core count
  → Often SLOWER due to scheduling and cache contention
```

### Recommendation

```bash
# Check your core count first
nproc                           # Linux
sysctl -n hw.ncpu               # macOS
wmic cpu get NumberOfCores      # Windows

# Then run with that number
./epidemic_omp <core_count>
mpiexec -n <core_count> epidemic_mpi.exe

# ⚠️ If you exceed physical core count, performance DEGRADES
```

---

## 📊 Performance Results & Speedup

| Version | Time (s) | Speedup | Efficiency | Platform |
|---------|---------|---------|-----------|----------|
| Sequential | 0.8354 | 1.00× | 100% | 1 CPU core |
| OpenMP (4T) | 0.2040 | **4.09×** | 102.3% | 4 CPU cores |
| MPI (1P) | 0.9099 | 0.92× | — | 1 process (overhead) |
| MPI (2P) | 0.4748 | 1.76× | 88% | 2 processes |
| MPI (4P) | 0.3043 | 2.74× | 68.6% | 4 processes |
| MPI (6P) | 0.2798 | 2.98× | 49.7% | 6 processes |
| CUDA | 0.1292 | **6.46×** | — | GPU (262K threads) |

```
  Execution Time (seconds)
  1.0 ┤
      │  ██  Sequential: 0.835s
  0.8 ┤  ██
      │  ██
  0.6 ┤  ██  MPI 1P: 0.910s
      │  ██  ██
  0.4 ┤  ██  ██  MPI 2P: 0.475s
      │  ██  ██  ██
  0.2 ┤  ██  ██  ██  OpenMP 4T: 0.204s  MPI 6P: 0.280s
      │  ██  ██  ██  ██  MPI 4P: 0.304s ██
  0.1 ┤  ██  ██  ██  ██  ██  ██  CUDA: 0.129s
      │  ██  ██  ██  ██  ██  ██  ██
  0.0 └──────────────────────────────────────────
      Seq  MPI1 MPI2 MPI4 MPI6 OMP4 CUDA
```

---

## 🚀 How to Run

### Sequential

```bash
cd sequential
make
make run
```

### OpenMP

```bash
cd openmp
make
make run T=4      # 4 threads
make run T=8      # 8 threads
make bench        # benchmark all thread counts
```

### MPI

```bash
cd mpi
make
make run P=4      # 4 processes
make bench        # benchmark 1,2,4,6 processes
```

### CUDA (Google Colab)

```python
# Upload epidemic_cuda.cu to Colab (GPU runtime required)
!nvcc -O2 -arch=sm_70 epidemic_cuda.cu -lcurand -o epidemic_cuda
!./epidemic_cuda
```

---

## 📚 References

- Anderson, R.M. & May, R.M. (1991). *Infectious Diseases of Humans*. Oxford University Press.
- Kermack, W.O. & McKendrick, A.G. (1927). *A Contribution to the Mathematical Theory of Epidemics*. Proc. Roy. Soc. London A, 115, 700–721.
- OpenMP Architecture Review Board. *OpenMP API Specification v5.2*.
- MPI Forum. *MPI: A Message-Passing Interface Standard v4.1*.
- NVIDIA Corporation. *CUDA Programming Guide v12*.
- Ferguson et al. (2020). *Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality*. Imperial College Report 9.

---

*Project 4 — Parallel Programming · ISET Sousse · 2025–2026*
*Rayen Marzouk · CCDAD1.1*
