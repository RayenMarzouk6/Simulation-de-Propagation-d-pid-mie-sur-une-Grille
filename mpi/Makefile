# ============================================================
#  Makefile — Version MPI  (MS-MPI sur Windows)
#  Usage : make           → compiler
#          make run       → executer avec 4 processus (defaut)
#          make run P=8   → executer avec 8 processus
#          make bench     → tester 1, 2, 4, 8 processus
#          make clean     → nettoyer
# ============================================================

CC      = gcc
TARGET  = epidemic_mpi
SRC     = epidemic_mpi.c
P       = 4

# ── Chemins MS-MPI (Windows) ────────────────────────────────
MPI_INC = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"	
MPI_LIB = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

CFLAGS  = -O2 -Wall -Wextra -I$(MPI_INC)
LFLAGS  = -L$(MPI_LIB) -lmsmpi -lm

# ── Compilation ─────────────────────────────────────────────
all: $(TARGET).exe

$(TARGET).exe: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET).exe $(SRC) $(LFLAGS)
	@echo ">>> Compilation OK : $(TARGET).exe"

# ── Execution (P = nombre de processus) ─────────────────────
run: $(TARGET).exe
	@echo ">>> Execution MPI avec $(P) processus..."
	mpiexec -n $(P) $(TARGET).exe

# ── Benchmark : tester plusieurs nombres de processus ───────
bench: $(TARGET).exe
	@echo ">>> Benchmark MPI..."
	@echo "--- 1 processus ---"
	mpiexec -n 1 $(TARGET).exe
	@echo "--- 2 processus ---"
	mpiexec -n 2 $(TARGET).exe
	@echo "--- 4 processus ---"
	mpiexec -n 4 $(TARGET).exe
	@echo "--- 8 processus ---"
	mpiexec -n 8 $(TARGET).exe

# ── Nettoyage ───────────────────────────────────────────────
clean:
	@echo ">>> Nettoyage..."
	del /Q $(TARGET).exe 2>nul || rm -f $(TARGET).exe
#     rm -f $(TARGET).exe
.PHONY: all run bench clean
