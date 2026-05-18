# ============================================================
#  Makefile — Version OpenMP
#  Usage : make           → compiler
#          make run       → executer avec 4 threads (defaut)
#          make run T=8   → executer avec 8 threads
#          make bench     → tester 1, 2, 4, 8 threads
#          make clean     → nettoyer
# ============================================================

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -fopenmp
TARGET  = epidemic_omp
SRC     = epidemic_omp.c
T       = 4

# ── Compilation ─────────────────────────────────────────────
all: $(TARGET).exe

$(TARGET).exe: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET).exe $(SRC) -lm
	@echo ">>> Compilation OK : $(TARGET).exe"

# ── Execution (T = nombre de threads) ───────────────────────
run: $(TARGET).exe
	@echo ">>> Execution OpenMP avec $(T) threads..."
	./$(TARGET).exe $(T)

# ── Benchmark : tester plusieurs nombres de threads ─────────
bench: $(TARGET).exe
	@echo ">>> Benchmark OpenMP..."
	@echo "--- 1 thread ---"
	./$(TARGET).exe 1
	@echo "--- 2 threads ---"
	./$(TARGET).exe 2
	@echo "--- 4 threads ---"
	./$(TARGET).exe 4
	@echo "--- 8 threads ---"
	./$(TARGET).exe 8

# ── Nettoyage ───────────────────────────────────────────────
clean:
	@echo ">>> Nettoyage..."
	del /Q $(TARGET).exe 2>nul || rm -f $(TARGET).exe

.PHONY: all run bench clean
