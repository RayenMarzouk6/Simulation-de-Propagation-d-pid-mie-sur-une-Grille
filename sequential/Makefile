# ============================================================
#  Makefile — Version Séquentielle
#  Usage : make        → compiler
#          make run    → executer
#          make clean  → nettoyer
# ============================================================

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
TARGET  = epidemic_seq
SRC     = epidemic_seq.c

# ── Compilation ─────────────────────────────────────────────
all: $(TARGET).exe

$(TARGET).exe: $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET).exe $(SRC) -lm
	@echo ">>> Compilation OK : $(TARGET).exe"

# ── Execution ───────────────────────────────────────────────
run: $(TARGET).exe
	@echo ">>> Execution sequentielle..."
	./$(TARGET).exe

# ── Nettoyage ───────────────────────────────────────────────
clean:
	@echo ">>> Nettoyage..."
	del /Q $(TARGET).exe 2>nul || rm -f $(TARGET).exe

.PHONY: all run clean
