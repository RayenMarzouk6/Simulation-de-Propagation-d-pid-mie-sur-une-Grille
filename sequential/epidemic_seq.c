#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N        512
#define T_MAX    200
#define BETA     0.3
#define GAMMA    0.05

#define S 0
#define I 1
#define R 2

int grid[N][N];
int new_grid[N][N];

// ===============================
// INITIALIZATION
// ===============================
void init_grid() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            grid[i][j] = S;

    int cx = N / 2, cy = N / 2;
    for (int i = cx - 2; i <= cx + 2; i++)
        for (int j = cy - 2; j <= cy + 2; j++)
            grid[i][j] = I;
}

// ===============================
// PRINT INITIAL MATRIX
// ===============================
void print_initial_grid() {
    int cx = N / 2, cy = N / 2;

    printf("\n========================================\n");
    printf(" INITIAL INFECTED ZONE (CENTER OF GRID)\n");
    printf("========================================\n\n");

    for (int i = cx - 4; i <= cx + 4; i++) {
        for (int j = cy - 4; j <= cy + 4; j++) {
            char c;
            if      (grid[i][j] == S) c = '.';
            else if (grid[i][j] == I) c = 'X';
            else                      c = 'R';
            printf("%c ", c);
        }
        printf("\n");
    }

    printf("\nLegend:\n");
    printf(". = Susceptible\n");
    printf("X = Infected\n");
    printf("R = Recovered\n");
    printf("========================================\n\n");
}

// ===============================
// COUNT INFECTED NEIGHBORS
// ===============================
int count_infected_neighbors(int row, int col) {
    int count = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = (row + di + N) % N;
            int nj = (col + dj + N) % N;
            if (grid[ni][nj] == I) count++;
        }
    }
    return count;
}

// ===============================
// ONE SIMULATION STEP
// ===============================
void step() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int state = grid[i][j];
            if (state == S) {
                int inf = count_infected_neighbors(i, j);
                double prob = 1.0;
                for (int k = 0; k < inf; k++) prob *= (1.0 - BETA);
                prob = 1.0 - prob;
                double r = (double)rand() / RAND_MAX;
                new_grid[i][j] = (r < prob) ? I : S;
            } else if (state == I) {
                double r = (double)rand() / RAND_MAX;
                new_grid[i][j] = (r < GAMMA) ? R : I;
            } else {
                new_grid[i][j] = R;
            }
        }
    }
    memcpy(grid, new_grid, sizeof(grid));
}

// ===============================
// PRINT STATISTICS
// ===============================
void print_stats(int t) {
    int s = 0, i = 0, r = 0;
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            if      (grid[x][y] == S) s++;
            else if (grid[x][y] == I) i++;
            else                      r++;
        }
    }
    printf("t=%3d | S=%6d | I=%6d | R=%6d\n", t, s, i, r);
}

// ===============================
// MAIN
// ===============================
int main() {
    srand(42);

    init_grid();
    print_initial_grid();
    print_stats(0);

    printf("\n=== EPIDEMIC EVOLUTION ===\n\n");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int t = 1; t <= T_MAX; t++) {
        step();
        if (t % 20 == 0) print_stats(t);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec  - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\n========================================\n");
    printf(" Sequential Simulation Finished\n");
    printf("========================================\n");
    printf("Threads      : 1 (sequential)\n");
    printf("Grid Size    : %d x %d\n", N, N);
    printf("Iterations   : %d\n", T_MAX);
    printf("Execution    : %.4f seconds\n", elapsed);
    printf("========================================\n");

    return 0;
}