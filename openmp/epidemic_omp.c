#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N        512
#define T_MAX    200
#define BETA     0.3
#define GAMMA    0.05

#define S 0
#define I 1
#define R 2

int grid[N][N];
int new_grid[N][N];


// ======================================
// THREAD-SAFE LOCAL RANDOM GENERATOR
// ======================================
static inline double local_rand(unsigned int *seed) {

    *seed = (*seed) * 1664525u + 1013904223u;

    return (*seed) / (double)0xFFFFFFFFu;
}


// ===============================
// INITIALIZATION
// ===============================
void init_grid() {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {

            grid[i][j] = S;
        }
    }

    // Initial infected zone (5x5 center)
    int cx = N / 2;
    int cy = N / 2;

    for (int i = cx - 2; i <= cx + 2; i++) {

        for (int j = cy - 2; j <= cy + 2; j++) {

            grid[i][j] = I;
        }
    }
}


// ===============================
// PRINT INITIAL MATRIX
// ===============================
void print_initial_grid() {

    int cx = N / 2;
    int cy = N / 2;

    printf("\n========================================\n");
    printf(" INITIAL INFECTED ZONE (CENTER OF GRID)\n");
    printf("========================================\n\n");

    for (int i = cx - 4; i <= cx + 4; i++) {

        for (int j = cy - 4; j <= cy + 4; j++) {

            char c;

            if (grid[i][j] == S)
                c = '.';

            else if (grid[i][j] == I)
                c = 'X';

            else
                c = 'R';

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

            if (di == 0 && dj == 0)
                continue;

            int ni = (row + di + N) % N;
            int nj = (col + dj + N) % N;

            if (grid[ni][nj] == I)
                count++;
        }
    }

    return count;
}


// ===============================
// ONE SIMULATION STEP
// ===============================
void step() {

    #pragma omp parallel for collapse(2) schedule(dynamic, 32)
    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {

            // Unique seed per thread + cell
            unsigned int seed =
                (unsigned int)(omp_get_thread_num() * 100003u
                + i * N + j);

            int state = grid[i][j];

            // -------------------------
            // SUSCEPTIBLE
            // -------------------------
            if (state == S) {

                int inf = count_infected_neighbors(i, j);

                double prob = 1.0;

                for (int k = 0; k < inf; k++) {

                    prob *= (1.0 - BETA);
                }

                prob = 1.0 - prob;

                double r = local_rand(&seed);

                new_grid[i][j] = (r < prob) ? I : S;
            }

            // -------------------------
            // INFECTED
            // -------------------------
            else if (state == I) {

                double r = local_rand(&seed);

                new_grid[i][j] = (r < GAMMA) ? R : I;
            }

            // -------------------------
            // RECOVERED
            // -------------------------
            else {

                new_grid[i][j] = R;
            }
        }
    }

    // Copy new state into current grid
    memcpy(grid, new_grid, sizeof(grid));
}


// ===============================
// PRINT STATISTICS
// ===============================
void print_stats(int t) {

    int s = 0;
    int i = 0;
    int r = 0;

    #pragma omp parallel for reduction(+:s,i,r) collapse(2)
    for (int x = 0; x < N; x++) {

        for (int y = 0; y < N; y++) {

            if (grid[x][y] == S)
                s++;

            else if (grid[x][y] == I)
                i++;

            else
                r++;
        }
    }

    printf("t=%3d | S=%6d | I=%6d | R=%6d\n",
           t, s, i, r);
}


// ===============================
// MAIN
// ===============================
int main(int argc, char *argv[]) {

    int nb_threads = (argc > 1) ? atoi(argv[1]) : 4;

    omp_set_num_threads(nb_threads);

    // Initialization
    init_grid();

    // Display initial state
    print_initial_grid();

    // Initial statistics
    print_stats(0);

    printf("\n=== EPIDEMIC EVOLUTION ===\n\n");

    double start = omp_get_wtime();

    // Simulation loop
    for (int t = 1; t <= T_MAX; t++) {

        step();

        if (t % 20 == 0) {

            print_stats(t);
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("\n========================================\n");
    printf(" OpenMP Simulation Finished\n");
    printf("========================================\n");

    printf("Threads      : %d\n", nb_threads);
    printf("Grid Size    : %d x %d\n", N, N);
    printf("Iterations   : %d\n", T_MAX);
    printf("Execution    : %.4f seconds\n", elapsed);

    printf("========================================\n");

    return 0;
}