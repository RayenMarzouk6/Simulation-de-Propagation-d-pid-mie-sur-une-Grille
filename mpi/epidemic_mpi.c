#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define N        512
#define T_MAX    200
#define BETA     0.3
#define GAMMA    0.05

#define S 0
#define I 1
#define R 2

// ===============================
// COUNT INFECTED NEIGHBORS
// (travaille sur la grille locale avec halos)
// ===============================
int count_infected(int *grid, int local_i, int local_j,
                   int grid_size, int halo_rows) {
    int cnt = 0;
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0) continue;
            int ni = local_i + di;
            int nj = local_j + dj;
            /* Bords gauche/droit periodiques */
            if (nj < 0)          nj = grid_size - 1;
            if (nj >= grid_size) nj = 0;
            /* Bords haut/bas : geres par les halos MPI */
            if (ni < 0 || ni >= halo_rows) continue;
            if (grid[ni * grid_size + nj] == I) cnt++;
        }
    }
    return cnt;
}

// ===============================
// COMPTER LES CELLULES LOCALES
// ===============================
void count_local(int *grid, int local_rows,
                 int *ls, int *li, int *lr) {
    *ls = 0; *li = 0; *lr = 0;
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < N; j++) {
            int v = grid[i * N + j];
            if      (v == S) (*ls)++;
            else if (v == I) (*li)++;
            else             (*lr)++;
        }
    }
}

// ===============================
// AFFICHER STATS GLOBALES
// ===============================
void print_stats_global(int *local_grid, int local_rows,
                         int t, MPI_Comm comm) {
    int ls, li, lr;
    count_local(local_grid, local_rows, &ls, &li, &lr);

    int gs, gi, gr;
    MPI_Reduce(&ls, &gs, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&li, &gi, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&lr, &gr, 1, MPI_INT, MPI_SUM, 0, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("t=%3d | S=%6d | I=%6d | R=%6d\n", t, gs, gi, gr);
        fflush(stdout);
    }
}

// ===============================
// AFFICHER LA ZONE INITIALE
// ===============================
void print_initial_grid_mpi(int *local_grid, int local_rows,
                              int start_row) {
    int cx = N / 2, cy = N / 2;
    int owns_center = (start_row <= cx && cx < start_row + local_rows);

    if (owns_center) {
        printf("\n========================================\n");
        printf(" INITIAL INFECTED ZONE (CENTER OF GRID)\n");
        printf("========================================\n\n");

        for (int gi = cx - 4; gi <= cx + 4; gi++) {
            int li = (gi - start_row) + 1;
            for (int j = cy - 4; j <= cy + 4; j++) {
                if (li >= 1 && li <= local_rows) {
                    int v = local_grid[li * N + j];
                    char c = (v == S) ? '.' : (v == I) ? 'X' : 'R';
                    printf("%c ", c);
                }
            }
            printf("\n");
        }

        printf("\nLegend:\n");
        printf(". = Susceptible\n");
        printf("X = Infected\n");
        printf("R = Recovered\n");
        printf("========================================\n\n");
        fflush(stdout);
    }
}

// ===============================
// MAIN
// ===============================
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ── Decomposition de domaine par bandes horizontales ── */
    int rows_per_proc = N / size;
    int remainder     = N % size;

    int local_rows, start_row;
    if (rank < remainder) {
        local_rows = rows_per_proc + 1;
        start_row  = rank * (rows_per_proc + 1);
    } else {
        local_rows = rows_per_proc;
        start_row  = remainder * (rows_per_proc + 1)
                   + (rank - remainder) * rows_per_proc;
    }

    int halo_rows   = local_rows + 2;
    int *local_grid = (int*)calloc(halo_rows * N, sizeof(int));
    int *local_new  = (int*)calloc(halo_rows * N, sizeof(int));

    if (!local_grid || !local_new) {
        fprintf(stderr, "[Rank %d] Erreur allocation\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ── Foyer infecte 5x5 au centre ── */
    int cx = N / 2, cy = N / 2;
    for (int i = 0; i < local_rows; i++) {
        int global_row = start_row + i;
        for (int j = 0; j < N; j++) {
            if (abs(global_row - cx) <= 2 && abs(j - cy) <= 2)
                local_grid[(i + 1) * N + j] = I;
        }
    }

    srand(rank + 42);

    /* ── Affichage initial (sequentiel par rang pour eviter melange) ── */
    for (int r = 0; r < size; r++) {
        if (rank == r)
            print_initial_grid_mpi(local_grid, local_rows, start_row);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    print_stats_global(local_grid, local_rows, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== EPIDEMIC EVOLUTION ===\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ══════════════════════════════════════
       Boucle principale de simulation
    ══════════════════════════════════════ */
    double start_time = MPI_Wtime();

    for (int t = 1; t <= T_MAX; t++) {

        /* ── Echange de lignes fantomes (halo swap) ── */
        int up_rank   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
        int down_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        // MPI_Sendrecv(
        //     &local_grid[1 * N],N, MPI_INT, up_rank,   0,
        //     &local_grid[0 * N],N, MPI_INT, up_rank,   0,
        //     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // MPI_Sendrecv(
        //     &local_grid[local_rows * N],       N, MPI_INT, down_rank, 1,
        //     &local_grid[(local_rows + 1) * N], N, MPI_INT, down_rank, 1,
        //     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Exchange top halo */
        MPI_Sendrecv(
            &local_grid[1 * N], N, MPI_INT, up_rank, 0,
            &local_grid[(local_rows + 1) * N], N, MPI_INT, down_rank, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Exchange bottom halo */
        MPI_Sendrecv(
            &local_grid[local_rows * N], N, MPI_INT, down_rank, 1,
            &local_grid[0 * N], N, MPI_INT, up_rank, 1,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* ── Mise a jour des cellules ── */
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < N; j++) {
                int state = local_grid[i * N + j];

                if (state == S) {
                    int inf = count_infected(local_grid, i, j, N, halo_rows);
                    double prob = 1.0;
                    for (int k = 0; k < inf; k++) prob *= (1.0 - BETA);
                    prob = 1.0 - prob;
                    double r = (double)rand() / RAND_MAX;
                    local_new[i * N + j] = (r < prob) ? I : S;
                } else if (state == I) {
                    double r = (double)rand() / RAND_MAX;
                    local_new[i * N + j] = (r < GAMMA) ? R : I;
                } else {
                    local_new[i * N + j] = R;
                }
            }
        }

        /* ── Copier new -> local ── */
        for (int i = 1; i <= local_rows; i++)
            memcpy(&local_grid[i * N], &local_new[i * N], N * sizeof(int));

        /* ── Stats toutes les 20 iterations ── */
        if (t % 20 == 0)
            print_stats_global(local_grid, local_rows, t, MPI_COMM_WORLD);
    }

    double elapsed = MPI_Wtime() - start_time;

    double global_elapsed;
    MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n========================================\n");
        printf(" MPI Simulation Finished\n");
        printf("========================================\n");
        printf("Processes    : %d\n", size);
        printf("Grid Size    : %d x %d\n", N, N);
        printf("Iterations   : %d\n", T_MAX);
        printf("Execution    : %.4f seconds\n", global_elapsed);
        printf("========================================\n");
        fflush(stdout);
    }

    free(local_grid);
    free(local_new);
    MPI_Finalize();
    return 0;
}