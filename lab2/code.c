#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#define PRECISION 0.000001
#define RANGESIZE 1

#define DATA 0
#define OFFSET 1

#define RET_DATA 2
#define RET_OFFSET 3

#define RESULT 1
#define FINISH 2

#define X_DATA_SIZE 700
#define Y_DATA_SIZE X_DATA_SIZE
#define DATA_SIZE (X_DATA_SIZE * Y_DATA_SIZE)

#define CHUNK_SIZE 2

static void process_data(double *data, unsigned w) {
	int i;
	for (i = 0; i < w; i++) {
		data[i] *= 2;
	}
}

static void prepare_rand_data(double data[][Y_DATA_SIZE]) {
		int i, j;
		for (i = 0; i < X_DATA_SIZE; i++) {
			for (j = 0; j < Y_DATA_SIZE; j++) {
				data[i][j] = rand() % DATA_SIZE;
			}
		}
}

static void prepare_chunk_from_data(int *x, int *y, double *dst,
		double data[][Y_DATA_SIZE])
{
	int j = 0;
	while (j < CHUNK_SIZE &&
			*x < X_DATA_SIZE && *y < Y_DATA_SIZE)
	{
		dst[j++] = data[*x][*y];
		if (++(*y) >= Y_DATA_SIZE) {
			*y = 0;
			(*x)++;
		}
	}
}

static void retrive_data_from_chunk(int offset, double *src,
		double dst[][Y_DATA_SIZE])
{
	int dst_x = offset / Y_DATA_SIZE;
	int dst_y = offset % Y_DATA_SIZE;
	int i;

	for (i = 0; i < CHUNK_SIZE; i++) {
		dst[dst_x][dst_y] = src[i];
		if (++dst_y >= Y_DATA_SIZE) {
			dst_y = 0;
			dst_x++;
		}
	}
}


int main (int argc, char **argv)
{
	int myrank, proccount;
	int i, j, k, l, offset;
	MPI_Status status;
	double tmp_arr[CHUNK_SIZE];

	srand(time(NULL));

	// Initialize MPI
	MPI_Init (&argc, &argv);

	// find out my rank
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size (MPI_COMM_WORLD, &proccount);

	if (proccount < 2)
	{
		printf ("Run with at least 2 processes");
		MPI_Finalize ();
		return -1;
	}

	fflush(stdout);
	if (myrank == 0)
	{
		double rand_arr[X_DATA_SIZE][Y_DATA_SIZE];
		double result_arr[X_DATA_SIZE][Y_DATA_SIZE];
		int x_pos = 0, y_pos = 0;
		int dst_x = 0, dst_y = 0;
		int proc_id;

		prepare_rand_data(rand_arr);

		for (proc_id = 1; proc_id < proccount; proc_id++) {
			offset = x_pos * Y_DATA_SIZE + y_pos;
			prepare_chunk_from_data(&x_pos, &y_pos, tmp_arr, rand_arr);

			MPI_Send(&offset, 1, MPI_UNSIGNED, proc_id, OFFSET, MPI_COMM_WORLD);
			MPI_Send(tmp_arr, CHUNK_SIZE, MPI_DOUBLE, proc_id, DATA,
					MPI_COMM_WORLD);
		}

		while (x_pos < X_DATA_SIZE) {
			MPI_Recv(&offset, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, OFFSET,
					MPI_COMM_WORLD, &status);
			proc_id = status.MPI_SOURCE;
			MPI_Recv(&tmp_arr, CHUNK_SIZE, MPI_DOUBLE, proc_id, DATA,
					MPI_COMM_WORLD, &status);

			retrive_data_from_chunk(offset, tmp_arr, result_arr);

			offset = x_pos * Y_DATA_SIZE + y_pos;
			prepare_chunk_from_data(&x_pos, &y_pos, tmp_arr, rand_arr);
			MPI_Send(&offset, 1, MPI_UNSIGNED, proc_id, OFFSET,
					MPI_COMM_WORLD);
			MPI_Send(tmp_arr, CHUNK_SIZE, MPI_DOUBLE, proc_id, DATA,
					MPI_COMM_WORLD);
		}

		for (i = 1; i < proccount; i++) {
			MPI_Recv(&offset, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, OFFSET,
					MPI_COMM_WORLD, &status);
			proc_id = status.MPI_SOURCE;
			MPI_Recv(&tmp_arr, CHUNK_SIZE, MPI_DOUBLE, proc_id, DATA,
					MPI_COMM_WORLD, &status);

			retrive_data_from_chunk(offset, tmp_arr, result_arr);
		}


		for (i = 1; i < proccount; i++)
			MPI_Send(NULL, 0, MPI_DOUBLE, i, FINISH, MPI_COMM_WORLD);

		for (i = 0; i < X_DATA_SIZE; i++) {
			for (j = 0; j < Y_DATA_SIZE; j++) {
				printf(" %f ", result_arr[i][j]);
			}
			printf("\n");
		}
		fflush (stdout);

	} else {
		// slave
		do {
			MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == OFFSET) {
				MPI_Recv(&offset, 1, MPI_UNSIGNED, 0, OFFSET,
						MPI_COMM_WORLD, &status);
				MPI_Recv(tmp_arr, CHUNK_SIZE, MPI_DOUBLE, 0, DATA,
						MPI_COMM_WORLD, &status);

				process_data(tmp_arr, CHUNK_SIZE);

				MPI_Send(&offset, 1, MPI_UNSIGNED, 0, OFFSET, MPI_COMM_WORLD);
				MPI_Send(tmp_arr, CHUNK_SIZE, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD);
			}
		} while (status.MPI_TAG != FINISH);
	}

	MPI_Finalize ();

	return 0;
}
