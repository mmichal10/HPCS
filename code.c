#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#define PRECISION 0.000001
#define RANGESIZE 1

#define DATA 0
#define OFFSET 1

#define RET_DATA 2
#define RET_OFFSET 3

#define RESULT 1
#define FINISH 2

#define X_DATA_SIZE 1000
#define Y_DATA_SIZE X_DATA_SIZE
#define DATA_SIZE (X_DATA_SIZE * Y_DATA_SIZE)

#define X_CHUNK 100
#define Y_CHUNK X_CHUNK
#define CHUNK_SIZE (X_CHUNK * Y_CHUNK)


//#define DEBUG
//
	double
f (double x)
{
	return sin (x) * sin (x) / x;
}

	double
SimpleIntegration (double a, double b)
{
	double i;
	double sum = 0;
	for (i = a; i < b; i += PRECISION)
		sum += f (i) * PRECISION;
	return sum;
}

	int
main (int argc, char **argv)
{
	int myrank, proccount;
	double a = 1, b = 100;
	double range[2];
	double rand_data[X_DATA_SIZE][Y_DATA_SIZE];
	double tmp_data[X_CHUNK][Y_CHUNK];
	double result = 0, resulttemp;
	unsigned long offset = 0;
	int sentcount = 0;
	int i, j, k, l;
	MPI_Status status;

	srand(time(NULL));

	// Initialize MPI
	MPI_Init (&argc, &argv);

	// find out my rank
	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size (MPI_COMM_WORLD, &proccount);

		printf("Line %u\n", __LINE__);
		fflush (stdout);

	if (proccount < 2)
	{
		printf ("Run with at least 2 processes");
		MPI_Finalize ();
		return -1;
	}

		printf("Line %u\n", __LINE__);
		fflush (stdout);

	if (((b - a) / RANGESIZE) < 2 * (proccount - 1))
	{
		printf ("More subranges needed");
		MPI_Finalize ();
		return -1;
	}

		printf("Line %u\n", __LINE__);
		fflush (stdout);

	// now the master will distribute the data and slave processes will perform computations
	if (myrank == 0)
	{
		int x_pos = 0, y_pos = 0;
		int rcv_offset;

		printf("Line %u\n", __LINE__);
		fflush (stdout);

		for (i = 0; i < X_DATA_SIZE; i++) {
			for (j = 0; j < Y_DATA_SIZE; j++)
				rand_data[i][j] = rand() % 1000;
		}

		printf("Line %u\n", __LINE__);
		fflush (stdout);

		for (i = 0; i < proccount; i++, offset++) {
			y_pos = 0;
			for (j = 0; j < X_CHUNK; j++) {
				for (k = 0; k < Y_CHUNK; k++) {
					tmp_data[j][k] = rand_data[x_pos + j][y_pos + k];
				}
				y_pos += Y_CHUNK;
			}
			x_pos += X_CHUNK;
			MPI_Send(&offset, 1, MPI_UNSIGNED, i, OFFSET, MPI_COMM_WORLD);
			MPI_Send(tmp_data, X_CHUNK * Y_CHUNK, MPI_DOUBLE, i, DATA,
					MPI_COMM_WORLD);
		}

		printf("Line %u - random array ready!\n", __LINE__);
		fflush (stdout);

		do
		{
			int x, y;
			int rcv_rank;
			// distribute remaining subranges to the processes which have completed their parts
			MPI_Recv (&rcv_offset, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
					RESULT,	MPI_COMM_WORLD, &status);
			rcv_rank = status.MPI_SOURCE;
			MPI_Recv (&tmp_data, X_CHUNK * Y_CHUNK, MPI_DOUBLE,
					rcv_rank, RESULT, MPI_COMM_WORLD, &status);

			x = rcv_offset / (X_DATA_SIZE / X_CHUNK);
			y = rcv_offset % (Y_DATA_SIZE / Y_CHUNK);
			for (i = 0; i < X_DATA_SIZE; i++) {
				for (j = 0; j < Y_DATA_SIZE; j++) {
					rand_data[x + i][y + i] = tmp_data[i][j];
				}
			}

			for (j = 0; j < X_CHUNK; j++) {
				for (k = 0; k < Y_CHUNK; k++) {
					tmp_data[j][k] = rand_data[x_pos + j][y_pos + k];
				}
				y_pos += Y_CHUNK;
			}

			offset++;
			if (y_pos >= X_DATA_SIZE) {
				x_pos += X_CHUNK;
				y_pos = 0;
			}

			MPI_Send(&offset, 1, MPI_UNSIGNED, i, OFFSET, MPI_COMM_WORLD);
			MPI_Send(tmp_data, X_CHUNK * Y_CHUNK, MPI_DOUBLE, i, DATA,
					MPI_COMM_WORLD);
		}

		while (x_pos * y_pos < DATA_SIZE);
		// now receive results from the processes
		for (i = 0; i < (proccount - 1); i++)
		{
			MPI_Recv (&resulttemp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, RESULT,
					MPI_COMM_WORLD, &status);
#ifdef DEBUG
			printf ("\nMaster received result %f from process %d",
					resulttemp, status.MPI_SOURCE);
			fflush (stdout);
#endif
			result += resulttemp;
		}
		// shut down the slaves
		for (i = 1; i < proccount; i++)
		{
			MPI_Send (NULL, 0, MPI_DOUBLE, i, FINISH, MPI_COMM_WORLD);
		}
		// now display the result
		printf ("\nHi, I am process 0, the result is %f\n", result);
	}
	else
	{				// slave

	goto end;
		// this is easy - just receive data and do the work
		do
		{
			MPI_Probe (0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if (status.MPI_TAG == OFFSET)
			{
				MPI_Recv (&offset, 1, MPI_DOUBLE, 0, OFFSET, MPI_COMM_WORLD,
						&status);
				MPI_Recv(tmp_data, CHUNK_SIZE, MPI_DOUBLE, 0, DATA,
						MPI_COMM_WORLD, &status);

				for (i = 0; i < X_CHUNK; i++) {
						for (j = 0; j < Y_CHUNK; j++) {
							tmp_data[i][j] = 1.0;
						}
					}

				MPI_Send (&offset, 1, MPI_DOUBLE, 0, RET_OFFSET,
						MPI_COMM_WORLD);
				MPI_Send (tmp_data, CHUNK_SIZE, MPI_DOUBLE, 0, RET_DATA,
						MPI_COMM_WORLD);
			}
		}
		while (status.MPI_TAG != FINISH);
end:
	;
	}

	// Shut down MPI
	MPI_Finalize ();

	return 0;
}
