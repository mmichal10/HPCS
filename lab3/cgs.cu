/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate graident solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas and cusparse */
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper function CUDA error checking and intialization

const char *sSDKname     = "conjugateGradient";

double mclock(){
     struct timeval tp;

     double sec,usec;
     gettimeofday( &tp, NULL );
     sec    = double( tp.tv_sec );
     usec   = double( tp.tv_usec )/1E6;
     return sec + usec;
}


#define dot_BS     32
#define kernel_BS  32

__global__
void saxpy(int n, float alpha, float *x, float *y) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < n)
		y[i] = alpha * x[i] + y[i];
}

__global__
void scal(int n, float alpha, float *y) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < n)
		y[i] = alpha * y[i];
}

__global__
void cpy(int n, float *src, float *dst) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < n)
		dst[i] = src[i];
}

__global__
void dot(int n, float *src, float *dst) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int tid = threadIdx.x;
	extern __shared__ float c_shared[];

	if (i < n) {
		c_shared[i] = src[i] * dst[i];
		__syncthreads();

		for (unsigned int s=1; s < blockDim.x; s *= 2) {
			if (tid % (2*s) == 0)
				c_shared[tid] += c_shared[tid + s];

			__syncthreads();
		}

		if (tid == 0)
			dst[blockIdx.x] = c_shared[0];
	}
}

__global__
void csrmv(int m, int n, int nnz, float alpha, float *csrValA, int *csrRowPtrA,
		int *csrColIdA, float *x, float beta, float *y)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int j;
	float sub = 0;
	if (i < n) {
		for (j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
			sub += csrValA[j] * x[csrColIdA[j]];
		y[i] = sub;
	}
}

__global__
void dot(int n, float *x, float *y, float *result)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i < n) {
		y[i] = x[i] * y[i];
	}
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    double RAND_MAXi = 1e6;
    double val_r     = 12.345 * 1e5;
    
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)val_r/RAND_MAXi + 10.0f;
    val[1] = (float)val_r/RAND_MAXi;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)val_r/RAND_MAXi + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)val_r/RAND_MAXi;
        }
    }

    I[N] = nz;
}


void cgs_basic(int argc, char **argv, int N, int M){

    //int M = 0, N = 0, 
    int nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-10f;
    const int max_iter = 1000;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    //M = N = 32*64;//10; //1048576;
    printf("M = %d, N = %d\n", M, N);
    nz = (N-2)*3 + 4;
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (float *)malloc(sizeof(float)*nz);
    genTridiag(I, J, val, N, nz);

    /*
    for (int i = 0; i < nz; i++){
        printf("%d\t", J[i]);
    }
    printf("\n");
    for (int i = 0; i < nz; i++){
        printf("%2f\t", val[i]);
    }
    */

    x = (float *)malloc(sizeof(float)*N);
    rhs = (float *)malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;


    double t_start = mclock();
    //cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
	csrmv<<<(N+255)/256, 256>>>(N, N, nz, alpha, d_val, d_row, d_col, d_x, beta, d_Ax);

    //cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);                                // PODMIEN FUNCKJE (I)
	saxpy<<<(N+255)/256, 256>>>(N, alpham1, d_Ax, d_r);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        // PODMIEN FUNCKJE (II)

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            //cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);                        // PODMIEN FUNCKJE (I)
			scal<<<(N+255)/256, 256>>>(N, b, d_p);
            //cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);            // PODMIEN FUNCKJE (I)
			saxpy<<<(N+255)/256, 256>>>(N, alpha, d_r, d_p);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);                    // PODMIEN FUNCKJE (I)
        }

        //cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax); // PODMIEN FUNCKJE (III)
		csrmv<<<(N+255)/256, 256>>>(N, N, nz, alpha, d_val, d_row, d_col, d_p, beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);                  // PODMIEN FUNCKJE (II)
        a = r1 / dot;

        //cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);                    // PODMIEN FUNCKJE (I)
		saxpy<<<(N+255)/256, 256>>>(N, a, d_p, d_x);
        na = -a;
        //cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);                  // PODMIEN FUNCKJE (I)
		saxpy<<<(N+255)/256, 256>>>(N, na, d_Ax, d_r);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                    // PODMIEN FUNCKJE (II)
        cudaThreadSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }
    printf("TIME OF CGS_BASIC = %f\n", mclock() - t_start);

    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    cudaDeviceReset();

    printf("Test Summary:  Error amount = %e\n", err);
    //exit((k <= max_iter) ? 0 : 1);


}
void cgs_TODO(int argc, char **argv, int N, int M){

    //int M = 0, N = 0, 
    int nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    const float tol = 1e-10f;
    const int max_iter = 1000;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    //M = N = 32*64;//10; //1048576;
    printf("M = %d, N = %d\n", M, N);
    nz = (N-2)*3 + 4;
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (float *)malloc(sizeof(float)*nz);
    genTridiag(I, J, val, N, nz);

    /*
    for (int i = 0; i < nz; i++){
        printf("%d\t", J[i]);
    }
    printf("\n");
    for (int i = 0; i < nz; i++){
        printf("%2f\t", val[i]);
    }
    */

    x = (float *)malloc(sizeof(float)*N);
    rhs = (float *)malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;


    // sparse matrix vector product: d_Ax = A * d_x
    //cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);  // PODMIEN FUNCKJE (ZADANIE-I)
	csrmv<<<(N+255)/256, 256>>>(N, N, nz, alpha, d_val, d_row, d_col, d_x, beta, d_Ax);


    //azpy: d_r = d_r + alpham1 * d_Ax
    //cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);        			    // PODMIEN FUNCKJE (ZADANIE-I)
	saxpy<<<(N+255)/256, 256>>>(N, alpham1, d_Ax, d_r);
    //dot:  r1 = d_r * d_r
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                        // PODMIEN FUNCKJE (ZADANIE-III)

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
	    //scal: d_p = b * d_p
            //cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);                        // PODMIEN FUNCKJE (ZADANIE-I)
			scal<<<(N+255)/256, 256>>>(N, b, d_p);
	    //axpy:  d_p = d_p + alpha * d_r
            //cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);            // PODMIEN FUNCKJE (ZADANIE-I)
			saxpy<<<(N+255)/256, 256>>>(N, alpha, d_r, d_p);
        }
        else
        {
            //cpy: d_p = d_r
            //cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);                    // PODMIEN FUNCKJE (ZADANIE-I)
			cpy<<<(N+255)/256, 256>>>(N, d_r, d_p);
        }

        //sparse matrix-vector product: d_Ax = A * d_p
        //cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax); // PODMIEN FUNCKJE (ZADANIE-II)
	csrmv<<<(N+255)/256, 256>>>(N, N, nz, alpha, d_val, d_row, d_col, d_p, beta, d_Ax);

        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);                  // PODMIEN FUNCKJE (ZADANIE-III)
        a = r1 / dot;

        //axpy: d_x = d_x + a*d_p
        //cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);                    // PODMIEN FUNCKJE (ZADANIE-I)
		saxpy<<<(N+255)/256, 256>>>(N, a, d_p, d_x);
        na = -a;
	 
        //axpy:  d_r = d_r + na * d_Ax
        //cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);                  // PODMIEN FUNCKJE (ZADANIE-I)
		saxpy<<<(N+255)/256, 256>>>(N, na, d_Ax, d_r);

        r0 = r1;
	
        //dot: r1 = d_r * d_r
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);                    // PODMIEN FUNCKJE (ZADANIE-III)
        cudaThreadSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    cudaDeviceReset();

    printf("Test Summary:  Error amount = %e\n", err);
    //exit((k <= max_iter) ? 0 : 1);


}

int main(int argc, char **argv)
{
    //int N = 1e6;//1 << 20;
    //int N = 256 * (1<<10)  -10 ; //1e6;//1 << 20;
    int N = 1e5;
    int M = N; 
    
    cgs_basic(argc, argv, N, M);
    
    cgs_TODO(argc, argv, N, M);
}
