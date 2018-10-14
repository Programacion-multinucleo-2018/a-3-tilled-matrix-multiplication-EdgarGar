#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;
//Tamaño de la matriz
#define NTM 2000
//Bloques en este caso se probara con tres 8*8, 16*16 y 32*32
#define DIM 8


//Funcion de llenado de la matriz en tre 0 y 10 obtenida de la primera tarea
void fillMat(float * ip, const int size) {
  int i;
  for(i = 0; i < size; i++) {
    ip[i] = (rand() / (float)RAND_MAX * 10.0f);
  }
}

/*
__global__ void matrixMultOnHostGPU(int *a, int *b, int *c) {
 int k, sum = 0;
 int col = threadIdx.x + blockDim.x * blockIdx.x;
 int fil = threadIdx.y + blockDim.y * blockIdx.y;

 if (col < N && fil < N) {
  for (k = 0; k < N; k++) {
   sum += a[fil * N + k] * b[k * N + col];
  }
  c[fil * N + col] = sum;
 }
}*/
__global__ void multMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix < nx && iy < ny) {
    for(int i = 0; i < ny; i++) {
      MatC[ix*ny+iy] += MatA[ix*ny+i] * MatB[i*ny+iy];
    }
  }
}

//Funcion de matrix mult con tiles
__global__ void multMatrixOnTiles(float *A, float *B, float *C, int nx, int ny) {
  float sum = 0;
  //Algunas partes del codigo fueron obtenidas de los demos vistos en clase
  unsigned int ix = threadIdx.x + blockIdx.x * DIM;
  unsigned int iy = threadIdx.y + blockIdx.y * DIM;

  __shared__ float matTempA[DIM][DIM];
  __shared__ float matTempB[DIM][DIM];

  //Llenamos las matrices shared y iniciado de 0
  for(int i = 0; i < DIM; i ++) {
    for(int j = 0; j < DIM; j++) {
      matTempA[i][j] = 0;
      matTempB[i][j] = 0;
    }
  }

  //vamos a traves de todos los tiles
  for(int i = (DIM + nx - 1)/DIM; i >= 0; i--) {
    if((i * DIM + threadIdx.x) < nx && (iy < ny)) {
      matTempA[threadIdx.y][threadIdx.x] = A[(iy*ny) + (i*DIM+threadIdx.x)];
    }

    if((i * DIM + threadIdx.y) < ny && (ix < nx)) {
      matTempB[threadIdx.y][threadIdx.x] = B[(i*DIM+threadIdx.y) * nx + ix];
    }
    /*//__syncthreads(); command is a block level synchronization barrier. That means it is safe to be used when all threads in a
    //block reach the barrier. It is also possible to use __syncthreads() in conditional code but only when all
    //threads evaluate identically such code otherwise
    //the execution is likely to hang or produce unintended side effects*/

    __syncthreads(); //Tenemos que utilizar syncthreads despues de modificar las matrices en threadIdx
    for(int j = 0; j < DIM; j++) {
      sum += matTempA[threadIdx.y][j] * matTempB[j][threadIdx.x];
    }
    __syncthreads();
  }
  if(ix < nx && iy < ny) {
    C[iy*ny+ix] = sum;
  }
}

//Funcion obtenida de la primera matriz
void multMat(float *A, float *B, float *C, const int nx, const int ny) {
  for(int i = 0; i < ny; i++) {
    for(int j = 0; j < nx; j++) {
      for(int k = 0; k < ny; k++) {
        //Operacion para hacer la regla del karatzo fila por culumna
        C[i * nx + j] += (A[i * nx + k] * B[k + nx * j]);
      }
    }
  }
}

//Funcion que checa el resultado el cual ya teniamos de la primera tarea
void checkResult(float *hostRef, float *gpuRef, const int N)
{
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N*N; i++){
    if (fabs(hostRef[i] - gpuRef[i]) > epsilon){
      match = 0;
      printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
      break;
    }
  }
  if (match)
    printf("Matrix multiplications from host and GPU match!.\n\n");
  else
    printf("Arrays do not match.\n\n");
}

//Main que ya teniamos de los otros ejemplos solo cambio nombres de la funciones y mandado a llamar de algunas
int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
    printf("Tarjeta %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");

    // set up data size of matrix
    int nx = NTM;
    int ny = NTM;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // Inicializar nuestros datos
    fillMat(h_A, nxy);
    fillMat(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result SAFE_CALLs
    auto start_cpu =  chrono::high_resolution_clock::now();
    multMat(h_A, h_B, hostRef, nx, ny);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("MultMat en Host elapsed %f ms\n\n", duration_ms.count());

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = DIM;
    int dimy = DIM;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    //MULTMAT ON GPU 2D_2D (ya se tenia)
    //Multiplicar matrices con cantidad de repeticiones
    int timeAverage = 0;
    // add matrix at host side for result SAFE_CALLs
    //Lo sacamos del ejemplo de clase
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    timeAverage += duration_ms.count();
    int performanceTime = timeAverage;
    printf("Ejecucion con GPU con threads: %d ms\n", performanceTime);
    printf("Matriz: %d x %d\n", nx, ny);

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    //MULTMAT CON TILING GPU
    timeAverage = 0;
    // add matrix at host side for result SAFE_CALLs
    start_cpu =  chrono::high_resolution_clock::now();
    multMatrixOnTiles<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    timeAverage += duration_ms.count();
    performanceTime = timeAverage;
    printf("Ejecucion con Tiling: %d x %d es alrededor de: %d ms\n", DIM, DIM, performanceTime);
    printf("Matriz: %d x %d\n", nx, ny);

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return (0);
}
