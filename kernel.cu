#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

using namespace std;

__global__ void deviceSum(int* array, int arraySize, int virtualSize) {

	int tid = threadIdx.x;

	if (tid >= arraySize)
		return;
	int nextElement = tid + (virtualSize >> 1);
	if (nextElement >= arraySize)
		return;

	array[tid] += array[nextElement];
}


cudaError_t arraySum(const int* array, const int arraySize)
{

	const int arrayMemory = arraySize * sizeof(int);
	cudaError_t cudaStatus;
	int gpuResult, cpuResult;
	int virtualSize, tempVirtualSize, tempThreadsPerBlock;


	int* hostArray;
	int* deviceArray;

	hostArray = (int*)malloc(arrayMemory);
	memcpy(hostArray, array, arrayMemory);

	for (virtualSize = 1; arraySize > virtualSize; virtualSize <<= 1);

	printf("Virtual Size: %d", virtualSize);
	cudaStatus = cudaMalloc((void**)&deviceArray, arrayMemory);

	cudaStatus = cudaMemcpy(deviceArray, hostArray, arrayMemory, cudaMemcpyHostToDevice);

	tempVirtualSize = virtualSize;
	tempThreadsPerBlock = virtualSize / 2;

	do
	{
		deviceSum << <1, tempThreadsPerBlock >> > (deviceArray, arraySize, tempVirtualSize);
		tempVirtualSize >>= 1;
		tempThreadsPerBlock >>= 1;
		cudaDeviceSynchronize();
	} while (tempThreadsPerBlock >= 1);


	cudaStatus = cudaMemcpy(&gpuResult, deviceArray, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Sum: %d", gpuResult);

	return cudaStatus;
}



int main() {
	cudaError_t cudaStatus;


	int arraySize;
	int threadsPerBlock;
	int maxElement;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("Error");
	}

	arraySize = 16;
	threadsPerBlock = arraySize / 2;
	int* array;
	const int arrayMemory = arraySize * sizeof(int);
	array = (int*)malloc(arrayMemory);

	srand(time(0));
	for (int i = 0; i < arraySize; i += 1) {
		array[i] = 1;
		printf("%d ", array[i]);
	}

	printf("\n\n");
	printf("===== Vector Sum =====\n");
	clock_t t = clock();
	arraySum(array, arraySize);
	double time_taken = ((double)t) / CLOCKS_PER_SEC;
	printf("   Time Taken:  %f", time_taken);
}