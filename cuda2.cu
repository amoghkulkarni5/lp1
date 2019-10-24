#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define MAX 128

using namespace std;

// =========================================================================================================================
// Vector Addition
// =========================================================================================================================
__global__ void deviceVectorAddition(int* result, const int* arr1, const int* arr2, const int arraySize) {
    int threadId = threadIdx.x;
	int threadsPerBlock = blockDim.x;
	int blockId = blockIdx.x;

	int index = blockId * threadsPerBlock + threadId;

	if(index < arraySize)
		result[index] = arr1[index] + arr2[index];
}

void hostVectorAddition(int* result, const int* arr1, const int* arr2, const int arraySize) {
	for (int i = 0; i < arraySize; i += 1) {
		result[i] = arr1[i] + arr2[i];
	}
}

cudaError_t vectorAddition(const int size, const int threadsPerBlock=32) {
	
	//size of array (no of elements)
	const int arraySize = size * size;
	//Mem required in bytes
	const int arrayMemory = arraySize * sizeof(int);


	// Array 1 and 2 will added and kept in Result again.
	int* hostArr1;
	int* hostArr2;
	int* hostResult;

	//Final answer vector that ill get from GPU
	int* GPUResult;

	// Three arrays for GPU
	int* deviceArr1;
	int* deviceArr2;
	int* deviceResult;

	
	hostArr1 = (int*) malloc(arrayMemory);
	hostArr2 = (int*)malloc(arrayMemory);
	hostResult = (int*)malloc(arrayMemory);

	//allocation of memories
	GPUResult = (int*)malloc(arrayMemory);

	//Changes the base random thing for rand() function
	srand(time(0));

	for (int i = 0; i < arraySize; i += 1) {
		hostArr1[i] = rand() % MAX;
		hostArr2[i] = rand() % MAX;
	}

	// Create event instances to measure time of execution on GPU
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaError_t cudaStatus;

	// Allocate GPU buffers for result vector
	cudaStatus = cudaMalloc((void**)&deviceResult, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Allocate GPU buffers for input vector
	cudaStatus = cudaMalloc((void**)&deviceArr1, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Allocate GPU buffers for input vector
	cudaStatus = cudaMalloc((void**)&deviceArr2, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers
	cudaStatus = cudaMemcpy(deviceArr1, hostArr1, arrayMemory, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(deviceArr2, hostArr2, arrayMemory, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	// Calculate the number of blocks based on the total elements and number of threads per block
	const int blocksPerGrid = ceil(arraySize / threadsPerBlock);

	// Record time of current event
	cudaEventRecord(cudaStart, 0);

	// Launch a kernel on the GPU with one thread for each element
	// 
	deviceVectorAddition <<< blocksPerGrid, threadsPerBlock >>> (deviceResult, deviceArr1, deviceArr2, arraySize);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
	cudaStatus = cudaDeviceSynchronize();

	// Record time of current event
	cudaEventRecord(cudaStop, 0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory
	cudaStatus = cudaMemcpy(GPUResult, deviceResult, arrayMemory, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	float elapsedTime = 0.0;

	// Calculate elapsed time and destory the event instances
	cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);

	printf("The time of execution on GPU is %lf ms.\n", elapsedTime);

	ios_base::sync_with_stdio(false);
	auto hostStart = chrono::high_resolution_clock::now();
	hostVectorAddition(hostResult, hostArr1, hostArr2, arraySize);
	auto hostStop = chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> time_taken = (hostStop - hostStart);

	printf("The time of execution on CPU is %lf ms.\n", time_taken);

	// Check if the results are same on GPU and CPU
	bool isResultSame = true;
	for (int i = 0; i < arraySize; i += 1) {
		if (hostResult[i] != GPUResult[i]) {
			isResultSame = false;
			break;
		}
	}

	if (isResultSame) {
		printf("The result of CPU and GPU is the same.\n");
	}
	else {
		printf("The result of CPU and GPU is different.\n");
	}

	cudaFree(deviceArr1);
	cudaFree(deviceArr2);
	cudaFree(deviceResult);

	return cudaSuccess;
}

// =========================================================================================================================
// Matrix Vector Multiplication
// =========================================================================================================================
__global__ void deviceVectorMatrixMultiplication(int* result, const int* array, const int* matrix, const int dimension) {
	int threadId = threadIdx.x;
	int threadPerBlock = blockDim.x;
	int blockId = blockIdx.x;

	int tid = threadId + threadPerBlock * blockId;
	if (tid >= dimension)
		return;

	int sum = 0;

	for (int k = 0; k < dimension; k += 1) {
		sum += (array[k] * matrix[tid*dimension + k]);
	}
	result[tid] = sum;
}

void hostVectorMatrixMultiplication(int* result, const int* array, const int* matrix, const int dimension) {
	int sum;
	for (int index = 0; index < dimension; index += 1) {
		sum = 0;

		for (int i = 0; i < dimension; i += 1) {
			sum += (array[i] * matrix[index * dimension + i]);
		}
		result[index] = sum;
	}
}

cudaError_t vectorMatrixMultiplication(const int dimension, const int threadsPerBlock = 32) {
	const int arrayMemory = dimension * sizeof(int);

	int* hostArray;
	int* hostMatrix;
	int* hostResult;

	int* GPUResult;

	int* deviceArray;
	int* deviceMatrix;
	int* deviceResult;

	hostArray = (int*)malloc(arrayMemory);
	hostMatrix = (int*)malloc(arrayMemory * dimension);
	hostResult = (int*)malloc(arrayMemory);

	GPUResult = (int*)malloc(arrayMemory);

	srand(time(0));

	for (int i = 0; i < dimension; i += 1) {
		hostArray[i] = rand() % MAX;
	}

	for (int i = 0; i < dimension * dimension; i += 1) {
		hostMatrix[i] = rand() % MAX;
	}

	// Create event instances to measure time of execution on GPU
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaError_t cudaStatus;

	// Allocate GPU buffers for result vector
	cudaStatus = cudaMalloc((void**)&deviceResult, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for input vector
	cudaStatus = cudaMalloc((void**)&deviceArray, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for input vector
	cudaStatus = cudaMalloc((void**)&deviceMatrix, arrayMemory * dimension);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers
	cudaStatus = cudaMemcpy(deviceArray, hostArray, arrayMemory, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(deviceMatrix, hostMatrix, arrayMemory * dimension, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Calculate the number of blocks based on the total elements and number of threads per block
	const int blocksPerGrid = ceil(dimension / threadsPerBlock);

	// Record time of current event
	cudaEventRecord(cudaStart, 0);

	// Launch a kernel on the GPU with one thread for each element
	deviceVectorMatrixMultiplication << < blocksPerGrid, threadsPerBlock >> > (deviceResult, deviceArray, deviceMatrix, dimension);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
	cudaStatus = cudaDeviceSynchronize();

	// Record time of current event
	cudaEventRecord(cudaStop, 0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory
	cudaStatus = cudaMemcpy(GPUResult, deviceResult, arrayMemory, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	float elapsedTime = 0.0;

	// Calculate elapsed time and destory the event instances
	cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);

	printf("The time of execution on GPU is %lf ms.\n", elapsedTime);

	auto hostStart = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);
	hostVectorMatrixMultiplication(hostResult, hostArray, hostMatrix, dimension);
	auto hostStop = chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> time_taken = (hostStop - hostStart);

	printf("The time of execution on CPU is %lf ms.\n", time_taken);

	// Check if the results are same on GPU and CPU
	bool isResultSame = true;
	for (int i = 0; i < dimension; i += 1) {
		if (hostResult[i] != GPUResult[i]) {
			isResultSame = false;
			break;
		}
	}

	if (isResultSame) {
		printf("The result of CPU and GPU is the same.\n");
	}
	else {
		printf("The result of CPU and GPU is different.\n");
	}

Error:
	cudaFree(deviceArray);
	cudaFree(deviceMatrix);
	cudaFree(deviceResult);

	return cudaSuccess;
}

// =========================================================================================================================
// Matrix Multiplication
// =========================================================================================================================
__global__ void deviceMatrixMultiplication(int* result, const int* mat1, const int* mat2, const int dimension) {
	int threadId = threadIdx.x;
	int threadPerBlock = blockDim.x;
	int blockId = blockIdx.x;

	int tid = threadId + threadPerBlock * blockId;
	if (tid >= dimension * dimension)
		return;

	int row = tid / dimension;
	int column = tid % dimension;
	
	int sum = 0;

	for (int k = 0; k < dimension; k += 1) {
		sum += (mat1[row * dimension + k] * mat2[k * dimension + column]);
	}
	result[tid] = sum;
}

void hostMatrixMultiplication(int* result, const int* mat1, const int* mat2, const int dimension) {
	int sum;
	for (int row = 0; row < dimension; row += 1) {
		for (int col = 0; col < dimension; col += 1) {
			sum = 0;
			for (int k = 0; k < dimension; k += 1) {
				sum += (mat1[row * dimension + k] * mat2[k * dimension + col]);
			}
			result[row * dimension + col] = sum;
		}
	}
}

cudaError_t matrixMultiplication(const int dimension, const int threadsPerBlock = 32) {
	const int arrayMemory = dimension * dimension * sizeof(int);

	int* hostArr1;
	int* hostArr2;
	int* hostResult;

	int* GPUResult;

	int* deviceArr1;
	int* deviceArr2;
	int* deviceResult;

	hostArr1 = (int*)malloc(arrayMemory);
	hostArr2 = (int*)malloc(arrayMemory);
	hostResult = (int*)malloc(arrayMemory);

	GPUResult = (int*)malloc(arrayMemory);

	srand(time(0));

	for (int i = 0; i < dimension * dimension; i += 1) {
		hostArr1[i] = rand() % MAX;
		hostArr2[i] = rand() % MAX;
	}

	// Create event instances to measure time of execution on GPU
	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaError_t cudaStatus;

	// Allocate GPU buffers for result vector
	cudaStatus = cudaMalloc((void**)&deviceResult, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for input vector
	cudaStatus = cudaMalloc((void**)&deviceArr1, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for input vector
	cudaStatus = cudaMalloc((void**)&deviceArr2, arrayMemory);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers
	cudaStatus = cudaMemcpy(deviceArr1, hostArr1, arrayMemory, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(deviceArr2, hostArr2, arrayMemory, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Calculate the number of blocks based on the total elements and number of threads per block
	const int blocksPerGrid = ceil((dimension * dimension) / threadsPerBlock);

	// Record time of current event
	cudaEventRecord(cudaStart, 0);

	// Launch a kernel on the GPU with one thread for each element
	deviceMatrixMultiplication <<< blocksPerGrid, threadsPerBlock >>> (deviceResult, deviceArr1, deviceArr2, dimension);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
	cudaStatus = cudaDeviceSynchronize();

	// Record time of current event
	cudaEventRecord(cudaStop, 0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory
	cudaStatus = cudaMemcpy(GPUResult, deviceResult, arrayMemory, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	float elapsedTime = 0.0;

	// Calculate elapsed time and destory the event instances
	cudaEventElapsedTime(&elapsedTime, cudaStart, cudaStop);
	cudaEventDestroy(cudaStart);
	cudaEventDestroy(cudaStop);

	printf("The time of execution on GPU is %lf ms.\n", elapsedTime);

	auto hostStart = chrono::high_resolution_clock::now();
	ios_base::sync_with_stdio(false);
	hostMatrixMultiplication(hostResult, hostArr1, hostArr2, dimension);
	auto hostStop = chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> time_taken = (hostStop - hostStart);

	printf("The time of execution on CPU is %lf ms.\n", time_taken);

	// Check if the results are same on GPU and CPU
	bool isResultSame = true;
	for (int i = 0; i < dimension * dimension; i += 1) {
		if (hostResult[i] != GPUResult[i]) {
			isResultSame = false;
			break;
		}
	}

	if (isResultSame) {
		printf("The result of CPU and GPU is the same.\n");
	}
	else {
		printf("The result of CPU and GPU is different.\n");
	}

Error:
	cudaFree(deviceArr1);
	cudaFree(deviceArr2);
	cudaFree(deviceResult);

	return cudaSuccess;
}

int main()
{
	int arraySize;
	int threadsPerBlock;

	printf("Enter the size of array and threads per block: : ");
	scanf("%d %d", &arraySize, &threadsPerBlock);

	// Choose which GPU to run on, change this on a multi-GPU system
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

    // Add vectors.
	printf("\n ===== Vector Addition =====\n");
	cudaStatus = vectorAddition(arraySize, threadsPerBlock);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	// Vector matrix multiplication
	printf("\n ===== Vector Matrix Multiplication =====\n");
	cudaStatus = vectorMatrixMultiplication(arraySize, threadsPerBlock);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// Matrix multiplication
	printf("\n ===== Matrix Multiplication =====\n");
	cudaStatus = matrixMultiplication(arraySize, threadsPerBlock);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

    // cudaDeviceReset must be called before exiting in order for profiling and tracing tools to show complete traces
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
