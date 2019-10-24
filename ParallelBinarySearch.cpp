


/*
mpic++ filename.cpp
mpirun -np no_of_threads
./a.out

*/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <cstdlib>

using namespace std;

const int arraySize = 4096;
const int key = 100;

void binarySearch(int *arr, int left, int right, int key, int rank) {
	while(left<=right) {
		
        int mid=(left+right)/2;

		if(arr[mid] == key) {
			printf("Element is found by processor %d at position %d\n", rank, mid);
			return;
		}
		else if(arr[mid] < key)
		{
			left=mid+1;
		}
		else
		{
			right=mid-1;
		}
	}
}
int main(int argc, char **argv) {
	int *arr;
	arr = (int*) malloc(arraySize * sizeof(int));

	for(int i=0; i<arraySize; i+=1) 
	{
		arr[i]=i+1;
	}
	MPI_Init(&argc, &argv);

	int rank,size;
	
	
	// Which processor the code will run on
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Capacity in batli terms
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("Processor rank: %d, Size: %d\n", rank, size);
	int blockCount = 2;
	int blockSize = arraySize / blockCount;


	if(rank==0) {
		double start = MPI_Wtime();
		binarySearch(arr, rank*blockSize, (rank+1)*blockSize-1, key, rank);
		double end = MPI_Wtime();
		printf("Execution time of Processor %d is %lf\n", rank, (end - start) * 1000);
	}
	else if(rank==1) {
		double start = MPI_Wtime();
		binarySearch(arr, rank*blockSize, (rank+1)*blockSize-1, key, rank);
		double end = MPI_Wtime();
		printf("Execution time of Processor %d is %lf\n", rank, (end - start) * 1000);
	}
	else if(rank==2) {
		double start = MPI_Wtime();
		binarySearch(arr, rank*blockSize, (rank+1)*blockSize-1, key, rank);
		double end = MPI_Wtime();
		printf("Execution time of Processor %d is %lf\n", rank, (end - start) * 1000);
	}

	// De-allocation
	MPI_Finalize();

	return 0;
}
