%%cu

#include<stdio.h>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"





__global__ void deviceSum(int *arr, int actualsize, int virtualsize)
{
	int tid=threadIdx.x;
	if(tid>actualsize)
		return;
	int nextelement= tid + (virtualsize>>1);
	if(nextelement>actualsize)
		return;
	
	arr[tid]=arr[tid]+arr[nextelement];
}


void vectoraddition(int *arr, int size)
{
	cudaError_t status;
	int actualsize=size;
	int *devicearray;
	int gpuresult;
	
	int arrayMemory= actualsize * sizeof(int);
	int virtualsize;
	
	status= cudaMalloc(&devicearray,actualsize);
	if(status!=cudaSuccess)
	{
		printf("Anarth-1");
	}
	status= cudaMemcpy(devicearray,arr,size,cudaMemcpyDeviceToHost);
	
	for(virtualsize=1; virtualsize < actualsize; virtualsize <<= 1);
	printf("Virutal Size: %d", virtualsize);
	
	int maxthreads= virtualsize /2;
	int tempthreads=maxthreads;
	do
	{
	deviceSum<<1,tempthreads>>(devicearray,actualsize,virtualsize);
	tempthreads >>=1;
	virtualsize >>=1;
	cudaDeviceSynchronize();
	} while(tempthreads>=1);
	
	status= cudaMemcpy(&gpuresult,devicearray,sizeof(int),cudaMemcpyDeviceToHost);
	printf("\nSum: %d",gpuresult);

}








int main()
{
	int *arr, size;
	size=15;
	int arrayMemory= size * sizeof(int);
	arr= (int*) malloc(arrayMemory);
	
	srand(time(0));
	for(int i=0;i<15;i++)
	{
		arr[i]= rand() % size;
	}
	
	printf("\nVector Addition\n");
	clock_t t= clock();
	vectoraddition(arr,size);
	clock_t time= clock()- t;
	printf("Time Required: %f",time);

}
