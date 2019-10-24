#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <mpi.h>
#include <math.h>

#define send_data_tag 2001
#define receive_data_tag 2002

#define searchElement 1024

int arr[100];
int array[100];

void display(int* arr, int end) {
	for(int i = 0; i<end; i++) {
		if (searchElement == arr[i]) {
			printf("Element Found.\n");
			exit(0);
		}
		printf("%d ", arr[i]);
	}
	printf("\n");
}

int main(int argc,char** argv) {
	srand(time(0));
	
	int N = 5;
	
	int ierr;
	
    MPI_Status status;

    int my_id, num_procs, newsize;

    double start, finish;

	ierr = MPI_Init(&argc, &argv); 
	start = MPI_Wtime();

    	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    	ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	
    	if(my_id == 0)
    	{
    		int size = ((int) pow(2.0, 5.0)) - 1;
    	
    		for(int i=0; i<size; i++)
		{
			arr[i] = i+1;
		}
	
    		printf("Array: \n");
		display(arr, size);
    		
    		int i = 1;
    		int sum = 0;
    		
    		for(int id = 1; id<=num_procs; id++)
    		{
	   	    	if(sum < size)
			{			
				ierr = MPI_Send(&i, 1, MPI_INT, id, send_data_tag, MPI_COMM_WORLD);
				ierr = MPI_Send(&arr[sum], i, MPI_INT, id, send_data_tag, MPI_COMM_WORLD);
			
				sum = sum + i;
				i <<= 1;
		    	}
		}
	}
	else
	{
		ierr = MPI_Recv(&newsize, 1, MPI_INT, 0, send_data_tag, MPI_COMM_WORLD, &status);
		ierr = MPI_Recv(&array, newsize, MPI_INT, 0, send_data_tag, MPI_COMM_WORLD, &status);
        	
		display(array, newsize);
	}
	finish = MPI_Wtime();
	printf("Execution time = %f seconds\n", (finish-start));	
	MPI_Finalize();

}

