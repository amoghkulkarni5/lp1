#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include<omp.h>
#include<time.h>

#define thread_count 4



/*
RUN WITH:
g++ filename.cpp -fopenmp
a.out file
*/

/*
    OPENMP PRAGMA USED:

    1) parallel : Forks additional threads to carry out the work in parallel
    2) num_threads : Sets the number of threads in thread team
    3) default : Specifies the behaviour of unscoped variables in parallel section
    4) shared : Specifies the variables that are shared among all threads
    5) private : Specifies each thread should have its own instance of the variable
    6) for : Causes the block in for loop to be divided among multiple threads

    7) task : Code blocks that are wrapped and executed in parallel
    8) firstprivate : Specifies each thread should have its own instance of variable and that it has to be initialized with value of variable as it exists brfore the parallel construct
    9) taskwait : Specifies a wait for child tasks generated by the current task to be completed
 */

void swap(int* num1, int* num2){
    int temp = *num1;
    *num1 = *num2;
    *num2 = temp;
}
// ==================== BUBBLE SORT ====================
double bubblesort_parallel(int* arr, int size){
	int i, j, first;
	
	double start, end;

    start = omp_get_wtime();
    
    #pragma omp parallel num_threads(thread_count) default(none) shared(arr, size) private(i, j, first)
    for(i=0; i<size - 1; i+=1){
        first = i & 1;

        #pragma omp for
        for(j=first; j<size - 1; j+=1){
            if(arr[j] > arr[j+1])
                swap(&arr[j], &arr[j+1]);
        }
        
    }

    end = omp_get_wtime();
    
    return (end - start);
}

double bubblesort_serial(int* arr, int size){
	int i, j;
	double start, end;
	
	start = omp_get_wtime();
	
	for(i=0; i<size - 1; i+=1){
        for(j=0; j<size-i - 1; j+=1){
            if(arr[j] > arr[j+1])
                swap(&arr[j], &arr[j+1]);
        }
    }
    
    end = omp_get_wtime();
    
    return (end - start);
}

void bubblesort(int *arr, int size){
    int arrayMemory = size * sizeof(int);
    int* serialArray = (int*) malloc(arrayMemory);
	int* parallelArray = (int*) malloc(arrayMemory);
	
    memcpy(serialArray, arr, arrayMemory);
    memcpy(parallelArray, arr, arrayMemory);
	
	double time_serial = bubblesort_serial(serialArray, size);
	double time_parallel = bubblesort_parallel(parallelArray, size);
	
	
	printf("\n*** Serial Bubble Sort ***\n");
    printf("Time Required: : %lf\n", time_serial);
    printf("The sorted elements are: : ");
    
	for(int i=0; i<size; i++)
		printf("%d ", serialArray[i]);
	
    printf("\n");

	printf("\n*** Parallel Bubble Sort ***\n");
    printf("Time Required: : %lf\n", time_parallel);
    printf("The sorted elements are: : ");
    
	for(int i=0; i<size; i++)
		printf("%d ", parallelArray[i]);

    printf("\n");
}

// ==================== MERGE SORT ====================
void merge(int* arr, int low, int high, int mid) {
    int temp[high - low + 1];
    int i = low;
    int j = mid + 1;
    int k = 0;

    while(i <= mid && j <= high) {
        if(arr[i] < arr[j]) {
            temp[k] = arr[i];
            k += 1;
            i += 1;
        }
        else {
            temp[k] = arr[j];
            k += 1;
            j += 1;
        }
    }

    while(i <= mid) {
        temp[k] = arr[i];
        k += 1;
        i += 1;
    }

    while(j <= high) {
        temp[k] = arr[j];
        k += 1;
        j += 1;
    }

    memcpy(arr + low, temp, k * sizeof(int));
}

void mergesortSerial(int* arr, int low, int high) {
    int mid;

    if(low < high) {
        mid = (low + high) >> 1;
        
        mergesortSerial(arr, low, mid);
        mergesortSerial(arr, mid + 1, high);

        merge(arr, low, high, mid);
    }
}

void mergesortParallel(int* arr, int low, int high) {
    int mid;

    if(low < high) {
        mid = (low + high) >> 1;
        
        #pragma omp parallel
        mergesortSerial(arr, low, mid);
        
        #pragma omp parallel
        mergesortSerial(arr, mid + 1, high);

        #pragma omp taskwait
        merge(arr, low, high, mid);
    }

}

void mergesort(int* arr, int size) {
    int arrayMemory = size * sizeof(int);
    int* serialArray = (int*) malloc(arrayMemory);
	int* parallelArray = (int*) malloc(arrayMemory);
	
    memcpy(serialArray, arr, arrayMemory);
    memcpy(parallelArray, arr, arrayMemory);

    double start, end;

    start = omp_get_wtime();
    mergesortSerial(serialArray, 0, size - 1);
    end = omp_get_wtime();
    
    double serialTime = end - start;

    start = omp_get_wtime();
    mergesortParallel(parallelArray, 0, size - 1);
    end = omp_get_wtime();
    
    double parallelTime = end - start;

    printf("\n*** Serial Merge Sort ***\n");
    printf("Time Required: : %lf\n", serialTime);

    printf("The sorted elements are: : ");
    for(int i=0; i<size; i++)
	    printf("%d ", serialArray[i]);

    printf("\n");

    printf("\n*** Parallel Merge Sort ***\n");
    printf("Time Required: : %lf\n", parallelTime);

    printf("The sorted elements are: : ");
    for(int i=0; i<size; i++)
	    printf("%d ", parallelArray[i]);

    printf("\n");

}

// ==================== MAIN FUNCTION ====================
int main(int argc, char** argv){
    int size;
    int* arr;
    
    // If size is not given as command line argument
    if(argc==1){
        scanf("%d", &size);
        arr = (int*) malloc(size * sizeof(int));
        
        for(int i=0; i<size; i+=1)
            scanf("%d", &arr[i]);
    }
    // If size is given as command line argument
    else{
        sscanf(argv[1], "%d", &size);
        arr = (int*) malloc(size * sizeof(int));
        
        srand(time(0));
        printf("The random elements are: : ");
        for(int i=0; i<size; i+=1){
            arr[i] = rand() % size;
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    printf("\n==================== BUBBLE SORT ====================\n");
    bubblesort(arr, size);

    printf("\n==================== MERGE SORT ====================\n");
	mergesort(arr, size);
    printf("\n");

    return 0;
}
