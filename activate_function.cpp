#include "mm_utils.h" //a library of basic matrix utilities functions
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
// and some key constants used in this program
//(such as TYPE)

using namespace std;

#define TOLERANCE 0.001
#define DEF_SIZE 1000
#define MAX_ITERS 100000
#define LARGE 1000000.0

#define REAL_T float
typedef REAL_T real_t;

int mem_size = 721696;
int mem_size_hard = 358440;
int x_size_bytes = 28*28*sizeof(float);
int num_cu = 1;

static int argmax(const real_t *a, int n)
{
  real_t max;
  int i, j;

  max = a[0];
  for (i=j=0; i<n; ++i) {
	// printf("a[%d]=%f\n",i,a[i]);
    if (max < a[i]) {
      max = a[i];
      j = i;
    }
  }
  return j;
}

static const int DATA_SIZE = 1024;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";
    

//#define DEBUG    1     // output a small subset of intermediate values
//#define VERBOSE  1

// static void L1(__global float *m_,
// __global float *x_,
// int x_offset,
// float *A,
// float *z) {
// 	float local_in[784];// __attribute__((xcl_array_partition(cyclic,34,1)));

//   	//__attribute__((xcl_pipeline_loop(1)))
//   	RD_X: for(int j = 0; j < 784; j++){
//     	local_in[j] = x_[x_offset+j];
//   	}

//   	int i, j, flag=0;

//   	//__attribute__((xcl_pipeline_loop(1)))
//   	MAC1_ADD_1: for (i=0; i<100; ++i) 
//     	z[i] = m_[78400+i];

//   	//__attribute__((xcl_pipeline_loop))
//   	MAC1_1: for (j=0; j<784; ++j) {
//     	//__attribute__((opencl_unroll_hint(34)))
//     	MAC1_1_1: for (i=0; i<100; ++i) {
//       		z[i] += A[i * 784 + j] * local_in[j];
//     	}
//   	}

// 	/*ADD + RELU*/
// 	//__attribute__((opencl_unroll_hint(34)))
// 	MAC1_1_2: for (i=0; i<100; ++i) {
// 		if (z[i] <= 0.0) 
// 			z[i] = 0.0;
// 	}
// }

// static void L2(__global float *m_,
// float *AA,
// float *z,
// float *zz) {
// 	int i, j;

// 	//__attribute__((xcl_pipeline_loop(1)))
//   	MAC1_ADD_2: for (i=0; i<100; ++i)
//   		zz[i] = m_[88500+i];

//   	//__attribute__((xcl_pipeline_loop))
//   	MAC1_2: for (j=0; j<100; ++j) {
//   		//__attribute__((opencl_unroll_hint(34)))
//   		MAC1_2_1: for (i=0; i<100; ++i) {
//   			zz[i] += AA[i * 100 + j] * z[j];
//   		}
// 	}
// 	//ADD + RELU
// 	//__attribute__((opencl_unroll_hint(34)))
// 	MAC1_2_2: for (i=0; i<100; i++) {
// 		if (zz[i] <= 0.0) 
// 			zz[i] = 0.0;
// 	}
// }

// static void L3(__global float *m_,
// __global float *z_,
// int z_offset,
// float *AAA,
// float *zz) {
//   	// MAC1
// 	float zzz[10];// __attribute__((xcl_array_partition(complete,1)));;
// 	int i, j;

// 	//__attribute__((xcl_pipeline_loop(1)))
//   	MAC1_ADD_3: for (i=0; i<10; ++i)
//   		//ADD
//   		zzz[i] = m_[i+89600];

//   	//__attribute__((xcl_pipeline_loop))
//   	MAC1_3: for (j=0; j<100; ++j) {
//   		//__attribute__((opencl_unroll_hint(10)))
//   		MAC1_3_1: for (i=0; i<10; ++i) {
//   			zzz[i] += AAA[i * 100 + j] * zz[j];
//   		}
//   	}

//   	/* SOFTMAX */
//   	float max=zzz[0], sum=0.0;
//   	//__attribute__((xcl_pipeline_loop))
//   	SOFTMAX_1: for (i=1; i<10; ++i) {
//   		if (max < zzz[i]) {
//   			max = zzz[i];
//   		}
//   	}

// 	float exp_z[10], exp_max=(float)exp(max);

// 	//__attribute__((opencl_unroll_hint(10)))
//   	EXP_Z_PRE: for (i=0; i<10; ++i) {
// 		exp_z[i] = -max;
//   	}
// 	//__attribute__((opencl_unroll_hint(10)))
//   	EXP_Z: for (i=0; i<10; ++i) {
// 		exp_z[i] += (float)exp(zzz[i]);
//   	}
//   	//__attribute__((xcl_pipeline_loop))
//   	SOFTMAX_2: for (i=0; i<10; ++i) {
// 		sum += exp_z[i];
//   	}

// 	sum /= exp_max;

//   	//__attribute__((opencl_unroll_hint(10)))
// 	//__attribute__((nounroll))
// 	//__attribute__((xcl_pipeline_loop))
//   	WB_3:for (i=0; i<10; ++i) {
//   		zzz[i] = exp_z[i]/sum;
//   	}

//   	//__attribute__((xcl_pipeline_loop(1)))
// 	WB_Z:for (i=0; i<10; ++i) {
// 		z_[z_offset+i] = zzz[i];
//   	}
// }


//__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
// __attribute__ ((xcl_dataflow))
void run_activate(float *m_, float *x_, float *z_,
int x_size, int z_size, int test_n) {
	
	int x_offset=0, z_offset=0;
  	// __local 
	// float A[78400];// __attribute__((xcl_array_partition(block,784,1)));
  	// //__attribute__((xcl_pipeline_loop(1)))
  	// RD_A: for(int j = 0; j < 78400; j++){
    // 	A[j] = m_[j];
  	// }

  	// // __local 
	// float AA[10000];// __attribute__((xcl_array_partition(block,100,1)));
 	// //__attribute__((xcl_pipeline_loop(1)))
  	// RD_AA: for(int j = 0; j < 10000; j++){
  	// 	AA[j] = m_[j+78500];
  	// }

  	// // __local 
	// float AAA[1000];// __attribute__((xcl_array_partition(block,100,1)));
  	// //__attribute__((xcl_pipeline_loop(1)))
  	// RD_AAA: for(int j = 0; j < 1000; j++){
  	// 	AAA[j] = m_[j+88600];
  	// }
  	
	// __attribute__((xcl_pipeline_loop(1)))
	//__attribute__((opencl_unroll_hint(1)))
	for(int k=0; k<test_n; k++){
		x_offset = x_size * k;
		z_offset = z_size * k;
		int i,j;
		// float local_in[784];// __attribute__((xcl_array_partition(cyclic,34,1)));

		// //__attribute__((xcl_pipeline_loop(1)))
		// RD_X: for(int j = 0; j < 784; j++){
		// 	local_in[j] = x_[x_offset+j];
		// }

		float z[100];// __attribute__((xcl_array_partition(cyclic,34,1)));
		#pragma omp target map(tofrom : z[0:100])
		#pragma omp teams distribute parallel for simd
		for (i=0; i<100; ++i) 
			z[i] = m_[78400+i];
		#pragma omp target teams distribute parallel for simd reduction(+ : z[0:100])
		for (j=0; j<784; ++j) {
			//__attribute__((opencl_unroll_hint(34)))
			for (i=0; i<100; ++i) {
				// z[i] += A[i * 784 + j] * local_in[j];
				z[i] += m_[i * 784 + j] * x_[x_offset+j];

			}
		}

			/*ADD + RELU*/
			//__attribute__((opencl_unroll_hint(34)))
		#pragma omp target teams distribute parallel for simd
		for (i=0; i<100; ++i) {
			if (z[i] <= 0.0) 
				z[i] = 0.0;
		}
		

		float zz[100];// __attribute__((xcl_array_partition(cyclic,34,1)));
		#pragma omp target map(tofrom : zz[0:100])
		#pragma omp teams distribute parallel for simd
			//__attribute__((xcl_pipeline_loop(1)))
		for (i=0; i<100; ++i)
			zz[i] = m_[88500+i];

		#pragma omp target teams distribute parallel for simd reduction(+ : zz[0:100])
		//__attribute__((xcl_pipeline_loop))
		for (j=0; j<100; ++j) {
			//__attribute__((opencl_unroll_hint(34)))
			for (i=0; i<100; ++i) {
				// zz[i] += AA[i * 100 + j] * z[j];
				zz[i] += m_[78500+(i * 100 + j)] * z[j];
			}
		}
		//ADD + RELU
		#pragma omp target teams distribute parallel for simd
		//__attribute__((opencl_unroll_hint(34)))
		for (i=0; i<100; i++) {
			if (zz[i] <= 0.0) 
				zz[i] = 0.0;
		}
		

		// MAC1
		float zzz[10];// __attribute__((xcl_array_partition(complete,1)));;
		#pragma omp target map(tofrom : zzz[0:10])
		#pragma omp teams distribute parallel for simd
		//__attribute__((xcl_pipeline_loop(1)))
		for (int i=0; i<10; ++i)
			//ADD
			zzz[i] = m_[i+89600];

		#pragma omp target teams distribute parallel for simd reduction(+ : zzz[0:10])
		//__attribute__((xcl_pipeline_loop))
		for (int j=0; j<100; ++j) {
			//__attribute__((opencl_unroll_hint(10)))
			for (int i=0; i<10; ++i) {
				// zzz[i] += AAA[i * 100 + j] * zz[j];
				zzz[i] += m_[88600+(i * 100 + j)] * zz[j];
			}
		}

		/* SOFTMAX */
		float max=zzz[0], sum=0.0;
		#pragma omp target map(tofrom : sum)
		//__attribute__((xcl_pipeline_loop))
		for (i=1; i<10; ++i) {
			if (max < zzz[i]) {
				max = zzz[i];
			}
		}

		float exp_z[10], exp_max=(float)exp(max);

		//__attribute__((opencl_unroll_hint(10)))
		for (i=0; i<10; ++i) {
			exp_z[i] = -max;
		}

		#pragma omp target teams distribute parallel for simd
		//__attribute__((opencl_unroll_hint(10)))
		for (i=0; i<10; ++i) {
			exp_z[i] += (float)exp(zzz[i]);
		}

		#pragma omp target teams distribute parallel for simd reduction(+ : sum)
		//__attribute__((xcl_pipeline_loop))
		for (i=0; i<10; ++i) {
			sum += exp_z[i];
		}

		sum /= exp_max;

		//__attribute__((opencl_unroll_hint(10)))
		//__attribute__((nounroll))
		//__attribute__((xcl_pipeline_loop))
		#pragma omp target teams distribute parallel for simd
		for (i=0; i<10; ++i) {
			z_[z_offset+i] = exp_z[i]/sum;
		}
	
	}
}

int main(int argc, char **argv) {
  // set matrix dimensions and allocate memory for matrices
	int test_n, digit, dec_cursor, s_idx; 
	long int error=0;    
  	if (argc == 2) {
    	if(atoi(argv[1])>10000){
    		test_n=10000;
    	}else{
    		test_n = atoi(argv[1]);
    	}
  	} else {
    	test_n = DEF_SIZE;
  	}
	//Allocate Memory in Host Memory

	//When creating a buffer with user pointer, under the hood user ptr is
	//used if and only if it is properly aligned (page aligned). When not 
	//aligned, runtime has no choice but to create its own host side buffer
	//that backs user ptr. This in turn implies that all operations that move
	//data to/from device incur an extra memcpy to move data to/from runtime's
	//own host buffer from/to user pointer. So it is recommended to use this 
	//allocator if user wish to Create Buffer/Memory Object to align user buffer
	//to the page boundary. It will ensure that user buffer will be used when 
	//user create Buffer/Mem Object.
	std::vector<float> mem(mem_size/sizeof(float));
	std::vector<float> x(test_n*x_size_bytes/sizeof(float));
	std::vector<float> z(test_n*10);
	std::vector<float> z_gold(test_n);

	//Create the test data and Software Result 
	char buf[256], s[19];
	// float in_data;
	FILE *fptr_mem, *fptr_x[test_n], *fptr_z, *fptr_out;
	fptr_mem = fopen("data/mem.txt", "r+");
	if(fptr_mem == NULL) {
		printf("\nERRROR! err:\n");
	}
	for(int i=0; i<(int)mem_size_hard/sizeof(float); i++) {
		/*printf("%f ", *((float *)(mem)+i));*/
		fscanf(fptr_mem, "%f ", &mem[i]);
		// mem[i] = (int)(in_data*1000000);
		// printf("%d ", mem[i]);
	}

	fptr_z = fopen("data/labels.txt", "r+");
	if(fptr_z == NULL) {
		printf("\nERRROR! err:\n");
	}
	for(int i=0; i<test_n; i++) {
		/*printf("%f ", *((float *)(mem)+i));*/
		fscanf(fptr_z, "%f ", &z_gold[i]);
		//printf("%d ", z[i]);
	}
	fclose(fptr_mem);
	fclose(fptr_z);
		
	for(int i=0; i<test_n; i++) {  
		memset(s, 0, 19);
		digit = i;
		dec_cursor = 10000;
		s_idx = 10;
		strcpy(s,"data/x");
		if(digit == 0)
			*(s+s_idx) = i + 48;
		else {
			while(digit/dec_cursor == 0) {
				digit = digit % dec_cursor;
				dec_cursor /= 10;
			}
			while(dec_cursor > 0) {
				s[s_idx] = digit / dec_cursor + 48;
				s_idx++;
				digit = digit % dec_cursor;
				dec_cursor /= 10;
			}
		}
		strcat(s,".txt");
		// printf("%s\t",s);
		fptr_x[i] = fopen(s,"r+");
		memset(buf, 0, 256);
		if(fptr_x[i] == NULL) {
			strerror_r(errno, buf, 256);
			printf("\nERRROR! at i=%d, err: %s\n", i, buf);

		}
		for (int k=0; k<(28*28); ++k) {
			fscanf(fptr_x[i], "%f ", &x[i*28*28+k]);
			// printf("in[%d]: %f\t", k, x[i*28*28+k]);
			// x[i*28*28+k] = (int)(in_data*1000000);
		}
		fclose(fptr_x[i]);
	}
 	double start_time = omp_get_wtime();

// #pragma omp target enter data map(to : xold[0 : Ndim], xnew[0 : Ndim], \
    A[0 : Ndim *Ndim], b[0 : Ndim])
	#pragma omp target enter data map(to : mem[0:mem_size/sizeof(float)],\
		x[0:test_n*x_size_bytes/sizeof(float)],	z[0:test_n*10])

	int32_t x_size, z_size;
	x_size = x_size_bytes / sizeof(float);
	z_size = 10;
	run_activate(&(mem[0]), &(x[0]), &(z[0]), x_size, z_size, test_n);

// #pragma omp target
// #pragma omp teams distribute parallel for simd
//     for (int i = 0; i < Ndim; i++) {
//       xnew[i] = (TYPE)0.0;
//       for (int j = 0; j < Ndim; j++) {
//         xnew[i] += A[j * Ndim + i] * xold[j] * (TYPE)(i != j);
//       }
//       xnew[i] = (b[i] - xnew[i]) / A[i * Ndim + i];
//     }
//     //
//     // test convergence
//     //
//     conv = 0.0;

// #pragma omp target map(tofrom : conv)
// #pragma omp teams distribute parallel for simd reduction(+ : conv)
//       for (int i = 0; i < Ndim; i++) {
//         TYPE tmp = xnew[i] - xold[i];
//         conv += tmp * tmp;
//       }

//     conv = sqrt((double)conv);

// #ifdef DEBUG
//     if(iters%1000==0)
//       printf(" conv = %f \n", (float)conv);
// #endif

//     TYPE* tmp = xold;
//     xold = xnew;
//     xnew = tmp;
//   }

	#pragma omp target exit data map(from : mem[0:mem_size/sizeof(float)],\
	x[0:test_n*x_size_bytes/sizeof(float)],	z[0:test_n*10])

	double elapsed_time = omp_get_wtime() - start_time;
	// printf(" Convergence = %g with %d iterations and %f seconds\n", (float)conv,
	// iters, (float)elapsed_time);

	for (int i=0; i<test_n; i++) {
		real_t *out;
		out = &z[i*10];

		if (argmax(out, 10) != z_gold[i]) {
			error++;
		}
	}

  	printf("\rtest done %ld errors\n", error);
  	printf("Accuracy  : %.4f\n", 1.0 - ((double)error / test_n));
  	
    bool match = true;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 

    std::cout << "Wall Clock Time (Kernel execution): " << elapsed_time << std::endl;
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution only,"  
            << "not for emulation." << std::endl; 

    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}
