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
    
void run_activate(float *m_, float *x_, float *z_,
int x_size, int z_size, int test_n) {
	
	int x_offset=0, z_offset=0;

	for(int k=0; k<test_n; k++){
		x_offset = x_size * k;
		z_offset = z_size * k;
		int i,j;


		float z[100];
		#pragma omp target map(tofrom : z[0:100])
		#pragma omp teams distribute parallel for simd
		for (i=0; i<100; ++i) 
			z[i] = m_[78400+i];
		#pragma omp target teams distribute parallel for simd reduction(+ : z[0:100])
		for (j=0; j<784; ++j) {
			for (i=0; i<100; ++i) {
				z[i] += m_[i * 784 + j] * x_[x_offset+j];
			}
		}

		#pragma omp target teams distribute parallel for simd
		for (i=0; i<100; ++i) {
			if (z[i] <= 0.0) 
				z[i] = 0.0;
		}
		

		float zz[100];
		#pragma omp target map(tofrom : zz[0:100])
		#pragma omp teams distribute parallel for simd
		for (i=0; i<100; ++i)
			zz[i] = m_[88500+i];

		#pragma omp target teams distribute parallel for simd reduction(+ : zz[0:100])
		for (j=0; j<100; ++j) {
			for (i=0; i<100; ++i) {
				zz[i] += m_[78500+(i * 100 + j)] * z[j];
			}
		}
		//ADD + RELU
		#pragma omp target teams distribute parallel for simd
		for (i=0; i<100; i++) {
			if (zz[i] <= 0.0) 
				zz[i] = 0.0;
		}
		

		// MAC1
		float zzz[10];
		#pragma omp target map(tofrom : zzz[0:10])
		#pragma omp teams distribute parallel for simd
		for (int i=0; i<10; ++i)
			//ADD
			zzz[i] = m_[i+89600];

		#pragma omp target teams distribute parallel for simd reduction(+ : zzz[0:10])
		for (int j=0; j<100; ++j) {
			for (int i=0; i<10; ++i) {
				zzz[i] += m_[88600+(i * 100 + j)] * zz[j];
			}
		}

		/* SOFTMAX */
		float max=zzz[0], sum=0.0;
		#pragma omp target map(tofrom : sum)
		for (i=1; i<10; ++i) {
			if (max < zzz[i]) {
				max = zzz[i];
			}
		}

		float exp_z[10], exp_max=(float)exp(max);

		for (i=0; i<10; ++i) {
			exp_z[i] = -max;
		}

		#pragma omp target teams distribute parallel for simd
		for (i=0; i<10; ++i) {
			exp_z[i] += (float)exp(zzz[i]);
		}

		#pragma omp target teams distribute parallel for simd reduction(+ : sum)
		for (i=0; i<10; ++i) {
			sum += exp_z[i];
		}

		sum /= exp_max;

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
		fscanf(fptr_mem, "%f ", &mem[i]);
	}

	fptr_z = fopen("data/labels.txt", "r+");
	if(fptr_z == NULL) {
		printf("\nERRROR! err:\n");
	}
	for(int i=0; i<test_n; i++) {
		fscanf(fptr_z, "%f ", &z_gold[i]);
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
		fptr_x[i] = fopen(s,"r+");
		memset(buf, 0, 256);
		if(fptr_x[i] == NULL) {
			strerror_r(errno, buf, 256);
			printf("\nERRROR! at i=%d, err: %s\n", i, buf);

		}
		for (int k=0; k<(28*28); ++k) {
			fscanf(fptr_x[i], "%f ", &x[i*28*28+k]);
		}
		fclose(fptr_x[i]);
	}
 	double start_time = omp_get_wtime();

	#pragma omp target enter data map(to : mem[0:mem_size/sizeof(float)],\
		x[0:test_n*x_size_bytes/sizeof(float)],	z[0:test_n*10])

	int32_t x_size, z_size;
	x_size = x_size_bytes / sizeof(float);
	z_size = 10;
	run_activate(&(mem[0]), &(x[0]), &(z[0]), x_size, z_size, test_n);

	#pragma omp target exit data map(from : mem[0:mem_size/sizeof(float)],\
	x[0:test_n*x_size_bytes/sizeof(float)],	z[0:test_n*10])

	double elapsed_time = omp_get_wtime() - start_time;

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
