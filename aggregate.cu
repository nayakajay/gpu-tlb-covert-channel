//compile nvcc *.cu -o test

# include <stdio.h>
# include <stdlib.h>
# include <stdint.h>
# include <time.h>
# include "cuda_runtime.h"

# define INVALID -1LL
# define ITER 4
# define L2_ENTRIES 64
# define GB (1024LU*1024LU*1024LU)
# define MB (1024LU*1024LU)
# define KB (1024LU)


__global__ void global_latency (unsigned long *, unsigned long, unsigned long, unsigned long);
void measure_global (unsigned int, unsigned int, unsigned int);

void create_pattern (unsigned long *arr, unsigned long stride, unsigned long size, unsigned long start_idx) {
    unsigned long temp = start_idx, i;
    // [Style 1]: Jump for a stride
    for (i = start_idx; i < size; i++) {
        temp = (i + stride);
        // Each index points to next index!
        arr[i] = (temp >= size) ? start_idx : temp;
    }
    // [Style 2]: Jump twice as much as stride
    /*stride = 2 * stride;
    i = 0;
    while (i < L2_ENTRIES) {
        arr[temp] = temp + stride;
        temp = temp + stride;
        i++;
    }
    // New temp will be (start_idx + original stride) (was doubled earlier)
    arr[temp] = start_idx + (stride / 2);
    i = 0;
    temp = start_idx + (stride / 2);
    // Jump twice as much as stride
    while (i < L2_ENTRIES) {
        arr[temp] = temp + stride;
        temp = temp + stride;
        i++;
    }
    // Roundabout!
    arr[temp] = start_idx;*/
    // [Style 3]: To create a stride like (1, 2, 4, 1 ...) x stride
    /* int cycle = 3;
    srand (time (0));
    unsigned long next_idx = start_idx;
    temp = 2 ^ (rand () % cycle) * stride;
    while (next_idx + temp < size) {
        arr[next_idx] = next_idx + temp;
        iteration += 1;
        next_idx = next_idx + temp;
        temp = 2 ^ (rand () % cycle) * stride;
    }
    // Roundabout!
    arr[next_idx] = start_idx;*/
    // [Style 4]: To create L2 amount of accesses and then somewhere in between!
    /*for (i = 0; i < L2_ENTRIES; i++) {
        arr[temp] = (temp + stride) % size;
        temp = (temp + stride) % size;
    }
    i = start_idx + (stride / 2);
    arr[temp] = i;
    temp = i + (stride);
    arr[i] = temp;
    // Roundabout!
    arr[temp] = start_idx;*/
    // [Style 5]: To create random accesses at stride size!
    /* long long probe_points[size];
    int count, idx;
    // Find out all elements that needs to be accessed!
    for (i = 0, count = 0; i < size; i++, count++) {
        temp = start_idx + (i + 1) * stride;
        if (temp >= size)
            break;
        probe_points[i] = temp;
    }
    // Now create access pattern out of them!
    srand (time (0));
    temp = start_idx;
    while (i > 0) {
        idx = rand () % count;
        if (probe_points[idx] != INVALID) {
            arr[temp] = (unsigned long) probe_points[idx];
            temp = (unsigned long) probe_points[idx];
            probe_points[idx] = INVALID;
            i--;
        }
    }
    // Roundabout
    arr[temp] = start_idx;*/
}

int main (int argc, char **argv) {

    if (argc < 4) {
        printf ("Usage: %s   from_mb   to_mb   stride_size_kb\n", argv[0]);
        return 0;
    }

    /* Array size in mega bytes 1st argument. */
    unsigned int from_mb = (unsigned int) atof (argv[1]);

    /* Array size in mega bytes 1st argument. */
    unsigned int to_mb = (unsigned int) atof (argv[2]);

    /* Stride size in kilo bytes 1st argument. */
    unsigned int stride_size_kb = (unsigned int) atof (argv[3]);

    measure_global (from_mb, to_mb, stride_size_kb);
    return 0;
}


void measure_global (unsigned int from_mb, unsigned int to_mb, unsigned int stride_kb) {

    cudaError_t error_id;
    
    int e_size = sizeof (unsigned long);
    unsigned long l_limit = (from_mb * MB) / e_size;
    unsigned long u_limit = (to_mb * MB) / e_size;
    unsigned long stride = (stride_kb * KB) / e_size;

    cudaSetDevice (0);
    /* allocate arrays on GPU */    
    unsigned long *d_a;

    error_id = cudaMallocManaged ((void **) &d_a, e_size * (u_limit + 2));
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    /* Find a start address which is good, i.e which we need */
    unsigned long start_idx = 0, temp, elements, k, j;    
    printf ("ArraySize (KB), clock\n");
    for (k = l_limit; k <= u_limit; k += stride) {

        create_pattern (d_a, stride, k, start_idx);
        d_a[k + 1] = 0;
        d_a[k + 2] = 0;

        cudaDeviceSynchronize ();
        dim3 Db = dim3 (1);
        dim3 Dg = dim3 (1, 1, 1);

        /* launch kernel*/
        elements = (k - start_idx) / stride;
        global_latency <<<Dg, Db>>>(d_a, k, start_idx, elements);

        cudaDeviceSynchronize ();

        error_id = cudaGetLastError ();
        if (error_id != cudaSuccess) {
            printf ("Error kernel is %s\n", cudaGetErrorString (error_id));
        }

        int sum = 0;
        temp = start_idx;
        for (j = 0; j < elements; j++) {
            sum += d_a[temp + 1];
            // printf ("%lu\n", d_a[temp + 1]);
            /* Pointer chase! */
            temp = d_a[temp];
        }
        // printf ("===");
        printf ("%3.2f, %f\n", (float) e_size * (k - start_idx) / KB, (float) sum / (elements));
    }
    cudaDeviceReset();
    /* free memory on GPU */
    cudaFree (d_a);
}


__global__ void global_latency (unsigned long *my_array, unsigned long array_length, unsigned long s_idx, unsigned long elements) {
    unsigned long j = s_idx, old_j, k, start_time, end_time, res = 0, res_1 = 0, iter = ITER;

    // first round, warm the TLB
    for (k = 0; k < elements; k++) {
        j = my_array[j];
    }

    // second round, begin timestamp
    // j = (j != s_idx) ? s_idx : j;
    for (k = 0; k < elements * iter; k++) {
        old_j = j;
        start_time = clock ();
        j = my_array[j];
        res += j;
        res_1 += res;
        end_time = clock ();
        /* We just used it for accessing the element that has to be timed. It will be
           at a boundary of the stride chosen. Keep it atleast greater than cache line
           size. */
        my_array[old_j + 1] = (end_time - start_time);
    }

    my_array[j + 2/*array_length + 1*/] = res + res_1;
    my_array[j + 3/*array_length + 2*/] = my_array[j];
}
