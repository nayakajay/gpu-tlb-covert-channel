#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "sync_utils.h"

static int limits[SETS] = {0};
///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU utility and driver functions

__global__ void Trojan (unsigned long *trojan, unsigned long *out, unsigned long* start, int its2) {
	int b_id = blockIdx.x;
    // printf ("BID: %d, SM: %d\n", b_id, get_smid ());
	int s_index = b_id * (BUCKETS / BLOCKS), t1 = 0, t2 = 0, i, k;
    unsigned long s1 = start[s_index], s2 = start[s_index + 1], s3 = start[s_index + 2];
    long long start_time, end_time, p, loop, duration;

    __shared__ unsigned long s_out;

    s_out = warmup (trojan, start + s_index + 1, 1, 2 * REPEAT);
    for (k = 0; k < BITS_TO_SEND; k++) {

        p = s3;
        /* Change here to change the message. Rightt now, it's all 1s */
        if (1/*k % 2 == 1*/) {
            /* Do nothing or wait for some time? */
        } else {
            for (i = 0; i < its2 * REPEAT; i++) {
                p = trojan[p];
                t1 += p;
                // t2 += t1;
            }
            s_out += t1/* + t2*/;
        }

        p = s1;
        start_time = clock ();
        for (i = 0; i < its2 * REPEAT; i++) {
            p = trojan[p];
            t1 += p;
            // t2 += t1;
        }
        end_time = clock ();
        s_out += t1/* + t2*/;
        // data[k] = (end_time - start_time) / (its2 * REPEAT);
        s_out += end_time - start_time;

        p = s2;
        loop = 0;
        do {
            start_time = clock();
            for (i = 0; i < its2 * REPEAT; i++) {
                p = trojan[p];
                t1 += p;
                // t2 += t1;
            }
            end_time = clock();
            s_out += t1/* + t2*/;
            duration = (end_time - start_time)/(its2 * REPEAT);
            loop++;
        } while ((duration < LATENCY_THRESHOLD)/* && loop < ITER_LIMIT*/);
        // printf ("[%d] T: %lld, L: %lld\n", k, duration, loop);
        // printf ("-");
    }
    // Copy from shared memory to ts to return message!
    // for (i = 0; i < BITS_TO_SEND; i++) {
    //     trojan[b_id * BITS_TO_SEND + i] = data[i];
    // }
    out[b_id * BITS_TO_SEND] = s_out;
    // __threadfence();
    // printf("[%dT]-Finish\n", b_id);
}


void cmem_stride () {

    cudaError_t error_id;
    unsigned long e_size = sizeof (unsigned long);
    unsigned long a_size = (8 * GB) / e_size;
    unsigned long stride = (1 * MB) / e_size;

    cudaSetDevice (DEVICE);
    /* allocate arrays on GPU, using cudaMallocManaged for change, might have to use cudaMalloc */
    unsigned long *d_trojan;
    error_id = cudaMallocManaged ((void **) &d_trojan, e_size * a_size);
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    unsigned long *d_out;
     error_id = cudaMallocManaged ((void **) &d_out, BLOCKS * BITS_TO_SEND * e_size);
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    unsigned long *s_trojan;
    error_id = cudaMallocManaged ((void **) &s_trojan, e_size * (BUCKETS + 6));
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }

    create_pattern (d_trojan, a_size, stride, s_trojan, limits);

    dim3 block_trojan = dim3 (THREADS);
    dim3 grid_trojan = dim3 (BLOCKS, 1, 1);

    cudaStream_t stream1, stream3;
    cudaStreamCreate (&stream1);
    cudaStreamCreate (&stream3);

    setPrefetchAsync (d_trojan, s_trojan, &stream1, /*SETS*/BUCKETS);

    cudaStreamSynchronize (stream1);

    float t1;
    cudaEvent_t start, end;
    cudaEventCreate (&start);
    cudaEventCreate (&end);

    Timer timer;
    int its = ITER;
    startTime (&timer);
    cudaEventRecord (start, stream1);
    // sender
    l_warmup<<<1, 1, 0, stream1>>>(d_trojan, s_trojan);
    Trojan<<<grid_trojan, block_trojan, 0, stream1>>> (d_trojan, d_out, s_trojan, its);

    cudaEventRecord (end, stream1);
    cudaEventSynchronize (end);
    // cudaDeviceSynchronize ();
    stopTime (&timer);
    cudaEventElapsedTime (&t1, start, end);

    float s = elapsedTime(timer);

    printf ("Res: %lu\n", d_out[BLOCKS * BITS_TO_SEND]);
    printf ("[END] %f ms, %f s, %f bps\n", t1, s, BLOCKS * BITS_TO_SEND/s);
    printf ("\n");
    cudaFree (d_trojan);
    cudaFree (s_trojan);
    cudaFree (d_out);
}


int main (int argc, char **argv) {
    /* Setting global constants! index 0 is executable, start from 1! */ 
    for (int i = 0; i < SETS; i++) {
        limits[i] = get_set_size (i);
        if (i + 1 < argc)
            limits[i] = (int) atoi (argv[i + 1]);
    }

    cmem_stride ();
    cudaDeviceReset ();
    return 0;
}
