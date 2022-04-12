#include <stdio.h>
#include <stdlib.h>
#include "sync_utils.h"

static int limits[SETS] = {0};
///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU utility and driver functions

__global__ void Spy (unsigned long *spy, unsigned long *ts, unsigned long *start, int its2) {
	int b_id = blockIdx.x;
    // printf ("BID: %d, SM: %d\n", b_id, get_smid ());
	int s_index = b_id * (BUCKETS / BLOCKS), t1 = 0, t2 = 0, i, k;
    unsigned long s1 = start[s_index], s2 = start[s_index + 1], s3 = start[s_index + 2];
    // int *lock = (int *) &spy[start[BUCKETS]];
    unsigned long start_time, end_time, p, loop, duration;

    __shared__ unsigned short data[BITS_TO_SEND + SPY_PADDING];
    /*__shared__*/ unsigned long s_out;

    s_out = warmup (spy, start + s_index + 2, 1, REPEAT);
    s_out += warmup (spy, start + s_index, 1, REPEAT);
    for (k = 0; k < BITS_TO_SEND + SPY_PADDING; k++) {

        /* wait() */
        p = s1;
        loop = 0;
        do {
            start_time = clock ();
            for (i = 0; i < its2 * REPEAT; i++) {
                p = spy[p];
                t1 += p;
                // t2 += t1;
            }
            end_time = clock ();
            s_out += t1 + t2;
            duration = (end_time - start_time)/(its2 * REPEAT);
            loop++;
        } while ((duration < LATENCY_THRESHOLD)/* && loop < ITER_LIMIT*/);

        /* msg[i] <- probe */
        p = s3;
        start_time = clock ();
        for (i = 0; i < its2 * REPEAT; i++) {
            p = spy[p];
            t1 += p;
            // t2 += t1;
        }
        end_time = clock ();
        s_out += t1 + t2;

        /* Use shared memory to do these changes, do not interfere
           with the TLB state while storing information! */
        data[k] = (end_time - start_time) / (its2 * REPEAT);

        /* AcceptedSignal() */
        p = s2;
        for (i = 0; i < its2 * REPEAT; i++) {
            p = spy[p];
            t1 += p;
            // t2 += t1;
        }
        s_out += t1 + t2;
        // printf("[%d] D: %d L: %lld\n", k, duration, loop);
    }

    // atomicAdd (lock, 1);
    // while (atomicAdd (lock, 0) != BLOCKS);

    // Copy from shared memory to ts to return message!
    for (k = 0; k < BITS_TO_SEND + SPY_PADDING; k++) {
        ts[b_id * (BITS_TO_SEND + SPY_PADDING) + k] = data[k];
    }
    ts[BLOCKS * (BITS_TO_SEND + SPY_PADDING) + b_id] = s_out;
    __threadfence();
    printf("[%ds]-Finish\n", b_id);
}


void cmem_stride () {

    cudaError_t error_id;
    unsigned long e_size = sizeof (unsigned long);
    unsigned long a_size = (8 * GB) / e_size;
    unsigned long stride = (1 * MB) / e_size;

    cudaSetDevice (DEVICE);
    /* allocate arrays on GPU, using cudaMallocManaged for change, might have to use cudaMalloc */
    unsigned long *d_spy;
    error_id = cudaMallocManaged ((void **) &d_spy, e_size * a_size);
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    unsigned long *d_ts;
    error_id = cudaMallocManaged ((void **) &d_ts, BLOCKS * (BITS_TO_SEND + SPY_PADDING + 2) * e_size);
    if (error_id != cudaSuccess) {
        printf ("Error 1.1 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    unsigned long *s_spy;
    error_id = cudaMallocManaged ((void **) &s_spy, e_size * (BUCKETS + 6));
    if (error_id != cudaSuccess) {
        printf ("Error 1.2 is %s\n", cudaGetErrorString (error_id));
        return;
    }

    create_pattern (d_spy, a_size, stride, s_spy, limits);

    dim3 block_spy = dim3 (THREADS);
    dim3 grid_spy = dim3 (BLOCKS, 1, 1);

    cudaStream_t stream2, stream4;
    cudaStreamCreate (&stream2);
    cudaStreamCreate (&stream4);

    setPrefetchAsync (d_spy, s_spy, &stream2, /*SETS*/BUCKETS);
    cudaStreamSynchronize (stream2);

    float t1;
    cudaEvent_t start, end;
    cudaEventCreate (&start);
    cudaEventCreate (&end);

    Timer timer;
    int its = ITER;
    startTime (&timer);
    cudaEventRecord (start, stream2);
    // reciver
    l_warmup<<<1, 1, 0, stream2>>> (d_spy, s_spy);
    Spy<<<grid_spy, block_spy, 0, stream2>>> (d_spy, d_ts, s_spy, its);
    cudaEventRecord (end, stream2);
    cudaEventSynchronize (end);
    // cudaDeviceSynchronize ();
    stopTime (&timer);
    cudaEventElapsedTime (&t1, start, end);

    float s = elapsedTime(timer);

    for (int j = 0; j < BLOCKS; j++) {
        for (int i = 0; i < (BITS_TO_SEND + SPY_PADDING); i++) {
            int bit = j * (BITS_TO_SEND + SPY_PADDING) + i;
            printf ("k: %d latency, %.3f clk\n", bit, (float) d_ts[bit]);
        }
        printf("=====BLOCK LIMITER=====\n");
    }

    printf ("Res: %lu\n", d_ts[BLOCKS * (BITS_TO_SEND + SPY_PADDING)]);
    printf ("[END] %f ms, %f s, %f bps\n", t1, s, BLOCKS * BITS_TO_SEND/s);
    printf ("\n");
    cudaFree (d_spy);
    cudaFree (s_spy);
    cudaFree (d_ts);
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

