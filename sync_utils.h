#include <sys/time.h>
#include <time.h>
#include "utils.h"

#define REPEAT 17.0
#define ITER_LIMIT 60UL

#define BLOCKS 14
#define THREADS 1
#define BITS_TO_SEND 20
#define SPY_PADDING 10
#define DEVICE 0
/**
 * This should be latency for L2 hit, anymore than that means atleast a single set
 * from L2 is thrashing! <380>
 */
#define LATENCY_THRESHOLD 450
#define ITER 7

#define SETS 128
#define COLUMNS 50
#define DEF_ENTRIES 5
#define NEXT_IDX 0
#define NEXT_IDX_VAL 1
#define DEF_VAL -1LL
#define BUCKETS 42


typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;


void startTime (Timer* timer) {
    gettimeofday (&(timer->startTime), NULL);
}


void stopTime (Timer* timer) {
    gettimeofday (&(timer->endTime), NULL);
}


float elapsedTime (Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                   + (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}


void _add(long long *temp, long long *sidx, int *start, int *count, 
          long long *indices, unsigned long *d_a, unsigned long *s_arr) {

    /* Check for invalidity, and initialize. */
    if (*temp == DEF_VAL) {
        *sidx = indices[NEXT_IDX_VAL];
        *temp = *sidx;
    }
    /* Add all the elements from this set to the pattern */
    for (int j = NEXT_IDX_VAL; j < indices[NEXT_IDX]; j++) {
        d_a[*temp] = indices[j];
        *temp = indices[j];
        *count += 1;
    }
    /* Check if necessary size is reached, if yes reset! */
    if (*count == REPEAT) {
        /* Roundabout pattern for circular access. */
        d_a[*temp] = *sidx;
        *temp = DEF_VAL;
        *count = 0;
        /* Save the pattern point so that kernels can use it! */
        // printf("NS:%d\n", *start);
        s_arr[*start] = *sidx;
        /* Tell where to insert next. */
        // *start += 2;
        *start += 1;
    }
}


void create_pattern (unsigned long *d_a, unsigned long a_size, unsigned long stride,
                     unsigned long *start, int *limits) {
    /* Reset for every call! */
    long long addr, i, j, temp = 0;
    int hash, hash_l1, hash_l2;
    long long indices[SETS][COLUMNS];
    /* Init the indices for the thingy if it's called multiple times! */
    for (i = 0; i < SETS; i++) {
        indices[i][NEXT_IDX] = NEXT_IDX_VAL;
    }
    for (i = 0; i < a_size; i += stride) {
        // Check if L2 interference is happening?
        addr = (long long) &d_a[i];
        hash_l1 = pascal_hash_function_l1 (addr);
        hash_l2 = pascal_hash_function_l2 (addr);
        /* This if case is for targeting L2 TLB, comment it out if playing around
           with L1. Also hash <= hash_l1 */
        if ((hash_l1 != 0 && hash_l1 != 1) || hash_l2 >= SETS)
            continue;

        hash = hash_l2;
        /* This index tells where to insert next */
        if (indices[hash][NEXT_IDX] <= limits[hash]) {
            // printf ("[%d],%lld\n", hash, i);
            temp = indices[hash][NEXT_IDX];
            indices[hash][temp] = i;
            indices[hash][NEXT_IDX] += 1;
            // i += 16;
        }
        for (j = 0; j < SETS; j++) {
            if (indices[j][NEXT_IDX] <= limits[j])
                break;
        }
        if (j == SETS)
            break;
    }

    long long temp_0 = DEF_VAL, temp_1 = DEF_VAL;
    long long sidx_0, sidx_1;
    int start_0 = 0, start_1 = 1;
    int count_0 = 0, count_1 = 0;
    for (i = 0; i < SETS; i++) {
        if (indices[i][NEXT_IDX] > NEXT_IDX_VAL) {
            hash_l1 = pascal_hash_function_l1 ((long long) &d_a[indices[i][NEXT_IDX_VAL]]);
            // printf ("L1:%d,L2:%lld,E:%lld\n", hash_l1, i, indices[i][NEXT_IDX]);
            if (hash_l1 == 0) {
                _add (&temp_0, &sidx_0, &start_0, &count_0, indices[i], d_a, start);
            } else if (hash_l1 == 1) {
                _add (&temp_1, &sidx_1, &start_0, &count_1, indices[i], d_a, start);
            } else {
                printf ("[Error] Pray that code won't reach here!");
            }
        }
    }
}


int get_set_size (int i) {
    /* At the moment we are going with 6 set pattern with 6 6 6 6 5 5 for 
       consecutive 6 sets of L2. Any change of pattern can be done here and it will
       reflect in both spy and trojan 
       Any change here needs to make sure that REPEAT has to atleast access all the
       elements once, in this case (17 ==> 6 6 5)
       Another possibility is (17 ==> 8 9) */
       return (i % 6 < 4) ? DEF_ENTRIES + 1 : DEF_ENTRIES;
       // return (i % 4 < 2) ? 9 : 8;
}


__global__ void l_warmup (unsigned long *arr, unsigned long *start) {
    unsigned long res = warmup (arr, start, BUCKETS, REPEAT);
    __threadfence();
    arr[1] = res;
}
