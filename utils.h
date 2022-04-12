# include "cuda_runtime.h"
# define DEVICE 0

#define GB (1024UL*1024UL*1024UL)
#define MB (1024UL*1024UL)
#define KB (1024UL)

/* 4KB smallest page size, 12 bits of offset. But we never see it's usage
   64KB is the next page size, 16 bits of offset.! */
/* If it's COLT, then doesn't it mean that we have to take offset from a different number? */
# define OFFSET 20
# define ZERO_MASK 0xffffffff0000LL

static int pascal_hash (long long addr, int bits, int sets) {
    /* Clear the least significant OFFSET bits! */
    unsigned long temp = addr & ZERO_MASK;
    temp >>= OFFSET;
    int hash = 0x0;
    int SETS = sets;
    int SHIFT = bits;
    int MASK = 0;
    /* Form mask with <b>bits</b> set as 1 */
    for (int i = 0; i < bits; i++) {
        MASK <<= 1;
        MASK |= 1;
    }
    while (SETS > 0) {
        /* Utilizing MASK bits at a time. */
        hash ^= (temp & MASK);
        /* Discarding the SHIFT bits just used! */
        temp >>= SHIFT;
        SETS--;
    }
    /* Ensure that MASK bit value is returned! */
    return (hash & MASK);
}


/* This is most likely the hash function for pascal micro-architecture. [L1]*/
inline int pascal_hash_function_l1 (long long addr) {
    return pascal_hash (addr, 3, 7);
}


/* This is most likely the hash function for pascal micro-architecture. [L2]*/
inline int pascal_hash_function_l2 (long long addr) {
    return pascal_hash (addr, 7, 3);
}


void setPrefetchAsync (unsigned long *arr, unsigned long *start, cudaStream_t *stream, int sets) {
    unsigned long s, t;
    /* This is the least size observed of data movement */
    unsigned long prefetch_size = 64 * KB;
    for (int i = 0; i < sets; i++) {
        s = start[i];
        t = s;
        do {
            cudaMemPrefetchAsync (&arr[t], prefetch_size, DEVICE, *stream);
            t = arr[t];
        } while (t != s);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline uint get_smid (void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}


__device__ unsigned long delay (long long clocks) {
    unsigned long start = clock ();
    while (clock () - start <= clocks);
    return start;
}


__device__ __noinline__  unsigned long warmup (unsigned long *arr, unsigned long* start, int sets, int repeat) {
    unsigned long temp, result = 0;
    for (int i = 0 ; i < sets; i++) {
        temp = start[i];
        for (int j = 0; j < repeat; j++)
            temp = arr[temp];
        result += temp;
    }
    return result;
}
