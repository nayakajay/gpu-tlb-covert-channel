#ifndef __SUPPH__
#define __SUPPH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void verify(float *A, float *B, float *C, unsigned int m, unsigned int k,
  unsigned int n);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#endif
