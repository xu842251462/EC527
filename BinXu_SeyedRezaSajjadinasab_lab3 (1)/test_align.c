/*************************************************************

You must use -O0 for this program to work as intended

   gcc -O0 -std=gnu99 test_align.c -lrt -o test_align

 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <mm_malloc.h>
#include <time.h>		// get time of day

#define OPTIONS 2

#define CPNS 2.0    /* Cycles per nanosecond -- Adjust to your computer,
                       for example a 3.2 GHz GPU, this would be 3.2 */

/* MODIFY the following line to use the values you determined
   when you worked on timers_1_scaling.c in assignment 0 */
#define TV_SCALING 1.0

#define ARR_SIZE 8*1024*1024    /* Large enough to exceed L3 cache size */
#define BOUNDARY_ALIGNMENT 64   /* cache block size */

#define OUTER_LOOPS 100
#define TEST_SIZE 10000



/* -=-=-=-=-= Time measurement by gettimeofday() -=-=-=-=- */

/* Turn the result from gettimeofday into a single number in units of seconds.
     gettimeofday is measuring the time since the UNIX epoch in 1970, which
   means we need a lot of digits to be able to measure intervals with
   microsecond resolution. Therefore we do the computation in "long double"
   which is good for about 19 decimal digits. */
#define timeval_to_secs(p_tv) \
   (   ( ((long double) (p_tv)->tv_sec) \
       + ((long double) (p_tv)->tv_usec) * 1.0e-6 ) \
   * TV_SCALING )
/*
     TV_SCALING constant is defined above

  How to use this method:

      struct timeval tv_start, tv_stop;
      gettimeofday(&tv_start, NULL);
      // DO SOMETHING THAT TAKES TIME
      gettimeofday(&tv_stop, NULL);
      measurement = timeval_to_secs(&tv_stop) - timeval_to_secs(&tv_start);
 */



/* -=-=-=-=- End of time measurement declarations =-=-=-=- */

/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int j;
  struct timeval tv_start, tv_stop;
  double quasi_random = 0;
  gettimeofday(&tv_start, 0);
  j = 100;
  while (meas < 1.0) {
    for (int i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    gettimeofday(&tv_stop, 0);
    meas = timeval_to_secs(&tv_stop) - timeval_to_secs(&tv_start);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/**************************************************************/
int main(int argc, char *argv[])
{
  int OPTION = 0;

  struct timeval tv_start, tv_stop;
  double time_stamp[OPTIONS][2*BOUNDARY_ALIGNMENT];
  long long int time_sec, time_ns;

  int ok;
  char *temp;
  double *x1, *x2, *x3;
  double wd = 0;

  wd = wakeup_delay();

#ifndef __APPLE__
  /* posix memalign array declaration -- x1 not used, demo only */
  ok = posix_memalign((void**)&x1, BOUNDARY_ALIGNMENT,
                                               ARR_SIZE * sizeof(double));
#endif

  /* a malloc() with an alignment wrapper works too */
  temp = (char*)_mm_malloc((ARR_SIZE+BOUNDARY_ALIGNMENT) * sizeof(double),
                                                    BOUNDARY_ALIGNMENT);

  x2 = (double*) (temp);  /* save original starting point for another test */

  printf("test_align  temp address = 0x%lx\n", (unsigned long)temp);

  /* Try different starting positions of x3, incrementing by 1 byte each iter.
     Use this to show what happens when you fetch doubles on addresses not
     aligned to doubles. */
  for (int k = 0; k < BOUNDARY_ALIGNMENT; k++) {
    x3 = (double*)(temp+k);
    gettimeofday(&tv_start, 0);
    for (int j=0; j<OUTER_LOOPS; j++) {

      for (int i = 0; i < TEST_SIZE; i++) {
        x3[i] += 1.3 + (double)(i);
      }

    }
    gettimeofday(&tv_stop, 0);
    time_stamp[OPTION][k] = timeval_to_secs(&tv_stop)-timeval_to_secs(&tv_start);
  }

  /* output times */
  printf("alignment, time\n");
  for (int j = 0; j < BOUNDARY_ALIGNMENT; j++) {
    printf(" %3d, %8ld\n", j,
                   (long int)((double)(CPNS) * 1.0e9 * time_stamp[OPTION][j]));
  }
  printf("\n");

  /* Again, try different starting positions of x3.  But this time, code in
     a way that has the following property:  In most cases it will cause one
     cache miss per fetch.  But for some values of k, each fetch causes two
     cache misses! */
  OPTION++;
  temp = (char*)(x2); // reset temp,
  for (int k = 0; k < BOUNDARY_ALIGNMENT; k++) { 
    x3 = (double*)(temp+k);                  
    gettimeofday(&tv_start, 0);
    for (int j=0; j<OUTER_LOOPS; j++) {

      /* NEW CODE GOES HERE */

    }
    gettimeofday(&tv_stop, 0);
    time_stamp[OPTION][k] = timeval_to_secs(&tv_stop)-timeval_to_secs(&tv_start);
  }

  /* output times */
  printf("alignment, time\n");
  for (int k = 0; k < BOUNDARY_ALIGNMENT; k++) {
    printf(" %3d, %8ld\n", k,
            (long int)((double)(CPNS)*(double) 1.0e9 * time_stamp[OPTION][k]));
  }
  printf("\n");

  printf("Wakeup delay calculated %f\n", wd);

} /* end main */