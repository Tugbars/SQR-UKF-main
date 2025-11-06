#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
  #include <windows.h>
  #include <malloc.h>
  #include <immintrin.h>
#else
  #include <time.h>
  #include <pthread.h>
  #include <unistd.h>
  #include <immintrin.h>
#endif

#include "../gemm/gemm.h"

static double now_sec(void){
#if defined(_WIN32)
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart/(double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
#endif
}

static void disable_denormals(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#ifdef _MM_DENORMALS_ZERO_ON
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

static void* aligned_alloc32(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 32);
#else
    void* p = NULL; if (posix_memalign(&p, 32, size)!=0) return NULL; return p;
#endif
}
static void aligned_free32(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static void naive(float* C, const float* A, const float* B, int M,int K,int N){
    for (int i=0;i<M;++i){
        for (int j=0;j<N;++j){
            float s=0.f;
            const float* arow = A + (size_t)i*K;
            for (int k=0;k<K;++k) s += arow[k]*B[(size_t)k*N + j];
            C[(size_t)i*N + j] = s;
        }
    }
}

static float rel_err_max(const float* x, const float* y, size_t n){
    float maxe=0.f, meanref=0.f;
    for(size_t i=0;i<n;++i){
        float e=fabsf(x[i]-y[i]); if(e>maxe)maxe=e;
        meanref += fabsf(y[i]);
    }
    return maxe/((meanref/(float)n)+1e-20f);
}

static int cmp_d(const void* a,const void* b){
    double da=*(const double*)a, db=*(const double*)b;
    return (da<db)?-1: (da>db);
}

void bench_case(int M, int K, int N, int reps, int verify)
{
    disable_denormals();

    size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
    float *A=(float*)aligned_alloc32(szA*sizeof(float));
    float *B=(float*)aligned_alloc32(szB*sizeof(float));
    float *C=(float*)aligned_alloc32(szC*sizeof(float));
    float *Cref = verify ? (float*)aligned_alloc32(szC*sizeof(float)) : NULL;

    if(!A||!B||!C||(verify&&!Cref)){ fprintf(stderr,"alloc failed\n"); goto out; }

    // init
    for(size_t i=0;i<szA;++i) A[i] = (float)((int32_t)(i*1103515245u+12345u)) / (float)INT32_MAX;
    for(size_t i=0;i<szB;++i) B[i] = (float)((int32_t)(i*1664525u+1013904223u)) / (float)INT32_MAX;

    // warmup
    memset(C,0,szC*sizeof(float));
    mul(C,A,B,(uint16_t)M,(uint16_t)K,(uint16_t)K,(uint16_t)N);
    if(verify){
        memset(Cref,0,szC*sizeof(float));
        naive(Cref,A,B,M,K,N);
        float r = rel_err_max(C,Cref,szC);
        printf("verify %dx%dx%d : rel_err=%.2e %s\n", M,K,N, r, (r<1e-3f?"OK":"!!"));
    }

    double *to=(double*)malloc(reps*sizeof(double));
    double *tn=(double*)malloc(reps*sizeof(double));
    if(!to||!tn){ fprintf(stderr,"times alloc failed\n"); goto out; }

    // measure optimized
    for(int r=0;r<reps;++r){
        memset(C,0,szC*sizeof(float));
        double t0=now_sec();
        mul(C,A,B,(uint16_t)M,(uint16_t)K,(uint16_t)K,(uint16_t)N);
        to[r]=now_sec()-t0;
    }
    // measure naive
    for(int r=0;r<reps;++r){
        memset(C,0,szC*sizeof(float));
        double t0=now_sec();
        naive(C,A,B,M,K,N);
        tn[r]=now_sec()-t0;
    }

    // stats
    qsort(to,reps,sizeof(double),cmp_d);
    qsort(tn,reps,sizeof(double),cmp_d);
    double sumo=0,sumn=0; for(int i=0;i<reps;++i){ sumo+=to[i]; sumn+=tn[i]; }
    double mino=to[0], medo=to[reps/2], meano=sumo/reps;
    double minn=tn[0], medn=tn[reps/2], meann=sumn/reps;

    double flops = 2.0*(double)M*(double)N*(double)K;

    printf("\n=== BENCH %dx%dx%d (reps=%d) ===\n", M,K,N,reps);
    printf("Optimized : min %.3f ms | med %.3f ms | mean %.3f ms | %.1f–%.1f GF/s\n",
           1e3*mino,1e3*medo,1e3*meano,(flops/meano)*1e-9,(flops/mino)*1e-9);
    printf("Naive     : min %.3f ms | med %.3f ms | mean %.3f ms | %.1f–%.1f GF/s\n",
           1e3*minn,1e3*medn,1e3*meann,(flops/meann)*1e-9,(flops/minn)*1e-9);
    printf("Speedup   : med x%.2f | best x%.2f\n", medn/medo, minn/mino);

    free(to); free(tn);

out:
    aligned_free32(A); aligned_free32(B); aligned_free32(C); aligned_free32(Cref);
}

void run_benchmark_suite(void)
{
    const int reps = 9, verify = 0;
    const int cases[][3] = {
        {64,64,64}, {128,128,128}, {128,256,256},
        {256,256,256}, {384,64,384}, {512,512,512},
        {129,257,255}, {100,200,150}
    };
    for (size_t i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
        bench_case(cases[i][0], cases[i][1], cases[i][2], reps, verify);
    }
}
