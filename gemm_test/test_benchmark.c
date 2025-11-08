/**
 * @file bench_gemm_planned.c
 * @brief Comprehensive benchmark for planning-based GEMM
 */

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

//==============================================================================
// BENCHMARK WITH PLANNING
//==============================================================================

void bench_case_planned(int M, int K, int N, int reps, int verify, const char* desc)
{
    disable_denormals();

    size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N;
    float *A=(float*)aligned_alloc32(szA*sizeof(float));
    float *B=(float*)aligned_alloc32(szB*sizeof(float));
    float *C=(float*)aligned_alloc32(szC*sizeof(float));
    float *Cref = verify ? (float*)aligned_alloc32(szC*sizeof(float)) : NULL;

    if(!A||!B||!C||(verify&&!Cref)){ 
        fprintf(stderr,"alloc failed\n"); 
        goto out; 
    }

    // Initialize matrices
    for(size_t i=0;i<szA;++i) A[i] = (float)((int32_t)(i*1103515245u+12345u)) / (float)INT32_MAX;
    for(size_t i=0;i<szB;++i) B[i] = (float)((int32_t)(i*1664525u+1013904223u)) / (float)INT32_MAX;

    // Verify correctness with auto dispatcher
    if(verify){
        memset(C,0,szC*sizeof(float));
        gemm_auto(C,A,B,M,K,N,1.0f,0.0f);
        memset(Cref,0,szC*sizeof(float));
        naive(Cref,A,B,M,K,N);
        float r = rel_err_max(C,Cref,szC);
        printf("verify %dx%dx%d : rel_err=%.2e %s\n", M,K,N, r, (r<1e-3f?"OK":"FAIL"));
        if(r >= 1e-3f) goto out;
    }

    double *t_auto = (double*)malloc(reps*sizeof(double));
    double *t_plan = (double*)malloc(reps*sizeof(double));
    double *t_exec = (double*)malloc(reps*sizeof(double));
    double *t_naive = (double*)malloc(reps*sizeof(double));
    
    if(!t_auto||!t_plan||!t_exec||!t_naive){ 
        fprintf(stderr,"times alloc failed\n"); 
        goto out; 
    }

    // 1. Benchmark AUTO dispatcher (includes routing overhead)
    for(int r=0;r<reps;++r){
        memset(C,0,szC*sizeof(float));
        double t0=now_sec();
        gemm_auto(C,A,B,M,K,N,1.0f,0.0f);
        t_auto[r]=now_sec()-t0;
    }

    // 2. Benchmark PLANNED (measure planning + execution)
    gemm_error_t error;
    for(int r=0;r<reps;++r){
        memset(C,0,szC*sizeof(float));
        double t0=now_sec();
        gemm_plan_t *plan = gemm_plan_create_safe(M,K,N,A,B,C,1.0f,0.0f,&error);
        gemm_execute_plan(plan,C,A,B,1.0f,0.0f);
        t_plan[r]=now_sec()-t0;
        gemm_plan_destroy(plan);
    }

    // 3. Benchmark EXECUTION ONLY (reuse plan - best case)
    gemm_plan_t *plan = gemm_plan_create_safe(M,K,N,A,B,C,1.0f,0.0f,&error);
    if(plan){
        for(int r=0;r<reps;++r){
            memset(C,0,szC*sizeof(float));
            double t0=now_sec();
            gemm_execute_plan(plan,C,A,B,1.0f,0.0f);
            t_exec[r]=now_sec()-t0;
        }
        gemm_plan_destroy(plan);
    } else {
        for(int r=0;r<reps;++r) t_exec[r] = t_plan[r];
    }

    // 4. Benchmark NAIVE
    for(int r=0;r<reps;++r){
        memset(C,0,szC*sizeof(float));
        double t0=now_sec();
        naive(C,A,B,M,K,N);
        t_naive[r]=now_sec()-t0;
    }

    // Statistics
    qsort(t_auto,reps,sizeof(double),cmp_d);
    qsort(t_plan,reps,sizeof(double),cmp_d);
    qsort(t_exec,reps,sizeof(double),cmp_d);
    qsort(t_naive,reps,sizeof(double),cmp_d);
    
    double min_auto=t_auto[0], med_auto=t_auto[reps/2];
    double min_plan=t_plan[0], med_plan=t_plan[reps/2];
    double min_exec=t_exec[0], med_exec=t_exec[reps/2];
    double min_naive=t_naive[0], med_naive=t_naive[reps/2];

    double flops = 2.0*(double)M*(double)N*(double)K;

    printf("\n=== %s: %dx%dx%d (reps=%d) ===\n", desc, M,K,N,reps);
    printf("Auto      : %.3f ms (%.1f GF/s)\n", 
           1e3*med_auto, (flops/med_auto)*1e-9);
    printf("Plan+Exec : %.3f ms (%.1f GF/s) - Planning overhead: %.3f ms\n", 
           1e3*med_plan, (flops/med_plan)*1e-9, 1e3*(med_plan-med_exec));
    printf("Exec only : %.3f ms (%.1f GF/s) - BEST with plan reuse\n", 
           1e3*med_exec, (flops/med_exec)*1e-9);
    printf("Naive     : %.3f ms (%.1f GF/s)\n", 
           1e3*med_naive, (flops/med_naive)*1e-9);
    printf("Speedup   : Auto x%.2f | Planned x%.2f | vs Naive x%.2f\n", 
           med_naive/med_auto, med_naive/med_exec, med_naive/med_exec);

    free(t_auto); free(t_plan); free(t_exec); free(t_naive);

out:
    aligned_free32(A); aligned_free32(B); aligned_free32(C); aligned_free32(Cref);
}

//==============================================================================
// KALMAN FILTER SIMULATION (shows planning benefit)
//==============================================================================

void bench_kalman_operations(void)
{
    printf("\n=== KALMAN FILTER OPERATIONS (1000 iterations) ===\n");
    
    const int n_iter = 1000;
    const int sizes[] = {4, 6, 8, 12};
    const char* names[] = {"4x4 (x,y,vx,vy)", "6x6 (+accel)", "8x8 (3D)", "12x12 (full 3D)"};
    
    for(int s=0; s<4; s++){
        int n = sizes[s];
        
        float *F = (float*)aligned_alloc32(n*n*sizeof(float));
        float *P = (float*)aligned_alloc32(n*n*sizeof(float));
        float *Q = (float*)aligned_alloc32(n*n*sizeof(float));
        float *temp = (float*)aligned_alloc32(n*n*sizeof(float));
        
        // Initialize matrices
        for(int i=0;i<n*n;i++){
            F[i] = (i%n == i/n) ? 1.0f : ((i%n == i/n-1) ? 0.01f : 0.0f); // State transition
            P[i] = (i%n == i/n) ? 1.0f : 0.0f; // Initial covariance
            Q[i] = (i%n == i/n) ? 0.001f : 0.0f; // Process noise
        }
        
        // Without planning - recreate plan each time
        double t0 = now_sec();
        for(int iter=0; iter<n_iter; iter++){
            // P = F*P*F^T + Q (simplified)
            gemm_auto(temp, F, P, n, n, n, 1.0f, 0.0f);  // temp = F*P
            gemm_auto(P, temp, F, n, n, n, 1.0f, 0.0f);  // P = temp*F^T
            for(int i=0;i<n*n;i++) P[i] += Q[i];         // P = P + Q
        }
        double t_auto = now_sec() - t0;
        
        // With planning - reuse plans
        gemm_error_t error;
        gemm_plan_t *plan1 = gemm_plan_create_safe(n,n,n,F,P,temp,1.0f,0.0f,&error);
        gemm_plan_t *plan2 = gemm_plan_create_safe(n,n,n,temp,F,P,1.0f,0.0f,&error);
        
        // Reset P
        for(int i=0;i<n*n;i++) P[i] = (i%n == i/n) ? 1.0f : 0.0f;
        
        t0 = now_sec();
        for(int iter=0; iter<n_iter; iter++){
            gemm_execute_plan(plan1, temp, F, P, 1.0f, 0.0f);
            gemm_execute_plan(plan2, P, temp, F, 1.0f, 0.0f);
            for(int i=0;i<n*n;i++) P[i] += Q[i];
        }
        double t_planned = now_sec() - t0;
        
        printf("  %s: Auto %.3fms | Planned %.3fms | Speedup x%.2f\n",
               names[s], t_auto*1000, t_planned*1000, t_auto/t_planned);
        
        gemm_plan_destroy(plan1);
        gemm_plan_destroy(plan2);
        aligned_free32(F); aligned_free32(P); aligned_free32(Q); aligned_free32(temp);
    }
}

//==============================================================================
// MAIN BENCHMARK SUITE
//==============================================================================

void run_benchmark_suite(void)
{
    const int reps = 9, verify = 1;
    
    printf("GEMM BENCHMARK - Planning Architecture\n");
    printf("=======================================\n");
    
    // Test different size categories
    printf("\n--- TINY MATRICES (Register-only) ---\n");
    bench_case_planned(4, 4, 4, reps, verify, "Tiny");
    bench_case_planned(8, 8, 8, reps, verify, "Tiny");
    bench_case_planned(12, 12, 12, reps, verify, "Tiny");
    
    printf("\n--- SMALL MATRICES (Direct kernels) ---\n");
    bench_case_planned(32, 32, 32, reps, verify, "Small");
    bench_case_planned(64, 64, 64, reps, verify, "Small");
    
    printf("\n--- MEDIUM MATRICES (Single blocking) ---\n");
    bench_case_planned(128, 128, 128, reps, verify, "Medium");
    bench_case_planned(256, 256, 256, reps, verify, "Medium");
    
    printf("\n--- LARGE MATRICES (Full blocking) ---\n");
    bench_case_planned(512, 512, 512, reps, verify, "Large");
    bench_case_planned(1024, 1024, 1024, reps, 0, "Large");  // Skip verify for speed
    
    printf("\n--- IRREGULAR SIZES ---\n");
    bench_case_planned(129, 257, 255, reps, verify, "Irregular");
    bench_case_planned(100, 200, 150, reps, verify, "Irregular");
    
    // Kalman filter simulation
    bench_kalman_operations();
}

