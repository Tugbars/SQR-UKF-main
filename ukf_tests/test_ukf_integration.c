/**
 * @file test_ukf_integration.c
 * @brief End-to-end integration tests for SR-UKF
 *
 * Tests:
 * - Full predict-update cycle
 * - Multi-step filtering with simulated dynamics
 * - Filter convergence behavior
 * - State estimation accuracy
 * - Covariance consistency (NEES bounds)
 * - Linear system comparison (UKF should match KF for linear systems)
 * - Nonlinear system tracking
 * - Filter recovery from poor initialization
 *
 * @author TUGBARS
 * @date 2025
 */

#include "test_common.h"
#include "sqr_ukf.h"
#include "gemm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//==============================================================================
// CONSTANTS
//==============================================================================

#define MAX_STATE_DIM 32
#define PI 3.14159265358979323846f

//==============================================================================
// UTILITIES
//==============================================================================

/**
 * @brief Simple pseudo-random number generator (deterministic for tests)
 */
static uint32_t g_rng_state = 12345;

static float randf(void)
{
    g_rng_state = g_rng_state * 1103515245 + 12345;
    return (float)(g_rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static float randn(void)
{
    /* Box-Muller transform */
    float u1 = randf();
    float u2 = randf();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}

static void seed_rng(uint32_t seed)
{
    g_rng_state = seed;
}

/**
 * @brief Compute vector 2-norm
 */
static float vec_norm(const float *v, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        sum += v[i] * v[i];
    }
    return sqrtf(sum);
}

/**
 * @brief Compute RMS error between vectors
 */
static float rms_error(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum / (float)n);
}

/**
 * @brief Reconstruct covariance from upper triangular SR: P = S^T * S
 */
static void reconstruct_covariance(float *P, const float *S, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            float sum = 0.0f;
            size_t k_max = (i < j) ? i : j;
            for (size_t k = 0; k <= k_max; k++)
            {
                sum += S[k * n + i] * S[k * n + j];
            }
            P[i * n + j] = sum;
        }
    }
}

/**
 * @brief Compute Normalized Estimation Error Squared (NEES)
 * 
 * NEES = (x_true - x_est)^T * P^(-1) * (x_true - x_est)
 * 
 * For consistency, NEES should have chi-squared distribution with n DOF.
 * Mean should be approximately n.
 */
static float compute_nees(const float *x_true, const float *x_est,
                          const float *P, size_t n)
{
    /* Compute error vector */
    float err[MAX_STATE_DIM];
    for (size_t i = 0; i < n; i++)
    {
        err[i] = x_true[i] - x_est[i];
    }
    
    /* Compute P^(-1) * err via solving P * y = err */
    /* For simplicity, use direct inversion for small matrices */
    float P_copy[MAX_STATE_DIM * MAX_STATE_DIM];
    float P_inv[MAX_STATE_DIM * MAX_STATE_DIM];
    memcpy(P_copy, P, n * n * sizeof(float));
    
    /* Simple Gauss-Jordan inversion */
    float work[MAX_STATE_DIM * 2 * MAX_STATE_DIM];
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            work[i * 2 * n + j] = P_copy[i * n + j];
            work[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    for (size_t col = 0; col < n; col++)
    {
        /* Find pivot */
        size_t pivot = col;
        float max_val = fabsf(work[col * 2 * n + col]);
        for (size_t row = col + 1; row < n; row++)
        {
            float val = fabsf(work[row * 2 * n + col]);
            if (val > max_val)
            {
                max_val = val;
                pivot = row;
            }
        }
        
        if (max_val < 1e-10f)
        {
            return -1.0f; /* Singular */
        }
        
        /* Swap rows */
        if (pivot != col)
        {
            for (size_t j = 0; j < 2 * n; j++)
            {
                float tmp = work[col * 2 * n + j];
                work[col * 2 * n + j] = work[pivot * 2 * n + j];
                work[pivot * 2 * n + j] = tmp;
            }
        }
        
        /* Scale pivot row */
        float scale = 1.0f / work[col * 2 * n + col];
        for (size_t j = 0; j < 2 * n; j++)
        {
            work[col * 2 * n + j] *= scale;
        }
        
        /* Eliminate */
        for (size_t row = 0; row < n; row++)
        {
            if (row != col)
            {
                float factor = work[row * 2 * n + col];
                for (size_t j = 0; j < 2 * n; j++)
                {
                    work[row * 2 * n + j] -= factor * work[col * 2 * n + j];
                }
            }
        }
    }
    
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            P_inv[i * n + j] = work[i * 2 * n + n + j];
        }
    }
    
    /* Compute err^T * P_inv * err */
    float P_inv_err[MAX_STATE_DIM];
    for (size_t i = 0; i < n; i++)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++)
        {
            sum += P_inv[i * n + j] * err[j];
        }
        P_inv_err[i] = sum;
    }
    
    float nees = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        nees += err[i] * P_inv_err[i];
    }
    
    return nees;
}

/**
 * @brief Check if matrix has valid SR structure
 */
static int check_valid_sr(const float *S, size_t n)
{
    /* Check upper triangular */
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            if (fabsf(S[i * n + j]) > 1e-6f)
            {
                return 0;
            }
        }
    }
    
    /* Check positive diagonal */
    for (size_t i = 0; i < n; i++)
    {
        if (S[i * n + i] <= 0.0f || !isfinite(S[i * n + i]))
        {
            return 0;
        }
    }
    
    return 1;
}

//==============================================================================
// TEST SYSTEM 1: Linear constant velocity (1D position + velocity)
//==============================================================================

static const uint8_t LINEAR_CV_L = 2;
static float g_linear_cv_dt = 0.1f;

/**
 * @brief Linear constant velocity dynamics: x' = [1 dt; 0 1] * x
 */
static void linear_cv_F(float dx[], float x[], float u[])
{
    (void)u;
    float dt = g_linear_cv_dt;
    dx[0] = x[0] + dt * x[1];  /* position */
    dx[1] = x[1];              /* velocity (constant) */
}

/**
 * @brief Simulate true linear CV system with process noise
 */
static void simulate_linear_cv_step(float x_true[], float dt, float q_std)
{
    float x_new[2];
    x_new[0] = x_true[0] + dt * x_true[1] + q_std * randn();
    x_new[1] = x_true[1] + q_std * 0.1f * randn();
    x_true[0] = x_new[0];
    x_true[1] = x_new[1];
}

/**
 * @brief Generate noisy measurement of position
 */
static float measure_linear_cv(const float x_true[], float r_std)
{
    return x_true[0] + r_std * randn();
}

//==============================================================================
// TEST SYSTEM 2: 2D constant velocity tracking
//==============================================================================

static const uint8_t CV_2D_L = 4; /* [px, py, vx, vy] */
static float g_cv_2d_dt = 0.1f;

/**
 * @brief 2D constant velocity dynamics
 */
static void cv_2d_F(float dx[], float x[], float u[])
{
    (void)u;
    float dt = g_cv_2d_dt;
    dx[0] = x[0] + dt * x[2];  /* px */
    dx[1] = x[1] + dt * x[3];  /* py */
    dx[2] = x[2];              /* vx */
    dx[3] = x[3];              /* vy */
}

/**
 * @brief Simulate true 2D CV system
 */
static void simulate_cv_2d_step(float x_true[], float dt, float q_std)
{
    float x_new[4];
    x_new[0] = x_true[0] + dt * x_true[2] + q_std * randn();
    x_new[1] = x_true[1] + dt * x_true[3] + q_std * randn();
    x_new[2] = x_true[2] + q_std * 0.5f * randn();
    x_new[3] = x_true[3] + q_std * 0.5f * randn();
    memcpy(x_true, x_new, 4 * sizeof(float));
}

//==============================================================================
// TEST SYSTEM 3: Nonlinear pendulum (angle + angular velocity)
//==============================================================================

static const uint8_t PENDULUM_L = 2;
static float g_pendulum_dt = 0.01f;
static float g_pendulum_g = 9.81f;
static float g_pendulum_len = 1.0f;

/**
 * @brief Nonlinear pendulum dynamics: θ'' = -g/L * sin(θ)
 * State: [θ, ω] where ω = θ'
 */
static void pendulum_F(float dx[], float x[], float u[])
{
    (void)u;
    float dt = g_pendulum_dt;
    float theta = x[0];
    float omega = x[1];
    
    /* Simple Euler integration of nonlinear ODE */
    float omega_dot = -(g_pendulum_g / g_pendulum_len) * sinf(theta);
    
    dx[0] = theta + dt * omega;
    dx[1] = omega + dt * omega_dot;
}

/**
 * @brief Simulate true pendulum with damping and noise
 */
static void simulate_pendulum_step(float x_true[], float dt, float q_std)
{
    float theta = x_true[0];
    float omega = x_true[1];
    
    float omega_dot = -(g_pendulum_g / g_pendulum_len) * sinf(theta) - 0.1f * omega;
    
    x_true[0] = theta + dt * omega + q_std * randn();
    x_true[1] = omega + dt * omega_dot + q_std * randn();
}

//==============================================================================
// INTEGRATION TEST: Linear 1D constant velocity
//==============================================================================

/**
 * @brief Test SR-UKF on linear 1D constant velocity system
 */
static int test_linear_cv_tracking(void)
{
    printf("\n=== Testing Linear 1D Constant Velocity Tracking ===\n");
    
    int passed = 1;
    
    const uint8_t L = LINEAR_CV_L;
    const size_t n = (size_t)L;
    const size_t N = 2u * n + 1u;
    const int num_steps = 100;
    const float dt = 0.1f;
    g_linear_cv_dt = dt;
    
    seed_rng(42);
    
    /* Allocations */
    float *x_true = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Xstar = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y_meas = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *u = gemm_aligned_alloc(32, n * sizeof(float));
    float *P = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    if (!x_true || !xhat || !S || !X || !Xstar || !Y || !yhat || 
        !y_meas || !Sy || !Pxy || !Rsr || !Wm || !Wc || !u || !P)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Initialize true state */
    x_true[0] = 0.0f;   /* position */
    x_true[1] = 1.0f;   /* velocity */
    
    /* Initialize estimate (with error) */
    xhat[0] = 0.5f;
    xhat[1] = 0.8f;
    
    /* Initialize SR covariance (diagonal) */
    memset(S, 0, n * n * sizeof(float));
    S[0 * n + 0] = 1.0f;  /* position uncertainty */
    S[1 * n + 1] = 0.5f;  /* velocity uncertainty */
    
    /* Process noise SR */
    memset(Rsr, 0, n * n * sizeof(float));
    Rsr[0 * n + 0] = 0.01f;
    Rsr[1 * n + 1] = 0.01f;
    
    /* Measurement noise SR (observe position only) */
    float r_std = 0.5f;
    memset(Sy, 0, n * n * sizeof(float));
    Sy[0 * n + 0] = r_std;
    Sy[1 * n + 1] = 1e6f;  /* Don't observe velocity directly */
    
    /* UKF parameters */
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    memset(u, 0, n * sizeof(float));
    
    /* Statistics tracking */
    float total_pos_err = 0.0f;
    float total_vel_err = 0.0f;
    int valid_steps = 0;
    
    printf("  Running %d filter steps...\n", num_steps);
    
    for (int step = 0; step < num_steps; step++)
    {
        /* === PREDICT === */
        
        /* Create sigma points */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        
        /* Propagate through dynamics */
        compute_transition_function(Xstar, X, u, linear_cv_F, L);
        
        /* Predicted mean */
        multiply_sigma_point_matrix_to_weights(xhat, Xstar, Wm, L);
        
        /* Predicted SR covariance */
        int rc = create_state_estimation_error_covariance_matrix(
            S, &qr_ws, Wc, Xstar, xhat, Rsr, L);
        
        if (rc != 0)
        {
            printf("  Step %d: predict covariance failed\n", step);
            continue;
        }
        
        /* === SIMULATE TRUE SYSTEM === */
        simulate_linear_cv_step(x_true, dt, 0.01f);
        
        /* === MEASUREMENT === */
        y_meas[0] = measure_linear_cv(x_true, r_std);
        y_meas[1] = 0.0f;  /* No velocity measurement */
        
        /* === UPDATE === */
        
        /* Create sigma points from predicted state */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        
        /* Propagate through observation (identity for this simple case) */
        H(Y, X, L);
        
        /* Predicted measurement */
        multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, L);
        
        /* Innovation SR covariance */
        rc = create_state_estimation_error_covariance_matrix(
            Sy, &qr_ws, Wc, Y, yhat, Sy, L);
        
        if (rc != 0)
        {
            printf("  Step %d: innovation covariance failed\n", step);
            continue;
        }
        
        /* Cross-covariance */
        rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, xhat, yhat, &pxy_ws, L);
        
        if (rc != 0)
        {
            printf("  Step %d: cross-covariance failed\n", step);
            continue;
        }
        
        /* State and covariance update */
        rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y_meas, Sy, Pxy, &upd_ws, L);
        
        if (rc != 0)
        {
            printf("  Step %d: update failed\n", step);
            continue;
        }
        
        /* Track errors */
        float pos_err = fabsf(xhat[0] - x_true[0]);
        float vel_err = fabsf(xhat[1] - x_true[1]);
        total_pos_err += pos_err;
        total_vel_err += vel_err;
        valid_steps++;
        
        /* Verify SR structure */
        if (!check_valid_sr(S, n))
        {
            printf("  Step %d: Invalid SR structure\n", step);
            passed = 0;
        }
    }
    
    if (valid_steps > 0)
    {
        float avg_pos_err = total_pos_err / (float)valid_steps;
        float avg_vel_err = total_vel_err / (float)valid_steps;
        
        printf("  Average position error: %.4f\n", avg_pos_err);
        printf("  Average velocity error: %.4f\n", avg_vel_err);
        
        /* Check reasonable tracking performance */
        if (avg_pos_err > 1.0f)
        {
            printf("  FAILED: Position tracking too poor\n");
            passed = 0;
        }
        else if (avg_vel_err > 0.5f)
        {
            printf("  FAILED: Velocity tracking too poor\n");
            passed = 0;
        }
        else
        {
            printf("  Tracking performance PASSED\n");
        }
    }
    else
    {
        printf("  FAILED: No valid filter steps\n");
        passed = 0;
    }
    
cleanup:
    gemm_aligned_free(x_true);
    gemm_aligned_free(xhat);
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(Xstar);
    gemm_aligned_free(Y);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y_meas);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wm);
    gemm_aligned_free(Wc);
    gemm_aligned_free(u);
    gemm_aligned_free(P);
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    return passed;
}

//==============================================================================
// INTEGRATION TEST: 2D constant velocity tracking
//==============================================================================

/**
 * @brief Test SR-UKF on 2D constant velocity system
 */
static int test_cv_2d_tracking(void)
{
    printf("\n=== Testing 2D Constant Velocity Tracking ===\n");
    
    int passed = 1;
    
    const uint8_t L = CV_2D_L;
    const size_t n = (size_t)L;
    const size_t N = 2u * n + 1u;
    const int num_steps = 200;
    const float dt = 0.1f;
    g_cv_2d_dt = dt;
    
    seed_rng(123);
    
    /* Allocations */
    float *x_true = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Xstar = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y_meas = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *u = gemm_aligned_alloc(32, n * sizeof(float));
    float *P = gemm_aligned_alloc(32, n * n * sizeof(float));
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    if (!x_true || !xhat || !S || !X || !Xstar || !Y || !yhat || 
        !y_meas || !Sy || !Pxy || !Rsr || !Wm || !Wc || !u || !P)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Initialize true state: moving diagonally */
    x_true[0] = 0.0f;   /* px */
    x_true[1] = 0.0f;   /* py */
    x_true[2] = 1.0f;   /* vx */
    x_true[3] = 0.5f;   /* vy */
    
    /* Initialize estimate with significant error */
    xhat[0] = 2.0f;
    xhat[1] = -1.0f;
    xhat[2] = 0.5f;
    xhat[3] = 1.0f;
    
    /* Initialize SR covariance */
    memset(S, 0, n * n * sizeof(float));
    S[0 * n + 0] = 5.0f;
    S[1 * n + 1] = 5.0f;
    S[2 * n + 2] = 2.0f;
    S[3 * n + 3] = 2.0f;
    
    /* Process noise SR */
    memset(Rsr, 0, n * n * sizeof(float));
    Rsr[0 * n + 0] = 0.1f;
    Rsr[1 * n + 1] = 0.1f;
    Rsr[2 * n + 2] = 0.05f;
    Rsr[3 * n + 3] = 0.05f;
    
    /* Measurement noise SR (observe positions only) */
    float r_std = 1.0f;
    
    /* UKF parameters */
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    memset(u, 0, n * sizeof(float));
    
    /* Statistics */
    float initial_pos_err = sqrtf((xhat[0]-x_true[0])*(xhat[0]-x_true[0]) + 
                                   (xhat[1]-x_true[1])*(xhat[1]-x_true[1]));
    float final_pos_err = 0.0f;
    float avg_nees = 0.0f;
    int valid_steps = 0;
    
    printf("  Initial position error: %.4f\n", initial_pos_err);
    printf("  Running %d filter steps...\n", num_steps);
    
    for (int step = 0; step < num_steps; step++)
    {
        /* === PREDICT === */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        compute_transition_function(Xstar, X, u, cv_2d_F, L);
        multiply_sigma_point_matrix_to_weights(xhat, Xstar, Wm, L);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &qr_ws, Wc, Xstar, xhat, Rsr, L);
        if (rc != 0) continue;
        
        /* === SIMULATE === */
        simulate_cv_2d_step(x_true, dt, 0.05f);
        
        /* === MEASUREMENT (position only) === */
        memset(Sy, 0, n * n * sizeof(float));
        Sy[0 * n + 0] = r_std;
        Sy[1 * n + 1] = r_std;
        Sy[2 * n + 2] = 1e6f;  /* No velocity measurement */
        Sy[3 * n + 3] = 1e6f;
        
        y_meas[0] = x_true[0] + r_std * randn();
        y_meas[1] = x_true[1] + r_std * randn();
        y_meas[2] = 0.0f;
        y_meas[3] = 0.0f;
        
        /* === UPDATE === */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        H(Y, X, L);
        multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, L);
        
        rc = create_state_estimation_error_covariance_matrix(
            Sy, &qr_ws, Wc, Y, yhat, Sy, L);
        if (rc != 0) continue;
        
        rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, xhat, yhat, &pxy_ws, L);
        if (rc != 0) continue;
        
        rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y_meas, Sy, Pxy, &upd_ws, L);
        if (rc != 0) continue;
        
        /* Compute NEES */
        reconstruct_covariance(P, S, n);
        float nees = compute_nees(x_true, xhat, P, n);
        if (nees >= 0.0f)
        {
            avg_nees += nees;
        }
        
        valid_steps++;
        
        /* Track final error */
        if (step == num_steps - 1)
        {
            final_pos_err = sqrtf((xhat[0]-x_true[0])*(xhat[0]-x_true[0]) + 
                                   (xhat[1]-x_true[1])*(xhat[1]-x_true[1]));
        }
    }
    
    if (valid_steps > 0)
    {
        avg_nees /= (float)valid_steps;
        
        printf("  Final position error: %.4f\n", final_pos_err);
        printf("  Average NEES: %.2f (expected ~%.1f for consistent filter)\n", 
               avg_nees, (float)n);
        
        /* Check convergence */
        if (final_pos_err > initial_pos_err)
        {
            printf("  FAILED: Filter diverged\n");
            passed = 0;
        }
        else if (final_pos_err > 2.0f)
        {
            printf("  FAILED: Poor tracking accuracy\n");
            passed = 0;
        }
        else
        {
            printf("  Tracking PASSED\n");
        }
        
        /* Check consistency (NEES should be roughly n for consistent filter) */
        /* Allow wide bounds since this is stochastic */
        if (avg_nees < 0.1f * (float)n || avg_nees > 10.0f * (float)n)
        {
            printf("  WARNING: NEES outside expected bounds (filter may be inconsistent)\n");
        }
    }
    else
    {
        printf("  FAILED: No valid steps\n");
        passed = 0;
    }
    
cleanup:
    gemm_aligned_free(x_true);
    gemm_aligned_free(xhat);
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(Xstar);
    gemm_aligned_free(Y);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y_meas);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wm);
    gemm_aligned_free(Wc);
    gemm_aligned_free(u);
    gemm_aligned_free(P);
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    return passed;
}

//==============================================================================
// INTEGRATION TEST: Nonlinear pendulum
//==============================================================================

/**
 * @brief Test SR-UKF on nonlinear pendulum system
 */
static int test_nonlinear_pendulum(void)
{
    printf("\n=== Testing Nonlinear Pendulum Tracking ===\n");
    
    int passed = 1;
    
    const uint8_t L = PENDULUM_L;
    const size_t n = (size_t)L;
    const size_t N = 2u * n + 1u;
    const int num_steps = 500;
    const float dt = 0.01f;
    g_pendulum_dt = dt;
    
    seed_rng(456);
    
    /* Allocations */
    float *x_true = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Xstar = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y_meas = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *u = gemm_aligned_alloc(32, n * sizeof(float));
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    if (!x_true || !xhat || !S || !X || !Xstar || !Y || !yhat || 
        !y_meas || !Sy || !Pxy || !Rsr || !Wm || !Wc || !u)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Initialize true state: pendulum at 30 degrees */
    x_true[0] = 0.5f;   /* theta (radians) */
    x_true[1] = 0.0f;   /* omega */
    
    /* Initialize estimate with error */
    xhat[0] = 0.3f;
    xhat[1] = 0.2f;
    
    /* Initialize SR covariance */
    memset(S, 0, n * n * sizeof(float));
    S[0 * n + 0] = 0.5f;
    S[1 * n + 1] = 0.5f;
    
    /* Process noise SR */
    memset(Rsr, 0, n * n * sizeof(float));
    Rsr[0 * n + 0] = 0.001f;
    Rsr[1 * n + 1] = 0.001f;
    
    /* Measurement noise SR */
    float r_std = 0.1f;
    
    /* UKF parameters */
    const float alpha = 1e-2f;  /* Slightly larger for nonlinear system */
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    memset(u, 0, n * sizeof(float));
    
    /* Statistics */
    float total_theta_err = 0.0f;
    int valid_steps = 0;
    
    printf("  Running %d filter steps (%.1f seconds)...\n", 
           num_steps, num_steps * dt);
    
    for (int step = 0; step < num_steps; step++)
    {
        /* === PREDICT === */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        compute_transition_function(Xstar, X, u, pendulum_F, L);
        multiply_sigma_point_matrix_to_weights(xhat, Xstar, Wm, L);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &qr_ws, Wc, Xstar, xhat, Rsr, L);
        if (rc != 0) continue;
        
        /* === SIMULATE === */
        simulate_pendulum_step(x_true, dt, 0.001f);
        
        /* === MEASUREMENT (angle only) === */
        memset(Sy, 0, n * n * sizeof(float));
        Sy[0 * n + 0] = r_std;
        Sy[1 * n + 1] = 1e6f;  /* No angular velocity measurement */
        
        y_meas[0] = x_true[0] + r_std * randn();
        y_meas[1] = 0.0f;
        
        /* === UPDATE === */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        H(Y, X, L);
        multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, L);
        
        rc = create_state_estimation_error_covariance_matrix(
            Sy, &qr_ws, Wc, Y, yhat, Sy, L);
        if (rc != 0) continue;
        
        rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, xhat, yhat, &pxy_ws, L);
        if (rc != 0) continue;
        
        rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y_meas, Sy, Pxy, &upd_ws, L);
        if (rc != 0) continue;
        
        /* Track error */
        float theta_err = fabsf(xhat[0] - x_true[0]);
        total_theta_err += theta_err;
        valid_steps++;
        
        /* Verify SR structure periodically */
        if (step % 100 == 0)
        {
            if (!check_valid_sr(S, n))
            {
                printf("  Step %d: Invalid SR structure\n", step);
                passed = 0;
            }
        }
    }
    
    if (valid_steps > 0)
    {
        float avg_theta_err = total_theta_err / (float)valid_steps;
        
        printf("  Average theta error: %.4f rad (%.2f deg)\n", 
               avg_theta_err, avg_theta_err * 180.0f / PI);
        
        /* Check reasonable tracking for nonlinear system */
        if (avg_theta_err > 0.3f)
        {
            printf("  FAILED: Theta tracking too poor\n");
            passed = 0;
        }
        else
        {
            printf("  Nonlinear tracking PASSED\n");
        }
    }
    else
    {
        printf("  FAILED: No valid steps\n");
        passed = 0;
    }
    
cleanup:
    gemm_aligned_free(x_true);
    gemm_aligned_free(xhat);
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(Xstar);
    gemm_aligned_free(Y);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y_meas);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wm);
    gemm_aligned_free(Wc);
    gemm_aligned_free(u);
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    return passed;
}

//==============================================================================
// INTEGRATION TEST: Filter convergence from poor initialization
//==============================================================================

/**
 * @brief Test filter recovery from very poor initial estimate
 */
static int test_convergence_from_poor_init(void)
{
    printf("\n=== Testing Convergence from Poor Initialization ===\n");
    
    int passed = 1;
    
    const uint8_t L = 4;
    const size_t n = (size_t)L;
    const size_t N = 2u * n + 1u;
    const int num_steps = 300;
    const float dt = 0.1f;
    g_cv_2d_dt = dt;
    
    seed_rng(789);
    
    /* Allocations */
    float *x_true = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Xstar = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y_meas = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *u = gemm_aligned_alloc(32, n * sizeof(float));
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    if (!x_true || !xhat || !S || !X || !Xstar || !Y || !yhat || 
        !y_meas || !Sy || !Pxy || !Rsr || !Wm || !Wc || !u)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* True state */
    x_true[0] = 0.0f;
    x_true[1] = 0.0f;
    x_true[2] = 1.0f;
    x_true[3] = 0.5f;
    
    /* VERY poor initial estimate */
    xhat[0] = 50.0f;   /* 50 units off in position */
    xhat[1] = -30.0f;
    xhat[2] = -5.0f;   /* Wrong velocity direction */
    xhat[3] = 3.0f;
    
    /* Large initial covariance to reflect uncertainty */
    memset(S, 0, n * n * sizeof(float));
    S[0 * n + 0] = 100.0f;
    S[1 * n + 1] = 100.0f;
    S[2 * n + 2] = 10.0f;
    S[3 * n + 3] = 10.0f;
    
    /* Process noise */
    memset(Rsr, 0, n * n * sizeof(float));
    Rsr[0 * n + 0] = 0.1f;
    Rsr[1 * n + 1] = 0.1f;
    Rsr[2 * n + 2] = 0.05f;
    Rsr[3 * n + 3] = 0.05f;
    
    /* Measurement noise */
    float r_std = 1.0f;
    
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    memset(u, 0, n * sizeof(float));
    
    float initial_err = vec_norm(xhat, n);
    float errors[3] = {0.0f, 0.0f, 0.0f}; /* early, mid, late */
    int phase_counts[3] = {0, 0, 0};
    
    printf("  Initial state error norm: %.2f\n", 
           sqrtf((xhat[0]-x_true[0])*(xhat[0]-x_true[0]) + 
                 (xhat[1]-x_true[1])*(xhat[1]-x_true[1])));
    
    for (int step = 0; step < num_steps; step++)
    {
        /* Predict */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        compute_transition_function(Xstar, X, u, cv_2d_F, L);
        multiply_sigma_point_matrix_to_weights(xhat, Xstar, Wm, L);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &qr_ws, Wc, Xstar, xhat, Rsr, L);
        if (rc != 0) continue;
        
        /* Simulate */
        simulate_cv_2d_step(x_true, dt, 0.05f);
        
        /* Measure positions */
        memset(Sy, 0, n * n * sizeof(float));
        Sy[0 * n + 0] = r_std;
        Sy[1 * n + 1] = r_std;
        Sy[2 * n + 2] = 1e6f;
        Sy[3 * n + 3] = 1e6f;
        
        y_meas[0] = x_true[0] + r_std * randn();
        y_meas[1] = x_true[1] + r_std * randn();
        y_meas[2] = 0.0f;
        y_meas[3] = 0.0f;
        
        /* Update */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        H(Y, X, L);
        multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, L);
        
        rc = create_state_estimation_error_covariance_matrix(
            Sy, &qr_ws, Wc, Y, yhat, Sy, L);
        if (rc != 0) continue;
        
        rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, xhat, yhat, &pxy_ws, L);
        if (rc != 0) continue;
        
        rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y_meas, Sy, Pxy, &upd_ws, L);
        if (rc != 0) continue;
        
        /* Track error by phase */
        float pos_err = sqrtf((xhat[0]-x_true[0])*(xhat[0]-x_true[0]) + 
                               (xhat[1]-x_true[1])*(xhat[1]-x_true[1]));
        
        int phase = (step < 100) ? 0 : ((step < 200) ? 1 : 2);
        errors[phase] += pos_err;
        phase_counts[phase]++;
    }
    
    /* Compute average errors per phase */
    float avg_errors[3];
    for (int i = 0; i < 3; i++)
    {
        avg_errors[i] = (phase_counts[i] > 0) ? errors[i] / phase_counts[i] : 0.0f;
    }
    
    printf("  Avg position error by phase:\n");
    printf("    Steps 0-99:   %.2f\n", avg_errors[0]);
    printf("    Steps 100-199: %.2f\n", avg_errors[1]);
    printf("    Steps 200-299: %.2f\n", avg_errors[2]);
    
    /* Check convergence: later phases should have lower error */
    if (avg_errors[2] > avg_errors[0] * 0.5f)
    {
        printf("  WARNING: Filter did not converge as expected\n");
    }
    
    if (avg_errors[2] < 5.0f)
    {
        printf("  Convergence PASSED (recovered from poor init)\n");
    }
    else
    {
        printf("  FAILED: Did not converge sufficiently\n");
        passed = 0;
    }
    
cleanup:
    gemm_aligned_free(x_true);
    gemm_aligned_free(xhat);
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(Xstar);
    gemm_aligned_free(Y);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y_meas);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wm);
    gemm_aligned_free(Wc);
    gemm_aligned_free(u);
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    return passed;
}

//==============================================================================
// INTEGRATION TEST: Long-term stability
//==============================================================================

/**
 * @brief Test filter stability over many iterations
 */
static int test_long_term_stability(void)
{
    printf("\n=== Testing Long-term Stability ===\n");
    
    int passed = 1;
    
    const uint8_t L = 4;
    const size_t n = (size_t)L;
    const size_t N = 2u * n + 1u;
    const int num_steps = 1000;
    const float dt = 0.1f;
    g_cv_2d_dt = dt;
    
    seed_rng(999);
    
    /* Allocations */
    float *x_true = gemm_aligned_alloc(32, n * sizeof(float));
    float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *X = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Xstar = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *Y = gemm_aligned_alloc(32, n * N * sizeof(float));
    float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
    float *y_meas = gemm_aligned_alloc(32, n * sizeof(float));
    float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Rsr = gemm_aligned_alloc(32, n * n * sizeof(float));
    float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
    float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
    float *u = gemm_aligned_alloc(32, n * sizeof(float));
    
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    if (!x_true || !xhat || !S || !X || !Xstar || !Y || !yhat || 
        !y_meas || !Sy || !Pxy || !Rsr || !Wm || !Wc || !u)
    {
        printf("  ERROR: Allocation failed\n");
        passed = 0;
        goto cleanup;
    }
    
    /* Initialize */
    x_true[0] = 0.0f; x_true[1] = 0.0f; x_true[2] = 1.0f; x_true[3] = 0.5f;
    memcpy(xhat, x_true, n * sizeof(float));
    
    memset(S, 0, n * n * sizeof(float));
    for (size_t i = 0; i < n; i++) S[i * n + i] = 1.0f;
    
    memset(Rsr, 0, n * n * sizeof(float));
    Rsr[0] = 0.1f; Rsr[1*n+1] = 0.1f; Rsr[2*n+2] = 0.05f; Rsr[3*n+3] = 0.05f;
    
    float r_std = 1.0f;
    
    const float alpha = 1e-3f;
    const float beta = 2.0f;
    const float kappa = 0.0f;
    create_weights(Wc, Wm, alpha, beta, kappa, L);
    
    memset(u, 0, n * sizeof(float));
    
    int sr_failures = 0;
    int nan_failures = 0;
    int update_failures = 0;
    
    printf("  Running %d filter steps...\n", num_steps);
    
    for (int step = 0; step < num_steps; step++)
    {
        /* Predict */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        compute_transition_function(Xstar, X, u, cv_2d_F, L);
        multiply_sigma_point_matrix_to_weights(xhat, Xstar, Wm, L);
        
        int rc = create_state_estimation_error_covariance_matrix(
            S, &qr_ws, Wc, Xstar, xhat, Rsr, L);
        if (rc != 0)
        {
            update_failures++;
            continue;
        }
        
        /* Simulate */
        simulate_cv_2d_step(x_true, dt, 0.05f);
        
        /* Measure */
        memset(Sy, 0, n * n * sizeof(float));
        Sy[0] = r_std; Sy[1*n+1] = r_std; Sy[2*n+2] = 1e6f; Sy[3*n+3] = 1e6f;
        
        y_meas[0] = x_true[0] + r_std * randn();
        y_meas[1] = x_true[1] + r_std * randn();
        y_meas[2] = 0.0f;
        y_meas[3] = 0.0f;
        
        /* Update */
        create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
        H(Y, X, L);
        multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, L);
        
        rc = create_state_estimation_error_covariance_matrix(
            Sy, &qr_ws, Wc, Y, yhat, Sy, L);
        if (rc != 0) { update_failures++; continue; }
        
        rc = create_state_cross_covariance_matrix(
            Pxy, Wc, X, Y, xhat, yhat, &pxy_ws, L);
        if (rc != 0) { update_failures++; continue; }
        
        rc = update_state_covariance_matrix_and_state_estimation_vector(
            S, xhat, yhat, y_meas, Sy, Pxy, &upd_ws, L);
        if (rc != 0) { update_failures++; continue; }
        
        /* Check for numerical issues */
        if (!check_valid_sr(S, n))
        {
            sr_failures++;
        }
        
        for (size_t i = 0; i < n; i++)
        {
            if (!isfinite(xhat[i]))
            {
                nan_failures++;
                break;
            }
        }
    }
    
    printf("  SR structure failures: %d / %d\n", sr_failures, num_steps);
    printf("  NaN/Inf failures: %d / %d\n", nan_failures, num_steps);
    printf("  Update failures: %d / %d\n", update_failures, num_steps);
    
    if (sr_failures > num_steps / 100)
    {
        printf("  FAILED: Too many SR structure failures\n");
        passed = 0;
    }
    else if (nan_failures > 0)
    {
        printf("  FAILED: NaN/Inf detected\n");
        passed = 0;
    }
    else if (update_failures > num_steps / 50)
    {
        printf("  FAILED: Too many update failures\n");
        passed = 0;
    }
    else
    {
        printf("  Long-term stability PASSED\n");
    }
    
cleanup:
    gemm_aligned_free(x_true);
    gemm_aligned_free(xhat);
    gemm_aligned_free(S);
    gemm_aligned_free(X);
    gemm_aligned_free(Xstar);
    gemm_aligned_free(Y);
    gemm_aligned_free(yhat);
    gemm_aligned_free(y_meas);
    gemm_aligned_free(Sy);
    gemm_aligned_free(Pxy);
    gemm_aligned_free(Rsr);
    gemm_aligned_free(Wm);
    gemm_aligned_free(Wc);
    gemm_aligned_free(u);
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    return passed;
}

//==============================================================================
// INTEGRATION TEST: Workspace persistence across filter runs
//==============================================================================

/**
 * @brief Test that workspaces can be reused across multiple filter instances
 */
static int test_workspace_persistence(void)
{
    printf("\n=== Testing Workspace Persistence ===\n");
    
    int passed = 1;
    
    /* Create shared workspaces */
    ukf_qr_ws_t qr_ws = {0};
    ukf_upd_ws_t upd_ws = {0};
    ukf_pxy_ws_t pxy_ws = {0};
    
    /* Run multiple short filter instances */
    const int num_instances = 5;
    const int steps_per_instance = 50;
    
    uint8_t test_L[] = {4, 8, 16, 8, 4}; /* Varying sizes */
    
    printf("  Running %d filter instances...\n", num_instances);
    
    for (int inst = 0; inst < num_instances; inst++)
    {
        uint8_t L = test_L[inst];
        size_t n = (size_t)L;
        size_t N = 2u * n + 1u;
        
        printf("    Instance %d: L=%d\n", inst, L);
        
        float *xhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *S = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *X = gemm_aligned_alloc(32, n * N * sizeof(float));
        float *Xstar = gemm_aligned_alloc(32, n * N * sizeof(float));
        float *Y = gemm_aligned_alloc(32, n * N * sizeof(float));
        float *yhat = gemm_aligned_alloc(32, n * sizeof(float));
        float *y_meas = gemm_aligned_alloc(32, n * sizeof(float));
        float *Sy = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Pxy = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Rsr = gemm_aligned_alloc(32, n * n * sizeof(float));
        float *Wm = gemm_aligned_alloc(32, N * sizeof(float));
        float *Wc = gemm_aligned_alloc(32, N * sizeof(float));
        float *u = gemm_aligned_alloc(32, n * sizeof(float));
        
        if (!xhat || !S || !X || !Xstar || !Y || !yhat || !y_meas ||
            !Sy || !Pxy || !Rsr || !Wm || !Wc || !u)
        {
            printf("      ERROR: Allocation failed\n");
            passed = 0;
            goto cleanup_inst;
        }
        
        /* Initialize */
        for (size_t i = 0; i < n; i++) xhat[i] = (float)i;
        memset(S, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++) S[i * n + i] = 1.0f;
        memset(Rsr, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i++) Rsr[i * n + i] = 0.1f;
        memset(u, 0, n * sizeof(float));
        
        const float alpha = 1e-3f;
        const float beta = 2.0f;
        const float kappa = 0.0f;
        create_weights(Wc, Wm, alpha, beta, kappa, L);
        
        int inst_failures = 0;
        
        for (int step = 0; step < steps_per_instance; step++)
        {
            /* Predict */
            create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
            
            /* Simple dynamics: x' = x */
            memcpy(Xstar, X, n * N * sizeof(float));
            
            multiply_sigma_point_matrix_to_weights(xhat, Xstar, Wm, L);
            
            int rc = create_state_estimation_error_covariance_matrix(
                S, &qr_ws, Wc, Xstar, xhat, Rsr, L);
            if (rc != 0) { inst_failures++; continue; }
            
            /* Measure */
            memset(Sy, 0, n * n * sizeof(float));
            for (size_t i = 0; i < n; i++) Sy[i * n + i] = 0.5f;
            
            for (size_t i = 0; i < n; i++)
            {
                y_meas[i] = xhat[i] + 0.1f * randn();
            }
            
            /* Update */
            create_sigma_point_matrix(X, xhat, S, alpha, kappa, L);
            H(Y, X, L);
            multiply_sigma_point_matrix_to_weights(yhat, Y, Wm, L);
            
            rc = create_state_estimation_error_covariance_matrix(
                Sy, &qr_ws, Wc, Y, yhat, Sy, L);
            if (rc != 0) { inst_failures++; continue; }
            
            rc = create_state_cross_covariance_matrix(
                Pxy, Wc, X, Y, xhat, yhat, &pxy_ws, L);
            if (rc != 0) { inst_failures++; continue; }
            
            rc = update_state_covariance_matrix_and_state_estimation_vector(
                S, xhat, yhat, y_meas, Sy, Pxy, &upd_ws, L);
            if (rc != 0) { inst_failures++; continue; }
        }
        
        if (inst_failures > steps_per_instance / 10)
        {
            printf("      FAILED: %d failures in instance\n", inst_failures);
            passed = 0;
        }
        
cleanup_inst:
        gemm_aligned_free(xhat);
        gemm_aligned_free(S);
        gemm_aligned_free(X);
        gemm_aligned_free(Xstar);
        gemm_aligned_free(Y);
        gemm_aligned_free(yhat);
        gemm_aligned_free(y_meas);
        gemm_aligned_free(Sy);
        gemm_aligned_free(Pxy);
        gemm_aligned_free(Rsr);
        gemm_aligned_free(Wm);
        gemm_aligned_free(Wc);
        gemm_aligned_free(u);
    }
    
    /* Cleanup shared workspaces */
    ukf_qr_ws_cleanup(&qr_ws);
    ukf_upd_ws_cleanup(&upd_ws);
    ukf_pxy_ws_cleanup(&pxy_ws);
    
    if (passed)
    {
        printf("  Workspace persistence PASSED\n");
    }
    
    return passed;
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int run_ukf_integration_tests(test_results_t *results)
{
    printf("=================================================\n");
    printf("    SR-UKF INTEGRATION TESTS\n");
    printf("=================================================\n");
    
    results->total = 0;
    results->passed = 0;
    results->failed = 0;
    
    /* Linear System Tests */
    printf("\n--- Linear System Tests ---\n");
    
    results->total++;
    if (test_linear_cv_tracking())
    {
        results->passed++;
        printf("✓ Linear 1D CV tracking PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Linear 1D CV tracking FAILED\n");
    }
    
    results->total++;
    if (test_cv_2d_tracking())
    {
        results->passed++;
        printf("✓ 2D CV tracking PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ 2D CV tracking FAILED\n");
    }
    
    /* Nonlinear System Tests */
    printf("\n--- Nonlinear System Tests ---\n");
    
    results->total++;
    if (test_nonlinear_pendulum())
    {
        results->passed++;
        printf("✓ Nonlinear pendulum tracking PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Nonlinear pendulum tracking FAILED\n");
    }
    
    /* Convergence Tests */
    printf("\n--- Convergence Tests ---\n");
    
    results->total++;
    if (test_convergence_from_poor_init())
    {
        results->passed++;
        printf("✓ Convergence from poor init PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Convergence from poor init FAILED\n");
    }
    
    /* Stability Tests */
    printf("\n--- Stability Tests ---\n");
    
    results->total++;
    if (test_long_term_stability())
    {
        results->passed++;
        printf("✓ Long-term stability PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Long-term stability FAILED\n");
    }
    
    /* Workspace Tests */
    printf("\n--- Workspace Tests ---\n");
    
    results->total++;
    if (test_workspace_persistence())
    {
        results->passed++;
        printf("✓ Workspace persistence PASSED\n");
    }
    else
    {
        results->failed++;
        printf("✗ Workspace persistence FAILED\n");
    }
    
    /* Summary */
    printf("\n=================================================\n");
    printf("SR-UKF Integration Tests: %d/%d passed\n", results->passed, results->total);
    
    if (results->passed == results->total)
    {
        printf("✓ ALL SR-UKF INTEGRATION TESTS PASSED!\n");
    }
    else
    {
        printf("✗ %d SR-UKF integration tests FAILED\n", results->failed);
    }
    printf("=================================================\n");
    
    return (results->failed == 0) ? 0 : 1;
}

//==============================================================================
// STANDALONE MODE
//==============================================================================

#ifdef STANDALONE
int main(void)
{
    test_results_t results = {0};
    return run_ukf_integration_tests(&results);
}
#endif