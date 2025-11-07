/**
 * @file test_correctness.c
 * @brief Round-trip A = Q*R verification
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "qr.h"

// Matrix multiply: C = A * B (row-major)
static void matmul(const float *A, const float *B, float *C,
                   uint16_t m, uint16_t k, uint16_t n)
{
    for (uint16_t i = 0; i < m; ++i)
    {
        for (uint16_t j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (uint16_t p = 0; p < k; ++p)
            {
                sum += A[(size_t)i * k + p] * B[(size_t)p * n + j];
            }
            C[(size_t)i * n + j] = sum;
        }
    }
}

// Frobenius norm
static float frobenius_norm(const float *A, uint16_t m, uint16_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < (size_t)m * n; ++i)
    {
        double v = (double)A[i];
        sum += v * v;
    }
    return (float)sqrt(sum);
}

static int test_roundtrip(uint16_t m, uint16_t n, const char *desc)
{
    printf("  [TEST] %s (%ux%u)... ", desc, m, n);

    qr_workspace *ws = qr_workspace_alloc(m, n, 0);
    if (!ws)
    {
        printf("FAIL (workspace alloc)\n");
        return 1;
    }

    float *A = (float *)malloc((size_t)m * n * sizeof(float));
    float *Q = (float *)malloc((size_t)m * m * sizeof(float));
    float *R = (float *)malloc((size_t)m * n * sizeof(float));
    float *QR = (float *)malloc((size_t)m * n * sizeof(float));

    if (!A || !Q || !R || !QR)
    {
        printf("FAIL (memory alloc)\n");
        free(A); free(Q); free(R); free(QR);
        qr_workspace_free(ws);
        return 1;
    }

    // Random test matrix
    for (size_t i = 0; i < (size_t)m * n; ++i)
    {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // Factor
    int ret = qr_ws(ws, A, Q, R, m, n, false);
    if (ret != 0)
    {
        printf("FAIL (qr_ws returned %d)\n", ret);
        free(A); free(Q); free(R); free(QR);
        qr_workspace_free(ws);
        return 1;
    }

    // Reconstruct
    matmul(Q, R, QR, m, m, n);

    // Error
    float normA = frobenius_norm(A, m, n);
    for (size_t i = 0; i < (size_t)m * n; ++i)
    {
        QR[i] = A[i] - QR[i];
    }
    float normErr = frobenius_norm(QR, m, n);
    float relErr = normErr / normA;

    free(A); free(Q); free(R); free(QR);
    qr_workspace_free(ws);

    const float tol = 1e-4f;
    if (relErr > tol)
    {
        printf("FAIL (rel_err=%.2e)\n", relErr);
        return 1;
    }

    printf("PASS (rel_err=%.2e)\n", relErr);
    return 0;
}

int test_correctness(void)
{
    int failures = 0;
    failures += test_roundtrip(64, 64, "Square 64x64");
    failures += test_roundtrip(128, 128, "Square 128x128");
    failures += test_roundtrip(256, 64, "Tall 256x64");
    failures += test_roundtrip(64, 256, "Wide 64x256");
    return failures;
}