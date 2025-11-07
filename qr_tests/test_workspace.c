/**
 * @file test_workspace.c
 * @brief Workspace allocation tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "qr.h"

static int test_alloc_free(void)
{
    printf("  [TEST] Workspace alloc/free... ");

    qr_workspace *ws = qr_workspace_alloc(512, 512, 64);
    if (!ws)
    {
        printf("FAIL (allocation failed)\n");
        return 1;
    }

    size_t bytes = qr_workspace_bytes(ws);
    if (bytes == 0)
    {
        printf("FAIL (zero bytes reported)\n");
        qr_workspace_free(ws);
        return 1;
    }

    qr_workspace_free(ws);
    printf("PASS (%zu bytes)\n", bytes);
    return 0;
}

static int test_null_handling(void)
{
    printf("  [TEST] NULL workspace handling... ");
    
    qr_workspace_free(NULL);  // Should not crash
    
    size_t bytes = qr_workspace_bytes(NULL);
    if (bytes != 0)
    {
        printf("FAIL (expected 0 bytes for NULL)\n");
        return 1;
    }
    
    printf("PASS\n");
    return 0;
}

int test_workspace(void)
{
    int failures = 0;
    failures += test_alloc_free();
    failures += test_null_handling();
    return failures;
}