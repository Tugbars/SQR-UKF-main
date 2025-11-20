/**
 * @brief Fast path for small square matrices (≤ 128×128)
 * 
 * Optimizations:
 * - Single block (ib = n, no blocking overhead)
 * - Cache-resident (entire matrix in L2)
 * - Simplified logic (no trailing updates)
 * - Leverages GEMM small-matrix kernels
 * 
 * @return 0 on success
 */
static int qr_small_square(const float *A, float *Q, float *R,
                           uint16_t n, bool only_R)
{
    // Allocate workspace (stack for small n)
    float *A_copy = (float *)alloca(n * n * sizeof(float));
    float *Y = (float *)alloca(n * n * sizeof(float));
    float *tau = (float *)alloca(n * sizeof(float));
    float *work = (float *)alloca(n * sizeof(float));
    
    // Copy input
    memcpy(A_copy, A, n * n * sizeof(float));
    memset(Y, 0, n * n * sizeof(float));
    
    //==========================================================================
    // Single-panel factorization (no blocking)
    //==========================================================================
    
    for (uint16_t j = 0; j < n; ++j) {
        uint16_t col_len = n - j;
        
        // Extract column j
        for (uint16_t i = 0; i < col_len; ++i)
            work[i] = A_copy[(j + i) * n + j];
        
        // Compute Householder
        float beta;
        compute_householder_robust(work, col_len, &tau[j], &beta);
        
        // Store in A_copy (R) and Y
        A_copy[j * n + j] = beta;
        for (uint16_t i = 1; i < col_len; ++i)
            A_copy[(j + i) * n + j] = 0.0f;
        
        for (uint16_t i = 0; i < col_len; ++i)
            Y[(j + i) * n + j] = work[i];
        
        // Apply to trailing columns (if any)
        if (j + 1 < n) {
            float *trailing = &A_copy[j * n + (j + 1)];
            apply_householder_clean(trailing, col_len, n - j - 1,
                                    n, work, tau[j]);
        }
    }
    
    //==========================================================================
    // Extract R
    //==========================================================================
    
    for (uint16_t i = 0; i < n; ++i)
        for (uint16_t j = 0; j < n; ++j)
            R[i * n + j] = (i <= j) ? A_copy[i * n + j] : 0.0f;
    
    if (only_R)
        return 0;
    
    //==========================================================================
    // Form Q (simplified for small matrices)
    //==========================================================================
    
    // Initialize Q = I
    memset(Q, 0, n * n * sizeof(float));
    for (uint16_t i = 0; i < n; ++i)
        Q[i * n + i] = 1.0f;
    
    // Apply reflectors in reverse (Q = H_0 · H_1 · ... · H_{n-1})
    for (int j = n - 1; j >= 0; --j) {
        if (tau[j] == 0.0f)
            continue;
        
        uint16_t col_len = n - j;
        
        // Extract reflector j from Y
        for (uint16_t i = 0; i < col_len; ++i)
            work[i] = Y[(j + i) * n + j];
        
        // Apply to all columns of Q
        float *q_start = &Q[j * n];
        apply_householder_clean(q_start, col_len, n, n, work, tau[j]);
    }
    
    return 0;
}

/**
 * @brief QR with automatic fast-path selection
 */
int qr_auto(const float *A, float *Q, float *R, 
            uint16_t m, uint16_t n, bool only_R)
{
    //==========================================================================
    // FAST PATH 1: Small square matrices (≤ 128×128)
    //==========================================================================
    
    if (m == n && m <= 128) {
        return qr_small_square(A, Q, R, m, only_R);
    }
    
    //==========================================================================
    // FAST PATH 2: Tiny matrices (≤ 32×32)
    //==========================================================================
    
    if (m <= 32 && n <= 32) {
        // Ultra-simple: no blocking at all
        qr_workspace *ws = qr_workspace_alloc_ex(m, n, n, !only_R);
        if (!ws) return -ENOMEM;
        int ret = qr_ws_blocked(ws, A, Q, R, m, n, only_R);
        qr_workspace_free(ws);
        return ret;
    }
    
    //==========================================================================
    // FAST PATH 3: Tall-skinny (m > 4n)
    //==========================================================================
    
    if (m > 4 * n && n <= 256) {
        return qr_tall_skinny(A, Q, R, m, n, only_R);
    }
    
    //==========================================================================
    // GENERAL PATH: Standard blocked QR
    //==========================================================================
    
    return qr_blocked(A, Q, R, m, n, only_R);
}

/**
 * @brief Optimized QR for tall-skinny matrices (m >> n)
 * 
 * Target: Least squares problems (overdetermined systems)
 * Example: 10000×50 matrix (200:1 aspect ratio)
 * 
 * Optimizations:
 * - Smaller block size (ib = 32 max)
 * - Column-major streaming
 * - Skip Q if only R needed
 * - Cache blocking optimized for tall matrices
 * 
 * @param m Number of rows (large)
 * @param n Number of columns (small, n << m)
 */
static int qr_tall_skinny(const float *A, float *Q, float *R,
                          uint16_t m, uint16_t n, bool only_R)
{
    // For tall-skinny, use smaller block size
    uint16_t ib = MIN(32, n);  // Smaller blocks for tall matrices
    
    qr_workspace *ws = qr_workspace_alloc_ex(m, n, ib, !only_R);
    if (!ws) return -ENOMEM;
    
    // Copy A to workspace
    memcpy(ws->Cpack, A, (size_t)m * n * sizeof(float));
    
    //==========================================================================
    // Factorization (optimized for tall-skinny)
    //==========================================================================
    
    uint16_t block_count = 0;
    
    for (uint16_t k = 0; k < n; k += ib) {
        uint16_t block_size = MIN(ib, n - k);
        uint16_t rows_below = m - k;
        uint16_t cols_right = (n > k + block_size) ? (n - k - block_size) : 0;
        
        //======================================================================
        // OPTIMIZATION 1: Use column-wise factorization for tall panels
        //======================================================================
        // For tall-skinny, column-wise access has better cache behavior
        // than row-wise (fewer cache lines touched)
        
        float *panel = &ws->Cpack[k * n + k];
        
        // Factor panel column-by-column (cache-friendly for tall)
        for (uint16_t j = 0; j < block_size && (k + j) < n; ++j) {
            uint16_t col_len = m - (k + j);
            float *col = &panel[j];
            
            // Extract column (strided, but streaming is OK for tall)
            for (uint16_t i = 0; i < col_len; ++i)
                ws->tmp[i] = col[i * n];
            
            // Compute Householder
            float beta;
            compute_householder_robust(ws->tmp, col_len, &ws->tau[k + j], &beta);
            
            // Write back
            col[0] = beta;
            for (uint16_t i = 1; i < col_len; ++i)
                col[i * n] = 0.0f;
            
            // Store in Y
            for (uint16_t i = 0; i < col_len; ++i)
                ws->Y[i * ws->ib + j] = ws->tmp[i];
            
            // Apply to remaining columns in panel
            if (j + 1 < block_size) {
                float *trailing = &panel[j + 1];
                apply_householder_clean(trailing, col_len, block_size - j - 1,
                                        n, ws->tmp, ws->tau[k + j]);
            }
        }
        
        //======================================================================
        // OPTIMIZATION 2: Skip trailing update if only R needed and last block
        //======================================================================
        
        if (only_R && k + block_size >= n) {
            // Last block and only need R → skip expensive trailing update
            break;
        }
        
        // Build T and apply to trailing (if needed)
        if (cols_right > 0) {
            build_T_matrix(ws->Y, &ws->tau[k], ws->T, rows_below, block_size, ws->ib);
            
            // Store for Q formation
            if (!only_R && ws->Y_stored && ws->T_stored) {
                size_t y_offset = block_count * ws->Y_block_stride;
                size_t t_offset = block_count * ws->T_block_stride;
                
                for (uint16_t i = 0; i < rows_below; ++i)
                    for (uint16_t j = 0; j < block_size; ++j)
                        ws->Y_stored[y_offset + i * block_size + j] = ws->Y[i * ws->ib + j];
                
                memcpy(&ws->T_stored[t_offset], ws->T, block_size * block_size * sizeof(float));
            }
            
            // Apply to trailing columns
            for (uint16_t j = 0; j < block_size && (k + j) < n; ++j) {
                uint16_t reflector_len = m - (k + j);
                ws->tmp[0] = 1.0f;
                for (uint16_t i = 1; i < reflector_len; ++i)
                    ws->tmp[i] = ws->Cpack[(k + j) * n + (k + j) + i * n];
                
                uint16_t row_start = k + j;
                float *trailing = &ws->Cpack[row_start * n + (k + block_size)];
                
                apply_householder_clean(trailing, reflector_len, cols_right,
                                        n, ws->tmp, ws->tau[k + j]);
            }
        }
        
        block_count++;
    }
    
    //==========================================================================
    // Extract R
    //==========================================================================
    
    qr_extract_r(R, ws->Cpack, m, n);
    
    //==========================================================================
    // Form Q (if needed)
    //==========================================================================
    
    if (!only_R && Q) {
        int ret = qr_form_q(ws, Q, m, n, block_count);
        if (ret != 0) {
            qr_workspace_free(ws);
            return ret;
        }
    }
    
    qr_workspace_free(ws);
    return 0;
}

/**
 * @brief Fast Householder application for contiguous memory
 * 
 * When stride = n (row-major, contiguous columns), we can treat
 * the matrix as a 1D array for better vectorization
 */
static void apply_householder_contiguous(float *restrict C, uint16_t m, uint16_t n,
                                         const float *restrict v, float tau)
{
    if (tau == 0.0f)
        return;
    
    // Treat as m×n contiguous array
    uint16_t total = m * n;
    
    //==========================================================================
    // Vectorized dot products (all columns at once)
    //==========================================================================
    
    float dots[n];
    
    for (uint16_t j = 0; j < n; ++j) {
        double dot = 0.0;
        
#ifdef __AVX2__
        __m256d sum_vec = _mm256_setzero_pd();
        
        for (uint16_t i = 0; i < m - 3; i += 4) {
            __m256d v_vec = _mm256_cvtps_pd(_mm_loadu_ps(&v[i]));
            __m256d c_vec = _mm256_cvtps_pd(_mm_loadu_ps(&C[i * n + j]));
            sum_vec = _mm256_fmadd_pd(v_vec, c_vec, sum_vec);
        }
        
        // Horizontal reduction
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        dot = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Scalar tail
        for (uint16_t i = (m/4)*4; i < m; ++i)
            dot += (double)v[i] * (double)C[i * n + j];
#else
        for (uint16_t i = 0; i < m; ++i)
            dot += (double)v[i] * (double)C[i * n + j];
#endif
        
        dots[j] = tau * (float)dot;
    }
    
    //==========================================================================
    // Vectorized updates (linear memory access)
    //==========================================================================
    
    for (uint16_t j = 0; j < n; ++j) {
        float tau_dot = dots[j];
        
#ifdef __AVX2__
        __m256 tau_dot_vec = _mm256_set1_ps(tau_dot);
        
        for (uint16_t i = 0; i < m - 7; i += 8) {
            __m256 v_vec = _mm256_loadu_ps(&v[i]);
            __m256 c_vec = _mm256_loadu_ps(&C[i * n + j]);
            c_vec = _mm256_fnmadd_ps(v_vec, tau_dot_vec, c_vec);  // c -= v * tau_dot
            _mm256_storeu_ps(&C[i * n + j], c_vec);
        }
        
        // Scalar tail
        for (uint16_t i = (m/8)*8; i < m; ++i)
            C[i * n + j] -= v[i] * tau_dot;
#else
        for (uint16_t i = 0; i < m; ++i)
            C[i * n + j] -= v[i] * tau_dot;
#endif
    }
}

/**
 * @brief Dispatch to contiguous or strided version
 */
static inline void apply_householder_auto(float *restrict C, uint16_t m, uint16_t n,
                                          uint16_t ldc, const float *restrict v, float tau)
{
    if (ldc == n) {
        // FAST PATH: Contiguous memory
        apply_householder_contiguous(C, m, n, v, tau);
    } else {
        // SLOW PATH: Strided memory
        apply_householder_clean(C, m, n, ldc, v, tau);
    }
}

/**
 * @brief QR decomposition with automatic optimization selection
 * 
 * Automatically selects best algorithm based on matrix shape:
 * - Small square (≤128×128): Single-block, cache-resident
 * - Tall-skinny (m>4n): Optimized for least squares
 * - General: Standard blocked algorithm
 */
int qr_auto(const float *A, float *Q, float *R,
            uint16_t m, uint16_t n, bool only_R)
{
    //==========================================================================
    // Input validation
    //==========================================================================
    
    if (!A || !R || m == 0 || n == 0)
        return -EINVAL;
    
    //==========================================================================
    // FAST PATH 1: Tiny matrices (≤ 32×32)
    //==========================================================================
    
    if (m <= 32 && n <= 32) {
        // Single block, minimal overhead
        qr_workspace *ws = qr_workspace_alloc_ex(m, n, MIN(m, n), !only_R);
        if (!ws) return -ENOMEM;
        int ret = qr_ws_blocked(ws, A, Q, R, m, n, only_R);
        qr_workspace_free(ws);
        return ret;
    }
    
    //==========================================================================
    // FAST PATH 2: Small square (≤ 128×128)
    //==========================================================================
    
    if (m == n && m <= 128) {
        return qr_small_square(A, Q, R, m, only_R);
    }
    
    //==========================================================================
    // FAST PATH 3: Tall-skinny (m > 4n, common in least squares)
    //==========================================================================
    
    if (m > 4 * n && n <= 256) {
        return qr_tall_skinny(A, Q, R, m, n, only_R);
    }
    
    //==========================================================================
    // FAST PATH 4: Wide (n > 4m, less common but optimize anyway)
    //==========================================================================
    
    if (n > 4 * m && m <= 256) {
        // Use smaller block size for wide matrices
        qr_workspace *ws = qr_workspace_alloc_ex(m, n, MIN(32, m), !only_R);
        if (!ws) return -ENOMEM;
        int ret = qr_ws_blocked(ws, A, Q, R, m, n, only_R);
        qr_workspace_free(ws);
        return ret;
    }
    
    //==========================================================================
    // GENERAL PATH: Standard blocked QR with adaptive blocking
    //==========================================================================
    
    return qr_blocked(A, Q, R, m, n, only_R);
}