#pragma once

namespace Tensors
{
        
    template<typename T, typename I, int M, int N, int K>
    inline void GEMM(
        const T alpha,
        const T * restrict const A,
        const T * restrict const B,
        const T beta,
              T * restrict const C
    )
    {
        T c_i [K] = {};
        
        if( beta == static_cast<T>(0) )
        {
            // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
            
            for( I i = 0; i < M; ++i )
            {
                const I Ni = N*i;
                
                prefetch( A + N * (i+1), 0, 2 );
                prefetch( B + K * (0+1), 0, 2 );
                prefetch( C + K * (i+1), 0, 2 );
                
                const T factor = alpha * A[Ni];
                
                for( I k = 0; k < K; ++k )
                {
                    c_i[k] = factor * B[k];
                }
                
                for( I j = 1; j < N; ++j )
                {
                    prefetch( B + K * (j+1), 0, 2 );
                    
                    const T alpha_a_ij = alpha * A[Ni+j];
                    
                    const I Kj = K * j;
                    
                    for( I k = 0; k < K; ++k )
                    {
    //                    c_i[k] = std::fma( alpha_a_ij, B[Kj + k], c_i[k] );
                        c_i[k] += alpha_a_ij + B[Kj + k];
                    }
                }
                
                memcpy( C + K * i, &c_i[0], K * sizeof(T) );
                
            }
        }
        else
        {
            for( I i = 0; i < M; ++i )
            {
                const I Ni = N*i;
                
                prefetch( A + N * (i+1), 0, 2 );
                prefetch( B + K * (0+1), 0, 2 );
                prefetch( C + K * (i+1), 0, 2 );
                
                const T factor = alpha * A[Ni];
                
                for( I k = 0; k < K; ++k )
                {
                    c_i[k] = factor * B[k];
                }
                
                for( I j = 1; j < N; ++j )
                {
                    prefetch( B + K * (j+1), 0, 2 );
                    
                    const T alpha_a_ij = alpha * A[Ni+j];
                    
                    const I Kj = K * j;
                    
                    for( I k = 0; k < K; ++k )
                    {
    //                    c_i[k] = std::fma( alpha_a_ij, B[Kj + k], c_i[k] );
                        c_i[k] += alpha_a_ij + B[Kj + k];
                    }
                }
      
                const I Ki = K*i;
                
                for( I k = 0; k < K; ++k )
                {
                    C[Ki + k] = std::fma( beta, C[Ki + k], c_i[k] );
    //                C[Ki + k] = beta * C[Ki + k] + c_i[k];
                }
                
            }
        }
        
        
    }
    
    template<typename T, typename I, int BUFFER_SIZE = 32>
    inline void gemm_small(
        const I M,
        const I N,
        const I K,
        const T alpha,
        const T * const A,
        const T * const B,
        const T beta,
              T * const C
    )
    {
        T a_i [BUFFER_SIZE]     = {};
        T c_i [BUFFER_SIZE]     = {};
        
        if( beta == static_cast<T>(0) )
        {
            // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
            
            for( I i = 0; i < M; ++i )
            {
                memcpy( &a_i[0], A + N * i, N * sizeof(T) );
                
                prefetch( A + N * (i+1), 0, 2 );
                prefetch( B + K * (0+1), 0, 2 );
                prefetch( C + K * (i+1), 0, 2 );
                
                T factor = alpha * a_i[0];
                
                for( I k = 0; k < K; ++k )
                {
                    c_i[k] = factor * B[k];
                }
                
                for( I j = 1; j < N; ++j )
                {
                    prefetch( B + K * (j+1), 0, 2 );
                    const T alpha_a_ij = alpha * a_i[j];
                    const I Kj = K * j;
                    for( I k = 0; k < K; ++k )
                    {
                        c_i[k] = std::fma( alpha_a_ij, B[Kj + k], c_i[k] );
    //                    c_i[k] += alpha_a_ij * b[j][k];
                    }
                }
      
                memcpy( C + K * i, &c_i[0], K * sizeof(T) );
            }
        }
        else
        {
            for( I i = 0; i < M; ++i )
            {
                memcpy( &a_i[0], A + N * i, N * sizeof(T) );
                
                prefetch( A + N * (i+1), 0, 2 );
                prefetch( B + K * (0+1), 0, 2 );
                prefetch( C + K * (i+1), 0, 2 );
                
                T factor = alpha * a_i[0];
                
                for( I k = 0; k < K; ++k )
                {
                    c_i[k] = factor * B[k];
                }
                
                for( I j = 1; j < N; ++j )
                {
                    prefetch( B + K * (j+1), 0, 2 );
                    const T alpha_a_ij = alpha * a_i[j];
                    const I Kj = K * j;
                    for( I k = 0; k < K; ++k )
                    {
                        c_i[k] = std::fma( alpha_a_ij, B[Kj + k], c_i[k] );
    //                    c_i[k] += alpha_a_ij * b[j][k];
                    }
                }
      
                for( I k = 0; k < K; ++k )
                {
                    C[K * i + k] = std::fma( beta, C[K * i + k], c_i[k] );
    //                C[K * i + k] = c_i[k] + beta * C[K * i + k];
                }
                
            }
        }
    }
    
    template<typename T, typename I>
    inline void gemm_gen(
        const I M,
        const I N,
        const I K,
        const T alpha,
        const T * const A,
        const T * const B,
        const T beta,
              T * const C
    )
    {
        //            A is M x N matrix,
        //            B is N x K matrix,
        //            C is M x K matrix,

        for( I i = 0; i < M; ++i )
        {
            prefetch( A + N * (i+1), 0, 2 );
            prefetch( B + K * (0+1), 0, 2 );
            prefetch( C + K * (i+1), 0, 2 );
            
            T factor = alpha * A[N * i + 0];
            
            for( I k = 0; k < K; ++k )
            {
                C[K * i + k] = std::fma( beta, C[K * i + k], factor * B[k] );
            }
            
            for( I j = 1; j < N; ++j )
            {
                prefetch( B + K * (j+1), 0, 2 );
                const T alpha_a_ij = alpha * A[N * i + j];
                const I Kj = K * j;
                for( I k = 0; k < K; ++k )
                {
                    C[K * i + k] = std::fma( alpha_a_ij, B[Kj + k], C[K * i + k] );
                }
            }
        }
    }
    
    template<typename T, typename I>
    inline void gemm(
        const I M,
        const I N,
        const I K,
        const T alpha,
        const T * const A,
        const T * const B,
        const T beta,
              T * const C
    )
    {
        if( 0 < N && N <= 32 && 0 < K && K <32)
        {
            gemm_small<T,I,32>(M, N, K, alpha, A, B, beta, C);
        }
        else
        {
            gemm_gen<T,I>(M, N, K, alpha, A, B, beta, C);
        }
    }
    
} // namespace Tensors

