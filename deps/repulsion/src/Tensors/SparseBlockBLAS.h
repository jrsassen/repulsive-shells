#pragma once

namespace Tensors
{
    template<typename T, typename I, int BLK_ROW, int BLK_COL, typename T_in = T, typename T_out = T>
    class SparseBlockBLAS
    {
        ASSERT_INT(I);
        
    public:
        SparseBlockBLAS()
        {
//            ptic("SparseBLAS()");
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                thread_count = static_cast<I>(omp_get_num_threads());
            }
//            ptoc("SparseBlockBLAS()");

        };
        
        explicit SparseBlockBLAS( const I thread_count_ )
        : thread_count(thread_count_)
        {};
        
        ~SparseBlockBLAS() = default;
        
    protected:
        
        I thread_count = 1;
        
    protected:
        
        void Scale( T_out * restrict const y, const T_out beta, const I size, const I thread_count_ )
        {
            #pragma omp parallel for num_threads( thread_count_ ) schedule( static )
            for( I i = 0; i < size; ++i )
            {
                y[i] *= beta;
            }
        }
        
    public:
        
        void Multiply_GeneralMatrix_DenseMatrix
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            T const * restrict const a,
            I const m,
            I const n,
            T_in  const * restrict const X,
            const T_out beta,
            T_out       * restrict const Y,
            const I cols
        )
        {
            auto job_ptr = BalanceWorkLoad<I>(m,rp,thread_count,false);
            
            Multiply_GeneralMatrix_DenseMatrix(alpha,rp,ci,a,m,n,X,beta,Y,cols,job_ptr);
        }
        
        void Multiply_GeneralMatrix_DenseMatrix
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            T const * restrict const a,
            I const m,
            I const n,
            T_in  const * restrict const X,
            const T_out beta,
            T_out       * restrict const Y,
            const I cols,
            const Tensor1<I,I> job_ptr
        )
        {
//            ptic(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix");
            
            if( rp[m] <= 0 )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    wprint(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix: No nonzeroes found and beta = 0. Overwriting by 0.");
                    memset( Y, 0, m * cols * sizeof(T_out) );
                }
                else
                {
                    if( beta == static_cast<T_out>(1) )
                    {
                        wprint(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix: No nonzeroes found and beta = 1. Doing nothing.");
                    }
                    else
                    {
                        wprint(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix: No nonzeroes found. Just scaling by beta = "+ToString(beta)+".");
                        
                        Scale( Y, beta, m * cols, job_ptr.Size()-1 );
                    }
                }
                goto exit;
            }
            
            switch( cols )
            {
                case 3:
                {
                    gemm<3>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 9:
                {
                    gemm<9>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 1:
                {
//                    Multiply_GeneralMatrix_Vector(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    gemm<1>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 2:
                {
                    gemm<2>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 4:
                {
                    gemm<4>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
//                case 6:
//                {
//                    gemm<6>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
//                    break;
//                }
                case 8:
                {
                    gemm<8>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
//                case 10:
//                {
//                    gemm<10>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);;
//                    break;
//                }
//                case 12:
//                {
//                    gemm<12>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
//                    break;
//                }
//                case 16:
//                {
//                    gemm<16>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
//                    break;
//                }
                default:
                {
                    wprint(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix: falling back to gemm_gen for cols = "+ToString(cols)+".");
//                    gemm_gen(alpha,rp,ci,a,m,n,X,beta,Y,cols,job_ptr);
                }
            }
            
            exit:
            
//            ptoc(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix");
            return;
        }
        
    protected:
        
        template<int COLS>
        void gemm
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            T const * restrict const a,
            I const b_m,
            I const b_n,
            T_in  const * restrict const X,
            const T_out beta,
            T_out       * restrict const Y,
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::gemm<"+ToString(cols)+">");
            
            if( beta == static_cast<T>(0) )
            {
                logprint("beta == 0");
                if( alpha == static_cast<T>(0) )
                {
                    logprint("alpha == 0");
                    memset( Y, 0, b_m * BLK_ROW * COLS * sizeof(T_out) );
                }
                else
                {
                    logprint("alpha != 0");
//                    loglogprint("alpha != 0");
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        T Y_ [BLK_ROW][COLS] = {};
                        
                        const I b_i_begin = job_ptr[thread  ];
                        const I b_i_end   = job_ptr[thread+1];
                        
                        for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
                        {
                            memset( &Y_[0], 0, BLK_ROW * COLS * sizeof(T) );
                            
                            const I b_l_begin = rp[b_i  ];
                            const I b_l_end   = rp[b_i+1];
                            
                            for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )
                            {
                                const I b_j = ci[b_l];
                                
                                const T    * restrict const a_ = &a[BLK_ROW * BLK_COL * b_l];
                                const T_in * restrict const X_ = &X[BLK_COL * COLS * b_j];
                                
                                for( I i = 0; i < BLK_ROW; ++i )
                                {
                                    const T a_i = &a_[BLK_COL * i];
                                    
                                    for( I j = 0; j < BLK_COL; ++j )
                                    {
                                        const T a_i_j = a_i[j];
                                        
                                        const T_in * restrict const X_j = &X_[COLS * j];
                                        
                                        for( I k = 0; k < COLS; ++k )
                                        {
                                            Y_[i][k] = std::fma( a_i_j , static_cast<T>(X_j[k]), Y_[i][k] );
                                        }
                                    }
                                    
                                } // for( I i = 0; i < BLK_ROW; ++i )

                            } // for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )

                            for( I i = 0; i < BLK_ROW; ++i )
                            {
                                T_out const * restrict const Y_i = &Y[BLK_ROW * COLS * b_i + COLS * i];
                                
                                for( I k = 0; k < COLS; ++k )
                                {
                                    Y_i[k] = std::fma( beta, Y_i[k], static_cast<T_out>(alpha * Y_[i][k]) );
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                logprint("beta != 0\n");
                if( alpha == static_cast<T>(0) )
                {
                    logprint("alpha == 0");
                    
                    Scale( Y, beta, b_m * BLK_ROW * COLS, job_ptr.Size()-1 );
                }
                else
                {
                    logprint("alpha != 0");
                    //                    logprint("alpha != 0\n");
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        T Y_ [BLK_ROW][  COLS] = {};
                        
                        const I b_i_begin = job_ptr[thread  ];
                        const I b_i_end   = job_ptr[thread+1];
                        
                        for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
                        {
                            memset( &Y_[0], 0, BLK_ROW * COLS * sizeof(T) );
                            
                            const I b_l_begin = rp[b_i  ];
                            const I b_l_end   = rp[b_i+1];
                            
                            for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )
                            {
                                const I b_j = ci[b_l];
                                
                                const T     * restrict const a_ = &a[BLK_ROW * BLK_COL * b_l];
                                const T_in  * restrict const X_ = &X[BLK_COL * COLS * b_j];
                                
                                for( I i = 0; i < BLK_ROW; ++i )
                                {
                                    T const * restrict const a_i = &a_[BLK_COL * i];
                                    for( I j = 0; j < BLK_COL; ++j )
                                    {
                                        const T a_i_j = a_i[j];
                                        
                                        const T_in * restrict const X_j = &X_[COLS * j];
                                        
                                        for( I k = 0; k < COLS; ++k )
                                        {
                                            Y_[i][k] =  std::fma( a_i_j, static_cast<T>(X_j[k]), Y_[i][k] );
                                        }
                                    }
                                }
                                
                            }

                            T_out const * restrict const Y__ = &Y[BLK_ROW * COLS * b_i];
                            
                            for( I i = 0; i < BLK_ROW; ++i )
                            {
                                T_out const * restrict const Y_i = &Y__[COLS * i];
                                
                                for( I k = 0; k < COLS; ++k )
                                {
                                    Y_i[k] = std::fma( beta, Y_i[k], static_cast<T_out>(alpha * Y_[i][k]) );
                                }
                            }
                        }
                    }
                }
            }
        }
        
        
//        template<int COLS>
//        void gemm2
//        (
//            const T alpha,
//            I const * restrict const rp,
//            I const * restrict const ci,
//            T const * restrict const a,
//            I const b_m,
//            I const b_n,
//            T_in  const * restrict const X,
//            const T_out beta,
//            T_out       * restrict const Y,
//            const Tensor1<I,I> & job_ptr
//        )
//        {
////            ptic(ClassName()+"::gemm<"+ToString(cols)+">");
//
//            if( beta == static_cast<T>(0) )
//            {
//                logprint("beta == 0");
//                if( alpha == static_cast<T>(0) )
//                {
//                    logprint("alpha == 0");
//                    memset( Y, 0, b_m * BLK_ROW * COLS * sizeof(T_out) );
//                }
//                else
//                {
////                    logprint("alpha != 0");
//                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
//
//                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
//                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
//                    {
//                        T A_ [BLK_ROW][BLK_COL] = {};
//                        T X_ [BLK_COL][  COLS] = {};
//                        T Y_ [BLK_ROW][  COLS] = {};
//
//                        const I b_i_begin = job_ptr[thread  ];
//                        const I b_i_end   = job_ptr[thread+1];
//
//                        for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
//                        {
//                            memset( &Y_[0], 0, BLK_ROW * COLS * sizeof(T) );
//
//                            const I b_l_begin = rp[b_i  ];
//                            const I b_l_end   = rp[b_i+1];
//
//                            for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )
//                            {
//                                const I b_j = ci[b_l];
//
//                                memcpy( &A_[0], a + BLK_ROW * BLK_COL * b_l, BLK_ROW * BLK_COL * sizeof(T) );
//                                memcpy( &X_[0], X + BLK_ROW * COLS   * b_j, BLK_ROW * COLS * sizeof(T_in) );
//
//                                for( I i = 0; i < BLK_ROW; ++i )
//                                {
//                                    for( I j = 0; j < BLK_COL; ++j )
//                                    {
//                                        for( I k = 0; k < COLS; ++k )
//                                        {
//                                            Y_[i][k] = std::fma(A_[i][j], X_[j][k], Y_[i][k]);
//                                        }
//                                    }
//                                }
//
//                            }
//
//                            T_out * restrict const Y__ = Y + BLK_ROW * COLS * b_i;
//
//                            for( I i = 0; i < BLK_ROW; ++i )
//                            {
//                                for( I k = 0; k < COLS; ++k )
//                                {
//                                    Y__[COLS * i + k] = static_cast<T_out>(alpha * Y_[i][k]);
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            else
//            {
//                logprint("beta != 0\n");
//                if( alpha == static_cast<T>(0) )
//                {
//                    logprint("alpha == 0");
//                    #pragma omp parallel for simd num_threads( job_ptr.Size()-1 ) schedule( static )
//                    for( I i = 0; i < b_m * BLK_ROW * COLS; ++i )
//                    {
//                        Y[i] *= beta;
//                    }
//                }
//                else
//                {
//                    //                    logprint("alpha != 0\n");
//                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
//                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
//                    {
//                        T A_ [BLK_ROW][BLK_COL] = {};
//                        T X_ [BLK_COL][  COLS] = {};
//                        T Y_ [BLK_ROW][  COLS] = {};
//
//                        const I b_i_begin = job_ptr[thread  ];
//                        const I b_i_end   = job_ptr[thread+1];
//
//                        for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
//                        {
//                            memset( &Y_[0], 0, BLK_ROW * COLS * sizeof(T) );
//
//                            const I b_l_begin = rp[b_i  ];
//                            const I b_l_end   = rp[b_i+1];
//
//                            for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )
//                            {
//                                const I b_j = ci[b_l];
//
//                                memcpy( &A_[0], a + BLK_ROW * BLK_COL * b_l, BLK_ROW * BLK_COL * sizeof(T) );
//                                memcpy( &X_[0], X + BLK_ROW * COLS   * b_j, BLK_ROW * COLS * sizeof(T_in) );
//
//                                for( I i = 0; i < BLK_ROW; ++i )
//                                {
//                                    for( I j = 0; j < BLK_COL; ++j )
//                                    {
//                                        for( I k = 0; k < COLS; ++k )
//                                        {
//                                            Y_[i][k] = std::fma(A_[i][j], X_[j][k], Y_[i][k]);
//                                        }
//                                    }
//                                }
//
//                            }
//
//                            T_out * restrict const Y__ = Y + BLK_ROW * COLS * b_i;
//
//                            for( I i = 0; i < BLK_ROW; ++i )
//                            {
//                                for( I k = 0; k < COLS; ++k )
//                                {
//                                    Y__[COLS * i + k] = std::fma(
//                                                            beta,
//                                                            Y__[COLS * i + k],
//                                                            static_cast<T_out>(alpha * Y_[i][k])
//                                                        );
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
        

//        void gemm_gen
//        (
//            const T alpha,
//            I const * restrict const rp,
//            I const * restrict const ci,
//            T const * restrict const a,
//            I const b_m,
//            I const b_n,
//            T_in  const * restrict const X,
//            const T_out beta,
//            T_out       * restrict const Y,
//            const I COLS,
//            const Tensor1<I,I> & job_ptr
//        )
//        {
////            ptic(ClassName()+"::gemm_gen ("+ToString(cols)+")");
//
//            if( beta == static_cast<T>(0) )
//            {
//                logprint("beta == 0");
//                if( alpha == static_cast<T>(0) )
//                {
//                    logprint("alpha == 0");
//                    memset( Y, 0, b_m * BLK_ROW * COLS * sizeof(T_out) );
//                }
//                else
//                {
////                    logprint("alpha != 0");
//                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
//
//                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
//                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
//                    {
//                        T A_ [BLK_ROW][BLK_COL] = {};
//                        T X_ [BLK_COL][COLS      ] = {};
//                        T Y_ [BLK_ROW][COLS      ] = {};
//
//                        const I b_i_begin = job_ptr[thread  ];
//                        const I b_i_end   = job_ptr[thread+1];
//
//                        for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
//                        {
//                            memset( &Y_[0], 0, BLK_ROW * COLS * sizeof(T) );
//
//                            const I b_l_begin = rp[b_i  ];
//                            const I b_l_end   = rp[b_i+1];
//
//                            for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )
//                            {
//                                const I b_j = ci[b_l];
//
//                                memcpy( &A_[0], a + BLK_ROW * BLK_COL * b_l, BLK_ROW * BLK_COL * sizeof(T) );
//                                memcpy( &X_[0], X + BLK_ROW * COLS   * b_j, BLK_ROW * COLS * sizeof(T_in) );
//
//                                for( I i = 0; i < BLK_ROW; ++i )
//                                {
//                                    for( I j = 0; j < BLK_COL; ++j )
//                                    {
//                                        const T A_i_j = A_[i][j];
//
//                                        for( I k = 0; k < COLS; ++k )
//                                        {
//                                            Y_[i][k] = std::fma(A_i_j, X_[j][k], Y_[i][k]);
//                                        }
//                                    }
//                                }
//
//                            }
//
//                            T_out * restrict const Y__ = Y + BLK_ROW * COLS * b_i;
//
//                            for( I i = 0; i < BLK_ROW; ++i )
//                            {
//                                for( I k = 0; k < COLS; ++k )
//                                {
//                                    Y__[COLS * i + k] = static_cast<T_out>(alpha * Y_[i][k]);
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            else
//            {
//                logprint("beta != 0\n");
//                if( alpha == static_cast<T>(0) )
//                {
//                    logprint("alpha == 0");
//                    #pragma omp parallel for simd num_threads( job_ptr.Size()-1 ) schedule( static )
//                    for( I i = 0; i < b_m * BLK_ROW * COLS; ++i )
//                    {
//                        Y[i] *= beta;
//                    }
//                }
//                else
//                {
//                    //                    logprint("alpha != 0\n");
//                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
//                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
//                    {
//                        T A_ [BLK_ROW][BLK_COL] = {};
//                        T X_ [BLK_COL][COLS      ] = {};
//                        T Y_ [BLK_ROW][COLS      ] = {};
//
//                        const I b_i_begin = job_ptr[thread  ];
//                        const I b_i_end   = job_ptr[thread+1];
//
//                        for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
//                        {
//                            memset( &Y_[0], 0, BLK_ROW * COLS * sizeof(T) );
//
//                            const I b_l_begin = rp[b_i  ];
//                            const I b_l_end   = rp[b_i+1];
//
//                            for( I b_l = b_l_begin; b_l < b_l_end; ++b_l )
//                            {
//                                const I b_j = ci[b_l];
//
//                                memcpy( &A_[0], a + BLK_ROW * BLK_COL * b_l, BLK_ROW * BLK_COL * sizeof(T) );
//                                memcpy( &X_[0], X + BLK_ROW * COLS   * b_j, BLK_ROW * COLS * sizeof(T_in) );
//
//                                for( I i = 0; i < BLK_ROW; ++i )
//                                {
//                                    for( I j = 0; j < BLK_COL; ++j )
//                                    {
//                                        const T A_i_j = A_[i][j];
//
//                                        for( I k = 0; k < COLS; ++k )
//                                        {
//                                            Y_[i][k] = std::fma(A_i_j, X_[j][k], Y_[i][k]);
//                                        }
//                                    }
//                                }
//
//                            }
//
//                            T_out * restrict const Y__ = Y + BLK_ROW * COLS * b_i;
//
//                            for( I i = 0; i < BLK_ROW; ++i )
//                            {
//                                for( I k = 0; k < COLS; ++k )
//                                {
//                                    Y__[COLS * i + k] = std::fma(
//                                                            beta,
//                                                            Y__[COLS * i + k],
//                                                            static_cast<T_out>(alpha * Y_[i][k])
//                                                        );
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
////            ptoc(ClassName()+"::gemm_gen ("+ToString(cols)+")");
//        }
        
        
    public:
        
        static std::string ClassName()
        {
            return "SparseBlockBLAS<"+TypeName<T>::Get()+","+TypeName<I>::Get()+","+ToString(BLK_ROW)+","+ToString(BLK_COL)+","+TypeName<T_in>::Get()+","+TypeName<T_out>::Get()+">";
        }
        
    }; // SparseBlockBLAS
    
    
} // namespace Tensors


