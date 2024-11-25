#pragma once

namespace Tensors
{
    template<typename T, typename I, typename T_in, typename T_out>
    class SparseBLAS
    {
        ASSERT_INT(I);

    public:
        SparseBLAS()
        {
//            ptic("SparseBLAS()");
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                thread_count = static_cast<I>(omp_get_num_threads());
            }
//            ptoc("SparseBLAS()");

        };
        
        explicit SparseBLAS( const I thread_count_ )
        : thread_count(thread_count_)
        {
//            ptic("SparseBLAS()");
//            ptoc("SparseBLAS()");
        };
        
        ~SparseBLAS() = default;
        
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
        
        void Multiply_GeneralMatrix_Vector
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            T const * restrict const a,
            I const m,
            I const n,
            T_in  const * restrict const x,
            const T_out beta,
            T_out       * restrict const y
        )
        {
            auto job_ptr = BalanceWorkLoad<I>(m,rp,thread_count,false);
            
            Multiply_GeneralMatrix_Vector(alpha,rp,ci,a,m,n,x,beta,y,job_ptr);
        }
        
        void Multiply_GeneralMatrix_Vector
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            T const * restrict const a,
            I const m,
            I const n,
            T_in  const * restrict const x,
            const T_out beta,
            T_out       * restrict const y,
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::Multiply_GeneralMatrix_Vector");
            
            if( rp[m] <= 0 )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    wprint(ClassName()+"::Multiply_GeneralMatrix_Vector: No nonzeroes found and beta = 0. Overwriting by 0.");
                    std::memset( y, 0, m * sizeof(T_out) );
                }
                else
                {
                    if( beta == static_cast<T_out>(1) )
                    {
                        wprint(ClassName()+"::Multiply_GeneralMatrix_Vector: No nonzeroes found and beta = 1. Doing nothing.");
                    }
                    else
                    {
                        wprint(ClassName()+"::Multiply_GeneralMatrix_Vector: No nonzeroes found. Just scaling by beta = "+ToString(beta)+".");
                        
                        
                        Scale( y, beta, m, job_ptr.Size()-1);
                    }
                }
                goto exit;
            }
            
            if( beta == static_cast<T_out>(0) )
            {
                if( alpha == static_cast<T>(0) )
                {
                    std::memset( y, 0, m * sizeof(T_out) );
                }
                else
                {
                    
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];

                        for( I i = i_begin; i < i_end; ++i )
                        {
                            T sum = static_cast<T>(0);

                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            __builtin_prefetch( ci + l_end );
                            __builtin_prefetch( a  + l_end );
                        
                            #pragma omp simd reduction( + : sum )
                            for( I l = l_begin; l < l_end; ++l )
                            {
                                const I j = ci[l];
                                
                                sum += a[l] * static_cast<T>(x[j]);
//                                sum = std::fma(a[l], x[j], sum);
                            }

                            y[i] = static_cast<T_out>(alpha * sum);
                        }
                    }
                }
            }
            else
            {
                if( alpha == static_cast<T>(0) )
                {
                    Scale( y, beta, m, job_ptr.Size()-1);
                }
                else
                {
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {

                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];

                        for( I i = i_begin; i < i_end; ++i )
                        {
                            T sum = static_cast<T>(0);

                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            __builtin_prefetch( ci + l_end );
                            __builtin_prefetch( a  + l_end );
                        
                            #pragma omp simd reduction( + : sum )
                            for( I l = l_begin; l < l_end; ++l )
                            {
                                const I j = ci[l];
                                
                                sum += a[l] * static_cast<T>(x[j]);
//                                sum = std::fma(a[l], x[j], sum);
                            }

                            y[i] = beta * y[i] + static_cast<T_out>(alpha * sum);
//                            y[i] = std::fma(beta, y[i], alpha * sum);
                        }
                    }
                }
            }

        exit:
            
//            ptoc(ClassName()+"::Multiply_GeneralMatrix_Vector");
            return;
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
                    std::memset( Y, 0, m * cols * sizeof(T_out) );
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

                        
                        Scale( Y, beta, m * cols, job_ptr.Size()-1);

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
                case 6:
                {
                    gemm<6>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 8:
                {
                    gemm<8>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 10:
                {
                    gemm<10>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);;
                    break;
                }
                case 12:
                {
                    gemm<12>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 16:
                {
                    gemm<16>(alpha,rp,ci,a,m,n,X,beta,Y,job_ptr);
                    break;
                }
                default:
                {
                    wprint(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix: falling back to gemm_gen for cols = "+ToString(cols)+".");
                    gemm_gen(alpha,rp,ci,a,m,n,X,beta,Y,cols,job_ptr);
                }
            }
            
            exit:
            
//            ptoc(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix");
            return;
        }
        
    protected:
        
        template<int cols>
        void gemm
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
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::gemm<"+ToString(cols)+">");
            
            if( beta == static_cast<T>(0) )
            {
//                logprint("beta == 0");
                if( alpha == static_cast<T>(0) )
                {
//                    logprint("alpha == 0");
                    std::memset( Y, 0, m * cols * sizeof(T_out) );
                }
                else
                {
//                    logprint("alpha != 0");
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        T sum [cols] = {};
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            if( l_end > l_begin)
                            {
                                const T a_l = a[l_begin];

                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = a_l * static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            // Add the others.
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T a_l = a[l];

                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] += a_l * static_cast<T>(x[k]);
//                                    sum[k] = std::fma( a_l, static_cast<T>(x[k]), sum[k]);
                                }
                            }

                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
                                y[k] = static_cast<T_out>(alpha * sum[k]);
                            }
                        }
                    }
                }
            }
            else
            {
//                logprint("beta != 0\n");
                if( alpha == static_cast<T>(0) )
                {
//                    logprint("alpha == 0");
                    
                    Scale( Y, beta, m * cols, job_ptr.Size()-1);
                }
                else
                {
//                    logprint("alpha != 0\n");
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        T sum [cols] = {};
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            if( l_end > l_begin)
                            {
                                const T a_l = a[l_begin];

                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = a_l * static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            // Add the others first,
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T a_l = a[l];

                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
                    //                            sum[k] += a_l * x[k];
                                    sum[k] = std::fma( a_l, static_cast<T>(x[k]), sum[k] );
                                }
                            }
                            
                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
//                                y[k] = beta * y[k] + static_cast<T_out>(alpha * sum[k]);
                                y[k] = std::fma(beta, y[k], static_cast<T_out>(alpha * sum[k]) );
                            }
                            
                        }
                    }
                }
            }
            
//            ptoc(ClassName()+"::gemm<"+ToString(cols)+">");
        }
        

        void gemm_gen
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
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::gemm_gen ("+ToString(cols)+")");
            
            if( beta == static_cast<T>(0) )
            {
                if( alpha == static_cast<T>(0) )
                {
                    std::memset( Y, 0, m * cols * sizeof(T_out) );
                }
                else
                {
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        Tensor1<T,I> sum_buffer (cols);
                        
                        T * restrict const sum = sum_buffer.data();
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            if( l_end > l_begin)
                            {
                                const T a_l = a[l_begin];

                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = a_l * static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T a_l = a[l];

                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
//                                    sum[k] += a_l * static_cast<T>(x[k]);
                                    sum[k] = std::fma( a_l, static_cast<T>(x[k]), sum[k] );
                                }
                            }
                            
                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
                                y[k] = static_cast<T_out>(alpha * sum[k]);
                            }
                        }
                    }
                }
            }
            else
            {
                if( alpha == static_cast<T>(0) )
                {
                    Scale( Y, beta, m * cols, job_ptr.Size()-1);
                }
                else
                {
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        Tensor1<T,I> sum_buffer (cols);
                        
                        T * restrict const sum = sum_buffer.data();
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            if( l_end >  l_begin)
                            {
                                const T a_l = a[l_begin];

                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = a_l * static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T a_l = a[l];

                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
//                                    sum[k] += a_l * static_cast<T>(x[k]);
                                    sum[k] = std::fma( a_l, static_cast<T>(x[k]), sum[k] );
                                }
                            }
                            
                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
//                                y[k] = beta * y[k] + static_cast<T_out>(alpha * sum[k]);
                                y[k] = std::fma(beta, y[k], static_cast<T_out>(alpha * sum[k]));
                            }
                            
                        }
                    }
                }
            }
            
//            ptoc(ClassName()+"::gemm_gen ("+ToString(cols)+")");
        }
        
        
        
        
//#######################################################################################
//####                               Binary matrices                                #####
//#######################################################################################
        
    public:
        
        void Multiply_BinaryMatrix_Vector
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            I const m,
            I const n,
            T_in  const * restrict const x,
            const T_out beta,
            T_out       * restrict const y
        )
        {
            auto job_ptr = BalanceWorkLoad<I>(m,rp,thread_count,false);
            
            Multiply_BinaryMatrix_Vector(alpha,rp,ci,m,n,x,beta,y,job_ptr);
        }
        
        void Multiply_BinaryMatrix_Vector
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            I const m,
            I const n,
            T_in  const * restrict const x,
            const T_out beta,
            T_out       * restrict const y,
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::Multiply_BinaryMatrix_Vector");
            
            if( rp[m] <= 0 )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    wprint(ClassName()+"::Multiply_BinaryMatrix_Vector: No nonzeroes found and beta = 0. Overwriting by 0.");
                    std::memset( y, 0, m * sizeof(T_out) );
                }
                else
                {
                    if( beta == static_cast<T_out>(1) )
                    {
                        wprint(ClassName()+"::Multiply_BinaryMatrix_DenseMatrix: No nonzeroes found and beta = 1. Doing nothing.");
                    }
                    else
                    {
                        wprint(ClassName()+"::Multiply_BinaryMatrix_DenseMatrix: No nonzeroes found. Just scaling by beta = "+ToString(beta)+".");

                        Scale( y, beta, m, job_ptr.Size()-1);
                    }
                }
                goto exit;
            }
            
            if( beta == static_cast<T_out>(0) )
            {
                if( alpha == static_cast<T>(0) )
                {
                    std::memset( y, 0, m * sizeof(T_out) );
                    goto exit;
                }
                else
                {
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];

                        for( I i = i_begin; i < i_end; ++i )
                        {
                            T sum = static_cast<T>(0);

                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            __builtin_prefetch( ci + l_end );
                        
                            #pragma omp simd reduction( + : sum )
                            for( I l = l_begin; l < l_end; ++l )
                            {
                                const I j = ci[l];
                                
                                sum += static_cast<T>(x[j]);
                            }

                            y[i] = static_cast<T_out>(alpha * sum);
                        }
                    }
                }
            }
            else
            {
                if( alpha == static_cast<T>(0) )
                {
                    Scale( y, beta, m, job_ptr.Size()-1);
                }
                else
                {
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];

                        for( I i = i_begin; i < i_end; ++i )
                        {
                            T sum = static_cast<T>(0);

                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            __builtin_prefetch( ci + l_end );
                        
                            #pragma omp simd reduction( + : sum )
                            for( I l = l_begin; l < l_end; ++l )
                            {
                                const I j = ci[l];
                                
                                sum += static_cast<T>(x[j]);
                            }

                            y[i] = beta * y[i] + static_cast<T_out>(alpha * sum);
//                            y[i] = std::fma(beta, y[i], alpha * sum);
                        }
                    }
                }
            }
            
        exit:
            
//            ptoc(ClassName()+"::Multiply_BinaryMatrix_Vector");
            return;
        }
        
        
        
        void Multiply_BinaryMatrix_DenseMatrix
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            I const m,
            I const n,
            T_in  const * restrict const X,
            const T_out beta,
            T_out       * restrict const Y,
            const I cols
        )
        {
            auto job_ptr = BalanceWorkLoad<I>(m,rp,thread_count,false);
            
            Multiply_BinaryMatrix_DenseMatrix(alpha,rp,ci,m,n,X,beta,Y,cols,job_ptr);
        }
        
        void Multiply_BinaryMatrix_DenseMatrix
        (
         const T alpha,
         I const * restrict const rp,
         I const * restrict const ci,
         I const m,
         I const n,
         T_in  const * restrict const X,
         const T_out beta,
         T_out       * restrict const Y,
         const I cols,
         const Tensor1<I,I> job_ptr
        )
        {
//            ptic("Multiply_BinaryMatrix_DenseMatrix");
            
            if( rp[m] <= 0 )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    wprint(ClassName()+"::Multiply_BinaryMatrix_DenseMatrix: No nonzeroes found and beta = 0. Overwriting by 0.");
                    std::memset( Y, 0, m * cols * sizeof(T_out) );
                }
                else
                {
                    if( beta == static_cast<T_out>(1) )
                    {
                        wprint(ClassName()+"::Multiply_BinaryMatrix_DenseMatrix: No nonzeroes found and beta = 1. Doing nothing.");
                    }
                    else
                    {
                        wprint(ClassName()+"::Multiply_BinaryMatrix_DenseMatrix: No nonzeroes found. Just scaling by beta = "+ToString(beta)+".");

                        Scale( Y, beta, m * cols, job_ptr.Size()-1);
                    }
                }
                goto exit;
            }
            
            switch( cols )
            {
                case 3:
                {
                    bimm<3>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 9:
                {
                    bimm<9>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 1:
                {
//                    Multiply_BinaryMatrix_Vector(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    bimm<1>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 2:
                {
                    bimm<2>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 4:
                {
                    bimm<4>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 6:
                {
                    bimm<6>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 8:
                {
                    bimm<8>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 10:
                {
                    bimm<10>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 12:
                {
                    bimm<12>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                case 16:
                {
                    bimm<16>(alpha,rp,ci,m,n,X,beta,Y,job_ptr);
                    break;
                }
                    
                default:
                {
                    wprint(ClassName()+"::Multiply_BinaryMatrix_DenseMatrix: falling back to bimm_gen for cols = "+ToString(cols)+".");
                    bimm_gen(alpha,rp,ci,m,n,X,beta,Y,cols,job_ptr);
                }
            }
            
        exit:
            
//            ptoc("Multiply_BinaryMatrix_DenseMatrix");
            return;
        }
        
    protected:
        
        template<int cols>
        void bimm
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            I const m,
            I const n,
            T_in  const * restrict const X,
            const T_out beta,
            T_out       * restrict const Y,
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::bimm<"+ToString(cols)+">");
            
            if( beta == static_cast<T>(0) )
            {
                if( alpha == static_cast<T>(0) )
                {
                    std::memset( Y, 0, m * cols * sizeof(T_out) );
                }
                else
                {
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        T sum [cols] = {};
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            if( l_end > l_begin)
                            {
                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] += static_cast<T>(x[k]);
                                }
                            }

                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
                                y[k] = static_cast<T_out>(alpha * sum[k]);
                            }
                        }
                    }
                }
            }
            else
            {
                if( alpha == static_cast<T>(0) )
                {
                    Scale( Y, beta, m * cols, job_ptr.Size()-1);
                }
                else
                {
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        T sum [cols] = {};
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
                            
                            if( l_end > l_begin)
                            {
                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] += static_cast<T>(x[k]);
                                }
                            }
                            
                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
//                                y[k] = beta * y[k] + static_cast<T_out>(sum[k]);
                                y[k] = std::fma(beta, y[k], static_cast<T_out>(alpha * sum[k]) );
                            }
                            
                        }
                    }
                }
            }
            
//            ptoc(ClassName()+"::bimm<"+ToString(cols)+">");
        }
        

        void bimm_gen
        (
            const T alpha,
            I const * restrict const rp,
            I const * restrict const ci,
            I const m,
            I const n,
            T_in  const * restrict const X,
            const T_out beta,
            T_out       * restrict const Y,
            const I cols,
            const Tensor1<I,I> & job_ptr
        )
        {
            if( beta == static_cast<T>(0) )
            {
                if( alpha == static_cast<T>(0) )
                {
                    std::memset( Y, 0, m * cols * sizeof(T_out) );
                }
                else
                {
                    // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                    
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        Tensor1<T,I> sum_buffer (cols);
                        
                        T * restrict const sum = sum_buffer.data();
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
  
                            if( l_end >  l_begin)
                            {
                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] += static_cast<T>(x[k]);
                                }
                            }
                            
                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
                                y[k] = static_cast<T_out>(alpha * sum[k]);
                            }
                        }
                    }
                }
            }
            else
            {
                if( alpha == static_cast<T>(0) )
                {
                    Scale( Y, beta, m * cols, job_ptr.Size()-1);
                }
                else
                {
                    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                    for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                    {
                        Tensor1<T,I> sum_buffer (cols);
                        
                        T * restrict const sum = sum_buffer.data();
                        
                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];
                        
                        for( I i = i_begin; i < i_end; ++i )
                        {
                            const I l_begin = rp[i  ];
                            const I l_end   = rp[i+1];
  
                            if( l_end >  l_begin)
                            {
                                const T_in * restrict const x = X + cols * ci[l_begin];

                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] = static_cast<T>(x[k]);
                                }
                            }
                            else
                            {
                                std::memset( &sum[0], 0, cols * sizeof(T) );
                            }
                            
                            for( I l = l_begin+1; l < l_end; ++l )
                            {
                                const T_in * restrict const x = X + cols * ci[l];
                                
                                for( I k = 0; k < cols; ++k )
                                {
                                    sum[k] += static_cast<T>(x[k]);
                                }
                            }
                            
                            T_out * restrict const y = Y + cols * i;
                            
                            for( I k = 0; k < cols; ++k )
                            {
//                                y[k] = beta * y[k] + static_cast<T_out>(alpha * sum[k]);
                                y[k] = std::fma(beta, y[k], static_cast<T_out>(alpha * sum[k]));
                            }
                            
                        }
                    }
                }
            }
            
        }
        
    public:
        
        static std::string ClassName()
        {
            return "SparseBLAS<"+TypeName<T>::Get()+","+TypeName<I>::Get()+","+TypeName<T_in>::Get()+","+TypeName<T_out>::Get()+">";
        }
        
        
        
        template<int cols>
        void transposed_gemm
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
            ThreadTensor3<T_out,I> & Y_buffer,
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::gemm<"+ToString(cols)+">");

            Y_buffer.SetZero();

            if( beta == static_cast<T>(0) )
            {
                std::memset( Y, 0, m * cols * sizeof(T_out) );
            }
            else
            {
                Scale( Y, beta, m * cols, job_ptr.Size()-1);
            }

            if( alpha != static_cast<T>(0) )
            {
//                    logprint("alpha != 0");
                // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                
                #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                {
                    T_out * restrict const Y_buf = Y_buffer[thread].data();
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I l_begin = rp[i  ];
                        const I l_end   = rp[i+1];
                        
                        // Add the others.
                        for( I l = l_begin; l < l_end; ++l )
                        {
                            const I j = ci[l];
                            
                            const T a_i_j = a[l];

                            const T_in  * restrict const X_i = X + cols * i;
                            
                                  T_out * restrict const Y_j = Y_buf + cols * j;
                            
                            for( I k = 0; k < cols; ++k )
                            {
                                Y_j[k] += a_i_j * static_cast<T>(X_i[k]);
                            }
                        }
                    }
                }
            }
        }
        
        
        template<int cols>
        void symm
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
                  Tensor3<T_out,I> & Y_buffer,
            const Tensor1<I,I> & job_ptr
        )
        {
//            ptic(ClassName()+"::gemm<"+ToString(cols)+">");

            Y_buffer.SetZero();

            if( alpha != static_cast<T>(0) )
            {
//                    logprint("alpha != 0");
                // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                
                #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
                {
                    T_out * restrict const Y_buf = Y_buffer.data(thread);
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I l_begin = rp[i  ];
                        const I l_end   = rp[i+1];
                        
                        // Add the others.
                        for( I l = l_begin; l < l_end; ++l )
                        {
                            const I j = ci[l];
                            
                            const T a_i_j = a[l];

                            const T_in  * restrict const X_i = X + cols * i;
                            const T_in  * restrict const X_j = X + cols * j;
                            
                                  T_out * restrict const Y_i = Y_buf + cols * i;
                                  T_out * restrict const Y_j = Y_buf + cols * j;
                            
                            for( I k = 0; k < cols; ++k )
                            {
//                                Y_i[k] += a_i_j * static_cast<T>(X_j[k]);
//                                Y_j[k] += a_i_j * static_cast<T>(X_i[k]);
                                Y_i[k] = std::fma(a_i_j, static_cast<T>(X_j[k]), Y_i[k]);
                                Y_j[k] = std::fma(a_i_j, static_cast<T>(X_i[k]), Y_i[k]);
                            }
                        }
                    }
                }
            }
            
//            #pragma omp parallel for simd num_threads( job_ptr.Size()-1 ) schedule( static )
//            for( I k = 0; k < m * cols; ++k )
//            {
//                Y[k] *= beta;
//
//                for( I thread = 0; thread < job_ptr.Size()-1; ++thread )
//                {
//                    Y[k] += Y_buffer[thread].data()[k];
//                }
//            }
            
            #pragma omp parallel for num_threads( job_ptr.Size()-1 ) schedule( static )
            for( I i = 0; i < m; ++i )
            {
                const I pos = cols*i;
//                T_out * restrict const T_i = Y + cols * i;

                for( I k = 0; k < cols; ++k )
                {
                    Y[pos+k] *= beta;
                }

                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const T_out * restrict const Y_buf = Y_buffer.data(thread,i);

                    for( I k = 0; k < cols; ++k )
                    {
                        Y[pos+k] += Y_buf[k];
                    }
                }
            }
        }

        
        
        
        
    }; // SparseBLAS
    
    
} // namespace Tensors


