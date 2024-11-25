// faster version of RequireDiag

#pragma once

#define CLASS SparseBinaryMatrixVBSR
#define BASE  SparseBinaryMatrixCSR<I>

namespace Tensors
{

    template<typename I>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::m;
        using BASE::n;
        using BASE::outer;
        using BASE::inner;
        using BASE::job_ptr;
        using BASE::diag_ptr;
        using BASE::thread_count;
        
        BASE blk_pat;
        
        Tensor1<I,I> blk_row_ptr;
        Tensor1<I,I> blk_col_ptr;
        
        mutable Tensor1<I,I> blk_ptr;
        // blk_ptr[k] is the index of the first nonzero entry of the
        
        // work load distribution among block rows for cycles over blocks
        mutable Tensor1<I,I>         blk_job_ptr;
        mutable Tensor1<I,I> upp_tri_blk_job_ptr;
        mutable Tensor1<I,I> low_tri_blk_job_ptr;
        
        mutable I max_row_count;

    public:
        
        CLASS() : BASE() {}
        
        CLASS(
          const std::vector<std::vector<I>> & idx,
          const std::vector<std::vector<I>> & jdx,
          const Tensor1<I,I> & blk_row_ptr_,
          const Tensor1<I,I> & blk_col_ptr_,
          const I thread_count_,
          const bool compress   = true,
          const int  symmetrize = 0
        )
        :   BASE( blk_row_ptr_[blk_row_ptr_.Size()-1], blk_col_ptr_[blk_col_ptr_.Size()-1], thread_count_ )
        ,   blk_pat         (
                              BASE(
                                   idx, jdx,
                                   blk_row_ptr_.Size()-1, blk_row_ptr_.Size()-1,
                                   thread_count_, compress, symmetrize
                              )
                            )
        ,   blk_row_ptr(blk_row_ptr_)
        ,   blk_col_ptr(blk_col_ptr_)
        {
            FullSparsityPattern();
        }
        
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE                ( other                     )
        ,   blk_pat             ( other.blk_pat             )
        ,   blk_row_ptr         ( other.blk_row_ptr         )
        ,   blk_col_ptr         ( other.blk_col_ptr         )
        ,   blk_ptr             ( other.blk_ptr             )
        ,   blk_job_ptr         ( other.blk_job_ptr         )
        ,   upp_tri_blk_job_ptr ( other.upp_tri_blk_job_ptr )
        ,   low_tri_blk_job_ptr ( other.low_tri_blk_job_ptr )
        ,   max_row_count       ( other.max_row_count       )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( static_cast<BASE&>(A), static_cast<BASE&>(B) );

            swap( A.blk_pat,                B.blk_pat               );
            swap( A.blk_row_ptr,            B.blk_row_ptr           );
            swap( A.blk_col_ptr,            B.blk_col_ptr           );
            swap( A.blk_ptr,                B.blk_ptr               );
            swap( A.blk_job_ptr,            B.blk_job_ptr           );
            swap( A.upp_tri_blk_job_ptr,    B.upp_tri_blk_job_ptr   );
            swap( A.low_tri_blk_job_ptr,    B.low_tri_blk_job_ptr   );
            swap( A.max_row_count,          B.max_row_count         );
        }
        
        // Copy assignment operator
        CLASS & operator=(CLASS other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept : CLASS()
        {
            swap(*this, other);
        }
    
        virtual ~CLASS() override = default;
        
        
    public:
        
        const BASE & BlockPattern() const
        {
            return blk_pat;
        }
        
        I BlockRowCount() const
        {
            return BlockPattern().RowCount();
        }
        
        I BlockColCount() const
        {
            return BlockPattern().ColCount();
        }
        
        I BlockNonzeroCount() const
        {
            return BlockPattern().NonzeroCount();
        }
        
        const Tensor1<I,I> & BlockRowPtr() const
        {
            return blk_row_ptr;
        }
        
        const Tensor1<I,I> & BlockColPtr() const
        {
            return blk_col_ptr;
        }
        
        const Tensor1<I,I> & BlockJobPtr() const
        {
            return blk_job_ptr;
        }
        
        const Tensor1<I,I> & BlockOuter() const
        {
            return BlockPattern().Outer();
        }
        
        const Tensor1<I,I> & BlockInner() const
        {
            return BlockPattern().Inner();
        }
        
        const Tensor1<I,I> & UpperTriangularBlockJobPtr() const
        {
            return upp_tri_blk_job_ptr;
        }
        
        const Tensor1<I,I> & LowerTriangularBlockJobPtr() const
        {
            return low_tri_blk_job_ptr;
        }

        
    protected:
        
        void FullSparsityPattern()
        {
            ptic(ClassName()+"::FullSparsityPattern");
            
            const I b_m = blk_pat.RowCount();
            
            outer[0] = static_cast<I>(0);
            
            Tensor1<I,I>         blk_row_ctr (b_m);
            // blk_row_ctr[b_i] for block row b_i is the number of nonzero elements
            // (which is constant among the rows contained in the block row_.
            
            Tensor1<I,I> upp_tri_blk_row_ctr (b_m);
            // upp_tri_blk_row_ctr[b_i] for block row b_i is the number of nonzero elements on or
            // above the diagonal; it is constant among the rows contained in the block row_.


            if( blk_pat.NonzeroCount() <= 0 )
            {
                ptoc(ClassName()+"::FullSparsityPattern");
                return;
            }
            
            const I * restrict const         b_row_ptr =          blk_row_ptr.data();
            const I * restrict const         b_col_ptr =          blk_col_ptr.data();
            const I * restrict const           b_outer =      blk_pat.Outer().data();
            const I * restrict const           b_inner =      blk_pat.Inner().data();
                  I * restrict const           outer__ =                outer.data();
                  I * restrict const         b_row_ctr =          blk_row_ctr.data();
                  I * restrict const upp_tri_b_row_ctr =  upp_tri_blk_row_ctr.data();
            
            auto & b_job_ptr = blk_pat.JobPtr();
            
            const I thread_count = b_job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                
                const I b_i_begin = b_job_ptr[thread  ];
                const I b_i_end   = b_job_ptr[thread+1];
                
                for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
                {
                    I         b_row_counter = static_cast<I>(0);
                    I upp_tri_b_row_counter = static_cast<I>(0);
                    
                    // Counting the number of entries per row in each block row to prepare outer.
                    {
                        const I b_k_begin = b_outer[b_i  ];
                        const I b_k_end   = b_outer[b_i+1];
                                    
                        for( I b_k = b_k_begin; b_k < b_k_end; ++b_k )
                        {
                            const I b_j = b_inner[b_k];

                            const I b_n_j = b_col_ptr[b_j+1] - b_col_ptr[b_j];

                            b_row_counter += b_n_j;
                            
                            if( b_i <= b_j )
                            {
                                upp_tri_b_row_counter += b_n_j;
                            }
                        }
                    }

                            b_row_ctr[b_i] =         b_row_counter;
                    upp_tri_b_row_ctr[b_i] = upp_tri_b_row_counter;
                    
                    {
                        const I b_k_begin = b_row_ptr[b_i];
                        const I b_k_end   = b_row_ptr[b_i+1];
                        // Each row in a block row has the same number of entries.
                        #pragma omp simd
                        for( I b_k = b_k_begin; b_k < b_k_end; ++b_k )
                        {
                            outer__[b_k+1] = b_row_counter;
                        }
                    }
                }
            }

            outer.Accumulate();

            const I nnz = outer.Last();

            if( nnz > 0 )
            {
                inner = Tensor1<I,I>( nnz );
                I * restrict const inner__ = inner.data();
                
                // Computing inner for CSR format.
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I b_i_begin = b_job_ptr[thread  ];
                    const I b_i_end   = b_job_ptr[thread+1];
                    
                    for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
                    {
                        // for each row i, write the column indices consecutively
                        
                        const I i_begin = b_row_ptr[b_i  ];
                        const I i_end   = b_row_ptr[b_i+1];
                        
                        const I k_begin = b_outer[b_i  ];
                        const I k_end   = b_outer[b_i+1];
                        
                        for( I i = i_begin; i < i_end; ++i ) // looping over all rows i in block row b_i
                        {
                            I ptr = outer__[i];                           // get first nonzero position in row i; ptr will be used to keep track of the current position within inner
                            
                            for( I k = k_begin; k < k_end; ++k )          // loop over all blocks in block row b_i
                            {
                                const I b_j = b_inner[k];               // we are in block {b_i, b_j}
                                
                                const I j_begin = b_col_ptr[b_j  ];
                                const I j_end   = b_col_ptr[b_j+1];
                                
                                for( I j = j_begin; j < j_end; ++j )      // write the column indices for row i
                                {
        //                                if( has_diagonal && (i == j) )
        //                                {
        //                                    diag_ptr__[i] = ptr;
        //                                }
                                    inner__[ptr] = j;
                                    ptr++;
                                }
                            }
                        }
                    }
                }
                
                blk_ptr = Tensor1<I,I>(blk_pat.NonzeroCount()+1);
                I * restrict const b_ptr = blk_ptr.data();
                blk_ptr[0] = 0;

                auto num_bef_b_row = Tensor1<I,I>(b_m+1);
                I * restrict const num_bef_b_row__ = num_bef_b_row.data();
                
                #pragma omp parallel for num_threads(thread_count)
                for( I b_i = 0; b_i < b_m; ++b_i )
                {
                    const I m_i = b_row_ptr[b_i+1] - b_row_ptr[b_i];
                    
                    num_bef_b_row__[b_i+1] = m_i * b_row_ctr[b_i];
                }

                num_bef_b_row.Accumulate();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I b_i_begin = b_job_ptr[thread  ];
                    const I b_i_end   = b_job_ptr[thread+1];
                    
                    for( I b_i = b_i_begin; b_i < b_i_end; ++b_i )
                    {
                        I num_bef_kth_b = num_bef_b_row__[b_i];
                        
                        const I m_b_i = b_row_ptr[b_i+1] - b_row_ptr[b_i];
                        
                        const I b_k_begin = b_outer[b_i  ];
                        const I b_k_end   = b_outer[b_i+1];
                        
                        for( I b_k = b_k_begin; b_k < b_k_end; ++b_k )
                        {
                            // b_k-th block is {b_i, b_j}
                            I   b_j = b_inner[b_k];
                            I n_b_j = b_col_ptr[b_j+1] - b_col_ptr[b_j];
                            // k-th block has size mi * nj
                            num_bef_kth_b += m_b_i * n_b_j;
                            b_ptr[b_k+1] = num_bef_kth_b;
                        }
                    }
                }

                // distribute workload
                Tensor1<I,I>         blk_costs (b_m+1);
                Tensor1<I,I> upp_tri_blk_costs (b_m+1);
                Tensor1<I,I> low_tri_blk_costs (b_m+1);
                
                        blk_costs[0] = static_cast<I>(0);
                upp_tri_blk_costs[0] = static_cast<I>(0);
                low_tri_blk_costs[0] = static_cast<I>(0);
                
                I * restrict const         b_costs =         blk_costs.data();
                I * restrict const upp_tri_b_costs = upp_tri_blk_costs.data();
                I * restrict const low_tri_b_costs = low_tri_blk_costs.data();
                
                #pragma omp parallel for num_threads(thread_count)
                for( I b_i = 0; b_i < b_m; ++b_i )
                {
                    const I rows = b_row_ptr[b_i+1] - b_row_ptr[b_i];
                    
                    const I cols     =         b_row_ctr[b_i];
                    const I upp_cols = upp_tri_b_row_ctr[b_i];
                    
                            b_costs[b_i+1] = rows * cols;
                    
                    upp_tri_b_costs[b_i+1] = rows * upp_cols;
                    
                    low_tri_b_costs[b_i+1] = rows * (cols-upp_cols);
                    
                }
                
                        blk_costs.Accumulate();
                upp_tri_blk_costs.Accumulate();
                low_tri_blk_costs.Accumulate();
                
                        blk_job_ptr = BalanceWorkLoad<I>(         blk_costs, thread_count, false );
                upp_tri_blk_job_ptr = BalanceWorkLoad<I>( upp_tri_blk_costs, thread_count, false );
                low_tri_blk_job_ptr = BalanceWorkLoad<I>( low_tri_blk_costs, thread_count, false );
                
                max_row_count = *std::max_element( b_row_ctr, b_row_ctr + b_m );
            }
            
            ptoc(ClassName()+"::FullSparsityPattern");
        }
    
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<I>::Get()+">";
        }
        
    }; // CLASS
    
    

    
} // namespace Tensors

#undef BASE
#undef CLASS
