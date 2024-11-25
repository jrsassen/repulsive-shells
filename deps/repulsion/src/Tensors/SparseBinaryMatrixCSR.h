#pragma once

#define CLASS SparseBinaryMatrixCSR

namespace Tensors
{
    
    template<typename I>
    inline void AccumulateAssemblyCounters( Tensor2<I,I> & counters )
    {
        ptic("AccumulateAssemblyCounters");
        
        const I thread_count = counters.Dimension(0);
        
        const I m = counters.Dimension(1);
        
        for( I i = 0; i < m; ++i )
        {
            if( i > 0 )
            {
                counters( 0, i ) += counters( thread_count-1, i-1 );
            }

            for( I thread = 1; thread < thread_count; ++thread )
            {
                counters( thread, i ) += counters( thread-1, i );
            }
        }
        
        ptoc("AccumulateAssemblyCounters");
    }
    
    template<typename I>
    inline Tensor2<I,I> AssemblyCounters(
        const I * const * const idx,
        const I * const * const jdx,
        const I * entry_counts,
        const I list_count,
        const I m,
        const int symmetrize = 0
    )
    {
        ptic("AssemblyCounters");
        
        Tensor2<I,I> counters (list_count, m, static_cast<I>(0));

        // https://en.wikipedia.org/wiki/Counting_sort
        // using parallel count sort to sort the cluster (i,j)-pairs according to i.
        // storing counters of each i-index in thread-interleaved format
        // TODO: Improve data layout (transpose counts).
        #pragma omp parallel for num_threads( list_count )
        for( I thread = 0; thread < list_count; ++thread )
        {
            
            const I * const thread_idx = idx[thread];
            const I * const thread_jdx = jdx[thread];
            
            const I entry_count = entry_counts[thread];
            
            I * restrict const c = counters.data(thread);
            
            for( I k = 0; k < entry_count; ++k )
            {
                const I i = thread_idx[k];
                const I j = thread_jdx[k];
                ++c[i];

                if( (symmetrize!=0) && (i != j) )
                {
                    ++c[j];
                }
            }
        }
                
        AccumulateAssemblyCounters(counters);
        
        ptoc("AssemblyCounters");
        
        return counters;
    }
    

    template<typename I>
    class CLASS
    {
        ASSERT_INT  (I);
        
    protected:
        
        Tensor1<I,I> outer;
        Tensor1<I,I> inner;
        
        I m;
        I n;
        
        I thread_count = 1;
        
        bool symmetric       = false;
        bool uppertriangular = false;
        bool lowertriangular = false;
        
        // diag_ptr[i] is the first nonzero element in row i such that inner[diag_ptr[i]] >= i
        mutable Tensor1<I,I> diag_ptr                  = Tensor1<I,I>();
        mutable Tensor1<I,I> job_ptr                   = Tensor1<I,I>();
        mutable Tensor1<I,I> upper_triangular_job_ptr  = Tensor1<I,I>();
        mutable Tensor1<I,I> lower_triangular_job_ptr  = Tensor1<I,I>();
        
    public:
        
        CLASS() : m(static_cast<I>(0)), n(static_cast<I>(0)) {}

        CLASS(
            const long long m_,
            const long long n_,
            const long long thread_count_
        )
        :   outer       ( Tensor1<I,I>(static_cast<I>(m_+1),static_cast<I>(0))  )
        ,   m           ( static_cast<I>(m_)                                    )
        ,   n           ( static_cast<I>(n_)                                    )
        ,   thread_count( static_cast<I>(thread_count_)                         )
        {
            Init();
        }
        
        CLASS(
            const long long m_,
            const long long n_,
            const long long nnz_,
            const long long thread_count_
        )
        :   outer       ( Tensor1<I,I>(static_cast<I>(m_+1),static_cast<I>(0))  )
        ,   inner       ( Tensor1<I,I>(static_cast<I>(nnz_) )                   )
        ,   m           ( static_cast<I>(m_)                                    )
        ,   n           ( static_cast<I>(n_)                                    )
        ,   thread_count( static_cast<I>(thread_count_)                         )
        {
            Init();
        }
        
        
        template<typename J0, typename J1>
        CLASS(
            const J0 * const outer_,
            const J1 * const inner_,
            const long long m_,
            const long long n_,
            const long long thread_count_
        )
        :   outer       ( ToTensor1<I,I>(outer_,static_cast<I>(m_+1))       )
        ,   inner       ( ToTensor1<I,I>(inner_,static_cast<I>(outer_[m_])) )
        ,   m           ( static_cast<I>(m_)                                )
        ,   n           ( static_cast<I>(n_)                                )
        ,   thread_count( static_cast<I>(thread_count_)                     )
        {
            Init();
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   outer       ( other.outer         )
        ,   inner       ( other.inner         )
        ,   m           ( other.m             )
        ,   n           ( other.n             )
        ,   thread_count( other.thread_count  )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.outer,        B.outer        );
            swap( A.inner,        B.inner        );
            swap( A.m,            B.m            );
            swap( A.n,            B.n            );
            swap( A.thread_count, B.thread_count );
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
        
        
//        CLASS(
//            const std::vector<std::vector<I>> & idx,
//            const std::vector<std::vector<I>> & jdx,
//            const I m_,
//            const I n_,
//            const I final_thread_count,
//            const bool compress   = true,
//            const int  symmetrize = 0
//        )
        
        CLASS(
            const std::vector<std::vector<I>> & idx,
            const std::vector<std::vector<I>> & jdx,
            const I m_,
            const I n_,
            const I final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        : CLASS ( m_, n_, static_cast<I>(idx.size()) )
        {
            I list_count = static_cast<I>(idx.size());
            Tensor1<const I*, I> i (list_count);
            Tensor1<const I*, I> j (list_count);

            Tensor1<I ,I> entry_counts (list_count);
            
            for( I thread = 0; thread < list_count; ++thread )
            {
                i[thread] = idx[thread].data();
                j[thread] = jdx[thread].data();
                entry_counts[thread] = static_cast<I>(idx[thread].size());
            }
            
//            FromPairs(idx,jdx,thread_count_,compress,symmetrize);
            
            FromPairs( i.data(), j.data(), entry_counts.data(),
                    list_count, final_thread_count, compress, symmetrize );
        }
    
        virtual ~CLASS() = default;
        

    public:
        
        I ThreadCount() const
        {
            return thread_count;
        }
        
        void SetThreadCount( const I thread_count_ )
        {
            thread_count = std::max( static_cast<I>(1), thread_count_);
        }
        
    protected:
        
        void Init()
        {
            outer[0] = static_cast<I>(0);
        }
        
        void FromPairs(
            const I * const * const idx,
            const I * const * const jdx,
            const I * entry_counts,
            const I list_count,
            const I final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        {
            ptic(ClassName()+"::FromPairs");
            
            if( symmetrize )
            {
                logprint(ClassName()+"::FromPairs symmetrize");
            }
            else
            {
                logprint(ClassName()+"::FromPairs no symmetrize");
            }
            
            if( compress )
            {
                logprint(ClassName()+"::FromPairs compress");
            }
            else
            {
                logprint(ClassName()+"::FromPairs no compress");
            }
            
//            Tensor2<I,I> counters = AssemblyCounters(idx,jdx,m,symmetrize);
            
            Tensor2<I,I> counters = AssemblyCounters(
                idx, jdx,entry_counts, list_count, m, symmetrize
            );
            
            const I nnz = counters(list_count-1,m-1);
            
            if( nnz > 0 )
            {
                inner = Tensor1<I,I>( nnz );
            
                I * restrict const outer__ = outer.data();
                I * restrict const inner__ = inner.data();

                memcpy( outer__+1, counters.data(list_count-1), m * sizeof(I) );

                // writing the j-indices into sep_column_indices
                // the counters array tells each thread where to write
                // since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                
                #pragma omp parallel for num_threads( list_count )
                for( I thread = 0; thread < list_count; ++thread )
                {
                    
                    const I entry_count = entry_counts[thread];
                    
                    const I * restrict const thread_idx = idx[thread];
                    const I * restrict const thread_jdx = jdx[thread];
                    
                          I * restrict const c = counters.data(thread);
                    
                    for( I k = entry_count - 1; k > -1; --k )
                    {
                        const I i = thread_idx[k];
                        const I j = thread_jdx[k];
                        {
                            const I pos  = --c[i];
                            inner__[pos] = j;
                        }
                        if( (symmetrize != 0) && (i != j) )
                        {
                            const I pos  = --c[j];
                            inner__[pos] = i;
                        }
                    }
                }

                // From here on, we may use as many threads as we want.
                SetThreadCount( final_thread_count );

                if( compress )
                {
                    Compress();
                }

                // We have to sort b_inner to be compatible with the CSR format.
                SortInner();
            }
            else
            {
                SetThreadCount( final_thread_count );
            }
            ptoc(ClassName()+"::FromPairs");
        }
        
        void RequireJobPtr() const
        {
            if( job_ptr.Size()-1 != thread_count )
            {
                ptic(ClassName()+"::RequireJobPtr");
                
                job_ptr = BalanceWorkLoad<I>( outer, thread_count, false );
                
                ptoc(ClassName()+"::RequireJobPtr");
            }
        }
        
        void RequireDiag() const
        {
            if( (diag_ptr.Size() != m) )
            {
                ptic(ClassName()+"::RequireDiag");
                
                if( outer.Last() <= 0 )
                {
                    diag_ptr = Tensor1<I,I>( outer.data()+1, m );
                }
                else
                {
                    RequireJobPtr();
                    
                    diag_ptr = Tensor1<I,I>( m );
                    
                          I * restrict const diag_ptr__ = diag_ptr.data();
                    const I * restrict const outer__    = outer.data();
                    const I * restrict const inner__    = inner.data();

                    #pragma omp parallel for num_threads( thread_count )
                    for( I thread = 0; thread < thread_count; ++thread )
                    {

                        const I i_begin = job_ptr[thread  ];
                        const I i_end   = job_ptr[thread+1];

                        for( I i = i_begin; i < i_end; ++ i )
                        {
                            const I k_begin = outer__[i  ];
                            const I k_end   = outer__[i+1];

                            I k = k_begin;

                            while( (k < k_end) && (inner__[k] < i)  )
                            {
                                ++k;
                            }
                            diag_ptr__[i] = k;
                        }
                    }
                }
                
                ptoc(ClassName()+"::RequireDiag");
            }
        }
        
        void RequireUpperTriangularJobPtr() const
        {
            if( (m > 0) && (upper_triangular_job_ptr.Size()-1 != thread_count) )
            {
                ptic(ClassName()+"::RequireUpperTriangularJobPtr");
                
                RequireDiag();
                
                Tensor1<I,I> costs = Tensor1<I,I>(m + 1);
                costs[0]=0;
                
                const I * restrict const diag_ptr__ = diag_ptr.data();
                const I * restrict const outer__    = outer.data();
                      I * restrict const costs__    = costs.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    for( I i = i_begin; i < i_end; ++ i )
                    {
                        costs__[i+1] = outer__[i+1] - diag_ptr__[i];
                    }
                }
                
                costs.Accumulate();
                
                upper_triangular_job_ptr = BalanceWorkLoad(costs, thread_count, false );
                
                ptoc(ClassName()+"::RequireUpperTriangularJobPtr");
            }
        }
        
        void RequireLowerTriangularJobPtr() const
        {
            if( (m > 0) && (lower_triangular_job_ptr.Size()-1 != thread_count) )
            {
                ptic(ClassName()+"::RequireLowerTriangularJobPtr");
                
                RequireDiag();
                                
                Tensor1<I,I> costs = Tensor1<I,I>(m + 1);
                costs[0]=0;
                
                const I * restrict const diag_ptr__ = diag_ptr.data();
                const I * restrict const outer__    = outer.data();
                      I * restrict const costs__    = costs.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    for( I i = i_begin; i < i_end; ++ i )
                    {
                        costs__[i+1] = diag_ptr__[i] - outer__[i];
                    }
                }
                
                costs.Accumulate();
                
                lower_triangular_job_ptr = BalanceWorkLoad(costs, thread_count, false );
                
                ptoc(ClassName()+"::RequireLowerTriangularJobPtr");
            }
        }
        
    public:

        I RowCount() const
        {
            return m;
        }
        
        I ColCount() const
        {
            return n;
        }
        
        I NonzeroCount() const
        {
            return inner.Size();
        }

        Tensor1<I,I> & Outer()
        {
            return outer;
        }
        
        const Tensor1<I,I> & Outer() const
        {
            return outer;
        }

        Tensor1<I,I> & Inner()
        {
            return inner;
        }
        
        const Tensor1<I,I> & Inner() const
        {
            return inner;
        }
        
        const Tensor1<I,I> & Diag() const
        {
            RequireDiag();
            
            return diag_ptr;
        }
        
        
        const Tensor1<I,I> & JobPtr() const
        {
            RequireJobPtr();
            
            return job_ptr;
        }
        
        
        
        const Tensor1<I,I> & UpperTriangularJobPtr() const
        {
            RequireUpperTriangularJobPtr();
            
            return upper_triangular_job_ptr;
        }
        
        
        const Tensor1<I,I> & LowerTriangularJobPtr() const
        {
            RequireLowerTriangularJobPtr();
            
            return lower_triangular_job_ptr;
        }
        
        
    protected:
        
        Tensor2<I,I> CreateTransposeCounters() const
        {
            ptic(ClassName()+"::CreateTransposeCounters");
            
            RequireJobPtr();
            
            Tensor2<I,I> counters ( thread_count, n, static_cast<I>(0) );
            
            if( WellFormed() )
            {
    //            ptic("Counting sort");
                // Use counting sort to sort outer indices of output matrix.
                // https://en.wikipedia.org/wiki/Counting_sort
    //            ptic("Counting");
                
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);
                    
                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
                                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = A_outer[i  ];
                        const I jj_end   = A_outer[i+1];
                        
                        for( I jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const I j = A_inner[jj];
                            ++c[ j ];
                        }
                    }
                }

                for( I i = 0; i < n; ++i )
                {
                    if( i > 0 )
                    {
                        counters( 0, i ) += counters( thread_count-1, i-1 );
                    }

                    for( I thread = 1; thread < thread_count ; ++thread )
                    {
                        counters( thread, i ) += counters( thread-1, i );
                    }
                }
            }
            
            ptoc(ClassName()+"::CreateTransposeCounters");
            
            return counters;
        }
        
    public:
        
        virtual void SortInner()
        {
            // Sorts the column indices of each matrix row.
            
            ptic(ClassName()+"::SortInner");
            
            if( WellFormed() )
            {
                RequireJobPtr();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I begin = outer[i  ];
                        const I end   = outer[i+1];
                        
                        std::sort( inner.data() + begin, inner.data() + end );
                    }
                }
            }
            
            ptoc(ClassName()+"::SortInner");
        }
        
    public:
        
        CLASS TransposeBinary() const
        {
            ptic(ClassName()+"::TransposeBinary");
            
            RequireJobPtr();
            
            Tensor2<I,I> counters = CreateTransposeCounters();
            
            CLASS<I> B ( n, m, outer[m], thread_count );

            memcpy( B.Outer().data() + 1, counters.data(thread_count-1), n * sizeof(I) );

            if( WellFormed() )
            {
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
      
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);

                          I * restrict const B_inner  = B.Inner().data();
    //                      T * restrict const B_values = B.Value().data();

                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
    //                const T * restrict const A_values = Value().data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I k_begin = A_outer[i  ];
                        const I k_end   = A_outer[i+1];
                        
                        for( I k = k_end-1; k > k_begin-1; --k )
                        {
                            const I j = A_inner[k];
                            const I pos = --c[ j ];
                            B_inner [pos] = i;
    //                        B_values[pos] = A_values[k];
                        }
                    }
                }
            }
            
            // Finished counting sort.

            // We only have to care about the correct ordering of inner indices and values.
            B.SortInner();

            ptoc(ClassName()+"::TransposeBinary");
            
            return B;
        }
        
        
        virtual void Compress()
        {
            ptic(ClassName()+"::Compress");
            
            if( WellFormed() )
            {
                RequireJobPtr();
                
                Tensor1<I,I> new_outer (outer.Size(),0);
                
                I * restrict const new_outer__ = new_outer.data();
                I * restrict const     outer__ = outer.data();
                I * restrict const     inner__ = inner.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    // To where we write.
                    I jj_new        = outer__[i_begin];
                    
                    // Memoize the next entry in outer because outer will be overwritten
                    I next_jj_begin = outer__[i_begin];
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = next_jj_begin;
                        const I jj_end   = outer__[i+1];
                        
                        // Memoize the next entry in outer because outer will be overwritten
                        next_jj_begin = jj_end;
                        
                        I row_nonzero_counter = static_cast<I>(0);
                        
                        // From where we read.
                        I jj = jj_begin;
                        
                        while( jj< jj_end )
                        {
                            I j = inner__[jj];
                            
                            {
                                if( jj > jj_new )
                                {
                                    inner__[jj] = static_cast<I>(0);
                                }
                                
                                ++jj;
                            }
            
                            while( (jj < jj_end) && (j == inner__[jj]) )
                            {
                                if( jj > jj_new )
                                {
                                    inner__[jj] = static_cast<I>(0);
                                }
                                ++jj;
                            }
                            
                            inner__[jj_new] = j;
                            
                            jj_new++;
                            row_nonzero_counter++;
                        }
                        new_outer__[i+1] = row_nonzero_counter;
                    }
                }
                
                // This is the new array of outer indices.
                new_outer.Accumulate();
                
                const I nnz = new_outer[m];
                
                Tensor1<I,I> new_inner (nnz,0);
                
                //TODO: Parallelization might be a bad idea here.
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    const I new_pos = new_outer__[i_begin];
                    const I     pos =     outer__[i_begin];

                    const I thread_nonzeroes = new_outer__[i_end] - new_outer__[i_begin];
                    
                    // Starting position of thread in inner list.
                    memcpy(
                        new_inner.data() + new_pos,
                            inner.data() +     pos,
                        thread_nonzeroes * sizeof(I)
                    );
                }
                
                swap( new_outer,  outer  );
                swap( new_inner,  inner  );
                
                job_ptr = Tensor1<I,I>();
            }
            
            ptoc(ClassName()+"::Compress");
        }
        
//##########################################################################################################
//####          Matrix Multiplication
//##########################################################################################################
        
        
        CLASS<I> DotBinary( const CLASS<I> & B ) const
        {
            ptic(ClassName()+"::DotBinary");
                        
            if( WellFormed() && B.WellFormed() )
            {
                RequireJobPtr();
                
                ptic("Create counters for counting sort");
                
                Tensor2<I,I> counters ( thread_count, m, static_cast<I>(0) );
                
                // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                // https://en.wikipedia.org/wiki/Counting_sort
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);
                    
                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
                    
                    const I * restrict const B_outer  = B.Outer().data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = A_outer[i  ];
                        const I jj_end   = A_outer[i+1];
                        
                        for( I jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const I j = A_inner[jj];
                            
                            c[i] += (B_outer[j+1] - B_outer[j]);
                        }
                    }
                }
                
                ptoc("Create counters for counting sort");
                
                AccumulateAssemblyCounters(counters);
                
    //            for( I i = 0; i < m; ++i )
    //            {
    //                if( i > 0 )
    //                {
    //                    counters( 0, i ) += counters( thread_count-1, i-1 );
    //                }
    //
    //                for( I thread = 1; thread < thread_count ; ++thread )
    //                {
    //                    counters( thread, i ) += counters( thread-1, i );
    //                }
    //            }
                
                const I nnz = counters.data(thread_count-1)[m-1];
                
                CLASS<I> C ( m, B.ColCount(), nnz, thread_count );
                
                memcpy( C.Outer().data() + 1, counters.data(thread_count-1), m * sizeof(I) );
                ptic("Counting sort");
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
      
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);

                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
    //                const T * restrict const A_values = Value().data();
                    
                    const I * restrict const B_outer  = B.Outer().data();
                    const I * restrict const B_inner  = B.Inner().data();
    //                const T * restrict const B_values = B.Value().data();
                    
                          I * restrict const C_inner  = C.Inner().data();
    //                      T * restrict const C_values = C.Value().data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = A_outer[i  ];
                        const I jj_end   = A_outer[i+1];
                        
                        for( I jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const I j = A_inner[jj];
                            
                            const I kk_begin = B_outer[j  ];
                            const I kk_end   = B_outer[j+1];
                            
                            for( I kk = kk_end-1; kk > kk_begin-1; --kk )
                            {
                                const I k = B_inner[kk];
                                const I pos = --c[ i ];
                                
                                C_inner [pos] = k;
    //                            C_values[pos] = A_values[jj] * B_values[kk];
                            }
                        }
                    }
                }
                // Finished expansion phase (counting sort).
                ptoc("Counting sort");
                
                // Now we have to care about the correct ordering of inner indices and values.
                C.SortInner();
                
                // Finally we compress duplicates in inner and values.
                C.Compress();
                
                ptoc(ClassName()+"::DotBinary");
                
                return C;
            }
            else
            {
                return CLASS<I> ();
            }
        }
        
        
        template<typename T, typename T_in, typename T_out>
        void Multiply_Vector
        (
            const T alpha,
            T     const * restrict const values,
            T_in  const * restrict const x,
            const T_out beta,
            T_out       * restrict const y
        ) const
        {
            SparseBLAS<T,I,T_in,T_out> spblas ( thread_count );
            
            spblas.Multiply_GeneralMatrix_Vector( alpha, outer, inner, values, m, n, x, beta, y, JobPtr() );
        }
        
        template<typename T, typename T_in, typename T_out>
        void Multiply_BinaryMatrix_Vector
        (
            const T alpha,
            T_in  const * restrict const x,
            const T_out beta,
            T_out       * restrict const y
        ) const
        {
            SparseBLAS<T,I,T_in,T_out> spblas ( thread_count );
            
            spblas.Multiply_BinaryMatrix_Vector( alpha, outer, inner, m, n, x, beta, y, JobPtr() );
        }
        
        template<typename T, typename T_in, typename T_out>
        void Multiply_BinaryMatrix_DenseMatrix(
            const T alpha,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const I cols = 1
        ) const
        {
            if( WellFormed() )
            {
                auto sblas = SparseBLAS<T,I,T_in,T_out>( thread_count );
                
                sblas.Multiply_BinaryMatrix_DenseMatrix(
                    alpha,outer.data(),inner.data(),m,n,X,beta,Y,cols,JobPtr());
                
            }
        }
        
        
        template<typename T, typename T_in, typename T_out>
        void Multiply_DenseMatrix(
            const T alpha,
            const T     * values,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const I cols = 1
        ) const
        {
            if( WellFormed() )
            {
                auto sblas = SparseBLAS<T,I,T_in,T_out>( thread_count );
                
                sblas.Multiply_GeneralMatrix_DenseMatrix(
                    alpha,outer.data(),inner.data(),values,m,n,X,beta,Y,cols,JobPtr());
            }
//            else
//            {
//                wprint(ClassName()+"::Multiply_DenseMatrix: No nonzeroes found. Doing nothing.");
//            }
        }
        
        
        template<typename T, typename T_in, typename T_out>
        void Multiply_Vector(
            const T alpha,
            const Tensor1<T,    I> & values,
            const Tensor1<T_in, I> & X,
            const T_out beta,
                  Tensor1<T_out,I> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Multiply_DenseMatrix( alpha, values.data(), X.data(), beta, Y.data(), 1 );
            }
            else
            {
                eprint(ClassName()+"::Multiply_Vector: shapes of matrix, input, and output do not match.");
            }
        }
        
        
        template<typename T, typename T_in, typename T_out>
        void Multiply_DenseMatrix(
            const T alpha,
            const Tensor1<T,    I> & values,
            const Tensor2<T_in, I> & X,
            const T beta,
                  Tensor2<T_out,I> & Y
        ) const
        {
            I cols = X.Dimension(1);
            if( cols == X.Dimension(1) && X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Multiply_DenseMatrix( alpha, values.data(), X.data(), beta, Y.data(), cols );
            }
            else
            {
                eprint(ClassName()+"::Multiply_DenseMatrix: shapes of matrix, input, and output do not match.");
            }
        }
        
        
        template<typename T, typename T_in, typename T_out>
        void Dot(
            const T     * values,
            const T_in  * X,
                  T_out * Y,
            const I cols,
            bool addTo = false
        ) const
        {
            Multiply_DenseMatrix( static_cast<T>(1), values, X, static_cast<T_out>(addTo), Y, cols );
        }

        
        template<typename T, typename T_in, typename T_out>
        void Dot(
            const Tensor1<T,    I> & values,
            const Tensor1<T_in, I> & X,
                  Tensor1<T_out,I> & Y,
            bool addTo = false
        ) const
        {
            Multiply_Vector( static_cast<T>(1), values, X, static_cast<T_out>(addTo), Y);
        }
        
        
        template<typename T, typename T_in, typename T_out>
        void Dot(
            const Tensor1<T,    I> & values,
            const Tensor2<T_in, I> & X,
                  Tensor2<T_out,I> & Y,
            bool addTo = false
        ) const
        {
            Multiply_DenseMatrix( static_cast<T>(1), values, X, static_cast<T_out>(addTo), Y);
        }
        
        
        template<typename T_in, typename T_out>
        void DotBinary(
            const T_in  * X,
                  T_out * Y,
            const I cols,
            bool addTo = false
        ) const
        {
            Multiply_BinaryMatrix_DenseMatrix( static_cast<T_out>(1), X, static_cast<T_out>(addTo), Y, cols );
        }
        

        template<typename T_in, typename T_out>
        void DotBinary(
            const Tensor1<T_in, I> & X,
                  Tensor1<T_out,I> & Y,
            bool addTo = false
        ) const
        {
            Multiply_BinaryMatrix_Vector( static_cast<T_out>(1), X, static_cast<T_out>(addTo), Y);
        }
        
        
        template<typename T_in, typename T_out>
        void DotBinary(
            const Tensor2<T_in, I> & X,
                  Tensor2<T_out,I> & Y,
            bool addTo = false
        ) const
        {
            Multiply_BinaryMatrix_DenseMatrix( static_cast<T_out>(1), X, static_cast<T_out>(addTo), Y);
        }
        

//##########################################################################################################
//####          Lookup Operations
//##########################################################################################################
        
        I FindNonzeroPosition( const I i, const I j ) const
        {
            // Looks up the entry {i,j}. If existent, its index within the list of nonzeroes is returned. Otherwise, a negative number is returned (-1 if simply not found and -2 if i is out of bounds).
            if( (0 <= i) && (i<m) )
            {
                const I * restrict const inner__ = inner.data();

                I L = outer[i  ];
                I R = outer[i+1]-1;
                while( L < R )
                {
                    const I k = R - (R-L)/static_cast<I>(2);
                    const I col = inner__[k];

                    if( col > j )
                    {
                        R = k-1;
                    }
                    else
                    {
                        L = k;
                    }
                }
                return (inner__[L]==j) ? L : static_cast<I>(-1);
            }
            else
            {
                wprint(ClassName()+"::FindNonzeroPosition: Row index i = "+ToString(i)+" is out of bounds {0,"+ToString(m)+"}.");
                return static_cast<I>(-2);
            }
        }
        
        
        template<typename S, typename T, typename J>
        void FillLowerTriangleFromUpperTriangle( std::map<S,Tensor1<T,J>> & values ) const
        {
            ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
            
            if( WellFormed() )
            {
                const I * restrict const diag__   = Diag().data();
                const I * restrict const outer__  = Outer().data();
                const I * restrict const inner__  = Inner().data();
                
                auto & job_ptr__ = LowerTriangularJobPtr();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    
                    const I i_begin = job_ptr__[thread];
                    const I i_end   = job_ptr__[thread+1];
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I k_begin = outer__[i];
                        const I k_end   =  diag__[i];
                        
                        for( I k = k_begin; k < k_end; ++k )
                        {
                            const I j = inner__[k];
                            
                            I L =  diag__[j];
                            I R = outer__[j+1]-1;
                            
                            while( L < R )
                            {
                                const I M   = R - (R-L)/static_cast<I>(2);
                                const I col = inner__[M];

                                if( col > i )
                                {
                                    R = M-1;
                                }
                                else
                                {
                                    L = M;
                                }
                            }
                            
                            for( auto & f : values )
                            {
                                f.second[k] = f.second[L];
                            }
                            
                        } // for( Int k = k_begin; k < k_end; ++k )

                    } // for( Int i = i_begin; i < i_end; ++i )
                    
                } // #pragma omp parallel
            }
            
            ptoc(ClassName()+"::FillLowerTriangleFromUpperTriangle");
        }
        
        bool WellFormed() const
        {
            return ( ( outer.Size() > 1 ) && ( outer.Last() > 0 ) );
        }
        
    public:
        
        virtual I Dimension( const bool dim )
        {
            return dim ? n : m;
        }
        
        std::string Stats() const
        {
            std::stringstream s;
            
            s
            << "\n==== "+ClassName()+" Stats ====" << "\n\n"
            << " RowCount()      = " << RowCount() << "\n"
            << " ColCount()      = " << ColCount() << "\n"
            << " NonzeroCount()  = " << NonzeroCount() << "\n"
            << " ThreadCount()   = " << ThreadCount() << "\n"
            << " Outer().Size()  = " << Outer().Size() << "\n"
            << " Inner().Size()  = " << Inner().Size() << "\n"
            << "\n==== "+ClassName()+" Stats ====\n" << std::endl;
            
            return s.str();
        }
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<I>::Get()+">";
        }
        
    }; // CLASS

    
} // namespace Tensors


#undef CLASS
