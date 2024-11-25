// faster version of RequireDiag

#pragma once

#define CLASS SparseMatrixBSR
#define BASE  SparseBinaryMatrixCSR<I>

namespace Tensors
{

    template<typename T, typename I, mint BLK_ROW, mint BLK_COL>
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
        
        mutable Tensor1<T,I> values;

    public:
        
        CLASS() : BASE() {}
        
        CLASS(
            const long long m_,
            const long long n_,
            const long long thread_count_
        )
        :   BASE(m_,n_,thread_count) {}
        
        CLASS(
            const long long m_,
            const long long n_,
            const long long nonzero_block_count_,
            const long long thread_count_
        )
        :   BASE    ( m_, n_, nonzero_block_count_, thread_count_ )
        ,   values  ( Tensor1<T,I>(static_cast<I>(nonzero_block_count_ * BLK_ROW * BLK_COL)) )
        {}
        
        template<typename S, typename J0, typename J1>
        CLASS(
            const J0 * const outer_,
            const J1 * const inner_,
            const S  * const values_,
            const long long m_,
            const long long n_,
                  long long thread_count_
        )
        :   BASE    (outer_, inner_, m_, n_, thread_count_)
        ,   values  ( ToTensor1<T,I>(values_,outer_[static_cast<I>(m_)] * static_cast<I>(BLK_ROW * BLK_COL)) )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE         ( other        )
        ,   values       ( other.values )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( static_cast<BASE&>(A), static_cast<BASE&>(B) );
            swap( A.values,              B.values              );
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
                     
        
        static constexpr I RowsPerBlock()
        {
            return BLK_ROW;
        }
        
        static constexpr I ColsPerBlock()
        {
            return BLK_COL;
        }
        
        virtual I Dimension( const bool dim ) override
        {
            return dim ? (BLK_COL * n) : (BLK_ROW * m);
        }
        
        I BlockRowCount()
        {
            return m;
        }
        
        I BlockColCount()
        {
            return n;
        }
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<T>::Get()+","+TypeName<I>::Get()+","+ToString(BLK_ROW)+","+ToString(BLK_COL)+">";
        }
        
    }; // CLASS
    
    

    
} // namespace Tensors

#undef BASE
#undef CLASS
