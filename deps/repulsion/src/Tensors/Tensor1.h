#pragma once

namespace Tensors {

#define TENSOR_T Tensor1

    template <typename T, typename I>
    class alignas( 2 * CACHE_LINE_WIDTH ) Tensor1 // Use this broad alignment to prevent false sharing.
    {

#include "Tensor_Common.h"
        
    protected:
        
        I dims[1] = {};     // dimensions visible to user
        
    public:
        
        template<typename J, IsInt(J)>
        explicit TENSOR_T( const J d0 )
        {
            dims[0] = static_cast<I>(d0);
            
            allocate();
        }
        
        template<typename S, typename J, IsInt(J)>
        TENSOR_T( const J d0, const S init )
        {
            dims[0] = static_cast<I>(d0);
            
            allocate();
            
            Fill( static_cast<T>(init) );
        }
        
        template<typename S, typename J, IsInt(J)>
        TENSOR_T( const S * a_, const J d0 )
        {
            dims[0] = static_cast<I>(d0);

            allocate();

            Read(a_);
        }

        // Copy constructor
        TENSOR_T( const TENSOR_T & B )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" copy constructor");
#endif
            dims[0] = B.Dimension(0);
            
            allocate();
            
            Read(B.a);
        }
        
        // Copy constructor
        template<typename S, typename J, IsInt(J)>
        explicit TENSOR_T( const TENSOR_T<S,J> & B )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" copy constructor");
#endif
            dims[0] = static_cast<I>(B.Dimension(0));
            
            allocate();
            
            Read(B.a);
        }
        
        inline friend void swap(TENSOR_T &A, TENSOR_T &B) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
#ifdef BOUND_CHECKS
            print(ClassName()+" swap");
#endif
            swap(A.dims[0], B.dims[0]);
            
            swap(A.size__, B.size__);
            swap(A.a, B.a);
        }
        
    public:
        
        static constexpr I Rank()
        {
            return static_cast<I>(1);
        }
        

        T * data( const I i )
        {
#ifdef BOUND_CHECKS
            if( i > (Dimension(0)) )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a + i;
        }
        
        const T * data( const I i ) const
        {
#ifdef BOUND_CHECKS
            if( i > (Dimension(0)) )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a + i;
        }
        
        const T & operator()( const I i) const
        {
#ifdef BOUND_CHECKS
            if( i >= (Dimension(0)) )
            {
                eprint(ClassName()+"::operator()(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a[ i ];
        }
        
        T & operator()( const I i)
        {
#ifdef BOUND_CHECKS
            if( i >= (Dimension(0)) )
            {
                eprint(ClassName()+"::operator()(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a[ i ];
        }
        
        const T & operator[]( const I i) const
        {
#ifdef BOUND_CHECKS
            if( i >= (Dimension(0)) )
            {
                eprint(ClassName()+"::operator[](i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a[ i ];
        }
        
        T & operator[]( const I i)
        {
#ifdef BOUND_CHECKS
            if( i >= (Dimension(0)) )
            {
                eprint(ClassName()+"::operator[](i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a[ i ];
        }
        
        T First() const
        {
            return a[0];
        }

        T Last() const
        {
            return a[size__-1];
        }
        
        void Accumulate()
        {
            I n = dims[0];
            
            for( I i = 1 ; i < n; ++i )
            {
                a[i] += a[i-1];
            }
        }
        
        T Dot( const Tensor1<T,I> & y )
        {
            T sum = static_cast<T>(0);
            
            const T * restrict const x__ =   data();
            const T * restrict const y__ = y.data();
            
            if( Size() != y.Size() )
            {
                eprint(ClassName()+"::Dot: Sizes of vectors differ. Doing nothing.");
                return sum;
            }
            const I n = std::min( Size(), y.Size() );

            #pragma omp simd aligned( x__, y__ : ALIGN ) reduction( + : sum )
            for( I i = 0; i < n; ++ i)
            {
                sum += x__[i] * y__[i];
            }

            return sum;
        }
        
        void iota()
        {
            T * restrict const a__ = a;
            
            const T i_begin = static_cast<T>(0);
            const T i_end   = static_cast<T>(size__);
            
            #pragma omp simd aligned( a__ : ALIGN )
            for( T i = i_begin; i < i_end; ++i )
            {
                a__[ i ] = i;
            }
        }
        
    private:
        
        void calculate_sizes()
        {
            size__ = Dimension(0);
            
#ifdef BOUND_CHECKS
            print(ClassName()+" { " + std::to_string(Dimension(0)) + " }");
#endif
            
        }

    public:
        
        inline friend std::ostream & operator<<( std::ostream & s, const TENSOR_T & tensor )
        {
            s << tensor.ToString();
            return s;
        }
        
        std::string ToString( const I n = 16) const
        {
            std::stringstream sout;
            sout.precision(n);
            sout << "{ ";
            if( Size() > 0 )
            {
                sout << a[0];
            }
            for( I i = 1; i<4; ++i )
            {
                sout << ", " << a[i];
            }
            sout << " }";
            return sout.str();
        }
        
        std::string ToString( const I i_begin, const I i_end, const I n = 16) const
        {
            std::stringstream sout;
            sout.precision(n);
            sout << "{ ";
            if( Size() >= i_end )
            {
                sout << a[i_begin];
            }
            for( I i = i_begin + 1; i < i_end; ++i )
            {
                sout << ", " << a[i];
            }
            sout << " }";
            return sout.str();
        }
        
        static std::string ClassName()
        {
            return "Tensor1<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
        
    }; // Tensor1
    
    template<typename T, typename I>
    Tensor1<T,I> iota( const I size_ )
    {
        auto v = Tensor1<T,I>(size_);
        
        v.iota();
        
        return v;
    }
    
    template<typename T, typename I, typename S, typename J, IsInt(I), IsInt(J)>
    Tensor1<T,I> ToTensor1( const S * a_, const J d0 )
    {
        Tensor1<T,I> result (static_cast<I>(d0));

        result.Read(a_);
        
        return result;
    }

#ifdef LTEMPLATE_H

    
    template<typename T, typename I, IsFloat(T)>
    mma::TensorRef<mreal> to_MTensorRef( const Tensor1<T,I> & A )
    {
        auto B = mma::makeVector<mreal>( A.Dimension(0) );

        A.Write(B.data());

        return B;
    }
    
    template<typename J, typename I, IsInt(J)>
    mma::TensorRef<mint> to_MTensorRef( const Tensor1<J,I> & A )
    {
        auto B = mma::makeVector<mint>( A.Dimension(0) );
        
        A.Write(B.data());

        return B;
    }
    
    
    template<typename T, typename I>
    Tensor1<T,I> from_VectorRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor1<T,I>( A.data(), A.dimensions()[0] );
    }
    
    template<typename T, typename I>
    Tensor1<T,I> from_VectorRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor1<T,I>( A.data(), A.dimensions()[0] );
    }

    
#endif
  template <typename T, typename I>
  std::ostream & operator<<( std::ostream & s, const Tensor1<T,I> & x )
  {
    s << "{ ";
    if( static_cast<I>(x.Size()) >= 1 )
    {
      s << x[0];
    }
    for( I i = 1; i < static_cast<I>(x.Size()); ++i )
    {
      s << ", " << x[i];
    }
    s << " }";
    return s;
  }
#include "Tensor_Common_External.h"
    
#undef TENSOR_T
} // namespace Tensors
