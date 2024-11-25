#pragma once

namespace Tensors {

#define TENSOR_T Tensor3

    template <typename T, typename I>
    class alignas( 2 * CACHE_LINE_WIDTH ) TENSOR_T
    {
        
#include "Tensor_Common.h"
        
    protected:
        
        I dims[3] = {};     // dimensions visible to user

    public:
        
        template< typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)>
        TENSOR_T( const J0 d0, const J1 d1, const J2 d2 )
        {
            dims[0] = static_cast<I>(d0);
            dims[1] = static_cast<I>(d1);
            dims[2] = static_cast<I>(d2);
            
            allocate();
        }
        
        template<typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2) >
        TENSOR_T( const J0 d0, const J1 d1, const J2 d2, const T init )
        {
            dims[0] = static_cast<I>(d0);
            dims[1] = static_cast<I>(d1);
            dims[2] = static_cast<I>(d2);
            
            allocate();
            
            Fill( static_cast<T>(init) );
        }
        
        template<typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)
        >
        TENSOR_T( const T * a_, const J0 d0, const J1 d1, const J2 d2 )
        {
            dims[0] = static_cast<I>(d0);
            dims[1] = static_cast<I>(d1);
            dims[2] = static_cast<I>(d2);

            allocate();

            Read(a_);
        }
        
        // Copy constructor
        template<typename S, typename J, IsInt(J)>
        TENSOR_T( const Tensor3<S,J> & B )
        {
            print(ClassName()+" copy constructor");
            dims[0] = static_cast<I>(B.Dimension(0));
            dims[1] = static_cast<I>(B.Dimension(1));
            dims[2] = static_cast<I>(B.Dimension(2));
            
            allocate();
            
            Read(B.a);
        }
        
        // Copy constructor
        TENSOR_T( const TENSOR_T & B )
        {
            print(ClassName()+" copy constructor");
            dims[0] = B.Dimension(0);
            dims[1] = B.Dimension(1);
            dims[2] = B.Dimension(2);
            
            allocate();
            
            Read(B.a);
        }
        
        friend void swap(TENSOR_T &A, TENSOR_T &B) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
#ifdef BOUND_CHECKS
            print(ClassName()+" swap");
#endif
            swap(A.dims[0], B.dims[0]);
            swap(A.dims[1], B.dims[1]);
            swap(A.dims[2], B.dims[2]);
            
            swap(A.size__, B.size__);
            swap(A.a, B.a);
        }
        
        static constexpr I Rank()
        {
            return static_cast<I>(3);
        }


        T * data( const I i )
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a +  i * Dimension(1) * Dimension(2);
        }
        
        const T * data( const I i ) const
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
#endif
            return a +  i * Dimension(1) * Dimension(2);
        }

        T * data( const I i, const I j )
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
            if( j > Dimension(1) )
            {
                eprint(ClassName()+"::data(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)-1) +" }.");
            }
#endif
            return a + ( i * Dimension(1) + j ) * Dimension(2);
        }
        
        const T * data( const I i, const I j ) const
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
            if( j > Dimension(1) )
            {
                eprint(ClassName()+"::data(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)-1) +" }.");
            }
#endif
            return a + ( i * Dimension(1) + j ) * Dimension(2);
        }

        T * data( const I i, const I j, const I k)
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i,j,k): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
            if( j > Dimension(1) )
            {
                eprint(ClassName()+"::data(i,j,k): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)-1) +" }.");
            }
            if( k > Dimension(2) )
            {
                eprint(ClassName()+"::data(i,j,k): third index " + std::to_string(k) + " is out of bounds { 0, " + std::to_string(Dimension(2)-1) +" }.");
            }
#endif
            return a + ( i *  Dimension(1) + j ) * Dimension(2) + k;
        }
        
        const T * data( const I i, const I j, const I k) const
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i,j,k): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
            if( j > Dimension(1) )
            {
                eprint(ClassName()+"::data(i,j,k): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)-1) +" }.");
            }
            if( k > Dimension(2) )
            {
                eprint(ClassName()+"::data(i,j,k): third index " + std::to_string(k) + " is out of bounds { 0, " + std::to_string(Dimension(2)-1) +" }.");
            }
#endif
            return a + ( i *  Dimension(1) + j ) * Dimension(2) + k;
        }
        
        T & operator()( const I i, const I j, const I k) const
        {
#ifdef BOUND_CHECKS
            if( i >= Dimension(0) )
            {
                eprint(ClassName()+"::operator()(i,j,k): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
            if( j >= Dimension(1) )
            {
                eprint(ClassName()+"::operator()(i,j,k): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)-1) +" }.");
            }
            if( k >= Dimension(2) )
            {
                eprint(ClassName()+"::operator()(i,j,k): third index " + std::to_string(k) + " is out of bounds { 0, " + std::to_string(Dimension(2)-1) +" }.");
            }
#endif
            return a[ ( i *  Dimension(1) + j ) * Dimension(2) + k ];
        }
        
        template< typename S>
        void Write( const I i, S * const b ) const
        {
//            memcpy( b, data(i), dims[1] * dims[2] * sizeof(T) );
//            std::copy( data(i), data(i) + dims[1] * dims[2], b );
            std::transform( data(i), data(i+1), b, static_caster<T,S>() );
        }
        
        template< typename S>
        void Write( const I i, const I j, T * const b ) const
        {
//            memcpy( b, data(i,j), dims[2] * sizeof(T) );
//            std::copy( data(i,j), data(i,j) + dims[2], b );
            std::transform( data(i,j), data(i,j+1), b, static_caster<T,S>() );
        }
        
        template< typename S>
        void Read( const I i, const S * const b )
        {
//            memcpy( data(i), b, dims[1] * dims[2] * sizeof(T) );
//            std::copy( b, b + dims[1] * dims[2], data(i) );
            std::transform( b, b + dims[1] * dims[2], data(i), static_caster<S,T>() );
        }
        
        template< typename S>
        void Read( const I i, const I j, const S * const b )
        {
//            memcpy( data(i,j), b, dims[2] * sizeof(T) );
//            std::copy( b, b + dims[2], data(i,j) );
            std::copy( b, b + dims[2], data(i,j), static_caster<S,T>() );
        }
        
        private :
        
        void calculate_sizes()
        {
            size__ = Dimension(0) * Dimension(1) * Dimension(2);
            
#ifdef BOUND_CHECKS
            print(ClassName()+" { " +
                  std::to_string(Dimension(0)) + ", " +
                  std::to_string(Dimension(1)) + ", " +
                  std::to_string(Dimension(2)) + " }" );
#endif
        }
        
    public:
        
        static std::string ClassName()
        {
            return "Tensor3<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
    }; // Tensor3
    
    
    template<typename T, typename I, typename S, typename J0, typename J1, typename J2, IsInt(I), IsInt(J0), IsInt(J1), IsInt(J2)
    >
    Tensor3<T,I> ToTensor3( const S * a_, const J0 d0, const J1 d1, const J2 d2 )
    {
        Tensor3<T,I> result (static_cast<I>(d0), static_cast<I>(d1), static_cast<I>(d2));

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename T, typename I, IsFloat(T)>
    mma::TensorRef<mreal> to_MTensorRef( const Tensor3<T,I> & A )
    {
        auto B = mma::makeCube<mreal>( A.Dimension(0), A.Dimension(1), A.Dimension(2) );
        
        A.Write(B.data());

        return B;
    }
    
    template<typename J, typename I, IsInt(J)>
    mma::TensorRef<mint> to_MTensorRef( const Tensor3<J,I> & A )
    {
        auto B = mma::makeCube<mint>( A.Dimension(0), A.Dimension(1), A.Dimension(2) );
        
        A.Write(B.data());

        return B;
    }
    
    
    template<typename T, typename I>
    Tensor3<T,I> from_CubeRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor3<T,I>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
    template<typename T, typename I>
    Tensor3<T,I> from_CubeRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor3<T,I>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
#endif

#include "Tensor_Common_External.h"
    
#undef TENSOR_T
} // namespace Tensors
