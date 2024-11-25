#pragma once

namespace Tensors {

#define TENSOR_T Tensor2

    template <typename T, typename I>
    class alignas( 2 * CACHE_LINE_WIDTH ) Tensor2
    {
        
#include "Tensor_Common.h"
        
    protected:
        
        I dims[2] = {};     // dimensions visible to user
        
    public:
        
        template<typename J0, typename J1, IsInt(J0), IsInt(J1)>
        TENSOR_T( const J0 d0, const J1 d1)
        {
            dims[0] = static_cast<I>(d0);
            dims[1] = static_cast<I>(d1);
            
            allocate();
        }
        
        template<typename S, typename J0, typename J1, IsInt(J0), IsInt(J1)>
        TENSOR_T( const J0 d0, const J1 d1, const S init )
        {
            dims[0] = static_cast<I>(d0);
            dims[1] = static_cast<I>(d1);
            
            allocate();
            
            Fill( static_cast<T>(init) );
        }
        
        template<typename S, typename J0, typename J1, IsFloat(S), IsInt(J0), IsInt(J1) >
        TENSOR_T( const S * a_, const J0 d0, const J1 d1 )
        {
            dims[0] = static_cast<I>(d0);
            dims[1] = static_cast<I>(d1);

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
            dims[1] = B.Dimension(1);
            
            allocate();
            
            Read(B.a);
        }
        
        // Copy constructor
        template<typename S, typename J, IsFloat(S),  IsInt(J) >
        explicit TENSOR_T( const Tensor2<S,J> & B )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" copy constructor");
#endif
            dims[0] = static_cast<I>(B.Dimension(0));
            dims[1] = static_cast<I>(B.Dimension(1));
            
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
            
            swap(A.size__, B.size__);
            swap(A.a, B.a);
        }
        
    public:
        
        static constexpr I Rank()
        {
            return static_cast<I>(2);
        }
        
        template< typename S>
        void WriteTransposed( S * const b )
        {
            const I d_0 = dims[0];
            const I d_1 = dims[1];
            
                  S * restrict b__ = b;
            const T * restrict a__ = a;
            
            for( I j = 0; j < d_1; ++j )
            {
                for( I i = 0; i < d_0; ++i )
                {
                    b__[ d_0 * j + i ] = static_cast<S>(a__[d_1 * i + j ]);
                }
            }
        }
        
        template< typename S>
        void ReadTransposed( const S * const b )
        {
            const I d_0 = dims[0];
            const I d_1 = dims[1];
            
            const S * restrict b__ = b;
                  T * restrict a__ = a;
            
            for( I i = 0; i < d_0; ++i )
            {
                for( I j = 0; j < d_1; ++j )
                {
                    a__[d_1 * i + j ] = static_cast<T>(b__[ d_0 * j + i ]);
                }
            }
        }
        
        // row-wise Write
        template< typename S>
        void Write( const I i, S * const b ) const
        {
//            memcpy( b, data(i), dims[1]  * sizeof(T) );
//            std::copy( data(i), data(i) + dims[1],  b );
            std::transform( data(i), data(i) + dims[1],  b, static_caster<T,S>() );
        }
        

        
        // row-wise Read
        template< typename S>
        void Read( const I i, const S * const b )
        {
//            std::copy( b, b + dims[1],  data(i) );
            std::transform( b, b + dims[1],  data(i), static_caster<S,T>() );
        }
        

        T * data( const I i )
        {
#ifdef BOUND_CHECKS
            if( i > Dimension(0) )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)) +" }.");
            }
#endif
            return a + i * (Dimension(1));
        }
        
        const T * data( const I i ) const
        {
#ifdef BOUND_CHECKS
            if( i > (Dimension(0)) )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)) +" }.");
            }
#endif
            return a + i * (Dimension(1));
        }

        T * data( const I i, const I j)
        {
#ifdef BOUND_CHECKS
            if( i > (Dimension(0)) )
            {
                eprint(ClassName()+"::data(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)) +" }.");
            }
            if( j > (Dimension(1)) )
            {
                eprint(ClassName()+"::data(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)) +" }.");
            }
#endif
            return a + i * (Dimension(1)) + j;
        }
        
        const T * data( const I i, const I j) const
        {
#ifdef BOUND_CHECKS
            if( i > (Dimension(0)) )
            {
                eprint(ClassName()+"::data(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)) +" }.");
            }
            if( j > (Dimension(1)) )
            {
                eprint(ClassName()+"::data(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)) +" }.");
            }
#endif
            return a + i * (Dimension(1)) + j;
        }
        
        T & operator()( const I i, const I j) const
        {
#ifdef BOUND_CHECKS
            if( i >= (Dimension(0)) )
            {
                eprint(ClassName()+"::operator()(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(Dimension(0)-1) +" }.");
            }
            if( j >= (Dimension(1)) )
            {
                eprint(ClassName()+"::operator(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(Dimension(1)-1) +" }.");
            }
#endif
            return a[ i * (Dimension(1)) + j ];
        }

        
        
    private:
        
        void calculate_sizes()
        {
            size__ = (Dimension(0)) * (Dimension(1));
            
#ifdef BOUND_CHECKS
            print(ClassName()+" { " +
                  std::to_string(Dimension(0)) + ", " +
                  std::to_string(Dimension(1)) + " }" );
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
            const I d_0 = Dimension(0);
            const I d_1 = Dimension(1);
            
            std::stringstream sout;
            sout.precision(n);
            sout << "{\n";
            if( Size() > 0 )
            {
                if( d_1 > 0 )
                {
                    sout << "\t{ ";
                    
                    sout << this->operator()(0,0);
                }
                for( I j = 1; j < d_1; ++j )
                {
                    sout << ", " << this->operator()(0,j);
                }
                
                for( I i = 1; i < d_0; ++i )
                {
                    sout << " },\n\t{ ";
                    
                    sout << this->operator()(i,0);
                    
                    for( I j = 1; j < d_1; ++j )
                    {
                        sout << ", " << this->operator()(i,j);
                    }
                }
            }
            sout << " }\n}";
            return sout.str();
        }
        
        static std::string ClassName()
        {
            return "Tensor2<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
    }; // Tensor2
    
//    template<typename I>
//    void GEMV( const double alpha, const Tensor2<double,I> & A, const CBLAS_TRANSPOSE transA,
//                                         const Tensor1<double,I> & x,
//                      const double beta,        Tensor1<double,I> & y )
//    {
//        I m = A.Dimension(0);
//        I n = A.Dimension(1);
//        I k = A.Dimension(transA == CblasNoTrans);
//        I stride_x = 1;
//        I stride_y = 1;
//        
//        cblas_dgemv( CblasRowMajor, transA, m, n, alpha, A.data(), k, x.data(), stride_x, beta, y.data(), stride_y );
//    }
//    
//    template<typename I>
//    void GEMM( const double alpha, const Tensor2<double,I> & A, const CBLAS_TRANSPOSE transA,
//                                          const Tensor2<double,I> & B, const CBLAS_TRANSPOSE transB,
//                      const double beta,        Tensor2<double,I> & C )
//    {
//        I m = C.Dimension(0);
//        I n = C.Dimension(1);
//        I k = A.Dimension(transA == CblasNoTrans);
//
//        cblas_dgemm( CblasRowMajor, transA, transB, m, n, k, alpha, A.data(), m, B.data(), n, beta, C.data(), n );
//    }
//
//    template<typename I>
//    void Dot ( const Tensor2<double,I> & A, const CBLAS_TRANSPOSE transA,
//                      const Tensor2<double,I> & B, const CBLAS_TRANSPOSE transB,
//                            Tensor2<double,I> & C, bool addTo = false )
//    {
//        double alpha = 1.;
//        I m = C.Dimension(0);
//        I n = C.Dimension(1);
//        I k = A.Dimension(transA == CblasNoTrans);
//
//        cblas_dgemm( CblasRowMajor, transA, transB, m, n, k, alpha, A.data(), k, B.data(), n, addTo, C.data(), n );
//    }

//    template<typename I>
//    int LinearSolve( const Tensor2<double,I> & A, const Tensor1<double,I> & b, Tensor1<double,I> & x )
//    {
//        I stride_x = 1;
//        I n = A.Dimension(0);
//        if( x.Dimension(0) != n )
//        {
//            x = Tensor1<double,I>(n);
//        }
//        x.Read(b.data());
//        Tensor1<I,I> ipiv ( n );
//        Tensor2<double,I> A1 ( A.data(), n , n );
//
//        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, stride_x, A1.data(), n, ipiv.data(), x.data(), stride_x );
////        valprint("stat",stat);
//        return stat;
//    }
//    
//    template<typename I>
//    int LinearSolve( Tensor2<double,I> & A, Tensor1<double,I> & b )
//    {
//        // in place variant
//        I stride_x = 1;
//        I stride_b = 1;
//        I n = A.Dimension(0);
//        if( b.Dimension(0) != n )
//        {
//            b = Tensor1<double,I>(n);
//        }
//        
//        Tensor1<I,I> ipiv ( n );
//        
//        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, stride_x, A.data(), n, ipiv.data(), b.data(), stride_b );
////        valprint("stat",stat);
//        return stat;
//    }
//    
//    template<typename I>
//    int Inverse( const Tensor2<double,I> & A, Tensor2<double,I> & Ainv )
//    {
//        I n = A.Dimension(0);
//        if( Ainv.Dimension(0) != n || Ainv.Dimension(1) != n )
//        {
//            Ainv = Tensor2<double,I>(n,n);
//        }
//        Tensor1<I,I> ipiv ( n );
//        Tensor2<double,I> A1 ( A.data(), n , n );
//        
//        double * b = Ainv.data();
//        #pragma omp simd collapse(2) aligned( b : ALIGN )
//        for( I i = 0; i < n; ++i )
//        {
//            for( I j = 0; j < n; ++j )
//            {
//                b[ n * i + j ] = static_cast<double>(i==j);
//            }
//        }
//        // LAPACKE_dgesv does not like its 4-th argument to be const.
//        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, n, A1.data(), n, ipiv.data(), Ainv.data(), n );
////        valprint("stat",stat);
//        return stat;
//    }

    template<typename T, typename I, typename S, typename J0, typename J1, IsInt(I), IsInt(J0), IsInt(J1)>
    Tensor2<T,I> ToTensor2( const S * a_, const J0 d0, const J1 d1, const bool transpose = false )
    {
        Tensor2<T,I> result (static_cast<I>(d0), static_cast<I>(d1));

        if( transpose )
        {
            result.ReadTransposed(a_);
        }
        else
        {
            result.Read(a_);
        }
        
        return result;
    }
    
#ifdef LTEMPLATE_H

    template<typename T, typename I, IsFloat(T)>
    mma::TensorRef<mreal> to_MTensorRef( const Tensor2<T,I> & A )
    {
        auto B = mma::makeMatrix<double>( A.Dimension(0), A.Dimension(1) );
        
        A.Write(B.data());

        return B;
    }
    
    template<typename J, typename I, IsInt(J)>
    mma::TensorRef<mint> to_MTensorRef( const Tensor2<J,I> & A )
    {
        auto B = mma::makeMatrix<mint>( A.Dimension(0), A.Dimension(1) );
        
        A.Write(B.data());

        return B;
    }
    
    
    template<typename T, typename I>
    Tensor2<T,I> from_MatrixRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor2<T,I>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    
    template<typename T, typename I>
    Tensor2<T,I> from_MatrixRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor2<T,I>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    

    template<typename T, typename I, IsFloat(T)>
    mma::MatrixRef<mreal> to_transposed_MTensorRef( const Tensor2<T,I> & B )
    {
        I rows = B.Dimension(0);
        I cols = B.Dimension(1);
        auto A = mma::makeMatrix<mreal>( cols, rows );

        double * a_out = A.data();

        #pragma omp parallel for collapse(2)
        for( I i = 0; i < rows; ++i )
        {
            for( I j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mreal>( B(i,j) );
            }
        }

        return A;
    }
    
    template<typename J, typename I, IsInt(J)>
    mma::MatrixRef<mint> to_transposed_MTensorRef( const Tensor2<J,I> & B )
    {
        I rows = B.Dimension(0);
        I cols = B.Dimension(1);
        auto A = mma::makeMatrix<mint>( cols, rows );

        double * a_out = A.data();

        #pragma omp parallel for collapse(2)
        for( I i = 0; i < rows; ++i )
        {
            for( I j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mint>( B(i,j) );
            }
        }

        return A;
    }

    
#endif
    
#include "Tensor_Common_External.h"
    
#undef TENSOR_T
} // namespace Tensors
