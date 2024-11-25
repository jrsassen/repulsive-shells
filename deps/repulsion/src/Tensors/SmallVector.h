#pragma once

#define CLASS SmallVector

namespace Tensors {

    template <typename T, int N>
    class CLASS
    {
        protected:
        
        T * a = nullptr;
        
        public:
        
        CLASS()
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" default constructor { " + std::to_string(N) + " }");
#endif
            safe_alloc(a,N);
        }
        
        explicit CLASS( const T init )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" fill constructor { " + std::to_string(N) + " }");
#endif
            safe_alloc(a,N);
            fill( init );
        }
        
        explicit CLASS( const T * a_ )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" pointer constructor { " + std::to_string(N) + " }");
#endif
            safe_alloc(a,N);
            Read(a_);
        }
        
        // Copy constructor
        explicit CLASS( const CLASS & B )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" copy constructor { " + std::to_string(N) + " }" );
#endif
            safe_alloc(a,N);
            Read(B.a);
        }
        
        // Move constructor
        explicit CLASS( CLASS && B ) noexcept
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" move constructor { " + std::to_string(N) + " }" );
#endif
            safe_free(a);
            a = B.a;
            B.a = nullptr;
        }
        
        friend void swap(CLASS &A, CLASS &B) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
//#ifdef BOUND_CHECKS
//            print(ClassName()+" swap { " + std::to_string(N) + " }" );
//#endif
            swap(A.a, B.a);
        }
        
        // copy-and-swap idiom
        CLASS & operator=(CLASS B)
        {
            // see https://stackoverflow.com/a/3279550/8248900 for details
#ifdef BOUND_CHECKS
            print("~CLASS copy-and-swap { " + std::to_string(N) + " }" );
#endif
            swap(*this, B);

            return *this;
            
        }
        
        ~CLASS()
        {
#ifdef BOUND_CHECKS
            print("~"+ClassName()+" { " + std::to_string(N) + " }" );
#endif
            safe_free(a);
        }
        
#ifdef LTEMPLATE_H
        mma::TensorRef<T> to_MTensorRef() const
        {
            auto A = mma::makeVector<T>( N );
            
            Write(A.data());

            return A;
        }
#endif
    
        void Read( const T * const a_ )
        {
            const T * restrict const a__ = a_;
            
            for( int k = 0; k < N; ++k )
            {
                a[k] = a__[k];
            }
        }
        
        void Write( T * a_ ) const
        {
            T * restrict const a__ = a_;
            
            for( int k = 0; k < N; ++k )
            {
                a__[k] = a[k];
            }
        }

        
        static constexpr int Rank()
        {
            return 1;
        }
        
        static constexpr int Size()
        {
            return N;
        }
        
        static int Dimension( const int i )
        {
            switch( i )
            {
                case 0:     return N;
                default:    return 0;
            }
        }

        T * begin()
        {
            return  a;
        }
        
        const T * begin() const
        {
            return  a;
        }
        
        T * end()
        {
            return  a+N;
        }
        
        const T * end() const
        {
            return  a+N;
        }
        
        
        T * data()
        {
            return  a;
        }
        
        const T * data() const
        {
            return  a;
        }

        T * data( const int i )
        {
#ifdef BOUND_CHECKS
            if( i > N )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
#endif
            return a + i;
        }
        
        const T * data( const int i ) const
        {
#ifdef BOUND_CHECKS
            if( i > N )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
#endif
            return a + i;
        }
        
        T & operator()( const int i)
        {
#ifdef BOUND_CHECKS
            if( i >= N )
            {
                eprint(ClassName()+"::operator()(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
#endif
            return a[i];
        }
        
        const T & operator()( const int i) const
        {
#ifdef BOUND_CHECKS
            if( i >= N )
            {
                eprint(ClassName()+"::operator()(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
#endif
            return a[i];
        }
        
        T & operator[]( const int i)
        {
#ifdef BOUND_CHECKS
            if( i >= N )
            {
                eprint(ClassName()+"::operator[](i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
#endif
            return a[i];
        }
        
        const T & operator[]( const int i) const
        {
#ifdef BOUND_CHECKS
            if( i >= N )
            {
                eprint(ClassName()+"::operator[](i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
#endif
            return a[i];
        }
        
        void fill( const T init )
        {
            for( int i = 0; i < N; ++i )
            {
                a[i] = init;
            }
        }
        
        void Fill( const T init )
        {
            for( int i = 0; i < N; ++i )
            {
                a[i] = init;
            }
        }
        
        void random()
        {
            std::uniform_real_distribution<double> unif(-1.,1.);
            std::default_random_engine re{static_cast<unsigned int>(time(0))};
            
            for( int i = 0; i < N; ++i )
            {
                a[i] = unif(re);
            }
        }
        
        std::string to_string( int begin_ = 0, int end_ = N, int prec = 16 ) const
        {
            std::ostringstream out;
            out.precision(prec);
            
            int begin__ = std::max( static_cast<int>(0),begin_);
            int end__ = std::min(static_cast<int>(N),end_);
            
            out << "{";
            if( N >= begin__ )
            {
                out << " " << a[begin__];
            }

            for( int k = begin__ + 1; k < end__ ; ++k )
            {
                out << ", ";
                out << a[k];
            }
            out << " }";
            return out.str();
        }
    
    public:
        
        template<typename T, int N>
        friend T Dot( const CLASS & x, const CLASS & y )
        {
            T sum = 0;
            
            const T * restrict const x__ = x.a;
            const T * restrict const y__ = y.a;
            
            for( int k = 0; k < N; ++k )
            {
                sum += x_a[k] * y_a[k];
            }
            
            return sum;
        }
        
        template<typename T, int N>
        friend void Subtract( const CLASS & x, const CLASS & y, CLASS & z )
        {
            const T * restrict const x__ = x.a;
            const T * restrict const y__ = y.a;
                  T * restrict const z__ = z.a;
            
            for( int k = 0; k < N; ++k )
            {
                z_a[k] = x_a[k] - y_a[k];
            }
        }
        
        template<typename T, int N>
        friend void Plus( const CLASS & x, const CLASS & y, CLASS & z )
        {
            const T * restrict const x__ = x.a;
            const T * restrict const y__ = y.a;
                  T * restrict const z__ = z.a;
            
            SMALL_UNROLL
            for( int k = 0; k < N; ++k )
            {
                z_a[k] = x_a[k] + y_a[k];
            }
        }
        
        friend void AXPY( const T a, const CLASS & x, CLASS & y )
        {
            if( a == static_cast<T>(1) )
            {
                y.Read(x);
            }
            else
            {
                const T * restrict const x_a = x.a;
                      T * restrict const y_a = y.a;
                
                SMALL_UNROLL
                for( int k = 0; k < N; ++k )
                {
                    y_a[k] = a * x_a[k] + y_a[k];
                }
            }
        }
        
        friend void Scale( const T a, CLASS & x )
        {
            T * restrict const x_a = x.a;
            
            SMALL_UNROLL
            for( int k = 0; k < N; ++k )
            {
                x_a[k] *= a;
            }
        }
        
        friend void Times( const T a, const CLASS & x, CLASS & y )
        {
            const T * restrict const x_a = x.a;
                  T * restrict const y_a = y.a;
            
            SMALL_UNROLL
            for( int k = 0; k < N; ++k )
            {
                y_a[k] = a * x_a[k];
            }
        }
        
        friend void Minus( const CLASS & x, CLASS & y )
        {
            const T * restrict const x_a = x.a;
                  T * restrict const y_a = y.a;
            
            SMALL_UNROLL
            for( int k = 0; k < N; ++k )
            {
                y_a[k] = -x_a[k];
            }
        }
        
        friend int IMAX( const CLASS & x )
        {
            const T * restrict const x_a = x.a;
            
            int pos = 0;
    //        T maximum = x[0];
            
            SMALL_UNROLL
            for( int k = 1; k < N; ++k )
            {
                pos = ( x_a[k] > x[pos] ) ? k : pos;
    //            maximum = std::max(maximum, x__[k]);
            }
            
            return pos;
        }
        
        friend int IMIN( const CLASS & x )
        {
            const T * restrict const x_a = x.a;
            
            int pos = 0;
    //        T minimum = x[0];

            SMALL_UNROLL
            for( int k = 1; k < N; ++k )
            {
                pos = ( x_a[k] < x[pos] ) ? k : pos;
    //            minimum = std::min(minimum, x__[k]);
            }
            
            return pos;
        }
        
    public:
        
        template<typename T, int N>
        std::string ToString( int prec = 16 )
        {
            std::ostringstream out;
            out.precision(prec);
            
            out << "{";
            out << " " << v[0];

            for( int k = 1; k < N; ++k )
            {
                out << ", ";
                out << v[k];
            }
            out << " }";
            return out.str();
        }
        
        static virtual std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<T>::Get()+","+ToString(N)+">";
        }
        
    }; // CLASS
    

    template<typename T, int N>
    inline SmallVector<T, N> iota()
    {
        auto v = SmallVector<T,N>();
        
        T * restrict const a__ = v.data();

        SMALL_UNROLL
        for( int i = 0; i < N; ++i )
        {
            a__[i] = static_cast<T>(i);
        }
        return v;
    }
    
    
#ifdef LTEMPLATE_H

    template<int N>
    mma::TensorRef<mreal> to_MTensorRef( const SmallVector<mreal,N> & B )
    {
        auto A = mma::makeVector<mreal>( N );

        B.Write(A.data());

        return A;
    }
    
    template<int N>
    mma::TensorRef<mreal> to_MTensorRef( const SmallVector<float,N> & B )
    {
        auto A = mma::makeVector<mreal>( N );

        mreal * restrict const a_out = A.data();
  
        for( int i = 0; i < N; ++i )
        {
            a_out[ i ] = static_cast<mreal>( B(i) );
        }

        return A;
    }
    
#endif
    
} // namespace Tensors

#undef CLASS
