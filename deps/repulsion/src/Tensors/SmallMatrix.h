#pragma once

namespace Tensors {

    template <typename T, int ROWS, int COLS>
    class SmallMatrix
    {
        protected :
        
        T * a = nullptr;
        
        public :
        
        SmallMatrix()
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" default constructor { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
#endif
            safe_alloc( a, ROWS * COLS );
        }
        
        explicit SmallMatrix( const T init )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" fill constructor { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
#endif
            safe_alloc( a, ROWS * COLS );
            Fill( init );
        }
        
        explicit SmallMatrix( const T * a_ )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" pointer constructor { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
#endif
            safe_alloc( a, ROWS * COLS );
            Read(a_);
        }
        
        // Copy constructor
        explicit SmallMatrix( const SmallMatrix & B )
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" copy constructor { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
#endif
            safe_alloc( a, ROWS * COLS );
            Read(B.a);
        }
        
        // Move constructor
        explicit SmallMatrix( SmallMatrix && B ) noexcept 
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" move constructor { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
#endif
            a = B.a;
            B.a = nullptr;
        }
        
        friend void swap(SmallMatrix &A, SmallMatrix &B) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
//#ifdef BOUND_CHECKS
//            print(ClassName()+" swap { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
//#endif
            
//            swap(A.size, B.size);
            swap(A.a, B.a);
        }
        
        // copy-and-swap idiom
        SmallMatrix & operator=(SmallMatrix B)
        {
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, B);

            return *this;
            
        }
        
        ~SmallMatrix(){
#ifdef BOUND_CHECKS
            print("~"+ClassName()+" { " + std::to_string(ROWS) + ", " + std::to_string(COLS) + " }" );
#endif
            safe_free(a);
        }
        
#ifdef LTEMPLATE_H
        
        mma::MatrixRef<T> to_MTensorRef() const
        {
            auto A = mma::makeMatrix<T>( ROWS, COLS );

            T * restrict const a_out = A.data();
            SMALL_UNROLL
            for( int i = 0; i < ROWS; ++i )
            {
                SMALL_UNROLL
                for( int j = 0; j < COLS; ++j )
                {
                    a_out[ COLS * i + j] = a[ COLS * i + j ];
                }
            }

            return A;
        }
        
        mma::MatrixRef<T> to_transposed_MTensorRef() const
        {

            auto A = mma::makeMatrix<T>( COLS, ROWS );

            T * restrict const a_out = A.data();

            SMALL_UNROLL
            for( int j = 0; j < COLS; ++j )
            {
                SMALL_UNROLL
                for( int i = 0; i < ROWS; ++i )
                {
                    a_out[ ROWS * j + i] = a[ COLS * i + j ];
                }
            }

            return A;
        }
        
#endif
        
        void Read( const T * const a_ )
        {
            const T * restrict const a__ = a_;
            
            SMALL_UNROLL
            for( int i = 0; i < ROWS; ++i )
            {
                SMALL_UNROLL
                for( int j = 0; j < COLS; ++j )
                {
                    a[ COLS * i + j ] = a__[ COLS * i + j];
                }
            }
        }
        
        void Write( T * a_ ) const
        {
            T * restrict const a__ = a_;
            
            SMALL_UNROLL
            for( int i = 0; i < ROWS; ++i )
            {
                SMALL_UNROLL
                for( int j = 0; j < COLS; ++j )
                {
                    a__[ COLS * j + i] = a[ COLS * i + j ];
                }
            }
        }

        T * data()
        {
            return a;
        }
        
        const T * data() const
        {
            return a;
        }

        T * data( const int i )
        {
#ifdef BOUND_CHECKS
            if( i > ROWS )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(ROWS) +" }.");
            }
#endif
            return a + COLS * i ;
        }
        
        const T * data( const int i ) const
        {
#ifdef BOUND_CHECKS
            if( i > ROWS )
            {
                eprint(ClassName()+"::data(i): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(ROWS) +" }.");
            }
#endif
            return a + COLS * i ;
        }

        T * data( const int i, const int j)
        {
#ifdef BOUND_CHECKS
            if( i > ROWS )
            {
                eprint(ClassName()+"::data(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(ROWS) +" }.");
            }
            if( j > COLS )
            {
                eprint(ClassName()+"::data(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(COLS) +" }.");
            }
#endif
            return a + COLS * i + j;
        }
        
        const T * data( const int i, const int j) const
        {
#ifdef BOUND_CHECKS
            if( i > ROWS )
            {
                eprint(ClassName()+"::data(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(ROWS) +" }.");
            }
            if( j > COLS )
            {
                eprint(ClassName()+"::data(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(COLS) +" }.");
            }
#endif
            return a + COLS * i + j;
        }
        
        const T & operator()( const int i, const int j) const
        {
#ifdef BOUND_CHECKS
            if( i >= ROWS )
            {
                eprint(ClassName()+"::operator()(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(ROWS-1) +" }.");
            }
            if( j >= COLS )
            {
                eprint(ClassName()+"::operator(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(COLS-1) +" }.");
            }
#endif
            return a[ COLS * i + j ];
        }
        
        T & operator()( const int i, const int j)
        {
#ifdef BOUND_CHECKS
            if( i >= ROWS )
            {
                eprint(ClassName()+"::operator()(i,j): first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(ROWS-1) +" }.");
            }
            if( j >= COLS )
            {
                eprint(ClassName()+"::operator(i,j): second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(COLS-1) +" }.");
            }
#endif
            return a[ COLS * i + j ];
        }
        
        static constexpr int Rank()
        {
            return 2;
        }

        static constexpr int Size()
        {
            return ROWS * COLS;
        }
        
        static int Dimension( const int i )
        {
            switch( i )
            {
                case 0:     return ROWS;
                case 1:     return COLS;
                default:    return 0;
            }
        }
        
        void fill( const T init )
        {
            SMALL_UNROLL
            for( int i = 0; i < ROWS; ++i )
            {
                SMALL_UNROLL
                for( int j = 0; j < COLS; ++j )
                {
                    a[ COLS * i + j ] = init;
                }
                
            }
        }
        
        void Fill( const T init )
        {
            SMALL_UNROLL
            for( int i = 0; i < ROWS; ++i )
            {
                SMALL_UNROLL
                for( int j = 0; j < COLS; ++j )
                {
                    a[ COLS * i + j ] = init;
                }
                
            }
        }
        
        void random()
        {
            std::uniform_real_distribution<double> unif(-1.,1.);
            std::default_random_engine re{static_cast<unsigned int>(time(0))};
            
            SMALL_UNROLL
            for( int i = 0; i < ROWS; ++i )
            {
                SMALL_UNROLL
                for( int j = 0; j < COLS; ++j )
                {
                    a[ COLS * i + j ] = unif(re);
                }
            }
        }
        
        std::string to_string( int row_begin = 0, int row_end = ROWS, int col_begin = 0, int col_end = COLS, int prec = 16 ) const
        {
            std::ostringstream out;
            out.precision(prec);
            
            int _row_begin = std::max( static_cast<int>(0),row_begin);
            int _row_end = std::min(static_cast<int>(ROWS),row_end);
            
            int _col_begin = std::max( static_cast<int>(0),col_begin);
            int _col_end = std::min(static_cast<int>(COLS),col_end);
            
            out << " {\n";
            SMALL_UNROLL
            for( int i = _row_begin; i < _row_end ; ++i )
            {
                out << " \t{";
                if( COLS >= _col_begin )
                {
                    out << a[ COLS * i + col_begin ];
                }
                SMALL_UNROLL
                for( int j = _col_begin + 1; j < _col_end ; ++j )
                {
                    out << ", ";
                    out << a[ COLS * i + j ];
                }
                out << " }";
                if( i < _row_end-1 )
                {
                    out << ",";
                }
                out << "\n";
            }
            out << " }";
            return out.str();
        }
        
        friend std::string to_string( const SmallMatrix & A, int prec = 16 )
        {
            std::ostringstream out;
            out.precision(prec);
            
            out << " {\n";
            SMALL_UNROLL
            for( int i = 0; i < ROWS ; ++i )
            {
                out << " \t{";
                out << A(i,0);
                SMALL_UNROLL
                for( int j = 1; j < COLS ; ++j )
                {
                    out << ", ";
                    out << A(i,j);
                }
                out << " }";
                if( i < ROWS-1 )
                {
                    out << ",";
                }
                out << "\n";
            }
            out << " }";
            return out.str();
        }
        
    public:
        
        static virtual std::string ClassName()
        {
            return "SmallMatrix<"+TypeName<T>::Get()+","+ToString(ROWS)+","+ToString(COLS)+">";
        }
        
    }; // SmallMatrix
    
    template <typename T, int ROWS, int COLS>
    inline void GEMV( const T alpha,     const SmallMatrix<T, ROWS, COLS> & A,
                                         const SmallVector<T, COLS> & x,
                      const double beta,        SmallVector<T, ROWS> & y )
    {
        const T * restrict const A__ = A.data();
        const T * restrict const x__ = x.data();
              T * restrict const y__ = y.data();
        
        SMALL_UNROLL
        for( int i = 0; i < ROWS; ++i )
        {
            SMALL_UNROLL
            for( int j = 0; j < COLS; ++j )
            {
                y__[i] = A__[ COLS * i + j] * x__[j];
            }
        }
    }

    template <int ROWS>
    int LinearSolve( const SmallMatrix<double, ROWS, ROWS> & A, const SmallVector<double, ROWS> & b, SmallVector<double, ROWS> & x )
    {
        x.Read(b.data());
        
        int ipiv[ROWS];

        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, ROWS, 1, A.data(), ROWS, ipiv.data(), x.data(), 1 );

        return stat;
    }
    
    template <int ROWS>
    int LinearSolve( const SmallMatrix<double, ROWS, ROWS> & A, SmallVector<double, ROWS> & b )
    {
        int ipiv[ROWS];

        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, ROWS, 1, A.data(), ROWS, ipiv.data(), b.data(), 1 );

        return stat;
    }

#ifdef LTEMPLATE_H

    template <int ROWS, int COLS>
    mma::MatrixRef<double> to_MTensorRef( const SmallMatrix<double, ROWS, COLS> & B )
    {
        auto A = mma::makeMatrix<double>( ROWS, COLS );

        double * restrict const a_out = A.data();

        SMALL_UNROLL
        for( int i = 0; i < ROWS; ++i )
        {
            SMALL_UNROLL
            for( int j = 0; j < COLS; ++j )
            {
                a_out[ COLS * i + j] = B(i,j);
            }
        }

        return A;
    }

    template <int ROWS, int COLS>
    mma::MatrixRef<double> to_transposed_MTensorRef( const SmallMatrix<double,ROWS, COLS > & B )
    {
        auto A = mma::makeMatrix<double>( COLS, ROWS );

        double * restrict const a_out = A.data();

        SMALL_UNROLL
        for( int i = 0; i < ROWS; ++i )
        {
            SMALL_UNROLL
            for( int j = 0; j < COLS; ++j )
            {
                a_out[ ROWS * j + i] = B(i,j);
            }
        }

        return A;
    }
    
    template <int ROWS, int COLS>
    mma::MatrixRef<double> to_MTensorRef( const SmallMatrix<float, ROWS, COLS> & B )
    {
        auto A = mma::makeMatrix<double>( ROWS, COLS );

        double * restrict const a_out = A.data();

        SMALL_UNROLL
        for( int i = 0; i < ROWS; ++i )
        {
            SMALL_UNROLL
            for( int j = 0; j < COLS; ++j )
            {
                a_out[ COLS * i + j] = static_cast<double>( B(i,j) );
            }
        }

        return A;
    }
    
    template <int ROWS, int COLS>
    mma::MatrixRef<double> to_transposed_MTensorRef( const SmallMatrix<float, ROWS, COLS> & B )
    {
        auto A = mma::makeMatrix<double>( COLS, ROWS );

        double * restrict const a_out = A.data();

        SMALL_UNROLL
        for( int i = 0; i < ROWS; ++i )
        {
            SMALL_UNROLL
            for( int j = 0; j < COLS; ++j )
            {
                a_out[ ROWS * j + i] = static_cast<double>( B(i,j) );
            }
        }

        return A;
    }
    
#endif
    
} // namespace Tensors
