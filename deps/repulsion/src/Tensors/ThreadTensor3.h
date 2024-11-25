#pragma once

namespace Tensors {

    template <typename T, typename I>
    class  alignas( 2 * CACHE_LINE_WIDTH ) ThreadTensor3
    {
        ASSERT_INT(I);
        
    private:
        
        std::vector<Tensor2<T,I>> tensors;
        I dims [3] = {};
        I size__ = 0;
        
    public:
        
        ThreadTensor3() {}
        
        template<typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)>
        ThreadTensor3( const J0 d0, const J1 d1, const J2 d2 )
        :
            tensors( std::vector<Tensor2<T,I>> (static_cast<I>(d0)) ),
            size__(static_cast<I>(d0 * d1 * d2))
        {
            dims[0]=static_cast<I>(d0);
            dims[1]=static_cast<I>(d1);
            dims[2]=static_cast<I>(d2);
            
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<T,I>( dims[1], dims[2] );
            }
        }
        
        template<typename S, typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)>
        ThreadTensor3( const J0 d0, const J1 d1, const J2 d2, const S init )
        :
            tensors( std::vector<Tensor2<T,I>> (static_cast<I>(d0)) ),
            size__(static_cast<I>(d0 * d1 * d2))
        {
            dims[0]=static_cast<I>(d0);
            dims[1]=static_cast<I>(d1);
            dims[2]=static_cast<I>(d2);
            
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<T,I>( dims[1], dims[2], static_cast<T>(init) );
            }
        }
        
        template<typename S, typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)>
        ThreadTensor3( const S * a_, const J0 d0, const J1 d1, const J2 d2 )
        :
            tensors( std::vector<Tensor2<T,I>> (static_cast<I>(d0)) ),
            size__(d0 * d1 * d2)
        {
            dims[0]=static_cast<I>(d0);
            dims[1]=static_cast<I>(d1);
            dims[2]=static_cast<I>(d2);
            
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<T,I>( dims[1], dims[2] );
                tensors[thread].Read( a_ + thread * d1 * d2 );
            }
        }
        
        // Copy constructor
        template<typename S, typename J, IsInt(J)>
        ThreadTensor3( const ThreadTensor3<S,J> & other )
        :
            tensors( std::vector<Tensor2<T,I>> (static_cast<I>(other.Dimension(0))) ),
            size__(static_cast<I>(other.Size()))
        {
            dims[0] = static_cast<I>(other.Dimension(0));
            dims[1] = static_cast<I>(other.Dimension(1));
            dims[2] = static_cast<I>(other.Dimension(2));
            
            print(ClassName()+" copy constructor");
            
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<T,I>( other[thread] );
            }
        }
        
        // Copy constructor
        ThreadTensor3( const ThreadTensor3 & other )
        :
            tensors( std::vector<Tensor2<T,I>> (other.Dimension(0)) ),
            size__(other.Size())
        {
            dims[0] = other.Dimension(0);
            dims[1] = other.Dimension(1);
            dims[2] = other.Dimension(2);
            
            print(ClassName()+" copy constructor");
            
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<T,I>( other[thread] );
            }
        }
        
        friend void swap(ThreadTensor3 &A, ThreadTensor3 &B) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
#ifdef BOUND_CHECKS
            print(ClassName()+" swap");
#endif
            swap(A.tensors, B.tensors);
            swap(A.dims[0], B.dims[0]);
            swap(A.dims[1], B.dims[1]);
            swap(A.dims[2], B.dims[2]);
            swap(A.size__ , B.size__ );
        }
        
        // copy-and-swap idiom
        ThreadTensor3 & operator=(ThreadTensor3 B)
        {
            // see https://stackoverflow.com/a/3279550/8248900 for details
#ifdef BOUND_CHECKS
            print(ClassName()+" copy-and-swap");
#endif
            swap(*this, B);

            return *this;
            
        }
        
        // Move constructor
        ThreadTensor3( ThreadTensor3 && other ) noexcept
        :   ThreadTensor3()
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" move constructor");
#endif
            swap(*this, other);
        }
        
        ~ThreadTensor3(){
#ifdef BOUND_CHECKS
            print("~"+ClassName()+" { " + ToString(dims[0]) + ", " + ToString(dims[1]) + " }" );
#endif
        }
        
        
        static constexpr I Rank()
        {
            return static_cast<I>(3);
        }

        T * data( const I i )
        {
#ifdef BOUND_CHECKS
            if( i > dims[0] )
            {
                eprint(ClassName()+"::data(i): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]) +" }.");
            }
#endif
            return tensors[i].data();
        }
        
        const T * data( const I i ) const
        {
#ifdef BOUND_CHECKS
            if( i > dims[0] )
            {
                eprint(ClassName()+"::data(i): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]) +" }.");
            }
#endif
            return tensors[i].data();
        }

        T * data( const I i, const I j)
        {
#ifdef BOUND_CHECKS
            if( i > dims[0] )
            {
                eprint(ClassName()+"::data(i,j): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]) +" }.");
            }
            if( j > dims[1] )
            {
                eprint(ClassName()+"::data(i,j): second index " + ToString(j) + " is out of bounds { 0, " + ToString(dims[1]) +" }.");
            }
#endif
            return tensors[i].data(j);
        }
        
        const T * data( const I i, const I j) const
        {
#ifdef BOUND_CHECKS
            if( i > dims[0] )
            {
                eprint(ClassName()+"::data(i,j): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]) +" }.");
            }
            if( j > dims[1] )
            {
                eprint(ClassName()+"::data(i,j): second index " + ToString(j) + " is out of bounds { 0, " + ToString(dims[1]) +" }.");
            }
#endif
            return tensors[i].data(j);
        }
        
        T * data( const I i, const I j, const I k)
        {
#ifdef BOUND_CHECKS
            if( i > dims[0] )
            {
                eprint(ClassName()+"::data(i,j,k): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]-1) +" }.");
            }
            if( j > dims[1] )
            {
                eprint(ClassName()+"::data(i,j,k): second index " + ToString(j) + " is out of bounds { 0, " + ToString(dims[1]-1) +" }.");
            }
            if( k > dims[2] )
            {
                eprint(ClassName()+"::data(i,j,k): third index " + ToString(k) + " is out of bounds { 0, " + ToString(dims[2]-1) +" }.");
            }
#endif
            return tensors[i].data(j,k);
        }
        
        const T * data( const I i, const I j, const I k) const
        {
#ifdef BOUND_CHECKS
            if( i > dims[0] )
            {
                eprint(ClassName()+"::data(i,j,k): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]-1) +" }.");
            }
            if( j > dims[1] )
            {
                eprint(ClassName()+"::data(i,j,k): second index " + ToString(j) + " is out of bounds { 0, " + ToString(dims[1]-1) +" }.");
            }
            if( k > dims[2] )
            {
                eprint(ClassName()+"::data(i,j,k): third index " + ToString(k) + " is out of bounds { 0, " + ToString(dims[2]-1) +" }.");
            }
#endif
            return tensors[i].data(j,k);
        }
        
        
        
        T & operator()( const I i, const I j, const I k) const
        {
#ifdef BOUND_CHECKS
            if( i >= dims[0] )
            {
                eprint(ClassName()+"::operator()(i,j): first index " + ToString(i) + " is out of bounds { 0, " + ToString(dims[0]-1) +" }.");
            }
            if( j >= dims[1] )
            {
                eprint(ClassName()+"::operator(i,j): second index " + ToString(j) + " is out of bounds { 0, " + ToString(dims[1]-1) +" }.");
            }
            if( k > dims[2] )
            {
                eprint(ClassName()+"::data(i,j,k): third index " + ToString(k) + " is out of bounds { 0, " + ToString(dims[2]-1) +" }.");
            }
#endif
            return tensors[i](j,k);
        }
        
        void Fill( const T init )
        {
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].fill( init );
            }
        }
        
        void SetZero()
        {
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].SetZero();
            }
        }

        void Write( T * const b ) const
        {
            const I thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( I thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].Write( b + dims[1] * dims[2] * thread );
            }
        }
        
        template<typename S>
        void Write( const I i, S * const b ) const
        {
            tensors[i].Write( b );
        }
        
        template<typename S>
        void Write( const I i, const I j, S * const b ) const
        {
            tensors[i].Write( j, b );
        }
        
        template<typename S>
        void Read( const I i, const S * const b )
        {
            tensors[i].Read( b );
        }
        
        template<typename S>
        void Read( const I i, const I j, const S * const b )
        {
            tensors[i].Read( j, b );
        }
        
    public:
        
        const I * Dimensions() const
        {
            return dims;
        }
        
        I Dimension( const I i ) const
        {
            return i < Rank() ? dims[i] : static_cast<I>(0);
        }
 
        I Size() const
        {
            return size__;
        }
        
//        void AdditiveReduction( Tensor2<T,I> & B ) const
//        {
//            if( (Dimension(1) == B.Dimension(0)) && (Dimension(2) == B.Dimension(1)) )
//            {
//                // Write first slice.
//                B.Read( tensors[0].data() );
//                
//                for( I i = 1; i < dims[0]; ++ i)
//                {
//                    B.AddFrom( tensors[i].data() );
//                }
//            }
//            else
//            {
//                eprint(ClassName()+"::AdditiveReduction : Dimensions not compatible.");
//            }
//        }
//        
//        void AdditiveReduction( T * const B ) const
//        {
//            // Write first slice.
//            tensors[0].Write(B);
//            
//            for( I i = 1; i < dims[0]; ++ i )
//            {
//                tensors[i].AddTo( B );
//            }
//        }
        
        I CountNan() const
        {
            I counter = 0;
            for( I thread = 0 ; thread < dims[0]; ++thread )
            {
                counter += tensors[thread].CountNan();
            }
            return counter;
        }
        
        
        Tensor2<T,I> & operator[]( const I thread )
        {
            return tensors[thread];
        }
        
        const Tensor2<T,I> & operator[]( const I thread ) const
        {
            return tensors[thread];
        }
        
    public:
        
        static std::string ClassName()
        {
            return "ThreadTensor3<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
    }; // ThreadTensor3
    
} // namespace Tensors
