ASSERT_INT  (I);

public:

TENSOR_T() = default;

// Copy assignment operator
TENSOR_T & operator=(TENSOR_T other)
{
    // copy-and-swap idiom
    // see https://stackoverflow.com/a/3279550/8248900 for details

    swap(*this, other);

    return *this;
}

// Move constructor
TENSOR_T( TENSOR_T && other ) noexcept : TENSOR_T()
{
    swap(*this, other);
}

~TENSOR_T()
{
    safe_free(a);
}

protected :

T * a = nullptr;

I size__ = 0;

public:

I Size() const
{
    return size__;
}

template<typename S>
void Read( const S * const a_ )
{
    std::transform( a_, a_ + size__ , a, static_caster<S,T>() );
}

template<typename S>
void Write( S * a_ ) const
{
    std::transform( a, a + size__ , a_, static_caster<T,S>() );
}

void Fill( const T init )
{
    std::fill( a, a + size__, init );
}

void SetZero()
{
    memset( a, 0, size__ * sizeof(T) );
}

void Random()
{
    std::uniform_real_distribution<double> unif(-1.,1.);
    std::default_random_engine re{static_cast<unsigned int>(time(0))};
    
    T * restrict const a__ = a;
    
    #pragma omp parallel for STATIC_SCHEDULE
    for( I i = 0; i < size__; ++i )
    {
        a__[ i ] = unif(re);
    }
}

protected:

void allocate()
{
    calculate_sizes();
    safe_alloc( a, std::max( static_cast<I>(0), size__) );
}

public:

T * begin()
{
    return a;
}

const T * begin() const
{
    return a;
}

T * end()
{
    return a + size__;
}

const T * end() const
{
    return a + size__;
}

const I * dimensions() const
{
    return dims;
}

const I * Dimensions() const
{
    return dims;
}

I Dimension( const I i ) const
{
    if( i < Rank() )
    {
        return dims[i];
    }
    else
    {
        return 0;
    }
}

public:

T * data()
{
    return  a;
}

const T * data() const
{
    return  a;
}


void AddFrom( const T * const b )
{
          T * restrict const a__ = a;
    const T * restrict const b__ = b;
    
    const I last = size__;
    
//    #pragma omp parallel for simd aligned( a__ : ALIGN ) STATIC_SCHEDULE
    for( I i = 0; i < last; ++i )
    {
        // cppcheck-suppress [arithOperationsOnVoidPointer]
        a__[i] += b__[i];
    }
}

void AddTo( const T * const b )
{
    const T * restrict const a__ = a;
          T * restrict const b__ = b;
    
    const I last = size__;
    
//    #pragma omp parallel for simd aligned( a__ : ALIGN ) STATIC_SCHEDULE
    for( I i = 0; i < last; ++i )
    {
        // cppcheck-suppress [arithOperationsOnVoidPointer]
        b__[i] += a__[i];
    }
}

I CountNan() const
{
    const T * restrict const a__ = a;
    const I last = size__;
    
    I counter = 0;
//    #pragma omp simd aligned( a__ : ALIGN ) reduction( + : counter )
    for( I i = 0 ; i < last; ++i)
    {
        counter += (a__[i] != a__[i]);
    }
    
    return counter;
}

T MaxNorm() const
{
    T result = static_cast<T>(0);
    const T * restrict const a__ = a;
    const I last = size__;
    
//    #pragma omp simd aligned( a__ : ALIGN ) reduction( max : result )
    for( I i = 0 ; i < last; ++i)
    {
        result = std::max( result, a__[i]);
    }
    
    return result;
}
