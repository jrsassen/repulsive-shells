template <typename T, typename I>
void Subtract( const TENSOR_T<T,I> & x, const TENSOR_T<T,I> & y, TENSOR_T<T,I> & z )
{
    const T * restrict const x__ = x.data();
    const T * restrict const y__ = y.data();
          T * restrict const z__ = z.data();
    
    const I last = x.Size();
    
    #pragma omp parallel for simd aligned( x__, y__, z__ : ALIGN ) schedule( static, (16 * CACHE_LINE_WIDTH) / sizeof(T) )
    for( I k = 0; k < last; ++ k)
    {
        z__[k] = x__[k] - y__[k];
    }
}

template <typename T, typename I>
void Plus( const TENSOR_T<T,I> & x, const TENSOR_T<T,I> & y, TENSOR_T<T,I> & z )
{
    const T * restrict const x__ = x.data();
    const T * restrict const y__ = y.data();
          T * restrict const z__ = z.data();
   
    const I last = x.Size();
    
    #pragma omp parallel for simd aligned( x__, y__, z__ : ALIGN ) schedule( static, (16 * CACHE_LINE_WIDTH) / sizeof(T) )
    for( I k = 0; k < last; ++ k)
    {
        z__[k] = x__[k] + y__[k];
    }
}

template <typename T, typename I>
void Times( const T a, const TENSOR_T<T,I> & x, TENSOR_T<T,I> & y )
{
    const T * restrict const x__ = x.data();
          T * restrict const y__ = y.data();
    
    const I last = x.Size();
    
    #pragma omp parallel for simd aligned( x__, y__ : ALIGN ) schedule( static, (16 * CACHE_LINE_WIDTH) / sizeof(T) )
    for( I k = 0; k < last; ++ k)
    {
        y__[k] = a * x__[k];
    }
}

//template <typename T, typename I>
//std::string to_string(const TENSOR_T<T,I> & x)
//{
//    return x.ToString();
//}


//template <typename T, typename I>
//void AXPY( const T a, const TENSOR_T<T,I> & x, TENSOR_T<T,I> & y )
//{
//    mint n = x.Size();
//    mint stride_x = 1;
//    mint stride_y = 1;
//    cblas_daxpy( n, a, x.data(), stride_x, y.data(), stride_y );
//}

//template <typename T, typename I>
//void Scale( const T a, TENSOR_T<T,I> & x )
//{
//    mint n = x.Size();
//    mint stride_x = 1;
//    cblas_dscal( n, a, x.data(),  stride_x );
//}






//
//template<>
//inline void TENSOR_T<double>::AddFrom( const double * const a_ )
//{
//    double alpha = static_cast<double>(1);
//    mint n = Size();
//    mint stride_a = 1;
//    mint stride_a_ = 1;
//    cblas_daxpy( n, alpha, a_, stride_a_, a, stride_a );
//}
//
//template<>
//inline void TENSOR_T<float>::AddFrom( const float * const a_ )
//{
//    float alpha = static_cast<float>(1);
//    mint n = Size();
//    mint stride_a = 1;
//    mint stride_a_ = 1;
//    cblas_saxpy( n, alpha, a_, stride_a_, a, stride_a );
//}

