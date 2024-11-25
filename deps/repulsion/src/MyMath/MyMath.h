namespace MyMath
{

    template<typename Real, typename T>
    std::enable_if_t<!std::is_integral_v<T>,Real> inline pow( const Real base, const T exponent )
    {
        // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
        return base > static_cast<Real>(0) ? std::exp2( static_cast<Real>(exponent) * std::log2(base) ) : ( static_cast<Real>(exponent)!=static_cast<Real>(0) ? static_cast<Real>(0) : static_cast<Real>(1) );
    } // pow
    
    template<typename Real, typename Int>
    std::enable_if_t<std::is_integral_v<Int>,Real> inline pow( const Real base, const Int exponent)
    {
        if( exponent >= 0)
        {
            switch( exponent )
            {
                case 0:
                {
                    return static_cast<Real>(1);
                }
                case 1:
                {
                    return base;
                }
                case 2:
                {
                    return base * base;
                }
                case 3:
                {
                    return base * base * base;
                }
                case 4:
                {
                    Real b2 = base * base;
                    return b2 * b2;
                }
                case 5:
                {
                    Real b2 = base * base;
                    return b2 * b2 * base;
                }
                case 6:
                {
                    Real b2 = base * base;
                    return b2 * b2 * b2;
                }
                case 7:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b2 * base;
                }
                case 8:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4;
                }
                case 9:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * base;
                }
                case 10:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * b2;
                }
                case 11:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * b2 * base;
                }
                case 12:
                {
                    Real b2 = base * base;
                    Real b4 = b2 * b2;
                    return b4 * b4 * b4;
                }

                default:
                {
                    Real exp = exponent;
                    return pow(base, exp);
                }
            }
        }
        else
        {
            return static_cast<Real>(1)/pow(base, -exponent);
        }
    } // pow
    
//    #pragma omp declare simd simdlen(4)
//    inline double pow( const double base, const double exponent )
//    {
//        // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
//        return base > 0. ? std::exp2( exponent * std::log2(base) ) : ( exponent!=0. ? 0. : 1. );
//    } // pow
//
//    #pragma omp declare simd simdlen(4)
//    inline double pow( const double base, const mint exponent)
//    {
//        if( exponent >= 0)
//        {
//            switch( exponent )
//            {
//                case 0:
//                {
//                    return 1.;
//                }
//                case 1:
//                {
//                    return base;
//                }
//                case 2:
//                {
//                    return base * base;
//                }
//                case 3:
//                {
//                    return base * base * base;
//                }
//                case 4:
//                {
//                    double b2 = base * base;
//                    return b2 * b2;
//                }
//                case 5:
//                {
//                    double b2 = base * base;
//                    return b2 * b2 * base;
//                }
//                case 6:
//                {
//                    double b2 = base * base;
//                    return b2 * b2 * b2;
//                }
//                case 7:
//                {
//                    double b2 = base * base;
//                    double b4 = b2 * b2;
//                    return b4 * b2 * base;
//                }
//                case 8:
//                {
//                    double b2 = base * base;
//                    double b4 = b2 * b2;
//                    return b4 * b4;
//                }
//                case 9:
//                {
//                    double b2 = base * base;
//                    double b4 = b2 * b2;
//                    return b4 * b4 * base;
//                }
//                case 10:
//                {
//                    double b2 = base * base;
//                    double b4 = b2 * b2;
//                    return b4 * b4 * b2;
//                }
//                case 11:
//                {
//                    double b2 = base * base;
//                    double b4 = b2 * b2;
//                    return b4 * b4 * b2 * base;
//                }
//                case 12:
//                {
//                    double b2 = base * base;
//                    double b4 = b2 * b2;
//                    return b4 * b4 * b4;
//                }
//
//                default:
//                {
//                    double exp = exponent;
//                    return pow(base, exp);
//                }
//            }
//        }
//        else
//        {
//            return 1./pow(base, -exponent);
//        }
//    } // pow
//    
//    
//    
//    #pragma omp declare simd simdlen(8)
//    inline float pow( const float base, const float exponent )
//    {
//        // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
//        return base > 0.0f ? std::exp2( exponent * std::log2(base) ) : ( exponent!=0.0f ? 0.0f : 1.0f );
//    } // pow
//    
//    #pragma omp declare simd simdlen(4)
//    inline float pow( const float base, const int exponent)
//    {
//        if( exponent >= 0)
//        {
//            switch( exponent )
//            {
//                case 0:
//                {
//                    return 1.0f;
//                }
//                case 1:
//                {
//                    return base;
//                }
//                case 2:
//                {
//                    return base * base;
//                }
//                case 3:
//                {
//                    return base * base * base;
//                }
//                case 4:
//                {
//                    float b2 = base * base;
//                    return b2 * b2;
//                }
//                case 5:
//                {
//                    float b2 = base * base;
//                    return b2 * b2 * base;
//                }
//                case 6:
//                {
//                    float b2 = base * base;
//                    return b2 * b2 * b2;
//                }
//                case 7:
//                {
//                    float b2 = base * base;
//                    float b4 = b2 * b2;
//                    return b4 * b2 * base;
//                }
//                case 8:
//                {
//                    float b2 = base * base;
//                    float b4 = b2 * b2;
//                    return b4 * b4;
//                }
//                case 9:
//                {
//                    float b2 = base * base;
//                    float b4 = b2 * b2;
//                    return b4 * b4 * base;
//                }
//                case 10:
//                {
//                    float b2 = base * base;
//                    float b4 = b2 * b2;
//                    return b4 * b4 * b2;
//                }
//                case 11:
//                {
//                    float b2 = base * base;
//                    float b4 = b2 * b2;
//                    return b4 * b4 * b2 * base;
//                }
//                case 12:
//                {
//                    float b2 = base * base;
//                    float b4 = b2 * b2;
//                    return b4 * b4 * b4;
//                }
//
//                default:
//                {
//                    float exp = exponent;
//                    return pow(base, exp);
//                }
//            }
//        }
//        else
//        {
//            return (1.0f)/pow(base, -exponent);
//        }
//    } // pow
    


    #pragma omp declare simd simdlen(4)
    inline double max( const double & a, const double & b )
    {
        return fmax(a,b);
    }

    #pragma omp declare simd simdlen(4)
    inline double min( const double & a, const double & b )
    {
        return fmin(a,b);
    }
    
    #pragma omp declare simd simdlen(4)
    inline long long sign( const double & a )
    {
        return static_cast<long long>( (a >= 0.) - (a <= 0.) );
    }
    
    #pragma omp declare simd simdlen(8)
    inline float max( const float & a, const float & b )
    {
        return fmax(a,b);
    }

    #pragma omp declare simd simdlen(8)
    inline float min( const float & a, const float & b )
    {
        return fmin(a,b);
    }
    
    
    
    template<typename T>
    inline T sign( const T val)
    {
        return static_cast<T>( (static_cast<T>(0) < val) - (val < static_cast<T>(0)) );
    }

}
