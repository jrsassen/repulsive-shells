#pragma once

namespace Collision
{
    
    
    template<mint AMB_DIM>
    void HilbertValue( const mint n, mint * const x )
    {
        eprint("HilbertValue<"+ToString(AMB_DIM)+"> has not been implemented, yet.");
    }
    
    template<mint AMB_DIM>
    void HilbertCurve( const mint n, const mint d, mint * const x )
    {
        eprint("HilbertCurve<"+ToString(AMB_DIM)+"> has not been implemented, yet.");
    }
    
    
        
    //rotate/flip a quadrant appropriately
    inline void rot( const mint n, mint * const x, mint * const r )
    {
        if( r[1] == 0 )
        {
            if (r[0] == 1)
            {
                x[0] = n-1 - x[0];
                x[1] = n-1 - x[1];
            }
            
            //Swap x and y
            mint t = x[0];
            x[0] = x[1];
            x[1] = t;
        }
    }


//    template<>
//    inline mint HilbertValue<2>( const mint n, mint * const x )
//    {
//        mint r [2];
//        mint s;
//        mint d = 0;
//
//        for (s=n/2; s>0; s/=2) {
//            r[0] = (x[0] & s) > 0;
//            r[1] = (x[1] & s) > 0;
//            d += s * s * ((3 * r[0]) ^ r[1]);
//            rot(n, x, r);
//        }
//
//        return d;
//    }
    
    //convert (x,y) to d
    template<>
    inline mint HilbertValue<2>( const mint n, mint * const x )
    {
        mint r [2];
        mint d = 0;
        for( mint s = n/2; s > 0; s/=2 )
        {
            r[0] = (x[0] & s) > 0;
            r[1] = (x[1] & s) > 0;
            d += s * s * ((3 * r[0]) ^ r[1]);
            rot(n, x, r);
        }
        return d;
    }

    template<>
    inline void HilbertCurve<2>( const mint n, const mint d, mint * const x )
    {
        mint r [2];
        mint t = d;

        x[0] = 0;
        x[1] = 0;
        for( mint s = 1; s < n; s *= 2 )
        {
            r[0] = 1 & (t/2);
            r[1] = 1 & (t ^ r[0]);
            rot(s, x, r);
            x[0] += s * r[0];
            x[1] += s * r[1];
            t /= 4;
        }
    }

    
} // namespace Collision
