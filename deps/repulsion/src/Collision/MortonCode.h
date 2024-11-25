#pragma once

namespace Collision
{
        
    
    template<size_t AMB_DIM>
    inline uint64_t InterleaveBits( const uint32_t * const x00 )
    {
        eprint("InterleaveBits<"+ToString(AMB_DIM)+"> not implemented. Returning 0.");
        return static_cast<uint64_t>(0);
    }
    
    // From : https://stackoverflow.com/a/18528775
    template<>
    inline uint64_t InterleaveBits<3>( const uint32_t * const x00 )
    {
        const uint32_t * restrict const x0 = x00;
        
        uint64_t u = static_cast<uint64_t>(0);
        
        for( size_t k = 0; k < 3; ++k )
        {
            uint64_t x = x0[k] & 0x1fffff;
            x = (x | x << 32) & 0x1f00000000ffff;
            x = (x | x << 16) & 0x1f0000ff0000ff;
            x = (x | x <<  8) & 0x100f00f00f00f00f;
            x = (x | x <<  4) & 0x10c30c30c30c30c3;
            x = (x | x <<  2) & 0x1249249249249249;
            
            u |= ( x << (2 - k ) );
        }
        
        return u;
    }
    
} // namespace Collision
