#pragma once

namespace Tensors
{
    template<typename T, typename I>
    class TwoArrayQuickSort
    {
        ASSERT_INT  (I);
        
    protected:
        
        Tensor1<I,I> stack;
        I stackptr = -1;
        
    public:
        
        void Sort( I * restrict const a, T * restrict const b, const I n )
        {
            // Recursion-free implementation from https://www.geeksforgeeks.org/iterative-quick-sort/
            // The handling of duplicates is taken from https://cs.stackexchange.com/a/104825
            if( n > 0)
            {
                if( n > stack.Size() )
                {
                    stack = Tensor1<I,I>( 2 * n );
                }
                
                stackptr = -1;
                
                stack[++stackptr] = 0;
                stack[++stackptr] = n-1;
                
                while( stackptr >= 0 )
                {
                    const I hi = stack[stackptr--];
                    const I lo = stack[stackptr--];
                    
                    I l = lo;
                    I r = lo;
                    I u = hi;
                    
                    // lo <= l <= r <= u <= hi
                    
                    // - elements in [lo,l) are smaller than pivot
                    // - elements in [l,r) are equal to pivot
                    // - elements in [r,u] are not examined, yet
                    // - elements in (u,hi] are greater than pivot
                    
                    const I pivot = a[lo];
                    
                    while( r <= u )
                    {
                        if( a[r] < pivot )
                        {
                            std::swap( a[l], a[r] );
                            std::swap( b[l], b[r] );
                            l++;
                            r++;
                        }
                        else if( a[r] > pivot )
                        {
                            std::swap( a[r], a[u] );
                            std::swap( b[r], b[u] );
                            u--;
                        }
                        else
                        {
                            // element a[r] is equal to pivot
                            r++;
                        }
                    }
                    
                    
                    if( l-1 > lo )
                    {
                        stack[++stackptr] = lo;
                        stack[++stackptr] = l-1;
                    }
                    
                    if( r < hi )
                    {
                        stack[++stackptr] = r;
                        stack[++stackptr] = hi;
                    }
                    
                    
                }
            }
        }
    }; // TwoArrayQuickSort
    
}
