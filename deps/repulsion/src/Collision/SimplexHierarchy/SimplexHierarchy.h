#pragma once


namespace Collision {

    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal>
    class alignas( 2 * CACHE_LINE_WIDTH ) SimplexHierarchy
    {
        
#include "SimplexHierarchy_Details.h"
        
        constexpr Int ChildCount() const
        {
            return 0;
        }
        
        void ToChild( const Int k )
        {
            print(ClassName()+"::ToChild not implemented.");
        }
        
        void ToParent()
        {
            print(ClassName()+"::ToParent not implemented.");
        }
        
    }; // SimplexHierarchy
    
} // Collision
