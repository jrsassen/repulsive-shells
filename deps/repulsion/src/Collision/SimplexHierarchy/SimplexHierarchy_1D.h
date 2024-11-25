#pragma once

namespace Collision {
    
#define DOM_DIM 1
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class alignas( 2 * CACHE_LINE_WIDTH ) SimplexHierarchy<DOM_DIM,AMB_DIM,Real,Int,SReal>
    {
        
#include "SimplexHierarchy_Details.h"
        
        constexpr Int ChildCount() const
        {
            return 2;
        }
        
        void ToChild( const Int k )
        {
            current_simplex_computed = false;
            
            former_child_id = child_id;
            child_id = k;
            column = ChildCount() * column + k;
            ++level;
            
            scale  *= static_cast<SReal>(0.5);
            weight *= static_cast<SReal>(0.5);
            
            if(k == 0)
            {
                corners[1][0] = corners[0][0] + static_cast<SReal>(0.5) * (corners[1][0] - corners[0][0]);
                corners[1][1] = corners[0][1] + static_cast<SReal>(0.5) * (corners[1][1] - corners[0][1]);
                
                center[0]= corners[0][0] + static_cast<SReal>(0.5) * (center[0] - corners[0][0]);
                center[1]= corners[0][1] + static_cast<SReal>(0.5) * (center[1] - corners[0][1]);
                
//                corners[1][0] = center[0];
//                corners[1][1] = center[1];
//
//                center[0] = static_cast<SReal>(0.5) * ( corners[0][0] + center[0] );
//                center[1] = static_cast<SReal>(0.5) * ( corners[0][1] + center[1] );
                
            }
            else // if( k == 1 )
            {
                corners[0][0] = corners[1][0] + static_cast<SReal>(0.5) * (corners[0][0] - corners[1][0]);
                corners[0][1] = corners[1][1] + static_cast<SReal>(0.5) * (corners[0][1] - corners[1][1]);
                
                center[0] = corners[1][0] + static_cast<SReal>(0.5) * (center[0] - corners[1][0]);
                center[1] = corners[1][1] + static_cast<SReal>(0.5) * (center[1] - corners[1][1]);
                
//                corners[0][0] = center[0];
//                corners[0][1] = center[1];
//
//                center[0] = static_cast<SReal>(0.5) * ( corners[1][0] + center[0] );
//                center[1] = static_cast<SReal>(0.5) * ( corners[1][1] + center[1] );
                

                
            }
        }
        
        void ToParent()
        {
            if( level > 0)
            {
                current_simplex_computed = false;
                
                Int k = child_id;
                
                column = (column-child_id) / 2;
                former_child_id = child_id;
                child_id = column % 2;
                --level;
                
                scale  *= static_cast<SReal>(2);
                weight *= static_cast<SReal>(2);
                
                if( k == 0 )
                {
                    corners[1][0] = corners[0][0] + static_cast<SReal>(2) * (corners[1][0] - corners[0][0]);
                    corners[1][1] = corners[0][1] + static_cast<SReal>(2) * (corners[1][1] - corners[0][1]);
                    
                    center[0] = corners[0][0] + static_cast<SReal>(2) * (center[0] - corners[0][0]);
                    center[1] = corners[0][1] + static_cast<SReal>(2) * (center[1] - corners[0][1]);
                    
//                    center[0] = corners[1][0];
//                    center[1] = corners[1][1];
//
//                    corners[1][0] = corners[0][0] + static_cast<SReal>(2) * (corners[1][0] - corners[0][0]);
//                    corners[1][1] = corners[0][1] + static_cast<SReal>(2) * (corners[1][1] - corners[0][1]);
                }
                else // if( k == 1 )
                {
                    
                    corners[0][0] = corners[1][0] + static_cast<SReal>(2) * (corners[0][0] - corners[1][0]);
                    corners[0][1] = corners[1][1] + static_cast<SReal>(2) * (corners[0][1] - corners[1][1]);
                
                    center[0] = corners[1][0] + static_cast<SReal>(2) * (center[0] - corners[1][0]);
                    center[1] = corners[1][1] + static_cast<SReal>(2) * (center[1] - corners[1][1]);
                    
//                    center[0] = corners[0][0];
//                    center[1] = corners[0][1];
//                    
//                    corners[0][0] = corners[1][0] + static_cast<SReal>(2) * (corners[0][0] - corners[1][0]);
//                    corners[0][1] = corners[1][1] + static_cast<SReal>(2) * (corners[0][1] - corners[1][1]);
                }
            }
            else
            {
                wprint("Level is equal to 0 already.");
            }
        }
        
    }; // SimplexHierarchy
    
#undef DOM_DIM
    
} // namespace Collision
