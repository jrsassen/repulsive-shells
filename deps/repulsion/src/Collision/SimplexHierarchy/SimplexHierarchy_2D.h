#pragma once

namespace Collision {
    
#define DOM_DIM 2
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class alignas( 2 * CACHE_LINE_WIDTH ) SimplexHierarchy<DOM_DIM,AMB_DIM,Real,Int,SReal>
    {
        
#include "SimplexHierarchy_Details.h"
        
        constexpr Int ChildCount() const
        {
            return 4;
        }
        
        void ToChild( const Int k )
        {
            current_simplex_computed = false;
            
            former_child_id = child_id;
            child_id = k;
            column = 4 * column + k;
            ++level;
            
            scale  *= static_cast<SReal>(0.5);
            weight *= static_cast<SReal>(0.25);
            
            if( (0<=k) && (k<=2) )
            {
                SMALL_UNROLL
                for( Int i = 0; i < VertexCount(); ++ i )
                {
                    if( i != k )
                    {
                        SMALL_UNROLL
                        for( Int j = 0; j < VertexCount(); ++j )
                        {
                            corners[i][j] = corners[k][j] + static_cast<SReal>(0.5) * (corners[i][j] - corners[k][j]);
                        }
                    }
                    else
                    {
                        SMALL_UNROLL
                        for( Int j = 0; j < VertexCount(); ++j )
                        {
                            center[j]     = corners[k][j] + static_cast<SReal>(0.5) * (center[j]     - corners[k][j]);
                        }
                    }
                }
            }
            else if( child_id == 3 )
            {
                SMALL_UNROLL
                for( Int i = 0; i < VertexCount(); ++ i )
                {
                    SMALL_UNROLL
                    for( Int j = 0; j < VertexCount(); ++j )
                    {
                        corners[i][j] = center[j] - static_cast<SReal>(0.5) * (corners[i][j] - center[j]);
                    }
                }
            }
        }
        
        void ToParent()
        {
            if( level > 0)
            {
                current_simplex_computed = false;
                
                Int k = child_id;
                
                column = (column-child_id) / 4;
                former_child_id = child_id;
                child_id = column % 4;
                --level;
                
                scale  *= static_cast<SReal>(2);
                weight *= static_cast<SReal>(4);
                
                if( (0<=k) && (k<=2) )
                {
                    SMALL_UNROLL
                    for( Int i = 0; i < VertexCount(); ++i )
                    {
                        if( i != k )
                        {
                            SMALL_UNROLL
                            for( Int j = 0; j < VertexCount(); ++j )
                            {
                                corners[i][j] = corners[k][j] + static_cast<SReal>(2) * (corners[i][j] - corners[k][j]);
                            }
                        }
                        else
                        {
                            SMALL_UNROLL
                            for( Int j = 0; j < VertexCount(); ++j )
                            {
                                center[j] = corners[k][j] + static_cast<SReal>(2) * (center[j]     - corners[k][j]);
                            }
                        }
                    }
                }
                else if( k == 3 )
                {
                    SMALL_UNROLL
                    for( Int i = 0; i < VertexCount(); ++ i )
                    {
                        SMALL_UNROLL
                        for( Int j = 0; j < VertexCount(); ++j )
                        {
                            corners[i][j] = center[j] - static_cast<SReal>(2) * (corners[i][j] - center[j]);
                        }
                    }
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
