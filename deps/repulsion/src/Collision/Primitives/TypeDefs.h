#pragma once

namespace Collision {
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    using Point = Polytope<1,AMB_DIM,Real,Int,SReal,ExtReal>;
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    using LineSegment = Polytope<2,AMB_DIM,Real,Int,SReal,ExtReal>;
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    using Triangle = Polytope<3,AMB_DIM,Real,Int,SReal,ExtReal>;
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    using Tetrahedron = Polytope<4,AMB_DIM,Real,Int,SReal,ExtReal>;
    
} // Collision
