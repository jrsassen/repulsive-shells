#pragma once

#include <omp.h>
#include <tuple>

#include "Tools.h"
#include "MyMath.h"

#include "Tensors.h"

namespace Collision {
    
    const static int digits = 16;
    
    using namespace Tools;
    using namespace Tensors;
    
} // namespace Collision

#include "src/Collision/Types.h"

// Load primitives and GJK_Algorithm object.
#include "GJK.h"

//#include "src/Collision/MortonTree.h"

#include "src/Collision/Cluster.h"
//#include "src/Collision/BoundingVolumeHierarchy.h"
//#include "src/Collision/BlockHierarchy.h"


#include "src/Collision/SimplexHierarchy/SimplexHierarchy.h"
#include "src/Collision/SimplexHierarchy/SimplexHierarchy_1D.h"
#include "src/Collision/SimplexHierarchy/SimplexHierarchy_2D.h"

//namespace Repulsion {
//    
//    using namespace Tools;
//    using namespace Tensors;
//    using namespace Collision;
//    
//} // namespace Repulsion
