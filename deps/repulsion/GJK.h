#pragma once

#include "Tools.h"
#include "MyMath.h"
#include "Tensors.h"

#define GJK_STRINGIFY(a) #a

namespace Collision {
    
    static int GJK_digits = 16;
    
    using namespace Tools;
    using namespace Tensors;
    
} // namespace Collision


#ifdef GJK_Report
#define GJK_DUMP(x) DUMP(x);

#define GJK_print(x) logprint(x);

#define GJK_tic(x) ptic(x);

#define GJK_toc(x) ptoc(x);

#else

#define GJK_DUMP(x)

#define GJK_tic(x)

#define GJK_toc(x)

#define GJK_print(x)

#endif

#include "src/Collision/MortonCode.h"
//#include "src/Collision/HilbertCurve.h"

// These primitives are not serializable -- for a reason.
#include "src/Collision/Primitives/PrimitiveBase.h"
#include "src/Collision/Primitives/ConvexHull.h"

// Serializable primitives:
#include "src/Collision/Primitives/PrimitiveSerialized.h"
#include "src/Collision/Primitives/PolytopeBase.h"
#include "src/Collision/Primitives/PolytopeExt.h"
#include "src/Collision/Primitives/Polytope.h"
#include "src/Collision/Primitives/Ellipsoid.h"
#include "src/Collision/Primitives/Parallelepiped.h"

//#include "src/Collision/Primitives/SpaceTimePrism.h"

#include "src/Collision/Primitives/TypeDefs.h"

// Bounding volume types:
#include "src/Collision/BoundingVolumes/BoundingVolumeBase.h"
#include "src/Collision/BoundingVolumes/AABB.h"
#include "src/Collision/BoundingVolumes/AABB_LongestAxisSplit.h"
#include "src/Collision/BoundingVolumes/AABB_MedianSplit.h"
#include "src/Collision/BoundingVolumes/AABB_PreorderedSplit.h"
//#include "src/Collision/BoundingVolumes/OBB.h"                    // requires eigensolve
//#include "src/Collision/BoundingVolumes/OBB_MedianSplit.h"        // requires eigensolve
//#include "src/Collision/BoundingVolumes/OBB_PreorderedSplit.h"    // requires eigensolve


// Actual GJK algorithms
#include "src/Collision/GJK/GJK_Algorithm.h"
#include "src/Collision/GJK/GJK_Batch.h"
#include "src/Collision/GJK/GJK_Offset_Batch.h"


#include "src/Collision/Primitives/MovingPolytopeBase.h"
#include "src/Collision/Primitives/MovingPolytopeExt.h"
#include "src/Collision/Primitives/MovingPolytope.h"

#include "src/Collision/CollisionFinderBase.h"
#include "src/Collision/CollisionFinder.h"
