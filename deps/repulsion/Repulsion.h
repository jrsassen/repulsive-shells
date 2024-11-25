#pragma once

#include <tuple>
#include <unordered_map>

#include "Tools.h"
#include "MyMath.h"

#include "Tensors.h"
#include "Collision.h"




namespace Repulsion {
    using namespace Tools;
    using namespace Tensors;
    using namespace Collision;
    
    typedef double GJK_Real;
}

#include "src/Repulsion/Types.h"
#include "src/Repulsion/Settings.h"
#include "src/Repulsion/MultipoleMoments/MultipoleMoments.h"
#include "src/Repulsion/ClusterTreeBase.h"
#include "src/Repulsion/ClusterTree.h"
#include "src/Repulsion/BlockClusterTreeBase.h"
#include "src/Repulsion/BlockClusterTree.h"
#include "src/Repulsion/CollisionTreeBase.h"
#include "src/Repulsion/CollisionTree.h"

//#include "src/Repulsion/MeshBase.h"
#include "src/Repulsion/SimplicialMesh/SimplicialMeshBase.h"
#include "src/Repulsion/SimplicialMesh/SimplicialMeshDetails.h"
#include "src/Repulsion/SimplicialMesh/SimplicialMesh.h"
//#include "src/Repulsion/SimplicialMesh/DNear_to_Hulls_Details.h"
//#include "src/Repulsion/SimplicialMesh/DFar_to_Hulls_Details.h"
//#include "src/Repulsion/SimplicialMesh/ComputeNearFarData_Details.h"
//#include "src/Repulsion/SimplicialMesh/ComputeNearFarDataOps_Details.h"

// TODO: Finalize this!
//#include "src/Repulsion/SimplicialRemesher/CombinatorialVertex.h"
//#include "src/Repulsion/SimplicialRemesher/CombinatorialEdge.h"
//#include "src/Repulsion/SimplicialRemesher/CombinatorialSimplex.h"
//#include "src/Repulsion/SimplicialRemesher/SimplicialRemesher.h"

//#include "src/Repulsion/DKT_Mesh/DKT_Mesh.h"

// toggle whether primitive data should be copied by kernels.
#define NearField_S_Copy
#define NearField_T_Copy
#include "src/Repulsion/Kernels/NearFieldKernelBase.h"

// toggle whether cluster data should be copied by kernels.
#define FarField_S_Copy
#define FarField_T_Copy
#include "src/Repulsion/Kernels/FarFieldKernelBase_FMM.h"

#include "Energy.h"
#include "Metric.h"
