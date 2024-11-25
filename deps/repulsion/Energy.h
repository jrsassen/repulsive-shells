#include "src/Repulsion/Energy/Energy__NearFieldKernel.h"
#include "src/Repulsion/Energy/Energy__NearFieldKernel_Adaptive.h"
#include "src/Repulsion/Energy/Energy__FarFieldKernel_FMM.h"
//
//// toggle whether cluster/primitive data should be copied by energy kernels.
////#define Energy_FarField_S_BH_Copy
//#define Energy_FarField_T_BH_Copy
//#include "src/Repulsion/Energy/Energy__FarFieldKernel_BH.h"
#include "src/Repulsion/Energy/EnergyBase.h"
#include "src/Repulsion/Energy/Energy_Restricted.h"
#include "src/Repulsion/Energy/Energy_Naive.h"
#include "src/Repulsion/Energy/Energy_FMM.h"
#include "src/Repulsion/Energy/Energy_FMM_Adaptive.h"
//#include "src/Repulsion/Energy/Energy_BH_BCT.h"
//#include "src/Repulsion/Energy/Energy_BH.h"

#include "src/Repulsion/Energy/ObstacleEnergy_FMM.h"
#include "src/Repulsion/Energy/ObstacleEnergy_FMM_Adaptive.h"


#include "src/Repulsion/TrivialEnergy/TrivialEnergy__NearFieldKernel.h"
#include "src/Repulsion/TrivialEnergy/TrivialEnergy__NearFieldKernel_Adaptive.h"
#include "src/Repulsion/TrivialEnergy/TrivialEnergy__FarFieldKernel_FMM.h"

#include "src/Repulsion/TrivialEnergy/TrivialEnergy_FMM_Adaptive.h"
#include "src/Repulsion/TrivialEnergy/TrivialObstacleEnergy_FMM_Adaptive.h"


#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy__NearFieldKernel.h"
#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy__FarFieldKernel_FMM.h"
#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy__NearFieldKernel_Adaptive.h"

#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy_Naive.h"
#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy_FMM.h"
#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy_FMM_Adaptive.h"
#include "src/Repulsion/TangentPointEnergy/TangentPointObstacleEnergy_FMM.h"
#include "src/Repulsion/TangentPointEnergy/TangentPointObstacleEnergy_FMM_Adaptive.h"

//#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy__FarFieldKernel_BH.h"
//#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy_BH_BCT_Adaptive.h"
//#include "src/Repulsion/TangentPointEnergy/TangentPointEnergy_BH_Adaptive.h"

