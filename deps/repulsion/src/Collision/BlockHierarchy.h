#pragma once

#define CLASS BlockHierarchy

namespace Collision
{
    
    struct BlockHierarchySettings
    {
        // handling symmetry properties of the BCT
        bool exploit_symmetry = true;
        
        bool near_upper_triangular = false;
        bool  far_upper_triangular = true;
        
        bool  far_fieldQ = true;
        bool near_fieldQ = true;
        // If exploit_symmetry == false, S == T is assumed and only roughly half the block clusters are generated during the split pass performed by Split.
        // If upper_triangular == true and if exploit_symmetry == true, only the upper triangle of the interaction matrices will be generated. => Split will be faster.
        // If exploit_symmetry == true and upper_triangular == false then the block cluster twins are generated _at the end_ of the splitting pass by Split.
        // CAUTION: Currently, we have no faster matrix-vector multiplication for upper triangular matrices, so upper_triangular == false is the default.

        
        int threads_available = 0;    // thread_count <= 0 means that we use all threads that we can get hold of
        
        int thread_count = 0;    // thread_count <= 0 means that we use all threads that we can get hold of
        
        double separation_parameter = 0.5;
        
        BlockSplitMethod block_split_method = BlockSplitMethod::Parallel;
        
        BlockHierarchySettings()
        {
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
        }
        
        ~BlockHierarchySettings(){};
        
        BlockHierarchySettings( const BlockHierarchySettings & settings_ )
        {
            // When initialized by another instance of BlockHierarchySettings, we do a couple of sanity checks. In particular, this will be used in the initialization list of the constructor(s) of BlockHierarchy. This allows us to make the settings const for each instantiated BlockHierarchy.
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
            
            separation_parameter = std::max(0., settings_.separation_parameter );
                        
            int one = 1;
            
            thread_count = (settings_.thread_count == 0) ? std::max( one, threads_available ) : std::max( one, settings_.thread_count );
            
            far_fieldQ  = settings_.far_fieldQ;
            near_fieldQ = settings_.near_fieldQ;
        }
    };

    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        
        using SETTING_T           = BlockHierarchySettings;
        using CLUSTER_TREE_BASE_T = BoundingVolumeHierarchy<AMB_DIM,Real,Int,SReal>;
        using ClusterTree_T      = BoundingVolumeHierarchy<AMB_DIM,Real,Int,SReal>;
        using BoundingVolume_T           = BoundingVolumeBase     <AMB_DIM,Real,Int,SReal>;
        using GJK_T               = GJK_Algorithm<AMB_DIM,Real,Int>;
        
        using NEAR_T  = SparseBinaryMatrixCSR<Int>;
        using  FAR_T  = SparseBinaryMatrixCSR<Int>;
        
#define OVERRIDE
#include "BlockHierarchy/BlockHierarchy_Details.h"
#undef OVERRIDE
        
        public :
            
        CLASS( const ClusterTree_T & S_, const ClusterTree_T & T_, SETTING_T settings_ = SETTING_T() )
            :
                S(S_),
                T(T_),
                settings(settings_),
                theta2( settings_.separation_parameter * settings_.separation_parameter ),
                thread_count( std::min( std::min(S_->TreeThreadCount(), T_->TreeThreadCount() ), settings.thread_count ) ),
                is_symmetric( std::addressof(S_) == std::addressof(T_) ),
                exploit_symmetry      ( is_symmetric && settings.exploit_symmetry      ),
                near_upper_triangular ( is_symmetric && settings.near_upper_triangular ),
                far_upper_triangular  ( is_symmetric && settings.far_upper_triangular  )
            {
                ptic(ClassName());
                
                valprint("ThreadCount()",ThreadCount());
                
                PreparePrototypes();
                
                ComputeBlocks();

//                RequireMetricSparsityPatterns();
                
                ptoc(ClassName());
            }; // Constructor
        
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS) + "<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    };
    
} //namespace Collision

#undef CLASS
