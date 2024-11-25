#pragma once

namespace Repulsion
{
    struct ClusterTreeSettings
    {
        // An auxilliary data structure in order to feed and hold options for ClusterTree.
        int split_threshold   = 1;
        int threads_available = 0;
        int thread_count      = 0;        // 0 means: use all threads that you can get.
        
//        TreePercolationAlgorithm tree_perc_alg = TreePercolationAlgorithm::Tasks;
//        TreePercolationAlgorithm tree_perc_alg = TreePercolationAlgorithm::Sequential;
        TreePercolationAlgorithm tree_perc_alg = TreePercolationAlgorithm::Recursive;

        BoundingVolumeType bounding_volume_type = BoundingVolumeType::AABB;
        
        ClusterTreeSettings()
        {
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
        }
        
        ClusterTreeSettings( const ClusterTreeSettings & settings_ )
        {
            // When initialized by another instance of ClusterTreeSettings, we do a couple of sanity checks. In particular, this will be used in the initialization list of the constructor(s) of ClusterTree. This allows us to make the settings const for each instantiated ClusterTree.
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
            
            int one = 1;
            
            thread_count = (settings_.thread_count == 0) ? std::max( one, threads_available ) : std::max( one, settings_.thread_count );
            
            split_threshold = std::max( static_cast<int>(1), settings_.split_threshold );
            
            tree_perc_alg = settings_.tree_perc_alg;
        }
        
        ~ClusterTreeSettings(){}
        
    }; // ClusterTreeSettings
    
    ClusterTreeSettings ClusterTreeDefaultSettings = ClusterTreeSettings();
    
    struct BlockClusterTreeSettings
    {
        // handling symmetry properties of the BCT
        bool exploit_symmetry = true;

        bool near_upper_triangular = true;
        bool  far_upper_triangular = true;
        
//        bool near_upper_triangular = false;
//        bool  far_upper_triangular = false;
        

        // If exploit_symmetry == false, S == T is assumed and only roughly half the block clusters are generated during the split pass performed by RequireBlockClusters.
        // If upper_triangular == true and if exploit_symmetry == true, only the upper triangle of the interaction matrices will be generated. --> RequireBlockClusters will be faster.
        // If exploit_symmetry == true and upper_triangular == false then the block cluster twins are generated _at the end_ of the splitting pass by RequireBlockClusters.
        // CAUTION: Currently, we have no faster matrix-vector multiplication for upper triangular matrices, so upper_triangular == false is the default.
        
        int threads_available = 1;
        
        double  far_field_separation_parameter = 0.5;
        double near_field_separation_parameter = 10.;
        double near_field_collision_parameter = 10000000000.;
        
        BlockSplitMethod block_split_method = BlockSplitMethod::Parallel;
        
        BlockClusterTreeSettings()
        {
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
        }
        
        ~BlockClusterTreeSettings(){};
        
        BlockClusterTreeSettings( const BlockClusterTreeSettings & settings_ )
        {
            // When initialized by another instance of BlockClusterTreeSettings, we do a couple of sanity checks. In particular, this will be used in the initialization list of the constructor(s) of BlockClusterTree. This allows us to make the settings const for each instantiated BlockClusterTree.
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }

          threads_available = (settings_.threads_available == 0) ? std::max( 1, threads_available ) : std::max( 1, settings_.threads_available );
            
             far_field_separation_parameter = std::max(0., settings_.far_field_separation_parameter );
            near_field_separation_parameter = std::max(0., settings_.near_field_separation_parameter );
                        
//            int one = 1;
            
//            far_field  = settings_.far_field;
//            near_field = settings_.near_field;
        }
    };
    
    BlockClusterTreeSettings BlockClusterTreeDefaultSettings = BlockClusterTreeSettings();
    
    struct AdaptivitySettings
    {
        int  max_level = 20;
        int  min_level = 0;
    //    double theta = 10.;
        double theta = 1.;
        double intersection_theta = 10000000000.;
        
        AdaptivitySettings() {}
        
        ~AdaptivitySettings(){}
    };

    AdaptivitySettings AdaptivityDefaultSettings = AdaptivitySettings();

} // namespace Repulsion
