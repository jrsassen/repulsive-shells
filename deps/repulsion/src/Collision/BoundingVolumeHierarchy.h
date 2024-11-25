#pragma once

#define CLASS BoundingVolumeHierarchy

namespace Collision
{
    
    // TODO: Have to replace option structs by key-value lists + parsing.
    struct BoundingVolumeHierarchySettings
    {
        // An auxilliary data structure in order to feed and hold options for ClusterTree.
        int split_threshold = 8;
        int threads_available = 0;
        int thread_count = 0;             // 0 means: use all threads that you can get.
        int tree_thread_count = 0;        // 0 means: use all threads that you can get.
        
        BoundingVolumeHierarchySettings()
        {
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
        }
        
        BoundingVolumeHierarchySettings( const BoundingVolumeHierarchySettings & settings_ )
        {
            // When initialized by another instance of ClusterTreeSettings, we do a couple of sanity checks. In particular, this will be used in the initialization list of the constructor(s) of ClusterTree. This allows us to make the settings const for each instantiated ClusterTree.
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
            
            int one = 1;
            
            thread_count = (settings_.thread_count == 0) ? std::max( one, threads_available ) : std::max( one, settings_.thread_count );
            
            tree_thread_count = (settings_.tree_thread_count == 0) ? std::max( one, threads_available ) : std::max( one, settings_.tree_thread_count );
            
            split_threshold = std::max( static_cast<int>(1), settings_.split_threshold );
        }
        
        ~BoundingVolumeHierarchySettings(){}
        
    }; // BoundingVolumeHierarchySettings
    

    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS // binary cluster tree
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        
        using Primitive_T = PrimitiveSerialized<AMB_DIM,Real,Int,SReal>;
        using BoundingVolume_T = BoundingVolumeBase <AMB_DIM,Real,Int,SReal>;
        

#define OVERRIDE
#include "BoundingVolumeHierarchy/BoundingVolumeHierarchy_Details.h"
#undef OVERRIDE
        
    public:
        
        const BoundingVolumeHierarchySettings settings;
        
        CLASS() {}
        
        // To allow polymorphism, we require the user to create instances of the desired types for the primitives and the bounding volumes, so that we can Clone() them.
        CLASS(
            const Primitive_T & P_proto_, const Tensor2<SReal,Int> & P_serialized_,
            const BoundingVolume_T & C_proto_, const Tensor1<  Int,Int> & P_ordering_,
            BoundingVolumeHierarchySettings settings_ = BoundingVolumeHierarchySettings()
        ) :
            P_serialized(P_serialized_),
            P_ordering(P_ordering_),
            settings(settings_)
        {
            ptic(ClassName());
            
            P_proto = std::vector<std::unique_ptr<Primitive_T>> ( TreeThreadCount() );
            C_proto = std::vector<std::unique_ptr<BoundingVolume_T>> ( TreeThreadCount() );
            
            for( Int thread = 0; thread < TreeThreadCount(); ++thread )
            {
                P_proto[thread] = P_proto_.Clone();
                C_proto[thread] = C_proto_.Clone();
            }
            
            ComputeClusters();
            
            ptoc(ClassName());
        }
        
        // To allow polymorphism, we require the user to create instances of the desired types for the primitives and the bounding volumes, so that we can Clone() them.
        // A slightly more general interface allowing input of pointer arrays.
        CLASS(
            const Primitive_T & P_proto_,
            const SReal * const P_serialized_,
            const Int primitive_count_,
            const BoundingVolume_T & C_proto_,
            BoundingVolumeHierarchySettings settings_ = BoundingVolumeHierarchySettings()
        )
        :   P_serialized( P_serialized_, primitive_count_, P_proto_.Size() )
        ,   settings(settings_)
        {
            ptic(ClassName());
            
            Int thread_count = std::max( TreeThreadCount(), ThreadCount() );
            
            P_proto = std::vector<std::unique_ptr<Primitive_T>> ( thread_count );
            C_proto = std::vector<std::unique_ptr<BoundingVolume_T>> ( thread_count );
            
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                P_proto[thread] = P_proto_.Clone();
                C_proto[thread] = C_proto_.Clone();
            }
            
            ComputeClusters();
            
            ptoc(ClassName());
        }
        
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS) + "<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    };
    
} // namespace Collision

#undef CLASS



