#pragma once

namespace Collision
{
    
    // TODO: Have to replace option structs by key-value lists + parsing.
    struct MortonTreeSettings
    {
        // An auxilliary data structure in order to feed and hold options for ClusterTree.
        mint split_threshold = 8;
        mint threads_available = 0;
        mint thread_count = 0;             // 0 means: use all threads that you can get.
        mint tree_thread_count = 0;        // 0 means: use all threads that you can get.
        
        MortonTreeSettings()
        {
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
        }
        
        MortonTreeSettings( const MortonTreeSettings & settings_ )
        {
            // When initialized by another instance of ClusterTreeSettings, we do a couple of sanity checks. In particular, this will be used in the initialization list of the constructor(s) of ClusterTree. This allows us to make the settings const for each instantiated ClusterTree.
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                threads_available = omp_get_num_threads();
            }
            
            mint one = 1;
            
            thread_count = (settings_.thread_count == 0) ? std::max( one, threads_available ) : std::max( one, settings_.thread_count );
            
            tree_thread_count = (settings_.tree_thread_count == 0) ? std::max( one, threads_available ) : std::max( one, settings_.tree_thread_count );
            
            split_threshold = std::max( static_cast<mint>(1), settings_.split_threshold );
        }
        
        ~MortonTreeSettings(){}
        
    }; // MortonTreeSettings
    
    
    

    template<mint AMB_DIM, typename Real, typename Int >
    class alignas( 2 * CACHE_LINE_WIDTH ) MortonTree // binary cluster tree
    {
        using Primitive_T = PrimitiveSerialized<AMB_DIM,Real,Int>;
        using BoundingVolume_T = AABB<AMB_DIM,Real,Int>;
        
//#include "BoundingVolumeHierarchy/BoundingVolumeHierarchy_Details.h"
        
    public:
        
        const MortonTreeSettings settings;
        
    protected:
        // "C_" stands for "cluster", "P_" stands for "primitive"
        
        // Each thread gets its own primitive prototype to avoid sharing conflicts.
        std::vector<std::unique_ptr<Primitive_T>> P_proto;           // False sharing prevented by alignment of PrimitiveBase.
        
        // Container for storing the serialized data of the primitives. Only meant to be accessed by primitive prototypes.
        // TODO: Make this private somehow?
        mutable Tensor2<Real,Int> P_serialized;
        
        // Each thread gets its own bounding volume prototype to avoid sharing conflicts.
        std::vector<std::unique_ptr<BoundingVolume_T>> C_proto;           // False sharing prevented by alignment of PrimitiveBase.
        
        // Container for storing the serialized data of the clusters. Only meant to be accessed by bounding volume prototypes.
        // TODO: Make this private somehow?
        mutable Tensor2<Real,Int> C_serialized;
        
    public:
        
        // Integer data for the combinatorics of the tree.
        
        Tensor1<Int,Int> P_ordering;           // Reordering of primitives; crucial for communication with outside world
        Tensor1<Int,Int> P_inverse_ordering;   // Inverse ordering of the above; crucial for communication with outside world

        Tensor1<Int,Int> C_begin;
        Tensor1<Int,Int> C_end;
        Tensor1<Int,Int> C_depth;
        Tensor1<Int,Int> C_next;
        Tensor1<Int,Int> C_left;  // list of index of left  children;  entry is -1 if no child is present
        Tensor1<Int,Int> C_right; // list of index of right children; entry is -1 if no child is present

        Int depth;
        
        
        MortonTree(){};
        
        // To allow polymorphism, we require the user to create instances of the desired types for the primitives and the bounding volumes, so that we can Clone() them.
        MortonTree(
            const Primitive_T & P_proto_,
            const Tensor2<Real,Int> & P_serialized_,
            const BoundingVolume_T & C_proto_,
            const Tensor1<Int,Int> & P_ordering_,
            MortonTreeSettings settings_ = MortonTreeSettings()
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
            
            MortonOrdering();
            
            ptoc(ClassName());
        }

        ~MortonTree() = dafault;
        
    protected:
        
        void MortonOrdering()
        {
            const Real * restrict const p = P_serialized.data()+1;
            const Int P_size = P_proto[0]->Size();
            const Int C_size = C_proto[0]->Size();
            
            auto job_ptr = BalanceWorkLoad<Int>( TreeThreadCount(), PrimitiveCount() );
            
            auto AABB_buffer  = Tensor2<Real,Int> ( TreeThreadCount(), C_size + 2 * CACHE_LINE_WIDTH );
            
            // Compute local bounding boxes in parallel
            #pragma omp parallel for num_threads( TreeThreadCount() )
            for( Int thread = 0; thread < TreeThreadCount() )
            {
                Real x [AMB_DIM];
                
                auto * P = P_proto[thread];
                auto * C = C_proto[thread];
                
                
                Int i_begin = job_ptr[thread];
                Int i_end   = job_ptr[thread+1];
                
                C->SetPointer( AABB_buffer.data(thread) );
                C->FromPrimitives( *P, P_serialized.data(), i_begin, i_end );
            }
            
            // Merge local bounding boxes (if there is more than one thread).
            if(TreeThreadCount() > 1)
            {
                
                BoundingVolume_T * C = C_proto[0];
                C->SetPointer( AABB_buffer.data(0) );
                
                for( Int thread = 1; thread < TreeThreadCount(); ++thread )
                {
                    C->Merge( AABB_buffer.data(thread) );
                }
            }
                
            // Now we have the global bounding box stored in AABB_buffer.data(0). Next we make it a cube.
            Real L = AABB_buffer[ 1 + AMB_DIM ];
            for( Int k = 1; k < AMB_DIM; ++ k)
            {
                L = std::max( L, AABB_buffer[ 1 + AMB_DIM + k ] );
            }
            for( Int k = 0; k < AMB_DIM; ++ k)
            {
                AABB_buffer[ 1 + AMB_DIM + k ] = L;
            }
                
            // Copy this bounding cube to where each thread will find it.
            if(TreeThreadCount() > 1)
            {
                BoundingVolume_T * C = C_proto[0];

                C->SetPointer( AABB_buffer.data(0) );
                        
                for( Int thread = 1; thread < TreeThreadCount(); ++thread )
                {
                    C->Write( AABB_buffer.data(thread) );
                }
            }
            
            
            auto z = Tensor1<uint64_t,Int>( PrimitiveCount() );
            
            #pragma omp parallel for num_threads( TreeThreadCount() )
            for( Int thread = 0; thread < TreeThreadCount() )
            {
                auto * P = P_proto[thread];
                auto * C = C_proto[thread];
                
                C->SetPointer( AABB_buffer.data(thread) );
                
                C->MortonCodes( *P, P_serialized.data(), job_ptr[thread], job_ptr[thread+1], z.data() );
            }
            
            P_ordering = iota( PrimitiveCount() );
            
            const uint64_t * restrict const z_ptr = z.data();
            std::sort(
                P_ordering.data(),
                P_ordering.data() + PrimitiveCount(),
                [z_ptr]( const Int & i, const Int & j ){ return z_ptr[i] < z_ptr[j]; }
            );
        }
        
        
        
    public:
        
        static constexpr Int AmbDim()
        {
            return AMB_DIM;
        }
        
        Int ThreadCount() const
        {
            return settings.thread_count;
        }
        
        Int TreeThreadCount() const
        {
            return settings.tree_thread_count;
        }
        
        Int SplitThreshold() const
        {
            return settings.split_threshold;
        }

        Int TreeDepth() const
        {
            return depth;
        }
                
        Int PrimitiveCount() const
        {
            return P_serialized.Dimension(0);
        }
        
        Int PrimitiveSize() const
        {
            return P_serialized.Dimension(1);
        }
        
        Int ClusterCount() const
        {
            return C_serialized.Dimension(0);
        }
        
        Int ClusterSize() const
        {
            return C_serialized.Dimension(1);
        }
        
//        Int LeafClusterCount() const
//        {
//            return leaf_clusters.Dimension(0);
//        }
//
//        const Tensor1<Int,Int> & LeafClusters() const
//        {
//            return leaf_clusters;
//        }
//
//        const Tensor1<Int,Int> & LeafClusterLookup() const
//        {
//            return leaf_cluster_lookup;
//        }
//
//        const Tensor1<Int,Int> & LeafClusterPointers() const
//        {
//            return leaf_cluster_ptr;
//        }
        
        const Tensor1<Int,Int> & PrimitiveOrdering() const
        {
            return P_ordering;
        }
        
        const Tensor1<Int,Int> & PrimitiveInverseOrdering() const
        {
            return P_inverse_ordering;
        }
        
        const Tensor1<Int,Int> & ClusterBegin() const
        {
            return C_begin;
        }

        const Tensor1<Int,Int> & ClusterEnd() const
        {
            return C_end;
        }
        
        const Tensor1<Int,Int> & ClusterLeft() const
        {
            return C_left;
        }
        
        const Tensor1<Int,Int> & ClusterRight() const
        {
            return C_right;
        }

        const Tensor1<Int,Int> & ClusterDepths() const
        {
            return C_depth;
        }
                
        Tensor2<Real,Int> & ClusterSerializedData() const
        {
            return C_serialized;
        }
        
        const BoundingVolume_T * ClusterPrototype() const
        {
            return C_proto[0];
        }
        
        Tensor2<Real,Int> & PrimitiveSerializedData() const
        {
            return P_serialized;
        }
        
        const Primitive_T * PrimitivePrototype() const
        {
            return P_proto[0];
        }
        
        
        static std::string ClassName()
        {
            return "MortonTree<"+TypeName<Real>::Get()+","+ToString(AMB_DIM)+">";
        }
    
    }; //MortonTree
    
} // namespace Collision
