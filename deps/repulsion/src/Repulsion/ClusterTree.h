#pragma once

#define CLASS ClusterTree
#define BASE  ClusterTreeBase<Real,Int,SReal,ExtReal>

#define REPULSION_CLUSTERTREE

namespace Repulsion
{
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:
        
        using       Primitive_T =       PolytopeBase<AMB_DIM,GJK_Real,Int,SReal>;
        using MovingPrimitive_T = MovingPolytopeBase<AMB_DIM,GJK_Real,Int,SReal>;
                                  
        using BoundingVolume_T  = AABB<AMB_DIM,GJK_Real,Int,SReal>;
        
        using ClusterTreeBase_T = BASE;
        
        using DataContainer_T   = typename BASE::DataContainer_T;
        using BufferContainer_T = typename BASE::BufferContainer_T;
     
        // In principle, ThreadTensor3<Real,Int> should have better scaling on multiple socket machines, because I tried to encourages that the thread-local arrays are allocated on local RAM. -- On my tiny Quad Core however, it performs a bit _WORSE_ than Tensor3<Real,Int>.
        using DerivativeContainer_T = typename BASE::DerivativeContainer_T;
        
        CLASS()
        : P_proto (1)
        , C_proto (1)
        , P_moving(1)
        {}
        
        // To allow polymorphism, we require the user to create instances of the desired types for the primitives and the bounding volumes, so that we can Clone() them.
        CLASS(
            const Primitive_T        & P_proto_,
            const Tensor2<SReal,Int> & P_serialized_,
            const BoundingVolume_T   & C_proto_,
            const Tensor1<Int ,Int>  & P_ordering_,
            const Tensor2<Real,Int>  & P_near_, // data used actual interaction computation; assumed to be of size PrimitiveCount() x NearDim(). For a triangle mesh in 3D, we want to feed each triangles i), area ii) barycenter and iii) normal as a 1 + 3 + 3 = 7 vector
            const Tensor2<Real,Int>  & P_far_, // data used actual interaction computation; assumed to be of size PrimitiveCount() x FarDim(). For a triangle mesh in 3D, we want to feed each triangles i), area ii) barycenter and iii) orthoprojector onto normal space as a 1 + 3 + 6 = 10 vector
            SparseMatrixCSR<Real,Int> & DiffOp,
            SparseMatrixCSR<Real,Int> & AvOp,
            ClusterTreeSettings settings_ = ClusterTreeDefaultSettings
        )
        :   BASE(settings_)
        ,   P_proto ( ThreadCount() )
        ,   C_proto ( ThreadCount() )
        ,   P_moving( ThreadCount() )
        {
            ptic(ClassName());
            
            P_serialized = P_serialized_;
            P_ordering   = P_ordering_;
            
//            P_proto  = std::vector<std::unique_ptr<      Primitive_T>> ( ThreadCount() );
//            P_moving = std::vector<std::unique_ptr<MovingPrimitive_T>> ( ThreadCount() );
//            C_proto  = std::vector<std::unique_ptr<BoundingVolume_T>>  ( ThreadCount() );
            
            #pragma omp parallel for num_threads( ThreadCount() )
            for( Int thread = 0; thread < ThreadCount(); ++thread )
            {
                P_proto[thread] = P_proto_.Clone();
                
                C_proto[thread] = C_proto_.Clone();
            }
            
            this->ComputeClusters();
            
            if( P_near_.Dimension(0) != PrimitiveCount() )
            {
                eprint(ClassName()+" : P_near_.Dimension(0) != PrimitiveCount()");
                return;
            }
            if( P_far_.Dimension(0) != PrimitiveCount() )
            {
                eprint(ClassName()+" : P_far_.Dimension(0) != PrimitiveCount()");
                return;
            }
    
            this->AllocateNearFarData( P_near_.Dimension(1), P_far_.Dimension(1) );

            this->ComputePrimitiveData( P_near_.data(), P_far_.data() );

            this->ComputeClusterData();
            
            this->ComputePrimitiveToClusterMatrix();
            
            this->ComputePrePost( DiffOp, AvOp );
            
            ptoc(ClassName());
        }
        
        virtual ~CLASS() override = default;

    
    public:
        
        using BASE::SplitThreshold;
        using BASE::ThreadCount;
        using BASE::PrimitiveCount;
        using BASE::ClusterCount;
        using BASE::LeafClusterCount;
        using BASE::RequireClusterMoments;
        
    protected:
        
        using BASE::P_ordering;
        using BASE::P_inverse_ordering;
        using BASE::P_score_buffer;
        using BASE::P_perm_buffer;
        using BASE::P_serialized;
        using BASE::P_updated_serialized;
        using BASE::P_velocities_serialized;
        using BASE::P_near;
        using BASE::P_D_near;
        using BASE::P_far;
        using BASE::P_D_far;
        using BASE::thread_P_D_near;
        using BASE::P_moments;
        using BASE::P_in;
        using BASE::P_out;
        
        using BASE::C_begin;
        using BASE::C_end;
        using BASE::C_depth;
        using BASE::C_right;
        using BASE::C_left;
        using BASE::C_next;
        using BASE::C_serialized;
        using BASE::C_updated_serialized;
        using BASE::C_thread_serialized;
        using BASE::C_far;
        using BASE::thread_C_D_far;
        using BASE::C_moments;
        using BASE::C_in;
        using BASE::C_out;
        
        using BASE::leaf_clusters;
        using BASE::leaf_cluster_ptr;
        using BASE::leaf_cluster_lookup;
        using BASE::thread_cluster_counter;

        using BASE::depth;
        using BASE::settings;
        
//        using BASE::stack_array;
//        using BASE::queue_array;
        
        using BASE::C_to_P;
        using BASE::P_to_C;
        using BASE::hi_pre;
        using BASE::lo_pre;
        using BASE::hi_post;
        using BASE::lo_post;
        
        
        // Each thread gets its own primitive prototype to avoid sharing conflicts.
        std::vector<std::unique_ptr<Primitive_T>> P_proto;
        // False sharing prevented by alignment of PrimitiveBase.
        
        // Each thread gets its own bounding volume prototype to avoid sharing conflicts.
        std::vector<std::unique_ptr<BoundingVolume_T>> C_proto;
        std::vector<std::unique_ptr<BoundingVolume_T>> C_proto_updated;
        // False sharing prevented by alignment of PrimitiveBase.
        
        mutable std::vector<std::unique_ptr<MovingPrimitive_T>> P_moving;
        
    public:

        mutable std::vector<std::unique_ptr<MultipoleMomentsBase<Real,Int>>> M_ker;
        mutable Int moment_degree = 0;
        
        
    private:

        void ComputeClusters()
        {
            ptic(ClassName()+"::ComputeClusters");
            
            // Request some temporary memory for threads.
            
    //        P_ordering         = iota<Int>   ( PrimitiveCount() );
            P_inverse_ordering = Tensor1<Int,Int>( PrimitiveCount() );
            
            // Padding every row to prevent false sharing.
            thread_cluster_counter = Tensor2<Int,Int>( ThreadCount(), 2 * ALIGN, 0 );
            
            C_thread_serialized = Tensor3<SReal,Int>( ThreadCount(), 2*PrimitiveCount(), C_proto[0]->Size() );
            
            const Int thread = 0;
            
            ++thread_cluster_counter(thread,0);
            
            auto * root = new Cluster<Int>( thread, 0, 0, PrimitiveCount(), 0 );
                            
            // Initial bounding volume of root node.
            C_proto[thread]->SetPointer( C_thread_serialized.data(thread), 0 );
            C_proto[thread]->FromPrimitives( *P_proto[thread], P_serialized.data(), 0, PrimitiveCount(), ThreadCount() );
            
            Split( root );
            
            Serialize( root );
            
            delete root;
            
            // Free memory for threads.
            C_thread_serialized = Tensor3<SReal,Int>();
            
            thread_cluster_counter = Tensor2<Int,Int>();
                
            ptoc(ClassName()+"::ComputeClusters");
        }
        
        void Split( Cluster<Int> * root )
        {
            ptic(ClassName()+"::Split");

            P_score_buffer = Tensor1<SReal,Int>( PrimitiveCount() );
            for( Int i = 0; i < PrimitiveCount(); ++i )
            {
                P_score_buffer[i] = static_cast<Real>(i);
            }
            P_perm_buffer = Tensor1<Int,Int>( PrimitiveCount() );
            
            #pragma omp parallel num_threads( ThreadCount() ) shared( root )
            {
                #pragma omp single nowait
                {
                    split( root, ThreadCount() );
                }
            }
            
            P_perm_buffer  = Tensor1<Int,Int>();
            P_score_buffer = Tensor1<SReal,Int>();
            
            ptoc(ClassName()+"::Split");
        } // Split
        
        void split( Cluster<Int> * C, const Int free_thread_count )
        {
            const Int thread = omp_get_thread_num();
            
            const Int begin = C->begin;
            const Int end   = C->end;
            
            const Int  left_ID = thread_cluster_counter(thread,0)+1;
            const Int right_ID = thread_cluster_counter(thread,0)+2;
            
            if( end - begin > SplitThreshold() )
            {
                // TODO: Many things to do here:
                // Split finds a nice split of the cluster and reorders the primitives begin,...,end-1 so that
                // primtives begin,...,split_index-1 belong to left  new cluster
                // primtives split_index-1,...,end-1 belong to right new cluster
                // Split has to return a number split_index <= begin if it is not successful and a value begin < split_index < end otherwise.
                // Split is also responsible for computing the bounding volumes of the children, if successful.
                // Remark: Some bounding volume types, e.g., AABBs can use some information from the Split pass to compute the children's bounding volumes. This is why we merge the splitting pass with the computation of the children's bounding columes.
                
                // Remark: Make sure that bounding volumes are already computed for the child clusters. Moreover, we want that the serialized data is stored in the thread's storage that _created_ the new clusters. This is why we do NOT compute the bounding volumes at the beginning of Split; C is possibly created by another thread and we _must not_ write to that thread's memory.

                const Int split_index = C_proto[thread]->Split(
                    *P_proto[thread],                                   // prototype for primitves
                    P_serialized.data(), begin, end,                    // which primitives are in question
                    P_ordering.data(),                                  // which primitives are in question
                    C_thread_serialized.data(C->thread),    C->ID,      // where to get   the bounding volume info for current cluster
                    C_thread_serialized.data(   thread),  left_ID,      // where to store the bounding volume info for left  child (if successful!)
                    C_thread_serialized.data(   thread), right_ID,      // where to store the bounding volume info for right child (if successful!)
                    P_score_buffer.data(),                              // some scratch space for storing local scores
                    P_perm_buffer.data(),                               // scratch space for storing local permutations
                    P_inverse_ordering.data(),                          // abusing P_inverse_ordering as scratch space for storing inverses of local permutations
                    free_thread_count
                );
                
                if( (begin < split_index) && (split_index < end) )
                {
                    // create new nodes...
                    thread_cluster_counter(thread,0) += 2;
                    // We use raw pointers for performance reasons because we want to delete Cluster instances in the parallel serialization pass.
                    C->left  = new Cluster<Int> ( thread,  left_ID, begin,       split_index, C->depth+1 );
                    C->right = new Cluster<Int> ( thread, right_ID, split_index, end,         C->depth+1 );

                    // ... and split them in parallel
                    #pragma omp task final(free_thread_count<1)
                    {
                        split( C->left, free_thread_count/2 );
                    }
                    #pragma omp task final(free_thread_count<1)
                    {
                        split( C->right, free_thread_count - free_thread_count/2 );
                    }
                    #pragma omp taskwait
                    
                    // collecting statistics for the later serialization
                    // counting ourselves as descendant, too!
                    C->descendant_count = 1 + C->left->descendant_count + C->right->descendant_count;
                    C->descendant_leaf_count = C->left->descendant_leaf_count + C->right->descendant_leaf_count;
                    C->max_depth = std::max( C->left->max_depth, C->right->max_depth );
                }
                else
                {
    //                wprint(ClassName()+"::split : Failed to split cluster. Creating leaf node with "+ToString(end-begin)+" primitives.");
                    // count cluster as leaf cluster
                    // counting ourselves as descendant, too!
                    C->descendant_count = 1;
                    C->descendant_leaf_count = 1;
                }
            }
            else
            {
                // count cluster as leaf cluster
                // counting ourselves as descendant, too!
                C->descendant_count = 1;
                C->descendant_leaf_count = 1;
                return;
            }
        } //split


        void Serialize( Cluster<Int> * const root )
        {
            ptic(ClassName()+"::Serialize");
            
    //            tree_max_depth = root->max_depth;
    //
            // We have to allocated these two arrays first, so that ClusterCount() and LeafClusterCount() return correct results.
            C_serialized  = Tensor2<SReal,Int>( root->descendant_count, C_proto[0]->Size() );
            leaf_clusters = Tensor1<Int,Int>( root->descendant_leaf_count );
            
            #pragma omp parallel
            {
                #pragma omp single nowait
                {
                    #pragma omp task
                    {
                        C_left = Tensor1<Int,Int>( ClusterCount() );
                    }
                    #pragma omp task
                    {
                        C_right = Tensor1<Int,Int>( ClusterCount() );
                    }
                    #pragma omp task
                    {
                        C_begin = Tensor1<Int,Int>( ClusterCount());
                    }
                    #pragma omp task
                    {
                        C_end = Tensor1<Int,Int>( ClusterCount() );
                    }
                    #pragma omp task
                    {
                        C_depth = Tensor1<Int,Int>( ClusterCount() );
                    }
                    #pragma omp task
                    {
                        C_next = Tensor1<Int,Int>( ClusterCount() );
                    }
                    #pragma omp task
                    {
                        leaf_cluster_lookup = Tensor1<Int,Int>( ClusterCount(), -1 );
                    }
//                    #pragma omp task
//                    {
//                        queue_array = Tensor1<Int,Int>( ClusterCount() );
//                    }
                    #pragma omp taskwait
                }
            }

            #pragma omp parallel num_threads( ThreadCount() )
            {
                #pragma omp single nowait
                {
                    serialize( root, 0, 0, ThreadCount() );
                }
            }

            depth = root->max_depth;
//            stack_array = Tensor1<Int,Int>( 2 * depth + 1 );
            
            // It is quite certainly _NOT_ a good idea to parallelize this loop (false sharing!).
            const Int last = PrimitiveCount();
    //            #pragma omp parallel for num_threads( ThreadCount() )
            for( Int i = 0; i < last; ++i )
            {
                P_inverse_ordering[P_ordering[i]] = i;
            }
            
            leaf_cluster_ptr = Tensor1<Int,Int> ( LeafClusterCount() + 1 );
            leaf_cluster_ptr[0] = 0;

            // This loop is probably too short to be parallelized.
    //            #pragma omp parallel for num_threads( ThreadCount() )
            for( Int i = 0; i < LeafClusterCount(); ++i )
            {
                leaf_cluster_ptr[ i + 1 ] = C_end[leaf_clusters[i]];
            }
            
            
            ptoc(ClassName()+"::Serialize");
            
        } // Serialize
        
        void serialize( Cluster<Int> * const C, const Int ID, const Int leaf_before_count, const Int free_thread_count )
        {
            const Int thread = omp_get_thread_num();
            
            // enumeration in depth-first order
            C_begin[ID] = C->begin;
            C_end  [ID] = C->end;
            C_depth[ID] = C->depth;
            C_next [ID] = ID + C->descendant_count;
            
            // TODO: Potentially, some write conflicts could occur here. But they should be seldom as we use depth-first order.
            C_proto[thread]->SetPointer( C_thread_serialized.data(C->thread), C->ID );
            C_proto[thread]->Write( C_serialized.data(), ID );
            
            if( ( C->left != nullptr ) && ( C->right != nullptr ) )
            {
                C_left [ID] = ID + 1;
                C_right[ID] = ID + 1 + C->left->descendant_count;
        //
                #pragma omp task final(free_thread_count<1)
                {
                    serialize( C->left, C_left[ID], leaf_before_count, free_thread_count/2 );
                }
                #pragma omp task final(free_thread_count<1)
                {
                    serialize( C->right, C_right[ID], leaf_before_count + C->left->descendant_leaf_count, free_thread_count - free_thread_count/2 );
                }
                #pragma omp taskwait
                
                // Cleaning up after ourselves to prevent a destructore cascade.
                delete C->left;
                delete C->right;
            }
            else
            {
                C_left [ID] = -1;
                C_right[ID] = -1;
                
                leaf_clusters[leaf_before_count] = ID;
                leaf_cluster_lookup[ID] = leaf_before_count;
            }
        } //serialize
            
    public:
        
        Int AmbDim() const override
        {
            return AMB_DIM;
        }
        
        const BoundingVolume_T & ClusterPrototype() const
        {
            return *C_proto[0];
        }
        
        const Primitive_T & PrimitivePrototype() const
        {
            return *P_proto[0];
        }
        
        const MovingPrimitive_T & MovingPrimitivePrototype() const
        {
            return *P_moving[0];
        }
        
        Int NearDim() const override
        {
            return P_near.Dimension(1);
        }
        
        Int FarDim() const override
        {
            return 1 + AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2;
        }
        
        
    private:
            
        void AllocateNearFarData( const Int near_dim, const Int far_dim ) // reordering and computing bounding boxes
        {
            ptic(ClassName()+"::AllocateNearFarData");
            
            P_near   = DataContainer_T( PrimitiveCount(), near_dim );
            P_D_near = DataContainer_T( PrimitiveCount(), near_dim );
            P_far    = DataContainer_T( PrimitiveCount(), far_dim  );
            P_D_far  = DataContainer_T( PrimitiveCount(), far_dim  );
            
            thread_P_D_near = DerivativeContainer_T( ThreadCount(), PrimitiveCount(), near_dim );
            
            C_far    = DataContainer_T( ClusterCount(), far_dim );
            thread_C_D_far  = DerivativeContainer_T( ThreadCount(), ClusterCount(), far_dim );
            
            ptoc(ClassName()+"::AllocateNearFarData");
        }
        
        void ComputePrimitiveData( const Real * restrict const P_near_, const Real * restrict const P_far_ )
        {
            ptic(ClassName()+"::ComputePrimitiveData");
            
            const Int primitive_count = PrimitiveCount();
            const Int near_dim = NearDim();
            const Int  far_dim =  FarDim();

            const Int * restrict const o = P_ordering.data();
            
            Real * restrict const P_near__ = P_near.data();
            Real * restrict const P_far__  = P_far.data();
            
            const size_t near_size = sizeof(Real) * near_dim;
            const size_t  far_size = sizeof(Real) *  far_dim;
            
            #pragma omp parallel for num_threads( ThreadCount() ) shared( P_near__, P_far__, P_near_, P_far_, o, near_size, far_size ) schedule( static )
            for( Int i = 0; i < primitive_count; ++i )
            {
                const Int j = o[i];

                std::memcpy( &P_near__[near_dim * i], &P_near_[near_dim * j], near_size );

                std::memcpy(  &P_far__[ far_dim * i],  &P_far_[ far_dim * j],  far_size );
            }
            
            ptoc(ClassName()+"::ComputePrimitiveData");
        } //ComputePrimitiveData

        
        void ComputeClusterData()
        {
            ptic(ClassName()+"::ComputeClusterData");
            
            // using the already serialized cluster tree
            #pragma omp parallel num_threads( ThreadCount() )
            {
                #pragma omp single nowait
                {
                    computeClusterData( 0, ThreadCount() );
                }
            }
            ptoc(ClassName()+"::ComputeClusterData");
        }; //ComputeClusterData


        void computeClusterData( const Int C, const Int free_thread_count ) // helper function for ComputeClusterData
        {
            const Int far_dim = FarDim();
            const Int L = C_left [C];
            const Int R = C_right[C];
            
            Real * restrict const C_C = C_far.data(C);
            
            if( L >= 0 && R >= 0 ){
                //C points to interior node.
                #pragma omp task final(free_thread_count<1)  shared( L )
                {
                    computeClusterData( L, free_thread_count/2 );
                }
                #pragma omp task final(free_thread_count<1)  shared( R )
                {
                    computeClusterData( R, free_thread_count-free_thread_count/2 );
                }
                #pragma omp taskwait

                const Real * restrict const C_L = C_far.data(L);
                const Real * restrict const C_R = C_far.data(R);
                
                Real L_weight = C_L[0];
                Real R_weight = C_R[0];
                
                const Real C_mass = L_weight + R_weight;
                C_C[0] = C_mass;
                const Real C_invmass = static_cast<Real>(1)/C_mass;
                
                L_weight *= C_invmass;
                R_weight *= C_invmass;
                
//                #pragma omp simd
                for( Int k = 1; k < far_dim; ++k )
                {
                    C_C[k] = L_weight * C_L[k] + R_weight * C_R[k] ;
                }
            }
            else
            {
                //C points to leaf node.
                //compute from primitives
                const Int begin = C_begin[C];
                const Int end   = C_end  [C];
                
                memset( C_C, 0, far_dim * sizeof(Real) );
                
                for( Int i = begin; i < end; ++i )
                {
                    const Real * restrict const P = P_far.data(i);
                    
                    const Real a = P[0];
                    
                    C_C[0] += a;
                    
                    for( Int j = 1; j < far_dim; ++j )
                    {
                        C_C[j] += a * P[j];
                    }
                }
                
                const Real C_invmass = static_cast<Real>(1)/C_C[0];
        
                for( Int j = 1; j < far_dim; ++j )
                {
                    C_C[j] *= C_invmass;
                }
            }
        }; //computeClusterData

        
        
    public:
        
        void RequireClusterMoments( const Int moment_degree_ ) const override
        {
            
            if( moment_degree_ > moment_degree )
            {
                ptic(ClassName()+"::RequireClusterMoments");
                
                MultipoleMomentsBase<Real,Int> * ptr;
            
                switch (moment_degree_)
                {
                    case 0:
                    {
                        moment_degree = 0;
                        return;
                    }
                    case 1:
                    {
                        moment_degree = 0;
                        return;
                    }
//                    case 2:
//                    {
//                        moment_degree = 2;
//                        ptr = new MultipoleMoments<1 + AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2,2,Real,Int>();
//                        break;
//                    }
//                    case 4:
//                    {
//                        moment_degree = 4;
//                        ptr = new MultipoleMoments<1 + AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2,4,Real,Int>();
//                        break;
//                    }
                    default:
                    {
                        eprint("Degree not available.");
                        return;
                    }
                }

                std::unique_ptr<MultipoleMomentsBase<Real,Int>> M (ptr);

                C_moments = DataContainer_T ( ClusterCount(), M->MomentCount(), static_cast<Real>(0) );

                M_ker = std::vector<std::unique_ptr<MultipoleMomentsBase<Real,Int>>>(  ThreadCount() );

                #pragma omp parallel for num_threads( ThreadCount() )
                for( Int thread = 0; thread < ThreadCount(); ++thread )
                {
                    M_ker[thread] = std::unique_ptr<MultipoleMomentsBase<Real,Int>>( static_cast<MultipoleMomentsBase<Real,Int> *>(M->Clone().release()) );
                }
                
//                 using the already serialized cluster tree
                #pragma omp parallel num_threads( ThreadCount() )
                {
                    #pragma omp single nowait
                    {
                        computeClusterMoments( 0, ThreadCount() );
                    }
                }

                ptoc(ClassName()+"::RequireClusterMoments");
            }
        
        }; //RequireClusterMoments
      
    protected:
        
        void computeClusterMoments( const Int C, const Int free_thread_count ) const // helper function for RequireMoments
        {
            const Int thread = omp_get_thread_num();
            const Int L = C_left [C];
            const Int R = C_right[C];
            
            if( L >= 0 /*&& R >= 0*/ ){
                //C points to interior node.
                #pragma omp task final(free_thread_count<1)  shared( L )
                {
                    computeClusterMoments( L, free_thread_count/2 );
                }
                #pragma omp task final(free_thread_count<1)  shared( R )
                {
                    computeClusterMoments( R, free_thread_count-free_thread_count/2 );
                }
                #pragma omp taskwait

                M_ker[thread]->ClusterToCluster( C_far.data(L), C_moments.data(L), C_far.data(C), C_moments.data(C) );
                
                M_ker[thread]->ClusterToCluster( C_far.data(R), C_moments.data(R), C_far.data(C), C_moments.data(C) );
                
            }
            else
            {
                //C points to leaf node.
                //compute from primitives
                const Int begin = C_begin[C];
                const Int end   = C_end  [C];
                
                for( Int i = begin; i < end; ++i )
                {
                    M_ker[thread]->PrimitiveToCluster( P_far.data(i), C_far.data(C), C_moments.data(C) );
                }
                
            }
        }; //computeClusterMoments
        
        void ComputePrimitiveToClusterMatrix()
        {
            ptic(ClassName()+"::ComputePrimitiveToClusterMatrix");
            
            P_to_C = SparseBinaryMatrixCSR<Int>(
                ClusterCount(), PrimitiveCount(), PrimitiveCount(), ThreadCount() );
            
            P_to_C.Outer()[0] = 0;
            
            C_to_P = SparseBinaryMatrixCSR<Int>(
                PrimitiveCount(), ClusterCount(), PrimitiveCount(), ThreadCount() );
            
            C_to_P.Outer()[PrimitiveCount()] = PrimitiveCount();
            
            {
                Int * restrict const inner__ = C_to_P.Inner().data();
                
                const Int leaf_cluster_count = LeafClusterCount();
                
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int i = 0; i < leaf_cluster_count; ++i )
                {
                    Int leaf  = leaf_clusters[i];
                    Int begin = C_begin[leaf];
                    Int end   = C_end  [leaf];
//                    #pragma omp simd aligned ( inner__ : ALIGN )
                    for( Int k = begin; k < end; ++k )
                    {
                        inner__[k] = leaf;
                    }
                }
                
            }
            
            {
                Int * restrict const i = C_to_P.Outer().data();
                Int * restrict const j = P_to_C.Inner().data();
                
                const Int primitive_count = PrimitiveCount();
                
//                #pragma omp simd aligned ( i, j : ALIGN )
                for( Int k = 0; k < primitive_count; ++k )
                {
                    i[k] = k;
                    j[k] = k;
                }
            }
            
            {
                Int cluster_count = ClusterCount();
                
                Int       * restrict const outer__ = P_to_C.Outer().data();
                Int const * restrict const left__  = C_left.data();
                Int const * restrict const begin__ = C_begin.data();
                Int const * restrict const end__   = C_end.data();
                
                for ( Int C = 0; C < cluster_count; ++C )
                {
                    outer__[C+1] = outer__[C] + static_cast<Int>(left__[C] < 0) * (end__[C] - begin__[C]);
                }
            }
            
            ptoc(ClassName()+"::ComputePrimitiveToClusterMatrix");
        }
        

        
        void ComputePrePost(
            SparseMatrixCSR<Real,Int> & DiffOp,
            SparseMatrixCSR<Real,Int> & AvOp
        )
        {
            if( !this->pre_post_initialized )
            {
                ptic(ClassName()+"::ComputePrePost");

                
                const Int primitive_count = PrimitiveCount();
                
                ptic("hi_pre");
                
                hi_pre = SparseMatrixCSR<Real,Int>(
                    DiffOp.RowCount(),
                    DiffOp.ColCount(),
                    DiffOp.NonzeroCount(),
                    ThreadCount()
                );
                
                //Assuming that all rows have the same number of entries!
                {
                    
                    hi_pre.Outer().Read( DiffOp.Outer().data() );
                    
                    Int  const * restrict const i_input  = DiffOp.Inner().data();
                    Int        * restrict const i_output = hi_pre.Inner().data();
                    Real const * restrict const r_input  = DiffOp.Values().data();
                    Real       * restrict const r_output = hi_pre.Values().data();
                
                    const Int size = DiffOp.NonzeroCount() / primitive_count;
                    
                    #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                    for( Int i = 0; i < primitive_count; ++i )
                    {
                        const Int from = size * P_ordering[i];
                        const Int to   = size * i;
                        const Real a = P_near( i, 0 );

                        for( Int k = 0; k < size; ++k )
                        {
                            i_output[to+k] =     i_input[from+k];
                            r_output[to+k] = a * r_input[from+k];
                        }
                    }
                }
                ptoc("hi_pre");
                
                ptic("hi_post");
                
                DUMP(hi_pre.Stats());
                
                hi_post = hi_pre.Transpose();
                
                ptoc("hi_post");
                
                ptic("lo_pre");
                lo_pre = SparseMatrixCSR<Real,Int>(
                    AvOp.RowCount(),
                    AvOp.ColCount(),
                    AvOp.NonzeroCount(),
                    ThreadCount()
                );
                
                //Assuming that all rows have the same number of entries!
                {
                    lo_pre.Outer().Read( AvOp.Outer().data() );
                                        
                    Int  const * restrict const i_input  =   AvOp.Inner().data();
                    Int        * restrict const i_output = lo_pre.Inner().data();
                    Real const * restrict const r_input  =   AvOp.Values().data();
                    Real       * restrict const r_output = lo_pre.Values().data();
                    
                    const Int size = AvOp.NonzeroCount() / primitive_count;
                    
                    #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                    for( Int i = 0; i < primitive_count; ++i )
                    {
                        const Int from = size * P_ordering[i];
                        const Int to   = size * i;
                        const Real a   = P_near( i, 0 );
//                        #pragma omp simd aligned( i_input, i_output, r_input, r_output : ALIGN )
                        for( Int k = 0; k < size; ++k )
                        {
                            i_output[ to + k ] =     i_input[ from + k ];
                            r_output[ to + k ] = a * r_input[ from + k ];
                        }
                    }
                }
                ptoc("lo_pre");
                
                ptic("lo_post");
                
                DUMP(lo_pre.Stats());
                
                lo_post = lo_pre.Transpose();
                ptoc("lo_post");
                
                this->pre_post_initialized = true;
                
                ptoc(ClassName()+"::ComputePrePost");
            }
        } // ComputePrePost
        
    public:
        
        // Updates only the computational data (primitive/cluster areas, centers of mass and normals).
        // All data related to clustering or multipole acceptance criteria remain are unchanged, as well
        // as the preprocessor and postprocessor matrices (that are needed for matrix-vector multiplies of the BCT.)
        
        void SemiStaticUpdate( const Tensor2<Real,Int> & P_near_, const Tensor2<Real,Int> & P_far_ ) override
        {
            if( P_near_.Dimension(0) != PrimitiveCount() )
            {
                eprint(ClassName()+"::SemiStaticUpdate: P_near_.Dimension(0) = "+ToString(P_near_.Dimension(0))+" != "+ToString(PrimitiveCount())+" = PrimitiveCount(). Skipping update.");
                return;
            }
            if( P_near_.Dimension(1) != NearDim() )
            {
                eprint(ClassName()+"::SemiStaticUpdate: P_near_.Dimension(1) "+ToString(P_near_.Dimension(1))+" != "+ToString(NearDim())+" = NearDim(). Skipping update.");
                return;
            }
            
            if( P_far_.Dimension(0) != PrimitiveCount() )
            {
                eprint(ClassName()+"::SemiStaticUpdate: P_far_.Dimension(0) "+ToString(P_far_.Dimension(0))+" != "+ToString(PrimitiveCount())+" = PrimitiveCount(). Skipping update.");
                return;
            }
            
            if( P_far_.Dimension(1) != FarDim() )
            {
                eprint(ClassName()+"::SemiStaticUpdate: P_far_.Dimension(1) "+ToString(P_far_.Dimension(1))+" != "+ToString(FarDim())+" = FarDim(). Skipping update.");
                return;
            }
            SemiStaticUpdate( P_near_.data(), P_far_.data() );
        } // SemiStaticUpdate
        
        void SemiStaticUpdate( const Real * restrict const P_near_, const Real * restrict const P_far_ ) override
        {
            // Updates only the computational data like primitive/cluster areas, centers of mass and normals. All data related to clustering or multipole acceptance criteria remain are unchanged.
            
            ptic(ClassName()+"::SemiStaticUpdate");
            
            this->ComputePrimitiveData( P_near_, P_far_ );
            
            this->ComputeClusterData();

            ptoc(ClassName()+"::SemiStaticUpdate");
            
        } // SemiStaticUpdate
        
        void TakeUpdateVectors(
            MovingPrimitive_T & P_moving_,
            Tensor2<SReal,Int> & P_velocities_serialized_,
            const SReal max_time
        ) const
        {
            ptic(ClassName()+"::TakeUpdateVectors");
            
            // ATTENTION: We assume here that P_velocities_serialized are handed over already in correct order and won't be used anymore and just destroyed by the caller. This allows us to just swap the Tensor2s here, saving us a big copy operation.
            swap( P_velocities_serialized, P_velocities_serialized_ );
            
            // TODO: Potentially superfluous container.
            P_updated_serialized = Tensor2<SReal,Int>( PrimitiveCount(), P_proto[0]->Size() );
            C_updated_serialized = Tensor2<SReal,Int>( ClusterCount(),   C_proto[0]->Size() );
            
            
            const SReal * restrict const P_p_ser = P_serialized.data();
            const SReal * restrict const P_v_ser = P_velocities_serialized.data();
            
                  SReal * restrict const P_up_ser = P_updated_serialized.data();
                  SReal * restrict const C_up_ser = C_updated_serialized.data();
            
            ptic(ClassName()+"::TakeUpdateVectors - Compute the primitives updated with max_time.");
            // TODO: Potentially wastful code.
            {
                const Tensor1<Int,Int> job_ptr = BalanceWorkLoad(PrimitiveCount(), ThreadCount() );
            
                #pragma omp parallel for num_threads( ThreadCount() )
                for( Int thread = 0; thread < ThreadCount(); ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    P_moving[thread] = P_moving_.Clone();
                    MovingPrimitive_T & P_mov = *P_moving[thread];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        P_mov.ReadCoordinatesSerialized ( P_p_ser           , i );
                        P_mov.ReadVelocitiesSerialized  ( P_v_ser           , i );
                        P_mov.WriteDeformedSerialized   ( P_up_ser, max_time, i );
                    }
                }
            }
            ptoc(ClassName()+"::TakeUpdateVectors - Compute the primitives updated with max_time.");
            
            ptic(ClassName()+"::TakeUpdateVectors - Compute the AABBs of the updated leaf clusters.");
            {
                const Tensor1<Int,Int> job_ptr = BalanceWorkLoad(LeafClusterCount(), ThreadCount() );
                
                #pragma omp parallel for num_threads(ThreadCount() )
                for( Int thread = 0; thread <ThreadCount(); ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                         Primitive_T & P_   = *P_proto[thread];
                    BoundingVolume_T & C_bv = *C_proto[thread];
                                        
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int C = leaf_clusters[i];
                        
                        C_bv.SetPointer( C_up_ser, C );
                        C_bv.FromPrimitives( P_, P_up_ser, C_begin[C], C_end[C], 1 );
                        
                    }
                }
            }
            ptoc(ClassName()+"::TakeUpdateVectors - Compute the AABBs of the updated leaf clusters.");
            
            ptic(ClassName()+"::TakeUpdateVectors - Upward pass for AABBs.");
            {
                Int stack_array[128];
                Int stack_ptr = 0;
                stack_array[stack_ptr] = 0;

                Tensor1<bool,Int> visited( ClusterCount(), false );

                BoundingVolume_T & C_bv = *C_proto[0];

                while( 0<=stack_ptr && stack_ptr < 126 )
                {
                    const Int C = stack_array[stack_ptr];
                    const Int L = C_left [C];
                    const Int R = C_right[C];

                    if( (L >= 0) /*&& (R >= 0)*/ )
                    {
                        if( visited[C] )
                        {
                            C_bv.SetPointer ( C_up_ser, C );
                            C_bv.Read       ( C_up_ser, L );
                            C_bv.Merge      ( C_up_ser, R );
                            stack_ptr--;
                        }
                        else
                        {
                            stack_array[++stack_ptr] = R;
                            stack_array[++stack_ptr] = L;
                            visited[C] = true;
                        }
                    }
                    else
                    {
                        stack_ptr--;
                    }
                }
            }
            ptoc(ClassName()+"::TakeUpdateVectors - Upward pass for AABBs.");
            
            ptoc(ClassName()+"::TakeUpdateVectors");
            
        } // TakeUpdateVectors
        
        void CollectNearFieldDerivatives( bool addto = false ) const override
        {
            // Collect derivative data from the primitives, so that an external routine can do the assembly.
            
            ptic(ClassName()+"::CollectNearFieldDerivatives");
            
            const Int near_dim = NearDim();
            const Int primitive_count = PrimitiveCount();
            
            if( !addto )
            {
                // Write first slice.
                const Real * restrict const from = thread_P_D_near.data(0);
                      Real * restrict const to   = P_D_near.data();
                
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int i = 0; i < primitive_count; ++i )
                {
                    const Int j = P_inverse_ordering[i];

                    for( Int k = 0; k < near_dim; ++k )
                    {
                        to[ near_dim * i + k ] = from[ near_dim * j + k  ];
                    }
                }
            }
            
            // Add the other slices. (Start with the first slice if we were asked to add into.)
            for( Int thread = 1-addto; thread < ThreadCount(); ++thread )
            {
                const Real * restrict const from = thread_P_D_near.data(thread);
                      Real * restrict const to   = P_D_near.data();
                
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int i = 0; i < primitive_count; ++i )
                {
                    const Int j = P_inverse_ordering[i];
                    
                    for( Int k = 0; k < near_dim; ++k )
                    {
                        to[ near_dim * i + k ] += from[ near_dim * j + k  ];
                    }
                }
            }
            
            ptoc(ClassName()+"::CollectNearFieldDerivatives");
            
        } // CollectNearFieldDerivatives
        
        
        void CollectFarFieldDerivatives( bool addto = false ) const override
        {
            // Collect derivative data from the clusters, so that an external routine can do the assembly.
            
            ptic(ClassName()+"::CollectFarFieldDerivatives");
            
            const Int far_dim         = FarDim();
            const Int cluster_count   = ClusterCount();
            const Int primitive_count = PrimitiveCount();
            const Int thread_count    = ThreadCount();
            
            this->RequireBuffers(far_dim);
            
            // Write first slice.
            std::memcpy( C_out.data(), thread_C_D_far.data(0), sizeof(Real) * cluster_count * far_dim );
            
            // Add the other slices.
            for( Int thread = 1; thread < thread_count; ++thread )
            {
                const Real * restrict const from = thread_C_D_far.data(thread);
                      Real * restrict const to   = C_out.data();
                
                #pragma omp parallel for simd num_threads( thread_count ) aligned( from, to : ALIGN ) schedule( static )
                for( Int i = 0; i < cluster_count * far_dim; ++i )
                {
                    to[i] += from[i];
                }
            }
            
            this->PercolateDown();
            
            this->ClustersToPrimitives(false);

            // Finally, permute data for the outside world.
            
            if(addto)
            {
                Real const * restrict const from = P_out.data();
                Real       * restrict const to   = P_D_far.data();
                #pragma omp parallel for num_threads( thread_count ) schedule( static )
                for( Int i = 0; i < primitive_count; ++i )
                {
                    const Int j = P_inverse_ordering[i];
                    
                    for( Int k = 0; k < far_dim; ++k )
                    {
                        to[ far_dim * i + k ] += from[ far_dim * j + k ];
                    }
                }
            }
            else
            {
                Real const * restrict const from = P_out.data();
                Real       * restrict const to   = P_D_far.data();
                
                const size_t far_size = sizeof(Real) * far_dim;
                
                #pragma omp parallel for num_threads( thread_count ) schedule( static )
                for( Int i = 0; i < primitive_count; ++i )
                {
                    const Int j = P_inverse_ordering[i];
                    
                    std::memcpy( &to[far_dim * i], &from[far_dim * j], far_size );
                }
                
            }
            
            ptoc(ClassName()+"::CollectFarFieldDerivatives");
            
        } // CollectFarFieldDerivatives
        
        
    
        
        const Tensor2<Real,Int> & ClusterMoments() const override
        {
            return C_moments;
        }
        
    public:
        
        
        std::string Stats() const override
        {
            std::stringstream s;
            
            s
            << "\n==== "+ClassName()+" Stats ====" << "\n\n"
            << " AmbDim()                    = " <<  this->AmbDim() << "\n"
            << " ThreadCount()               = " <<  this->ThreadCount() << "\n"
            << "\n"
            << " PrimitiveCount()            = " <<  this->PrimitiveCount() << "\n"
            << " ClusterCount()              = " <<  this->ClusterCount() << "\n"
            << " LeafClusterCount()          = " <<  this->LeafClusterCount() << "\n"
            << " PrimitivePrototype()        = " <<  this->PrimitivePrototype().ClassName() <<"\n"
            << " ClusterPrototype()          = " <<  this->ClusterPrototype().ClassName() <<"\n"
            
            << " SplitThreshold()            = " <<  this->SplitThreshold() << "\n"
            << " TreeDepth()                 = " <<  this->TreeDepth() << "\n"

            << "\n==== "+ClassName()+" Stats ====\n" << std::endl;

            
            return s.str();
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS) + "<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }

    }; // CLASS
    
} // namespace Repulsion

#undef BASE
#undef CLASS
