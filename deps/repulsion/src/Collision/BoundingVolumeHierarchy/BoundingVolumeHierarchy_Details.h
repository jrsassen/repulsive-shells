public:

~CLASS() = default;

private:
    // "C_" stands for "cluster", "P_" stands for "primitive"
    
    // Each thread gets its own primitive prototype to avoid sharing conflicts.
    std::vector<std::unique_ptr<Primitive_T>> P_proto;           // False sharing prevented by alignment of PrimitiveBase.
    
    // Container for storing the serialized data of the primitives. Only meant to be accessed by primitive prototypes.
    // TODO: Make this private somehow?
    mutable Tensor2<SReal,Int> P_serialized;
    
    // Each thread gets its own bounding volume prototype to avoid sharing conflicts.
    std::vector<std::unique_ptr<BoundingVolume_T>> C_proto;           // False sharing prevented by alignment of PrimitiveBase.
    
    // Container for storing the serialized data of the clusters. Only meant to be accessed by bounding volume prototypes.
    // TODO: Make this private somehow?
    mutable Tensor2<SReal,Int> C_serialized;
    
private:
    // Some temproray shared data that is required for the parallel construction and serialization of the tree.
    
    Tensor3<SReal,Int> C_thread_serialized;          // False sharing is unlikely as each thread's slice should already be quite large...
    
    Tensor2<Int,Int> thread_cluster_counter;                          // TODO: Avoid false sharing!
    
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
    
    Tensor1<Int,Int> leaf_clusters;
    Tensor1<Int,Int> leaf_cluster_lookup;
    Tensor1<Int,Int> leaf_cluster_ptr;

    Tensor1<SReal,Int> P_score_buffer;
    Tensor1<Int,Int> P_perm_buffer;

    mutable Tensor1<Int,Int> stack_array;
    mutable Tensor1<Int,Int> queue_array;

    Int depth;

private:

    void ComputeClusters()
    {
        ptic(ClassName()+"::ComputeClusters");
        
        Int thread_count = std::max( TreeThreadCount(), ThreadCount() );
        // Request some temporary memory for threads.
        
//        P_ordering         = iota<Int>   ( PrimitiveCount() );
        P_inverse_ordering = Tensor1<Int,Int>( PrimitiveCount() );
        
        // Padding every row to prevent false sharing.
        thread_cluster_counter = Tensor2<Int,Int>( thread_count, 2 * ALIGN, 0 );
        
        C_thread_serialized = Tensor3<SReal,Int>( thread_count, 2 * PrimitiveCount(), C_proto[0]->Size() );
        
        Int thread = 0;
        
        thread_cluster_counter(thread,0)++;
        
        auto * root = new Cluster<Int>( thread, 0, 0, PrimitiveCount(), 0 );
                        
        // Initial bounding volume of root node.
        C_proto[thread]->SetPointer( C_thread_serialized.data(thread), 0 );
        C_proto[thread]->FromPrimitives( *P_proto[thread], P_serialized.data(), 0, PrimitiveCount(), thread_count );
        
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
        P_perm_buffer  = Tensor1<Int,Int>( PrimitiveCount() );
        
        #pragma omp parallel num_threads( TreeThreadCount() ) shared( root )
        {
            #pragma omp single nowait
            {
                split( root, TreeThreadCount() );
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
//            print( "split { "+ToString(begin)+", "+ToString(split_index)+", "+ToString(end)+" } diff  = { "+ ToString(split_index-begin)+", "+ToString(end-split_index)+" }" );
            
            if( (begin < split_index) && (split_index < end) )
            {
                // create new nodes...
                thread_cluster_counter(thread,0) += 2;
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
                #pragma omp task
                {
                    queue_array = Tensor1<Int,Int>( ClusterCount() );
                }
                #pragma omp taskwait
            }
        }

        #pragma omp parallel num_threads( TreeThreadCount() )
        {
            #pragma omp single nowait
            {
                serialize( root, 0, 0, TreeThreadCount() );
            }
        }

        depth = root->max_depth;
        stack_array = Tensor1<Int,Int>( 2 * depth + 1 );
        
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
        Int thread = omp_get_thread_num();
        
        // enumeration in depth-first order
        C_begin[ID] = C->begin;
        C_end  [ID] = C->end;
        C_depth[ID] = C->depth;
        C_next [ID] = ID + C->descendant_count;
        
//        // TODO: Potentially, write conflicts could occur here. But they should be seldom as we use depth-first order.
//        C_proto[thread]->Copy(
//            C_thread_serialized.data(C->thread), C->ID,
//                   C_serialized.data(),             ID
//        );
        
        // TODO: Potentially, write conflicts could occur here. But they should be seldom as we use depth-first order.
        C_proto[thread]->SetPointer( C_thread_serialized.data(C->thread), C->ID );
        C_proto[thread]->Write( C_serialized.data(), ID );
        
        if( ( C->left != nullptr ) && ( C->right != nullptr ) )
        {
            C_left [ID] = ID + 1;
            C_right[ID] = ID + 1 + C->left->descendant_count;
    //
            #pragma omp task final(free_thread_count<1)  shared( C_left, C_right )
            {
                serialize( C->left, C_left[ID], leaf_before_count, free_thread_count/2 );
            }
            #pragma omp task final(free_thread_count<1)  shared( C_left, C_right )
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
    
    Int AmbDim() const OVERRIDE
    {
        return AMB_DIM;
    }
    
    Int ThreadCount() const OVERRIDE
    {
        return settings.thread_count;
    }
    
    Int TreeThreadCount() const OVERRIDE
    {
        return settings.tree_thread_count;
    }
    
    Int SplitThreshold() const OVERRIDE
    {
        return settings.split_threshold;
    }

    Int Depth() const OVERRIDE
    {
        return depth;
    }

    Int TreeDepth() const OVERRIDE
    {
        return depth;
    }
            
    Int PrimitiveCount() const OVERRIDE
    {
        return P_serialized.Dimension(0);
    }
    
    Int PrimitiveSize() const OVERRIDE
    {
        return P_serialized.Dimension(1);
    }
    
    Int ClusterCount() const OVERRIDE
    {
        return C_serialized.Dimension(0);
    }
    
    Int ClusterSize() const OVERRIDE
    {
        return C_serialized.Dimension(1);
    }
    
    Int LeafClusterCount() const OVERRIDE
    {
        return leaf_clusters.Dimension(0);
    }

    const Tensor1<Int,Int> & LeafClusters() const OVERRIDE
    {
        return leaf_clusters;
    }
    
    const Tensor1<Int,Int> & LeafClusterLookup() const OVERRIDE
    {
        return leaf_cluster_lookup;
    }
    
    const Tensor1<Int,Int> & LeafClusterPointers() const OVERRIDE
    {
        return leaf_cluster_ptr;
    }
    
    const Tensor1<Int,Int> & PrimitiveOrdering() const OVERRIDE
    {
        return P_ordering;
    }
    
    const Tensor1<Int,Int> & PrimitiveInverseOrdering() const OVERRIDE
    {
        return P_inverse_ordering;
    }
    
    const Tensor1<Int,Int> & ClusterBegin() const OVERRIDE
    {
        return C_begin;
    }

    const Tensor1<Int,Int> & ClusterEnd() const OVERRIDE
    {
        return C_end;
    }
    
    const Tensor1<Int,Int> & ClusterLeft() const OVERRIDE
    {
        return C_left;
    }
    
    const Tensor1<Int,Int> & ClusterRight() const OVERRIDE
    {
        return C_right;
    }

    const Tensor1<Int,Int> & ClusterDepths() const OVERRIDE
    {
        return C_depth;
    }
            
    Tensor2<SReal,Int> & ClusterSerializedData() const OVERRIDE
    {
        return C_serialized;
    }
    
    const BoundingVolume_T & ClusterPrototype() const
    {
        return *C_proto[0];
    }
    
    Tensor2<SReal,Int> & PrimitiveSerializedData() const OVERRIDE
    {
        return P_serialized;
    }
    
    const Primitive_T & PrimitivePrototype() const
    {
        return *P_proto[0];
    }
    
//        void PrintToFile(std::string filename) const
//        {
//            std::ofstream os;
//            std::cout << "Writing tree "+ClassName()+" to " << filename << "." << std::endl;
//            os.open(filename);
//            os << "ID" << "\t" << "left" << "\t" << "right" << "\t" << "next" << "\t" << "depth" << "\t"  << "begin" << "\t" << "end" << std::endl;
//            for( Int C = 0;  C < ClusterCount(); ++C)
//            {
//                os << C << "\t" << C_left[C] << "\t" << C_right[C] << "\t" << C_next[C] << "\t" << C_depth[C] << "\t" << C_begin[C] << "\t" << C_end[C] << std::endl;
//            }
//            os.close();
//            std::cout << "Done writing." << std::endl;
//        }

std::vector<Int> NearestPrimitive_DepthFirst( const Primitive_T & Q ) const
{
    auto a = NearestPrimitive_Intrinsic_DepthFirst(Q);
    for( Int i = 0; i < static_cast<Int>(a.size()); ++i )
    {
        a[i] = P_ordering[a[i]];
    }
    return a;
}

std::vector<Int> NearestPrimitive_Intrinsic_DepthFirst( const Primitive_T & Q ) const
{
    tic(ClassName()+"::NearestPrimitive_Intrinsic_DepthFirst");
    
    Int clusters_visited = 0;
    Int primitives_visited = 0;
    
    Real upper_bound = std::numeric_limits<Real>::max();
    std::vector<Int> nearest;
    
    const Real Q_r = sqrt( Q.SquaredRadius() );
    
    GJK_Algorithm<AMB_DIM,Real,Int> gjk;
    
    std::unique_ptr<Primitive_T> P   = P_proto[0]->Clone();
    std::unique_ptr<BoundingVolume_T> C_C = C_proto[0]->Clone();
    
    const Int * stack_begin = stack_array.data();
          Int * stack_ptr = stack_array.data();
    stack_ptr[0] = 0;
    
    while( stack_ptr >= stack_begin )
    {
        // pop
        const Int C = *(stack_ptr--);
        
        ++clusters_visited;
        
        C_C->SetPointer( C_serialized.data(), C );
        
        const Real C_r         = sqrt( C_C->SquaredRadius() );
        const Real center_dist = sqrt( gjk.InteriorPoints_SquaredDistance( Q, *C_C ) );

        Real max_squared_dist = center_dist + C_r + Q_r;
        Real min_squared_dist = std::max( static_cast<Real>(0) , center_dist - C_r - Q_r );
        
        max_squared_dist = max_squared_dist * max_squared_dist;
        min_squared_dist = min_squared_dist * min_squared_dist;
        
        if( max_squared_dist < upper_bound )
        {
            upper_bound = max_squared_dist;
//            valprint("upper_bound",upper_bound);
        }
//        max_squared_dist = std::min( max_squared_dist, upper_bound );

        if( min_squared_dist <= upper_bound )
        {
            min_squared_dist = gjk.SquaredDistance( Q, *C_C );
            
            if( min_squared_dist <= upper_bound )
            {
                const Int L = C_left [C];
                const Int R = C_right[C];
                
                if( L >= 0 /* && R >= 0 */ )
                {
                    // push
                    *(++stack_ptr) = R;
                    *(++stack_ptr) = L;
                }
                else
                {
//                    print("Going through all primitives of cluster "+ToString(C)+".");
                    
                    const Int i_begin = C_begin[C];
                    const Int i_end   = C_end  [C];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        
                        P->SetPointer( P_serialized.data(), i );
                        
                        const Real squared_dist = gjk.SquaredDistance( Q, *P );
                        
                        if( squared_dist <= upper_bound )
                        {
                            if( squared_dist < upper_bound )
                            {
                                nearest.clear();
                            }
                            upper_bound = squared_dist;
                            nearest.push_back( i );
                        }
                    }
                    
                    primitives_visited += (i_end - i_begin);
                }
            }
        }
    }

    toc(ClassName()+"::NearestPrimitive_Intrinsic_DepthFirst");
    
    valprint("distance",sqrt(upper_bound), 16 );
    valprint("clusters_visited",clusters_visited);
    valprint("primitives_visited",primitives_visited);

    return nearest;
} // NearestPrimitive_Intrinsic_DepthFirst


std::vector<Int> NearestPrimitive_BreadthFirst( const Primitive_T & Q ) const
{
    auto a = NearestPrimitive_Intrinsic_BreadthFirst(Q);
    for( Int i = 0; i < static_cast<Int>(a.size()); ++i )
    {
        a[i] = P_ordering[a[i]];
    }
    return a;
}

std::vector<Int> NearestPrimitive_Intrinsic_BreadthFirst( const Primitive_T & Q ) const
{
    tic(ClassName()+"::NearestPrimitive_Intrinsic_BreadthFirst");
    
    Int clusters_visited = 0;
    Int primitives_visited = 0;
    
    Real upper_bound = std::numeric_limits<Real>::max();
    std::vector<Int> nearest;
    
    const Real Q_r = sqrt( Q.SquaredRadius() );
    
    GJK_Algorithm<AMB_DIM,Real,Int> gjk;
    
    std::unique_ptr<Primitive_T> P   = P_proto[0]->Clone();
    std::unique_ptr<BoundingVolume_T> C_C = C_proto[0]->Clone();
    
    Int * queue_begin = queue_array.data();
    Int * queue_end   = queue_array.data();
    queue_begin[0] = 0;
    
    while( queue_end >= queue_begin )
    {
        // pop
        const Int C = *(queue_begin++);
        
        ++clusters_visited;
        
        C_C->SetPointer( C_serialized.data(), C );
        
        const Real C_r         = sqrt( C_C->SquaredRadius() );
        const Real center_dist = sqrt( gjk.InteriorPoints_SquaredDistance( Q, *C_C ) );

        Real max_squared_dist = center_dist + C_r + Q_r;
        max_squared_dist = max_squared_dist * max_squared_dist;
        
        Real min_squared_dist = std::max( static_cast<Real>(0), center_dist - C_r - Q_r );
        min_squared_dist = min_squared_dist * min_squared_dist;
        
        if( max_squared_dist < upper_bound )
        {
            upper_bound = max_squared_dist;
//            valprint("upper_bound",upper_bound);
        }
//        max_squared_dist = std::min( max_squared_dist, upper_bound );

        if( min_squared_dist <= upper_bound )
        {
            min_squared_dist = gjk.SquaredDistance( Q, *C_C );
            
            if( min_squared_dist <= upper_bound )
            {
                const Int L = C_left [C];
                const Int R = C_right[C];
                
                if( L >= 0 /* && R >= 0 */ )
                {
                    // push
                    *(++queue_end) = R;
                    *(++queue_end) = L;
                }
                else
                {
//                    print("Going through all primitives of cluster "+ToString(C)+".");
                    
                    const Int i_begin = C_begin[C];
                    const Int i_end   = C_end  [C];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        P->SetPointer( P_serialized.data(), i );
                        
                        const Real squared_dist = gjk.SquaredDistance( Q, *P );
                        
                        if( squared_dist <= upper_bound )
                        {
                            if( squared_dist < upper_bound )
                            {
                                nearest.clear();
                                upper_bound = squared_dist;
                            }
                            nearest.push_back( i );
                        }
                    }
                    
                    primitives_visited += (i_end - i_begin);
                }
            }
        }
    }
    
    toc(ClassName()+"::NearestPrimitive_Intrinsic_BreadthFirst");
    
    valprint("distance",sqrt(upper_bound), 16 );
    valprint("clusters_visited",clusters_visited);
    valprint("primitives_visited",primitives_visited);

    return nearest;
}
