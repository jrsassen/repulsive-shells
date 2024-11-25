public:

    ~CLASS() = default;

private:

    const ClusterTree_T * S; // "left"  BVH (output side of matrix-vector multiplication)
    const ClusterTree_T * T; // "right" BVH (input  side of matrix-vector multiplication)

    const SETTING_T settings;
    
    // Each thread gets its own bounding volume prototype to avoid sharing conflicts.
    mutable std::vector<std::unique_ptr<BoundingVolume_T>> S_proto;           // False sharing prevented by alignment of BoundingVolume_T.
    mutable std::vector<std::unique_ptr<BoundingVolume_T>> T_proto;           // False sharing prevented by alignment of BoundingVolume_T.
    
    // Adjacency matrices for far and near field.
    mutable  FAR_T  far;
    mutable NEAR_T near;

private:

    mutable std::vector<std::unique_ptr<GJK_T>> gjk;
    
    const SReal theta2 = static_cast<SReal>(0.25);
    
    const Int      thread_count = static_cast<Int>(1);
    const Int tree_thread_count = static_cast<Int>(1);

    const bool is_symmetric = false;
    const bool exploit_symmetry = false;

    const bool near_upper_triangular = false;
    const bool  far_upper_triangular = true;

    mutable bool blocks_initialized = false;
//    mutable bool metrics_initialized = false;
    
    // Containers for parallel aggregation of {i,j}-index pairs of separated and non-separated blocks.
    mutable std::vector<std::vector<Int>>  sep_i;
    mutable std::vector<std::vector<Int>>  sep_j;
    mutable std::vector<std::vector<Int>> nsep_i;
    mutable std::vector<std::vector<Int>> nsep_j;

    
    Tensor1<Int,Int> by_4;
    Tensor1<Int,Int> by_4_remainder;
    
    Tensor1<Int,Int> by_3;
    Tensor1<Int,Int> by_3_remainder;
    
    Tensor1<Int,Int> by_2;
    
public:

    virtual Int AmbDim() const OVERRIDE
    {
        return AMB_DIM;
    }
    
    virtual Int ThreadCount() const OVERRIDE
    {
        return thread_count;
    }

    virtual Int TreeThreadCount() const OVERRIDE
    {
        return tree_thread_count;
    }
    
    virtual Real SeparationParameter() const OVERRIDE
    {
        return sqrt(theta2);
    }
    
    virtual bool IsSymmetric() const OVERRIDE
    {
        return is_symmetric;
    }
    
    virtual bool ExploitSymmetry() const OVERRIDE
    {
        return exploit_symmetry;
    }
    
    virtual bool NearUpperTriangular() const OVERRIDE
    {
        return near_upper_triangular;
    }

    virtual bool FarUpperTriangular() const OVERRIDE
    {
        return far_upper_triangular;
    }

    virtual Int SeparatedBlockCount() const OVERRIDE
    {
        return far.NonzeroCount();
    }
    
    virtual Int NonseparatedBlockCount() const OVERRIDE
    {
        return near.BlockNonzeroCount();
    }
    
    virtual Int NearFieldInteractionCount() const OVERRIDE
    {
        return near.NonzeroCount();
    }

    virtual const FAR_T & Far() const OVERRIDE
    {
        return far;
    }
    
    virtual const NEAR_T & Near() const OVERRIDE
    {
        return near;
    }
    
    virtual const CLUSTER_TREE_BASE_T & GetS() const OVERRIDE
    {
        return *S;
    }
    
    virtual const CLUSTER_TREE_BASE_T & GetT() const OVERRIDE
    {
        return *T;
    }

    virtual const SETTING_T & Settings() const OVERRIDE
    {
        return settings;
    }
    
    virtual std::string Stats() const OVERRIDE
    {
        std::stringstream s;
        
        s
        << "\n==== "+ClassName()+" Stats ====" << "\n\n"
        << " AmbDim()                    = " <<  AmbDim() << "\n"
        << " SeparationParameter()       = " <<  SeparationParameter() << "\n"
        << " ThreadCount()               = " <<  ThreadCount() << "\n"
        << " TreeThreadCount()           = " <<  TreeThreadCount() << "\n"
        << "\n"
        << "Tree S"
        << S->Stats() << "\n"
        << T->Stats() << "\n"
        << " SeparatedBlockCount()       = " <<  SeparatedBlockCount() << "\n"
        << " NonseparatedBlockCount()    = " <<  NonseparatedBlockCount() << "\n"
        << " NearFieldInteractionCount() = " <<  NearFieldInteractionCount() << "\n"

        << "\n---- bool data ----" << "\n"

        << " IsSymmetric()               = " <<  IsSymmetric() << "\n"
        << " ExploitSymmetry()           = " <<  ExploitSymmetry() << "\n"
        << " NearUpperTriangular()       = " <<  NearUpperTriangular() << "\n"
        << " FarUpperTriangular()        = " <<  FarUpperTriangular() << "\n"

        << "\n==== "+ClassName()+" Stats ====\n" << std::endl;

        return s.str();
    }
    

    
    
//#####################################################################################################
//      Initialization
//#####################################################################################################

private:
    
    void PreparePrototypes() const
    {
        
        // This is for assuring that each thread gets a unique prototype for each tree, even if S == T.
        S_proto = std::vector<std::unique_ptr<BoundingVolume_T>> ( TreeThreadCount() );
        T_proto = std::vector<std::unique_ptr<BoundingVolume_T>> ( TreeThreadCount() );
        gjk     = std::vector<std::unique_ptr<GJK_T>>     ( TreeThreadCount() );
        
        #pragma omp parallel num_threads(TreeThreadCount())
        {
            Int thread = omp_get_thread_num();
            S_proto[thread] = S->ClusterPrototype().Clone();
            T_proto[thread] = T->ClusterPrototype().Clone();
            gjk[thread]     = std::make_unique<GJK_T>();
        }
        
    } // PreparePrototypes
    
    void ComputeBlocks()
    {
        if( blocks_initialized )
        {
            return;
        }
        
        ptic(ClassName()+"::ComputeBlocks");

         sep_i = std::vector<std::vector<Int>>(TreeThreadCount());
         sep_j = std::vector<std::vector<Int>>(TreeThreadCount());
        nsep_i = std::vector<std::vector<Int>>(TreeThreadCount());
        nsep_j = std::vector<std::vector<Int>>(TreeThreadCount());
    
        #pragma omp parallel num_threads(TreeThreadCount())
        {
            const Int thread = omp_get_thread_num();
            
            // TODO: Find a better initial guess for the number of admissable and inadmissable blocks.
            const Int expected = static_cast<Int>(10) * ( S->PrimitiveCount() + T->PrimitiveCount() );

             sep_i[thread] = std::vector<Int>();
             sep_j[thread] = std::vector<Int>();
            nsep_i[thread] = std::vector<Int>();
            nsep_j[thread] = std::vector<Int>();
            
             sep_i[thread].reserve(expected);
             sep_j[thread].reserve(expected);
            nsep_i[thread].reserve(expected);
            nsep_j[thread].reserve(expected);
        }
        
        switch( settings.block_split_method )
        {
            case BlockSplitMethod::Parallel:
            {
                Split_Parallel( static_cast<Int>(4) * TreeThreadCount() * TreeThreadCount() );
                break;
            }
            case BlockSplitMethod::Sequential:
            {
                Split_Sequential_DFS( static_cast<Int>(0), static_cast<Int>(0) );
                break;
            }
            case BlockSplitMethod::Recursive:
            {
                Split_Recursive();
                break;
            }
            default:
            {
                Split_Parallel( static_cast<Int>(4) * TreeThreadCount() * TreeThreadCount() );
            }
        }

        ptic(ClassName()+"  Far field interaction data");
        
        far = FAR_T(
            sep_i, sep_j,
            S->ClusterCount(), T->ClusterCount(),
            std::min(S->ThreadCount(), T->ThreadCount()),
            false, FarUpperTriangular()

        );
        
        DUMP(far.Stats());
        
        ptoc(ClassName()+"  Far field interaction data");
        
        ptic(ClassName()+"  Near field interaction data");
        
        near = NEAR_T(
            nsep_i, nsep_j,
            S->LeafClusterPointers(), T->LeafClusterPointers(),
            std::min(S->ThreadCount(), T->ThreadCount()),
            false, NearUpperTriangular()
        );
        
        DUMP(near.Stats());
        
        ptoc(ClassName()+"  Near field interaction data");
        
        
        blocks_initialized = true;
        
        // Free memory that is no longer used.
        #pragma omp parallel num_threads(TreeThreadCount())
        {
            const Int thread = omp_get_thread_num();

//             sep_i[thread].clear();
//             sep_j[thread].clear();
//            nsep_i[thread].clear();
//            nsep_j[thread].clear();
            
             sep_i[thread] = std::vector<Int>();
             sep_j[thread] = std::vector<Int>();
            nsep_i[thread] = std::vector<Int>();
            nsep_j[thread] = std::vector<Int>();
        }
        
        ptoc(ClassName()+"::ComputeBlocks");
            
    }; // ComputeBlocks

    void Split_Recursive()
    {
        ptic(ClassName()+"::Split_Recursive");
        
        // To avoid repeated integer division.
        by_4           = Tensor1<Int,Int> ( TreeThreadCount()+1 );
        by_4_remainder = Tensor1<Int,Int> ( TreeThreadCount()+1 );
        by_3           = Tensor1<Int,Int> ( TreeThreadCount()+1 );
        by_3_remainder = Tensor1<Int,Int> ( TreeThreadCount()+1 );
        by_2           = Tensor1<Int,Int> ( TreeThreadCount()+1 );
        
        for( Int thread = 0; thread < TreeThreadCount()+1; ++thread )
        {
            by_4          [thread] = thread / static_cast<Int>(4);
            by_4_remainder[thread] = thread % static_cast<Int>(4);

            by_3          [thread] = thread / static_cast<Int>(3);
            by_3_remainder[thread] = thread % static_cast<Int>(3);

            by_2          [thread] = thread / static_cast<Int>(2);
        }
        
        #pragma omp parallel num_threads(TreeThreadCount())
        {
            #pragma omp single nowait
            {
                split( static_cast<Int>(0), static_cast<Int>(0), TreeThreadCount() );
            }
        }
        ptoc(ClassName()+"::Split_Recursive");
        
    } // Split

    void split(
        const Int i,
        const Int j,
        const Int free_thread_count
    )
    {
        const Int thread = omp_get_thread_num();
        
        BoundingVolume_T & P = *S_proto[thread];

        P.SetPointer( S->ClusterSerializedData().data(), i );

        BoundingVolume_T & Q = *T_proto[thread];
        
        Q.SetPointer( T->ClusterSerializedData().data(), j );

        bool separatedQ = ( IsSymmetric() && (i == j) ) ? false : gjk[thread]->MultipoleAcceptanceCriterion( P, Q, theta2 );
        
        if( !separatedQ )
        {
            const Int lefti  = S->ClusterLeft()[i];
            const Int leftj  = T->ClusterLeft()[j];

            // Warning: This assumes that either both children are defined or empty.
            if( lefti >= static_cast<Int>(0) || leftj >= static_cast<Int>(0) )
            {
                const Int righti = S->ClusterRight()[i];
                const Int rightj = T->ClusterRight()[j];
                
                const SReal scorei = (lefti >= static_cast<SReal>(0))
                    ? P.SquaredRadius()
                    : static_cast<SReal>(0);
                const SReal scorej = (leftj >= static_cast<SReal>(0))
                    ? Q.SquaredRadius()
                    : static_cast<SReal>(0);

                if( scorei == scorej && scorei > static_cast<SReal>(0) && scorej > static_cast<SReal>(0) )
                {
                    // tie breaker: split both clusters

                    if( (exploit_symmetry) && (i == j) )
                    {
                        //  Creating 3 blockcluster children, since there is one block is just the mirror of another one.
                        
                        const Int spawncount = by_3          [free_thread_count];
                        const Int remainder  = by_3_remainder[free_thread_count];

                        #pragma omp task final(free_thread_count < 1)
                        split( lefti,  leftj,  spawncount + (remainder > static_cast<Int>(0)) );

                        #pragma omp task final(free_thread_count < 1)
                        split( lefti,  rightj, spawncount + (remainder > static_cast<Int>(1)) );
                        
                        #pragma omp task final(free_thread_count < 1)
                        split( righti, rightj, spawncount );
                        
                        #pragma omp taskwait
                    }
                    else
                    {
                        // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                        // This happens only if i and j represent_diffent clusters with same radii.
                        
                        const Int spawncount = by_4          [free_thread_count];
                        const Int remainder  = by_4_remainder[free_thread_count];

                        #pragma omp task final(free_thread_count < 1)
                        split( lefti,  leftj,  spawncount + (remainder > static_cast<Int>(0)) );
                        
                        #pragma omp task final(free_thread_count < 1)
                        split( righti, leftj,  spawncount + (remainder > static_cast<Int>(1)) );
                        
                        #pragma omp task final(free_thread_count < 1)
                        split( lefti,  rightj, spawncount + (remainder > static_cast<Int>(2)) );
                        
                        #pragma omp task final(free_thread_count < 1)
                        split( righti, rightj, spawncount );
                        
                        #pragma omp taskwait
                    }
                }
                else
                {
                    
                    const Int spawncount = by_2[free_thread_count];
                    
                    // split only larger cluster
                    if (scorei > scorej)
                    {
                        //split cluster i
                        #pragma omp task final(free_thread_count < 1)
                        split( lefti,  j, spawncount );
                        
                        #pragma omp task final(free_thread_count < 1)
                        split( righti, j, free_thread_count - spawncount );
                        
                        #pragma omp taskwait
                    }
                    else //scorei < scorej
                    {
//split cluster j
                        #pragma omp task final(free_thread_count < 1)
                        split( i, leftj,  spawncount );
                        
                        #pragma omp task final(free_thread_count < 1)
                        split( i, rightj, free_thread_count - spawncount );
                        
                        #pragma omp taskwait
                    }
                }
            }
            else
            {
                // create nonsep leaf blockcluster

                // i and j must be leaves of each ClusterTree S and T, so we directly store their position in the list leaf_clusters. This is important for the sparse matrix generation.

                //            In know  this is a very deep branching. I optimized it a bit for the case exploit_symmetry != 0 and upper_triangular == 0, though. That seemed to work best in regard of the matrix-vector multiplication.
                // TODO: Is there a clever way to avoid at least a bit of complixity of this branching? Would that speed up anything in the first place?
//                    if (exploit_symmetry)
                if (exploit_symmetry)
                {
                    if (!near_upper_triangular)
                    {
                        if (i != j)
                        {
                            // Generate also the twin to get a full matrix.
                            nsep_i[thread].push_back(S->leaf_cluster_lookup[i]);
                            nsep_i[thread].push_back(S->leaf_cluster_lookup[j]);
                            nsep_j[thread].push_back(T->leaf_cluster_lookup[j]);
                            nsep_j[thread].push_back(T->leaf_cluster_lookup[i]);
                        }
                        else
                        {
                            // This is a diagonal block; there is no twin to think about
                            nsep_i[thread].push_back(T->leaf_cluster_lookup[i]);
                            nsep_j[thread].push_back(S->leaf_cluster_lookup[i]);
                        }
                    }
                    else
                    {
                        // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                        if (i <= j)
                        {
                            nsep_i[thread].push_back(S->leaf_cluster_lookup[i]);
                            nsep_j[thread].push_back(T->leaf_cluster_lookup[j]);
                        }
                        else
                        {
                            nsep_i[thread].push_back(T->leaf_cluster_lookup[j]);
                            nsep_j[thread].push_back(S->leaf_cluster_lookup[i]);
                        }
                    }
                }
                else
                {
                    // No symmetry exploited.
                    nsep_i[thread].push_back(S->leaf_cluster_lookup[i]);
                    nsep_j[thread].push_back(T->leaf_cluster_lookup[j]);
                }
            }
        }
        else
        {
            //create sep leaf blockcluster
            if (exploit_symmetry)
            {
                if (!far_upper_triangular)
                {
                    // Generate also the twin to get a full matrix
                    sep_i[thread].push_back(i);
                    sep_i[thread].push_back(j);
                    sep_j[thread].push_back(j);
                    sep_j[thread].push_back(i);
                }
                else
                {
                    // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                    if (i <= j)
                    {
                        sep_i[thread].push_back(i);
                        sep_j[thread].push_back(j);
                    }
                    else
                    {
                        sep_i[thread].push_back(j);
                        sep_j[thread].push_back(i);
                    }
                }
            }
            else
            {
                // No symmetry exploited.
                sep_i[thread].push_back(i);
                sep_j[thread].push_back(j);
            }
        }
    }; // split


//#######################################################################################################
//      Initialization of matrices
//#######################################################################################################

public:


//Tensor1<SReal,Int> NearFieldInteriorPointsSquaredDistances() const
//{
//    tic(ClassName()+"::NearFieldInteriorPointsSquaredDistances");
//
//    Int sub_calls = 0;
//
//    Int const * restrict const b_row_ptr = near->b_row_ptr.data();
//    Int const * restrict const b_col_ptr = near->b_col_ptr.data();
//    Int const * restrict const b_outer   = near->b_outer.data();
//    Int const * restrict const b_inner   = near->b_inner.data();
//    Int const * restrict const outer     = near->outer.data();
//
//
//    auto values = Tensor1<Real,Int>(near->nnz);
//
//    Real * restrict const squared_dist = values.data();
//
//    #pragma omp parallel num_threads(ThreadCount()) reduction( + : sub_calls )
//    {
//        const Int thread = omp_get_thread_num();
//
//        auto & G = *gjk[thread];
//        auto & P = *S->PrimitivePrototype().Clone();
//        auto & Q = *T->PrimitivePrototype().Clone();
//
//        Real * restrict const X = S->PrimitiveSerializedData ().data();
//        Real * restrict const Y = T->PrimitiveSerializedData ().data();
//
//        const Int b_i_begin = near->job_ptr[thread];
//        const Int b_i_end   = near->job_ptr[thread+1];
//
//        for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
//        {
//            const Int k_begin = b_outer[b_i];
//            const Int k_end   = b_outer[b_i+1];
//
//            const Int i_begin = b_row_ptr[b_i];
//            const Int i_end   = b_row_ptr[b_i+1];
//
//            for( Int i = i_begin; i < i_end; ++i ) // looping over all rows i  in block row b_i
//            {
//                Int ptr = outer[i]; // get first nonzero position in row i; ptr will be used to keep track of the current position within values
//
//                P.SetPointer( X, i );
//
//                for( Int k = k_begin; k < k_end; ++k ) // loop over all blocks in block row b_i
//                {
//                    const Int b_j = b_inner[k]; // we are in block {b_i, b_j} now
//
//                    const Int j_begin = b_col_ptr[b_j];
//                    const Int j_end   = b_col_ptr[b_j+1];
//
//                    for( Int j = j_begin; j < j_end; ++j )
//                    {
//                        Q.SetPointer( Y, j );
//
//                        squared_dist[ptr] = G.InteriorPoints_SquaredDistance( P, Q );
//
//                        sub_calls += G.SubCalls();
//
//                        // Increment ptr, so that the next value is written to the next position.
//                        ++ptr;
//
//                    } // for( Int j = j_begin; j < j_end; ++j )
//
//                } // for( Int k = k_begin; k < k_end; ++k )
//
//            }//for( Int i = i_begin; i < i_end; ++i)
//
//        } // for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
//
//    }
//
//    print("GJK_Algorithm made " + ToString(sub_calls) + " subcalls for n = " + ToString(near->nnz) + " primitive pairs.");
//
//    toc(ClassName()+"::NearFieldInteriorPointsSquaredDistances");
//
//    return values;
//}
//
//
//Tensor1<SReal,Int> NearFieldSquaredDistances() const
//{
//    tic(ClassName()+"::NearFieldSquaredDistances");
//
//    Int sub_calls = 0;
//
//    Int const * restrict const b_row_ptr = near->b_row_ptr.data();
//    Int const * restrict const b_col_ptr = near->b_col_ptr.data();
//    Int const * restrict const b_outer   = near->b_outer.data();
//    Int const * restrict const b_inner   = near->b_inner.data();
//    Int const * restrict const outer     = near->outer.data();
//
//
//    auto values = Tensor1<Real,Int>(near->nnz);
//    Real * restrict const squared_dist = values.data();
//
//    #pragma omp parallel num_threads( ThreadCount() ) reduction( + : sub_calls )
//    {
//        const Int thread = omp_get_thread_num();
//
//        auto & G = *gjk[thread];
//        auto & P = *GetS().PrimitivePrototype().Clone();
//        auto & Q = *GetS().PrimitivePrototype().Clone();
//
//        Real * restrict const X = GetS().PrimitiveSerializedData ().data();
//        Real * restrict const Y = GetS().PrimitiveSerializedData ().data();
//
//        const Int b_i_begin = near->job_ptr[thread];
//        const Int b_i_end   = near->job_ptr[thread+1];
//
//        for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
//        {
//            const Int k_begin = b_outer[b_i];
//            const Int k_end   = b_outer[b_i+1];
//
//            const Int i_begin = b_row_ptr[b_i];
//            const Int i_end   = b_row_ptr[b_i+1];
//
//            for( Int i = i_begin; i < i_end; ++i ) // looping over all rows     i  in block row b_i
//            {
//                Int ptr = outer[i]; // get first nonzero position in row i; ptr     will be used to keep track of the current position within values
//
//                P.SetPointer( X, i );
//
//                for( Int k = k_begin; k < k_end; ++k ) // loop over all blocks in block row b_i
//                {
//                    Int b_j = b_inner[k]; // we are in block {b_i, b_j} now
//
//                    const Int j_begin = b_col_ptr[b_j];
//                    const Int j_end   = b_col_ptr[b_j+1];
//
//                    for( Int j = j_begin; j < j_end; ++j )
//                    {
//                        Q.SetPointer( Y, j );
//
//                        squared_dist[ptr] = G.SquaredDistance( *P, *Q );
//
//                        sub_calls += G.SubCalls();
//
//                        // Increment ptr, so that the next value is written to the next position.
//                        ++ptr;
//
//                    } // for( Int j = j_begin; j < j_end; ++j )
//
//                } // for( Int k = k_begin; k < k_end; ++k )
//
//            }//for( Int i = i_begin; i < i_end; ++i)
//
//        } // for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
//
//    }
//
//    print("GJK_Algorithm made " + ToString(sub_calls) + " subcalls for n = " + ToString(near->n) + " primitive pairs.");
//
//    toc(ClassName()+"::NearFieldSquaredDistances");
//
//    return values;
//}


    void Split_Sequential_DFS( const Int i0, const Int j0 )
    {
//        ptic(ClassName()+"::Split_Sequential_DFS");
        
        const Int thread = omp_get_thread_num();
        
        const Int max_depth = 100;
        Tensor1<Int,Int> i_stack (max_depth);
        Tensor1<Int,Int> j_stack (max_depth);
        Int stack_ptr = 0;
        i_stack[0] = i0;
        j_stack[0] = j0;
        
        std::vector<Int> &  sep_idx =  sep_i[thread];
        std::vector<Int> &  sep_jdx =  sep_j[thread];
        std::vector<Int> & nsep_idx = nsep_i[thread];
        std::vector<Int> & nsep_jdx = nsep_j[thread];
        
        BoundingVolume_T & P = *S_proto[thread];
        BoundingVolume_T & Q = *T_proto[thread];
        GJK_T     & G = *gjk    [thread];
        
        const Int   * restrict const P_left       = S->ClusterLeft().data();
        const Int   * restrict const P_right      = S->ClusterRight().data();
        
        const Int   * restrict const Q_left       = T->ClusterLeft().data();
        const Int   * restrict const Q_right      = T->ClusterRight().data();
        
              SReal * restrict const P_serialized = S->ClusterSerializedData().data();
              SReal * restrict const Q_serialized = T->ClusterSerializedData().data();
    
        const Int   * restrict const S_lookup     = S->leaf_cluster_lookup.data();
        const Int   * restrict const T_lookup     = T->leaf_cluster_lookup.data();
        
        while( (static_cast<Int>(0) <= stack_ptr) && (stack_ptr < max_depth) )
        {
            const Int i = i_stack[stack_ptr];
            const Int j = j_stack[stack_ptr];
            stack_ptr--;
            
//            print("{"+ToString(i)+","+ToString(j)+"}");
            
            P.SetPointer(P_serialized,i);
            Q.SetPointer(Q_serialized,j);
            
            const bool separatedQ = ( IsSymmetric() && (i == j) )
                ? false
                : G.MultipoleAcceptanceCriterion( P, Q, theta2 );
            
            
            if( !separatedQ )
            {
                const Int lefti = P_left[i];
                const Int leftj = Q_left[j];

                // Warning: This assumes that either both children are defined or empty.
                if( lefti >= static_cast<Int>(0) || leftj >= static_cast<Int>(0) )
                {

                    const Int righti = P_right[i];
                    const Int rightj = Q_right[j];
                    
                    const SReal scorei = (lefti >= static_cast<Int>(0))
                        ? P.SquaredRadius()
                        : static_cast<SReal>(0);
                    
                    const SReal scorej = (leftj >= static_cast<Int>(0))
                        ? Q.SquaredRadius()
                        : static_cast<SReal>(0);

                    if( scorei == scorej && scorei > static_cast<SReal>(0) && scorej > static_cast<SReal>(0) )
                    {
                        // tie breaker: split both clusters

                        if( (exploit_symmetry) && (i == j) )
                        {
                            //  Creating 3 blockcluster children, since there is one block is just the mirror of another one.
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = lefti;
                            j_stack[stack_ptr] = rightj;
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = righti;
                            j_stack[stack_ptr] = rightj;
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = lefti;
                            j_stack[stack_ptr] = leftj;
                        }
                        else
                        {
                            // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                            // This happens only if i and j represent_diffent clusters with same radii.
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = righti;
                            j_stack[stack_ptr] = rightj;
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = lefti;
                            j_stack[stack_ptr] = rightj;
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = righti;
                            j_stack[stack_ptr] = leftj;
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = lefti;
                            j_stack[stack_ptr] = leftj;
                            
                        }
                    }
                    else
                    {
                        // split only larger cluster
                        if (scorei > scorej)
                        {
                            ++stack_ptr;
                            i_stack[stack_ptr] = righti;
                            j_stack[stack_ptr] = j;
                            
                            //split cluster i
                            ++stack_ptr;
                            i_stack[stack_ptr] = lefti;
                            j_stack[stack_ptr] = j;
                        }
                        else //scorei < scorej
                        {
                            //split cluster j
                            ++stack_ptr;
                            i_stack[stack_ptr] = i;
                            j_stack[stack_ptr] = rightj;
                            
                            ++stack_ptr;
                            i_stack[stack_ptr] = i;
                            j_stack[stack_ptr] = leftj;
                        }
                    }
                }
                else
                {
                    // create nonsep leaf blockcluster

                    // i and j must be leaves of each ClusterTree S and T, so we directly store their position in the list leaf_clusters. This is important for the sparse matrix generation.

                    //            In know  this is a very deep branching. I optimized it a bit for the case exploit_symmetry != 0 and upper_triangular == 0, though. That seemed to work best in regard of the matrix-vector multiplication.
                    // TODO: Is there a clever way to avoid at least a bit of complixity of this branching? Would that speed up anything in the first place?
        //                    if (exploit_symmetry)
                    if (exploit_symmetry)
                    {
                        if (!near_upper_triangular)
                        {
                            if (i != j)
                            {
                                // Generate also the twin to get a full matrix.
                                nsep_idx.push_back(S_lookup[i]);
                                nsep_idx.push_back(S_lookup[j]);
                                nsep_jdx.push_back(T_lookup[j]);
                                nsep_jdx.push_back(T_lookup[i]);
                            }
                            else
                            {
                                // This is a diagonal block; there is no twin to think about
                                nsep_idx.push_back(T_lookup[i]);
                                nsep_jdx.push_back(S_lookup[i]);
                            }
                        }
                        else
                        {
                            // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                            if (i <= j)
                            {
                                nsep_idx.push_back(S_lookup[i]);
                                nsep_jdx.push_back(T_lookup[j]);
                            }
                            else
                            {
                                nsep_idx.push_back(T_lookup[j]);
                                nsep_jdx.push_back(S_lookup[i]);
                            }
                        }
                    }
                    else
                    {
                        // No symmetry exploited.
                        nsep_idx.push_back(S_lookup[i]);
                        nsep_jdx.push_back(T_lookup[j]);
                    }
                }
            }
            else
            {
                //create sep leaf blockcluster
                if (exploit_symmetry)
                {
                    if (!far_upper_triangular)
                    {
                        // Generate also the twin to get a full matrix
                        sep_idx.push_back(i);
                        sep_idx.push_back(j);
                        sep_jdx.push_back(j);
                        sep_jdx.push_back(i);
                    }
                    else
                    {
                        // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                        if (i <= j)
                        {
                            sep_idx.push_back(i);
                            sep_jdx.push_back(j);
                        }
                        else
                        {
                            sep_idx.push_back(j);
                            sep_jdx.push_back(i);
                        }
                    }
                }
                else
                {
                    // No symmetry exploited.
                    sep_idx.push_back(i);
                    sep_jdx.push_back(j);
                }
            }
        }
//        ptoc(ClassName()+"::Split_Sequential_DFS");
        
    } // Split


    void Split_Parallel( const Int max_leaves )
    {
        ptic(ClassName()+"::Split_Parallel");

        std::deque<Int> i_queue;
        std::deque<Int> j_queue;
        i_queue.push_back(static_cast<Int>(0));
        j_queue.push_back(static_cast<Int>(0));
        
        std::vector<Int> &  sep_idx =  sep_i[0];
        std::vector<Int> &  sep_jdx =  sep_j[0];
        std::vector<Int> & nsep_idx = nsep_i[0];
        std::vector<Int> & nsep_jdx = nsep_j[0];
        
        BoundingVolume_T & P = *S_proto[0];
        BoundingVolume_T & Q = *T_proto[0];
        GJK_T     & G = *gjk[0];
        
        const Int   * restrict const P_left       = S->ClusterLeft().data();
        const Int   * restrict const P_right      = S->ClusterRight().data();
        
        const Int   * restrict const Q_left       = T->ClusterLeft().data();
        const Int   * restrict const Q_right      = T->ClusterRight().data();
        
              SReal * restrict const P_serialized = S->ClusterSerializedData().data();
              SReal * restrict const Q_serialized = T->ClusterSerializedData().data();

        const Int   * restrict const S_lookup     = S->leaf_cluster_lookup.data();
        const Int   * restrict const T_lookup     = T->leaf_cluster_lookup.data();
        
        while( !i_queue.empty() && ( i_queue.size() < max_leaves ) )
        {
            const Int i = i_queue.front();
            const Int j = j_queue.front();
            i_queue.pop_front();
            j_queue.pop_front();

            P.SetPointer( P_serialized, i );
            Q.SetPointer( Q_serialized, j );
            
            bool separatedQ = ( IsSymmetric() && (i == j) )
                ? false
                : G.MultipoleAcceptanceCriterion( P, Q, theta2 );
            
            
            if( !separatedQ )
            {
                const Int lefti = P_left[i];
                const Int leftj = Q_left[j];

                // Warning: This assumes that either both children are defined or empty.
                if( lefti >= static_cast<Int>(0) || leftj >= static_cast<Int>(0) )
                {

                    const Int righti = P_right[i];
                    const Int rightj = Q_right[j];
                    
                    const SReal scorei = (lefti >= static_cast<Int>(0))
                        ? P.SquaredRadius()
                        : static_cast<SReal>(0);
                    
                    const SReal scorej = (leftj >= static_cast<Int>(0))
                        ? Q.SquaredRadius()
                        : static_cast<SReal>(0);

                    if( scorei == scorej && scorei > static_cast<SReal>(0) && scorej > static_cast<SReal>(0) )
                    {
                        // tie breaker: split both clusters

                        if( (exploit_symmetry) && (i == j) )
                        {
                            //  Creating 3 blockcluster children, since there is one block is just the mirror of another one.

                            i_queue.push_back(lefti);
                            j_queue.push_back(rightj);
                            
                            i_queue.push_back(righti);
                            j_queue.push_back(rightj);
                            
                            i_queue.push_back(lefti);
                            j_queue.push_back(leftj);
                        }
                        else
                        {
                            // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                            // This happens only if i and j represent_diffent clusters with same radii.
                            
                            i_queue.push_back(righti);
                            j_queue.push_back(rightj);
                            
                            i_queue.push_back(lefti);
                            j_queue.push_back(rightj);
                            
                            i_queue.push_back(righti);
                            j_queue.push_back(leftj);
                            
                            i_queue.push_back(lefti);
                            j_queue.push_back(leftj);
                        }
                    }
                    else
                    {
                        // split only the larger cluster
                        if (scorei > scorej)
                        {
                            i_queue.push_back(righti);
                            j_queue.push_back(j);
                            
                            //split cluster i
                            i_queue.push_back(lefti);
                            j_queue.push_back(j);
                        }
                        else //scorei < scorej
                        {
                            //split cluster j
                            i_queue.push_back(i);
                            j_queue.push_back(rightj);
                            
                            i_queue.push_back(i);
                            j_queue.push_back(leftj);
                        }
                    }
                }
                else
                {
                    // create nonsep leaf blockcluster

                    // i and j must be leaves of each ClusterTree S and T, so we directly store their position in the list leaf_clusters. This is important for the sparse matrix generation.

                    //            In know  this is a very deep branching. I optimized it a bit for the case exploit_symmetry != 0 and upper_triangular == 0, though. That seemed to work best in regard of the matrix-vector multiplication.
        
                    if (exploit_symmetry)
                    {
                        if (!near_upper_triangular)
                        {
                            if (i != j)
                            {
                                // Generate also the twin to get a full matrix.
                                nsep_idx.push_back(S_lookup[i]);
                                nsep_idx.push_back(S_lookup[j]);
                                nsep_jdx.push_back(T_lookup[j]);
                                nsep_jdx.push_back(T_lookup[i]);
                            }
                            else
                            {
                                // This is a diagonal block; there is no twin to think about
                                nsep_idx.push_back(T_lookup[i]);
                                nsep_jdx.push_back(S_lookup[i]);
                            }
                        }
                        else
                        {
                            // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                            if (i <= j)
                            {
                                nsep_idx.push_back(S_lookup[i]);
                                nsep_jdx.push_back(T_lookup[j]);
                            }
                            else
                            {
                                nsep_idx.push_back(T_lookup[j]);
                                nsep_jdx.push_back(S_lookup[i]);
                            }
                        }
                    }
                    else
                    {
                        // No symmetry exploited.
                        nsep_idx.push_back(S_lookup[i]);
                        nsep_jdx.push_back(T_lookup[j]);
                    }
                }
            }
            else
            {
                //create sep leaf blockcluster
                if (exploit_symmetry)
                {
                    if (!far_upper_triangular)
                    {
                        // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                        if (i <= j)
                        {
                            sep_idx.push_back(i);
                            sep_jdx.push_back(j);
                        }
                        else
                        {
                            sep_idx.push_back(j);
                            sep_jdx.push_back(i);
                        }
                    }
                    else
                    {
                        // Generate also the twin to get a full matrix
                        sep_idx.push_back(i);
                        sep_idx.push_back(j);
                        sep_jdx.push_back(j);
                        sep_jdx.push_back(i);
                    }
                }
                else
                {
                    // No symmetry exploited.
                    sep_idx.push_back(i);
                    sep_jdx.push_back(j);
                }
            }
        }
        
        #pragma omp parallel for num_threads( TreeThreadCount() ) schedule( dynamic )
        for( Int k = 0; k < i_queue.size(); ++k )
        {
            Split_Sequential_DFS(i_queue[k], j_queue[k]);
        }
              
        ptoc(ClassName()+"::Split_Parallel");

    } // Split_Parallel

