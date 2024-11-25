#pragma once

#define CLASS BlockClusterTree
#define BASE  BlockClusterTreeBase<Real,Int,SReal,ExtReal>

#define REPULSION_BLOCKCLUSTERTREE

namespace Repulsion
{
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        ASSERT_FLOAT(ExtReal);
        
    public:
        
        using BlockClusterTreeBase_T = BASE;
        
        using Setting_T = typename BASE::Setting_T;

        using Near_T            = typename BASE::Near_T;
        using  Far_T            = typename BASE::Far_T;
        
        using ClusterTree_T     = ClusterTree<AMB_DIM,Real,Int,SReal,ExtReal>;
        
        using Primitive_T       = typename ClusterTree_T::Primitive_T;
        using BoundingVolume_T  = typename ClusterTree_T::BoundingVolume_T;
        
        using GJK_T             = GJK_Algorithm<AMB_DIM,GJK_Real,Int>;
        
        using BASE::AmbDim;
        using BASE::ThreadCount;
        using BASE::SeparationParameter;
        using BASE::AdaptivityParameter;
        using BASE::IsSymmetric;
        using BASE::ExploitSymmetry;
        using BASE::NearUpperTriangular;
        using BASE::FarUpperTriangular;
        using BASE::SeparatedBlockCount;
        using BASE::NonseparatedBlockCount;
        using BASE::NearFieldInteractionCount;
        using BASE::Far;
        using BASE::Near;
        using BASE::GetS;
        using BASE::GetT;
        using BASE::Settings;
        
    public:

        // In order to prevent GetS() and GetT() shooting a segfault, we have to initialize S and T here. This is the only case in which CLASS owns these raw pointers.
        
        virtual ~CLASS() = default;
        
        CLASS(
            const ClusterTree_T & S_,
            const ClusterTree_T & T_,
            Setting_T settings_ = BlockClusterTreeDefaultSettings
        )
        :   S(S_)
        ,   T(T_)
        ,   settings(settings_)
        ,   near_theta2( pow(settings_.near_field_separation_parameter,2) )
        ,    far_theta2( pow(settings_. far_field_separation_parameter,2) )
        ,   thread_count( std::min(S_.ThreadCount(), T_.ThreadCount()) )
        ,   is_symmetric( std::addressof(S_) == std::addressof(T_) )
        ,   exploit_symmetry( is_symmetric && settings_.exploit_symmetry )
        ,   near_upper_triangular( is_symmetric && settings.near_upper_triangular )
        ,    far_upper_triangular( is_symmetric && settings. far_upper_triangular )
        {
            ptic(ClassName());
            
//            DUMP(settings.near_field_separation_parameter);
//            DUMP(near_theta2);
//
//            DUMP(settings.far_field_separation_parameter);
//            DUMP(far_theta2);
            
            if( std::min( S.PrimitiveCount(), T.PrimitiveCount()) > 0 )
            {
                PreparePrototypes();
                
                ComputeBlocks();
            }

            ptoc(ClassName());
        } // Constructor
        
    protected:

        // Not very elegant to use raw pointers here, but maybe acceptable due to constness.
        const ClusterTree_T & S; // "left"  BVH (output side of matrix-vector multiplication)
        const ClusterTree_T & T; // "right" BVH (input  side of matrix-vector multiplication)
        
        const Setting_T settings;
        
        // Each thread gets its own bounding volume prototype to avoid sharing conflicts.
        // False sharing prevented by alignment of BoundingVolume_T.
        mutable std::vector<std::unique_ptr<BoundingVolume_T>> S_bv;
        mutable std::vector<std::unique_ptr<BoundingVolume_T>> T_bv;
        
        // Adjacency matrices for far and near field.
        mutable  Far_T  far;
        mutable Near_T near;

        mutable std::vector<std::unique_ptr<GJK_T>> gjk;
        
        const SReal near_theta2 = static_cast<SReal>(10);
        const SReal  far_theta2 = static_cast<SReal>(0.25);
        
        const Int thread_count = static_cast<Int>(1);

        const bool is_symmetric = false;
        const bool exploit_symmetry = false;

        const bool near_upper_triangular = false;
        const bool  far_upper_triangular = true;

        mutable bool blocks_initialized = false;
        
        // Containers for parallel aggregation of {i,j}-index pairs of separated and non-separated blocks.
        mutable std::vector<std::vector<Int>>  sep_i;
        mutable std::vector<std::vector<Int>>  sep_j;
        mutable std::vector<std::vector<Int>>  sep_a;
        
        mutable std::vector<std::vector<Int>> nsep_i;
        mutable std::vector<std::vector<Int>> nsep_j;
        mutable std::vector<std::vector<Int>> nsep_a;
        
        mutable std::vector<std::vector<Int>>  col_i;
        mutable std::vector<std::vector<Int>>  col_j;
        mutable std::vector<std::vector<Int>>  col_a;

        mutable bool adaptive_subdivision_data_initialized = false;
        mutable Int collision_count = 0;

        mutable SparseMatrixCSR <Int,Int> to_subdivide;
        mutable SparseMatrixCSR <Int,Int> no_subdivide;
        mutable SparseMatrixCSR <Int,Int> collision_matrix;
        
        Tensor1<Int,Int> by_4;
        Tensor1<Int,Int> by_4_remainder;
        
        Tensor1<Int,Int> by_3;
        Tensor1<Int,Int> by_3_remainder;
        
        Tensor1<Int,Int> by_2;
        
    public:

        Int AmbDim() const override
        {
            return AMB_DIM;
        }
        
        Int ThreadCount() const override
        {
            return thread_count;
        }
        
        Real SeparationParameter() const override
        {
            return sqrt(far_theta2);
        }
                        
        Real AdaptivityParameter() const override
        {
            return sqrt(near_theta2);
        }
        
        bool IsSymmetric() const override
        {
            return is_symmetric;
        }
        
        bool ExploitSymmetry() const override
        {
            return exploit_symmetry;
        }
        
        bool NearUpperTriangular() const override
        {
            return near_upper_triangular;
        }

        bool FarUpperTriangular() const override
        {
            return far_upper_triangular;
        }

        Int SeparatedBlockCount() const override
        {
            return far.NonzeroCount();
        }
        
        Int NonseparatedBlockCount() const override
        {
            return near.BlockNonzeroCount();
        }
        
        Int NearFieldInteractionCount() const override
        {
            return near.NonzeroCount();
        }

        const Far_T & Far() const override
        {
            return far;
        }
        
        const Near_T & Near() const override
        {
            return near;
        }
        
        const ClusterTree_T & GetS() const override
        {
            return S;
        }
        
        const ClusterTree_T & GetT() const override
        {
            return T;
        }

        const Setting_T & Settings() const override
        {
            return settings;
        }
        
        virtual std::string Stats() const override
        {
            std::stringstream s;
            
            s
            << "\n==== "+ClassName()+" Stats ====" << "\n\n"
            << " AmbDim()                    = " <<  AmbDim() << "\n"
            << " SeparationParameter()       = " <<  SeparationParameter() << "\n"
            << " AdaptivityParameter()       = " <<  AdaptivityParameter() << "\n"
            << " ThreadCount()               = " <<  ThreadCount() << "\n"
            << "\n"
            << "Tree S \n"
            << S.Stats() << "\n"
            << "Tree T \n"
            << T.Stats() << "\n"
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
            S_bv = std::vector<std::unique_ptr<BoundingVolume_T>> ( ThreadCount() );
            T_bv = std::vector<std::unique_ptr<BoundingVolume_T>> ( ThreadCount() );
            gjk  = std::vector<std::unique_ptr<GJK_T>>            ( ThreadCount() );
            
            #pragma omp parallel for num_threads(ThreadCount())
            for( Int thread = 0; thread < ThreadCount(); ++thread )
            {
                S_bv[thread] = S.ClusterPrototype().Clone();
                T_bv[thread] = T.ClusterPrototype().Clone();
                gjk[thread]  = std::make_unique<GJK_T>();
            }
            
        } // PreparePrototypes
        
        void ComputeBlocks()
        {
            if( blocks_initialized )
            {
                return;
            }
            
            ptic(ClassName()+"::ComputeBlocks");

             sep_i = std::vector<std::vector<Int>>(ThreadCount());
             sep_j = std::vector<std::vector<Int>>(ThreadCount());
            nsep_i = std::vector<std::vector<Int>>(ThreadCount());
            nsep_j = std::vector<std::vector<Int>>(ThreadCount());
        
            #pragma omp parallel for num_threads(ThreadCount())
            for( Int thread = 0; thread < ThreadCount(); ++thread )
            {
                // TODO: Find a better initial guess for the number of admissable and inadmissable blocks.
                const Int expected = static_cast<Int>(10) * ( S.PrimitiveCount() + T.PrimitiveCount() );

                 sep_i[thread] = std::vector<Int>();
                 sep_j[thread] = std::vector<Int>();
                nsep_i[thread] = std::vector<Int>();
                nsep_j[thread] = std::vector<Int>();
                
                 sep_i[thread].reserve(expected);
                 sep_j[thread].reserve(expected);
                nsep_i[thread].reserve(expected);
                nsep_j[thread].reserve(expected);
            }
            
            DUMP(SeparationParameter());
            
            switch( settings.block_split_method )
            {
                case BlockSplitMethod::Parallel:
                {
                    Split_Parallel( static_cast<Int>(4) * ThreadCount() * ThreadCount() );
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
                    Split_Parallel( static_cast<Int>(4) * ThreadCount() * ThreadCount() );
                }
            }

            ptic(ClassName()+"  Far field interaction data");
            
            far = Far_T(
                sep_i, sep_j,
                S.ClusterCount(), T.ClusterCount(),
                std::min(S.ThreadCount(), T.ThreadCount()),
                false, FarUpperTriangular()
            );
            
            DUMP(far.Stats());
            
            ptoc(ClassName()+"  Far field interaction data");
            
            ptic(ClassName()+"  Near field interaction data");
            
            near = Near_T(
                nsep_i, nsep_j,
                S.LeafClusterPointers(), T.LeafClusterPointers(),
                std::min(S.ThreadCount(), T.ThreadCount()),
                false, NearUpperTriangular()
            );
            
            DUMP(near.Stats());
            
            ptoc(ClassName()+"  Near field interaction data");
            
            
            blocks_initialized = true;
            
            // Free memory that is no longer used.
            #pragma omp parallel for num_threads(ThreadCount())
            for( Int thread = 0; thread < ThreadCount(); ++thread )
            {
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
            by_4           = Tensor1<Int,Int> ( ThreadCount()+1 );
            by_4_remainder = Tensor1<Int,Int> ( ThreadCount()+1 );
            by_3           = Tensor1<Int,Int> ( ThreadCount()+1 );
            by_3_remainder = Tensor1<Int,Int> ( ThreadCount()+1 );
            by_2           = Tensor1<Int,Int> ( ThreadCount()+1 );
            
            for( Int thread = 0; thread < ThreadCount()+1; ++thread )
            {
                by_4          [thread] = thread / static_cast<Int>(4);
                by_4_remainder[thread] = thread % static_cast<Int>(4);

                by_3          [thread] = thread / static_cast<Int>(3);
                by_3_remainder[thread] = thread % static_cast<Int>(3);

                by_2          [thread] = thread / static_cast<Int>(2);
            }
            
            #pragma omp parallel num_threads(ThreadCount())
            {
                #pragma omp single nowait
                {
                    split( static_cast<Int>(0), static_cast<Int>(0), ThreadCount() );
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
            
            BoundingVolume_T & P = *S_bv[thread];

            P.SetPointer( S.ClusterSerializedData().data(), i );

            BoundingVolume_T & Q = *T_bv[thread];
            
            Q.SetPointer( T.ClusterSerializedData().data(), j );

            bool separatedQ = ( IsSymmetric() && (i == j) ) ? false : gjk[thread]->MultipoleAcceptanceCriterion( P, Q, far_theta2 );
            
        
            if( !separatedQ )
            {
                const Int left_i  = S.ClusterLeft()[i];
                const Int left_j  = T.ClusterLeft()[j];

                // Warning: This assumes that either both children are defined or empty.
                if( left_i >= static_cast<Int>(0) || left_j >= static_cast<Int>(0) )
                {
                    const Int right_i = S.ClusterRight()[i];
                    const Int right_j = T.ClusterRight()[j];
                    
                    const SReal score_i = (left_i>=static_cast<Int>(0)) * P.SquaredRadius();
                    const SReal score_j = (left_j>=static_cast<Int>(0)) * Q.SquaredRadius();

                    if( score_i == score_j && score_i > static_cast<SReal>(0) /*&& score_j > static_cast<SReal>(0)*/ )
                    {
                        // tie breaker: split both clusters

                        if( (exploit_symmetry) && (i == j) )
                        {
                            //  Creating 3 blockcluster children, since there is one block is just the mirror of another one.
                            
                            const Int spawncount = by_3          [free_thread_count];
                            const Int remainder  = by_3_remainder[free_thread_count];

                            #pragma omp task final(free_thread_count < 1)
                            split( left_i,  left_j,  spawncount + (remainder > static_cast<Int>(0)) );

                            #pragma omp task final(free_thread_count < 1)
                            split( left_i,  right_j, spawncount + (remainder > static_cast<Int>(1)) );
                            
                            #pragma omp task final(free_thread_count < 1)
                            split( right_i, right_j, spawncount );
                            
                            #pragma omp taskwait
                        }
                        else
                        {
                            // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                            // This happens only if i and j represent_diffent clusters with same radii.
                            
                            const Int spawncount = by_4          [free_thread_count];
                            const Int remainder  = by_4_remainder[free_thread_count];

                            #pragma omp task final(free_thread_count < 1)
                            split( left_i,  left_j,  spawncount + (remainder > static_cast<Int>(0)) );
                            
                            #pragma omp task final(free_thread_count < 1)
                            split( right_i, left_j,  spawncount + (remainder > static_cast<Int>(1)) );
                            
                            #pragma omp task final(free_thread_count < 1)
                            split( left_i,  right_j, spawncount + (remainder > static_cast<Int>(2)) );
                            
                            #pragma omp task final(free_thread_count < 1)
                            split( right_i, right_j, spawncount );
                            
                            #pragma omp taskwait
                        }
                    }
                    else
                    {
                        
                        const Int spawncount = by_2[free_thread_count];
                        
                        // split only larger cluster
                        if (score_i > score_j)
                        {
                            //split cluster i
                            #pragma omp task final(free_thread_count < 1)
                            split( left_i,  j, spawncount );
                            
                            #pragma omp task final(free_thread_count < 1)
                            split( right_i, j, free_thread_count - spawncount );
                            
                            #pragma omp taskwait
                        }
                        else //score_i < score_j
                        {
    //split cluster j
                            #pragma omp task final(free_thread_count < 1)
                            split( i, left_j,  spawncount );
                            
                            #pragma omp task final(free_thread_count < 1)
                            split( i, right_j, free_thread_count - spawncount );
                            
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
                                nsep_i[thread].push_back(S.LeafClusterLookup()[i]);
                                nsep_i[thread].push_back(S.LeafClusterLookup()[j]);
                                nsep_j[thread].push_back(T.LeafClusterLookup()[j]);
                                nsep_j[thread].push_back(T.LeafClusterLookup()[i]);
                            }
                            else
                            {
                                // This is a diagonal block; there is no twin to think about
                                nsep_i[thread].push_back(T.LeafClusterLookup()[i]);
                                nsep_j[thread].push_back(S.LeafClusterLookup()[i]);
                            }
                        }
                        else
                        {
                            // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                            if (i <= j)
                            {
                                nsep_i[thread].push_back(S.LeafClusterLookup()[i]);
                                nsep_j[thread].push_back(T.LeafClusterLookup()[j]);
                            }
                            else
                            {
                                nsep_i[thread].push_back(T.LeafClusterLookup()[j]);
                                nsep_j[thread].push_back(S.LeafClusterLookup()[i]);
                            }
                        }
                    }
                    else
                    {
                        // No symmetry exploited.
                        nsep_i[thread].push_back(S.LeafClusterLookup()[i]);
                        nsep_j[thread].push_back(T.LeafClusterLookup()[j]);
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

        void Split_Sequential_DFS( const Int i0, const Int j0 )
        {
    //        ptic(ClassName()+"::Split_Sequential_DFS");
            
            const Int thread = omp_get_thread_num();
            
            const Int max_depth = 128;
            Int i_stack[128] = {};
            Int j_stack[128] = {};
//            Tensor1<Int,Int> i_stack (max_depth);
//            Tensor1<Int,Int> j_stack (max_depth);
            Int stack_ptr = 0;
            i_stack[0] = i0;
            j_stack[0] = j0;
            
            std::vector<Int> &  sep_idx =  sep_i[thread];
            std::vector<Int> &  sep_jdx =  sep_j[thread];
            std::vector<Int> & nsep_idx = nsep_i[thread];
            std::vector<Int> & nsep_jdx = nsep_j[thread];
            
            BoundingVolume_T & P = *S_bv[thread];
            BoundingVolume_T & Q = *T_bv[thread];
            GJK_T            & G = *gjk [thread];
            
            const Int   * restrict const P_left       = S.ClusterLeft().data();
            const Int   * restrict const P_right      = S.ClusterRight().data();
            
            const Int   * restrict const Q_left       = T.ClusterLeft().data();
            const Int   * restrict const Q_right      = T.ClusterRight().data();
            
                  SReal * restrict const P_serialized = S.ClusterSerializedData().data();
                  SReal * restrict const Q_serialized = T.ClusterSerializedData().data();
        
            const Int   * restrict const S_lookup     = S.LeafClusterLookup().data();
            const Int   * restrict const T_lookup     = T.LeafClusterLookup().data();
            
            while( (static_cast<Int>(0) <= stack_ptr) && (stack_ptr < max_depth) )
            {
                const Int i = i_stack[stack_ptr];
                const Int j = j_stack[stack_ptr];
                stack_ptr--;
                
                P.SetPointer(P_serialized,i);
                Q.SetPointer(Q_serialized,j);
                
                const bool separatedQ = (!( IsSymmetric() && (i == j) )) && G.MultipoleAcceptanceCriterion( P, Q, far_theta2 );
                
                
                if( !separatedQ )
                {
                    const Int left_i = P_left[i];
                    const Int left_j = Q_left[j];

                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= static_cast<Int>(0) || left_j >= static_cast<Int>(0) )
                    {

                        const Int right_i = P_right[i];
                        const Int right_j = Q_right[j];
                        
                        const SReal score_i = (left_i>=static_cast<Int>(0)) * P.SquaredRadius();
                        const SReal score_j = (left_j>=static_cast<Int>(0)) * Q.SquaredRadius();

                        if( score_i == score_j && score_i > static_cast<SReal>(0) /*&& score_j > static_cast<SReal>(0)*/ )
                        {
                            // tie breaker: split both clusters

                            if( (exploit_symmetry) && (i == j) )
                            {
                                //  Creating 3 blockcluster children, since there is one block is just the mirror of another one.
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = left_i;
                                j_stack[stack_ptr] = right_j;
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = right_i;
                                j_stack[stack_ptr] = right_j;
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = left_i;
                                j_stack[stack_ptr] = left_j;
                            }
                            else
                            {
                                // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                                // This happens only if i and j represent_diffent clusters with same radii.
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = right_i;
                                j_stack[stack_ptr] = right_j;
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = left_i;
                                j_stack[stack_ptr] = right_j;
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = right_i;
                                j_stack[stack_ptr] = left_j;
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = left_i;
                                j_stack[stack_ptr] = left_j;
                                
                            }
                        }
                        else
                        {
                            // split only larger cluster
                            if (score_i > score_j)
                            {
                                ++stack_ptr;
                                i_stack[stack_ptr] = right_i;
                                j_stack[stack_ptr] = j;
                                
                                //split cluster i
                                ++stack_ptr;
                                i_stack[stack_ptr] = left_i;
                                j_stack[stack_ptr] = j;
                            }
                            else //score_i < score_j
                            {
                                //split cluster j
                                ++stack_ptr;
                                i_stack[stack_ptr] = i;
                                j_stack[stack_ptr] = right_j;
                                
                                ++stack_ptr;
                                i_stack[stack_ptr] = i;
                                j_stack[stack_ptr] = left_j;
                            }
                        }
                    }
                    else
                    {
                        // create nonsep leaf blockcluster

                        // i and j must be leaves of each ClusterTree S and T, so we directly store their position in the list leaf_clusters. This is important for the sparse matrix generation.

                        //  n know  this is a very deep branching. I optimized it a bit for the case exploit_symmetry != 0 and upper_triangular == 0, though. That seemed to work best in regard of the matrix-vector multiplication.
                        // TODO: Is there a clever way to avoid at least a bit of complixity of this branching? Would that speed up anything in the first place?
                        
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
            
        } // Split_Sequential_DFS


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
            
            BoundingVolume_T & P = *S_bv[0];
            BoundingVolume_T & Q = *T_bv[0];
            GJK_T            & G = *gjk [0];
            
            const Int   * restrict const P_left       = S.ClusterLeft().data();
            const Int   * restrict const P_right      = S.ClusterRight().data();
            
            const Int   * restrict const Q_left       = T.ClusterLeft().data();
            const Int   * restrict const Q_right      = T.ClusterRight().data();
            
                  SReal * restrict const P_serialized = S.ClusterSerializedData().data();
                  SReal * restrict const Q_serialized = T.ClusterSerializedData().data();

            const Int   * restrict const S_lookup     = S.LeafClusterLookup().data();
            const Int   * restrict const T_lookup     = T.LeafClusterLookup().data();
            
            while( !i_queue.empty() && ( static_cast<Int>(i_queue.size()) < max_leaves ) )
            {
                const Int i = i_queue.front();
                const Int j = j_queue.front();
                i_queue.pop_front();
                j_queue.pop_front();

                P.SetPointer( P_serialized, i );
                Q.SetPointer( Q_serialized, j );
                
                bool separatedQ = ( IsSymmetric() && (i == j) )
                    ? false
                    : G.MultipoleAcceptanceCriterion( P, Q, far_theta2 );
                
                
                if( !separatedQ )
                {
                    const Int left_i = P_left[i];
                    const Int left_j = Q_left[j];

                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= static_cast<Int>(0) || left_j >= static_cast<Int>(0) )
                    {

                        const Int right_i = P_right[i];
                        const Int right_j = Q_right[j];
                        
                        const SReal score_i = (left_i>=static_cast<Int>(0)) * P.SquaredRadius();
                        const SReal score_j = (left_j>=static_cast<Int>(0)) * Q.SquaredRadius();
                        
                        if( score_i == score_j && score_i > static_cast<SReal>(0) /*&& score_j > static_cast<SReal>(0)*/ )
                        {
                            // tie breaker: split both clusters

                            if( (exploit_symmetry) && (i == j) )
                            {
                                //  Creating 3 blockcluster children, since there is one block is just the mirror of another one.

                                i_queue.push_back(left_i);
                                j_queue.push_back(right_j);
                                
                                i_queue.push_back(right_i);
                                j_queue.push_back(right_j);
                                
                                i_queue.push_back(left_i);
                                j_queue.push_back(left_j);
                            }
                            else
                            {
                                // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                                // This happens only if i and j represent_diffent clusters with same radii.
                                
                                i_queue.push_back(right_i);
                                j_queue.push_back(right_j);
                                
                                i_queue.push_back(left_i);
                                j_queue.push_back(right_j);
                                
                                i_queue.push_back(right_i);
                                j_queue.push_back(left_j);
                                
                                i_queue.push_back(left_i);
                                j_queue.push_back(left_j);
                            }
                        }
                        else
                        {
                            // split only the larger cluster
                            if (score_i > score_j)
                            {
                                i_queue.push_back(right_i);
                                j_queue.push_back(j);
                                
                                //split cluster i
                                i_queue.push_back(left_i);
                                j_queue.push_back(j);
                            }
                            else //score_i < score_j
                            {
                                //split cluster j
                                i_queue.push_back(i);
                                j_queue.push_back(right_j);
                                
                                i_queue.push_back(i);
                                j_queue.push_back(left_j);
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
            
            #pragma omp parallel for num_threads( ThreadCount() ) schedule( dynamic )
            for( size_t k = 0; k < i_queue.size(); ++k )
            {
                Split_Sequential_DFS(i_queue[k], j_queue[k]);
            }
                  
            ptoc(ClassName()+"::Split_Parallel");

        } // Split_Parallel

            
        
    public:
        
        void RequireAdaptiveSubdivisionData() const override
        {
            if( adaptive_subdivision_data_initialized )
            {
                return;
            }
            
            ptic(ClassName()+"::RequireAdaptiveSubdivisionData");
            
            if( near.NonzeroCount() <= 0 )
            {
                ptoc(ClassName()+"::RequireAdaptiveSubdivisionData");
                return;
            }
            
            // cppcheck-suppress [knownConditionTrueFalse]
            const bool uppertriangular = NearUpperTriangular();
            
            // cppcheck-suppress [knownConditionTrueFalse]
            const Tensor1<Int,Int> & job_ptr = uppertriangular
                ? near.UpperTriangularJobPtr()
                : near.JobPtr();
            
            // cppcheck-suppress [knownConditionTrueFalse]
            Int const * restrict const row_begin = uppertriangular
                ? near.Diag ().data()
                : near.Outer().data();

            Int const * restrict const row_end = near.Outer().data()+1;
            Int const * restrict const inner   = near.Inner().data();

            const Int thread_count_ = job_ptr.Size()-1;
            
             col_i = std::vector<std::vector<Int>>(thread_count_);
             col_j = std::vector<std::vector<Int>>(thread_count_);
             col_a = std::vector<std::vector<Int>>(thread_count_);
            
             sep_a = std::vector<std::vector<Int>>(thread_count_);
            nsep_a = std::vector<std::vector<Int>>(thread_count_);
            
            if( IsSymmetric() )
            {
                //Ensure that the adjacency matrix is constructed before the parallel region.
                logprint("Tree is symmetric.");
                (void)S.PrimitiveAdjacencyMatrix();
            }
            else
            {
                logprint("Tree is not symmetric.");
            }
            
            DUMP(AdaptivityParameter());
            
            const Real intersection_theta2 = pow( settings.near_field_collision_parameter, 2 );
            
            collision_count = 0;
        
            #pragma omp parallel for num_threads( thread_count_ ) reduction( + : collision_count )
            for( Int thread = 0; thread < thread_count_; ++thread )
            {
                SReal * restrict const S_serialized = GetS().PrimitiveSerializedData().data();
                SReal * restrict const T_serialized = GetT().PrimitiveSerializedData().data();
                
//                 col_i[thread] = std::vector<Int>();
//                 col_j[thread] = std::vector<Int>();
//                 col_a[thread] = std::vector<Int>();
                
                 sep_i[thread] = std::vector<Int>();
                 sep_j[thread] = std::vector<Int>();
//                 sep_a[thread] = std::vector<Int>();
                
                nsep_i[thread] = std::vector<Int>();
                nsep_j[thread] = std::vector<Int>();
//                nsep_a[thread] = std::vector<Int>();
                
                std::vector<Int> &  col_idx =  col_i[thread];
                std::vector<Int> &  col_jdx =  col_j[thread];
                std::vector<Int> &  col_val =  col_a[thread];
                
                std::vector<Int> &  sep_idx =  sep_i[thread];
                std::vector<Int> &  sep_jdx =  sep_j[thread];
                std::vector<Int> &  sep_val =  sep_a[thread];
                
                std::vector<Int> & nsep_idx = nsep_i[thread];
                std::vector<Int> & nsep_jdx = nsep_j[thread];
                std::vector<Int> & nsep_val = nsep_a[thread];

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                std::unique_ptr<Primitive_T> P_ptr = S.PrimitivePrototype().Clone();
                std::unique_ptr<Primitive_T> Q_ptr = T.PrimitivePrototype().Clone();
                
                Primitive_T & P = *P_ptr;
                Primitive_T & Q = *Q_ptr;
                GJK_T       & G = *gjk[thread];
                
                // cppcheck-suppress [knownConditionTrueFalse]
                if( IsSymmetric() )
                {
                    const SparseBinaryMatrixCSR<Int> & A = S.PrimitiveAdjacencyMatrix();
                          
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                      if ( i < 0) {
                        std::cerr << " Gotcha! negative i in RequireAdaptiveSubdivisionData!" << std::endl;
                      }
                        P.SetPointer(S_serialized,i);

                        const Int k_begin = row_begin[i];
                        const Int k_end   = row_end  [i];
                        
                        for (Int k = k_begin; k < k_end; ++k )
                        {
                            if ( k < 0) {
                              std::cerr << " Gotcha! negative k in RequireAdaptiveSubdivisionData!" << std::endl;
                            }
                            const Int j = inner[k];
                            
                            Q.SetPointer(T_serialized,j);
                            
                            const bool neighbor_found = A.FindNonzeroPosition(i,j) >= 0;
                            
                            if( neighbor_found )
                            {
//                                logprint("neighbors");
                                sep_idx.push_back(i);
                                sep_jdx.push_back(j);
                                sep_val.push_back(k);
                            }
                            else
                            {
                                const bool admissable = G.MultipoleAcceptanceCriterion( P, Q, near_theta2);
                                
                                if( admissable )
                                {
//                                    logprint("admissable");
                                    sep_idx.push_back(i);
                                    sep_jdx.push_back(j);
                                    sep_val.push_back(k);
                                }
                                else
                                {
//                                    logprint("inadmissable");
                                    const bool intersecting = (!G.SeparatedQ()) && G.IntersectingQ( P, Q, intersection_theta2 );
                                    
                                    if( intersecting )
                                    {
                                        col_idx.push_back(i);
                                        col_jdx.push_back(j);
                                        col_val.push_back(k);
                                    }
                                    else
                                    {
                                        nsep_idx.push_back(i);
                                        nsep_jdx.push_back(j);
                                        nsep_val.push_back(k);
                                    }
                                }
                            }
                            
                        } // for (Int k = k_begin; k < k_end; ++k)
                        
                    } // for( Int i = i_begin; i < i_end; ++i )
                }
                else
                {
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                      if ( i < 0) {
                        std::cerr << " Gotcha! negative i in RequireAdaptiveSubdivisionData (nonsym)!" << std::endl;
                      }
                        P.SetPointer(S_serialized,i);

                        const Int k_begin = row_begin[i];
                        const Int k_end   = row_end  [i];
                        
                        for (Int k = k_begin; k < k_end; ++k )
                        {
                          if ( k < 0) {
                            std::cerr << " Gotcha! negative k in RequireAdaptiveSubdivisionData (nonsym)!" << std::endl;
                          }
                            const Int j = inner[k];
                            
                            Q.SetPointer(T_serialized,j);
                            
                            const bool admissable = G.MultipoleAcceptanceCriterion( P, Q, near_theta2);
                            
                            if( admissable )
                            {
//                                    logprint("admissable");
                                sep_idx.push_back(i);
                                sep_jdx.push_back(j);
                                sep_val.push_back(k);
                            }
                            else
                            {
//                                    logprint("inadmissable");
                                const bool intersecting = (!G.SeparatedQ()) && G.IntersectingQ( P, Q, intersection_theta2 );
                                
                                
                                if( intersecting )
                                {
                                    col_idx.push_back(i);
                                    col_jdx.push_back(j);
                                    col_val.push_back(k);
                                }
                                else
                                {
                                    nsep_idx.push_back(i);
                                    nsep_jdx.push_back(j);
                                    nsep_val.push_back(k);
                                }
                            }
                            
                        } // for (Int k = k_begin; k < k_end; ++k)
                        
                    } // for( Int i = i_begin; i < i_end; ++i )
                }
                
            } // #pragma omp parallel

            no_subdivide     = SparseMatrixCSR <Int,Int>(
                 sep_i,  sep_j,  sep_a, near.RowCount(), near.ColCount(), near.ThreadCount(), false, 0 );
            
            to_subdivide     = SparseMatrixCSR <Int,Int>(
                nsep_i, nsep_j, nsep_a, near.RowCount(), near.ColCount(), near.ThreadCount(), false, 0 );
            
            collision_matrix = SparseMatrixCSR<Int,Int>(
                 col_i,  col_j,  col_a, near.RowCount(), near.ColCount(), near.ThreadCount(), false, 0 );
            
            
            if( collision_matrix.Inner().Size() > 0 )
            {
                wprint(ClassName()+"::RequireAdaptiveSubdivisionData: "+ToString(collision_matrix.Inner().Size())+" collisions detected.");
                
                // for( Int thread = 0; thread < thread_count_; ++thread ) {
                //     for ( Int iii = 0; iii < col_i[thread].size(); iii++) {
                //         wprint(ClassName()+"::RequireAdaptiveSubdivisionData: Collision: "+ToString(col_i[thread][iii])+" - "+ToString(col_j[thread][iii])+" - "+ToString(col_a[thread][iii])+" ");
                //     }
                // }
                
            }
            
            #pragma omp parallel for num_threads(thread_count_)
            for( Int thread = 0; thread < thread_count_; ++thread )
            {
                col_i[thread] = std::vector<Int>();
                col_j[thread] = std::vector<Int>();
                col_a[thread] = std::vector<Int>();
                
                sep_i[thread] = std::vector<Int>();
                sep_j[thread] = std::vector<Int>();
                sep_a[thread] = std::vector<Int>();
                
               nsep_i[thread] = std::vector<Int>();
               nsep_j[thread] = std::vector<Int>();
               nsep_a[thread] = std::vector<Int>();
            }

            
            adaptive_subdivision_data_initialized = true;
            
            ptoc(ClassName()+"::RequireAdaptiveSubdivisionData");
        }


        Int CollisionCount() const override
        {
            RequireAdaptiveSubdivisionData();
            return collision_matrix.NonzeroCount();
            
        }
        
        const SparseMatrixCSR<Int,Int> & CollisionMatrix() const override
        {
            RequireAdaptiveSubdivisionData();
            return collision_matrix;
            
        }
        
        const SparseMatrixCSR<Int,Int> & AdaptiveNoSubdivisionData() const override
        {
            RequireAdaptiveSubdivisionData();
            return no_subdivide;
            
        }
        
        const SparseMatrixCSR<Int,Int> & AdaptiveSubdivisionData() const override
        {
            RequireAdaptiveSubdivisionData();
            return to_subdivide;
        }
        
        
        std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS) + "<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
      
        
    };
    
} //namespace Repulsion

#undef BASE
#undef CLASS
