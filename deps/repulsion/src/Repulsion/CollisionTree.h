#pragma once

#define CLASS CollisionTree
#define BASE CollisionTreeBase<Real,Int,SReal,ExtReal>

namespace Repulsion
{
    
    template <int AMB_DIM, typename SReal, typename Int>
    inline void AABB_CollisionTimeInterval(
       const SReal * restrict const p_0,
       const SReal * restrict const p_1,
       const SReal * restrict const q_0,
       const SReal * restrict const q_1,
       SReal & t_first,    // returning per reference to avoid std::pair
       SReal & t_last      // returning per reference to avoid std::pair
    )
    {
//        GJK_tic("AABB_CollisionTimeInterval");
        
        static constexpr SReal zero = static_cast<SReal>(0);
        static constexpr SReal one  = static_cast<SReal>(1);
        
        // Assuming p_0, p_1, q_0, q_1 are serizalizations in the format of some AABB<AMB_DIM,GJK_Real,Int,SReal>

        
        // Returns interval {t_first,t_last} of those t such that the AABBs defined by
        //     P(t) = (1-t) * p_0 + t * p_1
        // and
        //     Q(t) = (1-t) * q_0 + t * q_1
        // intersect.
    
        SReal t_0 = zero;
        SReal t_1 = one;
        
        for( Int k = 0; k < AMB_DIM; ++k )
        {
            // Compute the intervals of the four AABB in the k-th coordinate direction.
            
            const Int i = 1+k;
            const Int j = 1+k+AMB_DIM;
            
            const SReal P_min_0 = p_0[i] - p_0[j];
            const SReal P_max_0 = p_0[i] + p_0[j];
            
            const SReal P_min_1 = p_1[i] - p_1[j];
            const SReal P_max_1 = p_1[i] + p_1[j];
            
            const SReal Q_min_0 = q_0[i] - q_0[j];
            const SReal Q_max_0 = q_0[i] + q_0[j];
            
            const SReal Q_min_1 = q_1[i] - q_1[j];
            const SReal Q_max_1 = q_1[i] + q_1[j];
            
            bool initially_intersecting = (P_min_0 <= Q_max_0) && (Q_min_0 <= P_max_0);
//            bool finally_intersecting   = (P_min_1 <= Q_max_1) && (Q_min_1 <= P_max_1);
            
            const SReal m_0 = Q_max_0 - Q_max_1 - P_min_0 + P_min_1;
            const SReal m_1 = P_max_0 - P_max_1 - Q_min_0 + Q_min_1;
            
            SReal A = ( m_0 != zero ) ? std::min(one,(Q_max_0 - P_min_0)/m_0) : one;
            SReal B = ( m_1 != zero ) ? std::min(one,(P_max_0 - Q_min_0)/m_1) : one;
            
            if( A > B )
            {
                std::swap(A,B);
            }
            
            SReal a;
            SReal b;
            
//            GJK_DUMP(initially_intersecting);
//            GJK_DUMP(A);
//            GJK_DUMP(B);
            
            if( initially_intersecting )
            {
                a = zero;
                b = (A>zero) ? A : ( (B<=zero) ? one : B );
            }
            else
            {
                // initially_intersecting == false
                
                if( /*A < zero &&*/ B < zero )
                {
                    a = one;
                    b = one;
                }
                else
                {
                    if( A >= zero/* && B >= zero*/ )
                    {
                        a = A;
                        b = B;
                    }
                    else
                    {
                        a = B;
                        b = one;
                    }
                }
            }
            
            t_0 = std::max(t_0,a);
            t_1 = std::min(t_1,b);
        }
        
        if( t_1 < t_0 )
        {
            // empty intersection interval
            t_first = one;
            t_last  = one;
        }
        else
        {
            t_first = t_0;
            t_last  = t_1;
        }
 
//        GJK_toc("AABB_CollisionTimeInterval");
    }
    
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        ASSERT_FLOAT(ExtReal);        
        
    public:
        
        using ClusterTreeBase_T = BASE;
        
        using ClusterTree_T     = ClusterTree<AMB_DIM,Real,Int,SReal,ExtReal>;
        using CollisionMatrix_T = typename ClusterTreeBase_T::CollisionMatrix_T;
        
        using Primitive_T       = typename ClusterTree_T::Primitive_T;
        using MovingPrimitive_T = typename ClusterTree_T::MovingPrimitive_T;
        using BoundingVolume_T  = typename ClusterTree_T::BoundingVolume_T;
        
        using PrimitiveCollisionFinder_T  = CollisionFinder<AMB_DIM,Real,Int,SReal>;
        
        
        CLASS( const ClusterTree_T & S_, const ClusterTree_T & T_, const SReal t_init_ )
        :   S( S_ )
        ,   T( T_ )
        ,   t_init(t_init_)
        ,   thread_count( std::min(S_.ThreadCount(), T_.ThreadCount()) )
        ,   is_symmetric( std::addressof(S_) == std::addressof(T_) )
        ,   collision_i( thread_count )
        ,   collision_j( thread_count )
        ,   collision_t( thread_count )
        {
            ptic(ClassName()+"()");
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                collision_i[thread] = std::vector<Int>();
                collision_j[thread] = std::vector<Int>();
                collision_t[thread] = std::vector<SReal>();
            }
            ptoc(ClassName()+"()");
        }
        
//        // Copy constructor
//        CLASS( const CLASS & other )
//        :   S( other.S_ )
//        ,   T( other.T_ )
//        ,   t_init(other.t_init_)
//        ,   thread_count( other.thread_count )
//        ,   is_symmetric( other.is_symmetric )
//        ,   collision_i( thread_count )
//        ,   collision_j( thread_count )
//        ,   collision_t( thread_count )
//        {}
        
        
        virtual ~CLASS() = default;
        
    protected:
        
        // Not very elegant to use raw pointers here, but maybe acceptable due to constness.
//        const ClusterTree_T * S = nullptr; // "left"  BVH (output side of matrix-vector multiplication)
//        const ClusterTree_T * T = nullptr; // "right" BVH (input  side of matrix-vector multiplication)

        const ClusterTree_T & S; // "left"  BVH (output side of matrix-vector multiplication)
        const ClusterTree_T & T; // "right" BVH (input  side of matrix-vector multiplication)

        
        const SReal t_init = 1;
        
        const Int thread_count = 1;
        
        const bool is_symmetric = false;
        
        mutable std::deque<Int> i_queue;
        mutable std::deque<Int> j_queue;
        
        mutable std::vector<std::vector<Int>>   collision_i;
        mutable std::vector<std::vector<Int>>   collision_j;
        mutable std::vector<std::vector<SReal>> collision_t;
        
        mutable bool P_collision_matrix_initialized = false;
        mutable CollisionMatrix_T P_collision_matrix;
        mutable bool C_collision_matrix_initialized = false;
        mutable CollisionMatrix_T C_collision_matrix;
        
        static constexpr Int max_depth = 128;
        static constexpr Int null = static_cast<Int>(0);
        
    protected:
        
//####################################################################################################
//      FindCollidingPrimitives
//####################################################################################################
        
        void FindCollidingPrimitives() const
        {
            ptic(ClassName()+"::FindCollidingPrimitives");
            
            auto & C_matrix = ClusterCollisionMatrix();
            
            if( C_matrix.NonzeroCount() <= 0 )
            {
                P_collision_matrix = CollisionMatrix_T(0,0,0,1);
                
                DUMP(P_collision_matrix.Stats());
                
                ptoc(ClassName()+"::FindCollidingPrimitives");
                return;
            }
            
            
            DUMP(is_symmetric);
            
            const Int expected = is_symmetric ?
                    (S.PrimitiveCount() + S.PrimitiveAdjacencyMatrix().NonzeroCount()) : 0;
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                collision_i[thread] = std::vector<Int>();
                collision_j[thread] = std::vector<Int>();
                collision_t[thread] = std::vector<SReal>();
                
                collision_i[thread].reserve(expected);
                collision_j[thread].reserve(expected);
                collision_t[thread].reserve(expected);
            }
            
            Tensor1<Int,Int> job_ptr = C_matrix.JobPtr();
            
            const Int * restrict const S_begin = S.ClusterBegin().data();
            const Int * restrict const S_end   = S.ClusterEnd().data();
            const Int * restrict const T_begin = T.ClusterBegin().data();
            const Int * restrict const T_end   = T.ClusterEnd().data();
            
            const Tensor2<SReal,Int> & p = S.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & u = S.PrimitiveVelocitiesSerializedData();
            const Tensor2<SReal,Int> & q = T.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & v = T.PrimitiveVelocitiesSerializedData();
                
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                const Int b_i_begin = job_ptr[thread  ];
                const Int b_i_end   = job_ptr[thread+1];
                
                const Int * restrict const outer = C_matrix.Outer().data();
                const Int * restrict const inner = C_matrix.Inner().data();
                
                PrimitiveCollisionFinder_T C ( T.MovingPrimitivePrototype(), T.MovingPrimitivePrototype() );
                
                auto & collision_idx  = collision_i[thread];
                auto & collision_jdx  = collision_j[thread];
                auto & collision_time = collision_t[thread];
                
                if( is_symmetric )
                {
                    const SparseBinaryMatrixCSR<Int> & A = S.PrimitiveAdjacencyMatrix();
                    
                    for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
                    {
                        const Int b_k_begin = outer[b_i  ];
                        const Int b_k_end   = outer[b_i+1];
                        
                        for( Int b_k = b_k_begin; b_k < b_k_end; ++b_k )
                        {
                            const Int b_j = inner[b_k];
                            
                            const Int i_begin = S_begin[b_i];
                            const Int i_end   = S_end  [b_i];
                            
                            const Int j_end   = T_end  [b_j];
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const Int j_begin = (b_i==b_j) ? i+1 : T_begin[b_j];
                                
                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    if( A.FindNonzeroPosition(i,j) < 0 )
                                    {
                                        const SReal t = C.FindMaximumSafeStepSize(
                                            p.data(i), u.data(i),
                                            q.data(j), v.data(j),
                                            t_init, false
                                        );
                                        
                                        if( t < t_init )
                                        {
                                            collision_idx.push_back(i);
                                            collision_jdx.push_back(j);
                                            collision_time.push_back(t);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    //TODO: Code for asymmetric matrices has yet to be debugged.
                    wprint(ClassName()+"FindCollidingPrimitives: Code for asymmetric matrices has yet to be debugged.");
                    for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
                    {
                        const Int b_k_begin = outer[b_i  ];
                        const Int b_k_end   = outer[b_i+1];
                        
                        for( Int b_k = b_k_begin; b_k < b_k_end; ++b_k )
                        {
                            const Int b_j = inner[b_k];
                            
                            const Int i_begin = S_begin[b_i];
                            const Int i_end   = S_end  [b_i];
                            
                            const Int j_end   = T_end  [b_j];
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const Int j_begin = (b_i==b_j) ? i+1 : T_begin[b_j];
                                
                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    const SReal t = C.FindMaximumSafeStepSize(
                                        p.data(i), u.data(i),
                                        q.data(j), v.data(j),
                                        t_init, false
                                    );
                                    
                                    if( t < t_init )
                                    {
                                        collision_idx.push_back(i);
                                        collision_jdx.push_back(j);
                                        collision_time.push_back(t);
                                    }
                                }
                            }
                        }
                    }
                } // if( is_symmetric )
                
            } // #pragma omp parallel for num_threads( job_ptr.Size()-1 )

            
            P_collision_matrix = CollisionMatrix_T(
                collision_i, collision_j, collision_t,
                S.PrimitiveCount(), T.PrimitiveCount(),
                std::min(S.ThreadCount(), T.ThreadCount()),
                false, false
            );
            
            DUMP(P_collision_matrix.Stats());
            // Free memory that is no longer used.
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                collision_i[thread] = std::vector<Int>();
                collision_j[thread] = std::vector<Int>();
                collision_t[thread] = std::vector<SReal>();
            }
            
            ptoc(ClassName()+"::FindCollidingPrimitives");
        }
        
//####################################################################################################
//      FindCollidingClusters
//####################################################################################################
        
        void FindCollidingClusters() const
        {
            ptic(ClassName()+"::FindCollidingClusters");
            
            const Int expected = S.LeafClusterCount() + T.LeafClusterCount();
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                collision_i[thread] = std::vector<Int>();
                collision_j[thread] = std::vector<Int>();
                collision_t[thread] = std::vector<SReal>();
                
                collision_i[thread].reserve(expected);
                collision_j[thread].reserve(expected);
                collision_t[thread].reserve(expected);
            }
            
//            FindCollidingClusters_Sequential();
            
            FindCollidingClusters_Parallel();
            
            
            C_collision_matrix = CollisionMatrix_T(
                collision_i, collision_j,collision_t,
                S.ClusterCount(), T.ClusterCount(),
                std::min( S.ThreadCount(), T.ThreadCount() ),
                false, false
            );
            
            DUMP(C_collision_matrix.Stats());
            // Free memory that is no longer used.
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                collision_i[thread] = std::vector<Int>();
                collision_j[thread] = std::vector<Int>();
                collision_t[thread] = std::vector<SReal>();
            }
            
            ptoc(ClassName()+"::FindCollidingClusters");
        }
        
//####################################################################################################
//      FindCollidingClusters_Sequential
//####################################################################################################
                
        void FindCollidingClusters_Sequential() const
        {
            ptic(ClassName()+"::FindCollidingClusters_Sequential");
            
            FindCollidingClusters_Sequential_DFS(0,0);
            
            ptoc(ClassName()+"::FindCollidingClusters_Sequential");
        }
                
        void FindCollidingClusters_Sequential_DFS( const Int i_0, const Int j_0 ) const
        {
            //            ptic(ClassName()+"::FindCollidingClusters_Sequential_DFS");
            
            const Int thread = omp_get_thread_num();
            
            Int i_stack[max_depth] = {};
            Int j_stack[max_depth] = {};
            
            Int stack_ptr = null;
            i_stack[0] = i_0;
            j_stack[0] = j_0;
            
            std::vector<Int>   & collision_idx  = collision_i[thread];
            std::vector<Int>   & collision_jdx  = collision_j[thread];
            std::vector<SReal> & collision_time = collision_t[thread];
            
            const Int * restrict const S_left   = S.ClusterLeft().data();
            const Int * restrict const S_right  = S.ClusterRight().data();
            
            const Int * restrict const T_left   = T.ClusterLeft().data();
            const Int * restrict const T_right  = T.ClusterRight().data();
            
            const Tensor2<SReal,Int> & S_C_ser    = S.ClusterSerializedData();
            const Tensor2<SReal,Int> & S_C_up_ser = S.ClusterUpdatedSerializedData();
            const Tensor2<SReal,Int> & T_C_ser    = T.ClusterSerializedData();
            const Tensor2<SReal,Int> & T_C_up_ser = T.ClusterUpdatedSerializedData();
            //
            SReal a = static_cast<SReal>(0);
            SReal b = static_cast<SReal>(1);
            
            while( (null <= stack_ptr) && (stack_ptr < max_depth - 4) )
            {
                const Int i = i_stack[stack_ptr];
                const Int j = j_stack[stack_ptr];
                stack_ptr--;
                
                AABB_CollisionTimeInterval<AMB_DIM, SReal, Int> (
                    S_C_ser.data(i), S_C_up_ser.data(i),
                    T_C_ser.data(j), T_C_up_ser.data(j),
                    a, b
                );
                
                if( a < static_cast<SReal>(1) )
                {
                    const Int left_i = S_left[i];
                    const Int left_j = T_left[j];
                    
                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= null || left_j >= null )
                    {
                        
                        const Int right_i = S_right[i];
                        const Int right_j = T_right[j];
                        
                        // TODO: Improve score.
                        
                        const SReal score_i = static_cast<SReal>(left_i >= null) * S_C_ser(i,0);
                        const SReal score_j = static_cast<SReal>(left_j >= null) * T_C_ser(j,0);
                        
                        if( (score_i == score_j) /*&& (score_i > static_cast<SReal>(0)) && score_j > static_cast<SReal>(0)*/ )
                        {
                            if( (is_symmetric) && ( i == j ) )
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
                                // tie breaker: split both clusters
                                
                                // This is a very seldom case; still required to preserve symmetry.
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
                        collision_time.push_back(a * t_init);
                        
                        if( is_symmetric )
                        {
                            // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                            if (i <= j)
                            {
                                collision_idx.push_back(i);
                                collision_jdx.push_back(j);
                            }
                            else
                            {
                                collision_idx.push_back(j);
                                collision_jdx.push_back(i);
                            }
                        }
                        else
                        {
                            // No symmetry exploited.
                            collision_idx.push_back(i);
                            collision_jdx.push_back(j);
                        }
                    }
                }
            }
            //            ptoc(ClassName()+"::FindCollidingClusters_Sequential_DFS");
            
        } // FindCollidingClusters_Sequential_DFS
        
//####################################################################################################
//      FindCollidingClusters_Parallel
//####################################################################################################
        
        void FindCollidingClusters_Parallel() const
        {
            ptic(ClassName()+"::FindCollidingClusters_Parallel");
            
            i_queue = std::deque<Int>();
            j_queue = std::deque<Int>();
            
            FindCollidingClusters_BFS( static_cast<Int>(4) * ThreadCount() * ThreadCount() );
            
            #pragma omp parallel for num_threads( thread_count ) schedule( dynamic )
            for( size_t k = 0; k < i_queue.size(); ++k )
            {
                FindCollidingClusters_Sequential_DFS(i_queue[k], j_queue[k]);
            }
            
            i_queue = std::deque<Int>();
            j_queue = std::deque<Int>();
            
            ptoc(ClassName()+"::FindCollidingClusters_Parallel");
        }
        
        void FindCollidingClusters_BFS( const Int max_leaves ) const
        {
            i_queue.push_back(null);
            j_queue.push_back(null);
            
            std::vector<Int>   & collision_idx  = collision_i[0];
            std::vector<Int>   & collision_jdx  = collision_j[0];
            std::vector<SReal> & collision_time = collision_t[0];

            const Int * restrict const S_left   = S.ClusterLeft().data();
            const Int * restrict const S_right  = S.ClusterRight().data();
            const Int * restrict const T_left   = T.ClusterLeft().data();
            const Int * restrict const T_right  = T.ClusterRight().data();
            
            const Tensor2<SReal,Int> & S_C_ser    = S.ClusterSerializedData();
            const Tensor2<SReal,Int> & S_C_up_ser = S.ClusterUpdatedSerializedData();
            const Tensor2<SReal,Int> & T_C_ser    = T.ClusterSerializedData();
            const Tensor2<SReal,Int> & T_C_up_ser = T.ClusterUpdatedSerializedData();
//
            SReal a = static_cast<SReal>(0);
            SReal b = static_cast<SReal>(1);
            
            while( !i_queue.empty() && ( static_cast<Int>(i_queue.size()) < max_leaves ) )
            {
                const Int i = i_queue.front();
                const Int j = j_queue.front();
                i_queue.pop_front();
                j_queue.pop_front();

                AABB_CollisionTimeInterval<AMB_DIM, SReal, Int> (
                    S_C_ser.data(i), S_C_up_ser.data(i),
                    T_C_ser.data(j), T_C_up_ser.data(j),
                    a, b
                );
                
                if( a < static_cast<SReal>(1) )
                {
                    const Int left_i = S_left[i];
                    const Int left_j = T_left[j];

                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= null || left_j >= null )
                    {

                        const Int right_i = S_right[i];
                        const Int right_j = T_right[j];
                        
// TODO: Improve score.
                                                
                        const SReal score_i = (left_i>=null) * S_C_ser(i,0);
                        const SReal score_j = (left_j>=null) * T_C_ser(j,0);
                        
                        if( score_i == score_j )
                        {
                            // tie breaker: split both clusters

                            if( (is_symmetric) && (i == j) )
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
                        collision_time.push_back(a * t_init);
                        
                        if( is_symmetric )
                        {
                            // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                            if (i <= j)
                            {
                                collision_idx.push_back(i);
                                collision_jdx.push_back(j);
                            }
                            else
                            {
                                collision_idx.push_back(j);
                                collision_jdx.push_back(i);
                            }
                        }
                        else
                        {
                            // No symmetry exploited.
                            collision_idx.push_back(i);
                            collision_jdx.push_back(j);
                        }
                    }
                }
            }
        } // FindCollidingClusters_BFS
        
        
//####################################################################################################
//      FindMaximumSafeStepSize
//####################################################################################################
        
    public:
        
        SReal MaximumSafeStepSize() const override
        {
            ptic(ClassName()+"::MaximumSafeStepSize");
            
//            SReal t = MaximumSafeStepSize_Sequential();
            
            SReal t = MaximumSafeStepSize_Parallel();
            
            ptoc(ClassName()+"::MaximumSafeStepSize");
            
            return t;
        }
        
//####################################################################################################
//      MaximumSafeStepSize_Sequential
//####################################################################################################
               
    protected:
        
        SReal MaximumSafeStepSize_Sequential() const
        {
            ptic(ClassName()+"::MaximumSafeStepSize_Sequential");
            
            SReal t;
            
            if( is_symmetric )
            {
                t = MaximumSafeStepSize_DFS_Symmetric(0,0,t_init);
            }
            else
            {
                t = MaximumSafeStepSize_DFS_Asymmetric(0,0,t_init);
            }
            
            ptoc(ClassName()+"::MaximumSafeStepSize_Sequential");
            
            return t;
        }
                
        SReal MaximumSafeStepSize_DFS_Symmetric(
                const Int i_0, const Int j_0, const SReal t_init_ ) const
        {
            Int i_stack[max_depth] = {};
            Int j_stack[max_depth] = {};
            
            Int stack_ptr = null;
            i_stack[0] = i_0;
            j_stack[0] = j_0;
            
            const Int * restrict const S_left     = S.ClusterLeft().data();
            const Int * restrict const S_right    = S.ClusterRight().data();
            const Int * restrict const T_left     = T.ClusterLeft().data();
            const Int * restrict const T_right    = T.ClusterRight().data();
            
            const Int * restrict const S_begin    = S.ClusterBegin().data();
            const Int * restrict const S_end      = S.ClusterEnd().data();
            const Int * restrict const T_begin    = T.ClusterBegin().data();
            const Int * restrict const T_end      = T.ClusterEnd().data();
            
            const Tensor2<SReal,Int> & S_C_ser    = S.ClusterSerializedData();
            const Tensor2<SReal,Int> & S_C_up_ser = S.ClusterUpdatedSerializedData();
            const Tensor2<SReal,Int> & T_C_ser    = T.ClusterSerializedData();
            const Tensor2<SReal,Int> & T_C_up_ser = T.ClusterUpdatedSerializedData();
            
            const Tensor2<SReal,Int> & S_P_ser    = S.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & S_P_v_ser  = S.PrimitiveVelocitiesSerializedData();
            const Tensor2<SReal,Int> & T_P_ser    = T.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & T_P_v_ser  = T.PrimitiveVelocitiesSerializedData();

            const SparseBinaryMatrixCSR<Int> & A = S.PrimitiveAdjacencyMatrix();
            
            PrimitiveCollisionFinder_T C ( T.MovingPrimitivePrototype(), T.MovingPrimitivePrototype() );
            
            SReal a = static_cast<SReal>(0);
            SReal b = static_cast<SReal>(1);
            
            SReal t_max = t_init_;
            
            while( (null <= stack_ptr) && (stack_ptr < max_depth - 4) )
            {
                const Int i = i_stack[stack_ptr];
                const Int j = j_stack[stack_ptr];
                stack_ptr--;
                
                AABB_CollisionTimeInterval<AMB_DIM, SReal, Int> (
                    S_C_ser.data(i), S_C_up_ser.data(i),
                    T_C_ser.data(j), T_C_up_ser.data(j),
                    a, b
                );
                
                if( a * t_init_ < t_max )
                {
                    const Int left_i = S_left[i];
                    const Int left_j = T_left[j];
                    
                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= null || left_j >= null )
                    {
                        
                        const Int right_i = S_right[i];
                        const Int right_j = T_right[j];
                        
                        // TODO: Improve score.
                        
                        const SReal score_i = static_cast<SReal>(left_i >= null) * S_C_ser(i,0);
                        const SReal score_j = static_cast<SReal>(left_j >= null) * T_C_ser(j,0);
                        
                        if( (score_i == score_j) && (score_i > static_cast<SReal>(0)) /*&& score_j > static_cast<SReal>(0)*/ )
                        {
                            // tie breaker: split both clusters
                            
                            if( i == j )
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
                                // In case of is_symmetric !=0, this is a very seldom case; still requird to preserve symmetry.
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
                        const Int ii_begin = S_begin[i];
                        const Int ii_end   = S_end  [i];
                        
                        const Int jj_end   = T_end[j];
                        
                        for( Int ii = ii_begin; ii < ii_end; ++ii )
                        {
                            const Int jj_begin = ( i == j ) ? ii_begin+1 : T_begin[j];
                            
                            for( Int jj = jj_begin; jj < jj_end; ++jj )
                            {
                                if( A.FindNonzeroPosition(ii,jj) < null )
                                {
                                    const SReal t = C.FindMaximumSafeStepSize(
                                        S_P_ser.data(ii), S_P_v_ser.data(ii),
                                        T_P_ser.data(jj), T_P_v_ser.data(jj),
                                        t_max, false
                                    );
                                    
                                    t_max = std::min(t, t_max);
                                }
                                
                            } // for( Int jj = jj_begin; jj < jj_end; ++jj )
                            
                        } // for( Int ii = ii_begin; ii < ii_end; ++ii )
                    
                    } // if( left_i >= null || left_j >= null )
                }
            
            } // while
            
            
            return t_max;
            
        } // MaximumSafeStepSize_DFS_Symmetric
        
        
        SReal MaximumSafeStepSize_DFS_Asymmetric(
                const Int i_0, const Int j_0, const SReal t_init_ ) const
        {
            Int i_stack[max_depth] = {};
            Int j_stack[max_depth] = {};
            
            Int stack_ptr = null;
            i_stack[0] = i_0;
            j_stack[0] = j_0;
            
            const Int * restrict const S_left     = S.ClusterLeft().data();
            const Int * restrict const S_right    = S.ClusterRight().data();
            const Int * restrict const T_left     = T.ClusterLeft().data();
            const Int * restrict const T_right    = T.ClusterRight().data();
            
            const Int * restrict const S_begin    = S.ClusterBegin().data();
            const Int * restrict const S_end      = S.ClusterEnd().data();
            const Int * restrict const T_begin    = T.ClusterBegin().data();
            const Int * restrict const T_end      = T.ClusterEnd().data();
            
            const Tensor2<SReal,Int> & S_C_ser    = S.ClusterSerializedData();
            const Tensor2<SReal,Int> & S_C_up_ser = S.ClusterUpdatedSerializedData();
            const Tensor2<SReal,Int> & T_C_ser    = T.ClusterSerializedData();
            const Tensor2<SReal,Int> & T_C_up_ser = T.ClusterUpdatedSerializedData();
            
            const Tensor2<SReal,Int> & S_P_ser    = S.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & S_P_v_ser  = S.PrimitiveVelocitiesSerializedData();
            const Tensor2<SReal,Int> & T_P_ser    = T.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & T_P_v_ser  = T.PrimitiveVelocitiesSerializedData();

            PrimitiveCollisionFinder_T C ( T.MovingPrimitivePrototype(), T.MovingPrimitivePrototype() );
            
            SReal a = static_cast<SReal>(0);
            SReal b = static_cast<SReal>(1);
            
            SReal t_max = t_init_;
            
            while( (null <= stack_ptr) && (stack_ptr < max_depth - 4) )
            {
                const Int i = i_stack[stack_ptr];
                const Int j = j_stack[stack_ptr];
                stack_ptr--;
                
                AABB_CollisionTimeInterval<AMB_DIM, SReal, Int> (
                    S_C_ser.data(i), S_C_up_ser.data(i),
                    T_C_ser.data(j), T_C_up_ser.data(j),
                    a, b
                );
                
                if( a * t_init_ < t_max )
                {
                    const Int left_i = S_left[i];
                    const Int left_j = T_left[j];
                    
                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= null || left_j >= null )
                    {
                        
                        const Int right_i = S_right[i];
                        const Int right_j = T_right[j];
                        
                        // TODO: Improve score.
                        
                        const SReal score_i = static_cast<SReal>(left_i >= null) * S_C_ser(i,0);
                        const SReal score_j = static_cast<SReal>(left_j >= null) * T_C_ser(j,0);
                        
                        if( (score_i == score_j) /* && (score_i > static_cast<SReal>(0)) && score_j > static_cast<SReal>(0)*/ )
                        {
                            // tie breaker: split both clusters
                            
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
                        const Int ii_begin = S_begin[i];
                        const Int ii_end   = S_end  [i];
                        
                        const Int jj_begin = T_begin[j];
                        const Int jj_end   = T_end[j];
                        
                        for( Int ii = ii_begin; ii < ii_end; ++ii )
                        {
                            for( Int jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const SReal t = C.FindMaximumSafeStepSize(
                                    S_P_ser.data(ii), S_P_v_ser.data(ii),
                                    T_P_ser.data(jj), T_P_v_ser.data(jj),
                                    t_max, false
                                );
                                
                                t_max = std::min(t, t_max);
                                
                            } // for( Int jj = jj_begin; jj < jj_end; ++jj )
                            
                        } // for( Int ii = ii_begin; ii < ii_end; ++ii )
                    
                    } // if( left_i >= null || left_j >= null )
                }
            
            } // while
            
            
            return t_max;
            
        } // MaximumSafeStepSize_DFS_Asymmetric
        
        
        
//####################################################################################################
//      MaximumSafeStepSize_Parallel
//####################################################################################################
                       
    protected:
        
        SReal MaximumSafeStepSize_Parallel() const
        {
            ptic(ClassName()+"::MaximumSafeStepSize_Parallel");
            
            SReal t_max = t_init;
            
            const Int max_leaves = ( static_cast<Int>(4) * ThreadCount() * ThreadCount() );
            
            i_queue = std::deque<Int>();
            j_queue = std::deque<Int>();
            
            if( is_symmetric )
            {
                t_max = MaximumSafeStepSize_BFS_Symmetric ( max_leaves, t_max );
                
//                DUMP(t_max);
//                DUMP(ThreadCount());
                
                #pragma omp parallel for num_threads( thread_count ) schedule(dynamic)
                for( size_t k = 0; k < i_queue.size(); ++k )
                {
                    const SReal t = MaximumSafeStepSize_DFS_Symmetric( i_queue[k], j_queue[k], t_max );
                    
                    #pragma omp critical
                    {
                        t_max = std::min( t, t_max );
//                        DUMP(t_max);
                    }
                }
            }
            else
            {
                t_max = MaximumSafeStepSize_BFS_Asymmetric( max_leaves, t_max );
                
//                DUMP(t_max);
//                DUMP(ThreadCount());
                
                #pragma omp parallel for num_threads( thread_count ) schedule(dynamic)
                for( size_t k = 0; k < i_queue.size(); ++k )
                {
                    const SReal t = MaximumSafeStepSize_DFS_Asymmetric( i_queue[k], j_queue[k], t_max );
                    
                    #pragma omp critical
                    {
                        t_max = std::min( t, t_max );
//                        DUMP(t_max);
                    }
                }
            }

            i_queue = std::deque<Int>();
            j_queue = std::deque<Int>();
            
            ptoc(ClassName()+"::MaximumSafeStepSize_Parallel");
            
            DUMP(t_max);
            
            return t_max;
        }
        
        
        SReal MaximumSafeStepSize_BFS_Symmetric( const Int max_leaves, const SReal t_init_ ) const
        {
            i_queue.push_back(null);
            j_queue.push_back(null);

            const Int * restrict const S_left     = S.ClusterLeft().data();
            const Int * restrict const S_right    = S.ClusterRight().data();
            const Int * restrict const T_left     = T.ClusterLeft().data();
            const Int * restrict const T_right    = T.ClusterRight().data();
            
            const Int * restrict const S_begin    = S.ClusterBegin().data();
            const Int * restrict const S_end      = S.ClusterEnd().data();
            const Int * restrict const T_begin    = T.ClusterBegin().data();
            const Int * restrict const T_end      = T.ClusterEnd().data();
            
            const Tensor2<SReal,Int> & S_C_ser    = S.ClusterSerializedData();
            const Tensor2<SReal,Int> & S_C_up_ser = S.ClusterUpdatedSerializedData();
            const Tensor2<SReal,Int> & T_C_ser    = T.ClusterSerializedData();
            const Tensor2<SReal,Int> & T_C_up_ser = T.ClusterUpdatedSerializedData();
            
            const Tensor2<SReal,Int> & S_P_ser    = S.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & S_P_v_ser  = S.PrimitiveVelocitiesSerializedData();
            const Tensor2<SReal,Int> & T_P_ser    = T.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & T_P_v_ser  = T.PrimitiveVelocitiesSerializedData();

            const SparseBinaryMatrixCSR<Int> & A = S.PrimitiveAdjacencyMatrix();
            
            PrimitiveCollisionFinder_T C ( T.MovingPrimitivePrototype(), T.MovingPrimitivePrototype() );
            
            SReal a = static_cast<SReal>(0);
            SReal b = static_cast<SReal>(1);
            
            SReal t_max = t_init_;
            
            while( !i_queue.empty() && ( static_cast<Int>(i_queue.size()) < max_leaves ) )
            {
                const Int i = i_queue.front();
                const Int j = j_queue.front();
                
                i_queue.pop_front();
                j_queue.pop_front();

                AABB_CollisionTimeInterval<AMB_DIM, SReal, Int> (
                    S_C_ser.data(i), S_C_up_ser.data(i),
                    T_C_ser.data(j), T_C_up_ser.data(j),
                    a, b
                );
                
                if( a * t_init_ < t_max )
                {
                    const Int left_i = S_left[i];
                    const Int left_j = T_left[j];

                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= null /*|| left_j >= null*/ )
                    {

                        const Int right_i = S_right[i];
                        const Int right_j = T_right[j];
                                                
                        const SReal score_i = (left_i>=null) * S_C_ser(i,0);
                        const SReal score_j = (left_j>=null) * T_C_ser(j,0);
                        
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
                            if (score_i < score_j)
                            {
                                i_queue.push_back(i);
                                j_queue.push_back(right_j);
                                
                                i_queue.push_back(i);
                                j_queue.push_back(left_j);
                            }
                            else
                            {
                                // This is a very seldom case; still requird to preserve symmetry.
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
                    }
                    else
                    {
                        const Int ii_begin = S_begin[i];
                        const Int ii_end   = S_end  [i];
                        
                        const Int jj_end   = T_end  [j];
                        
                        for( Int ii = ii_begin; ii < ii_end; ++ii )
                        {
                            const Int jj_begin = ( i == j ) ? ii_begin+1 : T_begin[j];
                            
                            for( Int jj = jj_begin; jj < jj_end; ++jj )
                            {
                                if( A.FindNonzeroPosition(ii,jj) < null )
                                {
                                    const SReal t = C.FindMaximumSafeStepSize(
                                        S_P_ser.data(ii), S_P_v_ser.data(ii),
                                        T_P_ser.data(jj), T_P_v_ser.data(jj),
                                        t_max, false
                                    );
                                    
                                    t_max = std::min( t, t_max );
                                }
                                
                            } // for( Int jj = jj_begin; jj < jj_end; ++jj )
                            
                        } // for( Int ii = ii_begin; ii < ii_end; ++ii )
                    
                    } // if( score_i == score_j )
                    
                }
                
            }
            
            return t_max;
            
        } // MaximumSafeStepSize_BFS_Symmetric
        
        
        
        SReal MaximumSafeStepSize_BFS_Asymmetric( const Int max_leaves, const SReal t_init_ ) const
        {
            i_queue.push_back(null);
            j_queue.push_back(null);

            const Int * restrict const S_left     = S.ClusterLeft().data();
            const Int * restrict const S_right    = S.ClusterRight().data();
            const Int * restrict const T_left     = T.ClusterLeft().data();
            const Int * restrict const T_right    = T.ClusterRight().data();
            
            const Int * restrict const S_begin    = S.ClusterBegin().data();
            const Int * restrict const S_end      = S.ClusterEnd().data();
            const Int * restrict const T_begin    = T.ClusterBegin().data();
            const Int * restrict const T_end      = T.ClusterEnd().data();
            
            const Tensor2<SReal,Int> & S_C_ser    = S.ClusterSerializedData();
            const Tensor2<SReal,Int> & S_C_up_ser = S.ClusterUpdatedSerializedData();
            const Tensor2<SReal,Int> & T_C_ser    = T.ClusterSerializedData();
            const Tensor2<SReal,Int> & T_C_up_ser = T.ClusterUpdatedSerializedData();
            
            const Tensor2<SReal,Int> & S_P_ser    = S.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & S_P_v_ser  = S.PrimitiveVelocitiesSerializedData();
            const Tensor2<SReal,Int> & T_P_ser    = T.PrimitiveSerializedData();
            const Tensor2<SReal,Int> & T_P_v_ser  = T.PrimitiveVelocitiesSerializedData();

            PrimitiveCollisionFinder_T C (T.MovingPrimitivePrototype(), T.MovingPrimitivePrototype());
            
            SReal a = static_cast<SReal>(0);
            SReal b = static_cast<SReal>(1);
            
            SReal t_max = t_init_;
            
            while( !i_queue.empty() && ( static_cast<Int>(i_queue.size()) < max_leaves ) )
            {
                const Int i = i_queue.front();
                const Int j = j_queue.front();
                i_queue.pop_front();
                j_queue.pop_front();

                AABB_CollisionTimeInterval<AMB_DIM, SReal, Int> (
                    S_C_ser.data(i), S_C_up_ser.data(i),
                    T_C_ser.data(j), T_C_up_ser.data(j),
                    a, b
                );
                
                if( a * t_init_ < t_max )
                {
                    const Int left_i = S_left[i];
                    const Int left_j = T_left[j];

                    // Warning: This assumes that either both children are defined or empty.
                    if( left_i >= null /*|| left_j >= null*/ )
                    {

                        const Int right_i = S_right[i];
                        const Int right_j = T_right[j];
                                                
                        const SReal score_i = (left_i>=null) * S_C_ser(i,0);
                        const SReal score_j = (left_j>=null) * T_C_ser(j,0);
                        
                        if( score_i == score_j )
                        {
                            // tie breaker: split both clusters
                            
                            i_queue.push_back(right_i);
                            j_queue.push_back(right_j);
                            
                            i_queue.push_back(left_i);
                            j_queue.push_back(right_j);
                            
                            i_queue.push_back(right_i);
                            j_queue.push_back(left_j);
                            
                            i_queue.push_back(left_i);
                            j_queue.push_back(left_j);
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
                        const Int ii_begin = S_begin[i];
                        const Int ii_end   = S_end  [i];
                        
                        const Int jj_begin = T_begin[j];
                        const Int jj_end   = T_end  [j];
                        
                        for( Int ii = ii_begin; ii < ii_end; ++ii )
                        {
                            for( Int jj = jj_begin; jj < jj_end; ++jj )
                            {
                                const SReal t = C.FindMaximumSafeStepSize(
                                    S_P_ser.data(ii), S_P_v_ser.data(ii),
                                    T_P_ser.data(jj), T_P_v_ser.data(jj),
                                    t_max, false
                                );
                                
                                t_max = std::min( t, t_max );
                                
                            } // for( Int jj = jj_begin; jj < jj_end; ++jj )
                            
                        } // for( Int ii = ii_begin; ii < ii_end; ++ii )
                    
                    } // if( score_i == score_j )
                    
                }
                
            }
            
            return t_max;
            
        } // MaximumSafeStepSize_BFS_Asymmetric
        
    public:
        
        virtual Int ThreadCount() const override
        {
            return thread_count;
        }
        
        Int AmbDim() const override
        {
            return AMB_DIM;
        }
        
        bool IsSymmetric() const override
        {
            return is_symmetric;
        }
        
        const CollisionMatrix_T & ClusterCollisionMatrix() const override
        {
            if( !C_collision_matrix_initialized )
            {
                FindCollidingClusters();
                
                C_collision_matrix_initialized = true;
            }
            
            return C_collision_matrix;
        }
        
        const CollisionMatrix_T & PrimitiveCollisionMatrix() const override
        {
            if( !P_collision_matrix_initialized )
            {
                FindCollidingPrimitives();
                
                P_collision_matrix_initialized = true;
            }
            
            return P_collision_matrix;
        }
        
        const ClusterTree_T & GetS() const override
        {
            return S;
        }
        
        const ClusterTree_T & GetT() const override
        {
            return T;
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS) + "<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }

    }; // CLASS
    
} // namespace Repulsion

#undef BASE
#undef CLASS
