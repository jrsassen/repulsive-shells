#pragma once

#define BASE  SimplicialMeshBase<Real,Int,SReal,ExtReal>
#define CLASS SimplicialMesh

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class TangentPointEnergy_FMM_Adaptive;
    
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class TangentPointObstacleEnergy_FMM_Adaptive;
    
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class TangentPointMetric_FMM_Adaptive;
    
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class TrivialEnergy_FMM_Adaptive;
    
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class TrivialObstacleEnergy_FMM_Adaptive;
    
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class Energy_Restricted;
    
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public :
        
        using       Primitive_T  =        Polytope<DOM_DIM+1,AMB_DIM,GJK_Real,Int,SReal,ExtReal,Int>;
        using MovingPrimitive_T  =  MovingPolytope<DOM_DIM+1,AMB_DIM,GJK_Real,Int,SReal,ExtReal,Int>;
        using BoundingVolume_T   =           AABB<           AMB_DIM,GJK_Real,Int,SReal>;
        
        
        using ClusterTree_T      =       ClusterTree<        AMB_DIM,Real,Int,SReal,ExtReal>;
        using BlockClusterTree_T =  BlockClusterTree<        AMB_DIM,Real,Int,SReal,ExtReal>;
        using CollisionTree_T    =     CollisionTree<        AMB_DIM,Real,Int,SReal,ExtReal>;
        using Energy_T           = Energy_Restricted<DOM_DIM,AMB_DIM,Real,Int,SReal,ExtReal>;
        
        using Obstacle_T         = BASE;
        
        CLASS() {}

        CLASS(
            const Tensor2<ExtReal,Int> & V_coords_,
            // vertex coordinates; assumed to be of size vertex_count_ x AMB_DIM
            const Tensor2<Int,Int> & simplices_,
            // simplices; assumed to be of size simplex_count_ x (DOM_DIM+1)
            const long long thread_count_ = 1
        )
        :   CLASS(
                  V_coords_.data(),
                  V_coords_.Dimension(0),
                  false,
                  simplices_.data(),
                  simplices_.Dimension(0),
                  false,
                  static_cast<Int>(thread_count_)
            )
        {
            ptic(ClassName()+"()");
            if( V_coords_.Dimension(1) != AMB_DIM )
            {
                eprint(ClassName()+" : V_coords.Dimension(1) != AMB_DIM");
                ptoc(ClassName()+"()");
                return;
            }
            if( simplices_.Dimension(1) != DOM_DIM+1 )
            {
                eprint(ClassName()+" : simplices_.Dimension(1) != DOM_DIM+1");
                ptoc(ClassName()+"()");
                return;
            }
            ptoc(ClassName()+"()");
        }
        
#ifdef LTEMPLATE_H
        
        CLASS(
            const mma::TensorRef<ExtReal> & V_coords_,   // vertex coordinates; assumed to be of size vertex_count_ x AMB_DIM
            const mma::TensorRef<long long> & simplices_,   // simplices; assumed to be of size simplex_count_ x (DOM_DIM+1)
            const long long thread_count_ = 1
        )
        :   CLASS(
                V_coords_.data(),
                static_cast<Int>(V_coords_.dimensions()[0]),
                simplices_.data(),
                static_cast<Int>(simplices_.dimensions()[0]),
                static_cast<Int>(thread_count_)
            )
        {
            ptic(ClassName()+"() (from MTensor)");
            if( V_coords_.dimensions()[1] != AMB_DIM )
            {
                eprint(ClassName()+" : V_coords_.dimensions()[1] != AMB_DIM");
                ptoc(ClassName()+"() (from MTensor)");
                return;
            }
            if( simplices_.dimensions()[1] != DOM_DIM+1 )
            {
                eprint(ClassName()+" : simplices_.dimensions()[1] != DOM_DIM+1");
                ptoc(ClassName()+"() (from MTensor)");
                return;
            }
            ptoc(ClassName()+"() (from MTensor)");
        }
        
#endif

        template<typename SomeInt>
        CLASS(
            const ExtReal * V_coords_, // vertex coordinates; assumed to be of size vertex_count_ x AMB_DIM
            const long long vertex_count_,
            const SomeInt * simplices_, // simplices; assumed to be of size simplex_count_ x (DOM_DIM+1)
            const long long simplex_count_,
            const long long thread_count_ = 1
        )
        :   CLASS( V_coords_, vertex_count_, false, simplices_, simplex_count_, false, thread_count_)
        {}
        
        template<typename SomeInt>
        CLASS(
            const ExtReal * V_coords_, // vertex coordinates; assumed to be of size vertex_count_ x AMB_DIM
            const long long vertex_count_,
            const bool V_transpose,
            const SomeInt * simplices_, // simplices; assumed to be of size simplex_count_ x (DOM_DIM+1)
            const long long simplex_count_,
            const bool simplex_transpose,
            const long long thread_count_ = 1
        )
        :   BASE      ( static_cast<Int>(thread_count_) )
        ,   V_coords  ( ToTensor2<Real,Int>(
                            V_coords_,
                            static_cast<Int>(vertex_count_),
                            static_cast<Int>(AMB_DIM),
                            V_transpose
                        )
                      )
        ,   simplices ( ToTensor2<Int,Int>(
                            simplices_,
                            static_cast<Int>(simplex_count_),
                            static_cast<Int>(DOM_DIM+1),
                            simplex_transpose
                        )
                      )
        ,   details   ( static_cast<Int>(thread_count_) )
        {
            ptic(ClassName()+" (pointer)");
            ptoc(ClassName()+" (pointer)");
        }

        
        virtual ~CLASS() override = default;
        
    public:
        
        using BASE::block_cluster_tree_settings;
        using BASE::cluster_tree_settings;
        using BASE::adaptivity_settings;
        using BASE::ThreadCount;
//        using BASE::GetClusterTree;
//        using BASE::GetBlockClusterTree;
//        using BASE::GetCollisionTree;
//        using BASE::GetObstacleClusterTree;
//        using BASE::GetObstacleBlockClusterTree;
        
    protected:
        
        mutable bool cluster_tree_initialized = false;
        mutable std::unique_ptr<ClusterTree_T> cluster_tree
              = std::unique_ptr<ClusterTree_T>( new ClusterTree_T() );
        
        mutable bool block_cluster_tree_initialized = false;
        mutable std::unique_ptr<BlockClusterTree_T> block_cluster_tree
              = std::unique_ptr<BlockClusterTree_T>(
                    new BlockClusterTree_T(*cluster_tree, *cluster_tree) );
        
        mutable bool collision_tree_initialized = false;
        mutable std::unique_ptr<CollisionTree_T> collision_tree
              = std::unique_ptr<CollisionTree_T>(
                    new CollisionTree_T(*cluster_tree, *cluster_tree, static_cast<SReal>(1) ) );
        
        mutable SReal max_update_step_size = 0;
        
        Tensor2<Real,Int> V_coords;
        Tensor2<Real,Int> V_updates;
        Tensor2<Int,Int>  simplices;
        
        mutable bool derivative_assembler_initialized = false;
        mutable SparseBinaryMatrixCSR<Int> derivative_assembler;
        
        mutable Primitive_T P_proto;
        
        mutable Tensor1<Int,Int> simplex_row_pointers;
        mutable Tensor1<Int,Int> simplex_column_indices;
        
        SimplicialMeshDetails<DOM_DIM,AMB_DIM,Real,Int> details;

        
    public:
        
        virtual Int DomDim() const override
        {
            return DOM_DIM;
        }
        
        virtual Int AmbDim() const override
        {
            return AMB_DIM;
        }

        virtual const Tensor2<Real,Int> & VertexCoordinates() const override
        {
            return V_coords;
        }
        
        virtual const Tensor2<Int,Int> & Simplices() const override
        {
            return simplices;
        }

        virtual Int FarDim() const override
        {
            return 1 + AMB_DIM + (AMB_DIM * (AMB_DIM + 1))/2;
        }
        
        virtual Int NearDim() const override
        {
            return 1 + (DOM_DIM+1)*AMB_DIM + (AMB_DIM * (AMB_DIM + 1))/2;
        }
        
        virtual Int VertexCount() const override
        {
            return V_coords.Dimension(0);
        }
        
        virtual Int SimplexCount() const override
        {
            return simplices.Dimension(0);
        }
    
        virtual Int DofCount() const override
        {
            return VertexCount() * AMB_DIM;
        }
        
        virtual const Real * Dofs() const override
        {
            return V_coords.data();
        }
        
        virtual void SemiStaticUpdate( const ExtReal * restrict const new_V_coords_ ) override
        {
            ptic(ClassName()+"::SemiStaticUpdate");
            
            V_coords.Read(new_V_coords_);
            
            Tensor2<Real,Int> P_near( SimplexCount(), NearDim() );
            Tensor2<Real,Int> P_far ( SimplexCount(), FarDim()  );

            details.ComputeNearFarData( V_coords, simplices, P_near, P_far );
            
            cluster_tree->SemiStaticUpdate( P_near, P_far );
            
            ptoc(ClassName()+"::SemiStaticUpdate");
        }
       
        
        virtual void LoadUpdateVectors( const ExtReal * const vecs, const ExtReal max_time ) override
        {
            ptic(ClassName()+"::LoadUpdateVectors");
            
            max_update_step_size = static_cast<SReal>(max_time);
            collision_tree_initialized = false;
            
            V_updates = Tensor2<Real,Int>( VertexCount(), AMB_DIM );
            V_updates.Read(vecs);
            
            // ATTENTION: We reorder already here outside of the cluster tree to save a copy operation of a big Tensor2!
            
            const Int thread_count = GetClusterTree().ThreadCount();
            
            MovingPrimitive_T P_moving;
            
            Tensor2<SReal,Int> P_velocities_serialized ( SimplexCount(), P_moving.VelocitySize(), 0 );
            SReal * restrict const P_v_ser = P_velocities_serialized.data();
            
            const Tensor1<Int,Int> P_ordering = GetClusterTree().PrimitiveOrdering();

            Tensor1<Int,Int> job_ptr = BalanceWorkLoad( SimplexCount(), thread_count );
            
            #pragma omp parallel for num_threads(thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                MovingPrimitive_T P_mov;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const Int j = P_ordering[i];
                    P_mov.FromVelocitiesIndexList( vecs, simplices.data(), j );
                    P_mov.WriteVelocitiesSerialized( P_v_ser, i );
                }
            }
            
            // P_moving and P_velocities_serialized will be swapped against potentiall empty containers.
            GetClusterTree().TakeUpdateVectors(
                P_moving, P_velocities_serialized, static_cast<SReal>(max_time)
            );
            
            ptoc(ClassName()+"::LoadUpdateVectors");
        }
        
        virtual ExtReal MaximumSafeStepSize( const ExtReal * restrict const vecs, const ExtReal max_time ) override
        {
            ptic(ClassName()+"::LoadUpdateVectors");
            
            LoadUpdateVectors( vecs, max_time );
            
            ExtReal t = static_cast<ExtReal>( GetCollisionTree().MaximumSafeStepSize() );
            
            ptoc(ClassName()+"::LoadUpdateVectors");
            
            return t;
        }
        
        virtual const ClusterTree_T & GetClusterTree() const override
        {
            if( !cluster_tree_initialized )
            {
                ptic(ClassName()+"::GetClusterTree");
                if( (V_coords.Dimension(0) > 0) && (simplices.Dimension(0) > 0) )
                {
                    

                    ptic("Allocations");
                    auto P_coords      = Tensor2<Real,Int> ( SimplexCount(), AMB_DIM, static_cast<Real>(0) );
                    auto P_hull_coords = Tensor3<Real,Int> ( SimplexCount(), DOM_DIM+1, AMB_DIM );
                    auto P_near        = Tensor2<Real,Int> ( SimplexCount(), NearDim() );
                    auto P_far         = Tensor2<Real,Int> ( SimplexCount(), FarDim() );
                    auto P_ordering    = iota<Int>         ( SimplexCount() );
                    
                    Tensor2<SReal,Int> P_serialized ( SimplexCount(), P_proto.Size() );
                    
                    auto DiffOp = SparseMatrixCSR<Real,Int>(
                        SimplexCount() * AMB_DIM,
                        VertexCount(),
                        SimplexCount() * AMB_DIM * (DOM_DIM+1),
                        ThreadCount()
                    );
        
                    DiffOp.Outer()[SimplexCount() * AMB_DIM] = SimplexCount() * AMB_DIM * (DOM_DIM+1);
                    
                    auto AvOp = SparseMatrixCSR<Real,Int>(
                        SimplexCount(),
                        VertexCount(),
                        SimplexCount() * (DOM_DIM+1),
                        ThreadCount()
                    );
                    
                    AvOp.Outer()[SimplexCount()] = SimplexCount() * (DOM_DIM+1);
                                
                    ptoc("Allocations");
                    
                    // What remains is to compute P_coords, P_hull_coords, P_near and P_far and the nonzero values of DiffOp.
                    details.ComputeNearFarDataOps( V_coords, simplices, P_coords, P_hull_coords, P_near, P_far, DiffOp, AvOp );


                    auto  job_ptr = BalanceWorkLoad<Int>( SimplexCount(), ThreadCount() );
                    
                    ptic("Creating primitives");
                    const Int thread_count = job_ptr.Size()-1;
                    
                    #pragma omp parallel for num_threads(thread_count)
                    for( Int thread = 0; thread < thread_count; ++thread )
                    {

//                        std::unique_ptr<Primitive_T> P = P_proto.Clone();

                        Primitive_T P;
                        
                        Int i_begin = job_ptr[thread];
                        Int i_end   = job_ptr[thread+1];

                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            P.SetPointer( P_serialized.data(), i );
                            P.FromIndexList( V_coords.data(), simplices.data(), i );
                        }
                    }
                    ptoc("Creating primitives");
                    
                    ptic("Initializing cluster prototypes");
                    
                    std::unique_ptr<BoundingVolume_T> C_proto;

                    switch( cluster_tree_settings.bounding_volume_type )
                    {
                        case BoundingVolumeType::AABB_MedianSplit:
                        {
                            logprint("Using AABB_MedianSplit as bounding volume.");
                            C_proto = std::unique_ptr<BoundingVolume_T>(
                                static_cast<BoundingVolume_T*>(new AABB_MedianSplit<AMB_DIM,GJK_Real,Int,SReal>)
                            );
                            break;
                        }
                        case BoundingVolumeType::AABB_PreorderedSplit:
                        {
                            logprint("Using AABB_PreorderedSplit as bounding volume.");
                            C_proto = std::unique_ptr<BoundingVolume_T>(
                                static_cast<BoundingVolume_T*>(new AABB_PreorderedSplit<AMB_DIM,GJK_Real,Int,SReal>)
                            );
                            break;
                        }
                        default:
                        {
                            logprint("Using AABB_LongestAxisSplit as bounding volume.");
                            C_proto = std::unique_ptr<BoundingVolume_T>(
                                static_cast<BoundingVolume_T*>(new AABB_LongestAxisSplit<AMB_DIM,GJK_Real,Int,SReal>)
                            );
                        }
                    }

                    ptoc("Initializing cluster prototypes");
                    
                    cluster_tree = std::make_unique<ClusterTree_T>(
                        P_proto, P_serialized, *C_proto, P_ordering,
                        P_near, P_far,
                        DiffOp, AvOp,
                        cluster_tree_settings
                    );
                    
                    cluster_tree_initialized = true;
                }
                else
                {
                    cluster_tree = std::make_unique<ClusterTree_T>();
                }
                
                ptoc(ClassName()+"::GetClusterTree");
            }
            return *cluster_tree;
        }

        virtual const BlockClusterTree_T & GetBlockClusterTree() const override
        {
            if( !block_cluster_tree_initialized )
            {
                ptic(ClassName()+"::GetBlockClusterTree");
                
                (void)GetClusterTree();
                
                if( cluster_tree_initialized )
                {
                    block_cluster_tree_settings.near_field_separation_parameter = adaptivity_settings.theta;
                    block_cluster_tree_settings.near_field_collision_parameter  = adaptivity_settings.intersection_theta;
                    
                    block_cluster_tree = std::make_unique<BlockClusterTree_T>(
                        *cluster_tree, *cluster_tree,
                        block_cluster_tree_settings
                    );
                    
                    block_cluster_tree_initialized = true;
                }
//                else
//                {
//                    block_cluster_tree = std::make_unique<BlockClusterTree_T>();
//                }
                ptoc(ClassName()+"::GetBlockClusterTree");
            }
            return *block_cluster_tree;
        }
        
        virtual const CollisionTree_T & GetCollisionTree() const override
        {
            if( !collision_tree_initialized )
            {
                ptic(ClassName()+"::GetCollisionTree");
                
                (void)GetClusterTree();
        
                if( cluster_tree_initialized )
                {
                    collision_tree = std::make_unique<CollisionTree_T>(
                        *cluster_tree, *cluster_tree, max_update_step_size );

                    collision_tree_initialized = true;
                }
//                else
//                {
//                    collision_tree = std::make_unique<CollisionTree_T>();
//                }

                ptoc(ClassName()+"::GetCollisionTree");
            }
            return *collision_tree;
        }
        
        
        const SparseBinaryMatrixCSR<Int> & DerivativeAssembler() const override
        {
            if( !derivative_assembler_initialized )
            {
                ptic(ClassName()+"::DerivativeAssembler");
                
                auto A = SparseBinaryMatrixCSR<Int>(
                    SimplexCount() * (DOM_DIM+1),
                    VertexCount(),
                    SimplexCount() * (DOM_DIM+1),
                    ThreadCount()
                );
                
                A.Outer().iota();
                A.Inner().Read(simplices.data());
                
                derivative_assembler = A.TransposeBinary();
                
                derivative_assembler_initialized = true;
                
                ptoc(ClassName()+"::DerivativeAssembler");
            }
            return derivative_assembler;
            
        } // DerivativeAssembler
        
        void Assemble_ClusterTree_Derivatives( ExtReal * output, const ExtReal weight, bool addTo = false ) const override
        {
            ptic(ClassName()+"::Assemble_ClusterTree_Derivatives");
            
            Tensor3<Real,Int> buffer ( SimplexCount(), DOM_DIM+1, AMB_DIM );
            
            GetClusterTree().CollectDerivatives();
            
            details.DNearToHulls( V_coords, simplices, GetClusterTree().PrimitiveDNearFieldData(), buffer, false );
            
            DUMP(buffer.MaxNorm());

            details.DFarToHulls ( V_coords, simplices, GetClusterTree().PrimitiveDFarFieldData(), buffer, true );

            DUMP(buffer.MaxNorm());
            
            DerivativeAssembler().Multiply_BinaryMatrix_DenseMatrix(
                static_cast<Real>(weight), buffer.data(), static_cast<ExtReal>(addTo), output, AMB_DIM );
             
            ptoc(ClassName()+"::Assemble_ClusterTree_Derivatives");
        }

//####################################################################################################
//      Obstacle
//####################################################################################################
    
        
        mutable bool obstacle_initialized = false;
        mutable std::unique_ptr<Obstacle_T> obstacle = nullptr;
        
        mutable bool obstacle_block_cluster_tree_initialized = false;
        mutable std::unique_ptr<BlockClusterTree_T> obstacle_block_cluster_tree = nullptr;
//              = std::unique_ptr<BlockClusterTree_T>();
        
        bool ObstacleInitialized() const override
        {
            return obstacle != nullptr;
        }
        
        virtual void LoadObstacle( std::unique_ptr<Obstacle_T> obstacle_ ) override
        {
            // Input obstacle is moved.
            if( obstacle_->AmbDim() != AmbDim() )
            {
                eprint(ClassName()+"::LoadObstacle: Attempted to load obstacle of ambient dimension "+ToString(obstacle_->AmbDim())+" into mesh of ambient dimension "+ToString(AmbDim())+". Setting obstacle to nullptr."
                );
                obstacle = nullptr;
                return;
            }
            
            obstacle = std::move(obstacle_);
            
            obstacle_block_cluster_tree = nullptr;
            obstacle_block_cluster_tree_initialized = false;
            
            obstacle->cluster_tree_settings = cluster_tree_settings;
            obstacle_initialized = true;
            
            tpo_initialized = false;
            trivial_oe_initialized = false;
            
        }
        
        const Obstacle_T & GetObstacle() const override
        {
            if( !obstacle_initialized )
            {
                wprint(ClassName()+"::GetObstacle: Obstacle not initialized.");
                obstacle = std::unique_ptr<BASE>(new CLASS());
            }
            return *obstacle;
        }
        
        virtual const ClusterTree_T & GetObstacleClusterTree() const override
        {
            return *dynamic_cast<const ClusterTree_T *>( & (GetObstacle().GetClusterTree()) );
        }
        
        virtual const BlockClusterTree_T & GetObstacleBlockClusterTree() const override
        {
            if( !obstacle_block_cluster_tree_initialized )
            {
                ptic(ClassName()+"::GetObstacleBlockClusterTree");
                if( obstacle_initialized )
                {
                    (void)GetClusterTree();
                    obstacle_block_cluster_tree = std::make_unique<BlockClusterTree_T>(
                        *cluster_tree,
                        GetObstacleClusterTree(),
                        block_cluster_tree_settings
                    );
                    obstacle_block_cluster_tree_initialized = true;
                }
//                else
//                {
//                    // Print warning message.
//                    (void)GetObstacle();
//                    obstacle_block_cluster_tree = std::make_unique<BlockClusterTree_T>();
//                }
                ptoc(ClassName()+"::GetObstacleBlockClusterTree");
                
            }
            return *obstacle_block_cluster_tree;
        }
        
        
        
//####################################################################################################
//      Tangent-point
//####################################################################################################
     
    public:
        
        mutable Real    tp_alpha  = 2 * (DOM_DIM+1);
        mutable Real    tp_beta   = 4 * (DOM_DIM+1);
        mutable ExtReal tp_weight = 1;
        
        mutable bool tpe_initialized = false;
        mutable std::unique_ptr<TangentPointEnergy_FMM_Adaptive<DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>> tpe;

//        mutable bool tpo_initialized = false;
//        mutable std::unique_ptr<
//            TangentPointObstacleEnergy_FMM_Adaptive<DOM_DIM,DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>
//        > tpo;
        
        mutable bool tpo_initialized = false;
        mutable std::unique_ptr<Energy_T> tpo;

        mutable bool tpm_initialized = false;
        mutable std::unique_ptr<TangentPointMetric_FMM_Adaptive<DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>> tpm;

        
        void RequireTangentPointEnergy() const
        {
            if( !tpe_initialized )
            {
                tpe = std::make_unique<
                    TangentPointEnergy_FMM_Adaptive<DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>
                >( tp_alpha, tp_beta, tp_weight, adaptivity_settings );
                
                tpe_initialized = true;
            }
        }
        
        
        void RequireTangentPointObstacleEnergy() const
        {
            if( !tpo_initialized )
            {
                if( obstacle_initialized )
                {
                    Energy_T * r = nullptr;
                    
                    switch( GetObstacle().DomDim() )
                    {
                        case 1:
                        {
                            r = new TangentPointObstacleEnergy_FMM_Adaptive<DOM_DIM,1,AMB_DIM,0,Real,Int,SReal,ExtReal> ( tp_alpha, tp_beta, tp_weight, adaptivity_settings );
                            break;
                        }
                        case 2:
                        {
                            r = new TangentPointObstacleEnergy_FMM_Adaptive<DOM_DIM,2,AMB_DIM,0,Real,Int,SReal,ExtReal> ( tp_alpha, tp_beta, tp_weight, adaptivity_settings );
                            break;
                        }
                        default:
                        {
                            wprint(ClassName()+"RequireTangentPointObstacleEnergy : domain dimension "+ToString(GetObstacle().DomDim())+" invalid. Only obstacles of domain dimension 1 and 2 are implemented.");
                        }
        
                    }
                    
                    if( r != nullptr )
                    {
                        tpo = std::unique_ptr<Energy_T>(r);
                        
                        tpo_initialized = true;
                    }
                }
            }
        }
        
        void RequireTangentPointMetric() const
        {
            if( !tpm_initialized )
            {
                tpm = std::make_unique<
                        TangentPointMetric_FMM_Adaptive<DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>
                    >( GetBlockClusterTree(), tp_alpha, tp_beta, tp_weight, adaptivity_settings );
                    
                    
                tpm_initialized = true;
            }
        }

    public:
        
        ExtReal GetTangentPointWeight() const override
        {
           return tp_weight;
        }

        void SetTangentPointWeight( const ExtReal weight ) const override
        {
            tp_weight = weight;
            if( tpe_initialized )
            {
                tpe->SetWeight(tp_weight);
            }
            if( tpo_initialized )
            {
                tpo->SetWeight(tp_weight);
            }
            if( tpm_initialized )
            {
                tpm->SetWeight(tp_weight);
            }
        }
                                               
        std::pair<Real,Real> GetTangentPointExponents() const override
        {
           return std::pair<Real,Real>( tp_alpha, tp_beta );
        }

        void SetTangentPointExponents( const Real alpha_, const Real beta_ ) const override
        {
           if( alpha_ != tp_alpha || beta_ != tp_beta )
           {
               tp_alpha = alpha_;
               tp_beta  = beta_;
               
               tpe_initialized = false;
               tpo_initialized = false;
               tpm_initialized = false;
           }
        }
        

        virtual ExtReal TangentPointEnergy() const override
        {
            RequireTangentPointEnergy();
            
            return tpe->Value(*this);
        };

        virtual ExtReal TangentPointEnergy_Differential( ExtReal * output, bool addTo = false ) const override
        {
            RequireTangentPointEnergy();
            
            return tpe->Differential(*this, output, addTo);
        };



        virtual ExtReal TangentPointObstacleEnergy() const override
        {
            RequireTangentPointObstacleEnergy();
            
            return tpo->Value(*this);
        };

        virtual ExtReal TangentPointObstacleEnergy_Differential( ExtReal * output, bool addTo = false ) const override
        {
            RequireTangentPointObstacleEnergy();
            
            return tpo->Differential(*this, output, addTo);
        };


        virtual void TangentPointMetric_Multiply(
            const ExtReal alpha, const ExtReal * U,
            const ExtReal  beta,       ExtReal * V,
            Int cols,
            KernelType kernel
        ) const override
        {
            RequireTangentPointMetric();
            
            tpm->Multiply_DenseMatrix(
                static_cast<Real>(alpha), U, beta, V, cols, kernel );
        }
        
        virtual void TangentPointMetric_Multiply(
            const ExtReal alpha, const ExtReal * U,
            const ExtReal  beta,       ExtReal * V,
            Int cols
        ) const override
        {
            RequireTangentPointMetric();
            
            tpm->Multiply_DenseMatrix(
                static_cast<Real>(alpha), U, beta, V, cols, KernelType::HighOrder );
            
            tpm->Multiply_DenseMatrix(
                static_cast<Real>(alpha), U, beta, V, cols, KernelType::LowOrder );
        }
        
        
        virtual const Tensor1<Real,Int> & TangentPointMetric_Values(
            const bool farQ,
            const KernelType kernel
        ) const override
        {
            RequireTangentPointMetric();
            
            if( farQ )
            {
                return tpm->FarFieldValues().find(kernel)->second;
            }
            else
            {
                return tpm->NearFieldValues().find(kernel)->second;
            }
        }
        
        
        virtual void TangentPointMetric_ApplyKernel(
            const bool farQ,
            const KernelType kernel
        ) const override
        {
            RequireTangentPointMetric();
            
            if( farQ )
            {
                tpm->ApplyFarFieldKernel ( static_cast<Real>(1), kernel );
            }
            else
            {
                tpm->ApplyNearFieldKernel( static_cast<Real>(1), kernel );
            }
        }
        
//####################################################################################################
//      Custom energy
//####################################################################################################
    
    protected:
        
        mutable ExtReal custom_e_weight = 1;
        
        mutable std::unique_ptr<EnergyBase<Real,Int,SReal,ExtReal>> custom_e;
        
    
    public:
        
        void LoadCustomEnergy( std::unique_ptr<EnergyBase<Real,Int,SReal,ExtReal>> e ) const override
        {
            custom_e = std::move(e);
            if( custom_e != nullptr )
            {
                custom_e->SetWeight(custom_e_weight);
            }
        }
        
        virtual ExtReal GetCustomEnergyWeight() const override
        {
           return custom_e_weight;
        }

        virtual void SetCustomEnergyWeight( const ExtReal weight ) const override
        {
            custom_e_weight = weight;
            
            if( custom_e != nullptr )
            {
                custom_e->SetWeight(custom_e_weight);
            }
        }

        virtual ExtReal CustomEnergy() const override
        {
            if( custom_e != nullptr )
            {
                return custom_e->Value(*this);
            }
            else
            {
                return static_cast<ExtReal>(0);
            }
        };

        virtual ExtReal CustomEnergy_Differential( ExtReal * output, bool addTo = false ) const override
        {
            if( custom_e != nullptr )
            {
                return custom_e->Differential(*this, output, addTo);
            }
            else
            {
                if( !addTo )
                {
                    for( Int i = 0; i < DofCount(); ++i )
                    {
                        output[i] = static_cast<ExtReal>(0);
                    }
                }
                return static_cast<ExtReal>(0);
            }
        };
        
//####################################################################################################
//      Trivial energy
//####################################################################################################
        
    protected:
        
        mutable ExtReal trivial_e_weight = 1;
        
        mutable bool trivial_e_initialized = false;
        mutable std::unique_ptr<TrivialEnergy_FMM_Adaptive<DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>> trivial_e;
        
        mutable bool trivial_oe_initialized = false;
        mutable std::unique_ptr<Energy_T> trivial_oe;
        
        void RequireTrivialEnergy() const
        {
            if( !trivial_e_initialized )
            {
                trivial_e = std::make_unique< TrivialEnergy_FMM_Adaptive<DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>
                >(
                    trivial_e_weight,
                    adaptivity_settings
                );
                
                trivial_e_initialized = true;
            }
        }
        
//        void RequireTrivialObstacleEnergy() const
//        {
//            if( !trivial_oe_initialized )
//            {
//                trivial_oe = std::make_unique< TrivialObstacleEnergy_FMM_Adaptive<DOM_DIM,DOM_DIM,AMB_DIM,0,Real,Int,SReal,ExtReal>
//                >(
//                    trivial_e_weight,
//                    adaptivity_settings
//                );
//
//                trivial_oe_initialized = true;
//            }
//        }
        
        void RequireTrivialObstacleEnergy() const
        {
            if( !trivial_oe_initialized )
            {
                if( obstacle_initialized )
                {
                    Energy_T * r = nullptr;
                    
                    switch( GetObstacle().DomDim() )
                    {
                        case 1:
                        {
                            r = new TrivialObstacleEnergy_FMM_Adaptive<DOM_DIM,1,AMB_DIM,0,Real,Int,SReal,ExtReal>
                                (
                                    trivial_e_weight,
                                    adaptivity_settings
                                );
                            break;
                        }
                        case 2:
                        {
                            r = new TrivialObstacleEnergy_FMM_Adaptive<DOM_DIM,2,AMB_DIM,0,Real,Int,SReal,ExtReal>
                                (
                                    trivial_e_weight,
                                    adaptivity_settings
                                );
                            break;
                        }
                        default:
                        {
                            wprint(ClassName()+"RequireTrivialObstacleEnergy : domain dimension "+ToString(GetObstacle().DomDim())+" invalid. Only obstacles of domain dimension 1 and 2 are implemented.");
                        }
        
                    }
                    
                    if( r != nullptr )
                    {
                        trivial_oe = std::unique_ptr<Energy_T>(r);
                        
                        trivial_oe_initialized = true;
                    }
                }
            }
        }

    public:
        
        virtual ExtReal GetTrivialEnergyWeight() const override
        {
           return trivial_e_weight;
        }

        virtual void SetTrivialEnergyWeight( const ExtReal weight ) const override
        {
            trivial_e_weight = weight;
            if( trivial_e_initialized )
            {
                trivial_e->SetWeight(tp_weight);
            }
        }

        virtual ExtReal TrivialEnergy() const override
        {
            RequireTrivialEnergy();
            
            return trivial_e->Value(*this);
        };

        virtual ExtReal TrivialEnergy_Differential( ExtReal * output, bool addTo = false ) const override
        {
            RequireTrivialEnergy();
            
            return trivial_e->Differential(*this, output, addTo);
        };
        
        
        virtual ExtReal TrivialObstacleEnergy() const override
        {
            RequireTrivialObstacleEnergy();
            
            return trivial_oe->Value(*this);
        };

        virtual ExtReal TrivialObstacleEnergy_Differential( ExtReal * output, bool addTo = false ) const override
        {
            RequireTrivialObstacleEnergy();
            
            return trivial_oe->Differential(*this, output, addTo);
        };
        
        
    public:
        
        virtual CLASS & DownCast() override
        {
            return *this;
        }
        
        virtual const CLASS & DownCast() const override
        {
            return *this;
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
  
    };
    
    
    
    
    template<typename Real, typename Int, typename SReal, typename ExtReal, typename ExtInt>
    std::unique_ptr<BASE> MakeSimplicialMesh(
        const ExtReal * const V_coords_, // vertex coordinates; assumed to be of size vertex_count_ x AMB_DIM
        const long long vertex_count_,
        const int       amb_dim,
        const ExtInt  * const simplices_, // simplices; assumed to be of size simplex_count_ x (DOM_DIM+1)
        const long long simplex_count_,
        const int       simplex_size,
        const long long thread_count_ = 1
    )
    {
        ptic("MakeSimplicialMesh");
        BASE * r = nullptr;

        const int dom_dim = simplex_size-1;
        
        DUMP(dom_dim);
        DUMP(amb_dim);
        
        switch(dom_dim)
        {
            case 1:
            {
                switch(amb_dim)
                {
                    case 2:
                    {
                        r = new CLASS<1,2,Real,Int,SReal,ExtReal>( V_coords_, vertex_count_, simplices_, simplex_count_, thread_count_ );
                        break;
                    }
                    default:
                    {
                        r = new CLASS<1,3,Real,Int,SReal,ExtReal>( V_coords_, vertex_count_, simplices_, simplex_count_, thread_count_ );
                        break;
                    }
                }
                break;
            }
            case 2:
            {
                switch( amb_dim)
                {
                    default:
                    {
                        r = new CLASS<2,3,Real,Int,SReal,ExtReal>( V_coords_, vertex_count_, simplices_, simplex_count_, thread_count_ );
                        break;
                    }
                }
                break;
            }
            default:
            {
                r = new CLASS<2,3,Real,Int,SReal,ExtReal>( V_coords_, vertex_count_, simplices_, simplex_count_, thread_count_ );
                break;
            }
            
        }
        ptoc("MakeSimplicialMesh");
        return std::unique_ptr<BASE>(r);
    }
    
} // namespace Repulsion

#undef BASE
#undef CLASS
