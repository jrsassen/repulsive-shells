#pragma once

#define CLASS SimplicialMeshBase

namespace Repulsion
{
    template<typename Real, typename Int, typename SReal, typename ExtReal>
    class EnergyBase;
    
    template<typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS
    {
        ASSERT_FLOAT(Real);
        ASSERT_INT(Int);
        ASSERT_FLOAT(SReal);
        ASSERT_FLOAT(ExtReal);
        
        using ClusterTreeBase_T      =      ClusterTreeBase<Real,Int,SReal,ExtReal>;
        using BlockClusterTreeBase_T = BlockClusterTreeBase<Real,Int,SReal,ExtReal>;
        using CollisionTreeBase_T    =    CollisionTreeBase<Real,Int,SReal,ExtReal>;
        
    public:
        
        CLASS () {}
        
        explicit CLASS( const Int thread_count_ )
        :   thread_count( std::max( static_cast<Int>(1), thread_count_) )
        {};
        
        virtual ~CLASS() = default;

        mutable      ClusterTreeSettings       cluster_tree_settings =      ClusterTreeDefaultSettings;
        mutable BlockClusterTreeSettings block_cluster_tree_settings = BlockClusterTreeDefaultSettings;
        mutable       AdaptivitySettings         adaptivity_settings =       AdaptivityDefaultSettings;
        
    protected:
        
        const Int thread_count = 1;
            
    public:
        
        virtual Int DomDim() const = 0;
        
        virtual Int AmbDim() const = 0;

        virtual const Tensor2<Real,Int> & VertexCoordinates() const = 0;
        
        virtual const Tensor2<Int,Int> & Simplices() const = 0;

        virtual Int FarDim() const = 0;
        
        virtual Int NearDim() const = 0;
        
        virtual Int VertexCount() const = 0;
        
        virtual Int SimplexCount() const = 0;
        
        virtual Int DofCount() const = 0;
        
        virtual const Real * Dofs() const = 0;
        
        virtual Int ThreadCount() const
        {
            return thread_count;
        }
        
        virtual void SemiStaticUpdate( const ExtReal * restrict const V_coords_ ) = 0;
        
        virtual const ClusterTreeBase_T & GetClusterTree() const = 0;
        
        virtual const BlockClusterTreeBase_T & GetBlockClusterTree() const = 0;
        
        virtual const CollisionTreeBase_T & GetCollisionTree() const = 0;
                
        virtual const SparseBinaryMatrixCSR<Int> & DerivativeAssembler() const = 0;
        
        virtual void Assemble_ClusterTree_Derivatives(
            ExtReal * output,
            const ExtReal weight,
            bool addTo = false
        ) const = 0;
        
        virtual void LoadUpdateVectors( const ExtReal * restrict const vecs, const ExtReal max_time ) = 0;

        virtual SReal MaximumSafeStepSize( const ExtReal * restrict const vecs, const ExtReal max_time ) = 0;
        
//####################################################################################################
//      Obstacle
//####################################################################################################
        
    public:
        
        virtual void LoadObstacle( std::unique_ptr<CLASS> obstacle_ ) = 0;

        virtual const CLASS & GetObstacle() const = 0;
        
        virtual bool ObstacleInitialized() const = 0;
        
        virtual const ClusterTreeBase_T & GetObstacleClusterTree() const = 0;
        
        virtual const BlockClusterTreeBase_T & GetObstacleBlockClusterTree() const = 0;

//####################################################################################################
//      Tangent-point
//####################################################################################################

    public:
        
        virtual std::pair<Real,Real> GetTangentPointExponents() const = 0;
        
        virtual void SetTangentPointExponents( const Real alpha, const Real beta ) const = 0;
        
        virtual ExtReal GetTangentPointWeight() const = 0;
        
        virtual void SetTangentPointWeight( const ExtReal weight ) const = 0;
        
        virtual ExtReal TangentPointEnergy() const = 0;

        virtual ExtReal TangentPointEnergy_Differential( ExtReal * output, bool addTo = false ) const  = 0;
        
        ExtReal TangentPointEnergy_Differential( Tensor1<ExtReal,Int> & output, bool addTo = false ) const
        {
            return TangentPointEnergy_Differential( output.data(), addTo );
        }
        
        ExtReal TangentPointEnergy_Differential( Tensor2<ExtReal,Int> & output, bool addTo = false ) const
        {
            return TangentPointEnergy_Differential( output.data(), addTo );
        }

        virtual ExtReal TangentPointObstacleEnergy() const = 0;

        virtual ExtReal TangentPointObstacleEnergy_Differential( ExtReal * output, bool addTo = false ) const  = 0;
        
        ExtReal TangentPointObstacleEnergy_Differential( Tensor1<ExtReal,Int> & output, bool addTo = false ) const
        {
            return TangentPointObstacleEnergy_Differential( output.data(), addTo );
        }
        
        ExtReal TangentPointObstacleEnergy_Differential( Tensor2<ExtReal,Int> & output, bool addTo = false ) const
        {
            return TangentPointObstacleEnergy_Differential( output.data(), addTo );
        }
        
        
        virtual void TangentPointMetric_Multiply(
            const ExtReal alpha, const ExtReal * U,
            const ExtReal  beta,       ExtReal * V,
            Int cols
        ) const  = 0;
        
        void TangentPointMetric_Multiply(
            const ExtReal alpha, const Tensor1<ExtReal,Int> & U,
            const ExtReal  beta,       Tensor1<ExtReal,Int> & V
        ) const
        {
            TangentPointMetric_Multiply(alpha, U.data(), beta, V.data(), 1);
        }
        
        void TangentPointMetric_Multiply(
            const ExtReal alpha, const Tensor2<ExtReal,Int> & U,
            const ExtReal  beta,       Tensor2<ExtReal,Int> & V
        ) const
        {
            const Int cols = std::min( U.Dimension(2), V.Dimension(2) );
            TangentPointMetric_Multiply(alpha, U.data(), beta, V.data(), cols);
        }
        
        virtual void TangentPointMetric_Multiply(
            const ExtReal alpha, const ExtReal * U,
            const ExtReal  beta,       ExtReal * V,
            Int cols,
            KernelType kernel
        ) const = 0;
        
        void TangentPointMetric_Multiply(
            const ExtReal alpha, const Tensor1<ExtReal,Int> & U,
            const ExtReal  beta,       Tensor1<ExtReal,Int> & V,
            KernelType kernel
        ) const
        {
            TangentPointMetric_Multiply(alpha, U.data(), beta, V.data(), 1, kernel);
        }
        
        void TangentPointMetric_Multiply(
            const ExtReal alpha, const Tensor2<ExtReal,Int> & U,
            const ExtReal  beta,       Tensor2<ExtReal,Int> & V,
            KernelType kernel
        ) const
        {
            const Int cols = std::min( U.Dimension(2), V.Dimension(2) );
            TangentPointMetric_Multiply(alpha, U.data(), beta, V.data(), cols, kernel);
        }
        
        
        virtual const Tensor1<Real,Int> & TangentPointMetric_Values(
            const bool farQ,
            const KernelType kernel
        ) const = 0;
        
        virtual void TangentPointMetric_ApplyKernel(
            const bool farQ,
            const KernelType kernel
        ) const = 0;
   
//####################################################################################################
//      Custom energy (allows for loading arbitrary EnergyBase object )
//####################################################################################################
        
    public:
        
        virtual void LoadCustomEnergy( std::unique_ptr<EnergyBase<Real,Int,SReal,ExtReal>> e ) const = 0;
        
        virtual ExtReal GetCustomEnergyWeight() const = 0;

        virtual void SetCustomEnergyWeight( const ExtReal weight ) const = 0;

        virtual ExtReal CustomEnergy() const = 0;

        virtual ExtReal CustomEnergy_Differential( ExtReal * output, bool addTo = false ) const = 0;
 
        ExtReal CustomEnergy_Differential( Tensor1<ExtReal,Int> & output, bool addTo = false ) const
        {
            return CustomEnergy_Differential( output.data(), addTo );
        }
        
        ExtReal CustomEnergy_Differential( Tensor2<ExtReal,Int> & output, bool addTo = false ) const
        {
            return CustomEnergy_Differential( output.data(), addTo );
        }
        
//####################################################################################################
//      Trivial energy (for debugging purposes)
//####################################################################################################
        
    public:
        
        virtual ExtReal GetTrivialEnergyWeight() const = 0;

        virtual void SetTrivialEnergyWeight( const ExtReal weight ) const = 0;

        virtual ExtReal TrivialEnergy() const = 0;

        virtual ExtReal TrivialEnergy_Differential( ExtReal * output, bool addTo = false ) const = 0;

        ExtReal TrivialEnergy_Differential( Tensor1<ExtReal,Int> & output, bool addTo = false ) const
        {
            return CustomEnergy_Differential( output.data(), addTo );
        }
        
        ExtReal TrivialEnergy_Differential( Tensor2<ExtReal,Int> & output, bool addTo = false ) const
        {
            return CustomEnergy_Differential( output.data(), addTo );
        }
        
        virtual ExtReal TrivialObstacleEnergy() const = 0;

        virtual ExtReal TrivialObstacleEnergy_Differential( ExtReal * output, bool addTo = false ) const = 0;
        
        ExtReal TrivialObstacleEnergy_Differential( Tensor1<ExtReal,Int> & output, bool addTo = false ) const
        {
            return CustomEnergy_Differential( output.data(), addTo );
        }
        
        ExtReal TrivialObstacleEnergy_Differential( Tensor2<ExtReal,Int> & output, bool addTo = false ) const
        {
            return CustomEnergy_Differential( output.data(), addTo );
        }

        
    public:
                                                          
        virtual CLASS & DownCast() = 0;

        virtual const CLASS & DownCast() const = 0;
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
    };
    
} // namespace Repulsion

#undef CLASS
