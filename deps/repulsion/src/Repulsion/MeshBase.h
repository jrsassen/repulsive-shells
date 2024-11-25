#pragma once

namespace Repulsion
{
    
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class MeshBase
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        ASSERT_FLOAT(ExtReal);
        
        using ClusterTree_T       = ClusterTree     <AMB_DIM,Real,Int,SReal,ExtReal>;
        using BlockClusterTree_T = BlockClusterTree<AMB_DIM,Real,Int,SReal,ExtReal>;
        
    public:
        
        MeshBase( const Int thread_count_ )
        :   thread_count( std::max( static_cast<Int>(1), thread_count_) )
        {
//            ptic(ClassName());
//            ptoc(ClassName());
        };
        
        virtual ~MeshBase() = default;

             ClusterTreeSettings       cluster_tree_settings;
        BlockClusterTreeSettings block_cluster_tree_settings;
        
    protected:
        
        const Int thread_count;
        
    public:
        
        virtual Int DofCount() const = 0;
        
        virtual const Real * Dofs() const = 0;
        
        static constexpr Int DomDim()
        {
            return DOM_DIM;
        }
        
        static constexpr Int AmbDim()
        {
            return AMB_DIM;
        };
        
        virtual Int ThreadCount() const
        {
            return thread_count;
        }
        
        virtual void SemiStaticUpdate( const ExtReal * restrict const V_coords_ ) = 0;
        
        virtual const ClusterTree_T & GetClusterTree() const = 0;
        
        virtual const BlockClusterTree_T & GetBlockClusterTree() const = 0;
        
        virtual const ClusterTree_T & GetObstacleClusterTree() const = 0;
        
        virtual const BlockClusterTree_T & GetObstacleBlockClusterTree() const = 0;
        
        virtual void Assemble_ClusterTree_Derivatives(
            ExtReal * output,
            const Real weight,
            bool addTo = false
        ) const = 0;
        
        protected :
        
    public:
        
        virtual std::string ClassName() const
        {
            return "MeshBase<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
  
    }; // MeshBase
    
} // namespace Repulsion
