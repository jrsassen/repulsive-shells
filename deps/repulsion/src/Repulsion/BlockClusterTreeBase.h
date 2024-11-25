#pragma once

#define CLASS BlockClusterTreeBase

#define REPULSION_BLOCKCLUSTERTREE

namespace Repulsion
{
    
    template<typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        ASSERT_FLOAT(ExtReal);
        
    public:
        
        using Setting_T         = BlockClusterTreeSettings;
        
        using Near_T            = SparseBinaryMatrixVBSR<Int>;
        using  Far_T            = SparseBinaryMatrixCSR <Int>;
    
        using ClusterTreeBase_T = ClusterTreeBase<Real,Int,SReal,ExtReal>;

    protected:

    public:
        
        virtual ~CLASS() = default;
        
        virtual Int AmbDim() const = 0;

        virtual Int ThreadCount() const = 0;
        
        virtual Real SeparationParameter() const = 0;
        
        virtual Real AdaptivityParameter() const = 0;
        
        virtual bool IsSymmetric() const = 0;
        
        virtual bool ExploitSymmetry() const = 0;
        
        virtual bool NearUpperTriangular() const = 0;

        virtual bool FarUpperTriangular() const = 0;

        virtual Int SeparatedBlockCount() const = 0;
        
        virtual Int NonseparatedBlockCount() const = 0;
        
        virtual Int NearFieldInteractionCount() const = 0;

        virtual const Far_T & Far() const = 0;
        
        virtual const Near_T & Near() const = 0;
        
        virtual const ClusterTreeBase_T & GetS() const = 0;
        
        virtual const ClusterTreeBase_T & GetT() const = 0;
        
        virtual void RequireAdaptiveSubdivisionData() const = 0;
        
        virtual Int CollisionCount() const = 0;
        
        virtual const SparseMatrixCSR<Int,Int> & CollisionMatrix() const = 0;
        
        virtual const SparseMatrixCSR <Int,Int> & AdaptiveNoSubdivisionData() const = 0;
        
        virtual const SparseMatrixCSR <Int,Int> & AdaptiveSubdivisionData()   const = 0;

        virtual const Setting_T & Settings() const = 0;
        
        virtual std::string Stats() const = 0;
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS) + "<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
} //namespace Repulsion

#undef CLASS
