#pragma once

#define CLASS CollisionTreeBase

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
        
        using ClusterTreeBase_T = ClusterTreeBase<Real,Int,SReal,ExtReal>;
        
        using CollisionMatrix_T = SparseMatrixCSR<SReal,Int>;
        
        CLASS () {}
        
        virtual ~CLASS() = default;
        
    protected:
        
        
    public:
        
        virtual Int AmbDim() const = 0;

        virtual Int ThreadCount() const = 0;
        
        virtual bool IsSymmetric() const = 0;

        virtual const CollisionMatrix_T & ClusterCollisionMatrix() const = 0;
        
        virtual const CollisionMatrix_T & PrimitiveCollisionMatrix() const = 0;
        
        virtual const ClusterTreeBase_T & GetS() const = 0;
        
        virtual const ClusterTreeBase_T & GetT() const = 0;

        virtual SReal MaximumSafeStepSize() const = 0;
        
//        virtual Int CollisionCount() const = 0;
        
//        virtual std::string Stats() const = 0;
        

        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS) + "<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }

    }; // CLASS
    
} // namespace Repulsion

#undef CLASS
