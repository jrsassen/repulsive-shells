#pragma once

#define CLASS EnergyBase

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
        
        using MeshBase_T = SimplicialMeshBase<Real,Int,SReal,ExtReal>;

        explicit CLASS( ExtReal weight_ = static_cast<ExtReal>(1) )
        :
            weight(weight_)
        {}

        virtual ~CLASS() = default;
        
        // Returns the current value of the energy.
        virtual ExtReal Value( const MeshBase_T & M ) const = 0;
        
//        virtual ExtReal Value( const MESH_T & M ) const = 0;
        
        
        // Returns the current differential of the energy, stored in the given
        // V x AMB_DIM matrix, where each row holds the differential (a AMB_DIM-vector) with
        // respect to the corresponding vertex.
        virtual ExtReal Differential( const MeshBase_T & M, ExtReal * output, bool addTo = false ) const = 0;
        
//        virtual ExtReal Differential( const MESH_T & M, ExtReal * output, bool addTo = false ) const = 0;
        
        ExtReal GetWeight()  const
        {
            return weight;
        }
        
        void SetWeight( const ExtReal weight_ )
        {
            weight = weight_;
        }

        
    protected:
        
        ExtReal weight = static_cast<ExtReal>(1);
        
    public:
        
        virtual std::string Stats() const = 0;

        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };

}// namespace Repulsion

#undef CLASS
