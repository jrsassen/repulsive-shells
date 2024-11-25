#pragma once

#define CLASS Energy_Restricted
#define BASE  EnergyBase<Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:
        
        using MeshBase_T = typename BASE::MeshBase_T;
        using Mesh_T     = SimplicialMesh<DOM_DIM,AMB_DIM,Real,Int,SReal,ExtReal>;

        explicit CLASS( ExtReal weight_ = static_cast<ExtReal>(1) )
        :   BASE(weight_)
        {}

        virtual ~CLASS() override = default;

        // Returns the current value of the energy.
        ExtReal Value( const MeshBase_T & M ) const override
        {
            ptic(ClassName()+"::Value");
            
            const Mesh_T * Q = dynamic_cast<const Mesh_T *>(&M);
            
            ptoc(ClassName()+"::Value");
            
            if( Q != nullptr)
            {
                return Value(*Q);
            }
            else
            {
                eprint(ClassName()+"::Value: Could not downcast input to compatible type. Returning 0.");
                return static_cast<ExtReal>(0);
            }
        }
        
        // Returns the current value of the energy.
        virtual ExtReal Value( const Mesh_T & M ) const = 0;
        
        
        ExtReal Differential( const MeshBase_T & M, ExtReal * output, bool addTo = false ) const override
        {
            const Mesh_T * Q = dynamic_cast<const Mesh_T *>(&M);
                        
            if( Q != nullptr )
            {
                return Differential(*Q, output, addTo);
            }
            else
            {
                eprint(ClassName()+"::Differential: Could not downcast input to compatible type. Doing nothing.");
                return static_cast<ExtReal>(0);
            }
        }
        
        // Returns the current differential of the energy, stored in the given
        // V x AMB_DIM matrix, where each row holds the differential (a AMB_DIM-vector) with
        // respect to the corresponding vertex.
        virtual ExtReal Differential( const Mesh_T & M, ExtReal * output, bool addTo = false ) const = 0;

    public:
        
        virtual std::string Stats() const override = 0;
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };

}// namespace Repulsion

#undef BASE
#undef CLASS
