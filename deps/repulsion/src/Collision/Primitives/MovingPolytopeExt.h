#pragma once

#define CLASS MovingPolytopeExt
#define BASE  MovingPolytopeBase<AMB_DIM,Real,Int,SReal>

namespace Collision {
    
    template<int AMB_DIM, typename Real,typename Int, typename SReal, typename ExtReal, typename ExtInt>
    class CLASS : public BASE
    {
        ASSERT_FLOAT(ExtReal);
        
    protected:

        using BASE::a;
        using BASE::b;
        using BASE::T;
        
    public:
        
        CLASS() {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE(other)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :   BASE(other)
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
        
        virtual void FromCoordinates( const ExtReal * const p_, const Int i = 0 ) = 0;
        
        virtual void FromVelocities ( const ExtReal * const v_, const Int i = 0 ) = 0;
        
        virtual void FromVelocitiesIndexList(
            const ExtReal * const v_, const ExtInt * const tuples, const Int i ) = 0;
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };

} // Collision

#undef CLASS
#undef BASE
