#pragma once

#define CLASS Energy_FarFieldKernel_FMM
#define BASE  FarFieldKernelBase_FMM<AMB_DIM,DEGREE,Real,Int>

namespace Repulsion
{
    template<int AMB_DIM, int DEGREE, typename Real, typename Int>
    class CLASS : public BASE
    {
    public:
        
        CLASS() : BASE() {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}

        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE(other) {}

        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
        
        virtual Real Energy() = 0;
        
        virtual Real DEnergy() = 0;
        
    protected:
        
        using BASE::S_far;
        using BASE::S_D_far;
        
        using BASE::T_far;
        using BASE::T_D_far;
        
        using BASE::DX;
        using BASE::DY;
        
        using BASE::a;
        using BASE::x;
        using BASE::p;
        
        using BASE::b;
        using BASE::y;
        using BASE::q;
        
        using BASE::S_ID;
        using BASE::T_ID;
        
        using BASE::tri_i;
        using BASE::tri_j;
        using BASE::lin_k;
        
    public:
        
        using BASE::LoadFarField;
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef BASE
#undef CLASS
