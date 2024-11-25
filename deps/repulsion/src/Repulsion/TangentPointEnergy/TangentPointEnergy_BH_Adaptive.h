#pragma once

#define CLASS TangentPointEnergy_BH_Adaptive
#define BASE  Energy_BH<DOM_DIM,AMB_DIM,DEGREE,Real,Int>

namespace Repulsion
{
    template<mint DOM_DIM, mint AMB_DIM, mint DEGREE, typename Real, typename Int>
    class CLASS : public BASE
    {
    public:

        CLASS( const Real alpha, const Real beta, const Real theta_near, const Real theta_far, Real weight = static_cast<Real>(1) )
        :
        BASE(
            *Make_TangentPointEnergy_NearFieldKernel_Adaptive<DOM_DIM,AMB_DIM,Real,Int>( alpha, beta, theta_near ),
            *Make_TangentPointEnergy_FarFieldKernel_BH<DOM_DIM,AMB_DIM,DEGREE,Real,Int>( alpha, beta, theta_far ),
            weight
        )
        {}

        virtual ~CLASS() override = default;
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef BASE
#undef CLASS
