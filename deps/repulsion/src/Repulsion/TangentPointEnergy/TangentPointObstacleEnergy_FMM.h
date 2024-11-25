#pragma once

#define CLASS TangentPointObstacleEnergy_FMM
#define BASE  ObstacleEnergy_FMM<DOM_DIM1,DOM_DIM2,AMB_DIM,DEGREE,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:

        CLASS(
            const Real alpha_,
            const Real beta_,
            const ExtReal weight = static_cast<ExtReal>(1)
        )
        :   BASE(
            *Make_TangentPointEnergy_NearFieldKernel<DOM_DIM1,DOM_DIM2,AMB_DIM,Real,Int,SReal>( alpha_, beta_ ),
            *Make_TangentPointEnergy_FarFieldKernel_FMM<AMB_DIM,DEGREE,Real,Int>( alpha_, beta_ ),
            weight
             )
        ,   alpha(alpha_)
        ,   beta(beta_)
        {}

        virtual ~CLASS() override = default;
                
    protected:
        
        const Real alpha;
        const Real beta;
        
        
    public:
        
        Real GetAlpha() const
        {
            return alpha;
        }
        
        Real GetBeta() const
        {
            return beta;
        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM1)+","+ToString(DOM_DIM2)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef BASE
#undef CLASS
