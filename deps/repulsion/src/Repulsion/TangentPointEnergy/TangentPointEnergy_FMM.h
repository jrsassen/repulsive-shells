#pragma once

#define CLASS TangentPointEnergy_FMM
#define BASE  Energy_FMM<DOM_DIM,AMB_DIM,DEGREE,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:

        CLASS(
            const Real alpha_,
            const Real beta_,
            const ExtReal weight = static_cast<ExtReal>(1)
        )
        :   BASE(
            *Make_TangentPointEnergy_NearFieldKernel<DOM_DIM,DOM_DIM,AMB_DIM,Real,Int,SReal>( alpha_, beta_ ),
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
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
    template<int Degree, typename Real, typename Int, typename SReal, typename ExtReal>
    std::unique_ptr<EnergyBase<Real,Int,SReal,ExtReal>> CONCAT(Make_,CLASS) (
        const Int dom_dim, const Int amb_dim,
        const Real alpha, const Real beta, const Real weight
    )
    {
        EnergyBase<Real,Int,SReal,ExtReal> * r = nullptr;

        switch( dom_dim )
        {
            case 2:
            {
                switch( amb_dim )
                {
                    case 3:
                    {
                        r = new TangentPointEnergy_FMM<2,3,Degree,Real,Int,SReal,ExtReal>(
                            alpha, beta, weight
                        );
                        break;
                    }
                }
                break;
            }
            case 1:
            {
                switch( amb_dim )
                {
                    case 2:
                    {
                        r = new TangentPointEnergy_FMM<1,2,Degree,Real,Int,SReal,ExtReal>(
                            alpha, beta, weight
                        );
                        break;
                    }
                    case 3:
                    {
                        r = new TangentPointEnergy_FMM<1,3,Degree,Real,Int,SReal,ExtReal>(
                            alpha, beta, weight
                        );
                        break;
                    }
                }
                break;
            }
        }
        
        return std::unique_ptr<EnergyBase<Real,Int,SReal,ExtReal>>(r);
    }

}// namespace Repulsion

#undef BASE
#undef CLASS
