#pragma once

#define CLASS TangentPointMetric_FMM
#define BASE  Metric_FMM<DOM_DIM,AMB_DIM,DEGREE,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:

        using BlockClusterTree_T = BlockClusterTree<AMB_DIM,Real,Int,SReal,ExtReal>;
        
        CLASS() {}
        
        CLASS(
            const BlockClusterTree_T & bct_,
            const Real alpha_,
            const Real beta_,
            const ExtReal weight = static_cast<ExtReal>(1) )
        :
        BASE(
            bct_,
            *std::make_unique<
                TangentPointMetric_NearFieldKernel<DOM_DIM,AMB_DIM,Real,Int,SReal>
                >(alpha_,beta_),
             *std::make_unique<
                 TangentPointMetric_FarFieldKernel_FMM       <DOM_DIM,AMB_DIM,DEGREE,Real,Int>
             >(alpha_,beta_),
            weight
        )
        {
            this->RequireMetrics();
        }
        
        virtual ~CLASS() override = default;
                
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef BASE
#undef CLASS
