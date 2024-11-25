#pragma once

#define CLASS MetricBase

namespace Repulsion
{
    template<typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        ASSERT_FLOAT(ExtReal);
        
    protected:
        
        mutable ExtReal weight;
        
        virtual void RequireMetrics() const = 0;
        
    public:
        
        CLASS() {}
        
        CLASS( const ExtReal weight_ ) : weight(weight_) {}

        virtual ~CLASS() = default;

        
        virtual ExtReal GetWeight() const
        {
            return weight;
        }
        
        virtual ExtReal SetWeight( const ExtReal weight_ ) const
        {
            return weight = weight_;
        }
        
        virtual const std::map<KernelType, Tensor1<Real,Int>> & NearFieldValues() const = 0;

        virtual const std::map<KernelType, Tensor1<Real,Int>> & FarFieldValues() const = 0;

        
//#################################################################################################
//      Matrix multiplication
//#################################################################################################

        virtual void ApplyNearFieldKernel( const Real factor, const KernelType type ) const = 0;
        
        virtual void ApplyFarFieldKernel ( const Real factor, const KernelType type ) const = 0;
        
        virtual void Multiply_DenseMatrix(
            const Real    alpha, const ExtReal * X,
            const ExtReal  beta,       ExtReal * Y,
            const Int cols,
            const KernelType type
        ) const = 0;

        virtual void Multiply_DenseMatrix(
            const Real    alpha, const Tensor1<ExtReal,Int> & X,
            const ExtReal beta,        Tensor1<ExtReal,Int> & Y,
            KernelType type
        ) const = 0;
        
        virtual void Multiply_DenseMatrix(
            const Real    alpha, const Tensor2<ExtReal,Int> & X,
            const ExtReal beta,        Tensor2<ExtReal,Int> & Y,
            KernelType type
        ) const = 0;
        
        virtual void TestDiagonal() const = 0;
        
    public:
        
        virtual std::string Stats() const = 0;
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef CLASS
