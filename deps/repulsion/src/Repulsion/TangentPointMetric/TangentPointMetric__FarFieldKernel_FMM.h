#pragma once

#define CLASS TangentPointMetric_FarFieldKernel_FMM
#define BASE  Metric_FarFieldKernel_FMM<AMB_DIM,DEGREE,Real,Int>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::S_far;
        using BASE::T_far;
        
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

        using BASE::LoadS;
        using BASE::LoadT;
        
    public:
        
        CLASS( const Real alpha_, const Real beta_ )
        :
            BASE(),
            alpha(alpha_),
            beta(beta_),
            exp_s( (beta_ - DOM_DIM) / alpha_ ),
            hi_exponent( static_cast<Real>(-0.5) * (static_cast<Real>(2) * (exp_s - static_cast<Real>(1)) + DOM_DIM) ) // multiplying by 0.5 because we use r * r instead of r for saving a sqrt
        {}

        // Copy constructor
        CLASS( const CLASS & other )
        :
            BASE        ( other ),
            alpha       ( other.alpha ),
            beta        ( other.beta ),
            exp_s       ( other.exp_s ),
            hi_exponent ( other.hi_exponent ),
            fr_values   ( other.fr_values ),
            hi_values   ( other.hi_values ),
            lo_values   ( other.lo_values )
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :
            BASE        ( other ),
            alpha       ( other.alpha ),
            beta        ( other.beta ),
            exp_s       ( other.exp_s ),
            hi_exponent ( other.hi_exponent ),
            fr_values   ( other.fr_values ),
            hi_values   ( other.hi_values ),
            lo_values   ( other.lo_values )
        {
            other.fr_values = nullptr;
            other.hi_values = nullptr;
            other.lo_values = nullptr;
        }
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
    protected:
        
        const Real alpha       = static_cast<Real>(2) * (DOM_DIM + 1);
        const Real beta        = static_cast<Real>(4) * (DOM_DIM + 1);
        
        const Real exp_s       = (beta - DOM_DIM) / alpha; // differentiability of the energy space
        const Real hi_exponent = -static_cast<Real>(0.5) * (static_cast<Real>(2) * (exp_s - static_cast<Real>(1)) + DOM_DIM); // The only exponent we have to use for pow to compute matrix entries. All other exponents have been optimized away.
        
        Real * restrict fr_values = nullptr;
        Real * restrict hi_values = nullptr;
        Real * restrict lo_values = nullptr;
        
    public:

        virtual void AllocateValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & far_values, const Int nnz ) override
        {
            far_values[KernelType::FractionalOnly] = Tensor1<Real,Int> (nnz);
            far_values[KernelType::HighOrder]      = Tensor1<Real,Int> (nnz);
            far_values[KernelType::LowOrder]       = Tensor1<Real,Int> (nnz);
            
            fr_values = far_values[KernelType::FractionalOnly].data();
            hi_values = far_values[KernelType::HighOrder].data();
            lo_values = far_values[KernelType::LowOrder].data();
        }
        
        virtual void LoadValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & far_values ) override
        {
            fr_values = far_values[KernelType::FractionalOnly].data();
            hi_values = far_values[KernelType::HighOrder].data();
            lo_values = far_values[KernelType::LowOrder].data();
        }
        
        
        virtual void Metric( const Int pos ) override
        {
            Real v [AMB_DIM] = {};
            
            Real r2       = static_cast<Real>(0);
            Real rCosPhi2 = static_cast<Real>(0);
            Real rCosPsi2 = static_cast<Real>(0);
            
            SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                v[i] = y[i] - x[i];
                r2  += v[i] * v[i];
            }

            SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                Real Pv_i = static_cast<Real>(0);
                Real Qv_i = static_cast<Real>(0);
                
                SMALL_UNROLL
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    Int k = lin_k[i][j];
                    Pv_i += p[k] * v[j];
                    Qv_i += q[k] * v[j];
                }
                rCosPhi2 += v[i] * Pv_i;
                rCosPsi2 += v[i] * Qv_i;
            }

            const Real r4 = r2 * r2;
            const Real mul = r4 * MyMath::pow<Real,Int>( r2, static_cast<Int>(DOM_DIM-1) );
            
            // The following line makes up approx 2/3 of this function's runtime! This is why we avoid pow as much as possible and replace it with MyMath::pow.;
            // I got it down to this single call to pow. We might want to generate a lookup table for it...;
            // The factor of (-2.) is here, because we assemble the _metric_, not the kernel.;
            const Real power = static_cast<Real>(-2) * MyMath::pow(r2, hi_exponent);
            
            hi_values[pos] = power;

            fr_values[pos] = static_cast<Real>(4) / (power * mul);

            lo_values[pos] = static_cast<Real>(0.5) * (rCosPhi2 + rCosPsi2) / r4 * power;
        }
        
    public:
        
        virtual std::string Stats() const override
        {
            return ClassName()+": alpha = "+ToString(alpha)+", beta = "+ToString(beta);
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef CLASS
#undef BASE
