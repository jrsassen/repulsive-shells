#pragma once

#define CLASS TangentPointMetric_NearFieldKernel_Adaptive
#define BASE  Metric_NearFieldKernel_Adaptive<DOM_DIM,DOM_DIM,AMB_DIM,Real,Int,SReal>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::S_near;
        using BASE::S_D_near;
        using BASE::S_serialized;
        
        using BASE::T_near;
        using BASE::T_D_near;
        using BASE::T_serialized;
        
        using BASE::a;
        using BASE::x_buffer;
        using BASE::p;
        
        using BASE::b;
        using BASE::y_buffer;
        using BASE::q;

        using BASE::settings;
        using BASE::S;
        using BASE::T;
        
        using BASE::S_ID;
        using BASE::T_ID;
        
        using BASE::tri_i;
        using BASE::tri_j;
        using BASE::lin_k;
        
        using BASE::lambda;
        using BASE::mu;

    public:
        
        using BASE::NearDimS;
        using BASE::NearDimT;
        using BASE::CoordDimS;
        using BASE::CoordDimT;
        using BASE::ProjectorDim;
        
    public:
        
        CLASS( const Real alpha_, const Real beta_,
               const AdaptivitySettings & settings_ = AdaptivityDefaultSettings
        )
        :
            BASE( settings_ ),
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
            hi_exponent ( other.hi_exponent )
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :
            BASE        ( other ),
            alpha       ( other.alpha ),
            beta        ( other.beta ),
            exp_s       ( other.exp_s ),
            hi_exponent ( other.hi_exponent )
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
    protected:
            
            const Real alpha;
            const Real beta;
            
            const Real exp_s; // differentiability of the energy space
            const Real hi_exponent; // The only exponent we have to use for pow to compute matrix entries. All other exponents have been optimized away.
            
            Real * restrict fr_values = nullptr;
            Real * restrict hi_values = nullptr;
            Real * restrict lo_values = nullptr;
            
            Real fr_value = static_cast<Real>(0);
            Real hi_value = static_cast<Real>(0);
            Real lo_value = static_cast<Real>(0);
        
            Real hi_diag  = static_cast<Real>(0);
            Real fr_diag  = static_cast<Real>(0);
            Real lo_diag  = static_cast<Real>(0);
        
    public:
        
        virtual void AllocateValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & near_values, const Int nnz ) override
        {
            near_values[KernelType::FractionalOnly] = Tensor1<Real,Int> (nnz);
            near_values[KernelType::HighOrder]      = Tensor1<Real,Int> (nnz);
            near_values[KernelType::LowOrder]       = Tensor1<Real,Int> (nnz);
            
            fr_values = near_values[KernelType::FractionalOnly].data();
            hi_values = near_values[KernelType::HighOrder].data();
            lo_values = near_values[KernelType::LowOrder].data();
        }
        
        virtual void LoadValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & near_values ) override
        {
            fr_values = near_values[KernelType::FractionalOnly].data();
            hi_values = near_values[KernelType::HighOrder].data();
            lo_values = near_values[KernelType::LowOrder].data();
        }
        
//        virtual void LoadS( const Int i ) override
//        {
////            if( i != S_ID )
////            {
//                S_ID = i;
//                const Real * const X  = S_near + NearDim() * S_ID;
//        
//                S.RequireSimplex(S_serialized, S_ID);
//                
//                a = X[0];
//            
//#ifdef NearField_S_Copy
//                memcpy( &x_buffer[0], X+1,                CoordDim() * sizeof(Real) );
//                memcpy( &p[0],        X+1+CoordDim(), ProjectorDim() * sizeof(Real) );
//#else
//                x_buffer = X+1;
//                p        = X+1+CoordDim();
//#endif
////            }
//        }
//        
//        virtual void LoadT( const Int j ) override
//        {
////            if( j != T_ID )
////            {
//                T_ID = j;
//                const Real * const Y  = T_near + NearDim() * T_ID;
//        
//                T.RequireSimplex(T_serialized, T_ID);
//                
//                b = Y[0];
//            
//#ifdef NearField_T_Copy
//                memcpy( &y_buffer[0], Y+1,                CoordDim() * sizeof(Real) );
//                memcpy( &q[0],        Y+1+CoordDim(), ProjectorDim() * sizeof(Real) );
//#else
//                y_buffer = Y+1;
//                q        = Y+1+CoordDim();
//#endif
////            }
//        }
        
        
//        virtual void StartRow() override
//        {
//            hi_diag = static_cast<Real>(0);
//            fr_diag = static_cast<Real>(0);
//            lo_diag = static_cast<Real>(0);
//        }
        
        virtual void StartEntry() override
        {
            hi_value = static_cast<Real>(0);
            fr_value = static_cast<Real>(0);
            lo_value = static_cast<Real>(0);
        }
        
        virtual void FinishEntry( const Int pos ) override
        {
            hi_values[pos] = hi_value;
            fr_values[pos] = fr_value;
            lo_values[pos] = lo_value;
            
            hi_diag -= hi_value * b;
            fr_diag -= fr_value * b;
            lo_diag -= lo_value * b;
        }

        
//        virtual void FinishRow( const Int diag_pos ) override
//        {
//            const Real ainv = static_cast<Real>(1) / a;
//            fr_values[diag_pos] = fr_diag * ainv;
//            hi_values[diag_pos] = hi_diag * ainv;
//            lo_values[diag_pos] = lo_diag * ainv;
//        }

        virtual void metric() override
        {
            Real x [AMB_DIM] = {};
            Real y [AMB_DIM] = {};
            Real v [AMB_DIM] = {};
            
            const Real delta = static_cast<Real>(S_ID == T_ID);
            
            Real r2 = delta;
            Real rCosPhi2 = static_cast<Real>(0);
            Real rCosPsi2 = static_cast<Real>(0);
            
            SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                x[i] = lambda[0] * x_buffer[AMB_DIM*0 +i];
                y[i] = mu    [0] * y_buffer[AMB_DIM*0 +i];

                SMALL_UNROLL
                for( Int ii = 1; ii < DOM_DIM+1; ++ii )
                {
                    x[i] += lambda[ii] * x_buffer[AMB_DIM*ii +i];
                }
                
                SMALL_UNROLL
                for( Int ii = 1; ii < DOM_DIM+1; ++ii )
                {
                    y[i] += mu    [ii] * y_buffer[AMB_DIM*ii +i];
                }
            }
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

            const Real w = S.Weight() * T.Weight();
            
            hi_value += w * (static_cast<Real>(1) - delta) * power;
            
            fr_value += w * static_cast<Real>(4) * (static_cast<Real>(1) - delta) / (power * mul);
            
            lo_value += w * static_cast<Real>(0.5) * (static_cast<Real>(1) - delta) * (rCosPhi2 + rCosPsi2) / r4 * power;
        }
        
    public:
        
        virtual std::string Stats() const override
        {
            return ClassName()+": alpha = "+ToString(alpha)+", beta = "+ToString(beta);
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef BASE
#undef CLASS

