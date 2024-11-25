#pragma once

#define CLASS TangentPointEnergy_FarFieldKernel_BH
#define BASE  Energy_FarFieldKernel_BH<DOM_DIM,AMB_DIM,DEGREE,Real,Int>

namespace Repulsion
{
    template<mint DOM_DIM, mint AMB_DIM, mint DEGREE, typename T1, typename T2, typename Real, typename Int>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::S_far;
        using BASE::S_D_far;
        
        using BASE::T_near;
        using BASE::T_D_near;
        
        using BASE::DX;
        using BASE::DY;
        
        using BASE::a;
        using BASE::x;
//        using BASE::p;
        
        using BASE::b;
        using BASE::y;
        using BASE::q;
        
        using BASE::S_ID;
        using BASE::T_ID;
        
        using BASE::tri_i;
        using BASE::tri_j;
        using BASE::lin_k;
        
        using BASE::T_scale;
        
    public:
        
        CLASS( const T1 alphahalf_, const T2 betahalf_, const Real theta_ )
        :
                BASE(theta_),
                alpha(static_cast<Real>(2)*alphahalf_),
                alphahalf(alphahalf_),
                alphahalf_minus_1(alphahalf-1),
                beta(static_cast<Real>(2)*betahalf_),
                betahalf(betahalf_),
                minus_betahalf(-betahalf),
                minus_betahalf_minus_1(-betahalf-1)
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :
            BASE( other ),
            alpha(other.alpha),
            alphahalf(other.alphahalf),
            alphahalf_minus_1(other.alphahalf_minus_1),
            beta(other.beta),
            betahalf(other.betahalf),
            minus_betahalf(other.minus_betahalf),
            minus_betahalf_minus_1(other.minus_betahalf_minus_1)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :
            BASE( other ),
            alpha(other.alpha),
            alphahalf(other.alphahalf),
            alphahalf_minus_1(other.alphahalf_minus_1),
            beta(other.beta),
            betahalf(other.betahalf),
            minus_betahalf(other.minus_betahalf),
            minus_betahalf_minus_1(other.minus_betahalf_minus_1)
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
    protected:
        
        const Real alpha;
        const T1 alphahalf;
        const T1 alphahalf_minus_1;
        
        const Real beta;
        const T2 betahalf;
        const T2 minus_betahalf;
        const T2 minus_betahalf_minus_1;
        
    public:
        
        virtual Real Energy() override
        {
            Real v    [AMB_DIM] = {};
            
            Real r2 = static_cast<Real>(0);
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
                Real Qv_i = static_cast<Real>(0);
                
                SMALL_UNROLL
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    Int k = lin_k[i][j];
                    Qv_i += q[k] * v[j];
                }
                rCosPsi2 += v[i] * Qv_i;
            }

            const Real rCosPsiAlphaMinus2 = MyMath::pow<Real,T1>( fabs(rCosPsi2), alphahalf_minus_1);
            const Real rMinusBetaMinus2   = MyMath::pow<Real,T2>( r2, minus_betahalf_minus_1 );

            const Real rMinusBeta = rMinusBetaMinus2 * r2;
            const Real rCosPsiAlpha = rCosPsiAlphaMinus2 * rCosPsi2;
            
            const Real E = rCosPsiAlpha * rMinusBeta;
        
            return a * E * b;
        }
        
        virtual Real DEnergy() override
        {
            Real v    [AMB_DIM] = {};
            Real Qv   [AMB_DIM] = {};
            Real dEdv [AMB_DIM] = {};
            Real V    [(AMB_DIM*(AMB_DIM+1))/2] = {};
            
            Real r2 = static_cast<Real>(0);
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
                Qv[i] = static_cast<Real>(0);
                
                SMALL_UNROLL
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    Int k = lin_k[i][j];
                    Qv[i] += q[k] * v[j];
                    if( j >= i )
                    {
                        V[k] = (static_cast<Real>(1+i!=j))*v[i]*v[j];
                    }
                }
                rCosPsi2 += v[i] * Qv[i];
            }
            
            const Real rCosPsiAlphaMinus2 = MyMath::pow<Real,T1>( fabs(rCosPsi2), alphahalf_minus_1);
            const Real rMinusBetaMinus2   = MyMath::pow<Real,T2>( r2, minus_betahalf_minus_1 );

            const Real rMinusBeta = rMinusBetaMinus2 * r2;
            const Real rCosPsiAlpha = rCosPsiAlphaMinus2 * rCosPsi2;

            const Real E = rCosPsiAlpha * rMinusBeta;

            const Real factor = alphahalf * rMinusBeta;
            const Real G = factor * rCosPsiAlphaMinus2;
            const Real H = - beta * rMinusBetaMinus2 * rCosPsiAlpha;
        
            Real dEdvx = static_cast<Real>(0);
            Real dEdvy = static_cast<Real>(0);
            
            SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                dEdv[i] = static_cast<Real>(2) * G * Qv[i] + H * v[i];
                dEdvx += dEdv[i] * x[i];
                dEdvy += dEdv[i] * y[i];
                
                DX[1+i] -=  b * dEdv[i];
                for( Int ii = 0; ii < DOM_DIM+1; ++ii )
                {
                    this->DY[1+AMB_DIM*ii+i] += T_scale * a * dEdv[i];
                }
            }
        
            DX[0] += b * ( E + dEdvx );
            DY[0] += a * ( E - factor * rCosPsiAlpha - dEdvy );
            
            const Real aG = a * G;
            
            SMALL_UNROLL
            for( Int k = 0; k < (AMB_DIM*(AMB_DIM+1))/2; ++k )
            {
                DY[1+(DOM_DIM+1)*AMB_DIM+k] += aG * V[k];
            }
            
            return a * E * b;
        }
        

    public:
        
        virtual std:;string Stats() const override
        {
            return ClassName()+": alpha = "+ToString(alpha)+", beta = "+ToString(beta);
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<T1>::Get()+","+TypeName<T2>::Get()+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
    template<mint DOM_DIM, mint AMB_DIM, mint DEGREE, typename Real, typename Int>
    std::unique_ptr<BASE> CONCAT(Make_,CLASS)( const Real alpha, const Real beta, const Real theta_far  )
    {
        Real alphahalf_intpart;
        bool alphahalf_int = (std::modf( alpha/static_cast<Real>(2), &alphahalf_intpart) == static_cast<Real>(0));
        
        Real betahalf_intpart;
        bool betahalf_int = (std::modf( beta/static_cast<Real>(2), &betahalf_intpart) == static_cast<Real>(0));
        
        BASE * r;
        
        if( alphahalf_int )
        {
            if( betahalf_int )
            {
                r = new CLASS<DOM_DIM,AMB_DIM,DEGREE,Int,Int,Real,Int>(alphahalf_intpart,betahalf_intpart,theta_far);
            }
            else
            {
                r = new CLASS<DOM_DIM,AMB_DIM,DEGREE,Int,Real,Real,Int>(alphahalf_intpart,beta/static_cast<Real>(2),theta_far);
            }
        }
        else
        {
            if( betahalf_int )
            {
                r = new CLASS<DOM_DIM,AMB_DIM,DEGREE,Real,Int,Real,Int>(alpha/static_cast<Real>(2),betahalf_intpart,theta_far);
            }
            else
            {
                r = new CLASS<DOM_DIM,AMB_DIM,DEGREE,Real,Real,Real,Int>(alpha/static_cast<Real>(2),beta/static_cast<Real>(2),theta_far);
            }
        }
        
        return std::unique_ptr<BASE>(r);
    }
    
} // namespace Repulsion

#undef CLASS
#undef BASE
