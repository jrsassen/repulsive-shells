#pragma once

#define CLASS TangentPointEnergy_FarFieldKernel_FMM
#define BASE  Energy_FarFieldKernel_FMM<AMB_DIM,DEGREE,Real,Int>

namespace Repulsion
{
    template<int AMB_DIM, int DEGREE, typename T1, typename T2, typename Real, typename Int>
    class CLASS : public BASE
    {
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
        
        using BASE::CoordDim;
        using BASE::ProjectorDim;
        
        CLASS( const T1 alphahalf_, const T2 betahalf_ )
        :
                BASE(),
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
            
            Real r2       = static_cast<Real>(0);
            Real rCosPhi2 = static_cast<Real>(0);
            Real rCosPsi2 = static_cast<Real>(0);
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                v[i] = y[i] - x[i];
                r2  += v[i] * v[i];
            }

            for( Int i = 0; i < AMB_DIM; ++i )
            {
                Real Pv_i = static_cast<Real>(0);
                Real Qv_i = static_cast<Real>(0);
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    Int k = lin_k[i][j];
                    Pv_i += p[k] * v[j];
                    Qv_i += q[k] * v[j];
                }
                rCosPhi2 += v[i] * Pv_i;
                rCosPsi2 += v[i] * Qv_i;
            }

            const Real rCosPhiAlphaMinus2 = MyMath::pow<Real,T1>( fabs(rCosPhi2), alphahalf_minus_1);
            const Real rCosPsiAlphaMinus2 = MyMath::pow<Real,T1>( fabs(rCosPsi2), alphahalf_minus_1);
            const Real rMinusBetaMinus2   = MyMath::pow<Real,T2>( r2, minus_betahalf_minus_1 );

            const Real rMinusBeta   = rMinusBetaMinus2 * r2;
            const Real rCosPhiAlpha = rCosPhiAlphaMinus2 * rCosPhi2;
            const Real rCosPsiAlpha = rCosPsiAlphaMinus2 * rCosPsi2;
            const Real Num = ( rCosPhiAlpha + rCosPsiAlpha );
            
            const Real E = Num * rMinusBeta;
            
            return a * E * b;
        }
        
        virtual Real DEnergy() override
        {
            Real v    [AMB_DIM] = {};
            Real Pv   [AMB_DIM] = {};
            Real Qv   [AMB_DIM] = {};
            Real dEdv [AMB_DIM] = {};
            Real V    [(AMB_DIM*(AMB_DIM+1))/2] = {};
            
            Real r2       = static_cast<Real>(0);
            Real rCosPhi2 = static_cast<Real>(0);
            Real rCosPsi2 = static_cast<Real>(0);
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                v[i] = y[i] - x[i];
                r2  += v[i] * v[i];
            }

            for( Int i = 0; i < AMB_DIM; ++i )
            {
                Pv[i] = static_cast<Real>(0);
                Qv[i] = static_cast<Real>(0);
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    Int k = lin_k[i][j];
                    Pv[i] += p[k] * v[j];
                    Qv[i] += q[k] * v[j];
                    if( j >= i )
                    {
                        V[k] = ( static_cast<Real>(1) + static_cast<Real>( (i!=j) ) )*v[i]*v[j];
                    }
                }
                rCosPhi2 += v[i] * Pv[i];
                rCosPsi2 += v[i] * Qv[i];
            }
            
            const Real rCosPhiAlphaMinus2 = MyMath::pow<Real,T1>( fabs(rCosPhi2), alphahalf_minus_1);
            const Real rCosPsiAlphaMinus2 = MyMath::pow<Real,T1>( fabs(rCosPsi2), alphahalf_minus_1);
            const Real rMinusBetaMinus2   = MyMath::pow<Real,T2>( r2, minus_betahalf_minus_1 );

            const Real rMinusBeta   = rMinusBetaMinus2 * r2;
            const Real rCosPhiAlpha = rCosPhiAlphaMinus2 * rCosPhi2;
            const Real rCosPsiAlpha = rCosPsiAlphaMinus2 * rCosPsi2;
            const Real Num = ( rCosPhiAlpha + rCosPsiAlpha );

            const Real E = Num * rMinusBeta;

            const Real factor = alphahalf * rMinusBeta;
            const Real F = factor * rCosPhiAlphaMinus2;
            const Real G = factor * rCosPsiAlphaMinus2;
            const Real H = - beta * rMinusBetaMinus2 * Num;
            
            Real dEdvx = static_cast<Real>(0);
            Real dEdvy = static_cast<Real>(0);
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                dEdv[i] = static_cast<Real>(2) * ( F * Pv[i] + G * Qv[i] ) + H * v[i];
                dEdvx += dEdv[i] * x[i];
                dEdvy += dEdv[i] * y[i];
                
                DX[1+i] -= b * dEdv[i];
                DY[1+i] += a * dEdv[i];
            }
            
            DX[0] += b * ( E - factor * rCosPhiAlpha + dEdvx );
            DY[0] += a * ( E - factor * rCosPsiAlpha - dEdvy );
            
            const Real bF = b * F;
            const Real aG = a * G;
            
            for( Int k = 0; k < ProjectorDim(); ++k )
            {
                DX[1+CoordDim()+k] += bF * V[k];
                DY[1+CoordDim()+k] += aG * V[k];
            }
            
            return a * E * b;
        }
        
    public:
        
        virtual std::string Stats() const override
        {
            return ClassName()+": alpha = "+ToString(alpha)+", beta = "+ToString(beta);
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<T1>::Get()+","+TypeName<T2>::Get()+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
    template<int AMB_DIM, int DEGREE, typename Real, typename Int>
    std::unique_ptr<BASE> CONCAT(Make_,CLASS)( const Real alpha, const Real beta )
    {
        Real alphahalf_intpart;
        bool alphahalf_is_int = (std::modf( alpha/static_cast<Real>(2), &alphahalf_intpart) == static_cast<Real>(0));

        Real betahalf_intpart;
        bool betahalf_is_int = (std::modf( beta/static_cast<Real>(2), &betahalf_intpart) == static_cast<Real>(0));

        BASE * r;

        if( alphahalf_is_int )
        {
            if( betahalf_is_int )
            {
                r = new CLASS<AMB_DIM,DEGREE,Int,Int,Real,Int>(
                        static_cast<Int>(alphahalf_intpart),static_cast<Int>(betahalf_intpart)
                    );
            }
            else
            {
                r = new CLASS<AMB_DIM,DEGREE,Int,Real,Real,Int>(
                        static_cast<Int>(alphahalf_intpart),beta/static_cast<Real>(2)
                    );
            }
        }
        else
        {
            if( betahalf_is_int )
            {
                r = new CLASS<AMB_DIM,DEGREE,Real,Int,Real,Int>(
                        alpha/static_cast<Real>(2),static_cast<Int>(betahalf_intpart)
                    );
            }
            else
            {
                r = new CLASS<AMB_DIM,DEGREE,Real,Real,Real,Int>(
                        alpha/static_cast<Real>(2),beta/static_cast<Real>(2)
                    );
            }
        }
        
        return std::unique_ptr<BASE>(r);
    }
    
} // namespace Repulsion

#undef BASE
#undef CLASS
