#pragma once

#define CLASS TrivialEnergy_NearFieldKernel_Adaptive
#define BASE  Energy_NearFieldKernel_Adaptive<DOM_DIM1,DOM_DIM2,AMB_DIM,Real,Int,SReal>

namespace Repulsion
{
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::S_near;
        using BASE::S_D_near;
        
        using BASE::T_near;
        using BASE::T_D_near;

        using BASE::DX;
        using BASE::DY;
        
        using BASE::a;
        using BASE::x_buffer;
        using BASE::p;
        
        using BASE::b;
        using BASE::y_buffer;
        using BASE::q;

        using BASE::settings;
        using BASE::S;
        using BASE::T;
        
        using BASE::tri_i;
        using BASE::tri_j;
        using BASE::lin_k;
        
        using BASE::lambda;
        using BASE::mu;
        
    public:
        
        using BASE::CoordDimS;
        using BASE::CoordDimT;
        using BASE::ProjectorDim;
        
        explicit CLASS(
               const AdaptivitySettings & settings_ = AdaptivityDefaultSettings
        )
        :
            BASE( settings_ )
        {}

        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE(other)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept
        :   BASE(other)
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
    protected:
        
    public:

        virtual Real energy() const override
        {
            this->simplex_file
                << "{ "
                << S.CornerString()
                << ", "
                << T.CornerString()
                << " }" << std::endl;

            this->center_file
                << "{ "
                << S.CenterString()
                << ", "
                << T.CenterString()
                << " }"
                << std::endl;
            
            this->emb_simplex_file
                << "{ "
                << S.EmbeddedSimplexString()
                << ", "
                << T.EmbeddedSimplexString()
                << " }" << std::endl;

            
            Real x [AMB_DIM] = {};
            Real y [AMB_DIM] = {};
            
            SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                x[i] = lambda[0] * x_buffer[AMB_DIM*0 +i];
                y[i] = mu    [0] * y_buffer[AMB_DIM*0 +i];

                SMALL_UNROLL
                for( Int ii = 1; ii < DOM_DIM1+1; ++ii )
                {
                    x[i] += lambda[ii] * x_buffer[AMB_DIM*ii +i];
                }
                
                SMALL_UNROLL
                for( Int ii = 1; ii < DOM_DIM2+1; ++ii )
                {
                    y[i] += mu    [ii] * y_buffer[AMB_DIM*ii +i];
                }
            }
            
            this->emb_center_file << "{ ";
        
            this->emb_center_file << "{ ";
            this->emb_center_file << x[0];
            for( Int i = 1; i < AMB_DIM; ++i )
            {
                this->emb_center_file << ", " << x[i];
            }
            this->emb_center_file << " }";
            
            this->emb_center_file << ", ";
            
            this->emb_center_file << "{ ";
            this->emb_center_file << y[0];
            for( Int i = 1; i < AMB_DIM; ++i )
            {
                this->emb_center_file << ", " << y[i];
            }
            this->emb_center_file << " }";
            
            this->emb_center_file << " }";
            
            this->emb_center_file << std::endl;
            
            return static_cast<Real>(0);
        }
    
        
        virtual Real denergy() const override
        {
            return static_cast<Real>(0);
        }
        
    public:
        
        virtual bool IsRepulsive() const override
        {
            return false;
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM1)+","+"<"+ToString(DOM_DIM2)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef BASE
#undef CLASS
