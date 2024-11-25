#pragma once

#define CLASS FarFieldKernelBase_FMM

// This sets up the basic I/O routines for all FMM-like far field kernels, both for energies and metrics.
namespace Repulsion
{
    template<int AMB_DIM, int DEGREE, typename Real, typename Int>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        
    public:
        
        CLASS()
        {
            Init();
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   S_far       ( other.S_far   )
        ,   S_D_far     ( other.S_D_far )
        ,   T_far       ( other.T_far   )
        ,   T_D_far     ( other.T_D_far )
        {
            Init();
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :   S_far       ( other.S_far   )
        ,   S_D_far     ( other.S_D_far )
        ,   T_far       ( other.T_far   )
        ,   T_D_far     ( other.T_D_far )
        {
            Init();
        }

        virtual ~CLASS() = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_BASE_CLASS(CLASS)
        
    public:
        
        void Init()
        {
            Int k = 0;
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                lin_k[i][i] = k;
                ++k;
                
                for( Int j = i+1; j < AMB_DIM; ++j )
                {
                    tri_i[k] = i;
                    tri_j[k] = j;
                    lin_k[i][j] = lin_k[j][i] = k;
                    ++k;
                }
            }
        }
        
    protected :
        
        const Real * restrict S_far   = nullptr;
              Real * restrict S_D_far = nullptr;
        
        const Real * restrict T_far   = nullptr;
              Real * restrict T_D_far = nullptr;
        
#ifdef FarField_FMM_S_Copy
        mutable Real a = 0.;
        mutable Real x [AMB_DIM] = {};
        mutable Real p [(AMB_DIM*(AMB_DIM+1))/2] = {};
#else
        mutable Real a = 0.;
        mutable Real const * restrict x = nullptr;
        mutable Real const * restrict p = nullptr;
#endif
        
#ifdef FarField_FMM_T_Copy
        mutable Real b = 0.;
        mutable Real y [AMB_DIM] = {};
        mutable Real q [(AMB_DIM*(AMB_DIM+1))/2] = {};
#else
        mutable Real b = 0.;
        mutable Real const * restrict y = nullptr;
        mutable Real const * restrict q = nullptr;
#endif
        
        mutable Real DX [1 + AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2] = {};
        mutable Real DY [1 + AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2] = {};

        Int tri_i [(AMB_DIM*(AMB_DIM+1))/2] = {};
        Int tri_j [(AMB_DIM*(AMB_DIM+1))/2] = {};
        Int lin_k [AMB_DIM][AMB_DIM]        = {};
        
        mutable Int S_ID = -1;
        mutable Int T_ID = -1;
        
    public :
        
        bool LoadFarField( const Tensor2<Real,Int> & S_far_, const Tensor2<Real,Int> & T_far_ )
        {
            bool check = true;
            if( (S_far_.Dimension(1) != FarDim()) || (T_far_.Dimension(1) != FarDim()) )
            {
                eprint(ClassName()+"::LoadFarField : (S_far_.Dimension(1) != FarDim()) || (T_far_.Dimension(1) != FarDim()).");
                check = false;
            }
            else
            {
                S_far = S_far_.data();
                T_far = T_far_.data();
            }
            return check;
        }

        
        bool LoadDFarField( Tensor2<Real,Int> & S_D_far_, Tensor2<Real,Int> & T_D_far_ )
        {
            bool check = true;
            if( (S_D_far_.Dimension(1) != FarDim()) || (T_D_far_.Dimension(1) != FarDim()) )
            {
                eprint(ClassName()+"::LoadFarField : (S_D_far_.Dimension(1) != FarDim()) || (T_D_far_.Dimension(1) != FarDim()).");
                check = false;
            }
            else
            {
                S_D_far = S_D_far_.data();
                T_D_far = T_D_far_.data();
            }
            return check;
        }
        
        constexpr Int AmbDim() const
        {
            return AMB_DIM;
        }
        
        constexpr Int CoordDim() const
        {
            return AMB_DIM;
        }
        
        constexpr Int ProjectorDim() const
        {
            return (AMB_DIM*(AMB_DIM+1))/2;
        }
        
        constexpr Int FarDim() const
        {
            return 1 + CoordDim() + ProjectorDim();
        }
        
        constexpr size_t CoordSize()
        {
            return CoordDim() * sizeof(Real);
        }
        
        constexpr size_t ProjectorSize()
        {
            return ProjectorDim() * sizeof(Real);
        }
        
        constexpr size_t FarSize()
        {
            return FarDim() * sizeof(Real);
        }
        
        virtual void LoadS( const Int i )
        {
                S_ID = i;
                const Real * const restrict X = S_far + FarDim() * S_ID;

                a = X[0];
#ifdef FarField_FMM_S_Copy
                std::memcpy( &x[0],  &X[1],                CoordSize() );
                std::memcpy( &p[0],  &X[1+CoordDim()], ProjectorSize() );
#else
                x = X + 1;
                p = X + 1 + CoordDim();
#endif
        }
        
        
        virtual void CleanseDBufferS()
        {
            std::memset( &DX[0],0, FarSize() );
        }
        
        virtual void WriteDBufferS() const
        {
            Real * restrict const to = S_D_far + FarDim() * S_ID;
            
            for( Int k = 0; k < FarDim(); ++k )
            {
                to[k] += DX[k];
            }
        }
        
        virtual void LoadT( const Int j )
        {
                T_ID = j;
                const Real * const restrict Y = T_far + FarDim() * T_ID;

                b = Y[0];
#ifdef FarField_FMM_T_Copy
                std::memcpy( &y[0],  &Y[1],                CoordSize() );
                std::memcpy( &q[0],  &Y[1+CoordDim()], ProjectorSize() );
#else
                y = Y + 1;
                q = Y + 1 + CoordDim();
#endif
        }

        virtual void CleanseDBufferT()
        {
            memset( &DY[0],0, FarDim() * sizeof(Real) );
        }
        
        virtual void WriteDBufferT() const
        {
            Real * restrict const to = T_D_far + FarDim() * T_ID;
            
            for( Int k = 0; k < FarDim(); ++k )
            {
                to[k] += DY[k];
            }
        }
        
    public:
        
        virtual std::string Stats() const
        {
            return this->ClassName();
        }
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef CLASS

