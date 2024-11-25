#pragma once

#define CLASS Energy_FarFieldKernel_BH

namespace Repulsion
{
    template<mint DOM_DIM, mint AMB_DIM, mint DEGREE, typename Real, typename Int>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS
    {
    public :
        
        using ClusterTree_T = ClusterTree<AMB_DIM,Real,Int,Real>;
        
        CLASS()
        {
            Init();
        }
        
        explicit CLASS( const Real theta_ ) : theta(theta_) {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :
            S_far(other.S_far),
            S_D_far(other.S_D_far),
            T_near(other.T_near),
            T_D_near(other.T_D_near),
            theta(other.theta)
        {
            Init();
        }

        // Move constructor 
        CLASS( CLASS && other ) noexcept 
        :
            S_far(other.S_far),
            S_D_far(other.S_D_far),
            T_near(other.T_near),
            T_D_near(other.T_D_near),
            theta(other.theta)
        {
            Init();
        }
        
        virtual ~CLASS() = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_BASE_CLASS(CLASS)
        
    public:
        
        void Init()
        {
            Int k = 0;
            
            SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                lin_k[i][i] = k;
                ++k;
                
                SMALL_UNROLL
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
        
        const Real * restrict T_near   = nullptr;
              Real * restrict T_D_near = nullptr;
        
        mutable Real * restrict DX = nullptr;
        mutable Real * restrict DY = nullptr;
        
        mutable Real a = static_cast<Real>(0);
        
#ifdef Energy_FarField_S_BH_Copy
        mutable Real x [AMB_DIM] = {};
#else
        mutable Real const * restrict x = nullptr;
#endif
        
        mutable Real b = static_cast<Real>(0);
        mutable Real y[AMB_DIM] = {};
        
#ifdef Energy_FarField_T_BH_Copy
        mutable Real q [(AMB_DIM*(AMB_DIM+1))/2] = {};
#else
        mutable Real const * restrict q = nullptr;
#endif
        
        Int tri_i [(AMB_DIM*(AMB_DIM+1))/2] = {};
        Int tri_j [(AMB_DIM*(AMB_DIM+1))/2] = {};
        Int lin_k [AMB_DIM][AMB_DIM]        = {};
        
        mutable Int S_ID = -1;
        mutable Int T_ID = -1;
        
        const Real theta = static_cast<Real>(0);
        
        static const constexpr Real T_scale = static_cast<Real>(1)/static_cast<Real>(DOM_DIM+1);
        
    public :
        
        virtual std::string Stats() const
        {
            return ClassName();
        }
        
        constexpr Int NearDim() const
        {
            return 1 + (DOM_DIM+1) * AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2;
        }
        
        constexpr Int FarDim() const
        {
            return 1 + AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2;
        }
        
        constexpr Int AmbDim() const
        {
            return AMB_DIM;
        }
        
        constexpr Int DomDim() const
        {
            return DOM_DIM;
        }
        
        Real GetTheta() const
        {
            return theta;
        }
        
        bool LoadNearfield( const Real * const T_P_near )
        {
            bool check = true;
            if( T_P_near == nullptr )
            {
                eprint(ClassName()+"::LoadNearfield : T_P_near == nullptr.");
                check = false;
            }
            else
            {
                T_near = T_P_near;
            }
            return check;
        }
        
        bool LoadNearfield( const Tensor2<Real,Int> & T_P_near )
        {
            bool check = true;
            if( T_P_near.Dimension(1) != NearDim() )
            {
                eprint(ClassName()+"::LoadNearfield : T_P_near.Dimension(1) != NearDim().");
                check = false;
            }
            else
            {
                check = LoadNearfield( T_P_near.data() );
            }
            return check;
        }

        bool LoadDNearfield( Real * T_P_D_near )
        {
            bool check = true;
            if( T_P_D_near == nullptr )
            {
                eprint(ClassName()+"::LoadDNearfield : T_P_D_near == nullptr.");
                check = false;
            }
            else
            {
                T_D_near = T_P_D_near;
            }
            return check;
        }
        
        bool LoadDNearfield( Tensor2<Real,Int> & T_P_D_near )
        {
            bool check = true;
            if( T_P_D_near.Dimension(1) != NearDim() )
            {
                eprint(ClassName()+"::LoadDNearfield : T_P_D_near.Dimension(1) != NearDim().");
                check = false;
            }
            else
            {
                LoadDNearfield( T_P_D_near.data() );
            }
            return check;
        }
        
        
        
        bool LoadFarfield( const Real * const S_P_far )
        {
            bool check = true;
            if( S_P_far == nullptr )
            {
                eprint(ClassName()+"::LoadFarfield : S_P_far == nullptr.");
                check = false;
            }
            else
            {
                S_far = S_P_far;
            }
            return check;
        }
        
        bool LoadFarfield( const Tensor2<Real,Int> & S_P_far )
        {
            bool check = true;
            if( S_P_far.Dimension(1) != FarDim() )
            {
                eprint(ClassName()+"::LoadFarfield : S_P_far.Dimension(1) != FarDim().");
                check = false;
            }
            else
            {
                check = LoadFarfield( S_P_far.data() );
            }
            return check;
        }

        bool LoadDFarfield( Real * S_P_D_far )
        {
            bool check = true;
            if( S_P_D_far == nullptr )
            {
                eprint(ClassName()+"::LoadDFarfield : S_P_D_far == nullptr.");
                check = false;
            }
            else
            {
                S_D_far = S_P_D_far;
            }
            return check;
        }
        
        bool LoadDFarfield( Tensor2<Real,Int> & S_P_D_far )
        {
            bool check = true;
            if( S_P_D_far.Dimension(1) != FarDim() )
            {
                eprint(ClassName()+"::LoadDFarfield : S_P_D_far.Dimension(1) != FarDim().");
                check = false;
            }
            else
            {
                LoadDFarfield( S_P_D_far.data() );
            }
            return check;
        }
        
        
    
        
#ifdef REPULSION_CLUSTERTREE
        
        virtual bool LoadClusterTrees(
            const ClusterTree_T & S,
            const ClusterTree_T & T,
            const Int thread
        )
        {
            bool check = true;
            
            check = check && LoadFarfield( S.ClusterFarFieldData() );
            
            check = check && LoadDFarfield( S.ThreadClusterDFarFieldData()[thread] );
            
            check = check && LoadNearfield( T.PrimitiveNearFieldData() );
            
            check = check && LoadDNearfield( T.ThreadPrimitiveDNearFieldData()[thread] );
            
            return check;
        }
#endif
        
        virtual void LoadS( const Int i )
        {
            S_ID = i;
            
            Real const * restrict ptr = S_far + FarDim() * i;
            
            a = ptr[0];
#ifdef Energy_FarField_S_BH_Copy
            memcpy( &x[0], ptr + 1, AMB_DIM * sizeof(Real) );
#else
            x = ptr + 1;
#endif
            DX = S_D_far + FarDim() * i;
        }

    
        virtual void LoadT( const Int j )
        {
            T_ID = j;
            
            Real const * restrict ptr = T_near + NearDim() * j;
            
            b = ptr[0];
#ifdef Energy_FarField_T_BH_Copy
            memcpy( &q[0], ptr+1+(DOM_DIM+1)*AMB_DIM,(AMB_DIM*(AMB_DIM+1))/2 * sizeof(Real) );
#else
            q = ptr + 1 + (DOM_DIM+1)*AMB_DIM;
#endif
            DY = T_D_near + NearDim() * j;
            
            Real const * restrict Y = ptr + 1;

            SMALL_UNROLL
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                y[k] = T_scale * Y[AMB_DIM * 0 + k];
                
                SMALL_UNROLL
                for( Int kk = 1; kk < DOM_DIM+1; ++kk )
                {
                    y[k] += T_scale *  Y[AMB_DIM * kk + k];
                }
            }
        }
        
        virtual Real Energy() = 0;
        
        virtual Real DEnergy() = 0;
    
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef CLASS
#undef BASE
