#pragma once

#define CLASS NearFieldKernelBase

// This sets up the basic I/O routines for all near field kernels, both for energies and metrics.

namespace Repulsion
{
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, typename Real, typename Int, typename SReal>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        
    public:
        
        using S_TREE_T = SimplexHierarchy<DOM_DIM1,AMB_DIM,GJK_Real,Int,SReal>;
        using T_TREE_T = SimplexHierarchy<DOM_DIM2,AMB_DIM,GJK_Real,Int,SReal>;
        
        using GJK_T    = GJK_Algorithm<AMB_DIM,GJK_Real,Int>;
        
        CLASS()
        {
            Init();
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   S_near        ( other.S_near )
        ,   S_D_near      ( other.S_D_near )
        ,   S_serialized  ( other.S_serialized  )
        ,   T_near        ( other.T_near )
        ,   T_D_near      ( other.T_D_near )
        ,   T_serialized  ( other.T_serialized  )
        {
            Init();
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :   S_serialized  ( other.S_serialized  )
        ,   S_near        ( other.S_near )
        ,   S_D_near      ( other.S_D_near )
        ,   T_near        ( other.T_near )
        ,   T_D_near      ( other.T_D_near )
        ,   T_serialized  ( other.T_serialized  )
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

    protected:
        
        const  Real * restrict S_near       = nullptr;
               Real * restrict S_D_near     = nullptr;
        const SReal * restrict S_serialized = nullptr;

        
        const  Real * restrict T_near       = nullptr;
               Real * restrict T_D_near     = nullptr;
        const SReal * restrict T_serialized = nullptr;
        
        mutable Real a = static_cast<Real>(0);
        mutable Real b = static_cast<Real>(0);
        
        
#ifdef NearField_S_Copy
        mutable Real x [AMB_DIM] = {};
        mutable Real p [(AMB_DIM*(AMB_DIM+1))/2] = {};
        mutable Real x_buffer [(DOM_DIM1+1)*AMB_DIM] = {};
#else
        mutable Real x [AMB_DIM] = {};
        mutable Real const * restrict p = nullptr;
        mutable Real const * restrict x_buffer = nullptr;
#endif
        
#ifdef NearField_T_Copy
        mutable Real y [AMB_DIM] = {};
        mutable Real q [(AMB_DIM*(AMB_DIM+1))/2] = {};
        mutable Real y_buffer [(DOM_DIM2+1)*AMB_DIM] = {};
#else
        mutable Real y [AMB_DIM] = {};
        mutable Real const * restrict q = nullptr;
        mutable Real const * restrict y_buffer = nullptr;
#endif
        
        mutable Real DX [1 + (DOM_DIM1+1)*AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2] = {};
        mutable Real DY [1 + (DOM_DIM2+1)*AMB_DIM + (AMB_DIM*(AMB_DIM+1))/2] = {};
        
        Int tri_i [(AMB_DIM*(AMB_DIM+1))/2] = {};
        Int tri_j [(AMB_DIM*(AMB_DIM+1))/2] = {};
        Int lin_k [AMB_DIM][AMB_DIM]        = {};
        
        mutable Int S_ID = -1;
        mutable Int T_ID = -1;
        
        static const constexpr Real S_scale = static_cast<Real>(1)/static_cast<Real>(DOM_DIM1+1);
        static const constexpr Real T_scale = static_cast<Real>(1)/static_cast<Real>(DOM_DIM2+1);

        mutable S_TREE_T S;
        mutable T_TREE_T T;
        
        const SReal * restrict const lambda = S.Center();
        const SReal * restrict const mu     = T.Center();
        
        mutable GJK_T gjk;
        
    public:
        
        bool LoadNearField(
            const Tensor2<Real,Int> & S_near_,
            const Tensor2<Real,Int> & T_near_
        )
        {
            bool check = true;
            if( (S_near_.Dimension(1) != NearDimS()) || (T_near_.Dimension(1) != NearDimT()) )
            {
                eprint(ClassName()+"::LoadNearField : (S_near_.Dimension(1) != NearDimS()) || (T_near_.Dimension(1) != NearDimT()).");
                check = false;
            }
            else
            {
                S_near = S_near_.data();
                T_near = T_near_.data();
            }
            return check;
        }

        
        bool LoadDNearField(
            Tensor2<Real,Int> & S_D_near_,
            Tensor2<Real,Int> & T_D_near_
        )
        {
            bool check = true;
            if( (S_D_near_.Dimension(1) != NearDimS()) || (T_D_near_.Dimension(1) != NearDimT()) )
            {
                eprint(ClassName()+"::LoadDNearField : (S_D_near_.Dimension(1) != NearDimS()) || (T_D_near_.Dimension(1) != NearDimT()).");
                check = false;
            }
            else
            {
                S_D_near = S_D_near_.data();
                T_D_near = T_D_near_.data();
            }
            return check;
        }
        
        bool LoadPrimitiveSerializedData(
            const Tensor2<SReal,Int> & S_serialized_,
            const Tensor2<SReal,Int> & T_serialized_
        )
        {
            bool check = true;
            
            if( (S_serialized_.Dimension(1) != S.SimplexPrototype().Size()) || (T_serialized_.Dimension(1) != T.SimplexPrototype().Size()) )
            {
                check = false;
                eprint(ClassName()+"::LoadPrimitiveSerializedData.");
            }
            else
            {
                S_serialized = S_serialized_.data();
                T_serialized = T_serialized_.data();
            }
            
            return check;
        }
        
        constexpr Int AmbDim() const
        {
            return AMB_DIM;
        }
        
        constexpr Int DomDimS() const
        {
            return DOM_DIM1;
        }
        
        constexpr Int DomDimT() const
        {
            return DOM_DIM2;
        }
        
        constexpr Int CoordDimS() const
        {
            return (DOM_DIM1+1)*AMB_DIM;
        }
        
        constexpr Int CoordDimT() const
        {
            return (DOM_DIM2+1)*AMB_DIM;
        }
        
        constexpr Int ProjectorDim() const
        {
            return (AMB_DIM*(AMB_DIM+1))/2;
        }
        
        constexpr Int NearDimS() const
        {
            return 1 + CoordDimS() + ProjectorDim();
        }
        
        constexpr Int NearDimT() const
        {
            return 1 + CoordDimT() + ProjectorDim();
        }
        
        constexpr size_t CoordSizeS()
        {
            return CoordDimS() * sizeof(Real);
        }
        
        constexpr size_t CoordSizeT()
        {
            return CoordDimT() * sizeof(Real);
        }
        
        constexpr size_t ProjectorSize()
        {
            return ProjectorDim() * sizeof(Real);
        }
        
        constexpr size_t NearSizeS()
        {
            return NearDimS() * sizeof(Real);
        }
        
        constexpr size_t NearSizeT()
        {
            return NearDimT() * sizeof(Real);
        }
        
        virtual void LoadS( const Int i )
        {
            S_ID = i;
            const Real * const restrict X = S_near + NearDimS() * S_ID;
            
            a = X[0];
            
#ifdef NearField_S_Copy
            std::memcpy( &p[0], &X[1+CoordDimS()], ProjectorSize() );
#else
            p = &X[1+CoordDimS()];
#endif
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                x[k] = S_scale * X[1 + AMB_DIM * 0 + k];
                
                for( Int kk = 1; kk < DOM_DIM1+1; ++kk )
                {
                    x[k] += S_scale * X[1 + AMB_DIM * kk + k];
                }
            }
        }
        
        virtual void LoadT( const Int j )
        {
            T_ID = j;
            const Real * const restrict Y = T_near + NearDimT() * T_ID;
            
            b = Y[0];
            
#ifdef NearField_T_Copy
            std::memcpy( &q[0], &Y[1+CoordDimT()], ProjectorSize() );
#else
            q = &Y[1+CoordDimT()];
#endif
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                y[k] = T_scale * Y[1 + AMB_DIM * 0 + k];

                for( Int kk = 1; kk < DOM_DIM2+1; ++kk )
                {
                    y[k] += T_scale * Y[1 + AMB_DIM * kk + k];
                }
            }
        }
        
        virtual void CleanseDBufferS()
        {
            std::memset( &DX[0], 0, NearSizeS() );
        }
        
        virtual void WriteDBufferS()
        {
            Real * restrict const to = S_D_near + NearDimS() * S_ID;

            for( Int k = 0; k < NearDimS(); ++k )
            {
                to[k] += DX[k];
            }
        }
        
        virtual void CleanseDBufferT()
        {
            std::memset( &DY[0], 0, NearSizeT() );
        }
        
        virtual void WriteDBufferT()
        {
            Real * restrict const to = T_D_near + NearDimT() * T_ID;

            for( Int k = 0; k < NearDimT(); ++k )
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
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM1)+","+ToString(DOM_DIM2)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef CLASS


