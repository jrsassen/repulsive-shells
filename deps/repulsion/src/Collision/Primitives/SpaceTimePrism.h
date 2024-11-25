#pragma once

#define CLASS SpaceTimePrism
#define BASE  PrimitiveBase<AMB_DIM,Real,Int>
#define PROTO_T PrimitiveSerialized<AMB_DIM+1,Real,Int,SReal>

namespace Collision {
    
    // Adds some common I/O interface for all Polytope types to PrimitiveSerialized.

    template<int AMB_DIM,typename Real,typename Int,typename SReal>
    class CLASS : public BASE
    {
    protected:
    

        std::unique_ptr<PROTO_T> proto_0;
        std::unique_ptr<PROTO_T> proto_1;
        
        SReal t_0 = 0;
        SReal t_1 = 1;
    public:
        
        CLASS() {}
        
        explicit CLASS( const PROTO_T & proto )
        :   proto_0(proto->Clone())
        ,   proto_1(proto->Clone())
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE()
        ,   proto_0(other.proto_0->Clone())
        ,   proto_1(other.proto_1->Clone())
        ,   t_0(other.t_0)
        ,   t_1(other.t_1)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :   BASE()
        ,   proto_0(std::move(other.proto_0->Clone()))
        ,   proto_1(std::move(other.proto_1->Clone()))
        ,   t_0(other.t_0)
        ,   t_1(other.t_1)
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
        
        void SetFirstPointer( SReal * const data, const Int i )
        {
            proto_0->SetPointer( data, i );
        }
        
        void SetFirstTime( const SReal t )
        {
            t_0 = t;
        }
        
        void SetSecondPointer( SReal * const data, const Int i )
        {
            proto_1->SetPointer( data, i );
        }
        
        void SetSecondTime( const SReal t )
        {
            t_1 = t;
        }
        
        // Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
            Real supp_0 [AMB_DIM];
            Real supp_1 [AMB_DIM];
            
            Real h_0 = proto_0->MaxSupportVector(dir, supp);
            h_0 += dir[AMB_DIM] * t_0;
            Real h_1 = proto_0->MaxSupportVector(dir, supp);
            h_1 += dir[AMB_DIM] * t_1;
            
            if( h_0 >= h_1 )
            {
                GJK_SMALL_UNROLL
                for( Int i = 0; i < AMB_DIM; ++i )
                {
                    supp[i] = supp_0[i];
                }
                supp[AMB_DIM] = t_0;
                return h_0;
            }
            {
                GJK_SMALL_UNROLL
                for( Int i = 0; i < AMB_DIM; ++i )
                {
                    supp[i] = supp_1[i];
                }
                supp[AMB_DIM] = t_1;
                return h_1;
            }
        }
        
        // Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
            Real supp_0 [AMB_DIM];
            Real supp_1 [AMB_DIM];
            
            Real h_0 = dir[AMB_DIM] * t_0;
            h_0 += proto_0->MinSupportVector(dir, supp);
            
            Real h_1 = dir[AMB_DIM] * t_1;
            h_1 += proto_0->MinSupportVector(dir, supp);

            
            if( h_0 <= h_1 )
            {
                GJK_SMALL_UNROLL
                for( Int i = 0; i < AMB_DIM; ++i )
                {
                    supp[i] = supp_0[i];
                }
                supp[AMB_DIM] = t_0;
                return h_0;
            }
            {
                GJK_SMALL_UNROLL
                for( Int i = 0; i < AMB_DIM; ++i )
                {
                    supp[i] = supp_1[i];
                }
                supp[AMB_DIM] = t_1;
                return h_1;
            }
        }
        
        // Computes only the values of min/max support function. Usefull to compute bounding boxes.
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
            Real min_0;
            Real max_0;
            Real min_1;
            Real max_1;
            
            proto_0->MinMaxSupportValue( dir, min_0, max_0);
            proto_1->MinMaxSupportValue( dir, min_1, max_1);
            
            min_0 += t_0 * dir[AMB_DIM];
            max_0 += t_0 * dir[AMB_DIM];
            
            min_0 += t_1 * dir[AMB_DIM];
            max_0 += t_1 * dir[AMB_DIM];
            
            min_val = std::min(min_0,min_1);
            max_val = std::max(max_0,max_1);
        }
        
        // Returns some point within the primitive and writes it to p.
        virtual void InteriorPoint( Real * const p ) const override
        {
            Real p_0 [AMB_DIM];
            Real p_1 [AMB_DIM];
            proto_0->InteriorPoint( &p_0[0] );
            proto_1->InteriorPoint( &p_1[0] );
            
            GJK_SMALL_UNROLL
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                p[i] = static_cast<Real>(0.5) * ( p_0[i] + p_1[i] );
            }
            p[AMB_DIM] = static_cast<Real>(0.5) * ( t_0 + t_1 );
        }
        
        // Returns some (upper bound of the) squared radius of the primitive as measured from the result of InteriorPoint.
        virtual Real InteriorPoint( const Int k ) const override
        {
            if( k < AMB_DIM )
            {
                return static_cast<Real>(0.5) * ( proto_0->InteriorPoint(k) + proto_1->InteriorPoint(k) );
            }
            else
            {
                return static_cast<Real>(0.5) * ( t_0 + t_1 );
            }
        }
        
        // Returns some (upper bound of the) squared radius of the primitive as measured from the result of InteriorPoint.
        virtual Real SquaredRadius() const override
        {
            Real r_0 = std::sqrt(proto_0->SquaredRadius());
            Real r_1 = std::sqrt(proto_1->SquaredRadius());
            Real r   = static_cast<Real>(0.25) * std::abs( t_1 - t_0 ) + std::max(r_0,r_1);
            return r * r;
        }

        virtual std::string DataString() const override
        {
            std::stringstream s;
            
            s << ClassName() << ": ";
            s << " t_0 = " << t_0;
            s << " proto_0 = " << proto_0->DataString();
            s << " t_1 = " << t_1;
            s << " proto_1 = " << proto_1->DataString();
            return s.str();
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    };

    
} // namespace Collision

#undef PROTO_T
#undef CLASS
#undef BASE
