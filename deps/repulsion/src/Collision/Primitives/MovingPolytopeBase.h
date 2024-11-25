#pragma once

#define CLASS MovingPolytopeBase
#define BASE  PrimitiveBase<AMB_DIM+1,Real,Int>

namespace Collision {

    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
        ASSERT_FLOAT(SReal);
        
    protected:
    
        mutable SReal a = static_cast<SReal>(0);
        mutable SReal b = static_cast<SReal>(1);
        
        mutable SReal T = static_cast<SReal>(1);
        
    public:
        
        CLASS() {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE()
        ,   a(other.a)
        ,   b(other.b)
        ,   T(other.T)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :   BASE()
        ,   a(other.a)
        ,   b(other.b)
        ,   T(other.T)
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
        
        virtual Int PointCount() const = 0;
        
        virtual Int Size() const = 0;
        
        virtual Int CoordinateSize() const = 0;
        
        virtual Int VelocitySize() const  = 0;
        
        virtual void SetFirstTime( const SReal a_ ) const
        {
            a = a_;
        }
        
        virtual void SetSecondTime( const SReal b_ ) const
        {
            b = b_;
        }
        
        virtual void SetTimeScale( const SReal T_ ) const
        {
            T = T_;
        }
        
        virtual SReal TimeScale( const SReal T_ ) const
        {
            return T;
        }
        
        virtual void WriteCoordinatesSerialized(       SReal * const p_ser, const Int i = 0 ) const =0;
        virtual void ReadCoordinatesSerialized ( const SReal * const p_ser, const Int i = 0 ) = 0;
        
        virtual void WriteVelocitiesSerialized (       SReal * const v_ser, const Int i = 0 ) const =0;
        virtual void ReadVelocitiesSerialized  ( const SReal * const v_ser, const Int i = 0 ) = 0;
        
        virtual void WriteDeformedSerialized   (       SReal * const p_ser, const SReal t, const Int i = 0 ) const = 0;
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    };

    
} // Collision

#undef CLASS
#undef BASE
