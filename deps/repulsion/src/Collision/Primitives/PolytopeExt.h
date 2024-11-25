#pragma once

#define CLASS PolytopeExt
#define BASE  PolytopeBase<AMB_DIM,Real,Int,SReal>

namespace Collision {
    
    // Pure interface class

    template<int AMB_DIM, typename Real, typename Int,typename SReal, typename ExtReal, typename ExtInt>
    class CLASS : public BASE
    {        
        ASSERT_FLOAT (ExtReal);
        ASSERT_INT   (ExtInt );
        
    protected:
        
        // serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type Real by calling the member SetPointer.
        
        // DATA LAYOUT
        // serialized_data[0] = squared radius
        // serialized_data[1],...,serialized_data[AMB_DIM] = interior_point
        // serialized_data[1+ AMB_DIM],...,serialized_data[Size()] = data that defines the primitive.
        
    public:
        
        CLASS() : BASE () {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE( other)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE( other)
        {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
        
        virtual void FromCoordinates( const ExtReal * const hull_coords_, const Int i = 0 ) const = 0;
        
        virtual void FromIndexList( const ExtReal * const coords_, const ExtInt * const tuples, const Int i = 0 ) const = 0;
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    }; // PrimitiveBase

    
} // Collision

#undef CLASS
#undef BASE
