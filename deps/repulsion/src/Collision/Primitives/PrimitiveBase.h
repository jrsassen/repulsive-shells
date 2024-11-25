#pragma once

namespace Collision {
    
#define CLASS PrimitiveBase
    
    // Real  -  data type that will be handed to GJK; GJK typically needs doubles.
    // Int   -  integer type for return values and loops.
    
    template<int AMB_DIM, typename Real, typename Int>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS     // Use this broad alignment to prevent false sharing.
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        
    protected:

    public:
        
        CLASS() {}
        
        virtual ~CLASS() = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_BASE_CLASS(CLASS)
        
    public:
        
        static Int AmbDim()
        {
            return AMB_DIM;
        }
        
        // Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const = 0;
        
        // Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const = 0;
        
        // Computes only the values of min/max support function. Usefull to compute bounding boxes.
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const = 0;
        
        // Returns some point within the primitive and writes it to p.
        virtual void InteriorPoint( Real * const p ) const = 0;
    
        virtual Real InteriorPoint( const Int k ) const = 0;
        
        // Returns some (upper bound of the) squared radius of the primitive as measured from the result of InteriorPoint.
        virtual Real SquaredRadius() const = 0;

        virtual std::string DataString() const = 0;
        
//        // Copy constructor
//        PrimitiveBase( const PrimitiveBase & B )
//        {}
//        
//        // Move constructor
//        PrimitiveBase( PrimitiveBase && B ) noexcept 
//        {}
//        
//        friend void swap(PrimitiveBase &A, PrimitiveBase &B) noexcept
//        {
//            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//            using std::swap;
//            
//            swap( A.serialized_data, B.serialized_data );
//        }
//        
//        // copy-and-swap idiom
//        PrimitiveBase &operator=(PrimitiveBase B)
//        {
//            // see https://stackoverflow.com/a/3279550/8248900 for details
//
//            swap(*this, B);
//            return *this;
//            
//        }
        
        virtual std::string ClassName() const
        {
            return "PrimitiveBase<"+TypeName<Real>::Get()+","+ToString(AMB_DIM)+">";
        }
        
    }; // PrimitiveBase
    
} // Collision

#undef CLASS
