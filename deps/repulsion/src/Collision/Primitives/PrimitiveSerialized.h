#pragma once

#define CLASS PrimitiveSerialized
#define BASE  PrimitiveBase<AMB_DIM,Real,Int>

namespace Collision {
    
    // Adds some I/O routines to PrimitiveBase.

    // Real  -  data type that will be handed to GJK; GJK typically needs doubles.
    // SReal -  storage data type; could be float ("short real")
    // Int   -  integer type for return values and loops.
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal >
    class CLASS : public BASE
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );

    protected:
        
        // serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type Real by calling the member SetPointer.
        
        // DATA LAYOUT
        // serialized_data[0] = squared radius
        // serialized_data[1],...,serialized_data[AMB_DIM] = interior_point
        // serialized_data[1+ AMB_DIM],...,serialized_data[Size()] = data that defines the primitive.
        
        
        mutable SReal SReal_buffer [AMB_DIM * (AMB_DIM+1)];
        mutable  Real  Real_buffer [AMB_DIM * (AMB_DIM+1)];
        
        SReal * serialized_data = nullptr;
        
    public:
        
        CLASS() {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        {
            this->serialized_data = other.serialized_data;
        }

        // Move constructor
        CLASS( PrimitiveSerialized && other ) noexcept 
        {
            this->serialized_data = std::move(other.serialized_data);
        }
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
        
        virtual Int Size() const = 0;
        
        // Sets the classe's data pointer.
        // We assume that p_ is an array of sufficient size in which the primitive's data is found between
        //      begin = p_ + Size() * pos
        // and
        //      end   = p_ + Size() * (pos+1).
        virtual void SetPointer( SReal * const p_, const Int pos ) = 0;
        
        virtual void SetPointer( SReal * const p_ ) = 0;
        
        virtual void PrintPointer() const
        {
            print(ClassName()+"::PrintPointer: this->serialized_data = " + ToString(this->serialized_data) );
        }
        
        // Returns some point within the primitive and writes it to p.
        void InteriorPoint( Real * const point_out ) const override
        {
//            print(ClassName()+"::InteriorPoint");
            if( this->serialized_data )
            {
                       Real * restrict const point = point_out;
                const SReal * restrict const p = this->serialized_data + 1;
                
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    point[k] = static_cast<Real>(p[k]);
                }
            }
            else
            {
                print("this->serialized_data = " + ToString(this->serialized_data) );
                valprint("this->serialized_data[1]",this->serialized_data[1]);
                valprint("this->serialized_data[2]",this->serialized_data[2]);
                valprint("this->serialized_data[3]",this->serialized_data[3]);
                eprint(ClassName()+"::InteriorPoint: serialized_data is set to nullptr. Doing nothing.");
            }
        }
        
        virtual Real InteriorPoint( const Int k ) const override
        {
            if( this->serialized_data )
            {
                return static_cast<Real>(this->serialized_data[1+k]);
            }
            else
            {
//                valprint("this->serialized_data",this->serialized_data);
//                valprint("this->serialized_data[1+k]",this->serialized_data[1+k]);
                eprint(ClassName()+"::InteriorPoint: serialized_data is set to nullptr. Return 0.");
                return static_cast<Real>(0);
            }
        }
        
            
        // Returns some (upper bound of the) squared radius of the primitive as measured from the result of InteriorPoint.
        Real SquaredRadius() const override
        {
            return  this->serialized_data ? this->serialized_data[0] : std::numeric_limits<Real>::max();
        }
     

//        virtual void Copy( const Real * const p_in, Real * const q_out ) const = 0;
//
//        virtual void Copy( const Real * const p_in , const Int i, Real * const q_out, const Int j ) const = 0;

        
        virtual void Read( const SReal * const p_in ) const = 0;
        
        virtual void Read( const SReal * const p_in, const Int i ) const = 0;
        
        virtual void Write(      SReal * const q_out ) const = 0;
        
        virtual void Write(      SReal * const q_out, const Int j ) const = 0;
        
        virtual void Swap( SReal * const p_out, SReal * const q_out ) const = 0;
        
        virtual void Swap( SReal * const p_out, const Int i, SReal * const q_out, const Int j ) const = 0;
        
    public:



//
//        friend void swap(PrimitiveSerialized &A, PrimitiveSerialized &B) noexcept
//        {
//            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//            using std::swap;
//
//            swap( A.serialized_data, B.serialized_data );
//        }

//        // copy-and-swap idiom
//        virtual PrimitiveSerialized & operator=(PrimitiveSerialized other)
//        {
//            // see https://stackoverflow.com/a/3279550/8248900 for details
//
//            swap(*this, other);
//            return *this;
//
//        }
        
    
        
        
//        friend std::string to_string( const PrimitiveSerialized & P, Int prec = 16)
//        {
//            std::ostringstream out;
//            out.precision(prec);
//            
//            out << " {";
//            out << P.serialized_data[0];
//            for( Int i = 1; i < P.Size() ; ++i )
//            {
//                out << "," << P.serialized_data[i];
//            }
//            out << " }";
//            return out.str();
//        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Real>::Get()+","+ToString(AMB_DIM)+">";
        }
        
    }; // PrimitiveBase

    
} // Collision

#undef BASE
#undef CLASS
