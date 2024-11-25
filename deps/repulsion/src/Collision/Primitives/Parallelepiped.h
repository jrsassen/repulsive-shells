#pragma once

#define CLASS Parallelepiped
#define BASE  PrimitiveSerialized<AMB_DIM,Real,Int,SReal>

// TODO: Test MinMaxSupportValue

namespace Collision {
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    protected:
        
        // The Parallelepiped is the image of Cuboid[ {-1,...,-1}, {1,...,1} ] under the affine mapping x \mapsto transform * x+center.
        
        
        // serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type Real by calling the member SetPointer.
        
        // DATA LAYOUT
        // serialized_data[0] = squared radius
        // serialized_data[1],...,serialized_data[AMB_DIM] = center
        // serialized_data[AMB_DIM+1],...,serialized_data[AMB_DIM+POINT_COUNT x AMB_DIM] = transform.
        
    
    public:
        
        CLASS() : BASE() {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE( other ) {}

        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE( other ) {}

        virtual ~CLASS() override = default;
        
        constexpr Int Size() const override
        {
            return 1+AMB_DIM+AMB_DIM*AMB_DIM;
        }
        
    private:

        mutable SReal self_buffer [1+AMB_DIM+AMB_DIM*AMB_DIM];
        
    public:
        
#include "Primitive_BoilerPlate.h"
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
        void FromTransform( const SReal * const center, const SReal * const transform ) const
        {
            SReal & r2 = this->serialized_data[0];
            
                  SReal * restrict const x = this->serialized_data+1;
                  SReal * restrict const A = this->serialized_data+1+AMB_DIM;
            
            const SReal * restrict const __center     = center;
            const SReal * restrict const __transform  = transform;
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                x[k] = __center[k];
            }
            
            for( Int k = 0; k < AMB_DIM*AMB_DIM; ++k )
            {
                A[k] = __transform[k];
            }
            
            // Computes the sum of the squared half-axis lengths which equals the sum of squared singular values which equals the square of the Frobenius norm.
            
            r2 = static_cast<SReal>(0);

            for( Int i = 0; i < AMB_DIM; ++i )
            {
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    r2 += A[AMB_DIM*i+j]*A[AMB_DIM*i+j];
                }
            }
            // CAUTION: Up to this point, this only really the squared radius if the frame has orthogonal columns. Otherwise, r2 might be too small.
            
            
            // Adding a safety term that vanishes when frame has orthogonal column. This is to guarantee that r2 is indeed greater or equal to the actual squared radius.
            // This is superflous if the Parallelepiped is supposed to represent an OBB.
            SReal dots = static_cast<SReal>(0);

            for( Int i = 0; i < AMB_DIM; ++i )
            {
                for( Int j = i+1; j < AMB_DIM; ++j )
                {
                    SReal dot = A[i]*A[j];

                    for( Int k = 1; k < AMB_DIM; ++k )
                    {
                        dot += A[AMB_DIM*k+i]*A[AMB_DIM*k+j];
                    }
                    
                    dots += std::abs( dot );
                }

            }
            r2 += static_cast<SReal>(2)*dots;
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
            const SReal * restrict const x = this->serialized_data+1;
            const SReal * restrict const A = this->serialized_data+1+AMB_DIM;
            const  Real * restrict const v = dir;
                   Real * restrict const s = supp;

            Real R1;
            Real R2;
            Real R3 = static_cast<Real>(0);
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                s[i] = static_cast<Real>(x[i]);
            }
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                R1 = static_cast<Real>(A[i])*v[0];
                
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    R1 += v[j]*static_cast<Real>(A[AMB_DIM*j+i]);
                }
                
                R2 = (R1 >= static_cast<Real>(0)) ? static_cast<Real>(1) : static_cast<Real>(-1);
                
                R3 += static_cast<Real>(x[i])*v[i]+R1*R2;
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    s[j] += static_cast<Real>(A[AMB_DIM*j+i])*R2;
                }
            }

            return R3;
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
            const SReal * restrict const x = this->serialized_data+1;
            const SReal * restrict const A = this->serialized_data+1+AMB_DIM;
            const  Real * restrict const v = dir;
                   Real * restrict const s = supp;

            Real R1;
            Real R2;
            Real R3 = static_cast<Real>(0);
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                s[i] = static_cast<Real>(x[i]);
            }
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                R1 = static_cast<Real>(A[i])*v[0];
                
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    R1 += v[j]*static_cast<Real>(A[AMB_DIM*j+i]);
                }
                
                R2 = (R1 >= static_cast<Real>(0)) ? static_cast<Real>(-1) : static_cast<Real>(1);
                
                R3 += static_cast<Real>(x[i])*v[i]+R1*R2;
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    s[j] += static_cast<Real>(A[AMB_DIM*j+i])*R2;
                }
            }

            return R3;
        }
        
        // Computes only the values of min/max support function. Usefull to compute bounding boxes.
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
            const Real * restrict const x = this->serialized_data+1;
            const Real * restrict const A = this->serialized_data+1+AMB_DIM;
            const Real * restrict const v = dir;

            Real R1;
            Real R2;
            min_val = static_cast<Real>(0);
            max_val = static_cast<Real>(0);
            
            for( Int i = 0; i < AMB_DIM; ++i )
            {
                R1 = static_cast<Real>(A[i])*v[0];
                
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    R1 += v[j]*static_cast<Real>(A[AMB_DIM*j+i]);
                }
                
                R2 = (R1 >= static_cast<Real>(0)) ? static_cast<Real>(1) : static_cast<Real>(-1);
                Real x_i = static_cast<Real>(x[i]);
                min_val += x_i*v[i]-R1*R2;
                max_val += x_i*v[i]+R1*R2;
            }
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    }; // CLASS
    
} // namespe Collision

#undef CLASS
#undef BASE
