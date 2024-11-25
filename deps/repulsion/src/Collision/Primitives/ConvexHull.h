#pragma once

#define CLASS ConvexHull
#define BASE  PrimitiveBase<AMB_DIM,Real,Int>


//TODO: Test this thoroughly!

namespace Collision {
    
    template<int HULL_COUNT, int AMB_DIM, typename Real, typename Int>
    class CLASS : public BASE
    {
    protected:
        
        const BASE * primitive [HULL_COUNT];
        
        mutable Real buffer [AMB_DIM * (AMB_DIM+1)];
        mutable Real squared_radius = static_cast<Real>(-1);
        
    public:
        
        CLASS() : BASE()
        {
            for( Int i = 0; i < HULL_COUNT; ++i )
            {
                primitive[i] = nullptr;
            }
        }
        
        virtual ~CLASS() override = default;

        void Set( const Int i, const BASE * const  P )
        {
            if( 0 <= i < HULL_COUNT )
            {
                primitive[i] = P;
            }
        }
        
        bool IsNull( const Int i ) const
        {
            return ( primitive[i] == nullptr );
        }
        
        // Computes some point within the primitive and writes it to p.
        virtual void InteriorPoint( Real * const point ) const override
        {
                  Real * restrict const p = point;
            
            Int count = static_cast<Int>(0);
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                p[k] = static_cast<Real>(1);
            }
            
            for( Int i = 0; i < HULL_COUNT; ++i )
            {
                if( primitive[i] )
                {
                    ++count;
                    primitive[i]->InteriorPoint( &this->buffer[0] );
                    
                    for( Int k = 0; k < AMB_DIM; ++k )
                    {
                        p[k] += this->buffer[k];
                    }
                }
            }
            
            count = std::max( static_cast<Int>(1), count );
            
            Real scale = static_cast<Real>(1) / static_cast<Real>(count);
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                p[k] *= scale;
            }
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
            Real * restrict const s = supp;
            Real * restrict const b = &this->buffer[0];
            Real * restrict const b_max = &this->buffer[AMB_DIM];
            Real maximum = std::numeric_limits<Real>::lowest();
            Real value;
            
            for( Int i = 0; i < HULL_COUNT; ++i )
            {
                if( primitive[i] )
                {
                    value = primitive[i]->MaxSupportVector(dir, b);
                    
                    if( value > maximum )
                    {
                        maximum = value;
                        std::swap( b, b_max );
                    }
                }
            }
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                s[k] = b_max[k];
            }

            return maximum;
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
            Real * restrict const s = supp;
            Real * restrict const b = &this->buffer[0];
            Real * restrict const b_min = &this->buffer[AMB_DIM];
            Real minimum = std::numeric_limits<Real>::max();
            Real value;
            
            for( Int i = 0; i < HULL_COUNT; ++i )
            {
                if( primitive[i] )
                {
                    value = primitive[i]->MinSupportVector(dir, b);
                    
                    if( value < minimum )
                    {
                        minimum = value;
                        std::swap( b, b_min );
                    }
                }
            }
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                s[k] = b_min[k];
            }

            return minimum;
        }
        
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
            min_val = MinSupportVector( dir, &this->buffer[0] );
            max_val = MaxSupportVector( dir, &this->buffer[AMB_DIM] );
        }
        
        virtual Real SquaredRadius() const override
        {
            // Computes the sum of the squared axis lengths which equals the sum of squared singular values which equals the square of the Frobenius norm.
            
            if( squared_radius >= static_cast<Real>(0) )
            {
                return squared_radius;
            }
            
            squared_radius = static_cast<Real>(0);
            
            for( Int i = 0; i < HULL_COUNT; ++i )
            {
                if( primitive[i] )
                {
                    squared_radius = std::max( squared_radius, primitive[i]->SquaredRadius() );
                }
            }
            
            return squared_radius;
        }
        
        
        // Copy constructor
        ConvexHull( const ConvexHull & B )
        {
            for( Int i = 0; i < HULL_COUNT; ++ i)
            {
                primitive[i] = B.primitive[i];
            }
            squared_radius = B.squared_radius;
        }
        
        // Move constructor
        ConvexHull( ConvexHull && B ) noexcept
        {
            for( Int i = 0; i < HULL_COUNT; ++ i )
            {
                primitive[i] = B.primitive[i];
                B.primitive[i] = nullptr;
            }
            squared_radius = std::move(B.squared_radius);
        }
        
//        friend void swap(ConvexHull &A, ConvexHull &B) noexcept
//        {
//            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//            using std::swap;
//            
//            for( Int i = 0; i < HULL_COUNT; ++ i)
//            {
//                swap( A.primitive[i], B.primitive[i] );
//            }
//            swap( A.squared_radius, B.squared_radius );
//        }
//
//        // copy-and-swap idiom
//        ConvexHull & operator=(ConvexHull B)
//        {
//            // see https://stackoverflow.com/a/3279550/8248900 for details
//
//            swap(*this, B);
//            return *this;
//
//        }
        
    private:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(HULL_COUNT)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    }; // ConvexHull
    
} // namespe Collision

#undef CLASS
#undef BASE
