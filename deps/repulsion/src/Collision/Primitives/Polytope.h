#pragma once


#define CLASS Polytope
#define BASE  PolytopeExt<AMB_DIM,Real,Int,SReal,ExtReal,Int>

namespace Collision {
    
    template<int POINT_COUNT,int AMB_DIM,typename Real,typename Int,typename SReal,
                typename ExtReal = SReal,typename ExtInt = Int>
    class CLASS : public BASE
    {
        ASSERT_FLOAT (ExtReal );
        ASSERT_INT   (ExtInt  );
        
        
        // this->serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type Real by calling the member SetPointer.
        
        // DATA LAYOUT
        // serialized_data[0] = squared radius
        // serialized_data[1],...,serialized_data[AMB_DIM] = interior_point
        // serialized_data[AMB_DIM + 1],...,serialized_data[AMB_DIM + POINT_COUNT x AMB_DIM] = points whose convex hull defines the polytope.
        
    public:
        
        CLASS() : BASE() {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE( other ) {}

        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE( other ) {}

        virtual ~CLASS() override = default;
        
        Int Size() const override
        {
            return 1 + AMB_DIM + POINT_COUNT * AMB_DIM;
        }
        
        Int PointCount() const override
        {
            return POINT_COUNT;
        }

    protected:

        mutable SReal self_buffer [1 + AMB_DIM + POINT_COUNT * AMB_DIM];
        
#include "Primitive_BoilerPlate.h"
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
    public:
        
        virtual void FromCoordinates( const ExtReal * const hull_coords_, const Int i = 0 ) const override
        {
            constexpr SReal w = static_cast<SReal>(1)/static_cast<SReal>(POINT_COUNT);
            
            SReal & r2 = this->serialized_data[0];
            
            const ExtReal * restrict const A = hull_coords_ + i * POINT_COUNT * AMB_DIM;
                  SReal   * restrict const center = this->serialized_data + 1;
                  SReal   * restrict const hull_coords = this->serialized_data + 1 + AMB_DIM;

            
            // Copy the hull coordinates and compute average.
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                SReal x = static_cast<SReal>(A[ AMB_DIM * 0 + k ]);
                hull_coords[ AMB_DIM * 0 + k ] = x;
                center[k] = x;

                for( Int j = 1; j < POINT_COUNT; ++j )
                {
                    x = static_cast<SReal>(A[ AMB_DIM * j + k]);
                    hull_coords[ AMB_DIM * j + k ] = x;
                    center[k] += x;
                }
                center[k] *= w;
            }
            
            // Compute radius.
            r2 = static_cast<SReal>(0);

            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                SReal diff = A[ AMB_DIM * j] - center[0];
                SReal square = diff * diff;

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    diff = A[ AMB_DIM * j + k] - center[k];
                    square += diff * diff;
                }
                r2 = std::max( r2, square );
            }
        }
        
        virtual void FromIndexList( const ExtReal * const coords_, const ExtInt * const tuples, const Int i = 0 ) const override
        {
            const ExtInt * restrict const s = tuples + POINT_COUNT * i;
            
            constexpr SReal w = static_cast<SReal>(1)/static_cast<SReal>(POINT_COUNT);
            
            SReal & r2 = this->serialized_data[0];
            SReal * restrict const center = this->serialized_data + 1;
            SReal * restrict const hull_coords = this->serialized_data + 1 + AMB_DIM;
            
            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                std::transform(
                    coords_ + AMB_DIM * s[j],
                    coords_ + AMB_DIM * (s[j]+1),
                    hull_coords + AMB_DIM * j,
                    static_caster<ExtReal,SReal>()
                );
            }
            
            // Compute average.
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                center[k] = hull_coords[ AMB_DIM * 0 + k ];

                for( Int j = 1; j < POINT_COUNT; ++j )
                {
                    center[k] += hull_coords[ AMB_DIM * j + k ];
                }
                
                center[k] *= w;
            }
            
            // Compute radius.
            r2 = static_cast<Real>(0);

            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                SReal diff = hull_coords[ AMB_DIM * j] - center[0];
                SReal square = diff * diff;
                
                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    diff = hull_coords[ AMB_DIM * j + k] - center[k];
                    square += diff * diff;
                }
                r2 = std::max( r2, square );
            }
        }
        
        
        
        //Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
//            ptic(ClassName()+"::MinSupportVector");
            // apply dot product with direction to all the points that span the convex hull
            const SReal * restrict const A = this->serialized_data + 1 + AMB_DIM;
            const  Real * restrict const v = dir;
                   Real * restrict const s = supp;

            Real value = static_cast<Real>(A[0]) * v[0];

            for( Int k = 1; k < AMB_DIM; ++k )
            {
                value += static_cast<Real>(A[k]) * v[k];
            }

            Int pos = 0;
            Real minimum = value;

            for( Int j = 1; j < POINT_COUNT; ++j )
            {
                const SReal * restrict const w = &A[ AMB_DIM * j ];
                value = static_cast<Real>(w[0]) * v[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    value += static_cast<Real>(w[k]) * v[k];
                }

                if( value < minimum )
                {
                    pos = j;
                    minimum = value;
                }
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                s[k] = static_cast<Real>(A[ AMB_DIM * pos + k ]);
            }

//            ptoc(ClassName()+"::MinSupportVector");
            
            return minimum;
        }
        
        //Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
//            ptic(ClassName()+"::MaxSupportVector");
            
            // apply dot product with direction to all the points that span the convex hull
            const SReal * restrict const A = this->serialized_data + 1 + AMB_DIM;
            const  Real * restrict const v = dir;
                   Real * restrict const s = supp;

            Real value = static_cast<Real>(A[0]) * v[0];

            for( Int k = 1; k < AMB_DIM; ++k )
            {
                value += static_cast<Real>(A[k]) * v[k];
            }

            Int pos = 0;
            Real maximum = value;

            for( Int j = 1; j < POINT_COUNT; ++j )
            {
                const SReal * restrict const w = &A[ AMB_DIM * j ];
                value = static_cast<Real>(w[0]) * v[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    value += static_cast<Real>(w[k]) * v[k];
                }

                if( value > maximum )
                {
                    pos = j;
                    maximum = value;
                }
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                s[k] = static_cast<Real>(A[ AMB_DIM * pos + k ]);
            }

//            ptoc(ClassName()+"::MaxSupportVector");
            
            return maximum;
        }

        // Computes only the values of min/max support function. Usefull to compute bounding boxes.
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
//            ptic(ClassName()+"::MinMaxSupportValue");
            
            // apply dot product with direction to all the points that span the convex hull
            const SReal * restrict const A = this->serialized_data + 1 + AMB_DIM;
            const  Real * restrict const v = dir;

            Real value = static_cast<Real>(A[0]) * v[0];

            for( Int k = 1; k < AMB_DIM; ++k )
            {
                value += static_cast<Real>(A[k]) * v[k];
            }

            min_val = value;
            max_val = value;
            
            for( Int j = 1; j < POINT_COUNT; ++j )
            {
                value = static_cast<Real>(A[ AMB_DIM * j ]) * v[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    value += static_cast<Real>(A[ AMB_DIM * j + k]) * v[k];
                }

                min_val = std::min( min_val, value );
                max_val = std::max( max_val, value );
            }
            
//            ptoc(ClassName()+"::MinMaxSupportValue");
        }
        
        
        // Helper function to compute axis-aligned bounding boxes. in the format of box_min, box_max vector.
        // box_min, box_max are supposed to be vectors of size AMB_DIM.
        // BoxMinMax computes the "lower left" lo and "upper right" hi vectors of the primitives bounding box and sets box_min = min(lo, box_min) and box_max = min(h, box_max)
        virtual void BoxMinMax( SReal * const box_min_, SReal * const box_max_ ) const override
        {
//            ptic(ClassName()+"::BoxMinMax");
            
            const SReal * restrict const p = this->serialized_data + 1 + AMB_DIM;
                  SReal * restrict const box_min = box_min_;
                  SReal * restrict const box_max = box_max_;
            
            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    SReal x = p[ AMB_DIM * j + k ];
                    box_min[k] = std::min( box_min[k], x );
                    box_max[k] = std::max( box_max[k], x );
                }
            }
            
//            ptoc(ClassName()+"::BoxMinMax");
        }
        
        void PrintStats() const
        {
            print(ClassName()+"::PrintStats():");
            
            std::stringstream s;
            
            s << "serialized_data = { " << this->serialized_data[0];
            
            for( Int i = 1; i < Size(); ++ i )
            {
                s << ", " << this->serialized_data[i];
            }
        
            s <<" }";
            
            print(s.str());
        }
        
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(POINT_COUNT)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+","+TypeName<ExtInt>::Get()+">";
        }

    }; // Polytope
    
    template <int AMB_DIM, typename Real, typename Int, typename SReal,
        typename ExtReal = SReal, typename ExtInt = Int>
    std::unique_ptr<BASE> MakePolytope( const Int P_size )
    {
        BASE * r;
        switch(  P_size  )
        {
            case 1:
            {
                r = new Polytope<1,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 2:
            {
                r = new Polytope<2,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 3:
            {
                r = new Polytope<3,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 4:
            {
                r = new Polytope<4,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 5:
            {
                r = new Polytope<5,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 6:
            {
                r = new Polytope<6,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 7:
            {
                r = new Polytope<7,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 8:
            {
                r = new Polytope<8,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 9:
            {
                r = new Polytope<9,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 10:
            {
                r = new Polytope<10,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 11:
            {
                r = new Polytope<11,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 12:
            {
                r = new Polytope<12,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 13:
            {
                r = new Polytope<13,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 14:
            {
                r = new Polytope<14,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 15:
            {
                r = new Polytope<15,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 16:
            {
                r = new Polytope<16,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 17:
            {
                r = new Polytope<17,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 18:
            {
                r = new Polytope<18,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 19:
            {
                r = new Polytope<19,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            case 20:
            {
                r = new Polytope<20,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>();
                break;
            }
            default:
            {
                eprint("MakePolytope: Number of vertices of polytope = " + ToString( P_size ) + " is not in the range from 1 to 20. Returning nullptr.");
                return nullptr;
            }
        }
        
        return std::unique_ptr<BASE>(r);
        
    } // MakePolytope
    
} // namespe Collision

#undef CLASS
#undef BASE
