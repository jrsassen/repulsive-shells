#pragma once

#define CLASS AABB
#define BASE  BoundingVolumeBase<AMB_DIM,Real,Int,SReal>

namespace Collision {
    
    
    // serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type SReal by calling the member SetPointer.
    
    // DATA LAYOUT
    // serialized_data[0] = squared radius
    // serialized_data[1],...,serialized_data[AMB_DIM] = center
    // serialized_data[AMB_DIM + 1],...,serialized_data[AMB_DIM + AMB_DIM] = half the edge lengths.
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    protected:
        
        Real id_matrix[AMB_DIM][AMB_DIM];
    
        const int bits = static_cast<int>(64-1) / static_cast<int>(AMB_DIM);
        const SReal power  = static_cast<SReal>(0.9999) * std::pow( static_cast<SReal>(2), bits );
        const SReal offset = static_cast<SReal>(0.5) * power;
        
    protected:
        
        void Initialize()
        {
            for( Int k1 = 0; k1 < AMB_DIM; ++k1 )
            {
                for( Int k2 = 0; k2 < AMB_DIM; ++k2 )
                {
                    id_matrix[k1][k2] = static_cast<SReal>(k1==k2);
                }
            }
        }
    
    public:
        
        CLASS() : BASE()
        {
            Initialize();
        }

        // Copy constructor
        CLASS( const CLASS & other ) : BASE( other )
        {
            Initialize();
        }
        
        // Move constructor
        CLASS( CLASS && other ) noexcept  : BASE( other )
        {
            Initialize();
        }
        
        virtual ~CLASS() override = default;
        
        Int Size() const override
        {
            return 1 + AMB_DIM + AMB_DIM;
        }
        
    protected:

        mutable SReal self_buffer [1 + AMB_DIM + AMB_DIM];
 
    public:
        
#include "../Primitives/Primitive_BoilerPlate.h"
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
        
    public:
        
        // array p is suppose to represent a matrix of size N x AMB_DIM
        void FromPointCloud( const SReal * const coords_in, const Int N ) const override
        {
//            tic(ClassName()+"::FromPointCloud");
            SReal & r2 = serialized_data[0];
            
            // Abusing serialized_data temporily as working space.
                  SReal * restrict const box_min = serialized_data + 1;
                  SReal * restrict const box_max = serialized_data + 1 + AMB_DIM;
            
            const SReal * restrict const coords = coords_in;
            
            
//            valprint("Size()",Size());
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                box_min[k] = std::numeric_limits<SReal>::max();
                box_max[k] = std::numeric_limits<SReal>::lowest();
            }
            
            // No parallelization here because AABB objects are supposed to be created per thread anyways.
            // A desperate attempt to ask for simdization. The data layout of coords is wrong, so AVX will be useful only for AMB_DIM >=4. Nonetheless SSE instructs could be used for AMB_DIM = 2 and AMB_DIM = 3...
            
            
            for( Int i = 0; i < N; ++i )
            {
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    const SReal x = static_cast<SReal>(coords[ AMB_DIM * i + k ]);
                    box_min[k] = std::min( box_min[k], x );
                    box_max[k] = std::max( box_max[k], x );
                }
            }
            
            r2 = static_cast<SReal>(0);

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                const SReal diff = static_cast<SReal>(0.5) * (box_max[k] - box_min[k]);
                r2 += diff * diff;
                
                // adding half the edge length to obtain the k-th coordinate of the center
                box_min[k] += diff;
                // storing half the edge length in the designated storage.
                box_max[k]  = diff;
            }
//            toc(ClassName()+"::FromPointCloud");
        }
        

        // array p is supposed to represent a matrix of size N x AMB_DIM
        void FromPrimitives(
            PolytopeBase<AMB_DIM,Real,Int,SReal> & P,  // primitive prototype
            SReal * const P_serialized,                // serialized data of primitives
            const Int begin,                           // which _P_rimitives are in question
            const Int end,                             // which _P_rimitives are in question
            Int thread_count = 1                       // how many threads to utilize
        ) const
        {
//            ptic(ClassName()+"::FromPrimitives (PolytopeBase)");

            SReal & r2 = serialized_data[0];

            // Abusing serialized_data temporarily as working space.
            SReal * restrict const box_min = serialized_data + 1;
            SReal * restrict const box_max = serialized_data + 1 + AMB_DIM;
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                box_min[k] = std::numeric_limits<SReal>::max();
                box_max[k] = std::numeric_limits<SReal>::lowest();
            }
            // No parallelization here because AABB objects are supposed to be created per thread anyways.
            // A desperate attempt to ask for simdization. The data layout of coords is wrong, so AVX will be useful only for AMB_DIM >=4. Nonetheless SSE instructs could be used for AMB_DIM = 2 and AMB_DIM = 3...

            for( Int i = begin; i < end; ++i )
            {
                P.SetPointer( P_serialized, i );
                P.BoxMinMax( box_min, box_max );
            }

            r2 = static_cast<SReal>(0);

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                const SReal diff = static_cast<SReal>(0.5) * (box_max[k] - box_min[k]);
                r2 += diff * diff;

                // adding half the edge length to obtain the k-th coordinate of the center
                box_min[k] += diff;
                // storing half the edge length in the designated storage.
                box_max[k]  = diff;
            }
//            ptoc(ClassName()+"::FromPrimitives (PolytopeBase)");
        }
        
        // array p is supposed to represent a matrix of size N x AMB_DIM
        virtual void FromPrimitives(
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,      // primitive prototype
            SReal * const P_serialized,                  // serialized data of primitives
            const Int begin,                           // which _P_rimitives are in question
            const Int end,                             // which _P_rimitives are in question
            Int thread_count = 1                       // how many threads to utilize
        ) const override
        {
//            ptic(ClassName()+"::FromPrimitives (PrimitiveSerialized)");

            SReal & r2 = serialized_data[0];
            
            // Abusing serialized_data temporily as working space.
                  SReal * restrict const box_min = serialized_data + 1;
                  SReal * restrict const box_max = serialized_data + 1 + AMB_DIM;
            
//            valprint("Size()",Size());
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                box_min[k] = std::numeric_limits<SReal>::max();
                box_max[k] = std::numeric_limits<SReal>::lowest();
            }
            
            // TODO: Check whether reversing this loop would be helpful.
            // TODO: Can this loop be parallelized?
            for( Int i = begin; i < end; ++i )
            {
                P.SetPointer( P_serialized, i );
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    Real min_val;
                    Real max_val;
                    
                    P.MinMaxSupportValue( &id_matrix[j][0], min_val, max_val );
                    box_min[j] = std::min( box_min[j], static_cast<SReal>(min_val) );
                    box_max[j] = std::max( box_max[j], static_cast<SReal>(max_val) );
                }
            }

            r2 = static_cast<SReal>(0);

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                const SReal diff = static_cast<SReal>(0.5) * (box_max[k] - box_min[k]);
                r2 += diff * diff;
                
                // adding half the edge length to obtain the k-th coordinate of the center
                box_min[k] += diff;
                // storing half the edge length in the designated storage.
                box_max[k]  = diff;
            }
            
//            ptoc(ClassName()+"::FromPrimitives (PrimitiveSerialized)");
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
            const SReal * restrict const x = serialized_data + 1;
            const SReal * restrict const L = serialized_data + 1 + AMB_DIM;
            const  Real * restrict const v = dir;
                   Real * restrict const s = supp;
            
            Real R1;
            Real R2 = static_cast<Real>(0);
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                Real x_k = static_cast<Real>(x[k]);
                Real L_k = static_cast<Real>(L[k]);
                
                R1   = v[k] * L_k;
                R2  += v[k] * x_k + std::abs(R1);
                s[k] = x_k + MyMath::sign(R1) * L_k;
            }

            return R2;
        }


        //Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
            const SReal * restrict const x = serialized_data + 1;
            const SReal * restrict const L = serialized_data + 1 + AMB_DIM;
            const  Real * restrict const v = dir;
                   Real * restrict const s = supp;
            
            Real R1;
            Real R2 = static_cast<Real>(0);
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                Real x_k = static_cast<Real>(x[k]);
                Real L_k = static_cast<Real>(L[k]);
                
                R1   = v[k] * L_k;
                R2  += v[k] * x_k - std::abs(R1);
                s[k] = x_k - MyMath::sign(R1) * L_k;
            }

            return R2;
        }
                
        
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
            // Could be implemented more efficiently, but since this routine is unlikely to be used...
            min_val = MinSupportVector( dir, &this->Real_buffer[0] );
            max_val = MaxSupportVector( dir, &this->Real_buffer[AMB_DIM] );
        }
        
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
        
        inline friend Real AABB_SquaredDistance( const CLASS & P, const CLASS & Q )
        {
            const SReal * restrict const P_x = P.serialized_data+1;              // center of box P
            const SReal * restrict const P_L = P.serialized_data+1+AMB_DIM;      // edge half-lengths of box P
            
            const SReal * restrict const Q_x = Q.serialized_data+1;              // center of box Q
            const SReal * restrict const Q_L = Q.serialized_data+1+AMB_DIM;      // edge half-lengths of box Q
            
            Real d2 = static_cast<Real>(0);
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                Real x = static_cast<Real>(
                    std::max(
                        static_cast<SReal>(0),
                        std::max( P_x[k]-P_L[k], Q_x[k]-Q_L[k] )
                        -
                        std::min( P_x[k]+P_L[k], Q_x[k]+Q_L[k] )
                    )
                );
                d2 += x * x;
            }
            return d2;
        }
        
        void Merge( SReal * C_Serialized, const Int i = 0 ) const
        {
            SReal * p = C_Serialized + Size() * i;
            
            if( serialized_data != p )
            {
                
                SReal * restrict const x1 = serialized_data + 1;
                SReal * restrict const L1 = serialized_data + 1 + AMB_DIM;
                
                SReal * restrict const x2 = p + 1;
                SReal * restrict const L2 = p + 1 + AMB_DIM;
                
                SReal r2 = static_cast<SReal>(0);
                
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    const SReal box_min = std::min( x1[k] - L1[k], x2[k] - L2[k] );
                    const SReal box_max = std::max( x1[k] + L1[k], x2[k] + L2[k] );
                    
                    x1[k] = static_cast<SReal>(0.5) * ( box_max + box_min );
                    L1[k] = static_cast<SReal>(0.5) * ( box_max - box_min );
                    
                    r2 += L1[k] * L1[k];
                }
                serialized_data[0] = r2;
            }
        }
        
        virtual void MortonCodes(
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,
            SReal * const P_serialized,
            const Int begin,
            const Int end,
            uint64_t * const z
        ) const
        {
            
            SReal * restrict const c = serialized_data + 1;
            SReal * restrict const L = serialized_data + 1 + AMB_DIM;
            
            const Int P_Size = P.Size();
            
            SReal     scale    [AMB_DIM];
            uint32_t rescaled [AMB_DIM];
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                scale[k] = power * L[k];
            }
            
            
            for( Int i = begin; i < end; ++i )
            {
                const SReal * restrict const x = P_serialized + 1 + P_Size * i;
                
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    rescaled[k] = static_cast<uint32_t>( ( x[k] - c[k] ) * scale[k] + offset );
                }
                
                z[i] = InterleaveBits<AMB_DIM>( rescaled );
            }
        }
        

        
    }; // CLASS

} // namespace Collision

#undef CLASS
#undef BASE
    

