#pragma once

#define BASE  OBB<AMB_DIM,Real,Int,SReal>
#define CLASS OBB_MedianSplit

namespace Collision {
    
    // The OBB is the image of Cuboid[ {-L[0],...,-L[AMB_DIM-1]}, {L[0],...,L[AMB_DIM-1]} ] under the ORTHOGONAL mapping x \mapsto rotation * x + center.
    
    
    // serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type Real by calling the member SetPointer.
    
    // DATA LAYOUT
    // serialized_data[0] = squared radius
    // serialized_data[1],...,serialized_data[AMB_DIM] = center
    // serialized_data[1+AMB_DIM],...,serialized_data[AMB_DIM+AMB_DIM] = L (vector of half the edge lengths)
    // serialized_data[1+AMB_DIM+AMB_DIM],...,serialized_data[AMB_DIM + AMB_DIM + AMB_DIM x AMB_DIM] = rotation^T. BEWARE THE TRANSPOSITION!!!!!!!!!!!!!
    
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    public:
        
        CLASS() : BASE() {}

        // Copy constructor
        CLASS( const CLASS & other ) : BASE( other ) {}
        
        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE( other ) {}

        virtual ~CLASS() override = default;
        
        constexpr Int Size() const override
        {
            return 1 + AMB_DIM + AMB_DIM + AMB_DIM * AMB_DIM;
        }
        
    protected:

        using BASE::self_buffer;
 
    public:

#include "../Primitives/Primitive_BoilerPlate.h"
        
        
        // Array coords_in is supposes to represent a matrix of size n x AMB_DIM
        void FromPointCloud( const SReal * const coords_in, const Int n ) const override
        {
            if( n == 0 )
            {
                eprint(ClassName()+"::FromPointCloud : 0 members in point could.");
                return;
            }
            
            SReal n_inv = static_cast<SReal>(1)/static_cast<SReal>(n);
            
            // Zero bounding volume's data.
            for( Int k = 0; k < Size(); ++k )
            {
                serialized_data[k] = static_cast<Real>(0);
            }
            
            // Abusing serialized_data temporily as working space.
            SReal & r2 = serialized_data[0];
            SReal * restrict const average    = serialized_data + 1;
            SReal * restrict const covariance = serialized_data + 1 + AMB_DIM + AMB_DIM;
            
            // Compute average of the points
            for( Int i = 0; i < n; ++i )
            {
                const SReal * restrict const p = coords_in + AMB_DIM * i;

                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    average[k] += p[k];
                }
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                average[k] *= n_inv;
            }
            
            // Compute covariance matrix.
            for( Int i = 0; i < n; ++i )
            {
                const SReal * restrict const p = coords_in + AMB_DIM * i;

                for( Int k1 = 0; k1 < AMB_DIM; ++k1 )
                {
                    SReal delta = (p[k1] - average[k1]);

                    for( Int k2 = k1; k2 < AMB_DIM; ++k2 )
                    {
                        covariance[AMB_DIM * k1 + k2] += delta * (p[k2] - average[k2]);
                    }
                }
            }

            for( Int k1 = 0; k1 < AMB_DIM; ++k1 )
            {
                for( Int k2 = k1; k2 < AMB_DIM; ++k2 )
                {
                    covariance[AMB_DIM * k1 + k2] *= n_inv;
                }
            }
            
            // Abusing serialized_data temporily as working space.
                  SReal * restrict const box_min = serialized_data + 1;
                  SReal * restrict const box_max = serialized_data + 1 + AMB_DIM;
            
            (void)SymmetricEigenSolve( covariance, box_min );
            
            // Now "covariance" stores the eigenbasis.

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                box_min[k] = std::numeric_limits<SReal>::max();
                box_max[k] = std::numeric_limits<SReal>::lowest();
            }

            // TODO: Check whether reversing this loop would be helpful.
            // TODO: Can this loop be parallelized?
            for( Int i = 0; i < n; ++i )
            {
                const SReal * restrict const p = coords_in + AMB_DIM * i;

                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    const SReal * restrict const vec = covariance + AMB_DIM * j;

                    SReal x = p[0] * vec[0];

                    for( Int k = 1; k < AMB_DIM; ++k )
                    {
                        x += p[k] * vec[k];
                    }

                    box_min[j] = std::min( box_min[j], x );
                    box_max[j] = std::max( box_max[j], x );
                }
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                SReal diff = static_cast<SReal>(0.5) * (box_max[k] - box_min[k]);
                r2 += diff * diff;

                // adding half the edge length to obtain the k-th coordinate of the center
                this->SReal_buffer[k] = box_min[k] + diff;
        
                // storing half the edge length in the designated storage.
                box_max[k]  = diff;
            }
            
            SReal * restrict const center    = serialized_data + 1;
            SReal * restrict const rotationT = serialized_data + 1 + AMB_DIM + AMB_DIM;
            
            for( Int j = 0; j < AMB_DIM; ++j )
            {
                center[j] = rotationT[j] * this->SReal_buffer[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    center[j] += this->SReal_buffer[k] * rotationT[AMB_DIM * k + j];
                }
            }
        } // FromPointCloud

        
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
            if( begin >= end )
            {
                eprint(ClassName()+"FromPrimitives : begin = "+ToString(begin)+" >= "+ToString(end)+" = end");
                return;
            }
            
            Int P_Size = P.Size();
            SReal n_inv = static_cast<SReal>(1)/static_cast<SReal>(end-begin);
            
            // Zero bounding volume's data.
            for( Int k = 0; k < Size(); ++k )
            {
                serialized_data[k] = static_cast<SReal>(0);
            }
            
            // Abusing serialized_data temporily as working space.
            SReal & r2 = serialized_data[0];
            SReal * restrict const average    = serialized_data + 1;
            SReal * restrict const covariance = serialized_data + 1 + AMB_DIM + AMB_DIM;
            
            // Compute average of the InterPoints of all primitives.
            for( Int i = begin; i < end; ++i )
            {
                const SReal * restrict const p = P_serialized + 1 + P_Size * i;

                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    average[k] += p[k];
                }
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                average[k] *= n_inv;
            }
            
            // Compute covariance matrix.
            for( Int i = begin; i < end; ++i )
            {
                const SReal * restrict const p = P_serialized + 1 + P_Size * i;

                for( Int k1 = 0; k1 < AMB_DIM; ++k1 )
                {
                    const SReal delta = (p[k1] - average[k1]);

                    for( Int k2 = k1; k2 < AMB_DIM; ++k2 )
                    {
                        covariance[AMB_DIM * k1 + k2] += delta * (p[k2] - average[k2]);
                    }
                }
            }

            for( Int k1 = 0; k1 < AMB_DIM; ++k1 )
            {
                for( Int k2 = k1; k2 < AMB_DIM; ++k2 )
                {
                    covariance[AMB_DIM * k1 + k2] *= n_inv;
                }
            }
            
            // Abusing serialized_data temporily as working space.
                  SReal * restrict const box_min = serialized_data + 1;
                  SReal * restrict const box_max = serialized_data + 1 + AMB_DIM;
        
            (void)SymmetricEigenSolve( covariance, box_min );
            
            // Now "covariance" stores the eigenbasis.
            
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
                    SReal min_val;
                    SReal max_val;
                    P.MinMaxSupportValue( covariance + AMB_DIM * j, min_val, max_val );
                    box_min[j] = std::min( box_min[j], min_val );
                    box_max[j] = std::max( box_max[j], max_val );
                }
            }
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                SReal diff = 0.5 * (box_max[k] - box_min[k]);
                r2 += diff * diff;

                // adding half the edge length to obtain the k-th coordinate of the center (within the transformed coordinates)
                this->SReal_buffer[k] = box_min[k] + diff;
        
                // storing half the edge length in the designated storage.
                box_max[k]  = diff;
            }
            
            // Rotate this->SReal_buffer so that it falls onto the true center of the bounding box.
            
            SReal * restrict const center    = serialized_data + 1;
            SReal * restrict const rotationT = serialized_data + 1 + AMB_DIM + AMB_DIM;
            
            for( Int j = 0; j < AMB_DIM; ++j )
            {
                center[j] = rotationT[j] * this->SReal_buffer[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    center[j] += this->SReal_buffer[k] * rotationT[AMB_DIM * k + j];
                }
            }
//            ptoc(ClassName()+"::FromPrimitives (PrimitiveSerialized)");
        }
        
        virtual Int Split(
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,                          // primitive prototype; to be "mapped" over P_serialized, thus not const.
            SReal * const P_serialized, const Int begin, const Int end,    // which _P_rimitives are in question
            Int   * const P_ordering,                                        // to keep track of the permutation of the primitives
            SReal * const C_serialized, const Int C_ID,                     // where to get   the bounding volume info for _C_urrent bounding volume
            SReal * const L_serialized, const Int L_ID,                     // where to store the bounding volume info for _L_eft  child (if successful!)
            SReal * const R_serialized, const Int R_ID,                     // where to store the bounding volume info for _R_ight child (if successful!)
            SReal *       score,                                             // some scratch buffer for one scalar per primitive
            Int   *       perm,                                              // some scratch buffer for one Int per primitive (for storing local permutation)
            Int   *       inv_perm,                                          // some scratch buffer for one Int per primitive (for storing inverse of local permutation)
            Int thread_count = 1                                           // how many threads to utilize
        ) override
        {
//            ptic(ClassName()+"::Split");
//            valprint("begin",begin);
//            valprint("end",end);
            
            Int P_Size = P.Size();
            
            this->SetPointer( C_serialized, C_ID );
            
            const SReal * restrict const center    = serialized_data + 1;
            const SReal * restrict const L         = serialized_data + 1 + AMB_DIM;
            const SReal * restrict const rotationT = serialized_data + 1 + AMB_DIM + AMB_DIM;
            
            // Find the longest axis `split_dir` of primitives's bounding box.
            Int split_dir = 0;
            SReal L_max = L[0];
            
            for( Int k = 1; k < AMB_DIM; ++k )
            {
                if( L[k] > L_max )
                {
                    L_max = L[k];
                    split_dir = k;
                }
            }
            
            if( L_max <= static_cast<SReal>(0) )
            {
                eprint(ClassName()+"::Split: longest axis has length <=0.");
                return -1;
            }
            
            // Finding the "median". Adapted from https://stackoverflow.com/a/16798127/8248900 (using pointers instead of iterators)
            
            // Computing score as pojection of the primitives' InteriorPoints on the longest axis.
            // Fill perm with the indices.
            for( Int i = begin; i < end; ++i )
            {
                const SReal * restrict const p = P_serialized + 1 + P_Size * i;

                SReal x = rotationT[ AMB_DIM * split_dir ] * p[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    x += rotationT[ AMB_DIM * split_dir + k ] * p[k];
                }
                score[i] = x;
                perm[i]  = i;
            }
        
            Int split_index = begin + ((end-begin)/2);

            Int  * mid = perm + split_index;
            std::nth_element( perm + begin, mid, perm + end, [score](const Int i, const Int j) {return score[i] < score[j];} );
            // Now perm contains the desired ordering of score.

            // Invert permutation.
            for( Int i = begin; i < end; ++i )
            {
                inv_perm[perm[i]] = i;
            }

            // https://www.geeksforgeeks.org/permute-the-elements-of-an-array-following-given-order/
            // Reorder primitive according to perm, i.e., write primitive perm[i] to position i.
            for( Int i = begin; i < end; ++i )
            {
                Int next = i;

                while( inv_perm[next] >= 0 )
                {
                    Int temp = inv_perm[next];
                    std::swap( P_ordering   [i]  , P_ordering   [temp] );
                    P.Swap   ( P_serialized, i,    P_serialized, temp  );
                    inv_perm[next] = -1;
                    next = temp;
                }
            }
            
            // Prevent further computations with fewer than two primitives.
            if( ( begin+1 < split_index ) && ( split_index < end-1 ) )
            {
                // Compute bounding volume of left child.
                SetPointer( L_serialized, L_ID );
                FromPrimitives( P, P_serialized,   begin, split_index,   thread_count );
                // Compute bounding volume of right child.
                SetPointer( R_serialized, R_ID );
                FromPrimitives( P, P_serialized,   split_index, end,   thread_count );
            }
            else
            {
                split_index = -1;
            }
            // ... otherwise we assume that the bounding volume hierarchy / cluster tree won't do the split.
//            ptoc(ClassName()+"::Split");
            
            return split_index;
        
        } // Split
        
        
        //Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
            const SReal * restrict const x = serialized_data + 1;
            const SReal * restrict const L = serialized_data + 1 + AMB_DIM;
            const SReal * restrict const A = serialized_data + 1 + AMB_DIM + AMB_DIM;
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
                // Multiply v with i-th row.
                R1 = static_cast<Real>(A[ AMB_DIM * i]) * v[0];

                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    R1 += v[j] * static_cast<Real>(A[ AMB_DIM * i + j ]);
                }
                
                R2 = (R1 >= static_cast<Real>(0)) ? static_cast<Real>(L[i]) : -static_cast<Real>(L[i]);
                
                R3 += static_cast<Real>(x[i]) * v[i] + R1 * R2;
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    s[j] +=  static_cast<Real>(A[ AMB_DIM * i + j ]) * R2;
                }
            }

            return R3;
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
            const SReal * restrict const x = serialized_data + 1;
            const SReal * restrict const L = serialized_data + 1 + AMB_DIM;
            const SReal * restrict const A = serialized_data + 1 + AMB_DIM + AMB_DIM;
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
                // Multiply v with i-th row.
                R1 = static_cast<Real>(A[AMB_DIM *i]) * v[0];

                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    R1 += v[j] * static_cast<Real>(A[ AMB_DIM * i + j ]);
                }
                
                R2 = (R1 >= static_cast<Real>(0)) ? -static_cast<Real>(L[i]) : static_cast<Real>(L[i]);
                
                R3 += static_cast<Real>(x[i]) * v[i] + R1 * R2;
                
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    s[j] += static_cast<Real>(A[ AMB_DIM * i + j ]) * R2;
                }
            }

            return R3;
        }
                
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
            min_val = MinSupportVector( dir, &this->Real_buffer[0] );
            max_val = MaxSupportVector( dir, &this->Real_buffer[AMB_DIM] );
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    }; // CLASS
    
} // namespace Collision


#undef CLASS
#undef BASE
