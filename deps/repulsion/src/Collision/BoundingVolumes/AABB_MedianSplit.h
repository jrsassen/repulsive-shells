#pragma once

#define BASE  AABB<AMB_DIM,Real,Int,SReal>
#define CLASS AABB_MedianSplit

namespace Collision {
    
    
    // serialized_data is assumed to be an array of size Size(). Will never be allocated by class! Instead, it is meant to be mapped onto an array of type SReal by calling the member SetPointer.
    
    // DATA LAYOUT
    // serialized_data[0] = squared radius
    // serialized_data[1],...,serialized_data[AMB_DIM] = center
    // serialized_data[AMB_DIM + 1],...,serialized_data[AMB_DIM + AMB_DIM] = half the edge lengths.
    
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
        
    protected:
            
        using BASE::serialized_data;
        using BASE::Size;
        using BASE::self_buffer;
        using BASE::SetPointer;
        using BASE::FromPrimitives;
        
    public:
        
//#include "../Primitives/Primitive_BoilerPlate.h"
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
        virtual Int Split(
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,                          // primitive prototype; to be "mapped" over P_serialized, thus not const.
            SReal * const P_serialized, const Int begin, const Int end,    // which _P_rimitives are in question
            Int  * const P_ordering,                                        // to keep track of the permutation of the primitives
            SReal * const C_data, const Int C_ID,                           // where to get   the bounding volume info for _C_urrent bounding volume
            SReal * const L_serialized, const Int L_ID,                     // where to store the bounding volume info for _L_eft  child (if successful!)
            SReal * const R_serialized, const Int R_ID,                     // where to store the bounding volume info for _R_ight child (if successful!)
            SReal *       score,                                             // some scratch buffer for one scalar per primitive
            Int  *       perm,                                              // some scratch buffer for one Int per primitive (for storing local permutation)
            Int  *       inv_perm,                                          // some scratch buffer for one Int per primitive (for storing inverse of local permutation)
            Int thread_count = 1                                           // how many threads to utilize
        ) override
        {
//            ptic(ClassName()+"::Split");
//            valprint("begin",begin);
//            valprint("end",end);
            
            Int P_Size = P.Size();
            
            this->SetPointer( C_data, C_ID );
            
            // Find the longest axis `split_dir` of primitives's bounding box.
            Int split_dir = 0;
            
            SReal L_max = serialized_data[1 + AMB_DIM + split_dir];
            
            for( Int k = 1; k < AMB_DIM; ++k )
            {
                SReal L_k = serialized_data[1 + AMB_DIM + k];
                if( L_k > L_max )
                {
                    L_max = L_k;
                    split_dir = k;
                }
            }
            
            if( L_max <= static_cast<SReal>(0) )
            {
                eprint(ClassName()+"Split: longest axis has length <=0.");
                return -1;
            }
            
            
            // Finding the "median". Adapted from https://stackoverflow.com/a/16798127/8248900 (using pointers instead of iterators)

            // Computing score as pojection of the primitives' InteriorPoints on the longest axis.
            // Fill perm with the indices.

            const SReal * restrict const p = P_serialized + 1 + split_dir;

            for( Int i = begin; i < end; ++i )
            {
                score[i] = p[ P_Size * i ];
                perm [i] = i;
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
            
            if( ( begin < split_index ) && ( split_index < end ) )
            {
                // Compute bounding volume of left child.
                this->SetPointer( L_serialized, L_ID );
                this->FromPrimitives( P, P_serialized,   begin, split_index,   thread_count );
                // Compute bounding volume of right child.
                this->SetPointer( R_serialized, R_ID );
                this->FromPrimitives( P, P_serialized,   split_index, end,   thread_count );
            }
            // ... otherwise we assume that the bounding volume hierarchy / cluster tree won't do the split.
            
//            ptoc(ClassName()+"::Split");
            
            return split_index;
            
        } // Split
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    }; // CLASS

} // namespace Collision

#undef CLASS
#undef BASE

