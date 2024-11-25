#pragma once

#define BASE  AABB<AMB_DIM,Real,Int,SReal>
#define CLASS AABB_PreorderedSplit

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
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,               // primitive prototype; to be "mapped" over P_serialized, thus not const.
            SReal * const P_serialized, const Int begin, const Int end,    // which _P_rimitives are in question
            Int   * const P_ordering,                                       // to keep track of the permutation of the primitives
            SReal * const C_data, const Int C_ID,                           // where to get   the bounding volume info for _C_urrent bounding volume
            SReal * const L_serialized, const Int L_ID,                     // where to store the bounding volume info for _L_eft  child (if successful!)
            SReal * const R_serialized, const Int R_ID,                     // where to store the bounding volume info for _R_ight child (if successful!)
            SReal *       score,                                             // some scratch buffer for one scalar per primitive
            Int   *       perm,                                              // some scratch buffer for one Int per primitive (for storing local permutation)
            Int   *       inv_perm,                                          // some scratch buffer for one Int per primitive (for storing inverse of local permutation)
            Int thread_count = 1                                             // how many threads to utilize
        ) override
        {
//            ptic(ClassName()+"::Split");
            
            Int split_index = begin + ((end-begin)/2);
            
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

