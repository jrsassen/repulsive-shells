#pragma once

#define CLASS BoundingVolumeBase
#define BASE  PrimitiveSerialized<AMB_DIM,Real,Int,SReal>

namespace Collision {
    
    // Class that adds bounding volume features to a basic serializable primitive, like computing the bounding volume from a set of points.
    template<int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    public:
        
        CLASS() :BASE()
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other)
        {}
        
        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE(other)
        {}
        
        virtual ~CLASS() override = default;
        
//        virtual CLASS * Clone() const override = 0;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:

        // array p is suppose to represent a matrix of size N x AMB_DIM
        virtual void FromPointCloud( const SReal * const coords, const Int N ) const = 0;
        
        virtual void FromPrimitives(
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,
            SReal * const P_data,
            const Int begin,
            const Int end,           // which _P_rimitives are in question
            Int thread_count = 1     // how many threads to utilize
        ) const = 0;
        
        virtual Int Split(
            PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,                    // primitive prototype; to be "mapped" over P_data, thus not const.
            SReal * const P_data, const Int begin, const Int end,    // which _P_rimitives are in question
            Int  * const P_ordering,                                  // to keep track of the permutation of the primitives
            SReal * const C_data, const Int C_ID,                     // where to get   the bounding volume info for _C_urrent bounding volume
            SReal * const L_data, const Int L_ID,                     // where to store the bounding volume info for _L_eft  child (if successful!)
            SReal * const R_data, const Int R_ID,                     // where to store the bounding volume info for _R_ight child (if successful!)
            SReal *       score,                                       // some scratch buffer for one scalar per primitive
            Int  *       perm,                                        // some scratch buffer for one Int per primitive (for storing local permutation)
            Int  *       inv_perm,                                    // some scratch buffer for one Int per primitive (for storing inverse of local permutation)
            Int thread_count = 1                                     // how many threads to utilize
        ) = 0;
        
//        friend void swap(BoundingVolumeBase &A, BoundingVolumeBase &B) noexcept 
//        {
//            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//            using std::swap;
//        }
//
//        // copy-and-swap idiom
//        BoundingVolumeBase & operator=(BoundingVolumeBase B)
//        {
//            // see https://stackoverflow.com/a/3279550/8248900 for details
//
//            swap(*this, B);
//            return *this;
//
//        }

        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
        }
        
    }; // PrimitiveBase

    
} // Collision

#undef BASE
#undef CLASS
