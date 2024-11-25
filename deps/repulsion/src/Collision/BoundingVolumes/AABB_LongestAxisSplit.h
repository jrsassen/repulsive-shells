#pragma once

#define BASE  AABB<AMB_DIM,Real,Int,SReal>
#define CLASS AABB_LongestAxisSplit

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
            
    public:

        using BASE::serialized_data;
        using BASE::Size;
        using BASE::self_buffer;
        using BASE::SetPointer;
        using BASE::FromPrimitives;
        
    public:
        
//#include "../Primitives/Primitive_BoilerPlate.h"
        
        REPULSION__ADD_CLONE_CODE(CLASS)
    
        virtual Int Split(
                          PolytopeBase<AMB_DIM,Real,Int,SReal> & P,                                 // primitive prototype; to be "mapped" over P_serialized, thus not const.
                          SReal * const P_serialized, const Int begin, const Int end,    // which _P_rimitives are in question
                          Int   * const P_ordering,                                        // to keep track of the permutation of the primitives
                          SReal * const C_data, const Int C_ID,                           // where to get   the bounding volume info for _C_urrent bounding volume
                          SReal * const L_serialized, const Int L_ID,                     // where to store the bounding volume info for _L_eft  child (if successful!)
                          SReal * const R_serialized, const Int R_ID,                     // where to store the bounding volume info for _R_ight child (if successful!)
                          SReal *       score,                                             // some scratch buffer for one scalar per primitive
                          Int   *       perm,                                              // some scratch buffer for one Int per primitive (for storing local permutation)
                          Int   *       inv_perm,                                          // some scratch buffer for one Int per primitive (for storing inverse of local permutation)
                          Int thread_count = 1                                           // how many threads to utilize
        )
        {
//            ptic(ClassName()+"::Split (PolytopeBase)");
            
            const Int P_Size = P.Size();
            
            SetPointer( C_data, C_ID );
            
            Int split_index = begin;
            
            if( end - begin > 2 )
            {
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
                    //                ptoc(ClassName()+"::Split (PolytopeBase)");
                    return -1;
                }
                
                // The interesting coordinate of center position.
                const SReal mid = serialized_data[1 + split_dir];
                
                
                //            SReal box_min = std::numeric_limits<SReal>::max();
                //            SReal box_max = std::numeric_limits<SReal>::lowest();
                //
                //            for( Int i = begin; i < end; ++i )
                //            {
                //                P.SetPointer( P_serialized, i );
                //
                //                SReal x = SReal x = P.InteriorPoint( split_dir );
                //
                //                box_min = std::min( box_min, x );
                //                box_max = std::max( box_max, x );
                //            }
                //            SReal mid = 0.5*(box_min+box_max);
                
                //                split_index = begin;
                
                // Swap primitives according to their interior points in direction split_dir;
                SReal * restrict const p = &P_serialized[1 + split_dir];
                for( Int i = begin; i < end; ++i )
                {
                    //                P.SetPointer( P_serialized, i );
                    //
                    //                SReal x = P.InteriorPoint( split_dir );
                    
                    // WARNING: For performance reasons, we do NOT use P.InteriorPoint for reading out the coordinate x!
                    const SReal x = p[ P_Size * i ];
                    
                    if( x < mid )
                    {
                        std::swap( P_ordering[split_index], P_ordering[i] );
                        
                        P.Swap( P_serialized, split_index, P_serialized, i );
                        
                        ++split_index;
                    }
                }
            }
            else
            {
                split_index = begin+1;
            }
            
            // TODO:It can happen that the InteriorPoints of all primitives lie on one side of mid because mid is the center of the bounding box of the primitives --- and not the center of the bounding box of the InteriorPoints!! Typically, this happens only for clusters consisting of two points (thus is caught already in the above). But sometimes, this may happen also for larger clusters.
            if( ( begin == split_index ) || ( split_index == end ) )
            {
                //Just split "randomly" into two. This should be a very, very rare case (for "nice" meshes), so this won't break much.
                split_index = begin + (end - begin)/static_cast<Int>(2);
            }
            
            if( ( begin < split_index ) && ( split_index < end ) )
            {
                // Compute bounding volume of left child.
                SetPointer( L_serialized, L_ID );
                FromPrimitives( P, P_serialized, begin,       split_index,   thread_count );
                // Compute bounding volume of right child.
                SetPointer( R_serialized, R_ID );
                FromPrimitives( P, P_serialized, split_index, end,           thread_count );
            }
            else
            {
                // ... otherwise we assume that the bounding volume hierarchy / cluster tree won't do the split.
                eprint(ClassName()+"::Split: split_index coincides with one of the end points.");
            }
            
//            ptoc(ClassName()+"::Split (PolytopeBase)");
            
            return split_index;
            
        } // Split
        
        virtual Int Split(
                          PrimitiveSerialized<AMB_DIM,Real,Int,SReal> & P,                          // primitive prototype; to be "mapped" over P_serialized, thus not const.
                          SReal * const P_serialized, const Int begin, const Int end,    // which _P_rimitives are in question
                          Int   * const P_ordering,                                        // to keep track of the permutation of the primitives
                          SReal * const C_data, const Int C_ID,                           // where to get   the bounding volume info for _C_urrent bounding volume
                          SReal * const L_serialized, const Int L_ID,                     // where to store the bounding volume info for _L_eft  child (if successful!)
                          SReal * const R_serialized, const Int R_ID,                     // where to store the bounding volume info for _R_ight child (if successful!)
                          SReal *       score,                                             // some scratch buffer for one scalar per primitive
                          Int   *       perm,                                              // some scratch buffer for one Int per primitive (for storing local permutation)
                          Int   *       inv_perm,                                          // some scratch buffer for one Int per primitive (for storing inverse of local permutation)
                          Int thread_count = 1                                           // how many threads to utilize
        ) override
        {
//            ptic(ClassName()+"::Split (PrimitiveSerialized)");
            
            const Int P_Size = P.Size();
            
            SetPointer( C_data, C_ID );
            
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
                //                ptoc(ClassName()+"::Split (PrimitiveSerialized)");
                return -1;
            }
            
            // The interesting coordinate of center position.
            const SReal mid = serialized_data[1 + split_dir];
            
            
            //            SReal box_min = std::numeric_limits<SReal>::max();
            //            SReal box_max = std::numeric_limits<SReal>::lowest();
            //
            //            for( Int i = begin; i < end; ++i )
            //            {
            //                P.SetPointer( P_serialized, i );
            //
            //                SReal x = SReal x = P.InteriorPoint( split_dir );
            //
            //                box_min = std::min( box_min, x );
            //                box_max = std::max( box_max, x );
            //            }
            //            SReal mid = 0.5*(box_min+box_max);
            
            // TODO: It can happen then the InteriorPoints of all primitives lie on one side of mid because mid is the center of the bounding box of the primitives --- and not the center of the bounding box of the InteriorPoints!!
            Int split_index = begin;
            
            // Swap primitives according to their interior points in direction split_dir;
            SReal * restrict const p = &P_serialized[1 + split_dir];
            for( Int i = begin; i < end; ++i )
            {
                //                P.SetPointer( P_serialized, i );
                //
                //                SReal x = P.InteriorPoint( split_dir );
                
                // WARNING: For performance reasons, we do NOT use P.InteriorPoint for reading out the coordinate x!
                const SReal x = p[ P_Size * i ];
                
                if( x < mid )
                {
                    std::swap( P_ordering[split_index], P_ordering[i] );
                    
                    P.Swap( P_serialized, split_index, P_serialized, i );
                    
                    ++split_index;
                }
            }
            
            if( ( begin < split_index ) && ( split_index < end ) )
            {
                // Compute bounding volume of left child.
                SetPointer( L_serialized, L_ID );
                FromPrimitives( P, P_serialized,   begin, split_index,   thread_count );
                // Compute bounding volume of right child.
                SetPointer( R_serialized, R_ID );
                FromPrimitives( P, P_serialized,   split_index, end,   thread_count );
            }
            // ... otherwise we assume that the bounding volume hierarchy / cluster tree won't do the split.
            
//            ptoc(ClassName()+"::Split (PrimitiveSerialized)");
            
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

