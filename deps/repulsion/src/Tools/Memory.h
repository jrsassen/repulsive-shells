#pragma once

#define CACHE_LINE_WIDTH 64    // length of cache line measured in bytes
#define CACHE_LINE_LENGTH 8    // length of cache line measured in number of doubles
#define CACHE_SKIP 1           // There will be at least CACHE_SKIP-1 lines of full cache lines between two independent memory blocks.

#define restrict __restrict
#define prefetch __builtin_prefetch

#define SAFE_ALLOCATE_WARNINGS

#define ALIGN CACHE_LINE_WIDTH // MKL suggests 64-byte alignment
#define CHUNK_SIZE CACHE_LINE_LENGTH

#define RAGGED_SCHEDULE schedule( guided, CHUNK_SIZE)
#define STATIC_SCHEDULE schedule( static, 16 * CACHE_LINE_LENGTH )

namespace Tools
{
    
    inline void* aligned_malloc(size_t size, size_t align)
    {
        
        void *result;
        
#ifdef _MSC_VER
        result = _aligned_malloc(size, align);
        if( result == nullptr )
        {
            eprint("aligned_malloc: _aligned_malloc failed to allocate memory.");
        }
        #else
        bool failed = posix_memalign(&result, align, size);
        if( failed )
        {
             eprint("aligned_malloc: posix_memalign failed to allocate memory.");
             result = nullptr;
        }
        #endif
        
//        result = mkl_malloc ( size, static_cast<int>(align) );
//        
//        if( result!= nullptr)
//        {
//            print("aligned_malloc successfully allocated " + ToString(size) + " bytes.");
//        }
        
        return result;
    }

    inline void aligned_free(void *ptr)
    {
        
        #ifdef _MSC_VER
            _aligned_free(ptr);
        #else
          free(ptr);
        #endif
        
//        mkl_free(ptr);
    }
    
    
    template <typename T>
    int safe_free( T * & ptr )
    {
        int wasallocated = (ptr != nullptr);
        if( wasallocated ){ aligned_free(ptr); ptr = nullptr; }
        return !wasallocated;
    }
    
    template <typename T>
    int safe_alloc(T * & ptr, size_t size)
    {
        int wasallocated = (ptr != nullptr);
        if( wasallocated )
        {
#ifdef SAFE_ALLOCATE_WARNINGS
            wprint("safe_alloc: Pointer was not NULL. Calling safe_free to prevent memory leak.");
#endif
            safe_free(ptr);
        }
        ptr = (T *) aligned_malloc( size * sizeof(T), ALIGN );

        return wasallocated;
    }
    
    template <typename T>
    int safe_alloc( T * & ptr, size_t size, T init)
    {
        int wasallocated = safe_alloc(ptr, size);
        #pragma omp simd aligned( ptr : ALIGN )
        for( size_t i = 0; i < size; ++i )
        {
            ptr[i] = init;
        }
        return wasallocated;
    }
    
    template <typename T>
    int safe_iota(T * & ptr, size_t size, T step = static_cast<T>(1) )
    {
        int wasallocated = safe_alloc(ptr, size);
        #pragma omp simd aligned( ptr : ALIGN )
        for( size_t i = 0; i < size; i+=step )
        {
            ptr[i] = i;
        }
        return wasallocated;
    }
    
#if defined(__GNUC__) && !defined(__clang__)
// do this *only* for gcc
    
    // overload functions for restrict qualifier
    
     template <typename T>
     inline int safe_free( T * restrict & ptr )
     {
         int wasallocated = (ptr != nullptr);
         if( wasallocated ){ aligned_free(ptr); ptr = nullptr; }
         return !wasallocated;
     }
    
     template <typename T>
     int safe_alloc(T * restrict & ptr, size_t size)
     {
         int wasallocated = (ptr != nullptr);
         if( wasallocated )
         {
 #ifdef SAFE_ALLOCATE_WARNINGS
             wprint("safe_alloc: Pointer was not NULL. Calling safe_free to prevent memory leak.");
 #endif
             safe_free(ptr);
         }
         ptr = (T *) aligned_malloc( size * sizeof(T), ALIGN );
         
         return wasallocated;
     }
    
    template <typename T>
    inline int safe_alloc( T * restrict & ptr, size_t size, T init)
    {
        int wasallocated = safe_alloc(ptr, size);
        #pragma omp simd aligned( ptr : ALIGN )
        for( size_t i = 0; i < size; ++i )
        {
            ptr[i] = init;
        }
        return wasallocated;
    }
    
    template <typename T>
    inline int safe_iota(T * restrict & ptr, size_t size, T step = static_cast<T>(1) )
    {
        int wasallocated = safe_alloc(ptr, size);
        #pragma omp simd aligned( ptr : ALIGN )
        for( size_t i = 0; i < size; i+=step )
        {
            ptr[i] = i;
        }
        return wasallocated;
    }
#endif
    
    
    template <typename T>
    inline void partial_sum( T * begin, T * end)
    {
        std::partial_sum( begin, end, begin );
    }
    
    
} // namespace Tools

template<typename From, typename To>
struct static_caster
{
    To operator()(const From & p)
    {
        return static_cast<To>(p);
    }
};
