#pragma once

#include <deque>
#include <vector>
#include <unistd.h>
#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>
#include <type_traits>
#include <memory>
#include <numeric>
#include <filesystem>


#define STRINGIFY(a) #a
#define STRINGIFY2(a) STRINGIFY(a)

#define TO_STD_STRING(x) std::string(STRINGIFY(x))

#define CONCATT(id1, id2) id1##id2
#define CONCAT(id1, id2) CONCATT(id1, id2)

#define CONCATT3(id1, id2, id3) id1##id2##id3
#define CONCAT3(id1, id2, id3) CONCATT3(id1, id2, id3)

#define SMALL_UNROLL
//#define SMALL_UNROLL _Pragma( STRINGIFY( unroll ) )
//#define SMALL_UNROLL _Pragma( STRINGIFY( clang loop vectorize(disable) unroll(full) ) )



#define OMP_PLACES cores
#define OMP_WAIT_POLICY = active
#define OMP_DYNAMIC = false
//#define OMP_PROC_BIND = true
#define OMP_PROC_BIND spread // workers are spread across the available places to maximize the space inbetween two neighbouring threads
//#define OMP_PROC_BIND close // worker threads are close to the master in contiguous partitions, e. g. if the master is occupying hardware thread 0, worker 1 will be placed on hw thread 1, worker 2 on hw thread 2 and so on

#include <omp.h>

//#define MKL_DIRECT_CALL_SEQ_JIT
//
//#include <mkl.h>
//#include <mkl_cblas.h>
//#include <mkl_lapacke.h>
//
//void HelloMKL()
//{
//    MKLVersion Version;
//    mkl_get_version(&Version);
//    print(
//          "\n================================================================================================"
//    );
//    print("MKL version:            "+ToString(Version.MajorVersion)+"."+ToString(Version.MinorVersion)+"."+ToString(Version.UpdateVersion));
////        valprint("Major version:          ",Version.MajorVersion);
////        valprint("Minor version:          ",Version.MinorVersion);
////        valprint("Update version:         ",Version.UpdateVersion);
//    print("Product status:         "+ToString(Version.ProductStatus));
//    print("Build:                  "+ToString(Version.Build));
//    print("Platform:               "+ToString(Version.Platform));
//    print("Processor optimization: "+ToString(Version.Processor));
//    print(
//          "================================================================================================\n"
//    );
//    print("\n");
//}

#ifndef WOLFRAMLIBRARY_H
typedef double mreal;  // "machine real"
typedef long long mint;  // "machine integer"
#endif

#include "src/Tools/TypeName.h"
#include "src/Tools/Timers.h"
#include "src/Tools/Profiler.h"
#include "src/Tools/Memory.h"
//#include "src/Tools/CloneInherit.h"


//https://stackoverflow.com/a/43587319/8248900

#define REPULSION__ADD_CLONE_CODE_FOR_BASE_CLASS(BASE)                          \
public:                                                                         \
    std::unique_ptr<BASE> Clone () const                                        \
    {                                                                           \
        return std::unique_ptr<BASE>(CloneImplementation());                    \
    }                                                                           \
private:                                                                        \
    virtual BASE * CloneImplementation() const = 0;                             

#define REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)                     \
public:                                                                         \
    std::unique_ptr<CLASS> Clone () const                                       \
    {                                                                           \
        return std::unique_ptr<CLASS>(CloneImplementation());                   \
    }                                                                           \
private:                                                                        \
    virtual CLASS * CloneImplementation() const override = 0;


#define REPULSION__ADD_CLONE_CODE(DERIVED)                                      \
public:                                                                         \
    std::unique_ptr<DERIVED> Clone () const                                     \
    {                                                                           \
        return std::unique_ptr<DERIVED>(CloneImplementation());                 \
    }                                                                           \
                                                                                \
private:                                                                        \
    virtual DERIVED * CloneImplementation() const override                      \
    {                                                                           \
        return new DERIVED(*this);                                              \
    }





#define IsFloat(T)  class = typename std::enable_if_t<std::is_floating_point_v<T>>
#define IsInt(I)    class = typename std::enable_if_t<std::is_signed_v<I> && std::is_integral_v<I>>


#define ASSERT_INT(I) static_assert( std::is_signed_v<I> && std::is_integral_v<I>, "Template parameter " #I " must be integral type." );

#define ASSERT_FLOAT(type) static_assert( std::is_floating_point_v<type>, "Template parameter " #type " must be floating point type." );


// Copy swap idiom
