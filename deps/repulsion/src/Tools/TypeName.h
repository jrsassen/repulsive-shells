#pragma once

#include <stdint.h>

namespace Tools
{
    // Taken from https://stackoverflow.com/a/4485051/8248900
    
    // default implementation
    template <typename T>
    struct TypeName
    {
        static const std::string Get( void )
        {
            return typeid(T).name();
        }
    };

    template <>
    struct TypeName<int32_t>
    {
        static const std::string Get( void )
        {
            return std::string("Int32");
        }
    };
    
    template <>
    struct TypeName<int64_t>
    {
        static const std::string Get( void )
        {
            return std::string("Int64");
        }
    };

    template <>
    struct TypeName<size_t>
    {
        static const std::string Get( void )
        {
            return std::string("size_t");
        }
    };
    
    
    template <>
    struct TypeName<float>
    {
        static const std::string Get( void )
        {
            return std::string("Real32");
        }
    };
    
    template <>
    struct TypeName<double>
    {
        static const std::string Get( void )
        {
            return std::string("Real64");
        }
    };
    
    template <>
    struct TypeName<long double>
    {
        static const std::string Get( void )
        {
            return std::string("Real128");
        }
    };

    

    
} // namespace Tools
