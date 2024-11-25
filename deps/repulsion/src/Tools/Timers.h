#pragma once

namespace Tools
{
    
    template <typename T>
    std::string ToString(const T & a_value, const int n = 16)
    {
        std::stringstream sout;
        sout.precision(n);
        sout << a_value;
        return sout.str();
    }

    namespace Timers
    {
        std::deque<std::chrono::time_point<std::chrono::steady_clock>> time_stack;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> stop_time;
    }
    
#ifdef LTEMPLATE_H
    inline void print( const std::string & s )
    {
        mma::print( std::string( 2 * (Timers::time_stack.size()), ' ') + s );
//        mma::print( s );
    }
#else
    inline void print( const std::string & s )
    {
        std::cout << s << std::endl;
    }
#endif

    template<typename T>
    inline void valprint( const std::string & s, T value)
    {
        using namespace std;
        print( s + " = " + to_string(value) );
    }

    template<typename T>
    inline void varprint( const std::string & s, const T & value)
    {
        using namespace std;
        print( s + " = " + to_string(value) );
    }

    template<typename T>
    inline void valprint( const std::string & s, const T & value, const int prec)
    {
        print( s + " = " + ToString(value, prec) );
    }

    template<typename T>
    inline void varprint( const std::string & s, const T & value, const int prec)
    {
        print( s + " = " + ToString(value, prec) );
    }
    
    inline void tic(const std::string & s)
    {
        print(s + "...");
        Timers::time_stack.push_back(std::chrono::steady_clock::now());
    }

    inline double toc(const std::string & s)
    {
        if (!Timers::time_stack.empty())
        {
            auto start_time = Timers::time_stack.back();
            auto stop_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(stop_time - start_time).count();
            Timers::time_stack.pop_back();
            print( std::to_string(duration) + " s.");
            return duration;
        }
        else
        {
            print("Unmatched toc detected. Label =  " + s);
            return 0.;
        }
    }

//    inline void tic( void )
//    {
//        Timers::time_stack.push_back(std::chrono::steady_clock::now());
//    }
//
//    inline double toc( void )
//    {
//        if (!Timers::time_stack.empty())
//        {
//            auto start_time = Timers::time_stack.back();
//            auto stop_time = std::chrono::steady_clock::now();
//            double duration = std::chrono::duration<double>(stop_time - start_time).count();
//            Timers::time_stack.pop_back();
//            return duration;
//        }
//        else
//        {
//            print("Unmatched toc detected.");
//            return 0.;
//        }
//    }
}
