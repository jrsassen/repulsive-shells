#pragma once

namespace Tools
{
    namespace Profiler
    {
        std::ofstream prof ( "./Repulsion_Profile.tsv" );
        std::ofstream log ( "./Repulsion_Log.txt" );

        std::chrono::time_point<std::chrono::steady_clock> init_time = std::chrono::steady_clock::now();
        std::deque<std::chrono::time_point<std::chrono::steady_clock>> time_stack;
        std::deque<std::string> tag_stack(1,"root");
        std::deque<int> parent_stack (1, -0);
        std::deque<int> id_stack (1,0);
        int id_counter = 0;
        
        
        inline void Clear(const std::string & dir, const bool append = false)
        {
            std::filesystem::path profile_filename = dir;
            profile_filename /= "Repulsion_Profile.tsv";
            Profiler::prof.close();
            if( append )
            {
                Profiler::prof.open(profile_filename.c_str(), std::ios_base::app);
            }
            else
            {
                Profiler::prof.open(profile_filename.c_str());
            }
            std::cout << "Profile will be written to " << profile_filename << "." << std::endl;
            Profiler::log << std::setprecision(16);
            
            std::filesystem::path log_filename = dir;
            log_filename /= "Repulsion_Log.txt";
            Profiler::log.close();

            if( append )
            {
                Profiler::log.open(log_filename.c_str(), std::ios_base::app);
            }
            else
            {
                Profiler::log.open(log_filename.c_str());
            }
            
            std::cout << "Log will be written to " << log_filename << "." << std::endl;
            Profiler::log << std::setprecision(16);
    //        Profiler::prof << "ID" << "\t" << "Tag" << "\t" << "From" << "\t" << "Tic" << "\t" << "Toc" << "\t" << "Duration" << "\t" << "Depth" << "\n";
            Profiler::init_time = std::chrono::steady_clock::now();
            Profiler::time_stack.clear();
            Profiler::parent_stack.push_back(0.);
            Profiler::tag_stack.clear();
            Profiler::tag_stack.push_back("root");
            Profiler::parent_stack.clear();
            Profiler::parent_stack.push_back(-1);
            Profiler::id_stack.clear();
            Profiler::id_stack.push_back(0);
            Profiler::id_counter = 0;
        }
        
    }
    
    inline void logprint( const std::string & s )
    {
#if defined REPULSION_LOG
        for( size_t i = 0; i <= Profiler::time_stack.size(); ++i )
        {
            Profiler::log << "  ";
        }
        Profiler::log << s << "\n" << std::endl;
#endif
    }

#if defined REPULSION_LOG
#define DUMP(x)                                                                     \
    {                                                                               \
        for( size_t ___i = 0; ___i <= Profiler::time_stack.size(); ++___i )         \
        {                                                                           \
            Profiler::log << "  ";                                                  \
        }                                                                           \
        Profiler::log << std::string(#x) << std::string(" = ");                     \
        Profiler::log << x;                                                         \
        Profiler::log << "\n" << std::endl;                                         \
    }
#else
#define DUMP(x)
#endif

    inline void eprint( const std::string & s )
    {
        print(    std::string("ERROR: ") + s );
        logprint( std::string("ERROR: ") + s );
    }

    inline void wprint( const std::string & s )
    {
        print(    std::string("WARNING: ") + s );
        logprint( std::string("WARNING: ") + s );
    }
    
    inline void ptic_(const std::string & tag)
    {
        Profiler::time_stack.push_back(std::chrono::steady_clock::now());
        Profiler::parent_stack.push_back(Profiler::id_stack.back());
        Profiler::tag_stack.push_back(tag);
        Profiler::id_stack.push_back(++Profiler::id_counter);
        
        double start_time = std::chrono::duration<double>( Profiler::time_stack.back()      - Profiler::init_time ).count();
        
        for( size_t i = 0; i < Profiler::time_stack.size(); ++i )
        {
            Profiler::log << "  ";
        }
        Profiler::log
            << Profiler::tag_stack.back() << "\t started at \t" << start_time << "\n"
            << std::endl;
        
    }
    
    inline void ptoc_(const std::string & tag)
    {
        if( !Profiler::tag_stack.empty() )
        {
            if( tag == Profiler::tag_stack.back() )
            {
                double start_time = std::chrono::duration<double>( Profiler::time_stack.back()      - Profiler::init_time ).count();
                double stop_time  = std::chrono::duration<double>( std::chrono::steady_clock::now() - Profiler::init_time ).count();
                
                Profiler::prof
                    << Profiler::id_stack.back() <<  "\t"
                    << Profiler::tag_stack.back() << "\t"
                    << Profiler::parent_stack.back() << "\t"
                    << start_time << "\t"
                    << stop_time << "\t"
                    << stop_time-start_time << "\t"
                    << Profiler::tag_stack.size()-1
                    << std::endl;
                
                for( size_t i = 0; i < Profiler::time_stack.size(); ++i )
                {
                    Profiler::log << "  ";
                }
                Profiler::log
                    << Profiler::tag_stack.back() << "\t ended   at \t" << stop_time << "\n"
                    << std::endl;
                
                Profiler::id_stack.pop_back();
                Profiler::time_stack.pop_back();
                Profiler::tag_stack.pop_back();
                Profiler::parent_stack.pop_back();
            }
            else
            {
                eprint(std::string("Unmatched ptoc detected."));
                eprint(std::string("  Expected label =  ") + tag);
                eprint(std::string("  Visited label  =  ") + Profiler::tag_stack.back());
            }
        }
        else
        {
            std::cout << "Unmatched ptoc detected. Stack empty. Label =  " << tag << " detected." << std::endl;
        }
    }
    
#if defined REPULSION_PROFILE
    
#define ptic(tag) ptic_(tag)
#define ptoc(tag) ptoc_(tag)
    
#else
#define ptic(tag)
#define ptoc(tag)
//    inline void ptic(const std::string & tag) {};
//    inline void ptoc(const std::string & tag) {};
//    namespace Profiler
//    {
//        inline void Clear(const std::string & filename){}
//    }
#endif

}
