#pragma once


namespace Repulsion
{
    
    template<typename Real, typename Int>
    class alignas( 2 * CACHE_LINE_WIDTH ) MultipoleMomentsBase
    {
    public:
        
        MultipoleMomentsBase() {}
        
        // Copy constructor
        MultipoleMomentsBase( const MultipoleMomentsBase & other ) {}

        // Move constructor
        MultipoleMomentsBase( MultipoleMomentsBase && other ) noexcept  {}
        
        virtual ~MultipoleMomentsBase() = default;
        
    public:
        
        virtual Int FarDim() const = 0;
        
        virtual Int Degree() const = 0;

        virtual Int MomentCount() const = 0;
        
        virtual void PrimitiveToCluster(
            const Real * restrict const P_far,
            const Real * restrict const Q_far,       Real * restrict const Q_moments
        ) const = 0;
        
        virtual void ClusterToCluster(
            const Real * restrict const P_far, const Real * restrict const P_moments,
            const Real * restrict const Q_far,       Real * restrict const Q_moments
        ) const = 0;

        REPULSION__ADD_CLONE_CODE_FOR_BASE_CLASS(MultipoleMomentsBase);
        
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(MultipoleMomentsBase);
        }
        
        
    };
    
    
#define CLASS MultipoleMoments
#define BASE  MultipoleMomentsBase<Real,Int>
    
    template<int FAR_DIM, int DEGREE,typename Real, typename Int>
    class CLASS : public BASE
    {
    public:
        
        CLASS()
        {
            Init();
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        {
            Init();
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        {
            Init();
        }
        
        virtual ~CLASS() = default;
        
        REPULSION__ADD_CLONE_CODE(CLASS);
        
    protected:
        
        void Init()
        {
            Int moment_count;
            
            switch (DEGREE) {
                case 0:
                {
                    moment_count = 1;
                    break;
                }
                case 1:
                {
                    moment_count = 1;
                    break;
                }
                case 2:
                {
                    moment_count = (FAR_DIM*(1 + FAR_DIM))/2 - FAR_DIM + 1;
                    break;
                }
                case 3:
                {
                    moment_count = (FAR_DIM*(1 + FAR_DIM)*(2 + FAR_DIM))/6 - FAR_DIM + 1;
                    break;
                }
                case 4:
                {
                    moment_count = (FAR_DIM*(1 + FAR_DIM)*(2 + FAR_DIM)*(3 + FAR_DIM))/24 - FAR_DIM + 1;
                    break;
                }
                case 5:
                {
                    moment_count = (FAR_DIM*(1 + FAR_DIM)*(2 + FAR_DIM)*(3 + FAR_DIM)*(4 + FAR_DIM))/120 - FAR_DIM + 1;
                    break;
                }
                case 6:
                {
                    moment_count = (FAR_DIM*(1 + FAR_DIM)*(2 + FAR_DIM)*(3 + FAR_DIM)*(4 + FAR_DIM)*(5 + FAR_DIM))/720 - FAR_DIM + 1;
                    break;
                }
                default:
                {
                    moment_count = 0;
                    break;
                }
            };
            
            moment_buffer = Tensor1<Real,Int> (moment_count, 0.);
        }
        
    protected:
        
        mutable Real xs [FAR_DIM-1] = {};
        
        mutable const Real * restrict from = nullptr;
        mutable       Real * restrict to   = nullptr;
        
        mutable Tensor1<Real,Int> moment_buffer;
        
    public:
    
        virtual Int FarDim() const override
        {
            return FAR_DIM;
        }
        
        virtual Int Degree() const override
        {
            return DEGREE;
        }
        
        virtual Int MomentCount() const override
        {
            return moment_buffer.Dimension(0);
        }
            
        
        void PrimitiveToCluster(
            const Real * restrict const P_far,
            const Real * restrict const Q_far,       Real * restrict const Q_moments
        ) const override
        {
            moment_buffer[0] = P_far[0];
            SMALL_UNROLL
            for( Int k = 1; k < FAR_DIM; ++ k )
            {
                xs[k-1] = Q_far[k] - P_far[k];
            }

            from = moment_buffer.data();

            to   = Q_moments;

            this->shift();
        }
        
        void ClusterToCluster(
            const Real * restrict const P_far, const Real * restrict const P_moments,
            const Real * restrict const Q_far,       Real * restrict const Q_moments
        ) const override
        {
            SMALL_UNROLL
            for( Int k = 1; k < FAR_DIM; ++ k )
            {
                xs[k-1] = Q_far[k] - P_far[k];
            }
            
            from = P_moments;
            to   = Q_moments;
            
            this->shift();
        }
        
        void shift() const
        {
            eprint(ClassName()+"::Shift not implemented.");
        }
        
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS) + "<"+ToString(FAR_DIM)+","+ToString(DEGREE)+">";
        }
        
    };
   
    
#include "MultipoleMoments_Details.h"
    
} // namespace Repulsion

#undef BASE
#undef CLASS
