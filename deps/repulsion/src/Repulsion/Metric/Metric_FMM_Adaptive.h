#pragma once

#define CLASS Metric_FMM_Adaptive
#define BASE  Metric_FMM<DOM_DIM,AMB_DIM,DEGREE,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
        
    public:
        
        using BlockClusterTree_T = typename BASE::BlockClusterTree_T;
        using N_Kernel_T         = typename BASE::N_Kernel_T;
        using F_Kernel_T         = typename BASE::F_Kernel_T;
        using A_Kernel_T         = Metric_NearFieldKernel_Adaptive <DOM_DIM,DOM_DIM,AMB_DIM,Real,Int,SReal>;
        
        
        CLASS() {}
        
        CLASS(
            const BlockClusterTree_T & bct_,
            const N_Kernel_T & N,
            const A_Kernel_T & A,
            const F_Kernel_T & F,
            const Real weight_ = static_cast<Real>(1)
        )
        :   BASE  ( bct_, N, F, weight_  )
        ,   A_ker ( A.Clone()            )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE  ( other                )
        ,   A_ker ( other.A_ker->Clone() )
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept
        :   BASE  ( other                )
        ,   A_ker ( other.A_ker->Clone() )
        {}

        virtual ~CLASS() override = default;
        
        
        
    protected:
        
        using BASE::bct;
        using BASE::N_ker;
        using BASE::F_ker;
        using BASE::weight;
        using BASE::near_values;
        
        std::unique_ptr<A_Kernel_T> A_ker;
        
    protected:

        void NearField() const override
        {
//            ptic(ClassName()+"::NearField");
            
            N_ker->AllocateValueBuffers(near_values, bct->Near().NonzeroCount());
            
            bct->RequireAdaptiveSubdivisionData();
            
            NearField_Nonadaptive();
            
            NearField_Adaptive();
            
//            ptoc(ClassName()+"::NearField");
        }
        
        void NearField_Nonadaptive() const
        {
            ptic(ClassName()+"::NearField_Nonadaptive");
    
            auto & near = bct->AdaptiveNoSubdivisionData();
            
            if( near.NonzeroCount() <= 0 )
            {
                wprint(ClassName()+"::NearField_Nonadaptive no nonseparated blocks detected. Skipping.");
                ptoc(ClassName()+"::NearField_Nonadaptive");
                return;
            }
            
            auto & job_ptr = near.JobPtr();
                        
            Int const * restrict const outer = near.Outer().data();
            Int const * restrict const inner = near.Inner().data();
            Int const * restrict const pos   = near.Value().data();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                
                N->LoadValueBuffers(near_values);

                // This looks massive, but is only exchange of pointers.
                (void)N->LoadNearField(
                    bct->GetS().PrimitiveNearFieldData(),
                    bct->GetT().PrimitiveNearFieldData()
                );
                (void)N->LoadPrimitiveSerializedData(
                    bct->GetS().PrimitiveSerializedData(),
                    bct->GetT().PrimitiveSerializedData()
                );
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    N->LoadS(i);

                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];
                    
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];

                        N->LoadT(j);
                        
                        N->Metric(pos[k]);

                    } // for (Int k = k_begin; k < k_end; ++k)
                    
                } // for( Int i = i_begin; i < i_end; ++i )
                
            } // #pragma omp parallel

            ptoc(ClassName()+"::NearField_Nonadaptive");
        }
        
        void NearField_Adaptive() const
        {
            ptic(ClassName()+"::NearField_Adaptive");
            
            auto & near = bct->AdaptiveSubdivisionData();
            
            if( near.NonzeroCount() <= 0 )
            {
//                wprint(ClassName()+"::NearField_Adaptive no nonseparated blocks detected. Skipping.");
                ptoc(ClassName()+"::NearField_Adaptive");
                return;
            }
            
            auto & job_ptr = near.JobPtr();
            
            Int const * restrict const outer = near.Outer().data();
            Int const * restrict const inner = near.Inner().data();
            Int const * restrict const pos   = near.Value().data();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<N_Kernel_T> A = A_ker->Clone();
                
                A->LoadValueBuffers(near_values);

                // This looks massive, but is only exchange of pointers.
                (void)A->LoadNearField(
                    bct->GetS().PrimitiveNearFieldData(),
                    bct->GetT().PrimitiveNearFieldData()
                );
                (void)A->LoadPrimitiveSerializedData(
                    bct->GetS().PrimitiveSerializedData(),
                    bct->GetT().PrimitiveSerializedData()
                );
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    A->LoadS(i);

                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];
                    
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];
                        
                        A->LoadT(j);
                    
                        A->Metric(pos[k]);

                    } // for (Int k = k_begin; k < k_end; ++k)
                    
                } // for( Int i = i_begin; i < i_end; ++i )
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::NearField_Adaptive");
        }
        
    public:
        
        virtual std::string Stats() const override
        {
            std::stringstream s;
            
            s << ClassName() << ": \n\n";
            if( N_ker != nullptr )
            {
                s << "          NearFieldKernel          =" << N_ker->Stats() << "\n\n";
            }
            
            if( A_ker != nullptr )
            {
                s << "          adaptive NearFieldKernel =" << A_ker->Stats() << "\n\n";
            }
            
            if( F_ker != nullptr )
            {
                s << "          FarFieldKernel           = " << F_ker->Stats() << "\n";
            }
            return s.str();
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef BASE
#undef CLASS
