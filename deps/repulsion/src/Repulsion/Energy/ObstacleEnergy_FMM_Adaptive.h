#pragma once

#define CLASS ObstacleEnergy_FMM_Adaptive
#define BASE  ObstacleEnergy_FMM<DOM_DIM1,DOM_DIM2,AMB_DIM,DEGREE,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:
        
        using Mesh_T     = typename BASE::Mesh_T;
        using N_Kernel_T = typename BASE::N_Kernel_T;
        using F_Kernel_T = typename BASE::F_Kernel_T;
        using A_Kernel_T = Energy_NearFieldKernel_Adaptive<DOM_DIM1,DOM_DIM2,AMB_DIM,Real,Int,SReal>;
        
        CLASS(
            const N_Kernel_T & N,
            const A_Kernel_T & A,
            const F_Kernel_T & F,
            ExtReal weight_ = static_cast<ExtReal>(1)
        )
        :   BASE(N,F,weight_)
        ,   A_ker(A.Clone())
        {}

        virtual ~CLASS() override = default;

        
        A_Kernel_T & NearFieldKernelAdaptive()
        {
            return *A_ker;
        }
        
        const A_Kernel_T & NearFieldKernelAdaptive() const
        {
            return *A_ker;
        }
        
    protected:
        
        using BASE::F_ker;
        using BASE::N_ker;
        using BASE::weight;
        
        using BASE::FarField;
        using BASE::DFarField;
        
        mutable std::unique_ptr<A_Kernel_T> A_ker;

        virtual Real NearField( const Mesh_T & M ) const override
        {
            auto & bct  = M.GetObstacleBlockClusterTree();
            if ( bct.CollisionCount() > 0 )
              return std::numeric_limits<Real>::infinity();

            Real sum =  NearField_Adaptive(M) +  NearField_Nonadaptive(M);

            return sum;
        }
        
        virtual Real DNearField( const Mesh_T & M ) const override
        {
            auto & bct  = M.GetObstacleBlockClusterTree();
            if ( bct.CollisionCount() > 0 )
              return std::numeric_limits<Real>::infinity();

            Real sum = DNearField_Adaptive(M) + DNearField_Nonadaptive(M);
            
            return sum;
        }
        
        
        Real NearField_Nonadaptive( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::NearField_Nonadaptive");

            auto & bct  = M.GetObstacleBlockClusterTree();
            auto & near = bct.AdaptiveNoSubdivisionData();
            
            DUMP(near.NonzeroCount());
            
            if( near.NonzeroCount() <= 0 )
            {
                logprint("No near field found. Returning 0.");
                ptoc(ClassName()+"::NearField_Nonadaptive");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
            
            auto & job_ptr = near.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const outer = near.Outer().data();
                Int const * restrict const inner = near.Inner().data();

                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                
                // This looks massive, but is only exchange of pointers.
                (void)N->LoadNearField(
                    bct.GetS().PrimitiveNearFieldData(),
                    bct.GetT().PrimitiveNearFieldData()
                );
//                (void)N->LoadDNearField(
//                    bct.GetS().ThreadPrimitiveDNearFieldData()[thread],
//                    bct.GetT().ThreadPrimitiveDNearFieldData()[thread]
//                );
                (void)N->LoadPrimitiveSerializedData(
                    bct.GetS().PrimitiveSerializedData(),
                    bct.GetT().PrimitiveSerializedData()
                );
                
                Real local_sum = static_cast<Real>(0);

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
                    
                        local_sum += N->Energy();
                        
                    } // for( Int k = k_begin; k < k_end; ++k )
                    
                } //  for( Int i = i_begin; i < i_end; ++i )
                
                sum += local_sum;
    
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::NearField_Nonadaptive");
            return sum;
        }
        
        
        Real DNearField_Nonadaptive( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::DNearField_Nonadaptive");

            auto & bct  = M.GetObstacleBlockClusterTree();
            auto & near = bct.AdaptiveNoSubdivisionData();
            
            DUMP(near.NonzeroCount());
            
            if( near.NonzeroCount() <= 0 )
            {
                logprint("No near field found. Returning 0.");
                ptoc(ClassName()+"::DNearField_Nonadaptive");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
    
            auto & job_ptr = near.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const outer = near.Outer().data();
                Int const * restrict const inner = near.Inner().data();
                
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                
                // This looks massive, but is only exchange of pointers.
                (void)N->LoadNearField(
                    bct.GetS().PrimitiveNearFieldData(),
                    bct.GetT().PrimitiveNearFieldData()
                );
                (void)N->LoadDNearField(
                    bct.GetS().ThreadPrimitiveDNearFieldData()[thread],
                    bct.GetT().ThreadPrimitiveDNearFieldData()[thread]
                );
                (void)N->LoadPrimitiveSerializedData(
                    bct.GetS().PrimitiveSerializedData(),
                    bct.GetT().PrimitiveSerializedData()
                );
                
                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                for( Int i = i_begin; i < i_end; ++i )
                {
                    N->LoadS(i);
                    N->CleanseDBufferS();
                    
                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];
                     
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];
                        
                        N->LoadT(j);
                        N->CleanseDBufferT();
                        
                        local_sum += N->DEnergy();
                    
                        N->WriteDBufferT();
                        
                    } // for( Int k = k_begin; k < k_end; ++k )
                    
                    N->WriteDBufferS();
                    
                } // for( Int i = i_begin; i < i_end; ++i )
                 
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::DNearField_Nonadaptive");
            return sum;
        }
        
        
        
        Real NearField_Adaptive( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::NearField_Adaptive");

            auto & bct  = M.GetObstacleBlockClusterTree();
            auto & near = bct.AdaptiveSubdivisionData();
            
            DUMP(near.NonzeroCount());
            
            if( near.NonzeroCount() <= 0 )
            {
                logprint("No primitive pairs to subdivide found. Returning 0.");
                ptoc(ClassName()+"::NearField_Adaptive");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
    
            auto & job_ptr = near.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const outer = near.Outer().data();
                Int const * restrict const inner = near.Inner().data();
                
                std::unique_ptr<A_Kernel_T> A = A_ker->Clone();
                
#ifdef REPULSION__PRINT_REPORTS_FOR_ADAPTIVE_KERNELS
                A->CreateLogFile();
#endif
                
                // This looks massive, but is only exchange of pointers.
                (void)A->LoadNearField(
                    bct.GetS().PrimitiveNearFieldData(),
                    bct.GetT().PrimitiveNearFieldData()
                );
//                (void)A->LoadDNearField(
//                    bct.GetS().ThreadPrimitiveDNearFieldData()[thread],
//                    bct.GetT().ThreadPrimitiveDNearFieldData()[thread]
//                );
                (void)A->LoadPrimitiveSerializedData(
                    bct.GetS().PrimitiveSerializedData(),
                    bct.GetT().PrimitiveSerializedData()
                );
                
                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                for( Int i = i_begin; i < i_end; ++i )
                {
                    A->LoadS(i);
//                    A->CleanseDBufferS();
                    
                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];
                     
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];
                        
                        A->LoadT(j);
//                        A->CleanseDBufferT();
                        
                        local_sum += A->Energy();
                    
//                        A->WriteDBufferT();
                        
                    } // for( Int k = k_begin; k < k_end; ++k )
                    
//                    A->WriteDBufferS();
                    
                } // for( Int i = i_begin; i < i_end; ++i )
                 
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::NearField_Adaptive");
            return sum;
        }
        
        
        Real DNearField_Adaptive( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::DNearField_Adaptive");

            auto & bct  = M.GetObstacleBlockClusterTree();
            auto & near = bct.AdaptiveSubdivisionData();
            
            DUMP(near.NonzeroCount());
            
            if( near.NonzeroCount() <= 0 )
            {
                logprint("No primitive pairs to subdivide found. Returning 0.");
                ptoc(ClassName()+"::DNearField_Adaptive");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
    
            auto & job_ptr = near.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const outer = near.Outer().data();
                Int const * restrict const inner = near.Inner().data();
                
                std::unique_ptr<A_Kernel_T> A = A_ker->Clone();
                
                // This looks massive, but is only exchange of pointers.
                (void)A->LoadNearField(
                    bct.GetS().PrimitiveNearFieldData(),
                    bct.GetT().PrimitiveNearFieldData()
                );
                (void)A->LoadDNearField(
                    bct.GetS().ThreadPrimitiveDNearFieldData()[thread],
                    bct.GetT().ThreadPrimitiveDNearFieldData()[thread]
                );
                (void)A->LoadPrimitiveSerializedData(
                    bct.GetS().PrimitiveSerializedData(),
                    bct.GetT().PrimitiveSerializedData()
                );
                
                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                for( Int i = i_begin; i < i_end; ++i )
                {
                    A->LoadS(i);
                    A->CleanseDBufferS();
                    
                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];
                     
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];
                        
                        A->LoadT(j);
                        A->CleanseDBufferT();
                        
                        local_sum += A->DEnergy();
                    
                        A->WriteDBufferT();
                        
                    } // for( Int k = k_begin; k < k_end; ++k )
                    
                    A->WriteDBufferS();
                    
                } // for( Int i = i_begin; i < i_end; ++i )
                 
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::DNearField_Adaptive");
            return sum;
        }
        
    public:
        
        virtual std::string Stats() const override
        {
            std::stringstream s;
            
            s << ClassName() <<": \n\n";
            if( N_ker != nullptr )
            {
                s <<  "          NearFieldKernel          = " << N_ker->Stats() << "\n\n";
            }
            if( A_ker != nullptr )
            {
                s <<  "          NearFieldKernel_Adaptive = " << A_ker->Stats() << "\n\n";
            }
            if( F_ker != nullptr )
            {
                s <<  "          FarFieldKernel           = " << F_ker->Stats() << "\n\n";
            }
            return s.str();
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM1)+","+ToString(DOM_DIM2)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef BASE
#undef CLASS
