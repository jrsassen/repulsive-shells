#pragma once

#define CLASS ObstacleEnergy_FMM
#define BASE  Energy_Restricted<DOM_DIM1,AMB_DIM,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, int DEGREE, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:
        
        using Mesh_T     = typename BASE::Mesh_T;
        using N_Kernel_T = Energy_NearFieldKernel    <DOM_DIM1,DOM_DIM2,AMB_DIM,Real,Int,SReal>;
        using F_Kernel_T = Energy_FarFieldKernel_FMM <AMB_DIM,DEGREE,Real,Int>;
        
        CLASS(
            const N_Kernel_T & N,
            const F_Kernel_T & F,
            ExtReal weight_ = static_cast<ExtReal>(1))
        :   BASE(weight_)
        ,   N_ker(N.Clone())
        ,   F_ker(F.Clone())
        {}

        virtual ~CLASS() override = default;
        
        N_Kernel_T & NearFieldKernel()
        {
            return *N_ker;
        }
        
        const N_Kernel_T & NearFieldKernel() const
        {
            return *N_ker;
        }
        
        F_Kernel_T & FarFieldKernel()
        {
            return *F_ker;
        }
        
        const F_Kernel_T & FarFieldKernel() const
        {
            return *F_ker;
        }

    public:
        
        bool use_blocking = false;
        
    protected:
        
        using BASE::weight;
        
        mutable std::unique_ptr<N_Kernel_T> N_ker;
        mutable std::unique_ptr<F_Kernel_T> F_ker;
        
    public:
        
        // Returns the current value of the energy.
        ExtReal Value( const Mesh_T & M ) const override
        {
            ptic(ClassName()+"::Value");

            DUMP(Stats());
            
            Real near = static_cast<Real>(0);
            Real  far = static_cast<Real>(0);
            
            if( M.ObstacleInitialized() )
            {
                near = NearField(M);
                 far =  FarField(M);
            }
            
            DUMP(near);
            DUMP(far);
            
            ptoc(ClassName()+"::Value");
            return weight * static_cast<ExtReal>(near+far);
        }
        
        // Returns the current differential of the energy, stored in the given
        // V x AMB_DIM matrix, where each row holds the differential (a AMB_DIM-vector) with
        // respect to the corresponding vertex.
        ExtReal Differential( const Mesh_T & M, ExtReal * output, bool addTo = false ) const override
        {
            ptic(ClassName()+"::Differential");
            
            Real near = static_cast<Real>(0);
            Real  far = static_cast<Real>(0);
            
            if( M.ObstacleInitialized() )
            {
                M.GetObstacleBlockClusterTree().GetS().CleanseDerivativeBuffers();
                M.GetObstacleBlockClusterTree().GetT().CleanseDerivativeBuffers();
                
                near = DNearField(M);
                 far =  DFarField(M);
                
                M.Assemble_ClusterTree_Derivatives( output, weight, addTo );
            }
            else
            {
                if( !addTo )
                {
                    memset( output, 0, M.DofCount() * sizeof(ExtReal) );
                }
            }
            
            DUMP(near);
            DUMP(far);
            
            ptoc(ClassName()+"::Differential");
            return weight * static_cast<ExtReal>(near+far);
        }
        
    protected:
        
        virtual Real NearField( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::NearField");

            auto & bct  = M.GetObstacleBlockClusterTree();
            auto & near = bct.Near();
            
            if( near.NonzeroCount() <= 0 )
            {
//                wprint(ClassName()+"::NearField: no near field found. Returning 0." );
                ptoc(ClassName()+"::NearField");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
            
            auto & job_ptr = near.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count ) reduction( + : sum )
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
            
            ptoc(ClassName()+"::NearField");
            return sum;
        }
        
        
        virtual Real DNearField( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::DNearField");

            auto & bct  = M.GetObstacleBlockClusterTree();
            auto & near = bct.Near();
            
            if( near.NonzeroCount() <= 0 )
            {
//                wprint(ClassName()+"::DNearField: no near field found. Returning 0." );
                ptoc(ClassName()+"::DNearField");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
            
            auto & job_ptr = near.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count ) reduction( + : sum )
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
            
            ptoc(ClassName()+"::DNearField");
            return sum;
        }
        
        
        
//        Real NearField_Blocked( const Mesh_T & M ) const
//        {
//            ptic(ClassName()+"::NearField_Blocked");
//
//            auto & bct  = M.GetObstacleBlockClusterTree();
//            auto & near = bct.Near();
//
//            if( near.BlockNonzeroCount() <= 0 )
//            {
////                wprint(ClassName()+"::NearField_Blocked: no near field blocks found. Returning 0." );
//                ptoc(ClassName()+"::NearField_Blocked");
//                return static_cast<Real>(0);
//            }
//
//            Real sum = static_cast<Real>(0);
//
////            print("near.BlockJobPtr() = " + near.BlockJobPtr().ToString() );
////            print("near.b_outer = " + near.b_outer.ToString() );
////            print("near.b_inner = " + near.b_inner.ToString() );
////            print("near.b_row_ptr = " + near.b_row_ptr.ToString() );
////            print("near.b_col_ptr = " + near.b_col_ptr.ToString() );
//
//            auto & job_ptr = near.BlockJobPtr();
//
//            const Int thread_count = job_ptr.Size()-1;
//
//            #pragma omp parallel for num_threads( thread_count ) reduction( + : sum )
//            for( Int thread = 0; thread < thread_count; ++thread )
//            {
//                Int const * restrict const b_row_ptr = near.b_row_ptr.data();
//                Int const * restrict const b_col_ptr = near.b_col_ptr.data();
//                Int const * restrict const b_outer   = near.b_outer.data();
//                Int const * restrict const b_inner   = near.b_inner.data();
//
//                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
//
//                // This looks massive, but is only exchange of pointers.
//                (void)N->LoadNearField(
//                    bct.GetS().PrimitiveNearFieldData(),
//                    bct.GetT().PrimitiveNearFieldData()
//                );
////                (void)N->LoadDNearField(
////                    bct.GetS().ThreadPrimitiveDNearFieldData()[thread],
////                    bct.GetT().ThreadPrimitiveDNearFieldData()[thread]
////                );
//                (void)N->LoadPrimitiveSerializedData(
//                    bct.GetS().PrimitiveSerializedData(),
//                    bct.GetT().PrimitiveSerializedData()
//                );
//
//                Real local_sum = static_cast<Real>(0);
//
//                const Int b_i_begin = job_ptr[thread  ];
//                const Int b_i_end   = job_ptr[thread+1];
//
//
//                for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
//                {
//                    const Int i_begin = b_row_ptr[b_i  ];
//                    const Int i_end   = b_row_ptr[b_i+1];
//
//                    const Int k_begin = b_outer[b_i  ];
//                    const Int k_end   = b_outer[b_i+1];
//
//                    for( Int k = k_begin; k < k_end; ++k )
//                    {
//                        const Int b_j = b_inner[k];
//
//                        // we are in block (b_i, b_j)
//
//                        const Int j_begin = b_col_ptr[b_j  ];
//                        const Int j_end   = b_col_ptr[b_j+1];
//
//                        for( Int i = i_begin; i < i_end; ++i )
//                        {
//                            N->LoadS(i);
//
//                            for( Int j = j_begin; j < j_end; ++j )
//                            {
//                                N->LoadT(j);
//
//                                local_sum +=  N->Energy();
//
//                            } // for( Int j = j_begin; j < j_end; ++j )
//
//                        } // for( Int i = i_begin; i < i_end; ++i )
//
//                    } // for( Int k = k_begin; k < k_end; ++k )
//
//                } // for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
//
//                sum += local_sum;
//
//            } // #pragma omp parallel
//
//            ptoc(ClassName()+"::NearField_Blocked");
//            return sum;
//        }
        
        
//        Real DNearField_Blocked( const Mesh_T & M ) const
//        {
//            ptic(ClassName()+"::DNearField_Blocked");
//
//            auto & bct  = M.GetObstacleBlockClusterTree();
//            auto & near = bct.Near();
//
//            if( near.BlockNonzeroCount() <= 0 )
//            {
////                wprint(ClassName()+"::DNearField_Blocked: no near field blocks found. Returning 0." );
//                ptoc(ClassName()+"::DNearField_Blocked");
//                return static_cast<Real>(0);
//            }
//
//            Real sum = static_cast<Real>(0);
//
//            auto & job_ptr = near.BlockJobPtr();
//
//            const Int thread_count = job_ptr.Size()-1;
//
//            #pragma omp parallel for num_threads( thread_count ) reduction( + : sum )
//            for( Int thread = 0; thread < thread_count; ++thread )
//            {
//                Int const * restrict const b_row_ptr = near.b_row_ptr.data();
//                Int const * restrict const b_col_ptr = near.b_col_ptr.data();
//                Int const * restrict const b_outer   = near.b_outer.data();
//                Int const * restrict const b_inner   = near.b_inner.data();
//
//                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
//
//                // This looks massive, but is only exchange of pointers.
//                (void)N->LoadNearField(
//                    bct.GetS().PrimitiveNearFieldData(),
//                    bct.GetT().PrimitiveNearFieldData()
//                );
//                (void)N->LoadDNearField(
//                    bct.GetS().ThreadPrimitiveDNearFieldData()[thread],
//                    bct.GetT().ThreadPrimitiveDNearFieldData()[thread]
//                );
//                (void)N->LoadPrimitiveSerializedData(
//                    bct.GetS().PrimitiveSerializedData(),
//                    bct.GetT().PrimitiveSerializedData()
//                );
//
//                Real local_sum = static_cast<Real>(0);
//
//                const Int b_i_begin = job_ptr[thread  ];
//                const Int b_i_end   = job_ptr[thread+1];
//
//                for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
//                {
//                    const Int i_begin = b_row_ptr[b_i  ];
//                    const Int i_end   = b_row_ptr[b_i+1];
//
//                    const Int k_begin = b_outer[b_i  ];
//                    const Int k_end   = b_outer[b_i+1];
//
//                    for( Int k = k_begin; k < k_end; ++k )
//                    {
//                        const Int b_j = b_inner[k];
//
//                        // we are in block (b_i, b_j)
//                        const Int j_begin = b_col_ptr[b_j  ];
//                        const Int j_end   = b_col_ptr[b_j+1];
//
//                        for( Int i = i_begin; i < i_end; ++i )
//                        {
//                            N->LoadS(i);
//                            N->CleanseDBufferS();
//
//                            for( Int j = j_begin; j < j_end; ++j )
//                            {
//                                N->LoadT(j);
//                                N->CleanseDBufferT();
//
//                                local_sum += N->DEnergy();
//
//                                N->WriteDBufferT();
//
//                            } // for( Int j = j_begin; j < j_end; ++j )
//
//                            N->WriteDBufferS();
//
//                        } // for( Int i = i_begin; i < i_end; ++i )
//
//
//                    } // for( Int k = k_begin; k < k_end; ++k )
//
//                } // for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
//
//                sum += local_sum;
//
//            } // #pragma omp parallel
//
//            ptoc(ClassName()+"::DNearField_Blocked");
//            return sum;
//        }
        
        Real FarField( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::FarField");

            auto & bct = M.GetObstacleBlockClusterTree();
            auto & far = bct.Far();

            if( far.NonzeroCount() <= 0 )
            {
//                wprint(ClassName()+"::FarField: no admissible blocks found. Returning 0." );
                ptoc(ClassName()+"::FarField");
                return static_cast<Real>(0);
            }
            
            Real sum = static_cast<Real>(0);
            
            auto & job_ptr = far.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count ) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const outer = far.Outer().data();
                Int const * restrict const inner = far.Inner().data();
                
                std::unique_ptr<F_Kernel_T> F = F_ker->Clone();

                (void)F->LoadFarField(
                    bct.GetS().ClusterFarFieldData(),
                    bct.GetT().ClusterFarFieldData()
                );
//                (void)F->LoadDFarField(
//                    bct.GetS().ThreadClusterDFarFieldData()[thread],
//                    bct.GetT().ThreadClusterDFarFieldData()[thread]
//                );

                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                for( Int i = i_begin; i < i_end; ++i )
                {
                    F->LoadS(i);
                    
                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];

                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];

                        F->LoadT(j);
                    
                        local_sum += F->Energy();

                    } // for( Int k = k_begin; k < k_end; ++k )

                } // for( Int i = i_begin; i < i_end; ++i )

                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::FarField");
            return sum;
        }
        
        Real DFarField( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::DFarField");

            auto & bct = M.GetObstacleBlockClusterTree();
            auto & far = bct.Far();

            if( far.NonzeroCount() <= 0 )
            {
//                wprint(ClassName()+"::DFarField: no admissible blocks found. Returning 0." );
                ptoc(ClassName()+"::DFarField");
                return static_cast<Real>(0);
            }
          
            Real sum = static_cast<Real>(0);
            
            auto & job_ptr = far.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads( thread_count ) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const outer = far.Outer().data();
                Int const * restrict const inner = far.Inner().data();
                
                std::unique_ptr<F_Kernel_T> F = F_ker->Clone();
                
                (void)F->LoadFarField(
                    bct.GetS().ClusterFarFieldData(),
                    bct.GetT().ClusterFarFieldData()
                );
                (void)F->LoadDFarField(
                    bct.GetS().ThreadClusterDFarFieldData()[thread],
                    bct.GetT().ThreadClusterDFarFieldData()[thread]
                );
                
                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    F->LoadS(i);
                    F->CleanseDBufferS();
                    
                    const Int k_begin = outer[i  ];
                    const Int k_end   = outer[i+1];

                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];

                        F->LoadT(j);
                        F->CleanseDBufferT();
                        
                        local_sum += F->DEnergy();

                        F->WriteDBufferT();
                        
                    } // for( Int k = k_begin; k < k_end; ++k )

                    F->WriteDBufferS();
                    
                } // for( Int i = i_begin; i < i_end; ++i )

                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::DFarField");
            return sum;
        }
        
        
    public:
        
        virtual std::string Stats() const override
        {
            std::stringstream s;
            
            s << ClassName() <<": \n\n";
            if( N_ker != nullptr )
            {
                s <<  "          NearFieldKernel = " << N_ker->Stats() << "\n\n";
            }
            if( F_ker != nullptr )
            {
                s <<  "          FarFieldKernel  = " << F_ker->Stats() << "\n\n";
            }
            return s.str();
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM1)+","+"<"+ToString(DOM_DIM2)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef BASE
#undef CLASS
