#pragma once

#define CLASS Energy_BH_BCT

namespace Repulsion
{
    template<mint DOM_DIM, mint AMB_DIM, mint DEGREE, typename Real, typename Int>
    class CLASS
    {
    public:
        
        using MESH_T    = SimplicialMesh<DOM_DIM,AMB_DIM,Real,Int>;
        using N_Kernel_T = Energy_NearFieldKernel    <DOM_DIM,AMB_DIM,Real,Int>;
        using F_Kernel_T = Energy_FarFieldKernel_FMM <DOM_DIM,AMB_DIM,DEGREE,Real,Int>;

        CLASS( const N_Kernel_T & N, const F_Kernel_T & F, Real weight_ = static_cast<Real>(1))
        :
            N_ker(N.Clone()),
            F_ker(F.Clone()),
            weight(weight_)
        {}

        virtual ~CLASS() = default;

        virtual std::string Stats() const override
        {
            std::stringstream s;
            
            s << ClassName() << ": \n";
            
            if( N_ker != nullptr )
            {
                s << "          NearFieldKernel = " << N_ker->Stats() << "\n";
            }
            
            if( F_ker != nullptr )
            {
                s << "          FarFieldKernel  = " << F_ker->Stats() << "\n";
            }
            return s.str();
        }
        
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

        
        // Returns the current value of the energy.
        Real Value( const MESH_T & M )
        {
            ptic(ClassName()+"::Value");
            
            const Real near = NearField(M);
            const Real  far =  FarField(M);
        
//            valprint("near",near);
//            valprint("far",far);
            
            ptoc(ClassName()+"::Value");
            return weight * (near+far);
        }
        
        // Returns the current differential of the energy, stored in the given
        // V x AMB_DIM matrix, where each row holds the differential (a AMB_DIM-vector) with
        // respect to the corresponding vertex.
        Real Differential( const MESH_T & M, Real * output )
        {
            ptic(ClassName()+"::Differential");
            
            M.GetClusterTree().CleanseD();
            
            const Real near = DNearField(M);
            const Real  far =  DFarField(M);
            
//            valprint("near",near);
//            valprint("far",far);
            
            M.Assemble_ClusterTree_Derivatives( output, weight );
            
            ptoc(ClassName()+"::Differential");
            return weight * (near+far);
        }
        
        Real GetWeight()  const
        {
            return weight;
        }

        
    protected:
        
        mutable std::unique_ptr<N_Kernel_T> N_ker;
        mutable std::unique_ptr<F_Kernel_T> F_ker;
        
        const Real weight = static_cast<Real>(1);
        
        
        Real NearField( const MESH_T & M ) const
        {
            ptic(ClassName()+"::NearField");
            
//            print(N_ker->ClassName());

            auto & bct  = M.GetBlockClusterTree();
            auto & near = bct.Near();
            
            if( near.nnz <= 0 )
            {
                wprint(ClassName()+"::NearField: no near field blocks found. Returning 0." );
                ptoc(ClassName()+"::NearField");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
            
            const thread_count = near.triangular_block_job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const b_row_ptr = near.b_row_ptr.data();
                Int const * restrict const b_col_ptr = near.b_col_ptr.data();
                Int const * restrict const b_outer   = near.b_outer.data();
                Int const * restrict const b_inner   = near.b_inner.data();
                
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                N->LoadClusterTrees( bct.GetS(), bct.GetT(), thread );
                
                Real local_sum = static_cast<Real>(0);

                const Int b_i_begin = near.triangular_block_job_ptr[ thread ];
                const Int b_i_end   = near.triangular_block_job_ptr[ thread + 1 ];

                for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
                {
                    const Int i_begin = b_row_ptr[b_i];
                    const Int i_end   = b_row_ptr[b_i+1];
                    
                          Int k_begin = b_outer[b_i];
                    const Int k_end   = b_outer[b_i+1];
                    
                    // skip all blocks below the diagonal
                    while( k_begin < k_end && b_inner[k_begin] < b_i )
                    {
                        ++k_begin;
                    }
                     
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int b_j = b_inner[k];
                        
                        // we are in block (b_i, b_j)
                        
                        const Int j_end = b_col_ptr[b_j+1];

                        if( b_i != b_j )
                        {
                            // off-diagonal block
                            
                            const Int j_begin = b_col_ptr[b_j];

                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                N->LoadS(i);

                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    N->LoadT(j);
                                
                                    local_sum += N->Energy();
                                    
                                } // for( Int j = j_begin; j < j_end; ++j )
                                
                            } // for( Int i = i_begin; i < i_end; ++i )
                        }
                        else
                        {
                            // if b_i == b_j, we loop only over the upper triangular block, diagonal excluded
                            for( Int i = i_begin; i < i_end ; ++i )
                            {
                                const Int j_begin = i+1;
                                
                                N->LoadS(i);
                                
                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    N->LoadT(j);
                                    
                                    local_sum += N->Energy();
                                    
                                } // for( Int j = j_begin; j < j_end; ++j )
                                
                            } // for( Int i = i_begin; i < i_end ; ++i )
                        
                        } // if( b_i != b_j )
                     
                    } // for( Int k = k_begin; k < k_end; ++k )
                
                } // for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
                
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::NearField");
            return sum;
        }
        
        Real DNearField( const MESH_T & M ) const
        {
            ptic(ClassName()+"::DNearField");

            auto & bct  = M.GetBlockClusterTree();
            auto & near = bct.Near();
            
            if( near.nnz <= 0 )
            {
                wprint(ClassName()+"::DNearField: no near field blocks found. Returning 0." );
                ptoc(ClassName()+"::DNearField");
                return static_cast<Real>(0);
            }

            Real sum = static_cast<Real>(0);
            
            const thread_count = near.triangular_block_job_ptr.Size()-1;
                
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Int const * restrict const b_row_ptr = near.b_row_ptr.data();
                Int const * restrict const b_col_ptr = near.b_col_ptr.data();
                Int const * restrict const b_outer   = near.b_outer.data();
                Int const * restrict const b_inner   = near.b_inner.data();
                
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                N->LoadClusterTrees( bct.GetS(), bct.GetT(), thread );
                
                Real local_sum = static_cast<Real>(0);

                const Int b_i_begin = near.triangular_block_job_ptr[ thread ];
                const Int b_i_end   = near.triangular_block_job_ptr[ thread + 1 ];

                for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
                {
                    const Int i_begin = b_row_ptr[b_i];
                    const Int i_end   = b_row_ptr[b_i+1];
                    
                          Int k_begin = b_outer[b_i];
                    const Int k_end   = b_outer[b_i+1];
                    
                    // skip all blocks below the diagonal
                    while( k_begin < k_end && b_inner[k_begin] < b_i )
                    {
                        ++k_begin;
                    }
                     
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int b_j = b_inner[k];
                        
                        // we are in block (b_i, b_j)
                        
                        const Int j_end = b_col_ptr[b_j+1];

                        if( b_i != b_j )
                        {
                            // off-diagonal block
                            
                            const Int j_begin = b_col_ptr[b_j];

                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                N->LoadS(i);
                                N->CleanseDBufferS();
                                
                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    N->LoadT(j);
                                    N->CleanseDBufferT();
                                    
                                    local_sum += N->DEnergy();
                                    
                                    N->WriteDBufferT();
                                    
                                } // for( Int j = j_begin; j < j_end; ++j )
                                
                                N->WriteDBufferS();
                                
                            } // for( Int i = i_begin; i < i_end; ++i )
                        }
                        else
                        {
                            // if b_i == b_j, we loop only over the upper triangular block, diagonal excluded
                            for( Int i = i_begin; i < i_end ; ++i )
                            {
                                const Int j_begin = i+1;
                                
                                N->LoadS(i);
                                N->CleanseDBufferS();
                                
                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    N->LoadT(j);
                                    N->CleanseDBufferT();
                                    
                                    local_sum += N->DEnergy();
                                    
                                    N->WriteDBufferT();
                                    
                                } // for( Int j = j_begin; j < j_end; ++j )
                                
                                N->WriteDBufferS();
                                
                            } // for( Int i = i_begin; i < i_end ; ++i )
                        
                        } // if( b_i != b_j )
                     
                    } // for( Int k = k_begin; k < k_end; ++k )
                
                } // for( Int b_i = b_i_begin; b_i < b_i_end; ++ b_i)
                
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::DNearField");
            return sum;
        }
        
        Real FarField( const MESH_T & M ) const
        {
            ptic(ClassName()+"::FarField");

            auto & bct = M.GetBlockClusterTree();
            auto & far = bct.Far();

            if( far.nnz <= 0 )
            {
//                wprint(ClassName()+"::FarField: no admissible blocks found. Returning 0." );
                ptoc(ClassName()+"::FarField");
                return static_cast<Real>(0);
            }
          
            Real sum = static_cast<Real>(0);
            
            const thread_count = far.JobPtr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<F_Kernel_T> F = F_ker->Clone();
                F->LoadClusterTrees( bct.GetS(), bct.GetT(), thread );
                
                Int const * restrict const b_outer = far.b_outer.data();
                Int const * restrict const b_inner = far.b_inner.data();
                
                Int const * restrict const C_begin = bct.GetT().ClusterBegin().data();
                Int const * restrict const C_end   = bct.GetT().ClusterEnd().data();
                
                Real local_sum = static_cast<Real>(0);

                const Int b_i_begin = far.JobPtr()[thread];
                const Int b_i_end   = far.JobPtr()[thread+1];
                
                for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
                {
                    // Load cluster b_i;
                    F->LoadS(b_i);
                    
                    const Int k_begin = b_outer[b_i];
                    const Int k_end   = b_outer[b_i+1];

                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int b_j = b_inner[k];
                        
                        // we are in block {b_i, b_j}
                        
                        const Int j_begin = C_begin[b_j];
                        const Int j_end   = C_end  [b_j];
                        
                        for( Int j = j_begin; j < j_end; ++j )
                        {
                            // Load primitive j;
                            F->LoadT(j);

                            local_sum += F->Energy();
                            
                        } // for( Int j = j_begin; j < j_end; ++j )
                        
                    } // for( Int k = k_begin; k < k_end; ++k )
                    
                } // for( Int b_i = i_begin; b_i < i_end; ++b_i )
                
                sum += local_sum;
                
            } // #pragma omp parallel
            
            
            ptoc(ClassName()+"::FarField");
            return sum;
        }
        
        Real DFarField( const MESH_T & M ) const
        {
            ptic(ClassName()+"::DFarField");

            auto & bct = M.GetBlockClusterTree();
            auto & far = bct.Far();

            if( far.nnz <= 0 )
            {
//                wprint(ClassName()+"::DFarField: no admissible blocks found. Returning 0." );
                ptoc(ClassName()+"::DFarField");
                return static_cast<Real>(0);
            }
          
            Real sum = static_cast<Real>(0);
            
            const thread_count = far.JobPtr.Size()-1;
            
            #pragma omp parallel for  num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<F_Kernel_T> F = F_ker->Clone();
                F->LoadClusterTrees( bct.GetS(), bct.GetT(), thread );
                
                Int const * restrict const b_outer = far.b_outer.data();
                Int const * restrict const b_inner = far.b_inner.data();
                
                Int const * restrict const C_begin = bct.GetT().ClusterBegin().data();
                Int const * restrict const C_end   = bct.GetT().ClusterEnd().data();
                
                Real local_sum = static_cast<Real>(0);

                const Int b_i_begin = far.JobPtr()[thread];
                const Int b_i_end   = far.JobPtr()[thread+1];
                
                for( Int b_i = b_i_begin; b_i < b_i_end; ++b_i )
                {
                    // Load cluster b_i;
                    F->LoadS(b_i);
                    F->CleanseDBufferS();
                    
                    const Int k_begin = b_outer[b_i];
                    const Int k_end   = b_outer[b_i+1];

                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int b_j = b_inner[k];
                        
                        // we are in block {b_i, b_j}
                        
                        const Int j_begin = C_begin[b_j];
                        const Int j_end   = C_end  [b_j];
                        
                        for( Int j = j_begin; j < j_end; ++j )
                        {
                            // Load primitive j;
                            F->LoadT(j);
                            F->CleanseDBufferT();
                            
                            local_sum += F->DEnergy();
                            
                            F->WriteDBufferT();
                        }
                    }
                    
                    F->WriteDBufferS();
                }
                
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::DFarField");
            return sum;
        }
        
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef CLASS
