#pragma once

#define CLASS Energy_BH

namespace Repulsion
{
    template<mint DOM_DIM, mint AMB_DIM, mint DEGREE, typename Real, typename Int>
    class CLASS
    {
    protected:
        
        using MESH_T    = SimplicialMesh<DOM_DIM,AMB_DIM,Real,Int>;
        using N_Kernel_T = Energy_NearFieldKernel   <DOM_DIM,AMB_DIM,Real,Int>;
        using F_Kernel_T = Energy_FarFieldKernel_BH <DOM_DIM,AMB_DIM,DEGREE,Real,Int>;
        
        mutable std::unique_ptr<N_Kernel_T> N_ker;
        mutable std::unique_ptr<F_Kernel_T> F_ker;
        
        const Real weight =  static_cast<Real>(1);

        
    public:
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
            
            auto & T = M.GetClusterTree();

            Real sum = 0;
            
            const thread_count = T.ThreadCount();
            
            auto job_ptr = BalanceWorkLoad<Int>( T.LeafClusterCount(), thread_count );

            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                N->LoadClusterTrees( T, T, thread );
                
                std::unique_ptr<F_Kernel_T> F = F_ker->Clone();
                F->LoadClusterTrees( T, T, thread );
                
                const Real theta2 = F->GetTheta() * F->GetTheta();
                
                const Int depth = T.Depth();

                const Int  * restrict const C_left  = T.ClusterLeft().data();
                const Int  * restrict const C_right = T.ClusterRight().data();
                const Int  * restrict const C_begin = T.ClusterBegin().data();
                const Int  * restrict const C_end   = T.ClusterEnd().data();
                
                const Int  * restrict const leaf = T.LeafClusters().data();
                      Real * restrict const C_serialized = T.ClusterSerializedData().data();

                auto P = T.ClusterPrototype().Clone();
                auto Q = T.ClusterPrototype().Clone();
                
                auto gjk = GJK_Algorithm<AMB_DIM,Real,Int>();

                Int stack_array[2 * depth + 1];

                Int k_begin = job_ptr[thread];
                Int k_end   = job_ptr[thread+1];
        
                Real local_sum = 0.;
                
                // Go through leaf clusters.
                for( Int k = k_begin; k < k_end; ++k )
                {
                    //Reset stack.
                    stack_array[0] = 0;
                    Int * stack = stack_array;

                    const Int b_j = leaf[k];
                    const Int j_begin = C_begin[b_j];
                    const Int j_end   = C_end[b_j];
                    
                    Q->SetPointer( C_serialized, b_j );

                    while( stack >= stack_array )
                    {
                        Int C = *(stack--);
                        
                        P->SetPointer( C_serialized, C );

                        bool separatedQ = ( (b_j == C) ) ? false : gjk.MultipoleAcceptanceCriterion( *P, *Q, theta2 );

                        if( separatedQ )
                        {
                            F->LoadS(C);
                        
                            for( Int j = j_begin; j < j_end; ++j )
                            {
                                F->LoadT(j);

                                local_sum += F->Energy();
                            
                            } // for( Int j = j_begin; j < j_end; ++i )

                        }
                        else
                        {
                            const Int left  = C_left[C];
                            const Int right = C_right[C];

                            if( left >= 0 /*&& right >= 0*/ )
                            {
                                *(++stack) = right;
                                *(++stack) = left;
                            }
                            else
                            {
                                // near field loop
                                const Int i_begin = C_begin[C];
                                const Int i_end   = C_end[C];

                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    N->LoadT(j);
                                
                                    for( Int i = i_begin; i < i_end; ++i )
                                    {
                                        // Near field kernels are symmetrized.
                                        // So we have to make sure that we sum each unordered pair of primitives only once.
                                        if( i < j )
                                        {
                                            N->LoadS(i);
                                            local_sum += N->Energy();
                                        }
                                    }

                                } // for( Int j = j_begin; j < j_end; ++j )

                            } // if( left >= 0 /*&& right >= 0*/ )

                        } // if( separatedQ )

                    } // while( stack >= stack_array )

                } // for( Int k = k_begin; k < k_end; ++k )
                
                sum += local_sum;

            } // #pragma omp parallel
            
            ptoc(ClassName()+"::Value");
            return weight * sum;
        }
        
        // Returns the current differential of the energy, stored in the given
        // V x AMB_DIM matrix, where each row holds the differential (a AMB_DIM-vector) with
        // respect to the corresponding vertex.
        Real Differential( const MESH_T & M, Real * output, bool addTo = false )
        {
            ptic(ClassName()+"::Differential");
            
            M.GetClusterTree().CleanseD();
            
            auto & T = M.GetClusterTree();

            Real sum = 0;
            
            const Int thread_count = T.ThreadCount();
            
            auto job_ptr = BalanceWorkLoad<Int>( T.LeafClusterCount(), thread_count );

            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                N->LoadClusterTrees( T, T, thread );
                
                std::unique_ptr<F_Kernel_T> F = F_ker->Clone();
                F->LoadClusterTrees( T, T, thread );
                
                const Real theta2 = F->GetTheta() * F->GetTheta();
                
                const Int depth = T.Depth();

                const Int  * restrict const C_left  = T.ClusterLeft().data();
                const Int  * restrict const C_right = T.ClusterRight().data();
                const Int  * restrict const C_begin = T.ClusterBegin().data();
                const Int  * restrict const C_end   = T.ClusterEnd().data();
                
                const Int  * restrict const leaf = T.LeafClusters().data();
                      Real * restrict const C_serialized = T.ClusterSerializedData().data();

                auto P = T.ClusterPrototype().Clone();
                auto Q = T.ClusterPrototype().Clone();
                
                auto gjk = GJK_Algorithm<AMB_DIM,Real,Int>();

                Int stack_array[2 * depth + 1];

                Int k_begin = job_ptr[thread];
                Int k_end   = job_ptr[thread+1];
                
                Real local_sum = 0.;
                
                // Go through leaf clusters.
                for( Int k = k_begin; k < k_end; ++k )
                {
                    //Reset stack.
                    stack_array[0] = 0;
                    Int * stack = stack_array;

                    const Int b_j = leaf[k];
                    const Int j_begin = C_begin[b_j];
                    const Int j_end   = C_end[b_j];
                    
                    Q->SetPointer( C_serialized, b_j );

                    while( stack >= stack_array )
                    {
                        Int C = *(stack--);
                        
                        P->SetPointer( C_serialized, C );

                        bool separatedQ = ( (b_j == C) ) ? false : gjk.MultipoleAcceptanceCriterion( *P, *Q, theta2 );

                        if( separatedQ )
                        {
                            F->LoadS(C);
                        
                            for( Int j = j_begin; j < j_end; ++j )
                            {
                                F->LoadT(j);

                                local_sum += F->DEnergy();
                            
                            } // for( Int j = j_begin; j < j_end; ++i )

                        }
                        else
                        {
                            const Int left  = C_left[C];
                            const Int right = C_right[C];

                            if( left >= 0 /*&& right >= 0*/ )
                            {
                                *(++stack) = right;
                                *(++stack) = left;
                            }
                            else
                            {
                                // near field loop
                                const Int i_begin = C_begin[C];
                                const Int i_end   = C_end[C];

                                for( Int j = j_begin; j < j_end; ++j )
                                {
                                    N->LoadT(j);
                                
                                    for( Int i = i_begin; i < i_end; ++i )
                                    {
                                        // Near field kernels are symmetrized.
                                        // So we have to make sure that we sum each unordered pair of primitives only once.
                                        if( i < j )
                                        {
                                            N->LoadS(i);
                                            local_sum += N->DEnergy();
                                        }
                                    }

                                } // for( Int j = j_begin; j < j_end; ++j )

                            } // if( left >= 0 /*&& right >= 0*/ )

                        } // if( separatedQ )

                    } // while( stack >= stack_array )

                } // for( Int k = k_begin; k < k_end; ++k )
                
                sum += local_sum;

            } // #pragma omp parallel
            
            M.Assemble_ClusterTree_Derivatives( output, weight, addTo );
            
            ptoc(ClassName()+"::Differential");
            return weight * sum;
        }
        
        Real GetWeight()  const
        {
            return weight;
        }
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef CLASS
