#pragma once

#define CLASS Energy_Naive
#define BASE  Energy_Restricted<DOM_DIM,AMB_DIM,Real,Int,SReal,ExtReal>

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS : public BASE
    {
    public:
        
        using MeshBase_T = typename BASE::MeshBase_T;
        using Mesh_T     = typename BASE::Mesh_T;
        using N_Kernel_T = Energy_NearFieldKernel<DOM_DIM,DOM_DIM,AMB_DIM,Real,Int,SReal>;
        
        explicit CLASS(
            const N_Kernel_T & N,
            ExtReal weight_ = static_cast<ExtReal>(1)
        )
        :   BASE(weight_)
        ,   N_ker(N.Clone())
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
        
        // Returns the current value of the energy.
        ExtReal Value( const Mesh_T & M ) override
        {
            ptic(ClassName()+"::Value");
            
            const ExtReal near = NearField(M);
            
            ptoc(ClassName()+"::Value");
            return weight * static_cast<ExtReal>(near);
        }
        
        // Returns the current differential of the energy, stored in the given
        // V x AMB_DIM matrix, where each row holds the differential (a AMB_DIM-vector) with
        // respect to the corresponding vertex.
        ExtReal Differential( const Mesh_T & M, ExtReal * output, bool addTo = false )
        {
            ptic(ClassName()+"::Differential");
            
            M.GetClusterTree().CleanseDerivativeBuffers();
            
            const ExtReal near = DNearField(M);
            
            M.Assemble_ClusterTree_Derivatives( output, weight, addTo );
            
            ptoc(ClassName()+"::Differential");
            return weight * static_cast<ExtReal>(near);
        }
        
    protected:
        
        using BASE::weight;
        
        mutable std::unique_ptr<N_Kernel_T> N_ker;
                
        Real NearField( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::NearField");

            auto & S = M.GetClusterTree();
            Int n = S.PrimitiveCount();

            Real sum = static_cast<Real>(0);
            
            Tensor1<Int,Int> costs(n+1);
            for( Int k = 0; k < n+1; ++k )
            {
                costs[k] = (n-k);
            }
            
            auto job_ptr = BalanceWorkLoad<Int>( costs, S.ThreadCount() );
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                
                // This looks massive, but is only exchange of pointers.
                (void)N->LoadNearField(
                    S.PrimitiveNearFieldData(),
                    S.PrimitiveNearFieldData()
                );
//                (void)N->LoadDNearField(
//                    S.ThreadPrimitiveDNearFieldData()[thread],
//                    S.ThreadPrimitiveDNearFieldData()[thread]
//                );
                (void)N->LoadPrimitiveSerializedData(
                    S.PrimitiveSerializedData(),
                    S.PrimitiveSerializedData()
                );
                (void)N->LoadNeighborData(
                    S.PrimitiveNeighborRowPointers(),
                    S.PrimitiveNeighborColumnIndices()
                );
                
                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[ thread ];
                const Int i_end   = job_ptr[ thread + 1 ];

                for( Int i = i_begin; i < i_end; ++i)
                {
                    N->LoadS(i);
                    
                    const Int j_begin = i+1;
                    const Int j_end   = n;

                    for( Int j = j_begin; j < j_end; ++j )
                    {
                        N->LoadT(j);
                                
                        local_sum += N->Energy();
                                    
                    } // for( Int j = j_begin; j < j_end; ++j )
                                
                } // for( Int i = i_begin; i < i_end; ++i )
                
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::NearField");
            return sum;
        }
        
        Real DNearField( const Mesh_T & M ) const
        {
            ptic(ClassName()+"::DNearField");

            auto & S = M.GetClusterTree();
            Int n = S.PrimitiveCount();

            Real sum = static_cast<Real>(0);
            
            Tensor1<Int,Int> costs(n+1);
            for( Int k = 0; k < n+1; ++k )
            {
                costs[k] = (n-k);
            }
            
            auto job_ptr = BalanceWorkLoad<Int>( costs, S.ThreadCount(), true );
            
            const Int thread_count = job_ptr.Size()-1;
            
            #pragma omp parallel for num_threads(thread_count) reduction( + : sum )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                // This looks massive, but is only exchange of pointers.
                (void)N->LoadNearField(
                    S.PrimitiveNearFieldData(),
                    S.PrimitiveNearFieldData()
                );
                (void)N->LoadDNearField(
                    S.ThreadPrimitiveDNearFieldData()[thread],
                    S.ThreadPrimitiveDNearFieldData()[thread]
                );
                (void)N->LoadPrimitiveSerializedData(
                    S.PrimitiveSerializedData(),
                    S.PrimitiveSerializedData()
                );
                (void)N->LoadNeighborData(
                    S.PrimitiveNeighborRowPointers(),
                    S.PrimitiveNeighborColumnIndices()
                );
                
                Real local_sum = static_cast<Real>(0);

                const Int i_begin = job_ptr[ thread ];
                const Int i_end   = job_ptr[ thread + 1 ];

                for( Int i = i_begin; i < i_end; ++i)
                {
                    N->LoadS(i);
                    N->CleanseDBufferS();
                    
                    const Int j_begin = i+1;
                    const Int j_end   = n;

                    for( Int j = j_begin; j < j_end; ++j )
                    {
                        N->LoadT(j);
                        N->CleanseDBufferT();
                                
                        local_sum += N->DEnergy();
                        
                        N->WriteDBufferT();
                        
                    } // for( Int j = j_begin; j < j_end; ++j )
                                
                    N->WriteDBufferS();
                    
                } // for( Int i = i_begin; i < i_end; ++i )
                
                sum += local_sum;
                
            } // #pragma omp parallel
            
            ptoc(ClassName()+"::DNearField");
            return sum;
        }
        
        
    public:
        
        virtual std::string Stats() const override
        {
            std::stringstream s;
            
            s << ClassName() << ": \n\n";
            
            if( N_ker != nullptr )
            {
                s << "          NearFieldKernel = " << N_ker->Stats() << "\n";
            }

            return s.str();
        }
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    };

}// namespace Repulsion

#undef BASE
#undef CLASS
