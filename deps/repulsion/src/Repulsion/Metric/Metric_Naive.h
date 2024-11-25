#pragma once

#define CLASS Metric_Naive

namespace Repulsion
{
    template<int DOM_DIM, int AMB_DIM, typename Real, typename Int, typename SReal, typename ExtReal>
    class CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );
        ASSERT_FLOAT(SReal  );
        ASSERT_FLOAT(ExtReal);
        
        using ClusterTree_T = ClusterTree            <AMB_DIM,Real,Int,SReal,ExtReal>;
        using N_Kernel_T    = Metric_NearFieldKernel <DOM_DIM,DOM_DIM,AMB_DIM,Real,Int,SReal>;
        
    public:
        
        CLASS() {}
        
        CLASS( const ClusterTree_T & S_, const ClusterTree_T & T_, const N_Kernel_T & N, const Real weight_ = static_cast<Real>(1) )
        :
            S( &S_ ),
            T( &T_ ),
            N_ker(N.Clone()),
            weight( weight_)
        {
//            ptic(ClassName());
//            if( bct.get() == nullptr )
//            {
//                eprint(ClassName()+" : BlockClusterTree not initialized.");
//                ptoc(ClassName());
//                return;
//            }
//            
////            this->RequireMetrics();
//            
//            ptoc(ClassName());
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :
            S( other.S ),
            T( other.T ),
            N_ker( other.N_ker->Clone() ),
            weight ( other.weight )
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept 
        :
            S( other.S ),
            T( other.T ),
            N_ker( other.N_ker->Clone() ),
            weight ( other.weight )
        {}

        virtual ~CLASS() = default;
        
    protected:
        
        const ClusterTree_T * const S;
        const ClusterTree_T * const T;
        
        std::unique_ptr<N_Kernel_T> N_ker;

        const Real weight;
        
        mutable std::map<KernelType, Tensor1<Real,Int>> kernel_matrices;
        
        mutable bool metrics_initialized = false;
        
    public:

        std::string Stats() const
        {
            std::stringstream s;
            
            s << ClassName() << ": \n\n";
            if( N_ker != nullptr )
            {
                s << "NearFieldKernel = " << N_ker->Stats() <<"\n";
            }
            return s.str();
        }
        
        const ClusterTree_T & GetS() const
        {
            return S;
        }
        
        const ClusterTree_T & GetT() const
        {
            return T;
        }
        
        N_Kernel_T & NearFieldKernel()
        {
            return *N_ker;
        }
        
        const N_Kernel_T & NearFieldKernel() const
        {
            return *N_ker;
        }
        
        Real GetWeight()  const
        {
            return weight;
        }
        
        const std::map<KernelType, Tensor1<Real,Int>> & KernelMatrices() const
        {
            return kernel_matrices;
        }

        void RequireMetrics() const
        {
            if( metrics_initialized )
            {
                return;
            }
            
            ptic(ClassName()+"::RequireMetrics");
            
            NearField();
            
            metrics_initialized = true;
                
            ptoc(ClassName()+"::RequireMetrics");
            
        } // RequireMetrics
        
    protected:
        
        void NearField() const
        {
            ptic(ClassName()+"::NearField");
            
            const Int thread_count = std::min(S->ThreadCount(),T->ThreadCount());
            
            Int rows = S->PrimitiveCount();
            Int cols = T->PrimitiveCount();
            
            N_ker->AllocateValueBuffers( kernel_matrices, rows * cols );
            
            auto job_ptr = BalanceWorkLoad<Int>( rows, thread_count );
        
            #pragma omp parallel for num_threads(thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                std::unique_ptr<N_Kernel_T> N = N_ker->Clone();
                N->LoadValueBuffers( kernel_matrices );
                
                // This looks massive, but is only exchange of pointers.
                (void)N->LoadNearField(
                    S->PrimitiveNearFieldData(),
                    T->PrimitiveNearFieldData()
                );
                (void)N->LoadPrimitiveSerializedData(
                    S->PrimitiveSerializedData(),
                    T->PrimitiveSerializedData()
                );
                
                const Real * restrict const b =  T->PrimitiveNearFieldData().data();
                const Int near_dim = T->NearDim();
                
                const Int i_begin = job_ptr[thread];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    N->LoadS(i);
                    for( Int j = 0; j < cols; ++j )
                    {
                        if(i!=j)
                        {
                            N->LoadT(j);
                            N->Metric( cols * i + j);
                        }
                    }

                    const Real a_inv = static_cast<Real>(1) / S->PrimitiveNearFieldData()(i,0);
                    
                    for( auto & A : kernel_matrices )
                    {
                        Real * restrict const A_i = A.second.data() + cols * i;
                        
                        Real sum = static_cast<Real>(0);
                        
                        for( Int j = 0; j < cols; ++j )
                        {
                            sum += A_i[j] * b[near_dim * j];
                        }
                        
                        A_i[i] -= sum * a_inv;
                    }
                }
                
            } // #pragma omp parallel
            


            ptoc(ClassName()+"::NearField");
        }
        

    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
    
}// namespace Repulsion

#undef CLASS
