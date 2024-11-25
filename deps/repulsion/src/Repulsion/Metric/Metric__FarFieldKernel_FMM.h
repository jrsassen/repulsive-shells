#pragma once

#define CLASS Metric_FarFieldKernel_FMM
#define BASE  FarFieldKernelBase_FMM<AMB_DIM,DEGREE,Real,Int>

namespace Repulsion
{
    template<int AMB_DIM, int DEGREE, typename Real, typename Int>
    class CLASS : public BASE
    {
        public :
        
        CLASS() : BASE() {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}

        // Move constructor 
        CLASS( CLASS && other ) noexcept : BASE(other) {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        
    public:
    
        virtual void AllocateValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & far_values, const Int nnz ) = 0;
        
        virtual void LoadValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & far_values ) = 0;
        
        virtual void Metric( const Int ptr ) = 0;
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+ToString(DEGREE)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };
    
} // namespace Repulsion

#undef BASE
#undef CLASS
