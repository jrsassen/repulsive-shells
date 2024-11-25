#pragma once

#define CLASS Metric_NearFieldKernel
#define BASE  NearFieldKernelBase<DOM_DIM1,DOM_DIM2,AMB_DIM,Real,Int,SReal>

namespace Repulsion
{
    template<int DOM_DIM1, int DOM_DIM2, int AMB_DIM, typename Real, typename Int, typename SReal>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::S;
        using BASE::S_serialized;
        using BASE::S_near;
        using BASE::S_D_near;
        
        using BASE::T;
        using BASE::T_serialized;
        using BASE::T_near;
        using BASE::T_D_near;
        
        using BASE::DX;
        using BASE::DY;
        
        using BASE::a;
        using BASE::x;
        using BASE::p;
        using BASE::x_buffer;
        
        using BASE::b;
        using BASE::y;
        using BASE::q;
        using BASE::y_buffer;
        
        using BASE::S_ID;
        using BASE::T_ID;
        
        using BASE::tri_i;
        using BASE::tri_j;
        using BASE::lin_k;
                
        using BASE::gjk;
        
    public:
        
        using BASE::NearDimS;
        using BASE::NearDimT;
        
        CLASS() {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}

        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE(other) {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE_FOR_ABSTRACT_CLASS(CLASS)
        

    protected :

        
    public :
        
        virtual void AllocateValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & near_values, const Int nnz ) = 0;
        
        virtual void LoadValueBuffers(
            std::map<KernelType, Tensor1<Real,Int>> & near_values ) = 0;
        
        virtual void Metric ( const Int pos ) = 0;
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM1)+","+ToString(DOM_DIM2)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
  
    };

} // namespace Repulsion

#undef BASE
#undef CLASS
