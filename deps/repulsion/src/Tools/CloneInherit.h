#pragma once



namespace Tools
{
    
    // See // https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/ for explanations
    
    ///////////////////////////////////////////////////////////////////////////////
    
    template <typename T>
    class Abstract
    {
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    
    template <typename T>
    class VirtuallyInheritFrom : virtual public T
    {
        using T::T;
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    
    template <typename DERIVED, typename ... BASES>
    class alignas( 2 * CACHE_LINE_WIDTH ) CloneInherit : public BASES...
    {
    public:
        virtual ~CloneInherit() = default;
        
        std::unique_ptr<DERIVED> Clone() const
        {
            return std::unique_ptr<DERIVED>(static_cast<DERIVED *>(this->Clone_Implementation()));
        }
        
    protected:
        //         desirable, but impossible in C++17
        //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
        // using typename... BASES::BASES;
        
    private:
        virtual CloneInherit * Clone_Implementation() const override
        {
            return new DERIVED(static_cast<const DERIVED & >(*this));
        }
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    
    template <typename DERIVED, typename ... BASES>
    class alignas( 2 * CACHE_LINE_WIDTH ) CloneInherit<Abstract<DERIVED>, BASES...> : public BASES...
    {
    public:
        virtual ~CloneInherit() = default;
        
        std::unique_ptr<DERIVED> Clone() const
        {
            return std::unique_ptr<DERIVED>(static_cast<DERIVED *>(this->Clone_Implementation()));
        }
        
    protected:
        //         desirable, but impossible in C++17
        //         see: http://cplusplus.github.io/EWG/ewg-active.html#102
        // using typename... BASES::BASES;
        
    private:
        virtual CloneInherit * Clone_Implementation() const = 0;
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    
    template <typename DERIVED>
    class alignas( 2 * CACHE_LINE_WIDTH ) CloneInherit<DERIVED>
    {
    public:
        virtual ~CloneInherit() = default;
        
        std::unique_ptr<DERIVED> Clone() const
        {
            return std::unique_ptr<DERIVED>(static_cast<DERIVED *>(this->Clone_Implementation()));
        }
        
    private:
        virtual CloneInherit * Clone_Implementation() const override
        {
            return new DERIVED(static_cast<const DERIVED & >(*this));
        }
    };
    
    ///////////////////////////////////////////////////////////////////////////////
    
    template <typename DERIVED>
    class alignas( 2 * CACHE_LINE_WIDTH ) CloneInherit<Abstract<DERIVED>>
    {
    public:
        virtual ~CloneInherit() = default;
        
        std::unique_ptr<DERIVED> Clone() const
        {
            return std::unique_ptr<DERIVED>(static_cast<DERIVED *>(this->Clone_Implementation()));
        }
        
    private:
        virtual CloneInherit * Clone_Implementation() const = 0;
    };
    
    ///////////////////////////////////////////////////////////////////////////////
} // namespace Tools







//Example:
/////////////////////////////////////////////////////////////////////////////////
//
//class cloneable
//   : public CloneInherit<Abstract<cloneable>>
//{
//};
//
/////////////////////////////////////////////////////////////////////////////////
//
//class foo
//   : public CloneInherit<Abstract<foo>, VirtuallyInheritFrom<cloneable>>
//{
//};
//
/////////////////////////////////////////////////////////////////////////////////
//
//class bar
//   : public CloneInherit<Abstract<bar>, VirtuallyInheritFrom<cloneable>>
//{
//};
//
/////////////////////////////////////////////////////////////////////////////////
//
//class concrete
//   : public CloneInherit<concrete, foo, bar>
//{
//};
//
/////////////////////////////////////////////////////////////////////////////////
