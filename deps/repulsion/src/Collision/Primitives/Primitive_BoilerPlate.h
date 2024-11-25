//virtual CLASS * Clone() const override
//{
//    return new CLASS( *this );
//}

public:

using BASE::serialized_data;
using BASE::Size;


public:

// Sets the classe's data pointer.
// We assume that p_ is an array of sufficient size in which the primitive's data is found between
//      begin = p_ + Size() * pos
// and
//      end   = p_ + Size() * (pos+1).
virtual void SetPointer( SReal * const p_, const Int pos ) override
{
    serialized_data = p_ + Size() * pos;
}

virtual void SetPointer( SReal * const p_ ) override
{
    serialized_data = p_;
}


//template<typename OtherReal>
//virtual void Read( const OtherReal * const p_in, const Int i ) const override
//{
//    Read( p_in + Size() * i );
//}
//
//template<typename OtherReal>
//virtual void Read( const OtherReal * const p_in ) const override
//{
//    if( p_in != serialized_data )
//    {
//        std::copy( p_in, p_in + Size(), serialized_data);
//    }
//}
//
//template<typename OtherReal>
//virtual void Write(      OtherReal * const q_out, const Int j ) const override
//{
//    Write( q_out + Size() * j );
//}
//
//template<typename OtherReal>
//virtual void Write(      OtherReal * const q_out ) const override
//{
//    std::copy( serialized_data, serialized_data + Size(), q_out );
//}


virtual void Read( const SReal * const p_in, const Int i ) const override
{
    Read( p_in + Size() * i );
}

virtual void Read( const SReal * const p_in ) const override
{
    std::copy( p_in, p_in + Size(), serialized_data);
}


virtual void Write(      SReal * const q_out, const Int j ) const override
{
    Write( q_out + Size() * j );
}

virtual void Write(      SReal * const q_out ) const override
{
    std::copy( serialized_data, serialized_data + Size(), q_out );
}


virtual void Swap(
    SReal * const p_out, const Int i,
    SReal * const q_out, const Int j
)const override
{
    Swap( p_out + Size() * i, q_out + Size() * j );
}

virtual void Swap( SReal * const p, SReal * const q ) const override
{
    std::swap_ranges( q, q + Size(), p );


//    SReal * restrict b = &this->self_buffer[0];
//    std::copy( q, q + Size(), b );
//    std::copy( p, p + Size(), q );
//    std::copy( b, b + Size(), p );
}

virtual std::string DataString() const override
{
    std::stringstream s;
    
    const SReal * restrict const a = serialized_data;
    
    s << ClassName();
    s << ": data = { " << a[0];
    
    const Int k_begin = 1;
    const Int k_end   = Size();
    
    for( Int k = k_begin; k< k_end; ++k )
    {
        s << ", " << a[k];
    }
    s << " }";
    return s.str();
}

//friend void swap(CLASS &A, CLASS &B) noexcept
//{
//    // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//    using std::swap;
//
//    swap( A.serialized_data, B.serialized_data );
//}
//
//// copy-and-swap idiom
//CLASS &operator=(CLASS other)
//{
//    // see https://stackoverflow.com/a/3279550/8248900 for details
//
//    swap(*this, other);
//    return *this;
//
//}
