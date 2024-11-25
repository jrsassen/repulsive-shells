protected:
    
    SReal root_serialized    [1 + (DOM_DIM+2) * AMB_DIM] = {};
    SReal simplex_serialized [1 + (DOM_DIM+2) * AMB_DIM] = {};
    
    SReal corners [DOM_DIM+1][DOM_DIM+1] = {};
    SReal center  [DOM_DIM+1] = {};
    
    Polytope<DOM_DIM+1,AMB_DIM,Real,Int,SReal,SReal,Int> P;
    
    Int simplex_id = -1;
    Int level  = 0;
    Int column = 0;

    Int child_id =  0;
    Int former_child_id = -1;
    
    SReal scale  = static_cast<SReal>(1);
    SReal weight = static_cast<SReal>(1);
    
    bool current_simplex_computed = false;
    
public:
    
    SimplexHierarchy()
    {
        P.SetPointer(&simplex_serialized[0]);
    }

    virtual ~SimplexHierarchy() = default;
    
    constexpr Int VertexCount() const
    {
        return DOM_DIM+1;
    }

    constexpr Int DomDim() const
    {
        return DOM_DIM;
    }
    
    constexpr Int AmbDim() const
    {
        return AMB_DIM;
    }
    

    void RequireSimplex( const SReal * input_serialized, const Int k )
    {
//        if( simplex_id != k )
//        {
        
            current_simplex_computed = false;
            simplex_id = k;
            level  = 0;
            column = 0;
            child_id = 0;
            former_child_id = -1;
            
            scale  = static_cast<SReal>(1);
            weight = static_cast<SReal>(1);
            
            P.SetPointer(&root_serialized[0]);
            P.Read( input_serialized,k);
            P.SetPointer(&simplex_serialized[0]);
            P.Read( input_serialized,k);
        
            SMALL_UNROLL
            for( Int i = 0; i < DOM_DIM+1; ++i )
            {
                center[i] = static_cast<SReal>(1)/static_cast<SReal>(DOM_DIM+1);
            }
        
            SMALL_UNROLL
            for( Int i = 0; i < DOM_DIM+1; ++i )
            {
                SMALL_UNROLL
                for( Int j = 0; j < DOM_DIM+1; ++j )
                {
                    corners[i][j] = static_cast<SReal>(i == j);
                }
            }

//        }
    }
    
    void RequireSubsimplex()
    {
        if( !current_simplex_computed )
        {
            simplex_serialized[0] = scale * root_serialized[0];
            
            P.SetPointer( &simplex_serialized[0] );
            
            const SReal * restrict const root    =    &root_serialized[0] + 1 + AMB_DIM;
                  SReal * restrict const c       = &simplex_serialized[0] + 1           ;
                  SReal * restrict const current = &simplex_serialized[0] + 1 + AMB_DIM;

            SMALL_UNROLL
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                c[k] = center[0] * root[AMB_DIM * 0 + k];
            }
        
            SMALL_UNROLL
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                SMALL_UNROLL
                for( Int i = 1; i < DOM_DIM+1; ++i )
                {
                    c[k] += center[i] * root[ AMB_DIM * i + k];
                }
            }
        
            SMALL_UNROLL
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                SMALL_UNROLL
                for( Int i = 0; i < DOM_DIM+1; ++i )
                {
                    current[ AMB_DIM * i + k] = corners[i][0] * root[ AMB_DIM * 0 + k];
                    
                    SMALL_UNROLL
                    for( Int j = 1; j < DOM_DIM+1; ++j )
                    {
                        current[ AMB_DIM * i + k] += corners[i][j] * root[ AMB_DIM * j + k];
                    }
                }
            }
            
            current_simplex_computed = true;
        }
    }
    


    void RequireSimplexFromCoordinates( const SReal * coords, const Int k )
    {
    //        if( simplex_id != k )
    //        {
            simplex_id = k;
            level  = 0;
            column = 0;
            
            scale  = static_cast<SReal>(1);
            weight = static_cast<SReal>(1);

            P.SetPointer(&root_serialized[0]);

            P.FromCoordinates(coords,k);

            P.SetPointer(&simplex_serialized[0]);

            SMALL_UNROLL
            for( Int i = 0; i < DOM_DIM+1; ++i )
            {
                center[i] = static_cast<SReal>(1)/static_cast<SReal>(DOM_DIM+1);
            
                SMALL_UNROLL
                for( Int j = 0; j < DOM_DIM+1; ++j )
                {
                    corners[i][j] = static_cast<SReal>(i == j);
                }
            }
            
            current_simplex_computed = false;
            child_id = 0;
            former_child_id = -1;
    //        }
    }

    const Polytope<DOM_DIM+1,AMB_DIM,Real,Int,SReal,SReal,Int> & SimplexPrototype()
    {
        RequireSubsimplex();
        return P;
    }
    
    SReal Scale() const
    {
        return scale;
    }
    
    SReal Weight() const
    {
        return weight;
    }
    
    Int Level() const
    {
        return level;
    }
    
    Int Column() const
    {
        return column;
    }

    Int SimplexID() const
    {
        return simplex_id;
    }

    Int ChildID() const
    {
        return child_id;
    }
    Int FormerChildID() const
    {
        return former_child_id;
    }

    bool IsLastChild() const
    {
        return (0< child_id) && (child_id < ChildCount()-1);
    }


    const SReal * Center() const
    {
        return &center[0];
    }

    const SReal * BarycentricCoordinates() const
    {
        return &corners[0][0];
    }

    const SReal * SimplexSerialized() const
    {
        return simplex_serialized;
    }

    std::string CornerString() const
    {
        std::stringstream s;
        
        s << "{ ";
        
        s << "{ " << corners[0][0];
        SMALL_UNROLL
        for( Int j = 1; j < DOM_DIM+1; ++j )
        {
            s << ", " << corners[0][j];
        }
        s << " }";
        
        SMALL_UNROLL
        for( Int i = 1; i < DOM_DIM+1; ++i )
        {
            s << ", " << "{ " << corners[i][0];
            
            SMALL_UNROLL
            for( Int j = 1; j < DOM_DIM+1; ++j )
            {
                s << ", " << corners[i][j];
            }
            s << " }";
        }
        
        s << " }";
        return s.str();
    }

    std::string CenterString() const
    {
        std::stringstream s;

        s << "{ " << center[0];
        
        SMALL_UNROLL
        for( Int j = 1; j < DOM_DIM+1; ++j )
        {
            s << ", " << center[j];
        }
        s << " }";
        return s.str();
    }

    std::string EmbeddedSimplexString() const
    {
        std::stringstream s;
        
        s << "{ { ";

        const SReal * X = simplex_serialized+1+AMB_DIM;

        s << X[0];

        for( Int j = 1; j < AMB_DIM; ++j )
        {
            s << ", " << X[(DOM_DIM+1)*0+j];
        }

        for( Int i = 1; i < DOM_DIM+1; ++i )
        {
            s << " }" << ", " << "{ ";

            s << X[(DOM_DIM+1)*i+0];

            for( Int j = 1; j < AMB_DIM; ++j )
            {
                s << ", " << X[(DOM_DIM+1)*i+j];
            }
        }

        s << " } }";
        
        return s.str();
    }

    
#ifdef LTEMPLATE_H

    mma::TensorRef<mreal> BarycentricCoordinates_TensorRef()
    {
        auto A = mma::makeMatrix<mreal>( DOM_DIM+1, DOM_DIM+1 );
        
        mreal * a = A.data();
        
        SMALL_UNROLL
        for( Int i = 0; i < DOM_DIM+1; ++i )
        {
            SMALL_UNROLL
            for( Int j = 0; j < DOM_DIM+1; ++j )
            {
                a[ DOM_DIM+1 * i + j ] = static_cast<mreal>(corners[i][j]);
            }
        }
        
        return A;
    }

    mma::TensorRef<mreal> VertexCoordinates_TensorRef()
    {
        auto A = mma::makeMatrix<mreal>( DOM_DIM+1, AMB_DIM );
        
        mreal * a = A.data();
        
        RequireSubsimplex();
        
        SReal * restrict const q = &simplex_serialized[0] + 1 + AMB_DIM;
        
        SMALL_UNROLL
        for( Int i = 0; i < DOM_DIM+1 * AMB_DIM; ++i )
        {
            a[i] = static_cast<mreal>(q[i]);
        }
        
        return A;
    }

    mma::TensorRef<mreal> RootSerialized_TensorRef()
    {
        auto A = mma::makeVector<mreal>( P.Size() );

        P.SetPointer(&root_serialized[0]);
        
        P.Write(A.data());
        
        P.SetPointer(&simplex_serialized[0]);
        
        return A;
    }
    
    mma::TensorRef<mreal> SimplexSerialized_TensorRef()
    {
        auto A = mma::makeVector<mreal>( P.Size() );
    
        RequireSubsimplex();
        
        P.SetPointer(&simplex_serialized[0]);
        
        P.Write(A.data());
        
        return A;
    }
    
#endif
    
public:
    
    std::string ClassName() const
    {
        return TO_STD_STRING(CLASS)+"<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+">";
    }
