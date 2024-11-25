#pragma once

#define CLASS MovingPolytope
#define BASE  MovingPolytopeExt<AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>

namespace Collision {
    
    template<int POINT_COUNT, int AMB_DIM, typename Real, typename Int, typename SReal,
        typename ExtReal, typename ExtInt>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS : public BASE
    {

        ASSERT_FLOAT(ExtReal);
        ASSERT_INT  (ExtInt );
        
    protected:
        
        using BASE::a;
        using BASE::b;
        using BASE::T;
        
        mutable SReal r = 0;      // radius positions  as measured from av_position
        mutable SReal w = 0;      // radius velocities as measured from av_velocity
        mutable SReal v = 0;      // norm of average velocity
        
        mutable SReal av_position [AMB_DIM] = {};
        mutable SReal av_velocity [AMB_DIM] = {};
        
        mutable SReal positions   [POINT_COUNT][AMB_DIM] = {};
        mutable SReal velocities  [POINT_COUNT][AMB_DIM] = {};
        
    public:
        
        CLASS() : BASE() {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE(other)
        {}

        // Move constructor
        CLASS( CLASS && other ) noexcept : BASE(other) {}
        
        virtual ~CLASS() override = default;
        
        REPULSION__ADD_CLONE_CODE(CLASS)
        
    public:
        
        virtual Int PointCount() const override
        {
            return POINT_COUNT;
        }

        
        virtual Int CoordinateSize() const override
        {
            return 1 + (1 + POINT_COUNT ) * AMB_DIM;
        }
        
        virtual Int VelocitySize() const override
        {
            return 1 + (1 + POINT_COUNT ) * AMB_DIM + 1;
        }
        
        
        virtual Int Size() const override
        {
            return CoordinateSize() + VelocitySize();
        }
        
        virtual void FromCoordinates( const ExtReal * const p_, const Int i = 0 ) override
        {
            // Supposed to do the same as Polytope<POINT_COUNT,AMB_DIM,...>::FromCoordinates + ReadCoordinatesSerialized. Meant primarily for debugging purposes; in practice, we will typically used LoadCoordinatesSerialized from already serialized data.
            GJK_tic(ClassName()+"::FromCoordinates");
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                positions[0][k] = static_cast<SReal>(p_[(POINT_COUNT*AMB_DIM)*i+AMB_DIM*0+k]);
                av_position[k]  = positions[0][k];
            }

            for( Int j = 1; j < POINT_COUNT; ++j )
            {
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    positions[j][k] = static_cast<SReal>(p_[(POINT_COUNT*AMB_DIM)*i+AMB_DIM*j+k]);
                    av_position[k] += positions[j][k];
                }
            }

            constexpr SReal factor = (static_cast<SReal>(1)/static_cast<SReal>(POINT_COUNT));
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                av_position[k] *= factor;
            }

            // Compute radius.
            SReal r_2 = static_cast<SReal>(0);
            
            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                SReal diff = positions[j][0] - av_position[0];
                SReal square = diff * diff;
                
                    for( Int k = 1; k < AMB_DIM; ++k )
                {
                    diff = positions[j][k] - av_position[k];
                    square += diff * diff;
                }
                r_2 = std::max( r_2, square );
            }
            
            r = std::sqrt(r_2);
            
            GJK_toc(ClassName()+"::FromCoordinates");
        }
        
        virtual void WriteCoordinatesSerialized( SReal * const p_serialized, const Int i = 0 ) const override
        {
            // Reads from serialized data in the format of Polytope<POINT_COUNT,AMB_DIM,...>
            GJK_tic(ClassName()+"::ReadCoordinatesSerialized");
            
            SReal * restrict const p__ = p_serialized + CoordinateSize() * i;
            
            p__[0] = r * r;
            std::copy( &av_position[0],  &av_position[0] + AMB_DIM              , &p__[1]         );
            std::copy( &positions[0][0],&positions[0][0] + AMB_DIM * POINT_COUNT, &p__[1+AMB_DIM] );
            
            GJK_toc(ClassName()+"::ReadCoordinatesSerialized");
        }
        
        virtual void ReadCoordinatesSerialized( const SReal * const p_serialized, const Int i = 0 ) override
        {
            // Write to serialized data in the format of Polytope<POINT_COUNT,AMB_DIM,...>
            GJK_tic(ClassName()+"::WriteCoordinatesSerialized");
            
            const SReal * restrict const p__ = p_serialized + CoordinateSize() * i;
            
            r = std::sqrt(p__[0]);
            std::copy( &p__[1],         &p__[1]+AMB_DIM                        , &av_position[0]  );
            std::copy( &p__[1+AMB_DIM], &p__[1+AMB_DIM] + AMB_DIM * POINT_COUNT, &positions[0][0] );
            
            GJK_toc(ClassName()+"::WriteCoordinatesSerialized");
        }
        

        
        virtual void FromVelocities( const ExtReal * const v_, const Int i = 0 ) override
        {
            GJK_tic(ClassName()+"::FromVelocities");

            for( Int j = 0; j < AMB_DIM; ++j )
            {
                velocities[0][j] = static_cast<SReal>(v_[(POINT_COUNT*AMB_DIM)*i + AMB_DIM*0 + j]);
                av_velocity[j]   = velocities[0][j];
            }

            for( Int j = 1; j < POINT_COUNT; ++j )
            {
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    velocities[j][k] = static_cast<SReal>(v_[(POINT_COUNT*AMB_DIM)*i + AMB_DIM*j + k]);
                    av_velocity[k] += velocities[j][k];
                }
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                av_velocity[k] *= (static_cast<SReal>(1)/static_cast<SReal>(POINT_COUNT));
            }

            SReal v_2 = static_cast<SReal>(0);


            for( Int k = 0; k < AMB_DIM; ++k )
            {
                v_2 += av_velocity[k] * av_velocity[k];
            }
            v = std::sqrt(v_2);

            // Compute radius.
            SReal w_2 = static_cast<SReal>(0);
            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                SReal diff = velocities[j][0] - av_velocity[0];
                SReal square = diff * diff;

                    for( Int k = 1; k < AMB_DIM; ++k )
                {
                    diff = velocities[j][k] - av_velocity[k];
                    square += diff * diff;
                }
                w_2 = std::max( w_2, square );
            }
            w = std::sqrt(w_2);

            GJK_toc(ClassName()+"::FromVelocities");
        }
        
        virtual void FromVelocitiesIndexList( const ExtReal * const v_, const ExtInt * const tuples, const Int i = 0 ) override
        {
            GJK_tic(ClassName()+"::FromVelocitiesIndexList");
            
            const ExtInt * restrict const s = tuples + POINT_COUNT * i;
            
            {
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    velocities[0][k] = static_cast<SReal>(v_[AMB_DIM * s[0] + k]);
                    av_velocity[k]   = velocities[0][k];
                }
            }
            
            for( Int j = 1; j < POINT_COUNT; ++j )
            {
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    velocities[j][k] = static_cast<SReal>(v_[AMB_DIM * s[j] + k]);
                    av_velocity[k] += velocities[j][k];
                }
            }
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                av_velocity[k] *= (static_cast<SReal>(1)/static_cast<SReal>(POINT_COUNT));
            }
            
            SReal v_2 = static_cast<SReal>(0);
            
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                v_2 += av_velocity[k] * av_velocity[k];
            }
            v = std::sqrt(v_2);
            
            // Compute radius.
            SReal w_2 = static_cast<SReal>(0);
            for( Int j = 0; j < POINT_COUNT; ++j )
            {
                SReal diff = velocities[j][0] - av_velocity[0];
                SReal square = diff * diff;
                
                    for( Int k = 1; k < AMB_DIM; ++k )
                {
                    diff = velocities[j][k] - av_velocity[k];
                    square += diff * diff;
                }
                w_2 = std::max( w_2, square );
            }
            w = std::sqrt(w_2);
            
            GJK_toc(ClassName()+"::FromVelocitiesIndexList");
        }
        
        
        virtual void WriteVelocitiesSerialized( SReal * const v_serialized, const Int i = 0 ) const override
        {
            // Loads from serialized data as stored by Polytope<POINT_COUNT,AMB_DIM,...>
            GJK_tic(ClassName()+"::WriteVelocitiesSerialized");
            
            SReal * restrict const v__ = v_serialized + VelocitySize() * i;
            
            v__[0] = w;
            std::copy( &av_velocity[0],  &av_velocity[0] + AMB_DIM                , &v__[1]         );
            std::copy( &velocities[0][0],&velocities[0][0] + AMB_DIM * POINT_COUNT, &v__[1+AMB_DIM] );
            v__[VelocitySize()-1] = v;
            
            GJK_toc(ClassName()+"::WriteVelocitiesSerialized");
        }
        
        virtual void ReadVelocitiesSerialized( const SReal * const v_serialized, const Int i = 0 ) override
        {
            // Reads from serialized data as stored by WriteVelocitiesSerialized
            GJK_tic(ClassName()+"::ReadVelocitiesSerialized");
            
            const SReal * restrict const v__ = v_serialized + VelocitySize() * i;
            
            w = v__[0];
            std::copy( &v__[1],         &v__[1]+AMB_DIM                        , &av_velocity[0]   );
            std::copy( &v__[1+AMB_DIM], &v__[1+AMB_DIM] + AMB_DIM * POINT_COUNT, &velocities[0][0] );
            v = v__[VelocitySize()-1];
            
            GJK_toc(ClassName()+"::ReadVelocitiesSerialized");
        }

        
        
        virtual void WriteDeformedSerialized( SReal * const p_serialized, const SReal t, const Int i = 0 ) const override
        {
            // Reads from serialized data in the format of Polytope<POINT_COUNT,AMB_DIM,...>
            GJK_tic(ClassName()+"::WriteDeformedSerialized");
            
            SReal * restrict const p = p_serialized + CoordinateSize() * i;
            
//            p[0] = r * r;
            
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                p[1+k] = av_position[k] + t * av_velocity[k];
            }
                     
            SReal r2 = 0;
            
            for( Int j = 0; j < POINT_COUNT; ++ j)
            {
                SReal r2_local = 0;
                
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    p[1+AMB_DIM+AMB_DIM*j+k] = positions[j][k] + t * velocities[j][k];
                    
                    SReal diff = p[1+AMB_DIM+AMB_DIM*j+k] - p[1+k];
                    
                    r2_local += diff * diff;
                }
                
                r2 = std::max(r2,r2_local);
            }
            
            p[0] = r2;

            
            GJK_toc(ClassName()+"::WriteDeformedSerialized");
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MaxSupportVector( const Real * const dir, Real * const supp ) const override
        {
            GJK_tic(ClassName()+"::MaxSupportVector");
            
            SReal vec [AMB_DIM+1] = {};
            for( Int j = 0; j < AMB_DIM+1; ++j )
            {
                vec[j] = static_cast<SReal>(dir[j]);
            }
            
            SReal value_at_a;
            SReal value_at_b;
            
            // i = 0
            {
                {
                    const SReal x =  positions[0][0] * vec[0];
                    const SReal y = velocities[0][0] * vec[0];
                    
                    value_at_a  = x + a * y;
                    value_at_b  = x + b * y;
                }
                
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    const SReal x =  positions[0][j] * vec[j];
                    const SReal y = velocities[0][j] * vec[j];
                    
                    value_at_a += x + a * y;
                    value_at_b += x + b * y;
                }
            }
            
            SReal max_at_a = value_at_a;
            SReal max_at_b = value_at_b;
            
            Int pos_a = 0;
            Int pos_b = 0;
            
            for( Int i = 1; i < POINT_COUNT; ++i )
            {
                {
                    const SReal x =  positions[i][0] * vec[0];
                    const SReal y = velocities[i][0] * vec[0];
                    
                    value_at_a  = x + a * y;
                    value_at_b  = x + b * y;
                }
                
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    const SReal x =  positions[i][j] * vec[j];
                    const SReal y = velocities[i][j] * vec[j];
                    
                    value_at_a += x + a * y;
                    value_at_b += x + b * y;
                }
                
                if( value_at_a > max_at_a )
                {
                    pos_a = i;
                    max_at_a = value_at_a;
                }
                
                if( value_at_b > max_at_b )
                {
                    pos_b = i;
                    max_at_b = value_at_b;
                }
            }
            
            const SReal aT = a * T;
            const SReal bT = b * T;
            
            max_at_a += aT * vec[AMB_DIM];
            max_at_b += bT * vec[AMB_DIM];
            
            if( max_at_a >= max_at_b )
            {
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    supp[j] = static_cast<Real>(positions[pos_a][j] + a * velocities[pos_a][j]);
                }
                supp[AMB_DIM] = static_cast<Real>(aT);
                
                GJK_toc(ClassName()+"::MaxSupportVector");
                
                return static_cast<Real>(max_at_a);
            }
            else
            {
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    supp[j] = static_cast<Real>(positions[pos_b][j] + b * velocities[pos_b][j]);
                }
                supp[AMB_DIM] = static_cast<Real>(bT) ;
                
                GJK_toc(ClassName()+"::MaxSupportVector");
                
                return static_cast<Real>(max_at_b);
            }
        }
        
        
        //Computes support vector supp of dir.
        virtual Real MinSupportVector( const Real * const dir, Real * const supp ) const override
        {
            GJK_tic(ClassName()+"::MinSupportVector");
            
            SReal vec [AMB_DIM+1] = {};
            
            for( Int j = 0; j < AMB_DIM+1; ++j )
            {
                vec[j] = static_cast<SReal>(dir[j]);
            }
            
            SReal value_at_a;
            SReal value_at_b;
            
            {
                const SReal x =  positions[0][0] * vec[0];
                const SReal y = velocities[0][0] * vec[0];
                
                value_at_a  = x + a * y;
                value_at_b  = x + b * y;
            }
            for( Int j = 1; j < AMB_DIM; ++j )
            {
                const SReal x =  positions[0][j] * vec[j];
                const SReal y = velocities[0][j] * vec[j];
                
                value_at_a += x + a * y;
                value_at_b += x + b * y;
            }
            
            SReal min_at_a = value_at_a;
            SReal min_at_b = value_at_b;
            
            Int pos_a = 0;
            Int pos_b = 0;
            
            for( Int i = 1; i < POINT_COUNT; ++i )
            {
                {
                    const SReal x =  positions[i][0] * vec[0];
                    const SReal y = velocities[i][0] * vec[0];
                    
                    value_at_a  = x + a * y;
                    value_at_b  = x + b * y;
                }
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    const SReal x =  positions[i][j] * vec[j];
                    const SReal y = velocities[i][j] * vec[j];
                    
                    value_at_a += x + a * y;
                    value_at_b += x + b * y;
                }
                
                if( value_at_a < min_at_a )
                {
                    pos_a = i;
                    min_at_a = value_at_a;
                }
                
                if( value_at_b < min_at_b )
                {
                    pos_b = i;
                    min_at_b = value_at_b;
                }
            }
            
            const SReal aT = a * T;
            const SReal bT = b * T;
            
            min_at_a += aT * vec[AMB_DIM];
            min_at_b += bT * vec[AMB_DIM];
            
            if( min_at_a <= min_at_b )
            {
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    supp[j] = static_cast<Real>(positions[pos_a][j] + a * velocities[pos_a][j]);
                }
                supp[AMB_DIM] = static_cast<Real>(aT);
                
                GJK_toc(ClassName()+"::MinSupportVector");
                
                return static_cast<Real>(min_at_a);
            }
            else
            {
                for( Int j = 0; j < AMB_DIM; ++j )
                {
                    supp[j] = static_cast<Real>(positions[pos_b][j] + b * velocities[pos_b][j]);
                }
                supp[AMB_DIM] = static_cast<Real>(bT);
                
                GJK_toc(ClassName()+"::MinSupportVector");
                
                return static_cast<Real>(min_at_b);
            }
        }
        
        
        // Computes only the values of min/max support function. Usefull to compute bounding boxes.
        virtual void MinMaxSupportValue( const Real * const dir, Real & min_val, Real & max_val ) const override
        {
            GJK_tic(ClassName()+"::MinMaxSupportValue");
            
            SReal vec [AMB_DIM+1] = {};
            for( Int j = 1; j < AMB_DIM+1; ++j )
            {
                vec[j] = static_cast<SReal>(dir[j]);
            }
            
            SReal value_at_a;
            SReal value_at_b;
            
            {
                const SReal x =  positions[0][0] * vec[0];
                const SReal y = velocities[0][0] * vec[0];
                
                value_at_a  = x + a * y;
                value_at_b  = x + b * y;
            }
            for( Int j = 1; j < AMB_DIM; ++j )
            {
                const SReal x =  positions[0][j] * vec[j];
                const SReal y = velocities[0][j] * vec[j];
                
                value_at_a += x + a * y;
                value_at_b += x + b * y;
            }
            
            SReal min_at_a = value_at_a;
            SReal max_at_a = value_at_a;
            SReal min_at_b = value_at_b;
            SReal max_at_b = value_at_b;
            
            for( Int i = 1; i < POINT_COUNT; ++i )
            {
                {
                    const SReal x =  positions[i][0] * vec[0];
                    const SReal y = velocities[i][0] * vec[0];
                    
                    value_at_a  = x + a * y;
                    value_at_b  = x + b * y;
                }
                for( Int j = 1; j < AMB_DIM; ++j )
                {
                    const SReal x =  positions[i][j] * vec[j];
                    const SReal y = velocities[i][j] * vec[j];
                    
                    value_at_a += x + a * y;
                    value_at_b += x + b * y;
                }
                
                if( value_at_a < min_at_a )
                {
                    min_at_a = value_at_a;
                }
                
                if( value_at_b < min_at_b )
                {
                    min_at_b = value_at_b;
                }
                
                if( value_at_a > max_at_a )
                {
                    max_at_a = value_at_a;
                }
                
                if( value_at_b > max_at_b )
                {
                    max_at_b = value_at_b;
                }
            }
            
            const SReal aT = a * T;
            const SReal bT = b * T;
            
            min_at_a += aT * vec[AMB_DIM];
            max_at_a += aT * vec[AMB_DIM];
            min_at_b += bT * vec[AMB_DIM];
            max_at_b += bT * vec[AMB_DIM];
            
            min_val = static_cast<Real>(std::min(min_at_a, min_at_b));
            max_val = static_cast<Real>(std::max(max_at_a, max_at_b));
            
            GJK_toc(ClassName()+"::MinMaxSupportValue");
        }
        
        // Returns some point within the primitive and writes it to p.
        virtual void InteriorPoint( Real * const p ) const override
        {
            GJK_tic(ClassName()+"::InteriorPoint");
            
            const SReal t = static_cast<SReal>(0.5) * (a + b);
            
            for( Int k = 0; k < AMB_DIM+1; ++k )
            {
                p[k] = static_cast<Real>(av_position[k] + t * av_velocity[k]);
            }
            
            p[AMB_DIM] = t * T;
            
            GJK_toc(ClassName()+"::InteriorPoint");
        }
        
        // Returns some point within the primitive and writes it to p.
        virtual Real InteriorPoint( const Int k ) const override
        {
            GJK_tic(ClassName()+"::InteriorPoint");
            
            const SReal t = static_cast<SReal>(0.5) * (a + b);
            
            if( k == AMB_DIM )
            {
                return t * T;
            }
            else
            {
                return static_cast<Real>(av_position[k] + t * av_velocity[k]);
            }
            
            GJK_toc(ClassName()+"::InteriorPoint");
        }
        
        // Returns some (upper bound of the) squared radius of the primitive as measured from the result of InteriorPoint.
        virtual Real SquaredRadius() const override
        {
            GJK_tic(ClassName()+"::SquaredRadius");
            
            const SReal s = static_cast<SReal>(0.5) * std::abs(b-a);
            const SReal x = s * v + r + w * b;
            const SReal y = s * T;
            
            GJK_toc(ClassName()+"::SquaredRadius");
            
            return static_cast<Real>(x*x + y*y);
        }

        virtual std::string DataString() const override
        {
            std::stringstream s;
            
            s << ClassName() << ": ";
            s << " r = " << r <<", ";
            s << " v = " << v <<", ";
            s << " w = " << w <<", ";
            
            s << " av_position = { " << av_position[0];
            for( Int k = 1; k< AMB_DIM; ++k )
            {
                s << ", " << av_position[k];
            }
            s << " }";
            
            s << " av_velocity = { " << av_velocity[0];
            for( Int k = 1; k< AMB_DIM; ++k )
            {
                s << ", " << av_velocity[k];
            }
            s << " }";
            
            return s.str();
        }

        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(POINT_COUNT)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+","+TypeName<ExtInt>::Get()+">";
        }
        
    };

    
    
    
    template <int AMB_DIM, typename Real, typename Int, typename SReal,
        typename ExtReal = SReal, typename ExtInt = Int>
    std::unique_ptr<BASE> MakeMovingPolytope( const Int P_size )
    {
        switch(  P_size  )
        {
            case 1:
            {
                return std::unique_ptr<BASE>(MovingPolytope<1,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 2:
            {
                return std::unique_ptr<BASE>(MovingPolytope<2,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 3:
            {
                return std::unique_ptr<BASE>(MovingPolytope<3,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 4:
            {
                return std::unique_ptr<BASE>(MovingPolytope<4,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 5:
            {
                return std::unique_ptr<BASE>(MovingPolytope<5,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 6:
            {
                return std::unique_ptr<BASE>(MovingPolytope<6,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 7:
            {
                return std::unique_ptr<BASE>(MovingPolytope<7,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 8:
            {
                return std::unique_ptr<BASE>(MovingPolytope<8,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 9:
            {
                return std::unique_ptr<BASE>(MovingPolytope<9,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 10:
            {
                return std::unique_ptr<BASE>(MovingPolytope<10,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 11:
            {
                return std::unique_ptr<BASE>(MovingPolytope<11,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 12:
            {
                return std::unique_ptr<BASE>(MovingPolytope<12,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 13:
            {
                return std::unique_ptr<BASE>(MovingPolytope<13,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 14:
            {
                return std::unique_ptr<BASE>(MovingPolytope<14,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 15:
            {
                return std::unique_ptr<BASE>(MovingPolytope<15,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 16:
            {
                return std::unique_ptr<BASE>(MovingPolytope<16,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 17:
            {
                return std::unique_ptr<BASE>(MovingPolytope<17,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 18:
            {
                return std::unique_ptr<BASE>(MovingPolytope<18,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 19:
            {
                return std::unique_ptr<BASE>(MovingPolytope<19,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            case 20:
            {
                return std::unique_ptr<BASE>(MovingPolytope<20,AMB_DIM,Real,Int,SReal,ExtReal,ExtInt>());
            }
            default:
            {
                eprint("MakeMovingPolytope: Number of vertices of polytope = " + ToString( P_size ) + " is not in the range from 1 to 20. Returning nullptr.");
                return nullptr;
            }
        }
        
    } // MakeMovingPolytope
    
} // Collision

#undef CLASS
#undef BASE
