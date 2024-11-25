#pragma once

namespace Collision {
    
    // some template magic
    template<typename Int>
    constexpr Int face_count(Int amb_dim)
    {
        Int n = 2;
        for( Int k = 0; k < amb_dim; ++k )
        {
            n *= 2;
        }
        return n;
    }
    
    enum class GJK_Reason
    {
        NoReason,
        FullSimplex,
        InSimplex,
        MaxIteration,
        SmallProgress,
        SmallResidual,
        CollisionTolerance,
        Collision,
        Separated
    };
    
#define CLASS GJK_Algorithm
    
    template<int AMB_DIM, typename Real, typename Int>
    class alignas( 2 * CACHE_LINE_WIDTH ) CLASS
    {
        ASSERT_FLOAT(Real   );
        ASSERT_INT  (Int    );

    public:
        
        using PrimitiveBase_T = PrimitiveBase<AMB_DIM,Real,Int>;
        
        Real eps = std::sqrt(std::numeric_limits<Real>::epsilon());
        Real eps_squared = eps * eps;
        Int max_iter = 30;
        
    protected:
        
        Real coords [AMB_DIM+1][AMB_DIM  ] = {};  //  position of the corners of the simplex; only the first simplex_size rows are defined.
        Real P_supp [AMB_DIM+1][AMB_DIM  ] = {};  //  support points of the simplex in primitive P
        Real Q_supp [AMB_DIM+1][AMB_DIM  ] = {};  //  support points of the simplex in primitive Q

        Real dots   [AMB_DIM+1][AMB_DIM+1] = {};  // simplex_size x simplex_size matrix of dots products of  the vectors coords[0],..., coords[simplex_size-1];
        Real Gram   [AMB_DIM  ][AMB_DIM  ] = {};  // Gram matrix of size = (simplex_size-1) x (simplex_size-1) of frame spanned by the vectors coords[i] - coords[simplex_size-1];
        
        Real g      [AMB_DIM  ][AMB_DIM  ] = {}; // The local contiguous gram matrix for facets of size > 3.
        Real v                   [AMB_DIM  ] = {}; // current direction vector (quite hot)
        Real p                   [AMB_DIM  ] = {}; // just a buffer for storing current support point of p.
        Real Lambda              [AMB_DIM+1] = {}; // For the right hand sides of the linear equations to solve in DistanceSubalgorithm.
        Real best_lambda         [AMB_DIM+1] = {};
        Real facet_closest_point [AMB_DIM  ] = {};
        
        Int facet_sizes    [face_count(AMB_DIM)] = {};
        Int facet_vertices [face_count(AMB_DIM)][AMB_DIM+1] = {};
        Int facet_faces    [face_count(AMB_DIM)][AMB_DIM+1] = {};
        bool visited       [face_count(AMB_DIM)] = {};
        
        Real dotvv = std::numeric_limits<Real>::max();
        Real olddotvv = std::numeric_limits<Real>::max();
        Real dotvw = static_cast<Real>(0);
        Real TOL_squared = static_cast<Real>(0);
        Real theta_squared = static_cast<Real>(1);
        
        Int simplex_size = 0; //  index of last added point
        Int closest_facet;
        Int sub_calls = 0;
        GJK_Reason reason = GJK_Reason::NoReason;
        bool separatedQ = false;
        

    public:
        CLASS()
        {
            GJK_print("GJK_Algorithm()");
            GJK_DUMP(eps);
            GJK_DUMP(eps_squared);
            
            // Initializing facet_sizes, facet_vertices, and facet_faces.
            // TODO: In principle, the compiler should be possible to populate those at compile time. Dunno how to work this black magic.

            for( Int facet = 0; facet < face_count(AMB_DIM); ++ facet)
            {
                Int i = 0;
                
                Int * restrict const vertices = facet_vertices[facet];
                Int * restrict const faces = facet_faces[facet];

                for( Int vertex = 0; vertex < AMB_DIM+1; ++vertex )
                {
                    if( bit(facet,vertex) )
                    {
                        vertices[i] = vertex;
                        faces[i] = set_bit_to_zero(facet,vertex);
                        ++i;
                    }
                }
                
                facet_sizes[facet] = i;
                
                for( Int j = i; j < AMB_DIM+1; ++j )
                {
                    vertices[i] = -1;
                    faces[i] = -1;
                }
            }
            
            visited[0] = true;
        }

        CLASS( const CLASS & other ) : CLASS() {};
        
        CLASS( CLASS  && other ) : CLASS() {};
        
        ~CLASS() = default;
        
        constexpr Int AmbDim() const
        {
            return AMB_DIM;
        }
        
        Int Size() const
        {
            return simplex_size;
        }
        
        Real LeastSquaredDistance() const
        {
            return dotvv;
        }
        
        bool SeparatedQ() const
        {
            return separatedQ;
        }
        
        void ClosestPoint( Real * vec ) const
        {
            for( Int k = 0; k < AMB_DIM; ++k )
            {
                vec[k] = v[k];
            }
        }

        Int SubCalls() const
        {
            return sub_calls;
        }

    protected:
        
        static bool bit( const Int n, const Int k )
        {
            return ( (n & (1 << k)) >> k );
        }
        
        static  Int set_bit_to_zero( const Int n, const Int k )
        {
            return (n ^ (1 << k));
        }
        
        void Compute_dots()
        {
            GJK_tic(ClassName()+"::Compute_dots");
            // update matrix of dot products

            for( Int i = 0; i < simplex_size+1; ++i )
            {
                      Real * restrict const d_i = dots[i];
                const Real * restrict const c_i = coords[i];
                const Real * restrict const c_last = coords[simplex_size];
                
                d_i[simplex_size] = c_i[0] * c_last[0];

                for( Int k = 1; k < AMB_DIM; ++k )
                {
                    d_i[simplex_size] += c_i[k] * c_last[k];
                }
            }
            
            GJK_toc(ClassName()+"::Compute_dots");
        }
        
        int Compute_Gram()
        {
            GJK_tic(ClassName()+"::Compute_Gram");
            
            Real dots_last_last = dots[simplex_size][simplex_size];
            
            for( Int i = 0; i < simplex_size; ++i )
            {
                const Real * restrict const d_i = dots[i];
                      Real * restrict const G_i = Gram[i];

                const Real R1 = d_i[simplex_size];
                const Real R2 = dots_last_last - R1;

                Lambda[i] = R2;

                G_i[i] = d_i[i] + R2 - R1;

                // This is to guarantee that the current facet has full rank.
                // If g is not of full rank, then most recently added point (w) was already contained in the old simplex. (If another point were a problem, then  the code below would have aborted already the previous GJK iteration.
                if( G_i[i] <= 0 )
                {
                    GJK_toc(ClassName()+"::Compute_Gram");
                    return 1;
                }

                for( Int j = i+1; j < simplex_size; ++j )
                {
                    G_i[j] = d_i[j] - dots[j][simplex_size] + R2;
                }
                
//                const Real R1 = dots[i][simplex_size];
//                const Real R2 = dots_last_last - R1;
//
//                Lambda[i] = R2;
//
//                Gram[i][i] = dots[i][i] + R2 - R1;
//
//                // This is to guarantee that the current facet has full rank.
//                // If g is not of full rank, then most recently added point (w) was already contained in the old simplex. (If another point were a problem, then  the code below would have aborted already the previous GJK iteration.
//                if( Gram[i][i] <= 0 )
//                {
//                    return 1;
//                }
//
//                for( Int j = i+1; j < simplex_size; ++j )
//                {
//                    Gram[i][j] = dots[i][j] - dots[j][simplex_size] + R2;
//                }
            }
            
            GJK_toc(ClassName()+"::Compute_Gram");
            return 0;
        }
        
        int Push()
        {
            GJK_tic(ClassName()+"::Push");
            const Real * restrict const p__ = P_supp[simplex_size];
            const Real * restrict const q__ = Q_supp[simplex_size];
                  Real * restrict const c__ = coords[simplex_size];
            
            for( Int k = 0; k < AMB_DIM; ++k)
            {
                c__[k] = p__[k] - q__[k];
//                coords[simplex_size][k] = P_supp[simplex_size][k]-Q_supp[simplex_size][k];
            }

            Compute_dots();
            
            int stat = Compute_Gram();
            
            ++simplex_size;
            
            GJK_toc(ClassName()+"::Push");
            return stat;
            
        }

        Int PrepareDistanceSubalgorithm()
        {
            GJK_tic(ClassName()+"::PrepareDistanceSubalgorithm");

            // Compute starting facet.
            Int facet = 1;
            
            const Int end = simplex_size;
            
            for( Int k = 0; k < end; ++k )
            {
                facet *= 2;
            }
            --facet;

            closest_facet = facet;
            olddotvv = dotvv;
            
            dotvv = std::numeric_limits<Real>::max();
            
            // Mark subsimplices of `facet` that do not contain the last vertex of the simplex as visited before starting.
            // Facet itself is marked as unvisited; if `facet` is a reasonable starting facet, it _must_ be visited.
            visited[facet] = false;
            
            const Int top_index = end - 1;
            // Subsimplices of `facet` have _lower_ index than `facet`.

            for( Int i = 0; i < facet; ++i )
            {
                visited[i] = !bit(i,top_index);
            }
            
            GJK_toc(ClassName()+"::PrepareDistanceSubalgorithm");
            return facet;
        }
        
        // ################################################################
        // ###################   DistanceSubalgorithm   ###################
        // ################################################################
        
        void DistanceSubalgorithm( const Int facet )
        {
            GJK_tic(ClassName()+"::DistanceSubalgorithm");
            
            ++sub_calls;
            
            const Int facet_size = facet_sizes[facet];
            const Int * restrict const vertices = facet_vertices[facet];
            const Int * restrict const faces = facet_faces[facet];
            
            Real lambda [AMB_DIM+1] = {}; // The local contiguous version of Lambda for facets of size > 3.

            GJK_DUMP(facet);
            GJK_DUMP(facet_size);

#ifdef GJK_Report
            {
                std::ostringstream s;
                s<< "facet_vertices(facet,:) = { " << vertices[0];
                for( Int i = 1; i < facet_size; ++i )
                {
                    s << ", " << vertices[i];
                }
                s << " }";
                print(s.str());
            }
            {
                std::ostringstream s;
                s<< "facet_faces(facet,:) = { " << faces[0];
                for( Int i = 1; i < facet_size; ++i )
                {
                    s << ", " << faces[i];
                }
                s << " }";
                print(s.str());
            }
#endif
            
            visited[facet] = true;
            
            bool interior = true;
            
            const Int last = facet_size - 1;
                
            const Int i_last = vertices[last];
            
            const Real dots_i_last_i_last = dots[i_last][i_last];
            
//            Int switch_expr = facet_size <= AMB_DIM+1 ? facet_size :
            // Compute lambdas.
            switch( facet_size )
            {
//                case 0:
//                {
//                    eprint("Empty facet. This cannot happen.");
//                }
                case 1:
                {
                    if( dots_i_last_i_last < dotvv )
                    {
                        closest_facet = facet;
                        dotvv = dots_i_last_i_last;
                        best_lambda[last] = static_cast<Real>(1);
                        const Real * restrict const c = coords[i_last];
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
                            v[k] = c[k];
                        }
                    }
                    GJK_toc(ClassName()+"::DistanceSubalgorithm");
                    
                    return;
                }
                case 2:
                {
                    // Setting up linear system for the barycenter coordinates lambda.

                    // Find first vertex in facet.
                    const Int i_0 = vertices[0];

                    lambda[0] = Lambda[i_0] / Gram[i_0][i_0];
                    lambda[1] = static_cast<Real>(1) - lambda[0];

                    interior = (lambda[0] > eps) && (lambda[1] > eps);

                    break;
                }
#if AMB_DIM > 2
                case 3:
                {
                    // Setting up linear system for the barycenter coordinates lambda.

                    // Find first two vertices in facet.
                    const Int i_0 = vertices[0];
                    const Int i_1 = vertices[1];

                    const Real * restrict const G_i_0 = Gram[i_0];
                    const Real * restrict const G_i_1 = Gram[i_1];

                    // Using Cramer's rule to solve the linear system.
                    const Real inv_det = static_cast<Real>(1) / ( G_i_0[i_0] * G_i_1[i_1] - G_i_0[i_1] * G_i_0[i_1] );

                    lambda[0] = ( G_i_1[i_1] * Lambda[i_0] - G_i_0[i_1] * Lambda[i_1] ) * inv_det;
                    lambda[1] = ( G_i_0[i_0] * Lambda[i_1] - G_i_0[i_1] * Lambda[i_0] ) * inv_det;
                    lambda[2] = static_cast<Real>(1) - lambda[0] - lambda[1];

                    // Check the barycentric coordinates for positivity ( and compute the 0-th coordinate).
                    interior = (lambda[0] > eps) && (lambda[1] > eps) && (lambda[2] > eps);

                    break;
                }
#endif
//#if AMB_DIM > 3
//                case 4:
//                {
//                    // 3-dimensional simplex.
//                    // Setting up linear system for the barycenter coordinates lambda.
//
//                    const Int i_0 = vertices[0];
//                    const Int i_1 = vertices[1];
//                    const Int i_2 = vertices[2];
//
//                    const Real * restrict const G_i_0 = Gram[i_0];
//                    const Real * restrict const G_i_1 = Gram[i_1];
//                    const Real * restrict const G_i_2 = Gram[i_2];
//
//                    // Using Cramer's rule to solve the linear system.
//                    const Real G00 = G_i_1[i_1] * G_i_2[i_2] - G_i_1[i_2] * G_i_1[i_2];
//                    const Real G01 = G_i_1[i_2] * G_i_0[i_2] - G_i_0[i_1] * G_i_2[i_2];
//                    const Real G02 = G_i_0[i_1] * G_i_1[i_2] - G_i_1[i_1] * G_i_0[i_2];
//
//                    const Real G11 = G_i_0[i_0] * G_i_2[i_2] - G_i_0[i_2] * G_i_0[i_2];
//                    const Real G12 = G_i_0[i_1] * G_i_0[i_2] - G_i_0[i_0] * G_i_1[i_2];
//                    const Real G22 = G_i_0[i_0] * G_i_1[i_1] - G_i_0[i_1] * G_i_0[i_1];
//
//                    const Real det_inv = static_cast<Real>(1) / (G_i_0[i_0] * G00 + G_i_0[i_1] * G01 + G_i_0[i_2] * G02);
//
//                    lambda[0] = ( G00 * Lambda[i_0] + G01 * Lambda[i_1] + G02 * Lambda[i_2] ) * det_inv;
//                    lambda[1] = ( G01 * Lambda[i_0] + G11 * Lambda[i_1] + G12 * Lambda[i_2] ) * det_inv;
//                    lambda[2] = ( G02 * Lambda[i_0] + G12 * Lambda[i_1] + G22 * Lambda[i_2] ) * det_inv;
//                    lambda[3] = static_cast<Real>(1) - lambda[0] - lambda[1] - lambda[2];
//
//                    // Check the barycentric coordinates for positivity ( and compute the 0-th coordinate).
//                    interior = (lambda[0] > eps) && (lambda[1] > eps) && (lambda[2] > eps) && (lambda[3] > eps);
//
//                    break;
//                }
//#endif
                default:
                {
                    // Setting up linear system for the barycenter coordinates lambda.

                    GJK_print("Cholesky decomposition");
                    
                    for( Int i = 0; i < last; ++i )
                    {
                        const Int i_i = vertices[i];
                        
                        const Real * restrict const G_i_i = Gram[i_i];
                              Real * restrict const g_i   = g[i];
                        
                        lambda[i] = Lambda[i_i];
                        
                        g_i[i] = G_i_i[i_i];
                        
                        for( Int j = i+1; j < last; ++j )
                        {
                            Int j_j = vertices[j];
                            g[j][i] = g_i[j] = G_i_i[j_j];
                        }
                    }
                    
//                    print("g = " + g.to_string(0,last,0,last));
//                    print("lambda = " + lambda.to_string(0,last));
                    
//                    // beware: g and lambda are overwritten.
//                    int stat = Submatrix_LinearSolve<AMB_DIM>( &g[0][0], &lambda[0], last );
//
//                    if( stat )
//                    {
//                        eprint(ClassName()+"::DistanceSubalgorithm: LAPACKE_dgesv returned stat = " + ToString(stat));
//                    }
                    
                    // Cholesky decomposition
                    for( Int k = 0; k < last; ++k )
                    {
                        Real * restrict const g_k = g[k];

                        const Real a = g_k[k] = sqrt(g_k[k]);
                        const Real ainv = static_cast<Real>(1)/a;

                        for( Int j = k+1; j < last; ++j )
                        {
                            g[k][j] *= ainv;
                        }

                        for( Int i = k+1; i < last; ++i )
                        {
                            Real * restrict const g_i = g[i];

                            for( Int j = i; j < last; ++j )
                            {
                                g_i[j] -= g_k[i] * g_k[j];
                            }
                        }
                    }

                    // Lower triangular back substitution
                    for( Int i = 0; i < last; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            lambda[i] -= g[j][i] * lambda[j];
                        }
                        lambda[i] /= g[i][i];
                    }

                    // Upper triangular back substitution
                    for( Int i = last-1; i > -1; --i )
                    {
                        Real * restrict const g_i = g[i];

                        for( Int j = i+1; j < last; ++j )
                        {
                            lambda[i] -= g_i[j] * lambda[j];
                        }
                        lambda[i] /= g_i[i];
                    }
                    
                    // Check the barycentric coordinates for positivity ( and compute the 0-th coordinate).

                    lambda[last] = static_cast<Real>(1);
                    
                    for( Int k = 0; k < last; ++k)
                    {
                        lambda[last] -= lambda[k];
                        interior = interior && ( lambda[k] > eps );
                    }
                    interior = interior && (lambda[last] > eps );
                }
            }
            
            // If we arrive here, then facet_size>1.
            
            if( interior )
            {
                GJK_print("Interior point found.");
                // If the nearest point on facet lies in the interior, there is no point in searching its faces; mark them as visited.
                
                for( Int j = 0; j < facet_size; ++j )
                {
                    visited[ faces[j] ] = true;
                }
                
                Real facet_squared_dist = static_cast<Real>(0);
                
                // Computing facet_closest_point.
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    Real x = lambda[0] * coords[ vertices[0] ][ k ];
                    for( Int j = 1; j < facet_size; ++j )
                    {
                        x += lambda[j] * coords[ vertices[j] ][ k ];
                    }
                    facet_squared_dist += x * x;
                    facet_closest_point[k] = x;
                }
                
                if( facet_squared_dist < dotvv )
                {
                    closest_facet = facet;
                    dotvv = facet_squared_dist;
                    for( Int k = 0; k < AMB_DIM; ++k )
                    {
                        v[k] = facet_closest_point[k];
                    }
                    for( Int k = 0; k < AMB_DIM+1; ++k )
                    {
                        best_lambda[k] = lambda[k];
                    }
                }
            }
            else
            {
                GJK_tic("Going through facets");

                // Try to visit all faces of `facet`.
                for( Int j = 0; j < facet_size; ++j )
                {
                    const Int face = faces[j];
                    
                    if( !visited[face] )
                    {
                        // We may ignore the error code of DistanceSubalgorithm because the faces of the simplex should be nondegenerate if we arrive here.
                        DistanceSubalgorithm( face );
                    }
                }

                GJK_toc("Going through facets");
            }

            GJK_toc(ClassName()+"::DistanceSubalgorithm");
        }
        
    public:
    
        // ################################################################
        // ##########################  Compute  ###########################
        // ################################################################
        
        
        // If collision_only is set to true, then we abott if, at any time, we found a point x in P and a point y Q such that
        // theta_squared_ * (x-y)^T * (x-y) <= TOL_squared_.
        
        // Useful for intersections of thickened objects: Use TOL_squared_ = (r_P + r_Q)^2 (where r_P and r_Q are the thickening radii of P and Q) and theta_squared = 1.
        
        // Also useful for multipole acceptance criteria: Use TOL_squared_ = max(r_P^2,r_Q^2) where r_P and r_Q are the radii of P and Q) and theta_squared = chi * chi, where chi >= 0 is the separation parameter.
        
        void Compute(
            const PrimitiveBase_T & P,
            const PrimitiveBase_T & Q,
            const bool collision_only  = false,
            // If this is set to true, the algorithm will abort once a separating direction is found.
            const bool reuse_direction = false,
            const Real TOL_squared_    = static_cast<Real>(0),
            const Real theta_squared_  = static_cast<Real>(1)
        )
        {
            GJK_tic(ClassName()+"::Compute");
            
            separatedQ = false;
            
            Int iter = static_cast<Int>(0);

            int in_simplex;

            theta_squared = theta_squared_;

            if( TOL_squared_ > static_cast<Real>(0) )
            {
                TOL_squared = TOL_squared_;
            }
            else
            {
                GJK_print("Setting TOL_squared to "+ToString(eps)+" times minimum radius of primitives.");
                TOL_squared = eps * std::min( P.SquaredRadius(), Q.SquaredRadius() );
                if( TOL_squared <= static_cast<Real>(0) )
                {
                    GJK_print("Setting TOL_squared to "+ToString(eps)+" times maximum radius of primitives.");
                    TOL_squared = eps * std::max( P.SquaredRadius(), Q.SquaredRadius() );
                    
                    if( TOL_squared <= static_cast<Real>(0) )
                    {
                        dotvv = InteriorPoints_SquaredDistance(P,Q);
                        separatedQ = dotvv > static_cast<Real>(0);
                        
                        #pragma omp critical
                        {
                            wprint(ClassName()+"::Compute: Both primitives have nonpositive bounding radius: first radius = "+ToString(P.SquaredRadius())+", second radius = "+ToString(Q.SquaredRadius())+".");
                            DUMP(omp_get_thread_num());
                        }
                        
                        // TODO: set witnesses correctly.
                        GJK_toc(ClassName()+"::Compute");
                        return;
                    }
                }
            }
            
            GJK_DUMP(theta_squared);
            GJK_DUMP(TOL_squared);

            sub_calls = 0;
            reason = GJK_Reason::NoReason;
            simplex_size = 0;
            
            if( ! reuse_direction )
            {
                P.InteriorPoint(&v[0]);
                Q.InteriorPoint(&Q_supp[0][0]);
                
                for( Int k = 0; k < AMB_DIM; ++k )
                {
                    v[k] -= Q_supp[0][k];
                }
            }

            dotvv = v[0] * v[0];
            
            for( Int k = 1; k < AMB_DIM; ++k )
            {
                dotvv += v[k] * v[k];
            }
            olddotvv = dotvv;

            // Unrolling the first iteration to avoid a call to DistanceSubalgorithm.
            
            // We use w = p-q, but do not define it explicitly.
            Real a = P.MinSupportVector( &v[0], &P_supp[0][0] );
            Real b = Q.MaxSupportVector( &v[0], &Q_supp[0][0] );
            dotvw  = a-b;
            
            Push();
            
            closest_facet = 1;
            dotvv = dots[0][0];
            best_lambda[0] = static_cast<Real>(1);

            for( Int k = 0; k < AMB_DIM; ++k )
            {
                v[k] = coords[0][k];
            }
        
            while( true )
            {
                if( theta_squared * dotvv < TOL_squared )
                {
                    GJK_print("Stopped because theta_squared * dotvv < TOL_squared. ");
                    reason = GJK_Reason::CollisionTolerance;
                    break;
                }
                
                if( simplex_size >= AMB_DIM + 1 )
                {
                    reason = GJK_Reason::FullSimplex;
                    GJK_print("Stopped because simplex is full-dimensional. ");
                    break;
                }
                
                if( iter >= max_iter)
                {
                    reason = GJK_Reason::MaxIteration;
                    break;
                }
                
                ++iter;
                
                GJK_DUMP(iter);
                
                // We use w = p-q, but do not define it explicitly.
                a = P.MinSupportVector( &v[0], &P_supp[simplex_size][0] );
                b = Q.MaxSupportVector( &v[0], &Q_supp[simplex_size][0] );
                dotvw = a-b;
            
                if( collision_only && (dotvw > static_cast<Real>(0)) && (theta_squared * dotvw * dotvw > TOL_squared) )
                {
                    GJK_print("Stopped because separating plane was found. ");
                    separatedQ = true;
                    reason = GJK_Reason::Separated;
                    break;
                }
                
                
                if( abs(dotvv - dotvw) <= eps * dotvv )
                {
                    GJK_print("Stopping because of fabs(dotvv - dotvw) = "+ToString(fabs(dotvv - dotvw))+" <= "+ToString(eps * dotvv)+" = eps * dotvv.");
                    GJK_DUMP(dotvv);
                    GJK_DUMP(dotvw);
                    GJK_DUMP(eps);
                    reason = GJK_Reason::SmallResidual;
                    break;
                }
                
                in_simplex = Push();
                
                if( in_simplex  )
                {
                    GJK_print("Stopping because w=p-q was already in simplex.");
                    
                    // The vertex added most recently to the simplex coincides with one of the previous vertices.
                    // So we stop here and use the old best_lambda. All we have to do is to reduce the simplex_size by 1.
                    --simplex_size;
                    reason = GJK_Reason::InSimplex;
                    break;
                }
                
                Int initial_facet = PrepareDistanceSubalgorithm();
                
                // This computes closest_facet, v, and dotvv of the current simplex. If the simplex is degenerate, DistanceSubalgorithm terminates early and returns 1. Otherwise it returns 0.
                DistanceSubalgorithm( initial_facet );
                

                GJK_print("before reordering");
                GJK_DUMP(simplex_size);
                
                simplex_size = facet_sizes[closest_facet];
                const Int * restrict const vertices = facet_vertices[closest_facet];
                // Deleting superfluous vertices in simplex and writing everything to the beginning of the array.
                // TODO: Can we reduce the number of copies to be made here?
                
                for( Int i = 0; i < simplex_size; ++i )
                {
                    const Int i_i = vertices[i];
                    
                          Real * restrict const c_i   = coords[i];
                    const Real * restrict const c_i_i = coords[i_i];
                          Real * restrict const d_i   = dots[i];
                    const Real * restrict const d_i_i = dots[i_i];
//                    Real * restrict const p_i   = P_supp[i];
//                    Real * restrict const p_i_i = P_supp[i_i];
                          Real * restrict const q_i   = Q_supp[i];
                    const Real * restrict const q_i_i = Q_supp[i_i];
                    
                    for( Int k = 0; k < AMB_DIM; ++k )
                    {
                        c_i[k] = c_i_i[k];
//                        p_i[k] = p_i_i[k];
                        q_i[k] = q_i_i[k];
                    }

                    for( Int j = i; j < simplex_size; ++j )
                    {
                        d_i[j] = d_i_i[vertices[j]];
                    }
                }

                GJK_print("after reordering");
//                print( "coords = " + coords.to_string(0,simplex_size,0,AMB_DIM,GJK_digits) );
                GJK_DUMP(simplex_size);
                GJK_DUMP(olddotvv - dotvv);
                GJK_DUMP(olddotvv);
                GJK_DUMP(dotvv);
                
//                if( (iter>1) && (olddotvv - dotvv < -eps * dotvv) && ( theta_squared * dotvv > TOL_squared ) )
//                {
//                    eprint("-------------------------> Something is going wrong in iteration " + ToString(iter) + " : dotvv > olddotvv");
//                    valprint("Improvement", olddotvv - dotvv, GJK_digits );
//                    valprint("olddotvv ", olddotvv, GJK_digits );
//                    valprint("dotvv", dotvv, GJK_digits );
//
//                }
                
                if( abs(olddotvv - dotvv) <= eps * dotvv )
                {
                    reason = GJK_Reason::SmallProgress;
                    break;
                }
            }
            
#ifdef GJK_Report
            if( collision_only && separatedQ )
            {
                GJK_print("Stopped early because of collision_only = " + std::to_string(collision_only) + " and because a separating vector was found.");
            }
#endif
            separatedQ = separatedQ || ( TOL_squared < theta_squared * dotvv );
            
            if( iter >= max_iter)
            {
                #pragma omp critical
                {
//                    wprint(ClassName()+"::Compute: Stopped because iter = " + ToString(iter) + " >= " + ToString(max_iter) + " = max_iter iterations reached.");
//                    DUMP(omp_get_thread_num());
                }
                GJK_DUMP(simplex_size);
                GJK_DUMP(dotvv);
                GJK_DUMP(theta_squared);
                GJK_DUMP(TOL_squared);
                GJK_DUMP(theta_squared * dotvv);
                GJK_DUMP(abs(olddotvv - dotvv));
            }
            
#ifdef GJK_Report
            if( !separatedQ )
            {
                GJK_print("Stopped because of theta_squared * dotvv <= TOL_squared = " + ToString(TOL_squared) + " (intersection/nonseparability detected).");
                GJK_DUMP(theta_squared);
                GJK_DUMP(dotvv);
            }

            if( abs(olddotvv - dotvv) <= eps * dotvv  )
            {
                GJK_print("Converged after " + std::to_string(iter) + " iterations.");
            }
            

            
            if( in_simplex  )
            {
                GJK_print("Stopped because w was already in simplex.");
            }
            else
            {
                if( simplex_size >= AMB_DIM + 1 )
                {
                    GJK_print("Stopped because of simplex_size >=  AMB_DIM + 1 (intersection detected).");
                }
            }
#endif
            
            GJK_toc(ClassName()+"::Compute");
        } // Compute

        // ################################################################
        // #######################   IntersectingQ   ######################
        // ################################################################
        
        bool IntersectingQ(
            const PrimitiveBase_T & P,
            const PrimitiveBase_T & Q,
            const Real theta_squared_ = static_cast<Real>(1),
            const bool reuse_direction_ = false
        )
        {
            Compute(P, Q, true, reuse_direction_, static_cast<Real>(0), theta_squared_ );
            
            return !separatedQ;
        } // IntersectingQ
        
        // ################################################################
        // #####################   SquaredDistance   #####################
        // ################################################################
        
        Real SquaredDistance(
            const PrimitiveBase_T & P,
            const PrimitiveBase_T & Q,
            const bool reuse_direction_ = false
        )
        {
            Compute(P, Q, false, reuse_direction_, static_cast<Real>(0) );
            
            return dotvv;
        } // SquaredDistance
        
        
        // Faster overload for AABBs.
        template<typename SReal>
        Real SquaredDistance(
            const AABB<AMB_DIM,Real,Int,SReal> & P,
            const AABB<AMB_DIM,Real,Int,SReal> & Q
        )
        {
            return AABB_SquaredDistance( P, Q );
        } // SquaredDistance
        
        // ##########################################################################
        // ##############################   Witnesses   #############################
        // ##########################################################################
        
        Real Witnesses(
            const PrimitiveBase_T & P, Real * restrict const x,
            const PrimitiveBase_T & Q, Real * restrict const y,
            const bool reuse_direction_ = false
        )
        {
            Compute(P, Q, false, reuse_direction_, static_cast<Real>(0) );

            for( Int k = 0; k < AMB_DIM; ++ k)
            {
//                x[k] = best_lambda[0] * P_supp[0][k];
                y[k] = best_lambda[0] * Q_supp[0][k];
            }
            
            switch( simplex_size )
            {
                case 1:
                {
                    break;
                }
                case 2:
                {
                    for( Int k = 0; k < AMB_DIM; ++ k)
                    {
//                        x[k] += best_lambda[1] * P_supp[1][k];
                        y[k] += best_lambda[1] * Q_supp[1][k];
                    }
                    break;
                }
                case 3:
                {
                    for( Int j = 1; j < 3; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                    break;
                }
                case 4:
                {
                    for( Int j = 1; j < 4; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                    break;
                }
                case 5:
                {
                    for( Int j = 1; j < 5; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                    break;
                }
                default:
                {
                    // TODO: Could also be done by BLAS...
                    for( Int j = 1; j < simplex_size; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                }
            }
            
            for( Int k = 0; k < AMB_DIM; ++ k)
            {
                x[k] = y[k] + v[k];
            }
            
            return dotvv;
        } // Witnesses
        
        // ################################################################
        // ####################   Offset_IntersectingQ   ##################
        // ################################################################
        
        bool Offset_IntersectingQ(
            const PrimitiveBase_T & P, const Real P_offset,
            const PrimitiveBase_T & Q, const Real Q_offset,
            const bool reuse_direction_ = false
        )
        {
            const Real min_dist = P_offset + Q_offset;
            
            Compute(P, Q, true, reuse_direction_, min_dist * min_dist );
            
            return !separatedQ;
        } // Offset_IntersectingQ
        
        // ################################################################
        // #################   Offset_SquaredDistance   ##################
        // ################################################################
        
        Real Offset_SquaredDistance(
            const PrimitiveBase_T & P, const Real P_offset,
            const PrimitiveBase_T & Q, const Real Q_offset,
            const bool reuse_direction_ = false
        )
        {
            const Real min_dist = P_offset + Q_offset;
            
            Compute(P, Q, false, reuse_direction_, min_dist * min_dist );
            
            const Real dist = std::max( static_cast<Real>(0), sqrt(dotvv) - P_offset - Q_offset );
            
            return dist * dist;
        } // SquaredDistance
        
        // ##########################################################################
        // ###########################   Offset_Witnesses   #########################
        // ##########################################################################
        
        Real Offset_Witnesses(
            const PrimitiveBase_T & P, const Real P_offset, Real * restrict const x,
            const PrimitiveBase_T & Q, const Real Q_offset, Real * restrict const y,
            const bool reuse_direction_ = false
        )
        {
            // x and y are the return variables.
            const Real min_dist = P_offset + Q_offset;
            
            Compute(P, Q, false, reuse_direction_, min_dist * min_dist );
            
            const Real dist0 = sqrt(dotvv);
                  Real dist = dist0 - P_offset - Q_offset;
                  Real x_scale;
                  Real y_scale;
            
            if( dist > static_cast<Real>(0) )
            {
                // If no intersection was detected, we have to set the witnesses onto the bounday of the thickened primitives.

//                x_scale = -P_offset / dist0;
//                y_scale =  Q_offset / dist0;
                
                
                // We want y = y_0 + y_scale * v, where y_0 is constructed from Q_supp by barycentric coordinates
                y_scale =  Q_offset / dist0;
                // We want x = y + x_scale * v.
                x_scale = (dist0 - P_offset - Q_offset) / dist0;
            }
            else
            {
                // If an intersection was deteced, we just return the witnesses.
                dist = static_cast<Real>(0);
                // We want y = y_0 + y_scale * v, where y_0 is constructed from Q_supp by barycentric coordinates
                y_scale = static_cast<Real>(0);
                // We want x = y + x_scale * v.
                x_scale = static_cast<Real>(1);
                
            }

            for( Int k = 0; k < AMB_DIM; ++k )
            {
//                x[k] = x_scale * v[k] + best_lambda[0] * P_supp[0][k];
                y[k] = y_scale * v[k] + best_lambda[0] * Q_supp[0][k];
            }
            
            switch( simplex_size )
            {
                case 1:
                {
                    break;
                }
                case 2:
                {
                    for( Int k = 0; k < AMB_DIM; ++k )
                    {
//                        x[k] += best_lambda[1] * P_supp[1][k];
                        y[k] += best_lambda[1] * Q_supp[1][k];
                    }
                    break;
                }
                case 3:
                {
                    for( Int j = 1; j < 3; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                    break;
                }
                case 4:
                {
                    for( Int j = 1; j < 4; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                    break;
                }
                case 5:
                {
                    for( Int j = 1; j < 5; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                    break;
                }
                default:
                {
                    // TODO: Could also be done by BLAS...
                    for( Int j = 1; j < simplex_size; ++j )
                    {
                            for( Int k = 0; k < AMB_DIM; ++k )
                        {
//                            x[k] += best_lambda[j] * P_supp[j][k];
                            y[k] += best_lambda[j] * Q_supp[j][k];
                        }
                    }
                }
            }
            
            for( Int k = 0; k < AMB_DIM; ++ k)
            {
                x[k] = y[k] + x_scale * v[k];
            }
            
            return dist * dist;
        } // Offset_Witnesses

        
        // ########################################################################
        // ##################   InteriorPoints_SquaredDistance   ##################
        // ########################################################################
        
        Real InteriorPoints_SquaredDistance(
            const PrimitiveBase_T & P,
            const PrimitiveBase_T & Q
        )
        {
            P.InteriorPoint( &coords[0][0] );
            Q.InteriorPoint( &coords[1][0] );
            
            Real diff = coords[1][0] - coords[0][0];
            Real r2   = diff * diff;
            
            for( Int k = 1; k < AMB_DIM; ++ k)
            {
                diff = coords[1][k] - coords[0][k];
                r2  += diff * diff;
            }
            return r2;
            
        } // InteriorPoints_SquaredDistance
        
        
        // ########################################################################
        // ####################   MultipoleAcceptanceCriterion   ##################
        // ########################################################################
        
        bool MultipoleAcceptanceCriterion(
            const PrimitiveBase_T & P,
            const PrimitiveBase_T & Q,
            const Real theta_squared_,
            const bool reuse_direction_ = false
        )
        {
//            valprint("P.SquaredRadius()",P.SquaredRadius());
//            valprint("Q.SquaredRadius()",Q.SquaredRadius());
//            valprint("std::max( P.SquaredRadius(), Q.SquaredRadius() )",std::max( P.SquaredRadius(), Q.SquaredRadius() ) );
            
            Compute(P, Q, true, reuse_direction_, std::max( P.SquaredRadius(), Q.SquaredRadius() ), theta_squared_ );
            
            return separatedQ;
        } // MultipoleAcceptanceCriterion
        
        
        // Faster overload for AABBs.
        template<typename SReal>
        bool MultipoleAcceptanceCriterion(
            const AABB<AMB_DIM,Real,Int,SReal> & P,
            const AABB<AMB_DIM,Real,Int,SReal> & Q,
            const Real theta_squared_
        )
        {
            return std::max( P.SquaredRadius(), Q.SquaredRadius() ) < theta_squared_ * AABB_SquaredDistance( P, Q );
        } // MultipoleAcceptanceCriterion
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Real>::Get()+">";
        }
        
    }; // GJK_Algorithm
    
} // namespace GJK_Algorithm

#undef CLASS
