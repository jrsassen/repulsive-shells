// Machine generated code. Don't edit this file!

#pragma once

namespace Repulsion
{
	template<mint DOM_DIM, mint AMB_DIM, typename Real, typename Int>
    struct SimplicialMeshDetails
    {
		inline std::string ClassName() const
        {
            return "SimplicialMeshDetails<"+ToString(DOM_DIM)+","+ToString(AMB_DIM)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }

		
	void ComputeNearFarData(
		const Tensor2<Real,Int> & V_coords,
		const Tensor2<Int ,Int> & simplices,
			  Tensor2<Real,Int> & P_near,
			  Tensor2<Real,Int> & P_far
	) const
    {
        ptic(ClassName()+"::ComputeNearFarData");
        eprint(ClassName()+"::ComputeNearFarData not implemented. Doing nothing.");
        ptoc(ClassName()+"::ComputeNearFarData");
    }

	void ComputeNearFarDataOps( 
		const Tensor2<Real,Int> & V_coords,
        const Tensor2<Int ,Int> & simplices,
		      Tensor2<Real,Int> & P_coords,
		      Tensor3<Real,Int> & P_hull_coords,
		      Tensor2<Real,Int> & P_near,
		      Tensor2<Real,Int> & P_far,
		SparseMatrixCSR<Real,Int> & DiffOp,
		SparseMatrixCSR<Real,Int> & AvOp 
	) const
    {
        ptic(ClassName()+"::ComputeNearFarDataOps");
        eprint(ClassName()+"::ComputeNearFarDataOps not implemented. Doing nothing.");
        ptoc(ClassName()+"::ComputeNearFarDataOps");
    }

	void DNearToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_near, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DNearToHulls");
        eprint(ClassName()+"::DNearToHulls not implemented. Returning 0.");
		
		if(!addTo)
		{
			buffer.Fill(static_cast<Real>(0));
		}
        ptoc(ClassName()+"::DNearToHulls");
    }

	void DFarToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_far,
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DFarToHulls");
                eprint(ClassName()+"::DFarToHulls not implemented. Returning 0.");
		
		if(!addTo)
		{
			buffer.Fill(static_cast<Real>(0));
		}
        ptoc(ClassName()+"::DFarToHulls");
    }

	}; // SimplicialMeshDetails<DOM_DIM,AMB,Real,Int>

//----------------------------------------------------------------------------------------------

    
    template<typename Real, typename Int>
    struct SimplicialMeshDetails<1,2,Real,Int>
    {
	private:

		const Int thread_count = 1;

	public:

		SimplicialMeshDetails( const Int thread_count_ = 1 ) 
		:
			thread_count(std::max(static_cast<Int>(1),thread_count_))
		{}
	
		inline Int ThreadCount() const
		{
			return thread_count;
		}
	
		inline std::string ClassName() const
        {
            return "SimplicialMeshDetails<1,2,"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
		
	void ComputeNearFarData( 
		const Tensor2<Real,Int> & V_coords,
		const Tensor2<Int ,Int> & simplices,
			  Tensor2<Real,Int> & P_near,
			  Tensor2<Real,Int> & P_far
	) const
    {
        ptic(ClassName()+"::ComputeNearFarData");
        
        //Int size       = 2;
        //Int amb_dim    = 2;
        //Int dom_dim    = 1;

        auto job_ptr = BalanceWorkLoad<Int>( simplices.Dimension(0), ThreadCount() );
        
        #pragma omp parallel for num_threads( ThreadCount() )
        for( Int thread = 0; thread < ThreadCount(); ++thread )
        {
			const Real * restrict const V_coords__      = V_coords.data();	
			const Int  * restrict const simplices__     = simplices.data();

			Real hull    [2][2];
			Real df      [2][1];
			Real dfdagger[1][2];
			Real g       [1][1];
			Real ginv    [1][1];

			Int simplex  [2];
			
			const Int i_begin = job_ptr[thread];
			const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {
				Real * restrict const near = P_near.data(i);                    
				Real * restrict const far  = P_far.data(i);   
            
				simplex[0] = simplices__[2*i +0];
				simplex[1] = simplices__[2*i +1];

				near[1] = hull[0][0] = V_coords__[2*simplex[0]+0];
				near[2] = hull[0][1] = V_coords__[2*simplex[0]+1];
				near[3] = hull[1][0] = V_coords__[2*simplex[1]+0];
				near[4] = hull[1][1] = V_coords__[2*simplex[1]+1];

				far[1] = static_cast<Real>(0.5) * ( hull[0][0] + hull[1][0] );
				far[2] = static_cast<Real>(0.5) * ( hull[0][1] + hull[1][1] );

				df[0][0] = hull[1][0] - hull[0][0];
				df[1][0] = hull[1][1] - hull[0][1];

				g[0][0] = df[0][0] * df[0][0] + df[1][0] * df[1][0];

   
                near[0] = far[0] = sqrt( fabs(g[0][0]) );

                ginv[0][0] = static_cast<Real>(1)/g[0][0];
                
                //  dfdagger = g^{-1} * df^T (1 x 2 matrix)
				dfdagger[0][0] = ginv[0][0] * df[0][0];
				dfdagger[0][1] = ginv[0][0] * df[1][0];
            
				near[5] = far[3]  = static_cast<Real>(1) - df[0][0] * dfdagger[0][0];
				near[6] = far[4]  =    - df[0][0] * dfdagger[0][1];
				near[7] = far[5]  = static_cast<Real>(1) - df[1][0] * dfdagger[0][1];

            } // for( Int i = i_begin; i < i_end; ++i )

		} // #pragma omp parallel for num_threads( ThreadCount() )

        ptoc(ClassName()+"::ComputeNearFarData");
    }

    void ComputeNearFarDataOps( 
		const Tensor2<Real,Int> & V_coords,
        const Tensor2<Int ,Int> & simplices,
		      Tensor2<Real,Int> & P_coords,
		      Tensor3<Real,Int> & P_hull_coords,
		      Tensor2<Real,Int> & P_near,
		      Tensor2<Real,Int> & P_far,
		SparseMatrixCSR<Real,Int> & DiffOp,
		SparseMatrixCSR<Real,Int> & AvOp 
	) const
    {
        ptic(ClassName()+"::ComputeNearFarDataOps");
        
        //Int size       = 2;
        //Int amb_dim    = 2;
        //Int dom_dim    = 1;
        
        auto job_ptr = BalanceWorkLoad<Int>( simplices.Dimension(0), ThreadCount() );
        
        #pragma omp parallel for num_threads( ThreadCount() )
        for( Int thread = 0; thread < ThreadCount(); ++thread )
        {
			Int  * restrict const AvOp_outer = AvOp.Outer().data();
			Int  * restrict const AvOp_inner = AvOp.Inner().data();
			Real * restrict const AvOp_value = AvOp.Values().data();

			Int  * restrict const DiffOp_outer = DiffOp.Outer().data();
			Int  * restrict const DiffOp_inner = DiffOp.Inner().data();
			Real * restrict const DiffOp_value = DiffOp.Value().data();

			const Real * restrict const V_coords__      = V_coords.data();
			
			const Int  * restrict const simplices__     = simplices.data();
				  Real * restrict const P_hull_coords__ = P_hull_coords.data();
				  Real * restrict const P_coords__      = P_coords.data();

			Real df       [2][1];
			Real dfdagger [1][2];
			Real g        [1][1];
			Real ginv     [1][1];

			Int simplex        [2];
			Int sorted_simplex [2];

			const Int i_begin = job_ptr[thread];
			const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {

				Real * restrict const near = P_near.data(i);                    
				Real * restrict const far  = P_far.data(i);

				simplex[0] = sorted_simplex[0] = simplices__[2*i +0];
				simplex[1] = sorted_simplex[1] = simplices__[2*i +1];
                  
                // sorting simplex so that we do not have to sort the sparse arrays to achieve CSR format later
                std::sort( sorted_simplex, sorted_simplex + 2 );

				AvOp_outer[i+1] = (i+1) * 2;  
                      
				AvOp_inner[2*i+0] = sorted_simplex[0];
				AvOp_inner[2*i+1] = sorted_simplex[1];

				AvOp_value[2*i+0] = 0.5;
				AvOp_value[2*i+1] = 0.5;

				DiffOp_outer[2*i+0] = (2 * i + 0) * 2;
				DiffOp_outer[2*i+1] = (2 * i + 1) * 2;

				DiffOp_inner[(i*2+0)*2+0] = sorted_simplex[0];
				DiffOp_inner[(i*2+1)*2+0] = sorted_simplex[0];
				DiffOp_inner[(i*2+0)*2+1] = sorted_simplex[1];
				DiffOp_inner[(i*2+1)*2+1] = sorted_simplex[1];

				near[1] = P_hull_coords__[4*i+0] = V_coords__[2*simplex[0]+0];
				near[2] = P_hull_coords__[4*i+1] = V_coords__[2*simplex[0]+1];
				near[3] = P_hull_coords__[4*i+2] = V_coords__[2*simplex[1]+0];
				near[4] = P_hull_coords__[4*i+3] = V_coords__[2*simplex[1]+1];

				far[1] = P_coords__[2*i+0] = 0.5 * ( P_hull_coords__[4*i+0] + P_hull_coords__[4*i+2] );
				far[2] = P_coords__[2*i+1] = 0.5 * ( P_hull_coords__[4*i+1] + P_hull_coords__[4*i+3] );

				df[0][0] = V_coords__[2*sorted_simplex[1]+0] - V_coords__[2*sorted_simplex[0]+0];
				df[1][0] = V_coords__[2*sorted_simplex[1]+1] - V_coords__[2*sorted_simplex[0]+1];

				g[0][0] = df[0][0] * df[0][0] + df[1][0] * df[1][0];

                near[0] = far[0] = sqrt( fabs(g[0][0]) );

                ginv[0][0] =  static_cast<Real>(1)/g[0][0];
                
                //  dfdagger = g^{-1} * df^T (1 x 2 matrix)
				dfdagger[0][0] = ginv[0][0] * df[0][0];
				dfdagger[0][1] = ginv[0][0] * df[1][0];

				near[5] = far[3] = static_cast<Real>(1.0) - df[0][0] * dfdagger[0][0];
				near[6] = far[4] =    - df[0][0] * dfdagger[0][1];
				near[7] = far[5] = static_cast<Real>(1.0) - df[1][0] * dfdagger[0][1];

                // derivative operator  (2 x 2 matrix)

                Real * Df = &DiffOp_value[ 4 * i ];

				Df[0] = - dfdagger[0][0];
				Df[1] =   dfdagger[0][0];
				Df[2] = - dfdagger[0][1];
				Df[3] =   dfdagger[0][1];

            }
        }
        ptoc(ClassName()+"::ComputeNearFarDataOps");
    }

	void DNearToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_near, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DNearToHulls");

        if( P_D_near.Dimension(1) != 8 )
        {
            eprint("in DNearToHulls: P_D_near.Dimension(1) != 8. Aborting");
        }

		const Real * restrict const V_coords__  = V_coords.data();
		const Int  * restrict const simplices__ = simplices.data();
		const Real * restrict const P_D_near__  = P_D_near.data();
			  Real * restrict const buffer__    = buffer.data();
        
        if( addTo )
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[2*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[2*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[2*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[2*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = s4 + s9;
				const Real s11 = sqrt(s10);
				const Real s12 = 1/s11;
				const Real s13 = s10*s11;
				const Real s14 = 1/s13;
				const Real s15 = s3*s4;
				const Real s16 = s10*s10;
				const Real s17 = 1/s16;
				const Real s18 = 1/s10;
				const Real s19 = P_D_near__[8*i+0];
				const Real s20 = P_D_near__[8*i+1];
				const Real s21 = P_D_near__[8*i+3];
				const Real s22 = P_D_near__[8*i+4];
				const Real s23 = P_D_near__[8*i+6];
				const Real s24 = P_D_near__[8*i+2];
				const Real s25 = P_D_near__[8*i+5];
				const Real s26 = -(s18*s4);
				const Real s27 = 1 + s26;
				const Real s28 = P_D_near__[8*i+7];
				const Real s29 = s8*s9;
				const Real s30 = -(s18*s9);
				const Real s31 = 1 + s30;
				buffer__[4*i+0] += -(s12*s19*s3) - s12*s2*s21*s3 + s20*(s11 - s0*s12*s3) + s25*(-(s12*s27*s3) + s11*(-2*s15*s17 + 2*s18*s3)) - s12*s24*s3*s5 - s12*s22*s3*s7 + s23*(s12*s8 - s14*s4*s8) + s28*(-(s12*s3*s31) - 2*s14*s3*s9);
				buffer__[4*i+1] += -(s12*s19*s8) - s0*s12*s20*s8 - s12*s2*s21*s8 - s12*s22*s7*s8 + s25*(-(s12*s27*s8) - 2*s14*s4*s8) + s24*(s11 - s12*s5*s8) + s28*(-(s12*s31*s8) + s11*(-2*s17*s29 + 2*s18*s8)) + s23*(s12*s3 - s14*s3*s9);
				buffer__[4*i+2] += s12*s19*s3 + s0*s12*s20*s3 + s21*(s11 + s12*s2*s3) + s25*(s12*s27*s3 + s11*(2*s15*s17 - 2*s18*s3)) + s12*s24*s3*s5 + s12*s22*s3*s7 + s23*(-(s12*s8) + s14*s4*s8) + s28*(s12*s3*s31 + 2*s14*s3*s9);
			}
		}
		else
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[2*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[2*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[2*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[2*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = s4 + s9;
				const Real s11 = sqrt(s10);
				const Real s12 = 1/s11;
				const Real s13 = s10*s11;
				const Real s14 = 1/s13;
				const Real s15 = s3*s4;
				const Real s16 = s10*s10;
				const Real s17 = 1/s16;
				const Real s18 = 1/s10;
				const Real s19 = P_D_near__[8*i+0];
				const Real s20 = P_D_near__[8*i+1];
				const Real s21 = P_D_near__[8*i+3];
				const Real s22 = P_D_near__[8*i+4];
				const Real s23 = P_D_near__[8*i+6];
				const Real s24 = P_D_near__[8*i+2];
				const Real s25 = P_D_near__[8*i+5];
				const Real s26 = -(s18*s4);
				const Real s27 = 1 + s26;
				const Real s28 = P_D_near__[8*i+7];
				const Real s29 = s8*s9;
				const Real s30 = -(s18*s9);
				const Real s31 = 1 + s30;
				buffer__[4*i+0] = -(s12*s19*s3) - s12*s2*s21*s3 + s20*(s11 - s0*s12*s3) + s25*(-(s12*s27*s3) + s11*(-2*s15*s17 + 2*s18*s3)) - s12*s24*s3*s5 - s12*s22*s3*s7 + s23*(s12*s8 - s14*s4*s8) + s28*(-(s12*s3*s31) - 2*s14*s3*s9);
				buffer__[4*i+1] = -(s12*s19*s8) - s0*s12*s20*s8 - s12*s2*s21*s8 - s12*s22*s7*s8 + s25*(-(s12*s27*s8) - 2*s14*s4*s8) + s24*(s11 - s12*s5*s8) + s28*(-(s12*s31*s8) + s11*(-2*s17*s29 + 2*s18*s8)) + s23*(s12*s3 - s14*s3*s9);
				buffer__[4*i+2] = s12*s19*s3 + s0*s12*s20*s3 + s21*(s11 + s12*s2*s3) + s25*(s12*s27*s3 + s11*(2*s15*s17 - 2*s18*s3)) + s12*s24*s3*s5 + s12*s22*s3*s7 + s23*(-(s12*s8) + s14*s4*s8) + s28*(s12*s3*s31 + 2*s14*s3*s9);
			}
		}

        ptoc(ClassName()+"::DNearToHulls");
        
    }

	void DFarToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_far, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DFarToHulls");

        if( P_D_far.Dimension(1) != 6 )
        {
            eprint("in DFarToHulls: P_D_far.Dimension(1) != 6. Aborting");
        }

		const Real * restrict const V_coords__  = V_coords.data();
		const Int  * restrict const simplices__ = simplices.data();
		const Real * restrict const P_D_far__   = P_D_far.data();
			  Real * restrict const buffer__    = buffer.data();
        
        if( addTo )
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[2*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[2*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[2*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[2*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = s4 + s9;
				const Real s11 = sqrt(s10);
				const Real s12 = 1/s11;
				const Real s13 = s10*s11;
				const Real s14 = 1/s13;
				const Real s15 = s3*s4;
				const Real s16 = s10*s10;
				const Real s17 = 1/s16;
				const Real s18 = 1/s10;
				const Real s19 = P_D_far__[6*i+0];
				const Real s20 = P_D_far__[6*i+1];
				const Real s21 = s0 + s2;
				const Real s22 = P_D_far__[6*i+4];
				const Real s23 = P_D_far__[6*i+2];
				const Real s24 = s5 + s7;
				const Real s25 = s11/2.;
				const Real s26 = P_D_far__[6*i+3];
				const Real s27 = -(s18*s4);
				const Real s28 = 1 + s27;
				const Real s29 = P_D_far__[6*i+5];
				const Real s30 = s8*s9;
				const Real s31 = -(s18*s9);
				const Real s32 = 1 + s31;
				buffer__[4*i+0] += -(s12*s19*s3) - (s12*s23*s24*s3)/2. + s20*(s25 - (s12*s21*s3)/2.) + s26*(-(s12*s28*s3) + s11*(-2*s15*s17 + 2*s18*s3)) + s22*(s12*s8 - s14*s4*s8) + s29*(-(s12*s3*s32) - 2*s14*s3*s9);
				buffer__[4*i+1] += -(s12*s19*s8) - (s12*s20*s21*s8)/2. + s23*(s25 - (s12*s24*s8)/2.) + s26*(-(s12*s28*s8) - 2*s14*s4*s8) + s29*(-(s12*s32*s8) + s11*(-2*s17*s30 + 2*s18*s8)) + s22*(s12*s3 - s14*s3*s9);
				buffer__[4*i+2] += s12*s19*s3 + (s12*s23*s24*s3)/2. + s20*(s25 + (s12*s21*s3)/2.) + s26*(s12*s28*s3 + s11*(2*s15*s17 - 2*s18*s3)) + s22*(-(s12*s8) + s14*s4*s8) + s29*(s12*s3*s32 + 2*s14*s3*s9);
			}
		}
		else
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[2*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[2*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[2*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[2*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = s4 + s9;
				const Real s11 = sqrt(s10);
				const Real s12 = 1/s11;
				const Real s13 = s10*s11;
				const Real s14 = 1/s13;
				const Real s15 = s3*s4;
				const Real s16 = s10*s10;
				const Real s17 = 1/s16;
				const Real s18 = 1/s10;
				const Real s19 = P_D_far__[6*i+0];
				const Real s20 = P_D_far__[6*i+1];
				const Real s21 = s0 + s2;
				const Real s22 = P_D_far__[6*i+4];
				const Real s23 = P_D_far__[6*i+2];
				const Real s24 = s5 + s7;
				const Real s25 = s11/2.;
				const Real s26 = P_D_far__[6*i+3];
				const Real s27 = -(s18*s4);
				const Real s28 = 1 + s27;
				const Real s29 = P_D_far__[6*i+5];
				const Real s30 = s8*s9;
				const Real s31 = -(s18*s9);
				const Real s32 = 1 + s31;
				buffer__[4*i+0] = -(s12*s19*s3) - (s12*s23*s24*s3)/2. + s20*(s25 - (s12*s21*s3)/2.) + s26*(-(s12*s28*s3) + s11*(-2*s15*s17 + 2*s18*s3)) + s22*(s12*s8 - s14*s4*s8) + s29*(-(s12*s3*s32) - 2*s14*s3*s9);
				buffer__[4*i+1] = -(s12*s19*s8) - (s12*s20*s21*s8)/2. + s23*(s25 - (s12*s24*s8)/2.) + s26*(-(s12*s28*s8) - 2*s14*s4*s8) + s29*(-(s12*s32*s8) + s11*(-2*s17*s30 + 2*s18*s8)) + s22*(s12*s3 - s14*s3*s9);
				buffer__[4*i+2] = s12*s19*s3 + (s12*s23*s24*s3)/2. + s20*(s25 + (s12*s21*s3)/2.) + s26*(s12*s28*s3 + s11*(2*s15*s17 - 2*s18*s3)) + s22*(-(s12*s8) + s14*s4*s8) + s29*(s12*s3*s32 + 2*s14*s3*s9);
			}
		}

        ptoc(ClassName()+"::DFarToHulls");
        
    }

	}; // SimplicialMeshDetails<1,2,Real,Int>

//----------------------------------------------------------------------------------------------

    template<typename Real, typename Int>
    struct SimplicialMeshDetails<1,3,Real,Int>
    {
	private:

		const Int thread_count = 1;

	public:

		SimplicialMeshDetails( const Int thread_count_ = 1 ) 
		:
			thread_count(std::max(static_cast<Int>(1),thread_count_))
		{}
	
		inline Int ThreadCount() const
		{
			return thread_count;
		}
	
		inline std::string ClassName() const
        {
            return "SimplicialMeshDetails<1,3,"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
		
	void ComputeNearFarData( 
		const Tensor2<Real,Int> & V_coords,
		const Tensor2<Int ,Int> & simplices,
			  Tensor2<Real,Int> & P_near,
			  Tensor2<Real,Int> & P_far
	) const
    {
        ptic(ClassName()+"::ComputeNearFarData");
        
        //Int size       = 2;
        //Int amb_dim    = 3;
        //Int dom_dim    = 1;

        auto job_ptr = BalanceWorkLoad<Int>( simplices.Dimension(0), ThreadCount() );
        
        #pragma omp parallel for num_threads( ThreadCount() )
        for( Int thread = 0; thread < ThreadCount(); ++thread )
        {
			const Real * restrict const V_coords__      = V_coords.data();	
			const Int  * restrict const simplices__     = simplices.data();

			Real hull    [2][3];
			Real df      [3][1];
			Real dfdagger[1][3];
			Real g       [1][1];
			Real ginv    [1][1];

			Int simplex  [2];
			
			const Int i_begin = job_ptr[thread];
			const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {
				Real * restrict const near = P_near.data(i);                    
				Real * restrict const far  = P_far.data(i);   
            
				simplex[0] = simplices__[2*i +0];
				simplex[1] = simplices__[2*i +1];

				near[1] = hull[0][0] = V_coords__[3*simplex[0]+0];
				near[2] = hull[0][1] = V_coords__[3*simplex[0]+1];
				near[3] = hull[0][2] = V_coords__[3*simplex[0]+2];
				near[4] = hull[1][0] = V_coords__[3*simplex[1]+0];
				near[5] = hull[1][1] = V_coords__[3*simplex[1]+1];
				near[6] = hull[1][2] = V_coords__[3*simplex[1]+2];

				far[1] = static_cast<Real>(0.5) * ( hull[0][0] + hull[1][0] );
				far[2] = static_cast<Real>(0.5) * ( hull[0][1] + hull[1][1] );
				far[3] = static_cast<Real>(0.5) * ( hull[0][2] + hull[1][2] );

				df[0][0] = hull[1][0] - hull[0][0];
				df[1][0] = hull[1][1] - hull[0][1];
				df[2][0] = hull[1][2] - hull[0][2];

				g[0][0] = df[0][0] * df[0][0] + df[1][0] * df[1][0] + df[2][0] * df[2][0];

   
                near[0] = far[0] = sqrt( fabs(g[0][0]) );

                ginv[0][0] = static_cast<Real>(1)/g[0][0];
                
                //  dfdagger = g^{-1} * df^T (1 x 3 matrix)
				dfdagger[0][0] = ginv[0][0] * df[0][0];
				dfdagger[0][1] = ginv[0][0] * df[1][0];
				dfdagger[0][2] = ginv[0][0] * df[2][0];
            
				near[ 7] = far[ 4]  = static_cast<Real>(1) - df[0][0] * dfdagger[0][0];
				near[ 8] = far[ 5]  =    - df[0][0] * dfdagger[0][1];
				near[ 9] = far[ 6]  =    - df[0][0] * dfdagger[0][2];
				near[10] = far[ 7]  = static_cast<Real>(1) - df[1][0] * dfdagger[0][1];
				near[11] = far[ 8]  =    - df[1][0] * dfdagger[0][2];
				near[12] = far[ 9]  = static_cast<Real>(1) - df[2][0] * dfdagger[0][2];

            } // for( Int i = i_begin; i < i_end; ++i )

		} // #pragma omp parallel for num_threads( ThreadCount() )

        ptoc(ClassName()+"::ComputeNearFarData");
    }

    void ComputeNearFarDataOps( 
		const Tensor2<Real,Int> & V_coords,
        const Tensor2<Int ,Int> & simplices,
		      Tensor2<Real,Int> & P_coords,
		      Tensor3<Real,Int> & P_hull_coords,
		      Tensor2<Real,Int> & P_near,
		      Tensor2<Real,Int> & P_far,
		SparseMatrixCSR<Real,Int> & DiffOp,
		SparseMatrixCSR<Real,Int> & AvOp 
	) const
    {
        ptic(ClassName()+"::ComputeNearFarDataOps");
        
        //Int size       = 2;
        //Int amb_dim    = 3;
        //Int dom_dim    = 1;
        
        auto job_ptr = BalanceWorkLoad<Int>( simplices.Dimension(0), ThreadCount() );
        
        #pragma omp parallel for num_threads( ThreadCount() )
        for( Int thread = 0; thread < ThreadCount(); ++thread )
        {
			Int  * restrict const AvOp_outer = AvOp.Outer().data();
			Int  * restrict const AvOp_inner = AvOp.Inner().data();
			Real * restrict const AvOp_value = AvOp.Values().data();

			Int  * restrict const DiffOp_outer = DiffOp.Outer().data();
			Int  * restrict const DiffOp_inner = DiffOp.Inner().data();
			Real * restrict const DiffOp_value = DiffOp.Value().data();

			const Real * restrict const V_coords__      = V_coords.data();
			
			const Int  * restrict const simplices__     = simplices.data();
				  Real * restrict const P_hull_coords__ = P_hull_coords.data();
				  Real * restrict const P_coords__      = P_coords.data();

			Real df       [3][1];
			Real dfdagger [1][3];
			Real g        [1][1];
			Real ginv     [1][1];

			Int simplex        [2];
			Int sorted_simplex [2];

			const Int i_begin = job_ptr[thread];
			const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {

				Real * restrict const near = P_near.data(i);                    
				Real * restrict const far  = P_far.data(i);

				simplex[0] = sorted_simplex[0] = simplices__[2*i +0];
				simplex[1] = sorted_simplex[1] = simplices__[2*i +1];
                  
                // sorting simplex so that we do not have to sort the sparse arrays to achieve CSR format later
                std::sort( sorted_simplex, sorted_simplex + 2 );

				AvOp_outer[i+1] = (i+1) * 2;  
                      
				AvOp_inner[2*i+0] = sorted_simplex[0];
				AvOp_inner[2*i+1] = sorted_simplex[1];

				AvOp_value[2*i+0] = 0.5;
				AvOp_value[2*i+1] = 0.5;

				DiffOp_outer[3*i+0] = (3 * i + 0) * 2;
				DiffOp_outer[3*i+1] = (3 * i + 1) * 2;
				DiffOp_outer[3*i+2] = (3 * i + 2) * 2;

				DiffOp_inner[(i*3+0)*2+0] = sorted_simplex[0];
				DiffOp_inner[(i*3+1)*2+0] = sorted_simplex[0];
				DiffOp_inner[(i*3+2)*2+0] = sorted_simplex[0];
				DiffOp_inner[(i*3+0)*2+1] = sorted_simplex[1];
				DiffOp_inner[(i*3+1)*2+1] = sorted_simplex[1];
				DiffOp_inner[(i*3+2)*2+1] = sorted_simplex[1];

				near[1] = P_hull_coords__[6*i+0] = V_coords__[3*simplex[0]+0];
				near[2] = P_hull_coords__[6*i+1] = V_coords__[3*simplex[0]+1];
				near[3] = P_hull_coords__[6*i+2] = V_coords__[3*simplex[0]+2];
				near[4] = P_hull_coords__[6*i+3] = V_coords__[3*simplex[1]+0];
				near[5] = P_hull_coords__[6*i+4] = V_coords__[3*simplex[1]+1];
				near[6] = P_hull_coords__[6*i+5] = V_coords__[3*simplex[1]+2];

				far[1] = P_coords__[3*i+0] = 0.5 * ( P_hull_coords__[6*i+0] + P_hull_coords__[6*i+3] );
				far[2] = P_coords__[3*i+1] = 0.5 * ( P_hull_coords__[6*i+1] + P_hull_coords__[6*i+4] );
				far[3] = P_coords__[3*i+2] = 0.5 * ( P_hull_coords__[6*i+2] + P_hull_coords__[6*i+5] );

				df[0][0] = V_coords__[3*sorted_simplex[1]+0] - V_coords__[3*sorted_simplex[0]+0];
				df[1][0] = V_coords__[3*sorted_simplex[1]+1] - V_coords__[3*sorted_simplex[0]+1];
				df[2][0] = V_coords__[3*sorted_simplex[1]+2] - V_coords__[3*sorted_simplex[0]+2];

				g[0][0] = df[0][0] * df[0][0] + df[1][0] * df[1][0] + df[2][0] * df[2][0];

                near[0] = far[0] = sqrt( fabs(g[0][0]) );

                ginv[0][0] =  static_cast<Real>(1)/g[0][0];
                
                //  dfdagger = g^{-1} * df^T (1 x 3 matrix)
				dfdagger[0][0] = ginv[0][0] * df[0][0];
				dfdagger[0][1] = ginv[0][0] * df[1][0];
				dfdagger[0][2] = ginv[0][0] * df[2][0];

				near[ 7] = far[ 4] = static_cast<Real>(1.0) - df[0][0] * dfdagger[0][0];
				near[ 8] = far[ 5] =    - df[0][0] * dfdagger[0][1];
				near[ 9] = far[ 6] =    - df[0][0] * dfdagger[0][2];
				near[10] = far[ 7] = static_cast<Real>(1.0) - df[1][0] * dfdagger[0][1];
				near[11] = far[ 8] =    - df[1][0] * dfdagger[0][2];
				near[12] = far[ 9] = static_cast<Real>(1.0) - df[2][0] * dfdagger[0][2];

                // derivative operator  (3 x 2 matrix)

                Real * Df = &DiffOp_value[ 6 * i ];

				Df[ 0] = - dfdagger[0][0];
				Df[ 1] =   dfdagger[0][0];
				Df[ 2] = - dfdagger[0][1];
				Df[ 3] =   dfdagger[0][1];
				Df[ 4] = - dfdagger[0][2];
				Df[ 5] =   dfdagger[0][2];

            }
        }
        ptoc(ClassName()+"::ComputeNearFarDataOps");
    }

	void DNearToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_near, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DNearToHulls");

        if( P_D_near.Dimension(1) != 13 )
        {
            eprint("in DNearToHulls: P_D_near.Dimension(1) != 13. Aborting");
        }

		const Real * restrict const V_coords__  = V_coords.data();
		const Int  * restrict const simplices__ = simplices.data();
		const Real * restrict const P_D_near__  = P_D_near.data();
			  Real * restrict const buffer__    = buffer.data();
        
        if( addTo )
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[3*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[3*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[3*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = V_coords__[3*simplices__[2*i+0]+2];
				const Real s11 = -s10;
				const Real s12 = V_coords__[3*simplices__[2*i+1]+2];
				const Real s13 = s11 + s12;
				const Real s14 = s13*s13;
				const Real s15 = s14 + s4 + s9;
				const Real s16 = sqrt(s15);
				const Real s17 = s15*s16;
				const Real s18 = 1/s17;
				const Real s19 = 1/s16;
				const Real s20 = s3*s4;
				const Real s21 = s15*s15;
				const Real s22 = 1/s21;
				const Real s23 = 1/s15;
				const Real s24 = P_D_near__[13*i+9];
				const Real s25 = P_D_near__[13*i+0];
				const Real s26 = P_D_near__[13*i+1];
				const Real s27 = P_D_near__[13*i+3];
				const Real s28 = P_D_near__[13*i+4];
				const Real s29 = P_D_near__[13*i+5];
				const Real s30 = P_D_near__[13*i+6];
				const Real s31 = P_D_near__[13*i+8];
				const Real s32 = P_D_near__[13*i+11];
				const Real s33 = s13*s19;
				const Real s34 = P_D_near__[13*i+2];
				const Real s35 = P_D_near__[13*i+7];
				const Real s36 = -(s23*s4);
				const Real s37 = 1 + s36;
				const Real s38 = P_D_near__[13*i+10];
				const Real s39 = s8*s9;
				const Real s40 = -(s23*s9);
				const Real s41 = 1 + s40;
				const Real s42 = P_D_near__[13*i+12];
				const Real s43 = -(s14*s23);
				const Real s44 = 1 + s43;
				const Real s45 = s19*s3;
				const Real s46 = s19*s8;
				const Real s47 = s13*s14;
				const Real s48 = -(s13*s19);
				const Real s49 = -(s19*s3);
				const Real s50 = -(s19*s8);
				buffer__[6*i+0] += -(s19*s25*s3) - s10*s19*s27*s3 - s19*s2*s28*s3 + s26*(s16 - s0*s19*s3) - s12*s19*s3*s30 + s35*(s16*(-2*s20*s22 + 2*s23*s3) - s19*s3*s37) + s24*(s33 - s13*s18*s4) + s42*(-2*s14*s18*s3 - s19*s3*s44) - s19*s3*s34*s5 - s19*s29*s3*s7 - s13*s18*s3*s32*s8 + s31*(s46 - s18*s4*s8) + s38*(-(s19*s3*s41) - 2*s18*s3*s9);
				buffer__[6*i+1] += -(s19*s25*s8) - s0*s19*s26*s8 - s10*s19*s27*s8 - s19*s2*s28*s8 - s13*s18*s24*s3*s8 - s12*s19*s30*s8 - s19*s29*s7*s8 + s35*(-(s19*s37*s8) - 2*s18*s4*s8) + s42*(-2*s14*s18*s8 - s19*s44*s8) + s34*(s16 - s19*s5*s8) + s38*(-(s19*s41*s8) + s16*(-2*s22*s39 + 2*s23*s8)) + s32*(s33 - s13*s18*s9) + s31*(s45 - s18*s3*s9);
				buffer__[6*i+2] += -(s13*s19*s25) - s0*s13*s19*s26 + (s16 - s10*s13*s19)*s27 - s13*s19*s2*s28 - s12*s13*s19*s30 + s35*(-(s13*s19*s37) - 2*s13*s18*s4) + s24*(-(s14*s18*s3) + s45) + s42*(-(s13*s19*s44) + s16*(2*s13*s23 - 2*s22*s47)) - s13*s19*s34*s5 - s13*s19*s29*s7 - s13*s18*s3*s31*s8 + s32*(s46 - s14*s18*s8) + s38*(-(s13*s19*s41) - 2*s13*s18*s9);
				buffer__[6*i+3] += s19*s25*s3 + s0*s19*s26*s3 + s10*s19*s27*s3 + s28*(s16 + s19*s2*s3) + s12*s19*s3*s30 + s35*(s16*(2*s20*s22 - 2*s23*s3) + s19*s3*s37) + s42*(2*s14*s18*s3 + s19*s3*s44) + s24*(s13*s18*s4 + s48) + s19*s3*s34*s5 + s19*s29*s3*s7 + s13*s18*s3*s32*s8 + s31*(s50 + s18*s4*s8) + s38*(s19*s3*s41 + 2*s18*s3*s9);
				buffer__[6*i+4] += s19*s25*s8 + s0*s19*s26*s8 + s10*s19*s27*s8 + s19*s2*s28*s8 + s13*s18*s24*s3*s8 + s12*s19*s30*s8 + s19*s34*s5*s8 + s35*(s19*s37*s8 + 2*s18*s4*s8) + s42*(2*s14*s18*s8 + s19*s44*s8) + s29*(s16 + s19*s7*s8) + s38*(s19*s41*s8 + s16*(2*s22*s39 - 2*s23*s8)) + s32*(s48 + s13*s18*s9) + s31*(s49 + s18*s3*s9);
				buffer__[6*i+5] += s13*s19*s25 + s0*s13*s19*s26 + s10*s13*s19*s27 + s13*s19*s2*s28 + (s16 + s12*s13*s19)*s30 + s35*(s13*s19*s37 + 2*s13*s18*s4) + s42*(s13*s19*s44 + s16*(-2*s13*s23 + 2*s22*s47)) + s24*(s14*s18*s3 + s49) + s13*s19*s34*s5 + s13*s19*s29*s7 + s13*s18*s3*s31*s8 + s32*(s50 + s14*s18*s8) + s38*(s13*s19*s41 + 2*s13*s18*s9);
			}
		}
		else
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[3*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[3*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[3*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = V_coords__[3*simplices__[2*i+0]+2];
				const Real s11 = -s10;
				const Real s12 = V_coords__[3*simplices__[2*i+1]+2];
				const Real s13 = s11 + s12;
				const Real s14 = s13*s13;
				const Real s15 = s14 + s4 + s9;
				const Real s16 = sqrt(s15);
				const Real s17 = s15*s16;
				const Real s18 = 1/s17;
				const Real s19 = 1/s16;
				const Real s20 = s3*s4;
				const Real s21 = s15*s15;
				const Real s22 = 1/s21;
				const Real s23 = 1/s15;
				const Real s24 = P_D_near__[13*i+9];
				const Real s25 = P_D_near__[13*i+0];
				const Real s26 = P_D_near__[13*i+1];
				const Real s27 = P_D_near__[13*i+3];
				const Real s28 = P_D_near__[13*i+4];
				const Real s29 = P_D_near__[13*i+5];
				const Real s30 = P_D_near__[13*i+6];
				const Real s31 = P_D_near__[13*i+8];
				const Real s32 = P_D_near__[13*i+11];
				const Real s33 = s13*s19;
				const Real s34 = P_D_near__[13*i+2];
				const Real s35 = P_D_near__[13*i+7];
				const Real s36 = -(s23*s4);
				const Real s37 = 1 + s36;
				const Real s38 = P_D_near__[13*i+10];
				const Real s39 = s8*s9;
				const Real s40 = -(s23*s9);
				const Real s41 = 1 + s40;
				const Real s42 = P_D_near__[13*i+12];
				const Real s43 = -(s14*s23);
				const Real s44 = 1 + s43;
				const Real s45 = s19*s3;
				const Real s46 = s19*s8;
				const Real s47 = s13*s14;
				const Real s48 = -(s13*s19);
				const Real s49 = -(s19*s3);
				const Real s50 = -(s19*s8);
				buffer__[6*i+0] = -(s19*s25*s3) - s10*s19*s27*s3 - s19*s2*s28*s3 + s26*(s16 - s0*s19*s3) - s12*s19*s3*s30 + s35*(s16*(-2*s20*s22 + 2*s23*s3) - s19*s3*s37) + s24*(s33 - s13*s18*s4) + s42*(-2*s14*s18*s3 - s19*s3*s44) - s19*s3*s34*s5 - s19*s29*s3*s7 - s13*s18*s3*s32*s8 + s31*(s46 - s18*s4*s8) + s38*(-(s19*s3*s41) - 2*s18*s3*s9);
				buffer__[6*i+1] = -(s19*s25*s8) - s0*s19*s26*s8 - s10*s19*s27*s8 - s19*s2*s28*s8 - s13*s18*s24*s3*s8 - s12*s19*s30*s8 - s19*s29*s7*s8 + s35*(-(s19*s37*s8) - 2*s18*s4*s8) + s42*(-2*s14*s18*s8 - s19*s44*s8) + s34*(s16 - s19*s5*s8) + s38*(-(s19*s41*s8) + s16*(-2*s22*s39 + 2*s23*s8)) + s32*(s33 - s13*s18*s9) + s31*(s45 - s18*s3*s9);
				buffer__[6*i+2] = -(s13*s19*s25) - s0*s13*s19*s26 + (s16 - s10*s13*s19)*s27 - s13*s19*s2*s28 - s12*s13*s19*s30 + s35*(-(s13*s19*s37) - 2*s13*s18*s4) + s24*(-(s14*s18*s3) + s45) + s42*(-(s13*s19*s44) + s16*(2*s13*s23 - 2*s22*s47)) - s13*s19*s34*s5 - s13*s19*s29*s7 - s13*s18*s3*s31*s8 + s32*(s46 - s14*s18*s8) + s38*(-(s13*s19*s41) - 2*s13*s18*s9);
				buffer__[6*i+3] = s19*s25*s3 + s0*s19*s26*s3 + s10*s19*s27*s3 + s28*(s16 + s19*s2*s3) + s12*s19*s3*s30 + s35*(s16*(2*s20*s22 - 2*s23*s3) + s19*s3*s37) + s42*(2*s14*s18*s3 + s19*s3*s44) + s24*(s13*s18*s4 + s48) + s19*s3*s34*s5 + s19*s29*s3*s7 + s13*s18*s3*s32*s8 + s31*(s50 + s18*s4*s8) + s38*(s19*s3*s41 + 2*s18*s3*s9);
				buffer__[6*i+4] = s19*s25*s8 + s0*s19*s26*s8 + s10*s19*s27*s8 + s19*s2*s28*s8 + s13*s18*s24*s3*s8 + s12*s19*s30*s8 + s19*s34*s5*s8 + s35*(s19*s37*s8 + 2*s18*s4*s8) + s42*(2*s14*s18*s8 + s19*s44*s8) + s29*(s16 + s19*s7*s8) + s38*(s19*s41*s8 + s16*(2*s22*s39 - 2*s23*s8)) + s32*(s48 + s13*s18*s9) + s31*(s49 + s18*s3*s9);
				buffer__[6*i+5] = s13*s19*s25 + s0*s13*s19*s26 + s10*s13*s19*s27 + s13*s19*s2*s28 + (s16 + s12*s13*s19)*s30 + s35*(s13*s19*s37 + 2*s13*s18*s4) + s42*(s13*s19*s44 + s16*(-2*s13*s23 + 2*s22*s47)) + s24*(s14*s18*s3 + s49) + s13*s19*s34*s5 + s13*s19*s29*s7 + s13*s18*s3*s31*s8 + s32*(s50 + s14*s18*s8) + s38*(s13*s19*s41 + 2*s13*s18*s9);
			}
		}

        ptoc(ClassName()+"::DNearToHulls");
        
    }

	void DFarToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_far, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DFarToHulls");

        if( P_D_far.Dimension(1) != 10 )
        {
            eprint("in DFarToHulls: P_D_far.Dimension(1) != 10. Aborting");
        }

		const Real * restrict const V_coords__  = V_coords.data();
		const Int  * restrict const simplices__ = simplices.data();
		const Real * restrict const P_D_far__   = P_D_far.data();
			  Real * restrict const buffer__    = buffer.data();
        
        if( addTo )
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[3*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[3*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[3*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = V_coords__[3*simplices__[2*i+0]+2];
				const Real s11 = -s10;
				const Real s12 = V_coords__[3*simplices__[2*i+1]+2];
				const Real s13 = s11 + s12;
				const Real s14 = s13*s13;
				const Real s15 = s14 + s4 + s9;
				const Real s16 = sqrt(s15);
				const Real s17 = s15*s16;
				const Real s18 = 1/s17;
				const Real s19 = 1/s16;
				const Real s20 = s3*s4;
				const Real s21 = s15*s15;
				const Real s22 = 1/s21;
				const Real s23 = 1/s15;
				const Real s24 = P_D_far__[10*i+6];
				const Real s25 = P_D_far__[10*i+0];
				const Real s26 = P_D_far__[10*i+1];
				const Real s27 = s0 + s2;
				const Real s28 = P_D_far__[10*i+3];
				const Real s29 = s10 + s12;
				const Real s30 = P_D_far__[10*i+5];
				const Real s31 = P_D_far__[10*i+8];
				const Real s32 = s13*s19;
				const Real s33 = P_D_far__[10*i+2];
				const Real s34 = s5 + s7;
				const Real s35 = s16/2.;
				const Real s36 = P_D_far__[10*i+4];
				const Real s37 = -(s23*s4);
				const Real s38 = 1 + s37;
				const Real s39 = P_D_far__[10*i+7];
				const Real s40 = s8*s9;
				const Real s41 = -(s23*s9);
				const Real s42 = 1 + s41;
				const Real s43 = P_D_far__[10*i+9];
				const Real s44 = -(s14*s23);
				const Real s45 = 1 + s44;
				const Real s46 = s19*s3;
				const Real s47 = s19*s8;
				const Real s48 = s13*s14;
				const Real s49 = -(s13*s19);
				const Real s50 = -(s19*s3);
				const Real s51 = -(s19*s8);
				buffer__[6*i+0] += -(s19*s25*s3) - (s19*s28*s29*s3)/2. - (s19*s3*s33*s34)/2. + s26*(-0.5*(s19*s27*s3) + s35) + s36*(s16*(-2*s20*s22 + 2*s23*s3) - s19*s3*s38) + s24*(s32 - s13*s18*s4) + s43*(-2*s14*s18*s3 - s19*s3*s45) - s13*s18*s3*s31*s8 + s30*(s47 - s18*s4*s8) + s39*(-(s19*s3*s42) - 2*s18*s3*s9);
				buffer__[6*i+1] += -(s19*s25*s8) - (s19*s26*s27*s8)/2. - (s19*s28*s29*s8)/2. - s13*s18*s24*s3*s8 + s33*(s35 - (s19*s34*s8)/2.) + s36*(-(s19*s38*s8) - 2*s18*s4*s8) + s43*(-2*s14*s18*s8 - s19*s45*s8) + s39*(-(s19*s42*s8) + s16*(-2*s22*s40 + 2*s23*s8)) + s31*(s32 - s13*s18*s9) + s30*(s46 - s18*s3*s9);
				buffer__[6*i+2] += -(s13*s19*s25) - (s13*s19*s26*s27)/2. - (s13*s19*s33*s34)/2. + s28*(-0.5*(s13*s19*s29) + s35) + s36*(-(s13*s19*s38) - 2*s13*s18*s4) + s24*(-(s14*s18*s3) + s46) + s43*(-(s13*s19*s45) + s16*(2*s13*s23 - 2*s22*s48)) - s13*s18*s3*s30*s8 + s31*(s47 - s14*s18*s8) + s39*(-(s13*s19*s42) - 2*s13*s18*s9);
				buffer__[6*i+3] += s19*s25*s3 + (s19*s28*s29*s3)/2. + (s19*s3*s33*s34)/2. + s26*((s19*s27*s3)/2. + s35) + s36*(s16*(2*s20*s22 - 2*s23*s3) + s19*s3*s38) + s43*(2*s14*s18*s3 + s19*s3*s45) + s24*(s13*s18*s4 + s49) + s13*s18*s3*s31*s8 + s30*(s51 + s18*s4*s8) + s39*(s19*s3*s42 + 2*s18*s3*s9);
				buffer__[6*i+4] += s19*s25*s8 + (s19*s26*s27*s8)/2. + (s19*s28*s29*s8)/2. + s13*s18*s24*s3*s8 + s33*(s35 + (s19*s34*s8)/2.) + s36*(s19*s38*s8 + 2*s18*s4*s8) + s43*(2*s14*s18*s8 + s19*s45*s8) + s39*(s19*s42*s8 + s16*(2*s22*s40 - 2*s23*s8)) + s31*(s49 + s13*s18*s9) + s30*(s50 + s18*s3*s9);
				buffer__[6*i+5] += s13*s19*s25 + (s13*s19*s26*s27)/2. + (s13*s19*s33*s34)/2. + s28*((s13*s19*s29)/2. + s35) + s36*(s13*s19*s38 + 2*s13*s18*s4) + s43*(s13*s19*s45 + s16*(-2*s13*s23 + 2*s22*s48)) + s24*(s14*s18*s3 + s50) + s13*s18*s3*s30*s8 + s31*(s51 + s14*s18*s8) + s39*(s13*s19*s42 + 2*s13*s18*s9);
			}
		}
		else
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[2*i+0]+0];
				const Real s1 = -s0;
				const Real s2 = V_coords__[3*simplices__[2*i+1]+0];
				const Real s3 = s1 + s2;
				const Real s4 = s3*s3;
				const Real s5 = V_coords__[3*simplices__[2*i+0]+1];
				const Real s6 = -s5;
				const Real s7 = V_coords__[3*simplices__[2*i+1]+1];
				const Real s8 = s6 + s7;
				const Real s9 = s8*s8;
				const Real s10 = V_coords__[3*simplices__[2*i+0]+2];
				const Real s11 = -s10;
				const Real s12 = V_coords__[3*simplices__[2*i+1]+2];
				const Real s13 = s11 + s12;
				const Real s14 = s13*s13;
				const Real s15 = s14 + s4 + s9;
				const Real s16 = sqrt(s15);
				const Real s17 = s15*s16;
				const Real s18 = 1/s17;
				const Real s19 = 1/s16;
				const Real s20 = s3*s4;
				const Real s21 = s15*s15;
				const Real s22 = 1/s21;
				const Real s23 = 1/s15;
				const Real s24 = P_D_far__[10*i+6];
				const Real s25 = P_D_far__[10*i+0];
				const Real s26 = P_D_far__[10*i+1];
				const Real s27 = s0 + s2;
				const Real s28 = P_D_far__[10*i+3];
				const Real s29 = s10 + s12;
				const Real s30 = P_D_far__[10*i+5];
				const Real s31 = P_D_far__[10*i+8];
				const Real s32 = s13*s19;
				const Real s33 = P_D_far__[10*i+2];
				const Real s34 = s5 + s7;
				const Real s35 = s16/2.;
				const Real s36 = P_D_far__[10*i+4];
				const Real s37 = -(s23*s4);
				const Real s38 = 1 + s37;
				const Real s39 = P_D_far__[10*i+7];
				const Real s40 = s8*s9;
				const Real s41 = -(s23*s9);
				const Real s42 = 1 + s41;
				const Real s43 = P_D_far__[10*i+9];
				const Real s44 = -(s14*s23);
				const Real s45 = 1 + s44;
				const Real s46 = s19*s3;
				const Real s47 = s19*s8;
				const Real s48 = s13*s14;
				const Real s49 = -(s13*s19);
				const Real s50 = -(s19*s3);
				const Real s51 = -(s19*s8);
				buffer__[6*i+0] = -(s19*s25*s3) - (s19*s28*s29*s3)/2. - (s19*s3*s33*s34)/2. + s26*(-0.5*(s19*s27*s3) + s35) + s36*(s16*(-2*s20*s22 + 2*s23*s3) - s19*s3*s38) + s24*(s32 - s13*s18*s4) + s43*(-2*s14*s18*s3 - s19*s3*s45) - s13*s18*s3*s31*s8 + s30*(s47 - s18*s4*s8) + s39*(-(s19*s3*s42) - 2*s18*s3*s9);
				buffer__[6*i+1] = -(s19*s25*s8) - (s19*s26*s27*s8)/2. - (s19*s28*s29*s8)/2. - s13*s18*s24*s3*s8 + s33*(s35 - (s19*s34*s8)/2.) + s36*(-(s19*s38*s8) - 2*s18*s4*s8) + s43*(-2*s14*s18*s8 - s19*s45*s8) + s39*(-(s19*s42*s8) + s16*(-2*s22*s40 + 2*s23*s8)) + s31*(s32 - s13*s18*s9) + s30*(s46 - s18*s3*s9);
				buffer__[6*i+2] = -(s13*s19*s25) - (s13*s19*s26*s27)/2. - (s13*s19*s33*s34)/2. + s28*(-0.5*(s13*s19*s29) + s35) + s36*(-(s13*s19*s38) - 2*s13*s18*s4) + s24*(-(s14*s18*s3) + s46) + s43*(-(s13*s19*s45) + s16*(2*s13*s23 - 2*s22*s48)) - s13*s18*s3*s30*s8 + s31*(s47 - s14*s18*s8) + s39*(-(s13*s19*s42) - 2*s13*s18*s9);
				buffer__[6*i+3] = s19*s25*s3 + (s19*s28*s29*s3)/2. + (s19*s3*s33*s34)/2. + s26*((s19*s27*s3)/2. + s35) + s36*(s16*(2*s20*s22 - 2*s23*s3) + s19*s3*s38) + s43*(2*s14*s18*s3 + s19*s3*s45) + s24*(s13*s18*s4 + s49) + s13*s18*s3*s31*s8 + s30*(s51 + s18*s4*s8) + s39*(s19*s3*s42 + 2*s18*s3*s9);
				buffer__[6*i+4] = s19*s25*s8 + (s19*s26*s27*s8)/2. + (s19*s28*s29*s8)/2. + s13*s18*s24*s3*s8 + s33*(s35 + (s19*s34*s8)/2.) + s36*(s19*s38*s8 + 2*s18*s4*s8) + s43*(2*s14*s18*s8 + s19*s45*s8) + s39*(s19*s42*s8 + s16*(2*s22*s40 - 2*s23*s8)) + s31*(s49 + s13*s18*s9) + s30*(s50 + s18*s3*s9);
				buffer__[6*i+5] = s13*s19*s25 + (s13*s19*s26*s27)/2. + (s13*s19*s33*s34)/2. + s28*((s13*s19*s29)/2. + s35) + s36*(s13*s19*s38 + 2*s13*s18*s4) + s43*(s13*s19*s45 + s16*(-2*s13*s23 + 2*s22*s48)) + s24*(s14*s18*s3 + s50) + s13*s18*s3*s30*s8 + s31*(s51 + s14*s18*s8) + s39*(s13*s19*s42 + 2*s13*s18*s9);
			}
		}

        ptoc(ClassName()+"::DFarToHulls");
        
    }

	}; // SimplicialMeshDetails<1,3,Real,Int>

//----------------------------------------------------------------------------------------------

    template<typename Real, typename Int>
    struct SimplicialMeshDetails<2,3,Real,Int>
    {
	private:

		const Int thread_count = 1;

	public:

		SimplicialMeshDetails( const Int thread_count_ = 1 ) 
		:
			thread_count(std::max(static_cast<Int>(1),thread_count_))
		{}
	
		inline Int ThreadCount() const
		{
			return thread_count;
		}
	
		inline std::string ClassName() const
        {
            return "SimplicialMeshDetails<2,3,"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
		
	void ComputeNearFarData( 
		const Tensor2<Real,Int> & V_coords,
		const Tensor2<Int ,Int> & simplices,
			  Tensor2<Real,Int> & P_near,
			  Tensor2<Real,Int> & P_far
	) const
    {
        ptic(ClassName()+"::ComputeNearFarData");
        
        //Int size       = 3;
        //Int amb_dim    = 3;
        //Int dom_dim    = 2;

        auto job_ptr = BalanceWorkLoad<Int>( simplices.Dimension(0), ThreadCount() );
        
        #pragma omp parallel for num_threads( ThreadCount() )
        for( Int thread = 0; thread < ThreadCount(); ++thread )
        {
			const Real * restrict const V_coords__      = V_coords.data();	
			const Int  * restrict const simplices__     = simplices.data();

			Real hull    [3][3];
			Real df      [3][2];
			Real dfdagger[2][3];
			Real g       [2][2];
			Real ginv    [2][2];

			Int simplex  [3];
			
			const Int i_begin = job_ptr[thread];
			const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {
				Real * restrict const near = P_near.data(i);                    
				Real * restrict const far  = P_far.data(i);   
            
				simplex[0] = simplices__[3*i +0];
				simplex[1] = simplices__[3*i +1];
				simplex[2] = simplices__[3*i +2];

				near[1] = hull[0][0] = V_coords__[3*simplex[0]+0];
				near[2] = hull[0][1] = V_coords__[3*simplex[0]+1];
				near[3] = hull[0][2] = V_coords__[3*simplex[0]+2];
				near[4] = hull[1][0] = V_coords__[3*simplex[1]+0];
				near[5] = hull[1][1] = V_coords__[3*simplex[1]+1];
				near[6] = hull[1][2] = V_coords__[3*simplex[1]+2];
				near[7] = hull[2][0] = V_coords__[3*simplex[2]+0];
				near[8] = hull[2][1] = V_coords__[3*simplex[2]+1];
				near[9] = hull[2][2] = V_coords__[3*simplex[2]+2];

				far[1] = static_cast<Real>(0.3333333333333333) * ( hull[0][0] + hull[1][0] + hull[2][0] );
				far[2] = static_cast<Real>(0.3333333333333333) * ( hull[0][1] + hull[1][1] + hull[2][1] );
				far[3] = static_cast<Real>(0.3333333333333333) * ( hull[0][2] + hull[1][2] + hull[2][2] );

				df[0][0] = hull[1][0] - hull[0][0];
				df[0][1] = hull[2][0] - hull[0][0];
				df[1][0] = hull[1][1] - hull[0][1];
				df[1][1] = hull[2][1] - hull[0][1];
				df[2][0] = hull[1][2] - hull[0][2];
				df[2][1] = hull[2][2] - hull[0][2];

				g[0][0] = df[0][0] * df[0][0] + df[1][0] * df[1][0] + df[2][0] * df[2][0];
				g[0][1] = df[0][0] * df[0][1] + df[1][0] * df[1][1] + df[2][0] * df[2][1];
				g[1][0] = df[0][1] * df[0][0] + df[1][1] * df[1][0] + df[2][1] * df[2][0];
				g[1][1] = df[0][1] * df[0][1] + df[1][1] * df[1][1] + df[2][1] * df[2][1];

                Real det = g[0][0] * g[1][1] - g[0][1] * g[0][1];

                near[0] = far[0] = sqrt( fabs(det) ) * static_cast<Real>(0.5);

                Real invdet = static_cast<Real>(1)/det;
                ginv[0][0] =  g[1][1] * invdet;
                ginv[0][1] = -g[0][1] * invdet;
                ginv[1][1] =  g[0][0] * invdet;
                
                //  dfdagger = g^{-1} * df^T (2 x 3 matrix)
				dfdagger[0][0] = ginv[0][0] * df[0][0] + ginv[0][1] * df[0][1];
				dfdagger[0][1] = ginv[0][0] * df[1][0] + ginv[0][1] * df[1][1];
				dfdagger[0][2] = ginv[0][0] * df[2][0] + ginv[0][1] * df[2][1];
				dfdagger[1][0] = ginv[0][1] * df[0][0] + ginv[1][1] * df[0][1];
				dfdagger[1][1] = ginv[0][1] * df[1][0] + ginv[1][1] * df[1][1];
				dfdagger[1][2] = ginv[0][1] * df[2][0] + ginv[1][1] * df[2][1];
            
				near[10] = far[ 4]  = static_cast<Real>(1) - df[0][0] * dfdagger[0][0] - df[0][1] * dfdagger[1][0];
				near[11] = far[ 5]  =    - df[0][0] * dfdagger[0][1] - df[0][1] * dfdagger[1][1];
				near[12] = far[ 6]  =    - df[0][0] * dfdagger[0][2] - df[0][1] * dfdagger[1][2];
				near[13] = far[ 7]  = static_cast<Real>(1) - df[1][0] * dfdagger[0][1] - df[1][1] * dfdagger[1][1];
				near[14] = far[ 8]  =    - df[1][0] * dfdagger[0][2] - df[1][1] * dfdagger[1][2];
				near[15] = far[ 9]  = static_cast<Real>(1) - df[2][0] * dfdagger[0][2] - df[2][1] * dfdagger[1][2];

            } // for( Int i = i_begin; i < i_end; ++i )

        } // #pragma omp parallel for num_threads( ThreadCount() )

        ptoc(ClassName()+"::ComputeNearFarData");
    }

	void ComputeNearFarDataOps(
		const Tensor2<Real,Int> & V_coords,
        const Tensor2<Int ,Int> & simplices,
		      Tensor2<Real,Int> & P_coords,
		      Tensor3<Real,Int> & P_hull_coords,
		      Tensor2<Real,Int> & P_near,
		      Tensor2<Real,Int> & P_far,
		SparseMatrixCSR<Real,Int> & DiffOp,
		SparseMatrixCSR<Real,Int> & AvOp 
	) const
    {
        ptic(ClassName()+"::ComputeNearFarDataOps");

        //Int size       = 3;
        //Int amb_dim    = 3;
        //Int dom_dim    = 2;

        auto job_ptr = BalanceWorkLoad<Int>( simplices.Dimension(0), ThreadCount() );

        #pragma omp parallel for num_threads( ThreadCount() )
        for( Int thread = 0; thread < ThreadCount(); ++thread )
        {
			Int  * restrict const AvOp_outer = AvOp.Outer().data();
			Int  * restrict const AvOp_inner = AvOp.Inner().data();
			Real * restrict const AvOp_value = AvOp.Values().data();

			Int  * restrict const DiffOp_outer = DiffOp.Outer().data();
			Int  * restrict const DiffOp_inner = DiffOp.Inner().data();
			Real * restrict const DiffOp_value = DiffOp.Values().data();

			const Real * restrict const V_coords__      = V_coords.data();
			
			const Int  * restrict const simplices__     = simplices.data();
				  Real * restrict const P_hull_coords__ = P_hull_coords.data();
				  Real * restrict const P_coords__      = P_coords.data();

			Real df       [3][2];
			Real dfdagger [2][3];
			Real g        [2][2];
			Real ginv     [2][2];

			Int simplex        [3];
            Int sorted_simplex [3];

			const Int i_begin = job_ptr[thread];
			const Int i_end   = job_ptr[thread+1];

            for( Int i = i_begin; i < i_end; ++i )
            {

				Real * restrict const near = P_near.data(i);                    
				Real * restrict const far  = P_far.data(i);

				simplex[0] = sorted_simplex[0] = simplices__[3*i +0];
				simplex[1] = sorted_simplex[1] = simplices__[3*i +1];
				simplex[2] = sorted_simplex[2] = simplices__[3*i +2];
                  
                // sorting simplex so that we do not have to sort the sparse arrays to achieve CSR format later
                std::sort( sorted_simplex, sorted_simplex + 3 );

				AvOp_outer[i+1] = (i+1) * 3;                      
				AvOp_inner[3*i+0] = sorted_simplex[0];
				AvOp_inner[3*i+1] = sorted_simplex[1];
				AvOp_inner[3*i+2] = sorted_simplex[2];

				AvOp_value[3*i+0] = 0.3333333333333333;
				AvOp_value[3*i+1] = 0.3333333333333333;
				AvOp_value[3*i+2] = 0.3333333333333333;

				DiffOp_outer[3*i+0] = (3 * i + 0) * 3;
				DiffOp_outer[3*i+1] = (3 * i + 1) * 3;
				DiffOp_outer[3*i+2] = (3 * i + 2) * 3;

				DiffOp_inner[(i * 3 + 0) * 3 + 0 ] = sorted_simplex[0];
				DiffOp_inner[(i * 3 + 0) * 3 + 1 ] = sorted_simplex[1];
				DiffOp_inner[(i * 3 + 0) * 3 + 2 ] = sorted_simplex[2];
				DiffOp_inner[(i * 3 + 1) * 3 + 0 ] = sorted_simplex[0];
				DiffOp_inner[(i * 3 + 1) * 3 + 1 ] = sorted_simplex[1];
				DiffOp_inner[(i * 3 + 1) * 3 + 2 ] = sorted_simplex[2];
				DiffOp_inner[(i * 3 + 2) * 3 + 0 ] = sorted_simplex[0];
				DiffOp_inner[(i * 3 + 2) * 3 + 1 ] = sorted_simplex[1];
				DiffOp_inner[(i * 3 + 2) * 3 + 2 ] = sorted_simplex[2];

				near[1] = P_hull_coords__[9*i+0] = V_coords__[3*simplex[0]+0];
				near[2] = P_hull_coords__[9*i+1] = V_coords__[3*simplex[0]+1];
				near[3] = P_hull_coords__[9*i+2] = V_coords__[3*simplex[0]+2];
				near[4] = P_hull_coords__[9*i+3] = V_coords__[3*simplex[1]+0];
				near[5] = P_hull_coords__[9*i+4] = V_coords__[3*simplex[1]+1];
				near[6] = P_hull_coords__[9*i+5] = V_coords__[3*simplex[1]+2];
				near[7] = P_hull_coords__[9*i+6] = V_coords__[3*simplex[2]+0];
				near[8] = P_hull_coords__[9*i+7] = V_coords__[3*simplex[2]+1];
				near[9] = P_hull_coords__[9*i+8] = V_coords__[3*simplex[2]+2];

				far[1] = P_coords__[3*i+0] = 0.3333333333333333 * ( P_hull_coords__[9*i+0] + P_hull_coords__[9*i+3] + P_hull_coords__[9*i+6] );
				far[2] = P_coords__[3*i+1] = 0.3333333333333333 * ( P_hull_coords__[9*i+1] + P_hull_coords__[9*i+4] + P_hull_coords__[9*i+7] );
				far[3] = P_coords__[3*i+2] = 0.3333333333333333 * ( P_hull_coords__[9*i+2] + P_hull_coords__[9*i+5] + P_hull_coords__[9*i+8] );

				df[0][0] = V_coords__[3*sorted_simplex[1]+0] - V_coords__[3*sorted_simplex[0]+0];
				df[0][1] = V_coords__[3*sorted_simplex[2]+0] - V_coords__[3*sorted_simplex[0]+0];
				df[1][0] = V_coords__[3*sorted_simplex[1]+1] - V_coords__[3*sorted_simplex[0]+1];
				df[1][1] = V_coords__[3*sorted_simplex[2]+1] - V_coords__[3*sorted_simplex[0]+1];
				df[2][0] = V_coords__[3*sorted_simplex[1]+2] - V_coords__[3*sorted_simplex[0]+2];
				df[2][1] = V_coords__[3*sorted_simplex[2]+2] - V_coords__[3*sorted_simplex[0]+2];

				g[0][0] = df[0][0] * df[0][0] + df[1][0] * df[1][0] + df[2][0] * df[2][0];
				g[0][1] = df[0][0] * df[0][1] + df[1][0] * df[1][1] + df[2][0] * df[2][1];
				g[1][0] = df[0][1] * df[0][0] + df[1][1] * df[1][0] + df[2][1] * df[2][0];
				g[1][1] = df[0][1] * df[0][1] + df[1][1] * df[1][1] + df[2][1] * df[2][1];

                Real det = g[0][0] * g[1][1] - g[0][1] * g[0][1];

                near[0] = far[0] = sqrt( fabs(det) ) * static_cast<Real>(0.5);

                Real invdet = static_cast<Real>(1.0)/det;
                ginv[0][0] =  g[1][1] * invdet;
                ginv[0][1] = -g[0][1] * invdet;
                ginv[1][1] =  g[0][0] * invdet;
                
                //  dfdagger = g^{-1} * df^T (2 x 3 matrix)
				dfdagger[0][0] = ginv[0][0] * df[0][0] + ginv[0][1] * df[0][1];
				dfdagger[0][1] = ginv[0][0] * df[1][0] + ginv[0][1] * df[1][1];
				dfdagger[0][2] = ginv[0][0] * df[2][0] + ginv[0][1] * df[2][1];
				dfdagger[1][0] = ginv[0][1] * df[0][0] + ginv[1][1] * df[0][1];
				dfdagger[1][1] = ginv[0][1] * df[1][0] + ginv[1][1] * df[1][1];
				dfdagger[1][2] = ginv[0][1] * df[2][0] + ginv[1][1] * df[2][1];
            
				near[10] = far[ 4]  = static_cast<Real>(1.0) - df[0][0] * dfdagger[0][0] - df[0][1] * dfdagger[1][0];
				near[11] = far[ 5]  =    - df[0][0] * dfdagger[0][1] - df[0][1] * dfdagger[1][1];
				near[12] = far[ 6]  =    - df[0][0] * dfdagger[0][2] - df[0][1] * dfdagger[1][2];
				near[13] = far[ 7]  = static_cast<Real>(1.0) - df[1][0] * dfdagger[0][1] - df[1][1] * dfdagger[1][1];
				near[14] = far[ 8]  =    - df[1][0] * dfdagger[0][2] - df[1][1] * dfdagger[1][2];
				near[15] = far[ 9]  = static_cast<Real>(1.0) - df[2][0] * dfdagger[0][2] - df[2][1] * dfdagger[1][2];

                // derivative operator  (3 x 3 matrix)

                Real * restrict const Df = &DiffOp_value[ 9 * i ];

				Df[ 0] = - dfdagger[0][0] - dfdagger[1][0];
				Df[ 1] =   dfdagger[0][0];
				Df[ 2] =   dfdagger[1][0];
				Df[ 3] = - dfdagger[0][1] - dfdagger[1][1];
				Df[ 4] =   dfdagger[0][1];
				Df[ 5] =   dfdagger[1][1];
				Df[ 6] = - dfdagger[0][2] - dfdagger[1][2];
				Df[ 7] =   dfdagger[0][2];
				Df[ 8] =   dfdagger[1][2];

            }
        }
        ptoc(ClassName()+"::ComputeNearFarDataOps");
    }

    void DNearToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_near, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DNearToHulls");

        if( P_D_near.Dimension(1) != 16 )
        {
            eprint("in DNearToHulls: P_D_near.Dimension(1) != 16. Aborting");
        }
        
		const Real * restrict const V_coords__  = V_coords.data();
        const Int  * restrict const simplices__ = simplices.data();
		const Real * restrict const P_D_near__  = P_D_near.data();
              Real * restrict const buffer__    = buffer.data();

		if( addTo )
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[3*i+0]+2];
				const Real s1 = V_coords__[3*simplices__[3*i+1]+1];
				const Real s2 = -(s0*s1);
				const Real s3 = V_coords__[3*simplices__[3*i+0]+1];
				const Real s4 = V_coords__[3*simplices__[3*i+1]+2];
				const Real s5 = s3*s4;
				const Real s6 = V_coords__[3*simplices__[3*i+2]+1];
				const Real s7 = s0*s6;
				const Real s8 = -(s4*s6);
				const Real s9 = V_coords__[3*simplices__[3*i+2]+2];
				const Real s10 = -(s3*s9);
				const Real s11 = s1*s9;
				const Real s12 = s10 + s11 + s2 + s5 + s7 + s8;
				const Real s13 = s12*s12;
				const Real s14 = V_coords__[3*simplices__[3*i+2]+0];
				const Real s15 = V_coords__[3*simplices__[3*i+0]+0];
				const Real s16 = V_coords__[3*simplices__[3*i+1]+0];
				const Real s17 = -(s16*s3);
				const Real s18 = s1*s15;
				const Real s19 = s14*s3;
				const Real s20 = -(s1*s14);
				const Real s21 = -(s15*s6);
				const Real s22 = s16*s6;
				const Real s23 = s17 + s18 + s19 + s20 + s21 + s22;
				const Real s24 = s23*s23;
				const Real s25 = s0*s16;
				const Real s26 = -(s15*s4);
				const Real s27 = -(s0*s14);
				const Real s28 = s14*s4;
				const Real s29 = s15*s9;
				const Real s30 = -(s16*s9);
				const Real s31 = s25 + s26 + s27 + s28 + s29 + s30;
				const Real s32 = s31*s31;
				const Real s33 = s13 + s24 + s32;
				const Real s34 = sqrt(s33);
				const Real s35 = s33*s34;
				const Real s36 = 1/s35;
				const Real s37 = -s6;
				const Real s38 = s1 + s37;
				const Real s39 = 2*s23*s38;
				const Real s40 = -s4;
				const Real s41 = s40 + s9;
				const Real s42 = 2*s31*s41;
				const Real s43 = s39 + s42;
				const Real s44 = 1/s34;
				const Real s45 = P_D_near__[16*i+13];
				const Real s46 = P_D_near__[16*i+0];
				const Real s47 = -s16;
				const Real s48 = s14 + s47;
				const Real s49 = 2*s23*s48;
				const Real s50 = -s9;
				const Real s51 = s4 + s50;
				const Real s52 = 2*s12*s51;
				const Real s53 = s49 + s52;
				const Real s54 = P_D_near__[16*i+1];
				const Real s55 = P_D_near__[16*i+3];
				const Real s56 = P_D_near__[16*i+4];
				const Real s57 = P_D_near__[16*i+5];
				const Real s58 = P_D_near__[16*i+6];
				const Real s59 = P_D_near__[16*i+7];
				const Real s60 = P_D_near__[16*i+8];
				const Real s61 = P_D_near__[16*i+9];
				const Real s62 = P_D_near__[16*i+15];
				const Real s63 = P_D_near__[16*i+14];
				const Real s64 = P_D_near__[16*i+11];
				const Real s65 = P_D_near__[16*i+12];
				const Real s66 = P_D_near__[16*i+10];
				const Real s67 = P_D_near__[16*i+2];
				const Real s68 = s34/2.;
				const Real s69 = -s14;
				const Real s70 = s16 + s69;
				const Real s71 = 2*s31*s70;
				const Real s72 = -s1;
				const Real s73 = s6 + s72;
				const Real s74 = 2*s12*s73;
				const Real s75 = s71 + s74;
				const Real s76 = -s3;
				const Real s77 = s6 + s76;
				const Real s78 = 2*s23*s77;
				const Real s79 = s0 + s50;
				const Real s80 = 2*s31*s79;
				const Real s81 = s78 + s80;
				const Real s82 = s15 + s69;
				const Real s83 = 2*s23*s82;
				const Real s84 = -s0;
				const Real s85 = s84 + s9;
				const Real s86 = 2*s12*s85;
				const Real s87 = s83 + s86;
				const Real s88 = -s15;
				const Real s89 = s14 + s88;
				const Real s90 = 2*s31*s89;
				const Real s91 = s3 + s37;
				const Real s92 = 2*s12*s91;
				const Real s93 = s90 + s92;
				const Real s94 = s3 + s72;
				const Real s95 = 2*s23*s94;
				const Real s96 = s4 + s84;
				const Real s97 = 2*s31*s96;
				const Real s98 = s95 + s97;
				const Real s99 = s16 + s88;
				const Real s100 = 2*s23*s99;
				const Real s101 = s0 + s40;
				const Real s102 = 2*s101*s12;
				const Real s103 = s100 + s102;
				const Real s104 = s15 + s47;
				const Real s105 = 2*s104*s31;
				const Real s106 = s1 + s76;
				const Real s107 = 2*s106*s12;
				const Real s108 = s105 + s107;
				buffer__[9*i+0] += (-0.25*(s32*s36*s43) + s31*s41*s44)*s45 + (s43*s44*s46)/4. + (s0*s43*s44*s55)/4. + (s16*s43*s44*s56)/4. + (s1*s43*s44*s57)/4. + (s4*s43*s44*s58)/4. + (s14*s43*s44*s59)/4. + (s43*s44*s6*s60)/4. + (-0.25*(s24*s36*s43) + s23*s38*s44)*s62 + (-0.25*(s23*s31*s36*s43) + (s31*s38*s44)/2. + (s23*s41*s44)/2.)*s63 + (-0.25*(s12*s31*s36*s43) + (s12*s41*s44)/2.)*s64 + (-0.25*(s12*s23*s36*s43) + (s12*s38*s44)/2.)*s65 - (s13*s36*s43*s66)/4. + (s3*s43*s44*s67)/4. + s54*((s15*s43*s44)/4. + s68) + (s43*s44*s61*s9)/4.;
				buffer__[9*i+1] += -0.25*(s32*s36*s45*s53) + (s44*s46*s53)/4. + (s15*s44*s53*s54)/4. + (s0*s44*s53*s55)/4. + (s16*s44*s53*s56)/4. + (s1*s44*s53*s57)/4. + (s4*s44*s53*s58)/4. + (s14*s44*s53*s59)/4. + (s44*s53*s6*s60)/4. + (s23*s44*s48 - (s24*s36*s53)/4.)*s62 + ((s31*s44*s48)/2. - (s23*s31*s36*s53)/4.)*s63 + ((s31*s44*s51)/2. - (s12*s31*s36*s53)/4.)*s64 + ((s12*s44*s48)/2. + (s23*s44*s51)/2. - (s12*s23*s36*s53)/4.)*s65 + (s12*s44*s51 - (s13*s36*s53)/4.)*s66 + s67*((s3*s44*s53)/4. + s68) + (s44*s53*s61*s9)/4.;
				buffer__[9*i+2] += (s44*s46*s75)/4. + (s15*s44*s54*s75)/4. + (s16*s44*s56*s75)/4. + (s1*s44*s57*s75)/4. + (s4*s44*s58*s75)/4. + (s14*s44*s59*s75)/4. + (s44*s6*s60*s75)/4. - (s24*s36*s62*s75)/4. + (s3*s44*s67*s75)/4. + s66*(s12*s44*s73 - (s13*s36*s75)/4.) + s65*((s23*s44*s73)/2. - (s12*s23*s36*s75)/4.) + s64*((s12*s44*s70)/2. + (s31*s44*s73)/2. - (s12*s31*s36*s75)/4.) + s63*((s23*s44*s70)/2. - (s23*s31*s36*s75)/4.) + s45*(s31*s44*s70 - (s32*s36*s75)/4.) + s55*(s68 + (s0*s44*s75)/4.) + (s44*s61*s75*s9)/4.;
				buffer__[9*i+3] += (s44*s46*s81)/4. + (s15*s44*s54*s81)/4. + (s0*s44*s55*s81)/4. + (s1*s44*s57*s81)/4. + (s4*s44*s58*s81)/4. + (s14*s44*s59*s81)/4. + (s44*s6*s60*s81)/4. - (s13*s36*s66*s81)/4. + (s3*s44*s67*s81)/4. + s65*((s12*s44*s77)/2. - (s12*s23*s36*s81)/4.) + s62*(s23*s44*s77 - (s24*s36*s81)/4.) + s64*((s12*s44*s79)/2. - (s12*s31*s36*s81)/4.) + s63*((s31*s44*s77)/2. + (s23*s44*s79)/2. - (s23*s31*s36*s81)/4.) + s45*(s31*s44*s79 - (s32*s36*s81)/4.) + s56*(s68 + (s16*s44*s81)/4.) + (s44*s61*s81*s9)/4.;
				buffer__[9*i+4] += -0.25*(s32*s36*s45*s87) + (s44*s46*s87)/4. + (s15*s44*s54*s87)/4. + (s0*s44*s55*s87)/4. + (s16*s44*s56*s87)/4. + (s4*s44*s58*s87)/4. + (s14*s44*s59*s87)/4. + (s44*s6*s60*s87)/4. + (s3*s44*s67*s87)/4. + s66*(s12*s44*s85 - (s13*s36*s87)/4.) + s65*((s12*s44*s82)/2. + (s23*s44*s85)/2. - (s12*s23*s36*s87)/4.) + s62*(s23*s44*s82 - (s24*s36*s87)/4.) + s64*((s31*s44*s85)/2. - (s12*s31*s36*s87)/4.) + s63*((s31*s44*s82)/2. - (s23*s31*s36*s87)/4.) + s57*(s68 + (s1*s44*s87)/4.) + (s44*s61*s87*s9)/4.;
				buffer__[9*i+5] += (s44*s46*s93)/4. + (s15*s44*s54*s93)/4. + (s0*s44*s55*s93)/4. + (s16*s44*s56*s93)/4. + (s1*s44*s57*s93)/4. + (s14*s44*s59*s93)/4. + (s44*s6*s60*s93)/4. - (s24*s36*s62*s93)/4. + (s3*s44*s67*s93)/4. + (s44*s61*s9*s93)/4. + s66*(s12*s44*s91 - (s13*s36*s93)/4.) + s65*((s23*s44*s91)/2. - (s12*s23*s36*s93)/4.) + s64*((s12*s44*s89)/2. + (s31*s44*s91)/2. - (s12*s31*s36*s93)/4.) + s63*((s23*s44*s89)/2. - (s23*s31*s36*s93)/4.) + s45*(s31*s44*s89 - (s32*s36*s93)/4.) + s58*(s68 + (s4*s44*s93)/4.);
				buffer__[9*i+6] += (s44*s46*s98)/4. + (s15*s44*s54*s98)/4. + (s0*s44*s55*s98)/4. + (s16*s44*s56*s98)/4. + (s1*s44*s57*s98)/4. + (s4*s44*s58*s98)/4. + (s44*s6*s60*s98)/4. - (s13*s36*s66*s98)/4. + (s3*s44*s67*s98)/4. + (s44*s61*s9*s98)/4. + s65*((s12*s44*s94)/2. - (s12*s23*s36*s98)/4.) + s62*(s23*s44*s94 - (s24*s36*s98)/4.) + s64*((s12*s44*s96)/2. - (s12*s31*s36*s98)/4.) + s63*((s31*s44*s94)/2. + (s23*s44*s96)/2. - (s23*s31*s36*s98)/4.) + s45*(s31*s44*s96 - (s32*s36*s98)/4.) + s59*(s68 + (s14*s44*s98)/4.);
				buffer__[9*i+7] += -0.25*(s103*s32*s36*s45) + (s103*s44*s46)/4. + (s103*s15*s44*s54)/4. + (s0*s103*s44*s55)/4. + (s103*s16*s44*s56)/4. + (s1*s103*s44*s57)/4. + (s103*s4*s44*s58)/4. + (s103*s14*s44*s59)/4. + (-0.25*(s103*s12*s31*s36) + (s101*s31*s44)/2.)*s64 + (-0.25*(s103*s13*s36) + s101*s12*s44)*s66 + (s103*s3*s44*s67)/4. + s60*((s103*s44*s6)/4. + s68) + (s103*s44*s61*s9)/4. + s65*(-0.25*(s103*s12*s23*s36) + (s101*s23*s44)/2. + (s12*s44*s99)/2.) + s62*(-0.25*(s103*s24*s36) + s23*s44*s99) + s63*(-0.25*(s103*s23*s31*s36) + (s31*s44*s99)/2.);
				buffer__[9*i+8] += (-0.25*(s108*s32*s36) + s104*s31*s44)*s45 + (s108*s44*s46)/4. + (s108*s15*s44*s54)/4. + (s0*s108*s44*s55)/4. + (s108*s16*s44*s56)/4. + (s1*s108*s44*s57)/4. + (s108*s4*s44*s58)/4. + (s108*s14*s44*s59)/4. + (s108*s44*s6*s60)/4. - (s108*s24*s36*s62)/4. + (-0.25*(s108*s23*s31*s36) + (s104*s23*s44)/2.)*s63 + (-0.25*(s108*s12*s31*s36) + (s104*s12*s44)/2. + (s106*s31*s44)/2.)*s64 + (-0.25*(s108*s12*s23*s36) + (s106*s23*s44)/2.)*s65 + (-0.25*(s108*s13*s36) + s106*s12*s44)*s66 + (s108*s3*s44*s67)/4. + s61*(s68 + (s108*s44*s9)/4.);
			}
		}
		else
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[3*i+0]+2];
				const Real s1 = V_coords__[3*simplices__[3*i+1]+1];
				const Real s2 = -(s0*s1);
				const Real s3 = V_coords__[3*simplices__[3*i+0]+1];
				const Real s4 = V_coords__[3*simplices__[3*i+1]+2];
				const Real s5 = s3*s4;
				const Real s6 = V_coords__[3*simplices__[3*i+2]+1];
				const Real s7 = s0*s6;
				const Real s8 = -(s4*s6);
				const Real s9 = V_coords__[3*simplices__[3*i+2]+2];
				const Real s10 = -(s3*s9);
				const Real s11 = s1*s9;
				const Real s12 = s10 + s11 + s2 + s5 + s7 + s8;
				const Real s13 = s12*s12;
				const Real s14 = V_coords__[3*simplices__[3*i+2]+0];
				const Real s15 = V_coords__[3*simplices__[3*i+0]+0];
				const Real s16 = V_coords__[3*simplices__[3*i+1]+0];
				const Real s17 = -(s16*s3);
				const Real s18 = s1*s15;
				const Real s19 = s14*s3;
				const Real s20 = -(s1*s14);
				const Real s21 = -(s15*s6);
				const Real s22 = s16*s6;
				const Real s23 = s17 + s18 + s19 + s20 + s21 + s22;
				const Real s24 = s23*s23;
				const Real s25 = s0*s16;
				const Real s26 = -(s15*s4);
				const Real s27 = -(s0*s14);
				const Real s28 = s14*s4;
				const Real s29 = s15*s9;
				const Real s30 = -(s16*s9);
				const Real s31 = s25 + s26 + s27 + s28 + s29 + s30;
				const Real s32 = s31*s31;
				const Real s33 = s13 + s24 + s32;
				const Real s34 = sqrt(s33);
				const Real s35 = s33*s34;
				const Real s36 = 1/s35;
				const Real s37 = -s6;
				const Real s38 = s1 + s37;
				const Real s39 = 2*s23*s38;
				const Real s40 = -s4;
				const Real s41 = s40 + s9;
				const Real s42 = 2*s31*s41;
				const Real s43 = s39 + s42;
				const Real s44 = 1/s34;
				const Real s45 = P_D_near__[16*i+13];
				const Real s46 = P_D_near__[16*i+0];
				const Real s47 = -s16;
				const Real s48 = s14 + s47;
				const Real s49 = 2*s23*s48;
				const Real s50 = -s9;
				const Real s51 = s4 + s50;
				const Real s52 = 2*s12*s51;
				const Real s53 = s49 + s52;
				const Real s54 = P_D_near__[16*i+1];
				const Real s55 = P_D_near__[16*i+3];
				const Real s56 = P_D_near__[16*i+4];
				const Real s57 = P_D_near__[16*i+5];
				const Real s58 = P_D_near__[16*i+6];
				const Real s59 = P_D_near__[16*i+7];
				const Real s60 = P_D_near__[16*i+8];
				const Real s61 = P_D_near__[16*i+9];
				const Real s62 = P_D_near__[16*i+15];
				const Real s63 = P_D_near__[16*i+14];
				const Real s64 = P_D_near__[16*i+11];
				const Real s65 = P_D_near__[16*i+12];
				const Real s66 = P_D_near__[16*i+10];
				const Real s67 = P_D_near__[16*i+2];
				const Real s68 = s34/2.;
				const Real s69 = -s14;
				const Real s70 = s16 + s69;
				const Real s71 = 2*s31*s70;
				const Real s72 = -s1;
				const Real s73 = s6 + s72;
				const Real s74 = 2*s12*s73;
				const Real s75 = s71 + s74;
				const Real s76 = -s3;
				const Real s77 = s6 + s76;
				const Real s78 = 2*s23*s77;
				const Real s79 = s0 + s50;
				const Real s80 = 2*s31*s79;
				const Real s81 = s78 + s80;
				const Real s82 = s15 + s69;
				const Real s83 = 2*s23*s82;
				const Real s84 = -s0;
				const Real s85 = s84 + s9;
				const Real s86 = 2*s12*s85;
				const Real s87 = s83 + s86;
				const Real s88 = -s15;
				const Real s89 = s14 + s88;
				const Real s90 = 2*s31*s89;
				const Real s91 = s3 + s37;
				const Real s92 = 2*s12*s91;
				const Real s93 = s90 + s92;
				const Real s94 = s3 + s72;
				const Real s95 = 2*s23*s94;
				const Real s96 = s4 + s84;
				const Real s97 = 2*s31*s96;
				const Real s98 = s95 + s97;
				const Real s99 = s16 + s88;
				const Real s100 = 2*s23*s99;
				const Real s101 = s0 + s40;
				const Real s102 = 2*s101*s12;
				const Real s103 = s100 + s102;
				const Real s104 = s15 + s47;
				const Real s105 = 2*s104*s31;
				const Real s106 = s1 + s76;
				const Real s107 = 2*s106*s12;
				const Real s108 = s105 + s107;
				buffer__[9*i+0] = (-0.25*(s32*s36*s43) + s31*s41*s44)*s45 + (s43*s44*s46)/4. + (s0*s43*s44*s55)/4. + (s16*s43*s44*s56)/4. + (s1*s43*s44*s57)/4. + (s4*s43*s44*s58)/4. + (s14*s43*s44*s59)/4. + (s43*s44*s6*s60)/4. + (-0.25*(s24*s36*s43) + s23*s38*s44)*s62 + (-0.25*(s23*s31*s36*s43) + (s31*s38*s44)/2. + (s23*s41*s44)/2.)*s63 + (-0.25*(s12*s31*s36*s43) + (s12*s41*s44)/2.)*s64 + (-0.25*(s12*s23*s36*s43) + (s12*s38*s44)/2.)*s65 - (s13*s36*s43*s66)/4. + (s3*s43*s44*s67)/4. + s54*((s15*s43*s44)/4. + s68) + (s43*s44*s61*s9)/4.;
				buffer__[9*i+1] = -0.25*(s32*s36*s45*s53) + (s44*s46*s53)/4. + (s15*s44*s53*s54)/4. + (s0*s44*s53*s55)/4. + (s16*s44*s53*s56)/4. + (s1*s44*s53*s57)/4. + (s4*s44*s53*s58)/4. + (s14*s44*s53*s59)/4. + (s44*s53*s6*s60)/4. + (s23*s44*s48 - (s24*s36*s53)/4.)*s62 + ((s31*s44*s48)/2. - (s23*s31*s36*s53)/4.)*s63 + ((s31*s44*s51)/2. - (s12*s31*s36*s53)/4.)*s64 + ((s12*s44*s48)/2. + (s23*s44*s51)/2. - (s12*s23*s36*s53)/4.)*s65 + (s12*s44*s51 - (s13*s36*s53)/4.)*s66 + s67*((s3*s44*s53)/4. + s68) + (s44*s53*s61*s9)/4.;
				buffer__[9*i+2] = (s44*s46*s75)/4. + (s15*s44*s54*s75)/4. + (s16*s44*s56*s75)/4. + (s1*s44*s57*s75)/4. + (s4*s44*s58*s75)/4. + (s14*s44*s59*s75)/4. + (s44*s6*s60*s75)/4. - (s24*s36*s62*s75)/4. + (s3*s44*s67*s75)/4. + s66*(s12*s44*s73 - (s13*s36*s75)/4.) + s65*((s23*s44*s73)/2. - (s12*s23*s36*s75)/4.) + s64*((s12*s44*s70)/2. + (s31*s44*s73)/2. - (s12*s31*s36*s75)/4.) + s63*((s23*s44*s70)/2. - (s23*s31*s36*s75)/4.) + s45*(s31*s44*s70 - (s32*s36*s75)/4.) + s55*(s68 + (s0*s44*s75)/4.) + (s44*s61*s75*s9)/4.;
				buffer__[9*i+3] = (s44*s46*s81)/4. + (s15*s44*s54*s81)/4. + (s0*s44*s55*s81)/4. + (s1*s44*s57*s81)/4. + (s4*s44*s58*s81)/4. + (s14*s44*s59*s81)/4. + (s44*s6*s60*s81)/4. - (s13*s36*s66*s81)/4. + (s3*s44*s67*s81)/4. + s65*((s12*s44*s77)/2. - (s12*s23*s36*s81)/4.) + s62*(s23*s44*s77 - (s24*s36*s81)/4.) + s64*((s12*s44*s79)/2. - (s12*s31*s36*s81)/4.) + s63*((s31*s44*s77)/2. + (s23*s44*s79)/2. - (s23*s31*s36*s81)/4.) + s45*(s31*s44*s79 - (s32*s36*s81)/4.) + s56*(s68 + (s16*s44*s81)/4.) + (s44*s61*s81*s9)/4.;
				buffer__[9*i+4] = -0.25*(s32*s36*s45*s87) + (s44*s46*s87)/4. + (s15*s44*s54*s87)/4. + (s0*s44*s55*s87)/4. + (s16*s44*s56*s87)/4. + (s4*s44*s58*s87)/4. + (s14*s44*s59*s87)/4. + (s44*s6*s60*s87)/4. + (s3*s44*s67*s87)/4. + s66*(s12*s44*s85 - (s13*s36*s87)/4.) + s65*((s12*s44*s82)/2. + (s23*s44*s85)/2. - (s12*s23*s36*s87)/4.) + s62*(s23*s44*s82 - (s24*s36*s87)/4.) + s64*((s31*s44*s85)/2. - (s12*s31*s36*s87)/4.) + s63*((s31*s44*s82)/2. - (s23*s31*s36*s87)/4.) + s57*(s68 + (s1*s44*s87)/4.) + (s44*s61*s87*s9)/4.;
				buffer__[9*i+5] = (s44*s46*s93)/4. + (s15*s44*s54*s93)/4. + (s0*s44*s55*s93)/4. + (s16*s44*s56*s93)/4. + (s1*s44*s57*s93)/4. + (s14*s44*s59*s93)/4. + (s44*s6*s60*s93)/4. - (s24*s36*s62*s93)/4. + (s3*s44*s67*s93)/4. + (s44*s61*s9*s93)/4. + s66*(s12*s44*s91 - (s13*s36*s93)/4.) + s65*((s23*s44*s91)/2. - (s12*s23*s36*s93)/4.) + s64*((s12*s44*s89)/2. + (s31*s44*s91)/2. - (s12*s31*s36*s93)/4.) + s63*((s23*s44*s89)/2. - (s23*s31*s36*s93)/4.) + s45*(s31*s44*s89 - (s32*s36*s93)/4.) + s58*(s68 + (s4*s44*s93)/4.);
				buffer__[9*i+6] = (s44*s46*s98)/4. + (s15*s44*s54*s98)/4. + (s0*s44*s55*s98)/4. + (s16*s44*s56*s98)/4. + (s1*s44*s57*s98)/4. + (s4*s44*s58*s98)/4. + (s44*s6*s60*s98)/4. - (s13*s36*s66*s98)/4. + (s3*s44*s67*s98)/4. + (s44*s61*s9*s98)/4. + s65*((s12*s44*s94)/2. - (s12*s23*s36*s98)/4.) + s62*(s23*s44*s94 - (s24*s36*s98)/4.) + s64*((s12*s44*s96)/2. - (s12*s31*s36*s98)/4.) + s63*((s31*s44*s94)/2. + (s23*s44*s96)/2. - (s23*s31*s36*s98)/4.) + s45*(s31*s44*s96 - (s32*s36*s98)/4.) + s59*(s68 + (s14*s44*s98)/4.);
				buffer__[9*i+7] = -0.25*(s103*s32*s36*s45) + (s103*s44*s46)/4. + (s103*s15*s44*s54)/4. + (s0*s103*s44*s55)/4. + (s103*s16*s44*s56)/4. + (s1*s103*s44*s57)/4. + (s103*s4*s44*s58)/4. + (s103*s14*s44*s59)/4. + (-0.25*(s103*s12*s31*s36) + (s101*s31*s44)/2.)*s64 + (-0.25*(s103*s13*s36) + s101*s12*s44)*s66 + (s103*s3*s44*s67)/4. + s60*((s103*s44*s6)/4. + s68) + (s103*s44*s61*s9)/4. + s65*(-0.25*(s103*s12*s23*s36) + (s101*s23*s44)/2. + (s12*s44*s99)/2.) + s62*(-0.25*(s103*s24*s36) + s23*s44*s99) + s63*(-0.25*(s103*s23*s31*s36) + (s31*s44*s99)/2.);
				buffer__[9*i+8] = (-0.25*(s108*s32*s36) + s104*s31*s44)*s45 + (s108*s44*s46)/4. + (s108*s15*s44*s54)/4. + (s0*s108*s44*s55)/4. + (s108*s16*s44*s56)/4. + (s1*s108*s44*s57)/4. + (s108*s4*s44*s58)/4. + (s108*s14*s44*s59)/4. + (s108*s44*s6*s60)/4. - (s108*s24*s36*s62)/4. + (-0.25*(s108*s23*s31*s36) + (s104*s23*s44)/2.)*s63 + (-0.25*(s108*s12*s31*s36) + (s104*s12*s44)/2. + (s106*s31*s44)/2.)*s64 + (-0.25*(s108*s12*s23*s36) + (s106*s23*s44)/2.)*s65 + (-0.25*(s108*s13*s36) + s106*s12*s44)*s66 + (s108*s3*s44*s67)/4. + s61*(s68 + (s108*s44*s9)/4.);
			}
		}

        ptoc(ClassName()+"::DNearToHulls");
        
    }

    void DFarToHulls( 
		const Tensor2<Real,Int> & V_coords, 
		const Tensor2<Int ,Int> & simplices, 
		const Tensor2<Real,Int> & P_D_far, 
        // cppcheck-suppress [constParameter]
		      Tensor3<Real,Int> & buffer, 
		bool addTo 
	) const
    {
        ptic(ClassName()+"::DFarToHulls");

        if( P_D_far.Dimension(1) != 10 )
        {
            eprint("in DFarToHulls: P_D_far.Dimension(1) != 10. Aborting");
        }
        
		const Real * restrict const V_coords__  = V_coords.data();
        const Int  * restrict const simplices__ = simplices.data();
		const Real * restrict const P_D_far__   = P_D_far.data();
              Real * restrict const buffer__    = buffer.data();

		if( addTo )
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[3*i+0]+2];
				const Real s1 = V_coords__[3*simplices__[3*i+1]+1];
				const Real s2 = -(s0*s1);
				const Real s3 = V_coords__[3*simplices__[3*i+0]+1];
				const Real s4 = V_coords__[3*simplices__[3*i+1]+2];
				const Real s5 = s3*s4;
				const Real s6 = V_coords__[3*simplices__[3*i+2]+1];
				const Real s7 = s0*s6;
				const Real s8 = -(s4*s6);
				const Real s9 = V_coords__[3*simplices__[3*i+2]+2];
				const Real s10 = -(s3*s9);
				const Real s11 = s1*s9;
				const Real s12 = s10 + s11 + s2 + s5 + s7 + s8;
				const Real s13 = s12*s12;
				const Real s14 = V_coords__[3*simplices__[3*i+2]+0];
				const Real s15 = V_coords__[3*simplices__[3*i+0]+0];
				const Real s16 = V_coords__[3*simplices__[3*i+1]+0];
				const Real s17 = -(s16*s3);
				const Real s18 = s1*s15;
				const Real s19 = s14*s3;
				const Real s20 = -(s1*s14);
				const Real s21 = -(s15*s6);
				const Real s22 = s16*s6;
				const Real s23 = s17 + s18 + s19 + s20 + s21 + s22;
				const Real s24 = s23*s23;
				const Real s25 = s0*s16;
				const Real s26 = -(s15*s4);
				const Real s27 = -(s0*s14);
				const Real s28 = s14*s4;
				const Real s29 = s15*s9;
				const Real s30 = -(s16*s9);
				const Real s31 = s25 + s26 + s27 + s28 + s29 + s30;
				const Real s32 = s31*s31;
				const Real s33 = s13 + s24 + s32;
				const Real s34 = sqrt(s33);
				const Real s35 = s33*s34;
				const Real s36 = 1/s35;
				const Real s37 = -s6;
				const Real s38 = s1 + s37;
				const Real s39 = 2*s23*s38;
				const Real s40 = -s4;
				const Real s41 = s40 + s9;
				const Real s42 = 2*s31*s41;
				const Real s43 = s39 + s42;
				const Real s44 = 1/s34;
				const Real s45 = P_D_far__[10*i+7];
				const Real s46 = P_D_far__[10*i+0];
				const Real s47 = -s16;
				const Real s48 = s14 + s47;
				const Real s49 = 2*s23*s48;
				const Real s50 = -s9;
				const Real s51 = s4 + s50;
				const Real s52 = 2*s12*s51;
				const Real s53 = s49 + s52;
				const Real s54 = P_D_far__[10*i+1];
				const Real s55 = s14 + s15 + s16;
				const Real s56 = P_D_far__[10*i+3];
				const Real s57 = s0 + s4 + s9;
				const Real s58 = P_D_far__[10*i+9];
				const Real s59 = P_D_far__[10*i+8];
				const Real s60 = P_D_far__[10*i+5];
				const Real s61 = P_D_far__[10*i+6];
				const Real s62 = P_D_far__[10*i+4];
				const Real s63 = P_D_far__[10*i+2];
				const Real s64 = s1 + s3 + s6;
				const Real s65 = s34/6.;
				const Real s66 = -s14;
				const Real s67 = s16 + s66;
				const Real s68 = 2*s31*s67;
				const Real s69 = -s1;
				const Real s70 = s6 + s69;
				const Real s71 = 2*s12*s70;
				const Real s72 = s68 + s71;
				const Real s73 = -s3;
				const Real s74 = s6 + s73;
				const Real s75 = 2*s23*s74;
				const Real s76 = s0 + s50;
				const Real s77 = 2*s31*s76;
				const Real s78 = s75 + s77;
				const Real s79 = s15 + s66;
				const Real s80 = 2*s23*s79;
				const Real s81 = -s0;
				const Real s82 = s81 + s9;
				const Real s83 = 2*s12*s82;
				const Real s84 = s80 + s83;
				const Real s85 = -s15;
				const Real s86 = s14 + s85;
				const Real s87 = 2*s31*s86;
				const Real s88 = s3 + s37;
				const Real s89 = 2*s12*s88;
				const Real s90 = s87 + s89;
				const Real s91 = s3 + s69;
				const Real s92 = 2*s23*s91;
				const Real s93 = s4 + s81;
				const Real s94 = 2*s31*s93;
				const Real s95 = s92 + s94;
				const Real s96 = s16 + s85;
				const Real s97 = 2*s23*s96;
				const Real s98 = s0 + s40;
				const Real s99 = 2*s12*s98;
				const Real s100 = s97 + s99;
				const Real s101 = s15 + s47;
				const Real s102 = 2*s101*s31;
				const Real s103 = s1 + s73;
				const Real s104 = 2*s103*s12;
				const Real s105 = s102 + s104;
				buffer__[9*i+0] += (-0.25*(s32*s36*s43) + s31*s41*s44)*s45 + (s43*s44*s46)/4. + (s43*s44*s56*s57)/12. + (-0.25*(s24*s36*s43) + s23*s38*s44)*s58 + (-0.25*(s23*s31*s36*s43) + (s31*s38*s44)/2. + (s23*s41*s44)/2.)*s59 + (-0.25*(s12*s31*s36*s43) + (s12*s41*s44)/2.)*s60 + (-0.25*(s12*s23*s36*s43) + (s12*s38*s44)/2.)*s61 - (s13*s36*s43*s62)/4. + (s43*s44*s63*s64)/12. + s54*((s43*s44*s55)/12. + s65);
				buffer__[9*i+1] += -0.25*(s32*s36*s45*s53) + (s44*s46*s53)/4. + (s44*s53*s54*s55)/12. + (s44*s53*s56*s57)/12. + (s23*s44*s48 - (s24*s36*s53)/4.)*s58 + ((s31*s44*s48)/2. - (s23*s31*s36*s53)/4.)*s59 + ((s31*s44*s51)/2. - (s12*s31*s36*s53)/4.)*s60 + ((s12*s44*s48)/2. + (s23*s44*s51)/2. - (s12*s23*s36*s53)/4.)*s61 + (s12*s44*s51 - (s13*s36*s53)/4.)*s62 + s63*((s44*s53*s64)/12. + s65);
				buffer__[9*i+2] += (s44*s46*s72)/4. + (s44*s54*s55*s72)/12. - (s24*s36*s58*s72)/4. + (s44*s63*s64*s72)/12. + s62*(s12*s44*s70 - (s13*s36*s72)/4.) + s61*((s23*s44*s70)/2. - (s12*s23*s36*s72)/4.) + s60*((s12*s44*s67)/2. + (s31*s44*s70)/2. - (s12*s31*s36*s72)/4.) + s59*((s23*s44*s67)/2. - (s23*s31*s36*s72)/4.) + s45*(s31*s44*s67 - (s32*s36*s72)/4.) + s56*(s65 + (s44*s57*s72)/12.);
				buffer__[9*i+3] += (s44*s46*s78)/4. + (s44*s56*s57*s78)/12. - (s13*s36*s62*s78)/4. + (s44*s63*s64*s78)/12. + s61*((s12*s44*s74)/2. - (s12*s23*s36*s78)/4.) + s58*(s23*s44*s74 - (s24*s36*s78)/4.) + s60*((s12*s44*s76)/2. - (s12*s31*s36*s78)/4.) + s59*((s31*s44*s74)/2. + (s23*s44*s76)/2. - (s23*s31*s36*s78)/4.) + s45*(s31*s44*s76 - (s32*s36*s78)/4.) + s54*(s65 + (s44*s55*s78)/12.);
				buffer__[9*i+4] += -0.25*(s32*s36*s45*s84) + (s44*s46*s84)/4. + (s44*s54*s55*s84)/12. + (s44*s56*s57*s84)/12. + s62*(s12*s44*s82 - (s13*s36*s84)/4.) + s61*((s12*s44*s79)/2. + (s23*s44*s82)/2. - (s12*s23*s36*s84)/4.) + s58*(s23*s44*s79 - (s24*s36*s84)/4.) + s60*((s31*s44*s82)/2. - (s12*s31*s36*s84)/4.) + s59*((s31*s44*s79)/2. - (s23*s31*s36*s84)/4.) + s63*(s65 + (s44*s64*s84)/12.);
				buffer__[9*i+5] += (s44*s46*s90)/4. + (s44*s54*s55*s90)/12. - (s24*s36*s58*s90)/4. + (s44*s63*s64*s90)/12. + s62*(s12*s44*s88 - (s13*s36*s90)/4.) + s61*((s23*s44*s88)/2. - (s12*s23*s36*s90)/4.) + s60*((s12*s44*s86)/2. + (s31*s44*s88)/2. - (s12*s31*s36*s90)/4.) + s59*((s23*s44*s86)/2. - (s23*s31*s36*s90)/4.) + s45*(s31*s44*s86 - (s32*s36*s90)/4.) + s56*(s65 + (s44*s57*s90)/12.);
				buffer__[9*i+6] += (s44*s46*s95)/4. + (s44*s56*s57*s95)/12. - (s13*s36*s62*s95)/4. + (s44*s63*s64*s95)/12. + s61*((s12*s44*s91)/2. - (s12*s23*s36*s95)/4.) + s58*(s23*s44*s91 - (s24*s36*s95)/4.) + s60*((s12*s44*s93)/2. - (s12*s31*s36*s95)/4.) + s59*((s31*s44*s91)/2. + (s23*s44*s93)/2. - (s23*s31*s36*s95)/4.) + s45*(s31*s44*s93 - (s32*s36*s95)/4.) + s54*(s65 + (s44*s55*s95)/12.);
				buffer__[9*i+7] += -0.25*(s100*s32*s36*s45) + (s100*s44*s46)/4. + (s100*s44*s54*s55)/12. + (s100*s44*s56*s57)/12. + s63*((s100*s44*s64)/12. + s65) + s58*(-0.25*(s100*s24*s36) + s23*s44*s96) + s59*(-0.25*(s100*s23*s31*s36) + (s31*s44*s96)/2.) + s62*(-0.25*(s100*s13*s36) + s12*s44*s98) + s61*(-0.25*(s100*s12*s23*s36) + (s12*s44*s96)/2. + (s23*s44*s98)/2.) + s60*(-0.25*(s100*s12*s31*s36) + (s31*s44*s98)/2.);
				buffer__[9*i+8] += (-0.25*(s105*s32*s36) + s101*s31*s44)*s45 + (s105*s44*s46)/4. + (s105*s44*s54*s55)/12. - (s105*s24*s36*s58)/4. + (-0.25*(s105*s23*s31*s36) + (s101*s23*s44)/2.)*s59 + (-0.25*(s105*s12*s31*s36) + (s101*s12*s44)/2. + (s103*s31*s44)/2.)*s60 + (-0.25*(s105*s12*s23*s36) + (s103*s23*s44)/2.)*s61 + (-0.25*(s105*s13*s36) + s103*s12*s44)*s62 + (s105*s44*s63*s64)/12. + s56*((s105*s44*s57)/12. + s65);
			}
		}
		else
		{
			#pragma omp parallel for num_threads( ThreadCount() )
			for( Int i = 0; i < simplices.Dimension(0); ++i )
			{
				const Real s0 = V_coords__[3*simplices__[3*i+0]+2];
				const Real s1 = V_coords__[3*simplices__[3*i+1]+1];
				const Real s2 = -(s0*s1);
				const Real s3 = V_coords__[3*simplices__[3*i+0]+1];
				const Real s4 = V_coords__[3*simplices__[3*i+1]+2];
				const Real s5 = s3*s4;
				const Real s6 = V_coords__[3*simplices__[3*i+2]+1];
				const Real s7 = s0*s6;
				const Real s8 = -(s4*s6);
				const Real s9 = V_coords__[3*simplices__[3*i+2]+2];
				const Real s10 = -(s3*s9);
				const Real s11 = s1*s9;
				const Real s12 = s10 + s11 + s2 + s5 + s7 + s8;
				const Real s13 = s12*s12;
				const Real s14 = V_coords__[3*simplices__[3*i+2]+0];
				const Real s15 = V_coords__[3*simplices__[3*i+0]+0];
				const Real s16 = V_coords__[3*simplices__[3*i+1]+0];
				const Real s17 = -(s16*s3);
				const Real s18 = s1*s15;
				const Real s19 = s14*s3;
				const Real s20 = -(s1*s14);
				const Real s21 = -(s15*s6);
				const Real s22 = s16*s6;
				const Real s23 = s17 + s18 + s19 + s20 + s21 + s22;
				const Real s24 = s23*s23;
				const Real s25 = s0*s16;
				const Real s26 = -(s15*s4);
				const Real s27 = -(s0*s14);
				const Real s28 = s14*s4;
				const Real s29 = s15*s9;
				const Real s30 = -(s16*s9);
				const Real s31 = s25 + s26 + s27 + s28 + s29 + s30;
				const Real s32 = s31*s31;
				const Real s33 = s13 + s24 + s32;
				const Real s34 = sqrt(s33);
				const Real s35 = s33*s34;
				const Real s36 = 1/s35;
				const Real s37 = -s6;
				const Real s38 = s1 + s37;
				const Real s39 = 2*s23*s38;
				const Real s40 = -s4;
				const Real s41 = s40 + s9;
				const Real s42 = 2*s31*s41;
				const Real s43 = s39 + s42;
				const Real s44 = 1/s34;
				const Real s45 = P_D_far__[10*i+7];
				const Real s46 = P_D_far__[10*i+0];
				const Real s47 = -s16;
				const Real s48 = s14 + s47;
				const Real s49 = 2*s23*s48;
				const Real s50 = -s9;
				const Real s51 = s4 + s50;
				const Real s52 = 2*s12*s51;
				const Real s53 = s49 + s52;
				const Real s54 = P_D_far__[10*i+1];
				const Real s55 = s14 + s15 + s16;
				const Real s56 = P_D_far__[10*i+3];
				const Real s57 = s0 + s4 + s9;
				const Real s58 = P_D_far__[10*i+9];
				const Real s59 = P_D_far__[10*i+8];
				const Real s60 = P_D_far__[10*i+5];
				const Real s61 = P_D_far__[10*i+6];
				const Real s62 = P_D_far__[10*i+4];
				const Real s63 = P_D_far__[10*i+2];
				const Real s64 = s1 + s3 + s6;
				const Real s65 = s34/6.;
				const Real s66 = -s14;
				const Real s67 = s16 + s66;
				const Real s68 = 2*s31*s67;
				const Real s69 = -s1;
				const Real s70 = s6 + s69;
				const Real s71 = 2*s12*s70;
				const Real s72 = s68 + s71;
				const Real s73 = -s3;
				const Real s74 = s6 + s73;
				const Real s75 = 2*s23*s74;
				const Real s76 = s0 + s50;
				const Real s77 = 2*s31*s76;
				const Real s78 = s75 + s77;
				const Real s79 = s15 + s66;
				const Real s80 = 2*s23*s79;
				const Real s81 = -s0;
				const Real s82 = s81 + s9;
				const Real s83 = 2*s12*s82;
				const Real s84 = s80 + s83;
				const Real s85 = -s15;
				const Real s86 = s14 + s85;
				const Real s87 = 2*s31*s86;
				const Real s88 = s3 + s37;
				const Real s89 = 2*s12*s88;
				const Real s90 = s87 + s89;
				const Real s91 = s3 + s69;
				const Real s92 = 2*s23*s91;
				const Real s93 = s4 + s81;
				const Real s94 = 2*s31*s93;
				const Real s95 = s92 + s94;
				const Real s96 = s16 + s85;
				const Real s97 = 2*s23*s96;
				const Real s98 = s0 + s40;
				const Real s99 = 2*s12*s98;
				const Real s100 = s97 + s99;
				const Real s101 = s15 + s47;
				const Real s102 = 2*s101*s31;
				const Real s103 = s1 + s73;
				const Real s104 = 2*s103*s12;
				const Real s105 = s102 + s104;
				buffer__[9*i+0] = (-0.25*(s32*s36*s43) + s31*s41*s44)*s45 + (s43*s44*s46)/4. + (s43*s44*s56*s57)/12. + (-0.25*(s24*s36*s43) + s23*s38*s44)*s58 + (-0.25*(s23*s31*s36*s43) + (s31*s38*s44)/2. + (s23*s41*s44)/2.)*s59 + (-0.25*(s12*s31*s36*s43) + (s12*s41*s44)/2.)*s60 + (-0.25*(s12*s23*s36*s43) + (s12*s38*s44)/2.)*s61 - (s13*s36*s43*s62)/4. + (s43*s44*s63*s64)/12. + s54*((s43*s44*s55)/12. + s65);
				buffer__[9*i+1] = -0.25*(s32*s36*s45*s53) + (s44*s46*s53)/4. + (s44*s53*s54*s55)/12. + (s44*s53*s56*s57)/12. + (s23*s44*s48 - (s24*s36*s53)/4.)*s58 + ((s31*s44*s48)/2. - (s23*s31*s36*s53)/4.)*s59 + ((s31*s44*s51)/2. - (s12*s31*s36*s53)/4.)*s60 + ((s12*s44*s48)/2. + (s23*s44*s51)/2. - (s12*s23*s36*s53)/4.)*s61 + (s12*s44*s51 - (s13*s36*s53)/4.)*s62 + s63*((s44*s53*s64)/12. + s65);
				buffer__[9*i+2] = (s44*s46*s72)/4. + (s44*s54*s55*s72)/12. - (s24*s36*s58*s72)/4. + (s44*s63*s64*s72)/12. + s62*(s12*s44*s70 - (s13*s36*s72)/4.) + s61*((s23*s44*s70)/2. - (s12*s23*s36*s72)/4.) + s60*((s12*s44*s67)/2. + (s31*s44*s70)/2. - (s12*s31*s36*s72)/4.) + s59*((s23*s44*s67)/2. - (s23*s31*s36*s72)/4.) + s45*(s31*s44*s67 - (s32*s36*s72)/4.) + s56*(s65 + (s44*s57*s72)/12.);
				buffer__[9*i+3] = (s44*s46*s78)/4. + (s44*s56*s57*s78)/12. - (s13*s36*s62*s78)/4. + (s44*s63*s64*s78)/12. + s61*((s12*s44*s74)/2. - (s12*s23*s36*s78)/4.) + s58*(s23*s44*s74 - (s24*s36*s78)/4.) + s60*((s12*s44*s76)/2. - (s12*s31*s36*s78)/4.) + s59*((s31*s44*s74)/2. + (s23*s44*s76)/2. - (s23*s31*s36*s78)/4.) + s45*(s31*s44*s76 - (s32*s36*s78)/4.) + s54*(s65 + (s44*s55*s78)/12.);
				buffer__[9*i+4] = -0.25*(s32*s36*s45*s84) + (s44*s46*s84)/4. + (s44*s54*s55*s84)/12. + (s44*s56*s57*s84)/12. + s62*(s12*s44*s82 - (s13*s36*s84)/4.) + s61*((s12*s44*s79)/2. + (s23*s44*s82)/2. - (s12*s23*s36*s84)/4.) + s58*(s23*s44*s79 - (s24*s36*s84)/4.) + s60*((s31*s44*s82)/2. - (s12*s31*s36*s84)/4.) + s59*((s31*s44*s79)/2. - (s23*s31*s36*s84)/4.) + s63*(s65 + (s44*s64*s84)/12.);
				buffer__[9*i+5] = (s44*s46*s90)/4. + (s44*s54*s55*s90)/12. - (s24*s36*s58*s90)/4. + (s44*s63*s64*s90)/12. + s62*(s12*s44*s88 - (s13*s36*s90)/4.) + s61*((s23*s44*s88)/2. - (s12*s23*s36*s90)/4.) + s60*((s12*s44*s86)/2. + (s31*s44*s88)/2. - (s12*s31*s36*s90)/4.) + s59*((s23*s44*s86)/2. - (s23*s31*s36*s90)/4.) + s45*(s31*s44*s86 - (s32*s36*s90)/4.) + s56*(s65 + (s44*s57*s90)/12.);
				buffer__[9*i+6] = (s44*s46*s95)/4. + (s44*s56*s57*s95)/12. - (s13*s36*s62*s95)/4. + (s44*s63*s64*s95)/12. + s61*((s12*s44*s91)/2. - (s12*s23*s36*s95)/4.) + s58*(s23*s44*s91 - (s24*s36*s95)/4.) + s60*((s12*s44*s93)/2. - (s12*s31*s36*s95)/4.) + s59*((s31*s44*s91)/2. + (s23*s44*s93)/2. - (s23*s31*s36*s95)/4.) + s45*(s31*s44*s93 - (s32*s36*s95)/4.) + s54*(s65 + (s44*s55*s95)/12.);
				buffer__[9*i+7] = -0.25*(s100*s32*s36*s45) + (s100*s44*s46)/4. + (s100*s44*s54*s55)/12. + (s100*s44*s56*s57)/12. + s63*((s100*s44*s64)/12. + s65) + s58*(-0.25*(s100*s24*s36) + s23*s44*s96) + s59*(-0.25*(s100*s23*s31*s36) + (s31*s44*s96)/2.) + s62*(-0.25*(s100*s13*s36) + s12*s44*s98) + s61*(-0.25*(s100*s12*s23*s36) + (s12*s44*s96)/2. + (s23*s44*s98)/2.) + s60*(-0.25*(s100*s12*s31*s36) + (s31*s44*s98)/2.);
				buffer__[9*i+8] = (-0.25*(s105*s32*s36) + s101*s31*s44)*s45 + (s105*s44*s46)/4. + (s105*s44*s54*s55)/12. - (s105*s24*s36*s58)/4. + (-0.25*(s105*s23*s31*s36) + (s101*s23*s44)/2.)*s59 + (-0.25*(s105*s12*s31*s36) + (s101*s12*s44)/2. + (s103*s31*s44)/2.)*s60 + (-0.25*(s105*s12*s23*s36) + (s103*s23*s44)/2.)*s61 + (-0.25*(s105*s13*s36) + s103*s12*s44)*s62 + (s105*s44*s63*s64)/12. + s56*((s105*s44*s57)/12. + s65);
			}
		}

        ptoc(ClassName()+"::DFarToHulls");
        
    }

	}; // SimplicialMeshDetails<2,3,Real,Int>

} // namespace Repulsion
