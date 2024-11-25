#pragma once

#include <goast/Optimization/optInterface.h>
#include <goast/Optimization/optParameters.h>
#include <goast/Optimization/optUtils.h>

#include <goast/Optimization/interfaces/CholmodInterface.h>

namespace NewOpt {
  template<typename ConfiguratorType=DefaultConfigurator>
  class LineSearchNewtonCG : public OptimizationBase<ConfiguratorType> {
    using RealType = typename ConfiguratorType::RealType;

    using VectorType = typename ConfiguratorType::VectorType;
    using MatrixType = typename ConfiguratorType::SparseMatrixType;
    using FullMatrixType = typename ConfiguratorType::FullMatrixType;
    using TripletType = typename ConfiguratorType::TripletType;
    using TensorType = typename ConfiguratorType::TensorType;

    // Epsilon for safe-guarding computation of reduction against numerically instabilities
    const RealType eps_red = 10 * std::numeric_limits<RealType>::epsilon();

    // Functionals
    const BaseOp<VectorType, RealType> &_F;
    const BaseOp<VectorType, VectorType> &_DF;
    const MapToLinOp<ConfiguratorType> &_D2F;
    const MapToLinOp<ConfiguratorType> *m_Pre = nullptr;


    // Linear solver
    int m_cgIterations = 1000;

    // Stopping criteria
    int _maxIterations;
    RealType _stopEpsilon;

    // Line search
    TIMESTEP_CONTROLLER _linesearchMethod = ARMIJO;
    RealType _sigma = 0.1;
    RealType _beta = 0.9;
    RealType _start_tau = 1.0;
    RealType _tau_min = 1e-12;
    RealType _tau_max = 4.;

    const std::vector<int> *_bdryMask;

    QUIET_MODE _quietMode;

  public:
    LineSearchNewtonCG( const BaseOp<VectorType, RealType> &F,
                        const BaseOp<VectorType, VectorType> &DF,
                        const MapToLinOp<ConfiguratorType> &D2F,
                        const OptimizationParameters<ConfiguratorType> &optPars )
            : _F( F ), _DF( DF ), _D2F( D2F ),
              _maxIterations( optPars.getNewtonIterations()),
              _stopEpsilon( optPars.getStopEpsilon()),
              _bdryMask( nullptr ),
              _quietMode( optPars.getQuietMode()) {}

    LineSearchNewtonCG( const BaseOp<VectorType, RealType> &F,
                        const BaseOp<VectorType, VectorType> &DF,
                        const MapToLinOp<ConfiguratorType> &D2F,
                        const RealType stopEpsilon = 1e-8,
                        const int maxIterations = 100,
                        QUIET_MODE quietMode = SUPERQUIET ) : _F( F ), _DF( DF ), _D2F( D2F ),
                                                              _maxIterations( maxIterations ),
                                                              _stopEpsilon( stopEpsilon ), _bdryMask( nullptr ),
                                                              _quietMode( quietMode ) {}
    LineSearchNewtonCG( const BaseOp<VectorType, RealType> &F,
                        const BaseOp<VectorType, VectorType> &DF,
                        const MapToLinOp<ConfiguratorType> &D2F,
                        const MapToLinOp<ConfiguratorType> &Pre,
                        const RealType stopEpsilon = 1e-8,
                        const int maxIterations = 100,
                        QUIET_MODE quietMode = SUPERQUIET ) : _F( F ), _DF( DF ), _D2F( D2F ), m_Pre(&Pre),
                                                              _maxIterations( maxIterations ),
                                                              _stopEpsilon( stopEpsilon ), _bdryMask( nullptr ),
                                                              _quietMode( quietMode ) {}

    LineSearchNewtonCG( const BaseOp<VectorType, RealType> &F,
                        const BaseOp<VectorType, VectorType> &DF,
                        const MapToLinOp<ConfiguratorType> &D2F,
                        const RealType stopEpsilon = 1e-8,
                        const int maxIterations = 100,
                        bool quiet = false ) : _F( F ), _DF( DF ), _D2F( D2F ), _maxIterations( maxIterations ),
                                               _stopEpsilon( stopEpsilon ), _bdryMask( nullptr ),
                                               _quietMode( quiet ? SUPERQUIET : SHOW_ALL ) {}

    void setBoundaryMask( const std::vector<int> &Mask ) {
      _bdryMask = &Mask;
    }


    void solve( const VectorType &x_0, VectorType &x_k ) const override {
      this->status.Iteration = 0;
      this->status.totalTime = 0.;
      this->status.additionalTimings["Evaluation"] = 0.;
      this->status.additionalTimings["Direction"] = 0.;
      this->status.additionalTimings["LineSearch"] = 0.;

      auto t_start_setup = std::chrono::high_resolution_clock::now();

      int numDofs = x_0.size();
      int numDofs_red = x_0.size();

      x_k = x_0;

      VectorType p_k( x_0.size());
      p_k.setZero();

      RealType eta_k, eps_k;
      eps_k = 1.e-1;
//      eps_k = std::numeric_limits<RealType>::infinity();

      RealType alpha_k = _start_tau;
      RealType tau_k = 0;

      VectorType tmp_x_k( x_0.size());
      RealType F_k, tmp_F_k;

      VectorType grad_F_k;
      std::unique_ptr<LinearOperator<ConfiguratorType>> Hess_F_k;
      std::unique_ptr<LinearOperator<ConfiguratorType>> Dinv = std::make_unique<IdentityOperator<ConfiguratorType>>();

      // initial evaluation of F, DF and D^2F
      auto t_start_eval = std::chrono::high_resolution_clock::now();
      _F.apply( x_k, F_k );
      // compute gradient and check norm
      _DF.apply( x_k, grad_F_k );
      if ( _bdryMask )
        applyMaskToVector<VectorType>( *_bdryMask, grad_F_k );

      RealType gradNorm = grad_F_k.norm();
      if ( gradNorm < _stopEpsilon ) {
        auto t_end_eval = std::chrono::high_resolution_clock::now();
        this->status.additionalTimings["Evaluation"] += std::chrono::duration<RealType, std::milli>(
                t_end_eval - t_start_eval ).count();

        if ( _quietMode == SHOW_ALL || _quietMode == SHOW_TERMINATION_INFO )
          std::cout << " -- LSNCG -- Initial gradient norm below epsilon." << std::endl;

        return;
      }
      // compute Hessian (and preconditioner)
      Hess_F_k = _D2F( x_k );
      if ( m_Pre )
        Dinv = (*m_Pre)( x_k );
//      _D2F.apply( x_k, Hess_F_k );
//      if ( _bdryMask )
//        applyMaskToSymmetricMatrix<MatrixType>( *_bdryMask, Hess_F_k );

      VectorType c = ( *Dinv )( grad_F_k );

      auto t_end_eval = std::chrono::high_resolution_clock::now();
      this->status.additionalTimings["Evaluation"] += std::chrono::duration<RealType, std::milli>(
              t_end_eval - t_start_eval ).count();

      StepsizeControl<ConfiguratorType> stepsizeControl( _F, _DF, _linesearchMethod, _sigma, _beta, _start_tau,
                                                         _tau_min, _tau_max );

      auto t_end_setup = std::chrono::high_resolution_clock::now();
      this->status.additionalTimings["Setup"] = std::chrono::duration<RealType, std::milli>(
              t_end_setup - t_start_setup ).count();


      // print header for console output
      if ( _quietMode == SHOW_ALL )
        printHeader();
      if ( _quietMode == SHOW_ALL )
        printConsoleOutput<std::chrono::duration<RealType, std::milli>>( 0, F_k, grad_F_k.norm(), 0, 0, 0, 0,
                                                                         t_end_setup - t_start_setup,
                                                                         std::chrono::duration<RealType, std::milli>(0),
                                                                         std::chrono::duration<RealType, std::milli>(0),
                                                                                 t_end_eval - t_start_eval );

      // start Newton iteration
      for ( int k = 1; k <= _maxIterations; k++ ) {
        this->status.Iteration = k;

        auto t_start = std::chrono::high_resolution_clock::now();

        // Step 1: Compute descent direction
        auto t_start_dir = std::chrono::high_resolution_clock::now();

        RealType cNorm = std::sqrt( grad_F_k.dot( c ));

        eta_k = std::min( 0.5, cNorm );
        eps_k = std::min(eps_k, eta_k * cNorm);
//        eps_k = eta_k * cNorm;

        // Actually compute descent direction
        auto cgStatus = solveCG( *Hess_F_k, grad_F_k, *Dinv, eps_k, p_k );


//        std::cout << "pk Norm: " << std::scientific << p_k.norm() << std::endl;
//      std::cout << "pk_red Norm: " << (bndPermutation.inverse() * p_k).segment(0, numDofs_red).norm() << std::endl;
//      std::cout << "pk_nonred Norm: " << (bndPermutation.inverse() * p_k).segment(numDofs_red, numDofs - numDofs_red).norm() << std::endl;

        auto t_end_dir = std::chrono::high_resolution_clock::now();
        this->status.additionalTimings["Direction"] += std::chrono::duration<RealType, std::milli>(
                t_end_dir - t_start_dir ).count();

        // Step 2: Line search
        auto t_start_ls = std::chrono::high_resolution_clock::now();
        alpha_k = stepsizeControl.getStepsize( x_k, grad_F_k, p_k, alpha_k, F_k );
        auto t_end_ls = std::chrono::high_resolution_clock::now();
        this->status.additionalTimings["LineSearch"] += std::chrono::duration<RealType, std::milli>(
                t_end_ls - t_start_ls ).count();

        // step size zero -> TERMINATE!
        if ( alpha_k == 0. ) {
          auto t_end = std::chrono::high_resolution_clock::now();
          this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();

          if ( _quietMode == SHOW_ALL ) {
            printConsoleOutput( k, F_k, grad_F_k.norm(), alpha_k, cgStatus.Residual, cgStatus.reasonOfTermination,
                                cgStatus.Iteration, t_end - t_start, t_end_dir - t_start_dir,
                                t_end_ls - t_start_ls, t_end_eval - t_start_eval );
            std::cout << " -- LSNCG -- Iter " << std::setw( 3 ) << k << ": " << "Step size too small." << std::endl;
          }
          break;
        }

        // actual update of position
        x_k += alpha_k * p_k;

        // start new evaluation
        t_start_eval = std::chrono::high_resolution_clock::now();
        _F.apply( x_k, F_k );
        _DF.apply( x_k, grad_F_k );
        if ( _bdryMask )
          applyMaskToVector<VectorType>( *_bdryMask, grad_F_k );

        Hess_F_k = _D2F( x_k );
        if ( m_Pre )
          Dinv = (*m_Pre)( x_k );
        c = ( *Dinv )( grad_F_k );


        gradNorm = grad_F_k.norm();
        t_end_eval = std::chrono::high_resolution_clock::now();
        this->status.additionalTimings["Evaluation"] += std::chrono::duration<RealType, std::milli>(
                t_end_eval - t_start_eval ).count();

        // stop total timing
        auto t_end = std::chrono::high_resolution_clock::now();
        this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();

        // gradient norm small enough -> TERMINATE!
        if ( gradNorm < _stopEpsilon ) {
          if ( _quietMode == SHOW_ALL ) {
            printConsoleOutput( k, F_k, gradNorm, alpha_k, cgStatus.Residual, cgStatus.reasonOfTermination,
                                cgStatus.Iteration, t_end - t_start, t_end_dir - t_start_dir,
                                t_end_ls - t_start_ls, t_end_eval - t_start_eval );
            std::cout << " -- LSNCG -- Iter " << std::setw( 3 ) << k << ": " << "Gradient norm below epsilon."
                      << std::endl;
          }
          break;
        }

        // console output for that iteration
        if ( _quietMode == SHOW_ALL )
          printConsoleOutput( k, F_k, gradNorm, alpha_k, cgStatus.Residual, cgStatus.reasonOfTermination,
                              cgStatus.Iteration, t_end - t_start, t_end_dir - t_start_dir,
                              t_end_ls - t_start_ls, t_end_eval - t_start_eval );

      } // end of Newton iteration

      // final console output
      if (( _quietMode == SHOW_ALL ) || ( _quietMode == SHOW_TERMINATION_INFO ) ||
          (( _quietMode == SHOW_ONLY_IF_FAILED ) && ( gradNorm > _stopEpsilon ))) {
        std::cout << " -- LSNCG -- Final   : " << std::scientific << std::setprecision( 6 )
                  << F_k
                  << " || " << gradNorm
                  << " ||===|| " << std::fixed << std::setprecision( 2 ) << this->status.totalTime
                  << " || " << this->status.additionalTimings["Direction"]
                  << " || " << this->status.additionalTimings["LineSearch"]
                  << " || " << this->status.additionalTimings["Evaluation"]
                  << std::endl;
      }
    }

    void setParameter( const std::string &name, std::string value ) override {
      throw std::runtime_error( "LineSearchNewton::setParameter(): Unknown string parameter '" + name + "'." );
    }

    void setParameter( const std::string &name, RealType value ) override {
      if ( name == "sigma" )
        _sigma = value;
      else if ( name == "beta" )
        _beta = value;
      else if ( name == "initial_stepsize" )
        _start_tau = value;
      else if ( name == "minimal_stepsize" )
        _tau_min = value;
      else if ( name == "maximal_stepsize" )
        _tau_max = value;
      else if ( name == "tolerance" )
        _stopEpsilon = value;
      else
        throw std::runtime_error( "LineSearchNewton::setParameter(): Unknown real parameter '" + name + "'." );
    }

    void setParameter( const std::string &name, int value ) override {
      if ( name == "maximum_iterations" )
        _maxIterations = value;
      else if ( name == "stepsize_control" )
        _linesearchMethod = static_cast<TIMESTEP_CONTROLLER> (value);
      else if ( name == "print_level" )
        _quietMode = static_cast<QUIET_MODE> (value);
      else if ( name == "cg_iterations" )
        m_cgIterations = value;
      else
        throw std::runtime_error( "LineSearchNewton::setParameter(): Unknown integer parameter '" + name + "'." );
    }

  protected:
    SolverStatus<ConfiguratorType> solveCG( const LinearOperator<ConfiguratorType> &H, const VectorType &c,
                                            const LinearOperator<ConfiguratorType> &Dinv, const RealType &eps_k,
                                            VectorType &p ) const {
      SolverStatus<ConfiguratorType> Status;
      Status.Iteration = 0;
      Status.reasonOfTermination = -1;

      p.resize( c.size());
      p.setZero();

//    VectorType r_k( _Dinv * _c );
      VectorType r_k = c;
      VectorType y_k = Dinv(r_k);
//      VectorType y_k = r_k;

      if ( _bdryMask )
        applyMaskToVector( *_bdryMask, y_k );

      VectorType d_k = -y_k;

      // Stop uf 0 is already a good enough solution
      RealType r_sqNorm = r_k.dot( y_k ); // r_k^T r_k
      Status.Residual = std::sqrt( r_sqNorm );
      if ( Status.Residual < eps_k ) {
        std::cout << "stopped with initial residual " << r_k.norm() << " vs. " << eps_k << std::endl;
        return Status;
      }

      // Intermediate and temporary quantities
      VectorType Hd; // H * d_k
      RealType alpha_k, beta_k, temp;
//      std::cerr << " .. r_sqNorm = " << r_sqNorm << std::endl;
      VectorType tmpVector;

      for ( int k = 0; k < m_cgIterations; k++ ) {
        Status.Iteration++;

        Hd = H( d_k );

        if ( _bdryMask )
          applyMaskToVector( *_bdryMask, Hd );

        // Step 1: Check if current search direction is one of nonpositive curvature
        RealType curvature = d_k.transpose() * Hd;
        if ( curvature <= 0 ) {
          if ( k == 0 || p.norm() > 1.e+8 )
            p = -c;

          std::cout << " -- LSNCG -- CG stopped in iter " << k << " due to nonpositive curvature: " << std::scientific
                    << std::setprecision( 6 ) << curvature << " - norm " << d_k.norm() << std::endl;

          Status.reasonOfTermination = 1;

          return Status;
        }

        // Step 2: Compute new iterate
        alpha_k = r_sqNorm / curvature;
        p += alpha_k * d_k;

        // Step 3: Update residual and check for convergence
        r_k += alpha_k * Hd; // r_{k+1} = r_k + alpha_k * H * d_k
        y_k = Dinv( r_k );
//        y_k = r_k;
        if ( _bdryMask )
          applyMaskToVector( *_bdryMask, y_k );

        temp = r_sqNorm;
        r_sqNorm = r_k.dot(y_k);

        Status.Residual = std::sqrt(r_sqNorm);
        if ( Status.Residual < eps_k ) {
          Status.reasonOfTermination = 0;
          return Status;
        }

        // Step 4: Compute new search direction

//        std::cerr << " .. r_sqNorm = " << r_sqNorm << std::endl;
        beta_k = r_sqNorm / temp; // beta_{k+1} = r_{k+1}^T r_{k+1} / r_k^T r_k
//        std::cerr << " .. beta_k = " << beta_k << std::endl;
        d_k *= beta_k;
        d_k -= y_k; // d_{k+1} = -r_{k+1} + beta_{k+1} * d_k

      }

      return Status;
    }

    void printHeader() const {
      std::cout << " -- LSNCG --         : " << std::scientific << std::setprecision( 6 )
                << "  F[x_k]  " << " || " << "  |DF[x_k]|  " << " ||===|| "
                << " stepsize " << " || " << "  shift  " << " || " << "num LS" << " ||===|| "
                << "total time" << " || " << "time dir" << " || " << "time ls" << " || " << "time eval"
                << std::endl;
    }

    template<typename IntType>
    void printConsoleOutput( int iteration,
                             RealType F,
                             RealType NormDF,
                             RealType stepsize,
                             RealType cgResidual,
                             int cgTermination,
                             int cgIter,
                             IntType DeltaTotalTime,
                             IntType DeltaDirTime,
                             IntType DeltaLSTime,
                             IntType DeltaEvalTime ) const {
      std::cout << " -- LSNCG -- Iter " << std::setw( 3 ) << iteration << ": " << std::scientific
                << std::setprecision( 6 )
                << F
                << " || " << NormDF
                << " ||===|| "
                << std::setw( 13 ) << stepsize
                << " ||===|| " << cgResidual
                << " || " << cgTermination
                << " || " << cgIter
                << std::setprecision( 2 ) << std::fixed
                << " ||===|| " << std::setw( 6 )
                << std::chrono::duration<RealType, std::milli>( DeltaTotalTime ).count()
                << " || " << std::setw( 6 ) << std::chrono::duration<RealType, std::milli>( DeltaDirTime ).count()
                << " || " << std::setw( 6 ) << std::chrono::duration<RealType, std::milli>( DeltaLSTime ).count()
                << " || " << std::setw( 6 ) << std::chrono::duration<RealType, std::milli>( DeltaEvalTime ).count()
                << std::endl;
    }

  };
}
