#pragma once

#include <memory>

#include <goast/Optimization/optInterface.h>

#include "SteihaugCG.h"
#include "Preconditioners.h"

namespace NewOpt {
  template<typename ConfiguratorType=DefaultConfigurator>
  class TrustRegionNewton : public OptimizationBase<ConfiguratorType> {
  public:
    using RealType = typename ConfiguratorType::RealType;
    using VectorType = typename ConfiguratorType::VectorType;
    using MatrixType = typename ConfiguratorType::SparseMatrixType;

    // Preconditioner
    enum PreconditionerType {
      PROVIDED = -1,
      NONE = 0,
      DIAGONAL = 1,
      INCOMPLETE_CHOLESKY = 2,
      CHOLESKY = 3,
    };

  protected:
    // Epsilon for safe-guarding computation of reduction against numerically instabilities
    const RealType eps_red = 100 * std::numeric_limits<RealType>::epsilon();

    // Functionals
    const BaseOp<VectorType, RealType> &m_F;
    const BaseOp<VectorType, VectorType> &m_DF;
    const MapToLinOp<ConfiguratorType> &m_D2F;
    const MapToLinOp<ConfiguratorType> *m_P = nullptr;

    // Trust region size
    RealType m_maxRadius;
    RealType m_minRadius = 1e-8;
    RealType m_initRadius;
    RealType m_eta;

    // Stopping criteria
    int m_maxIterations;
    RealType m_stopEpsilon;
    RealType m_minStepsize = 1.e-9;
    RealType m_minReduction = 1.e-14;

    // Preconditioning
    PreconditionerType m_preconditioner = DIAGONAL;

    // Output
    bool m_quiet;

    // Subproblem solver
    std::string m_subproblemSolver = "SteihaugCG";
    int m_cgIterations;

    // Fixed variables and bounds
    const std::vector<int> *m_fixedVariables;
    const VectorType *m_lowerBounds = nullptr;
    const VectorType *m_upperBounds = nullptr;

    // Trust-region subproblem solver parameters
    std::map<std::string, int> m_trsolverIntParameters;
    std::map<std::string, RealType> m_trsolverRealParameters;
    std::map<std::string, std::string> m_trsolverStringParameters;

    // Callbacks
    std::vector<std::function<void( int, const VectorType &, const RealType &, const VectorType & )>> m_callbackFcts;


  public:
    TrustRegionNewton( const BaseOp<VectorType, RealType> &F,
                       const BaseOp<VectorType, VectorType> &DF,
                       const MapToLinOp<ConfiguratorType> &D2F,
                       const RealType initRadius,
                       const RealType maxRadius,
                       const RealType stopEpsilon = 1e-8,
                       const int maxIterations = 100,
                       const int cgIterations = 100,
                       const RealType eta = 0.25,
                       bool quiet = false ) : m_F( F ), m_DF( DF ), m_D2F( D2F ), m_maxRadius( maxRadius ),
                                              m_initRadius( initRadius ), m_eta( eta ),
                                              m_maxIterations( maxIterations ), m_cgIterations( cgIterations ),
                                              m_stopEpsilon( stopEpsilon ), m_fixedVariables( nullptr ), m_quiet( quiet ) {}

    // For legacy compatability
    void setBoundaryMask( const std::vector<int> &Mask ) {
      setFixedVariables(Mask);
    }

    void addCallbackFunction (const std::function<void( int, const VectorType &, const RealType &, const VectorType & )> &F) {
      m_callbackFcts.push_back(F);
    }

    void setFixedVariables( const std::vector<int> &fixedVariables ) override {
      m_fixedVariables = &fixedVariables;
    }

    void setVariableBounds( const VectorType &lowerBounds, const VectorType &upperBounds ) override {
      m_lowerBounds = &lowerBounds;
      m_upperBounds = &upperBounds;
    }

    void setPreconditioner( const MapToLinOp<ConfiguratorType> &P ) {
      m_P = &P;
    }


    void solve( const VectorType &x_0, VectorType &x_k ) const override {
      this->status.Iteration = 0;
      this->status.totalTime = 0.;
      this->status.additionalTimings["Preconditioner"] = 0.;
      this->status.additionalTimings["Subproblem"] = 0.;
      this->status.additionalIterations["Subproblem"] = 0;
      this->status.additionalTimings["Evaluation"] = 0.;

      RealType trRadius = m_initRadius;
      RealType eta_k, eps_k, rho_k;
      eps_k = std::numeric_limits<RealType>::infinity();
      eps_k = 1.e-1;

      x_k = x_0;

      VectorType p_k( x_0.size());
      p_k.setZero();

      VectorType tmp_x_k( x_0.size());
      RealType F_k, tmp_F_k;
      RealType m_red, f_red; // predicted and actual reduction

      VectorType grad_F_k;
      std::unique_ptr<LinearOperator<ConfiguratorType>> Hess_F_k;
      m_DF.apply( x_k, grad_F_k );
      m_F.apply( x_k, F_k );
      Hess_F_k = m_D2F(x_k);
      if ( m_fixedVariables )
        applyMaskToVector<VectorType>( *m_fixedVariables, grad_F_k );

      RealType initGradNorm = grad_F_k.norm();

      // modified bounds
      VectorType lb, ub;

      auto t_start_eval = std::chrono::high_resolution_clock::now();
      auto t_end_eval = std::chrono::high_resolution_clock::now();
      auto t_start_eval_a = std::chrono::high_resolution_clock::now();
      auto t_end_eval_a = std::chrono::high_resolution_clock::now();
      auto t_start_pre = std::chrono::high_resolution_clock::now();
      auto t_end_pre = std::chrono::high_resolution_clock::now();

      std::unique_ptr<LinearOperator<ConfiguratorType>> Dinv = std::make_unique<IdentityOperator<ConfiguratorType>>();

      if ( m_preconditioner == DIAGONAL )
        Dinv = std::make_unique<DiagonalPreconditioner<ConfiguratorType>>( *Hess_F_k );
      else if ( m_preconditioner == INCOMPLETE_CHOLESKY )
        Dinv = std::make_unique<IncompleteCholeskyPreconditioner<ConfiguratorType>>( *Hess_F_k, *m_fixedVariables );
      else if ( m_preconditioner == CHOLESKY )
        Dinv = std::make_unique<CholeskyPreconditioner<ConfiguratorType>>( *Hess_F_k, *m_fixedVariables );
      else if ( m_preconditioner == PROVIDED && m_P )
        Dinv = ( *m_P )( x_k );

      VectorType c = ( *Dinv )( grad_F_k );

      for ( int k = 0; k < m_maxIterations; k++ ) {
        // Step 1: Solve trust-region subproblem
        this->status.Iteration = k;

        auto t_start = std::chrono::high_resolution_clock::now();



        // Compute forcing sequence / epsilon
//        eta_k = std::min( 0.5, std::sqrt( c.norm()));
//        eps_k = eta_k * c.norm();

        RealType gradNorm = std::sqrt( grad_F_k.dot( c ));

        eta_k = std::min( 0.5, gradNorm );
        eps_k = std::min( eps_k, eta_k * gradNorm );
//        eps_k =  eta_k * gradNorm;

        // Apply trust-region subproblem solver
        auto t_start_inner = std::chrono::high_resolution_clock::now();
        if (m_lowerBounds && m_upperBounds) {
          lb = *m_lowerBounds - x_k;
          ub = *m_upperBounds - x_k;
        }

        SolverStatus<ConfiguratorType> trsolverStatus = solveTrustRegionSubproblem( *Hess_F_k, grad_F_k, *Dinv, trRadius, eps_k, lb, ub, p_k );
        auto t_end_inner = std::chrono::high_resolution_clock::now();
        this->status.additionalTimings["Subproblem"] += std::chrono::duration<RealType, std::milli>(
                t_end_inner - t_start_inner ).count();
        this->status.additionalIterations["Subproblem"] += trsolverStatus.Iteration;

//        VectorType tmpVec = p_k;
//        p_k = (*Dinv)(tmpVec);

//        RealType pkn = std::sqrt(p_k.dot((*Dinv)(p_k)));
        RealType pkn = p_k.norm();

//        Dinv->apply( p_k, p_k );

        // Step 2: Determine reduction ration
        t_start_eval_a = std::chrono::high_resolution_clock::now();
        tmp_x_k = x_k + p_k; // temporary new iterate
        m_F.apply( tmp_x_k, tmp_F_k ); // temporary new function value

        m_red = -grad_F_k.dot( p_k ) - 0.5 * p_k.dot( (*Hess_F_k)( p_k ) ); // predicted reduction i.e. in the quadratic model
        f_red = (F_k - tmp_F_k);
        t_end_eval_a = std::chrono::high_resolution_clock::now();
        this->status.additionalTimings["Evaluation"] += std::chrono::duration<RealType, std::milli>(
                t_end_eval_a - t_start_eval_a ).count();

        if ((std::abs( f_red ) < eps_red && std::abs( m_red ) < eps_red) || std::abs( f_red - m_red ) < eps_red ) {
          if ( !m_quiet )
            std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << "Cutoff active in rho-computation: "
                      << std::scientific << std::setprecision(6) << f_red << " vs " << m_red << std::endl;
          rho_k = 1.;
        }
        else
          rho_k = f_red / m_red; // actual over predicted reduction
//        std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << " f_red = " << std::scientific
//                  << std::setprecision( 6 ) << f_red << "; m_red = " << m_red << std::endl;

        // Step 3: Update trust region radius
        if ( rho_k < 0.25 || std::isnan(rho_k) || std::isinf(rho_k) )
          trRadius = trRadius / 4.;
        else if ( rho_k > 0.75 && std::abs( pkn - trRadius ) <= eps_red )
          trRadius = std::min( 2 * trRadius, m_maxRadius );

        // Step 4: Accept or decline new iterate
        if ( rho_k > m_eta && !std::isnan( rho_k ) && !std::isinf( rho_k )) {
          x_k = tmp_x_k;
          F_k = tmp_F_k;

          t_start_eval = std::chrono::high_resolution_clock::now();
          m_DF.apply( x_k, grad_F_k );
          m_F.apply( x_k, F_k );
          Hess_F_k = m_D2F(x_k);
          if ( m_fixedVariables )
            applyMaskToVector<VectorType>( *m_fixedVariables, grad_F_k );

          t_end_eval = std::chrono::high_resolution_clock::now();
          this->status.additionalTimings["Evaluation"] += std::chrono::duration<RealType, std::milli>(
                  t_end_eval - t_start_eval ).count();

          if ( grad_F_k.norm() < m_stopEpsilon ) {
            if ( !m_quiet )
              std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << "Gradient norm below epsilon."
                        << std::endl;
            auto t_end = std::chrono::high_resolution_clock::now();
            this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();
            break;
          }

          // Preconditioning
          t_start_pre = std::chrono::high_resolution_clock::now();

          if ( m_preconditioner == DIAGONAL )
            Dinv = std::make_unique<DiagonalPreconditioner<ConfiguratorType>>( *Hess_F_k );
          else if ( m_preconditioner == INCOMPLETE_CHOLESKY )
            Dinv = std::make_unique<IncompleteCholeskyPreconditioner<ConfiguratorType>>( *Hess_F_k, *m_fixedVariables );
          else if ( m_preconditioner == CHOLESKY )
            Dinv = std::make_unique<CholeskyPreconditioner<ConfiguratorType>>( *Hess_F_k, *m_fixedVariables );
          else if ( m_preconditioner == PROVIDED && m_P )
            Dinv = ( *m_P )( x_k );

          c = ( *Dinv )( grad_F_k );
          t_end_pre = std::chrono::high_resolution_clock::now();
          this->status.additionalTimings["Preconditioner"] += std::chrono::duration<RealType, std::milli>(
                  t_end_pre - t_start_pre ).count();
        }
        else {
          t_end_eval = t_start_eval; // No time spent on evaluation.. Just for the console output.
          t_end_pre = t_start_pre; // No time spent on preconditioner.. Just for the console output.
        }

        if ( trRadius < m_minRadius ) {
          if ( !m_quiet )
            std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << "Trust region too small." << std::endl;
          auto t_end = std::chrono::high_resolution_clock::now();
          this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();
          break;
        }

        if ( p_k.norm() < m_minStepsize ) { // template lpNorm<Eigen::Infinity>
          if ( !m_quiet )
            std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << "Step size too small." << std::endl;
          auto t_end = std::chrono::high_resolution_clock::now();
          this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();
          break;
        }

        if ( std::abs( f_red ) < m_minReduction ) {
          if ( !m_quiet )
            std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << "Value reduction too small." << std::endl;
          auto t_end = std::chrono::high_resolution_clock::now();
          this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();
          break;
        }


        auto t_end = std::chrono::high_resolution_clock::now();
        this->status.totalTime += std::chrono::duration<RealType, std::milli>( t_end - t_start ).count();

        for ( auto &F: m_callbackFcts ) {
          F( k, x_k, F_k, grad_F_k );
        }


        if ( !m_quiet )
          std::cout << " -- TRN -- Iter " << std::setw( 3 ) << k << ": " << std::scientific << std::setprecision( 6 )
                    << F_k
                    << " || " << grad_F_k.norm()
                    << " ||===|| " << std::setw( 13 ) << rho_k
                    << " || " << p_k.template lpNorm<Eigen::Infinity>()
                    << " || " << trRadius
                    << " ||===|| " << trsolverStatus.Residual
                    << " || " << std::fixed << std::setw( 2 ) << trsolverStatus.reasonOfTermination
                    << " || " << std::setw( 4 ) << trsolverStatus.Iteration
                    << std::setprecision( 2 )
                    << " ||===|| " << std::setw( 6 )
                    << std::chrono::duration<RealType, std::milli>( t_end - t_start ).count()
                    << " || " << std::setw( 6 )
                    << std::chrono::duration<RealType, std::milli>( t_end_inner - t_start_inner ).count()
                    << " || " << std::setw( 6 )
                    << std::chrono::duration<RealType, std::milli>( t_end_pre - t_start_pre ).count()
                    << " || " << std::setw( 6 )
                    << std::chrono::duration<RealType, std::milli>( t_end_eval - t_start_eval ).count() +
                       std::chrono::duration<RealType, std::milli>( t_end_eval_a - t_start_eval_a ).count()
                    << std::endl;
      }

      if ( !m_quiet )
        std::cout << " -- TRN -- Final   : " << std::scientific << std::setprecision(6)
                  << F_k
                  << " || " << grad_F_k.norm()
                  << " ||===|| " << std::setw( 13 ) << rho_k
                  << " || " << p_k.template lpNorm<Eigen::Infinity>()
                  << " || " << trRadius
                  << " ||===|| " << std::fixed << std::setprecision(2) << this->status.totalTime
                  << " || " << this->status.additionalTimings["Subproblem"]
                  << std::endl;
    }

    SolverStatus<ConfiguratorType> solveTrustRegionSubproblem( const LinearOperator<ConfiguratorType> &H, const VectorType &c,
                                                               const LinearOperator<ConfiguratorType> &Dinv,
                                                               const RealType &trRadius, const RealType &eps_k,
                                                               const VectorType &lb, VectorType &ub,
                                                               VectorType &p ) const {
      if ( m_subproblemSolver == "SteihaugCG" ) {
        NewOpt::SteihaugCGMethod<ConfiguratorType> trSolver( H, c, Dinv, trRadius, eps_k, m_cgIterations, true );
        trSolver.setParameters( m_trsolverIntParameters );
        trSolver.setParameters( m_trsolverRealParameters );
        trSolver.setParameters( m_trsolverStringParameters );
        trSolver.setParameter( "tolerance", eps_k ); // Just to be safe

        if ( m_fixedVariables )
          trSolver.setFixedVariables( *m_fixedVariables );

        if ( lb.size() > 0 && ub.size() > 0 )
          trSolver.setVariableBounds( lb, ub );


        trSolver.solve( p );

        return trSolver.status;
      }
      else {
        throw std::runtime_error( "TrustRegionNewton::solveTrustRegionSubproblem(): Unknown subproblem method!" );
      }
    }

    void setParameter( const std::string &name, std::string value ) override {
      if ( name == "subproblem_solver" )
        m_subproblemSolver = value;
      else if ( name.rfind( "trsolver__", 0 ) == 0 )
        m_trsolverStringParameters[name.substr( 10, std::string::npos )] = value;
      else
        throw std::runtime_error( "TrustRegionNewton::setParameter(): Unknown parameter '" + name + "'." );
    }

    void setParameter( const std::string &name, RealType value ) override {
      if ( name == "maximal_radius" )
        m_maxRadius = value;
      else if ( name == "initial_radius" )
        m_initRadius = value;
      else if ( name == "minimal_radius" )
        m_minRadius = value;
      else if ( name == "minimal_stepsize" )
        m_minStepsize = value;
      else if ( name == "minimal_reduction" )
        m_minReduction = value;
      else if ( name == "accept_reduction_ratio" )
        m_eta = value;
      else if ( name == "tolerance" )
        m_stopEpsilon = value;
      else if ( name.rfind( "trsolver__", 0 ) == 0 )
        m_trsolverRealParameters[name.substr( 10, std::string::npos )] = value;
      else
        throw std::runtime_error( "TrustRegionNewton::setParameter(): Unknown parameter '" + name + "'." );
    }

    void setParameter( const std::string &name, int value ) override {
      if ( name == "maximum_iterations" )
        m_maxIterations = value;
      else if ( name == "cg_iterations" )
        m_cgIterations = value;
      else if ( name == "print_level" )
        m_quiet = ( value < 5);
      else if ( name == "preconditioner" )
        m_preconditioner = static_cast<PreconditionerType>(value);
      else if ( name.rfind( "trsolver__", 0 ) == 0 )
        m_trsolverIntParameters[name.substr( 10, std::string::npos )] = value;
      else
        throw std::runtime_error( "TrustRegionNewton::setParameter(): Unknown parameter '" + name + "'." );
    }
  };
}
