#pragma once

#include <goast/Optimization/optInterface.h>
#include <goast/Optimization/stepsizeControl.h>
#include <goast/Optimization/optParameters.h>
#include <goast/Optimization/simpleNewton.h>
#include <goast/Core/LinearSolver.h>

namespace NewOpt {
  /**
   * \brief Stepsize control for Newton
   * \author Heeren
   */
  template<typename ConfiguratorType=DefaultConfigurator>
  class StepsizeControlForNewton : public StepsizeControlInterface<ConfiguratorType> {

  protected:
    using RealType = typename ConfiguratorType::RealType;
    using MatrixType = typename ConfiguratorType::SparseMatrixType;
    using VectorType = typename ConfiguratorType::VectorType;

    const BaseOp<VectorType, VectorType> &m_F;
    const std::unique_ptr<LinearOperator<ConfiguratorType>> &m_Jacobi;

  public:
    StepsizeControlForNewton( const BaseOp<VectorType, VectorType> &F,
                              const std::unique_ptr<LinearOperator<ConfiguratorType>> &Jacobi,
                              TIMESTEP_CONTROLLER timestepController = NEWTON_OPTIMAL,
                              RealType sigma = 0.1,
                              RealType beta = 0.9,
                              RealType startTau = 1.0,
                              RealType tauMin = 1e-12,
                              RealType tauMax = 4. ) : StepsizeControlInterface<ConfiguratorType>( sigma, beta,
                                                                                                   timestepController,
                                                                                                   startTau, tauMin,
                                                                                                   tauMax ), m_F( F ),
                                                       m_Jacobi( Jacobi ) {}

    //Returns the scalar objective function evaluated at CurrentPosition.
    RealType evaluateLinesearchFnc( const VectorType &Position ) const {
      VectorType temp( Position.size());
      m_F.apply( Position, temp );
      if ( this->_bdryMask )
        applyMaskToVector( *this->_bdryMask, temp );
      return 0.5 * temp.squaredNorm();
    }

    //Returns the dot product of the energy derivative and the descent direction.
    RealType evaluateLinesearchGrad( const VectorType &Position, const VectorType &DescentDir ) const {
      VectorType outer = m_Jacobi->operator()( DescentDir );
      VectorType grad( DescentDir.size());
      m_F.apply( Position, grad );
      if ( this->_bdryMask )
        applyMaskToVector( *this->_bdryMask, grad );
      return outer.dot( grad );
    }

    RealType evaluateLinesearchFnc( const VectorType &CurrentPosition, const VectorType &DescentDir,
                                    RealType timestepWidth ) const {
      return evaluateLinesearchFnc( CurrentPosition + timestepWidth * DescentDir );
    }

    // Calculates the step size based on "Schaback, Werner - Numerische Mathematik, 4. Auflage, Seite 129". Is global convergent, as long as F fulfils certain regularity conditions.
    RealType getNewtonOptimalTimestepWidth( const VectorType &CurrentPosition, const VectorType &DescentDir ) const {

      const RealType DescentDirNormSqr = DescentDir.squaredNorm();
//      std::cout << " .. DescentDirNormSqr = " << DescentDirNormSqr << std::endl;

      if ( !( DescentDirNormSqr > 0. ))
        return 0.;

      // initial guess for L: L = ||F(y)-F(x)-F'(x)(y-x)|| / ||y-x||^2, where x = CurrentPosition, y = x + DescentDir
      VectorType newPosition = CurrentPosition + 1.e-1 * DescentDir;

      VectorType pTmp( DescentDir.size());
      m_F.apply( newPosition, pTmp );
      if ( this->_bdryMask )
        applyMaskToVector( *this->_bdryMask, pTmp );
      pTmp -= m_Jacobi->operator()( DescentDir );
      VectorType Fx( DescentDir.size());
      m_F.apply( CurrentPosition, Fx );
      if ( this->_bdryMask )
        applyMaskToVector( *this->_bdryMask, Fx );
      pTmp -= Fx;
      RealType L = pTmp.norm() / DescentDirNormSqr;

//      std::cout << " .. L = " << L << std::endl;
//      std::cout << " .. pTmp.norm = " <<  pTmp.norm() << std::endl;
//      std::cout << " .. Fx.norm = " <<  Fx.norm() << std::endl;
//      std::cout << " .. m_F(newPosition).norm = " << m_F( newPosition ).norm() << std::endl;
//      std::cout << " .. m_F(CurrentPosition).norm = " << m_F( CurrentPosition ).norm() << std::endl;
//      std::cout << " .. m_Jacobi*DescentDir = " << m_Jacobi->operator()( DescentDir ).norm() << std::endl;
//
//      std::cout << " .. DescentDirNormSqr = " << DescentDirNormSqr << std::endl;

      // If F is locally linear and matches the gradient with numerical perfection take the full step
      if ( L < 1e-15 )
        return 1.e-1;

      RealType fNorm = Fx.norm();
      RealType tau = std::min( fNorm / ( 2. * L * DescentDirNormSqr ), 1.0 );
      RealType temp1 = fNorm - std::sqrt( 2. * evaluateLinesearchFnc( CurrentPosition, DescentDir, tau ));
      RealType temp2 = tau * ( fNorm - L * tau * DescentDirNormSqr );
      if (L == std::numeric_limits<RealType>::infinity())
        temp2 = -std::numeric_limits<RealType>::infinity();

      RealType tauCandidate;
      if ( temp1 < temp2 ) {
        do {
          L *= 2.;
          tau = std::min( fNorm / ( 2. * L * DescentDirNormSqr ), 1.0 );
          temp1 = fNorm - sqrt( 2. * evaluateLinesearchFnc( CurrentPosition, DescentDir, tau ));
          temp2 = tau * ( fNorm - L * tau * DescentDirNormSqr );
          // Prevent the while loop from getting stuck if DescentDir is not a descent direction.
          if ( tau < this->_tauMin ) {
            temp1 = temp2;
            tau = 0.;
          }
        } while ( temp1 < temp2 );
      }
      else {
        do {
          // Compute tauCandidate, temp1 and temp2 with L/2 instead of L to find out if a bigger step size is still feasible.
          tauCandidate = std::min( fNorm / ( L * DescentDirNormSqr ), 1.0 );
          if ( tauCandidate == 1.0 )
            break;
          temp1 = fNorm - std::sqrt( 2. * evaluateLinesearchFnc( CurrentPosition, DescentDir, tauCandidate ));
          temp2 = tauCandidate * ( fNorm - 0.5 * L * tauCandidate * DescentDirNormSqr );
          if (L == std::numeric_limits<RealType>::infinity())
            temp2 = -std::numeric_limits<RealType>::infinity();

//          std::cout << " .. fNorm = " << fNorm << std::endl;
//          std::cout << " .. L = " << L << std::endl;
//          std::cout << " .. tauCandidate = " << tauCandidate << std::endl;
//          std::cout << " .. DescentDirNormSqr = " << DescentDirNormSqr << std::endl;
//          std::cout << " .. temp1 = " << temp1 << std::endl;
//          std::cout << " .. temp2 = " << temp2 << std::endl;
//          std::cout << " .. evaluateLinesearchFnc = " << evaluateLinesearchFnc( CurrentPosition, DescentDir, tauCandidate ) << std::endl;

          if ( temp1 >= temp2 ) {
            L /= 2.;
            tau = tauCandidate;

            if (tau <= this->_tauMin)
              break;
          }
        } while ( temp1 >= temp2 );
      }

      // return
      return tau > this->_tauMin ? tau : 0.;

    }

    RealType getStepsize( const VectorType &CurrentPosition, const VectorType &CurrentGradient,
                          const VectorType &descentDir, RealType tau_before, RealType currEnergy = -1. ) const {
      switch ( this->_timestepController ) {
        case NEWTON_OPTIMAL:
          return getNewtonOptimalTimestepWidth( CurrentPosition, descentDir );
        case ARMIJO:
          return StepsizeControlInterface<ConfiguratorType>::getTimestepWidthWithArmijoLineSearch( CurrentPosition, -CurrentGradient, descentDir, tau_before, currEnergy );
        default:
          return StepsizeControlInterface<ConfiguratorType>::getStepsize( CurrentPosition, CurrentGradient, descentDir,
                                                                          tau_before, currEnergy );
      }
    }

  };


  /**
   * \brief Newton method to find a root of a vector valued functional F: \R^n -> \R^n
   * \author Heeren
   */
  template<typename ConfiguratorType=DefaultConfigurator>
  class NewtonMethod {

  protected:
    using RealType = typename ConfiguratorType::RealType;

    using VectorType = typename ConfiguratorType::VectorType;
    using MatrixType = typename ConfiguratorType::SparseMatrixType;

    const BaseOp<VectorType, VectorType> &m_F;
    const MapToLinOp<ConfiguratorType> &m_DF;
    const MapToLinOp<ConfiguratorType> *m_invDF;
    mutable std::unique_ptr<LinearOperator<ConfiguratorType>> m_Jacobi;

    TIMESTEP_CONTROLLER m_timestepController;
    std::unique_ptr<StepsizeControlInterface<ConfiguratorType>> _stepsizeControlPtr;

    LINEAR_SOLVER_TYPE _solverType;
    int _maxIterations;
    RealType _stopEpsilon;
    QUIET_MODE _quietMode;
    const std::vector<int> *_bdryMask;

  public:
    mutable SolverStatus<ConfiguratorType> status;

    NewtonMethod( const BaseOp<VectorType, VectorType> &F,
                  const MapToLinOp<ConfiguratorType> &DF,
                  int MaxIterations,
                  RealType StopEpsilon,
                  TIMESTEP_CONTROLLER TimestepController,
                  bool quiet,
                  RealType sigma = 0.1,
                  RealType tauMin = 1.e-6,
                  RealType tauMax = 4. )
            : m_F( F ), m_DF( DF ), m_timestepController( static_cast<TIMESTEP_CONTROLLER>(TimestepController)),
              _solverType( UMFPACK_LU_FACT ), _maxIterations( MaxIterations ), _stopEpsilon( StopEpsilon ),
              _quietMode( quiet ? SUPERQUIET : SHOW_ALL ), _bdryMask( nullptr ), m_invDF( nullptr ) {

      _stepsizeControlPtr = std::make_unique<NewOpt::StepsizeControlForNewton<ConfiguratorType>>( m_F, m_Jacobi,
                                                                                          m_timestepController, sigma,
                                                                                          0.9, 1., tauMin,
                                                                                          tauMax );
    }

    NewtonMethod( const BaseOp<VectorType, VectorType> &F,
                  const MapToLinOp<ConfiguratorType> &DF,
                  int MaxIterations = 1000,
                  RealType StopEpsilon = 1e-8,
                  TIMESTEP_CONTROLLER TimestepController = NEWTON_OPTIMAL,
                  QUIET_MODE quietMode = SUPERQUIET,
                  RealType sigma = 0.1,
                  RealType tauMin = 1.e-6,
                  RealType tauMax = 4. )
            : m_F( F ), m_DF( DF ), m_timestepController( static_cast<TIMESTEP_CONTROLLER>(TimestepController)),
              _solverType( UMFPACK_LU_FACT ), _maxIterations( MaxIterations ), _stopEpsilon( StopEpsilon ),
              _quietMode( quietMode ), _bdryMask( nullptr ), m_invDF( nullptr ) {

      _stepsizeControlPtr = std::make_unique<NewOpt::StepsizeControlForNewton<ConfiguratorType>>( m_F, m_Jacobi,
                                                                                          m_timestepController, sigma,
                                                                                          0.9, 1., tauMin,
                                                                                          tauMax );
    }

    NewtonMethod( const BaseOp<VectorType, VectorType> &F,
                  const MapToLinOp<ConfiguratorType> &DF,
                  const MapToLinOp<ConfiguratorType> &invDF,
                  int MaxIterations = 1000,
                  RealType StopEpsilon = 1e-8,
                  TIMESTEP_CONTROLLER TimestepController = NEWTON_OPTIMAL,
                  QUIET_MODE quietMode = SUPERQUIET,
                  RealType sigma = 0.1,
                  RealType tauMin = 1.e-6,
                  RealType tauMax = 4. )
            : m_F( F ), m_DF( DF ), m_invDF( &invDF ),
              m_timestepController( static_cast<TIMESTEP_CONTROLLER>(TimestepController)),
              _solverType( UMFPACK_LU_FACT ), _maxIterations( MaxIterations ), _stopEpsilon( StopEpsilon ),
              _quietMode( quietMode ), _bdryMask( nullptr ) {

      _stepsizeControlPtr = std::make_unique<NewOpt::StepsizeControlForNewton<ConfiguratorType>>( m_F, m_Jacobi,
                                                                                          m_timestepController, sigma,
                                                                                          0.9, 1., tauMin,
                                                                                          tauMax );
    }

    NewtonMethod( const BaseOp<VectorType, VectorType> &F,
                  const MapToLinOp<ConfiguratorType> &invDF,
                  const OptimizationParameters<ConfiguratorType> &optPars )
            : m_F( F ), m_invDF( invDF ),
              m_timestepController( static_cast<TIMESTEP_CONTROLLER>(optPars.getNewtonTimeStepping())),
              _solverType( static_cast<LINEAR_SOLVER_TYPE>(optPars.getSolverType())),
              _maxIterations( optPars.getNewtonIterations()), _stopEpsilon( optPars.getStopEpsilon()),
              _quietMode( optPars.getQuietMode()), _bdryMask( nullptr ) {

      _stepsizeControlPtr = std::make_unique<NewOpt::StepsizeControlForNewton<ConfiguratorType>>( m_F, m_Jacobi,
                                                                                          m_timestepController,
                                                                                          optPars.getSigma(),
                                                                                          optPars.getBeta(),
                                                                                          optPars.getStartTau(),
                                                                                          optPars.getTauMin(),
                                                                                          optPars.getTauMax());
    }

    // residuum in iteration
    RealType computeErrorNorm( const VectorType &x_k, const VectorType &delta_x_k, const VectorType &F_x_k,
                               RealType tau, bool initial ) const {
//      if ( initial )
//        return 1.;
//      else
//        return delta_x_k.norm();

      return F_x_k.norm();
    }

    //
    void setSolver( LINEAR_SOLVER_TYPE solverType ) {
      _solverType = solverType;
    }

    // set boundary mask
    void setBoundaryMask( const std::vector<int> &Mask ) {
      _bdryMask = &Mask;
      _stepsizeControlPtr->setBoundaryMask( Mask );
    }

    // x^{k+1} = x^k + tau * d^k, where d^k solves D^2E[x^k] d^k = - DE[x^k]
    bool solve( const VectorType &x_0, VectorType &x_k ) const {
      status.Iteration = 0;
      status.totalTime = 0.;

      x_k = x_0;

      VectorType F_x_k( x_k.size());
      std::unique_ptr<LinearOperator<ConfiguratorType>> invDF_k;
      VectorType delta_x_k( x_k.size());
      m_F.apply( x_k, F_x_k );
      if ( _bdryMask )
        applyMaskToVector( *_bdryMask, F_x_k );

      MatrixType JacobiMat( x_k.size(), x_k.size());

      RealType tau = _stepsizeControlPtr->getStartTau();
      RealType FNorm = computeErrorNorm( x_k, delta_x_k, F_x_k, tau, true );

      if ( _maxIterations == 0 )
        return FNorm <= _stopEpsilon;

      if ( _quietMode == SHOW_ALL ) {
        std::cout << "========================================================================================="
                  << std::endl;
        std::cout << "Start Newton method with " << _maxIterations << " iterations and eps = " << _stopEpsilon << "."
                  << std::endl;
        writeOutput( x_k, 0, tau, FNorm, false );
        std::cout << "========================================================================================="
                  << std::endl;
      }


      int iterations = 0;
      while ( FNorm > _stopEpsilon && ( iterations < _maxIterations ) && tau > 0. ) {
        iterations++;

        auto t_start = std::chrono::high_resolution_clock::now();

        // Newton iteration given by x^{k+1} = x^k - tau D2F(x^k)^{-1}(DF(x^k))
        m_Jacobi = m_DF( x_k );
        //TODO check whether mask also works for non-symmetric matrices!
        if ( _bdryMask )
          applyMaskToVector( *_bdryMask, F_x_k );

        //std::cout << _Jacobi << std::endl;
        VectorType rhs = -F_x_k;
        if ( m_invDF ) {
          invDF_k = m_invDF->operator()( x_k );
          invDF_k->apply( rhs, delta_x_k );
        }
        else {
          m_Jacobi->assembleTransformationMatrix( JacobiMat );
          if ( _bdryMask )
            applyMaskToSymmetricMatrix( *_bdryMask, JacobiMat );
          LinearSolver<ConfiguratorType>( _solverType ).solve( JacobiMat, rhs, delta_x_k );
        }

        // get tau
        tau = _stepsizeControlPtr->getStepsize( x_k, F_x_k, delta_x_k, tau );

        if ( tau > 0 ) {
          // update position and descent direction
          x_k += tau * delta_x_k;
          m_F.apply( x_k, F_x_k );
          if ( _bdryMask )
            applyMaskToVector( *_bdryMask, F_x_k );
          FNorm = computeErrorNorm( x_k, delta_x_k, F_x_k, tau, false );

          status.Iteration++;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        status.totalTime += std::chrono::duration<double, std::milli>( t_end - t_start ).count();

        if ( _quietMode == SHOW_ALL )
          writeOutput( x_k, iterations, tau, FNorm,
                       std::chrono::duration<double, std::milli>( t_end - t_start ).count());
      } // end while

      if (( _quietMode == SHOW_ALL ) || ( _quietMode == SHOW_TERMINATION_INFO ) ||
          (( _quietMode == SHOW_ONLY_IF_FAILED ) && ( FNorm > _stopEpsilon ))) {
        std::cout << "========================================================================================="
                  << std::scientific << std::endl;
        std::cout << "Finished Newton's method after " << iterations << " steps (max. steps = " << _maxIterations
                  << ", tol = " << _stopEpsilon << ")." << std::endl;
        std::cout << "Final stepsize = " << tau << ", final norm = " << FNorm << std::endl;
        writeOutput( x_k, iterations, tau, FNorm, status.totalTime, false );
        std::cout << "========================================================================================="
                  << std::endl << std::endl;
      }

      return FNorm <= _stopEpsilon;

    }

  protected:

    virtual void writeOutput( const VectorType &/*x_k*/, int iterations, RealType tau, RealType norm, RealType time,
                              bool intermediate = true ) const {
      if ( intermediate )
        std::cout << std::scientific << "step = " << iterations << ", stepsize = " << tau << ", norm = " << norm
                  << std::fixed << ", t = " << time << "ms" << std::endl;
    }


  };
}
