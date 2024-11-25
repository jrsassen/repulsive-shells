#pragma once

#include <goast/Optimization/optInterface.h>
#include <goast/Optimization/optParameters.h>
#include <goast/Optimization/stepsizeControl.h>
#include <goast/Optimization/LinearOperator.h>

#include "Preconditioners.h"

namespace NewOpt {
/**
 * \brief Simple gradient descent method
 * \tparam ConfiguratorType Container with data types
 * \author Heeren
 */
  template<typename ConfiguratorType=DefaultConfigurator>
  class PreconditionedGradientDescent {

  protected:
    using RealType = typename ConfiguratorType::RealType;

    using VectorType = typename ConfiguratorType::VectorType;

    const BaseOp<VectorType, RealType> &_E;
    const BaseOp<VectorType, VectorType> &_DE;
    const MapToLinOp<ConfiguratorType> &m_Pre;

    TIMESTEP_CONTROLLER _timestepController;
    StepsizeControl <ConfiguratorType> _stepsizeControl;

    int _maxIterations;
    RealType _stopEpsilon;
    bool _useNonlinearCG, _useGradientBasedStopping;
    QUIET_MODE _quietMode;
    const std::vector<int> *_bdryMask;

  public:
    PreconditionedGradientDescent( const BaseOp <VectorType, RealType> &E,
                                   const BaseOp <VectorType, VectorType> &DE,
                                   const MapToLinOp<ConfiguratorType> &Pre,
                                   int MaxIterations,
                                   RealType StopEpsilon,
                                   TIMESTEP_CONTROLLER TimestepController,
                                   bool quiet,
                                   RealType sigma = 0.1,
                                   RealType tauMin = 1.e-6,
                                   RealType tauMax = 4. )
            : _E( E ), _DE( DE ), _timestepController( static_cast<TIMESTEP_CONTROLLER>(TimestepController)),
              _stepsizeControl( _E, _DE, _timestepController, sigma, 0.9, 1., tauMin, tauMax ),
              _maxIterations( MaxIterations ), _stopEpsilon( StopEpsilon ), _useNonlinearCG( false ),
              _useGradientBasedStopping( true ), _quietMode( quiet ? SUPERQUIET : SHOW_ALL ), _bdryMask( nullptr ),
              m_Pre( Pre ) {}

    PreconditionedGradientDescent( const BaseOp <VectorType, RealType> &E,
                                   const BaseOp <VectorType, VectorType> &DE,
                                   const MapToLinOp<ConfiguratorType> &Pre,
                                   int MaxIterations = 1000,
                                   RealType StopEpsilon = 1e-8,
                                   TIMESTEP_CONTROLLER TimestepController = ARMIJO,
                                   QUIET_MODE quietMode = SUPERQUIET,
                                   RealType sigma = 0.1,
                                   RealType tauMin = 1.e-6,
                                   RealType tauMax = 4. )
            : _E( E ), _DE( DE ), _timestepController( static_cast<TIMESTEP_CONTROLLER>(TimestepController)),
              _stepsizeControl( _E, _DE, _timestepController, sigma, 0.9, 1., tauMin, tauMax ),
              _maxIterations( MaxIterations ), _stopEpsilon( StopEpsilon ), _useNonlinearCG( false ),
              _useGradientBasedStopping( true ), _quietMode( quietMode ), _bdryMask( nullptr ),
              m_Pre( Pre ) {}

    PreconditionedGradientDescent( const BaseOp <VectorType, RealType> &E,
                                   const BaseOp <VectorType, VectorType> &DE,
                                   const MapToLinOp<ConfiguratorType> &Pre,
                                   const OptimizationParameters <ConfiguratorType> &optPars )
            : _E( E ), _DE( DE ), _timestepController( static_cast<TIMESTEP_CONTROLLER>(optPars.getGradTimeStepping())),
              _stepsizeControl( _E, _DE, _timestepController, optPars.getSigma(), optPars.getBeta(),
                                optPars.getStartTau(), optPars.getTauMin(), optPars.getTauMax()),
              _maxIterations( optPars.getGradientIterations()), _stopEpsilon( optPars.getStopEpsilon()),
              _useNonlinearCG( false ), _useGradientBasedStopping( true ), _quietMode( optPars.getQuietMode()),
              _bdryMask( nullptr ),
              m_Pre( Pre ) {}

    void setGradientBasedStopping( bool useGradientBasedStopping = true ) {
      _useGradientBasedStopping = useGradientBasedStopping;
    }

    void setBoundaryMask( const std::vector<int> &Mask ) {
      _bdryMask = &Mask;
      _stepsizeControl.setBoundaryMask( Mask );
    }


    void setNonlinearCG( const bool value ) {
      _useNonlinearCG = value;
    }

    // x^{k+1} = x^k - tau * E'[x^k]
    void solve( const VectorType &x_0, VectorType &x_k ) const {

      if ( _maxIterations == 0 )
        return;

      x_k = x_0;
      RealType tau = _stepsizeControl.getStartTau();

      // compute initla energy
      RealType energyScalar;
      _E.apply( x_k, energyScalar );                             // just for the stopping criterion
      RealType energy = energyScalar;
      RealType energyNew = energyScalar;

      if ( _quietMode == SHOW_ALL ) {
        std::cout << "========================================================================================="
                  << std::endl;
        std::cout << "Start gradient descent with " << _maxIterations << " iterations and eps = " << _stopEpsilon << "."
                  << std::endl;
        std::cout << "Initial energy: " << energy << std::endl;
        std::cout << "========================================================================================="
                  << std::endl;
      }

      int iterations = 0;
      bool stoppingCriterion;
      RealType error;

      // store old gradient and direction for nonlinear CG
      VectorType oldDirection( x_k.size()), oldGradient( x_k.size()), currGradient;
      _DE.apply( x_k, currGradient );
      if ( _bdryMask )
        applyMaskToVector( *_bdryMask, currGradient );

      // iteration loop
      do {
        auto t_start = std::chrono::high_resolution_clock::now();
        iterations++;
        // Save the energy at the current position.
        energy = energyNew;

        // compute descent diraction (ie. negative gradient)
        VectorType descentDir( currGradient );
        descentDir *= -1.;

        // apply preconditioner to descent direction if given
        auto P = m_Pre( x_k );
        descentDir = (*P)( descentDir );


        // Fletcher-Reeves nonlinear conjugate gradient method.
        if ( _useNonlinearCG ) {
          const RealType nonlinCGBeta = (( iterations - 1 ) % 10 ) ? ( descentDir.squaredNorm() /
                                                                       oldGradient.squaredNorm())
                                                                   : 0.;
          oldGradient = descentDir;
          if ( nonlinCGBeta > 0. )
            descentDir += nonlinCGBeta * oldDirection;
          oldDirection = descentDir;
        }

        // compute tau
        tau = _stepsizeControl.getStepsize( x_k, currGradient, descentDir, tau, energy );

        // If the nonlinear CG direction is not a descent direction, use the normal descent direction instead.
        if ( _useNonlinearCG && ( tau == 0 )) {
          descentDir = oldGradient;
          oldDirection.setZero();
          tau = _stepsizeControl.getStepsize( x_k, currGradient, descentDir, tau, energyNew );
        }

        // update position
        x_k += tau * descentDir;

        // Calculate energy and gradient at the new position
        _DE.apply( x_k, currGradient );
        if ( _bdryMask )
          applyMaskToVector( *_bdryMask, currGradient );
        _E.apply( x_k, energyScalar );
        energyNew = energyScalar;

        // compute error
        error = _useGradientBasedStopping ? currGradient.norm() : energy - energyNew;

        auto t_end = std::chrono::high_resolution_clock::now();

        if ( _quietMode == SHOW_ALL )
          std::cout << "step = " << iterations
                    << std::scientific << " , stepsize = " << tau << ", energy = " << energyNew << ", error = " << error
                    << std::fixed << ", time = " << std::chrono::duration<double, std::milli>( t_end - t_start ).count()
                    << "ms" << std::endl;

        // stoppingCriterion == true means the iteration will be continued
        stoppingCriterion = ( error > _stopEpsilon ) && ( iterations < _maxIterations ) && ( tau > 0 );

      } while ( stoppingCriterion ); // end iteration loop

      if ( _quietMode != SUPERQUIET ) {
        std::cout << "========================================================================================="
                  << std::endl;
        std::cout << "Finished gradient descent after " << iterations << " steps (max. steps = " << _maxIterations
                  << ", tol = " << _stopEpsilon << ")." << std::endl;
        std::cout << "Final stepsize = " << std::scientific << tau << ", energy = " << energyNew << ", error = "
                  << error
                  << std::endl;
        std::cout << "========================================================================================="
                  << std::endl;
      }

    }


  };

}

