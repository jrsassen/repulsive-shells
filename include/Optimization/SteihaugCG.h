#pragma once

#include <goast/Core.h>
#include <goast/Optimization/optUtils.h>
#include <goast/Optimization/optInterface.h>
#include <goast/Optimization/LinearOperator.h>

namespace NewOpt {

//  template<typename ConfiguratorType=DefaultConfigurator>
//  std::tuple<typename ConfiguratorType::RealType, typename ConfiguratorType::RealType, bool> lineBallIntersection( const typename ConfiguratorType::VectorType &u,
//                                                                                                                   const typename ConfiguratorType::VectorType &v,
//                                                                                                                   const LinearOperator<ConfiguratorType> &G,
//                                                                                                                   const typename ConfiguratorType::RealType radius,
//                                                                                                                   const typename ConfiguratorType::RealType min_t = -std::numeric_limits<typename ConfiguratorType::RealType>::infinity(),
//                                                                                                                   const typename ConfiguratorType::RealType max_t = std::numeric_limits<typename ConfiguratorType::RealType>::infinity()) {
//    using RealType = typename ConfiguratorType::RealType;
//    if ( v.norm() == 0 )
//      return std::make_tuple( 0., 0., false );
//
//    if ( radius == std::numeric_limits<RealType>::infinity())
//      return std::make_tuple( min_t, max_t, true );
//
//    // coefficients of quadratic equation
//    RealType a = v.dot( G( v ));
//    RealType b = 2 * u.dot( G( v ));
//    RealType c = u.dot( G( u )) - radius * radius;
//
//    // abc-Formula
//    RealType discriminant = b * b - 4 * a * c;
//
//    if ( discriminant < 0 ) {
//      std::cout << "DISC NEGATIVE" << std::endl;
//      return std::make_tuple( 0., 0., false );
//    }
//
//    // For stability reasons we first determine the root where b and the discriminant have the same sign
//    // and then derive the by the relation r_2 = (-c/a)/r_1
//    RealType aux = b + std::copysign( std::sqrt( discriminant ), b );
//    RealType r_1 = -aux / (2 * a);
//    RealType r_2 = -2 * c / aux;
//
//    RealType r_a, r_b;
//    std::tie( r_a, r_b ) = std::minmax( r_1, r_2 );
//
//    if ( r_b < min_t || r_a > max_t )
//      return std::make_tuple( 0., 0., false );
//    else
//      return std::make_tuple( std::max( min_t, r_a ), std::min( max_t, r_b ), true );
//  }

  template<typename ConfiguratorType=DefaultConfigurator>
  class SteihaugCGMethod : public OptimizationBase<ConfiguratorType>,
                           public TimedClass<SteihaugCGMethod<ConfiguratorType>> {
    using RealType = typename ConfiguratorType::RealType;

    using VectorType = typename ConfiguratorType::VectorType;
    using MatrixType = typename ConfiguratorType::SparseMatrixType;
    using FullMatrixType = typename ConfiguratorType::FullMatrixType;
    using TripletType = typename ConfiguratorType::TripletType;
    using TensorType = typename ConfiguratorType::TensorType;

    using typename TimedClass<SteihaugCGMethod<ConfiguratorType>>::ScopeTimer;

    const LinearOperator<ConfiguratorType> &m_H;
    const LinearOperator<ConfiguratorType> &m_Dinv;
    const VectorType &m_c;

    const RealType m_radius;
    RealType m_tolerance;
    int m_maxIterations;
    int m_maxInfeasibleIterations = 10;

    bool m_quiet;

    const std::vector<int> *m_fixedVariables = nullptr;
    const VectorType *m_lowerBounds = nullptr;
    const VectorType *m_upperBounds = nullptr;

  public:
    SteihaugCGMethod( const LinearOperator<ConfiguratorType> &H, const VectorType &c,
                      const LinearOperator<ConfiguratorType> &Dinv, const RealType radius,
                      const RealType tolerance, const int maxIterations = 1000, bool quiet = false )
            : m_H( H ), m_c( c ), m_Dinv( Dinv ), m_tolerance( tolerance ), m_radius( radius ),
              m_maxIterations( maxIterations ), m_quiet( quiet ) {}

    void setFixedVariables( const std::vector<int> &fixedVariables ) override {
      m_fixedVariables = &fixedVariables;
    }

    void setVariableBounds( const VectorType &lowerBounds, const VectorType &upperBounds ) override {
      m_lowerBounds = &lowerBounds;
      m_upperBounds = &upperBounds;
    }

    void solve( const VectorType & /*start*/, VectorType &s ) const override {
      solve( s );
    }

    void solve( VectorType &z_k ) const {
      ScopeTimer timer( "solve" );
      this->status.Iteration = 0;
      this->status.reasonOfTermination = -1;

      z_k.resize( m_c.size());
      z_k.setZero();

//    VectorType r_k( _Dinv * _c );
      VectorType r_k = m_c;
      VectorType y_k;

      {
        ScopeTimer innerTimer( "solve::evaluatePreconditioner" );
        y_k = m_Dinv( r_k );
        if ( m_fixedVariables )
          applyMaskToVector( *m_fixedVariables, y_k );
      }

      VectorType d_k = -y_k;


      RealType r_sqNorm = r_k.dot( y_k ); // r_k^T r_k
      this->status.Residual = std::sqrt( r_sqNorm );

      // Stop if 0 is already a good enough solution
      if ( this->status.Residual  < m_tolerance ) {
        std::cout << "stopped with inital residual " << this->status.Residual << " vs. " << m_tolerance << std::endl;
        this->status.reasonOfTermination = 0;
        return;
      }

      // Intermediate and temporary quantities
      VectorType Hd; // H * d_k
      RealType alpha_k, beta_k, temp;

      VectorType tmpVector;

      // Helper variables for line / trust-region intersection
      RealType tau_0, tau_1;
      bool intersected;

      // Helper variables for box constraints
      int feasible_counter = 0;
      VectorType last_feasible_z( z_k.size());

      for ( int k = 0; k < m_maxIterations; k++ ) {
        this->status.Iteration++;

        {
          ScopeTimer innerTimer( "solve::evaluateOperatorProduct" );
          Hd = m_H( d_k );
        }

        if ( m_fixedVariables )
          applyMaskToVector( *m_fixedVariables, Hd );

        // Step 1: Check if current search direction is one of nonpositive curvature and if yes return intersection with
        // boundary of trust region
        RealType curvature = d_k.transpose() * Hd;
        if ( curvature <= 0 ) {
          // TODO: replace by proper lineSphereIntersection
          if ( m_lowerBounds && m_upperBounds )
            std::tie( tau_0, tau_1, intersected) = lineBoxBallIntersections( z_k, d_k, m_radius,
                                                                             *m_lowerBounds, *m_upperBounds,
                                                                             -std::numeric_limits<RealType>::infinity(),
                                                                             std::numeric_limits<RealType>::infinity());
          else
            std::tie( tau_0, tau_1, intersected ) = lineSphereIntersection<RealType, VectorType>( z_k, d_k, m_radius );

          if ( !intersected )
            throw std::runtime_error( "SteihaugCGMethod: Negative curvature line does not intersect with ball!" );

          z_k += tau_1 * d_k;

          this->status.reasonOfTermination = 1;

          return;
        }

        // Step 2: Compute new iterate
        alpha_k = r_sqNorm / curvature;
        tmpVector = z_k + alpha_k * d_k;

        // Step 2a: If new iterate is outside of trust region then return intersection with boundary of trust region
        if ( tmpVector.norm() >= m_radius ) {
          if ( m_lowerBounds && m_upperBounds )
            std::tie( tau_0, tau_1, intersected ) = lineBoxBallIntersections( z_k, d_k, m_radius, *m_lowerBounds,
                                                                              *m_upperBounds, 0., alpha_k );
          else
            std::tie( tau_0, tau_1, intersected ) = lineBallIntersection<RealType, VectorType>( z_k, d_k, m_radius, 0.,
                                                                                                alpha_k );

          if ( !intersected )
            throw std::runtime_error( "SteihaugCGMethod: Search direction does not intersect with ball!" );
          if ( tau_1 < 0 )
            throw std::runtime_error( "SteihaugCGMethod: tau_1 < 0" );

          z_k += tau_1 * d_k;

          this->status.reasonOfTermination = 2;

          return;
        }

        // Step 2b: If we have box constraints and new iterate violates them, then project it onto the box
        if ( m_lowerBounds && m_upperBounds ) {
          if ( insideBox( tmpVector, *m_lowerBounds, *m_upperBounds )) {
            feasible_counter = 0;
          }
          else {
            feasible_counter += 1;

            std::tie( tau_0, tau_1, intersected ) = lineBoxBallIntersections( z_k, d_k, m_radius, *m_lowerBounds,
                                                                              *m_upperBounds, 0., alpha_k );
            if ( intersected ) {
              last_feasible_z = z_k + tau_1 * d_k;
              feasible_counter = 0;
            }
          }

          if ( feasible_counter > m_maxInfeasibleIterations ) {
            if ( !m_quiet )
              std::cout << "SCG -- Iter " << std::setw( 3 ) << k << ": " << "Too many infeasible iterations: "
                        << feasible_counter << std::endl;
            this->status.reasonOfTermination = 3;
            break;
          }
        }

        z_k = tmpVector; // Accept iterate

        // Step 3: Update residual and check for convergence
        r_k += alpha_k * Hd; // r_{k+1} = r_k + alpha_k * H * d_k
        {
          ScopeTimer innerTimer( "solve::evaluatePreconditioner" );
          y_k = m_Dinv( r_k );
          if ( m_fixedVariables )
            applyMaskToVector( *m_fixedVariables, y_k );
        }

        temp = r_sqNorm;
        r_sqNorm = r_k.dot(y_k);

        this->status.Residual = std::sqrt(r_sqNorm);
        if ( this->status.Residual < m_tolerance ) {
          this->status.reasonOfTermination = 0;
          return;
        }

        // Step 4: Compute new search direction
        beta_k = r_sqNorm / temp; // beta_{k+1} = r_{k+1}^T r_{k+1} / r_k^T r_k
        d_k *= beta_k;
        d_k -= y_k; // d_{k+1} = -r_{k+1} + beta_{k+1} * d_k

        if (!m_quiet)
          std::cout << "SCG -- Iter " << k << ": ||r_k|| = " << r_k.norm() << std::endl;
      }

      if ( m_lowerBounds && m_upperBounds )
        if ( !insideBox( z_k, *m_lowerBounds, *m_upperBounds ))
          z_k = last_feasible_z;
    }

    void setParameter( const std::string &name, std::string value ) override {
      throw std::runtime_error( "SteihaugCGMethod::setParameter(): This class has no string parameters.." );
    }

    void setParameter( const std::string &name, RealType value ) override {
      if ( name == "tolerance" )
        m_tolerance = value;
      else
        throw std::runtime_error( "SteihaugCGMethod::setParameter(): Unknown parameter '" + name + "'." );
    }

    void setParameter( const std::string &name, int value ) override {
      if ( name == "maximum_iterations" )
        m_maxIterations = value;
      else if ( name == "maximum_infeasible_iterations" )
        m_maxInfeasibleIterations = value;
      else if ( name == "print_level" )
        m_quiet = ( value < 5);
      else
        throw std::runtime_error( "SteihaugCGMethod::setParameter(): Unknown parameter '" + name + "'." );
    }
  };
}
