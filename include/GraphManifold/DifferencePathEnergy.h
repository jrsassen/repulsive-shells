#pragma once

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>


template<typename ConfiguratorType, template<typename> typename EnergyType>
class DifferencePathEnergy : public ObjectiveFunctional<ConfiguratorType>,
                             public TimedClass<DifferencePathEnergy<ConfiguratorType, EnergyType>> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  using typename TimedClass<DifferencePathEnergy<ConfiguratorType, EnergyType>>::ScopeTimer;

  std::vector<std::unique_ptr<EnergyType<ConfiguratorType>>> m_E;

  int m_K;
  int m_numOfFreeShapes;

  RealType m_startValue, m_endValue;

  const int m_numLocalDofs;

  // TPE cache
  mutable VectorType m_oldPoint;
  mutable VectorType m_Values;
  mutable std::vector<VectorType> m_Gradients;

public:
  template<class... Ts>
  DifferencePathEnergy( const int K,
                        const VectorType &start,
                        const VectorType &end,
                        Ts&&... args ) : m_K( K ),
                                       m_numOfFreeShapes( K - 1 ),
                                       m_numLocalDofs( start.size() ) {
    assert( start.size() == end.size() && "DifferencePathEnergy: Match of start and end does not match" );
    m_Values.resize( m_numOfFreeShapes );
    m_Gradients.resize( m_numOfFreeShapes );
    m_oldPoint = VectorType::Zero( m_numOfFreeShapes * start.size() );

    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      m_Gradients[k].resize( start.size() );
      m_Gradients[k].setZero();

      m_E.emplace_back( std::make_unique<EnergyType<ConfiguratorType>>( std::forward<Ts>(args)... ) );
    }


    // Compute start and end values via normal energy
    m_startValue = ( *m_E[0] )( start );
    m_endValue = ( *m_E[0] )( end );
  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    ScopeTimer timer( "evaluate" );

    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluate: wrong number of dofs!" );

    evaluateEnergiesAndGradients( Point );

    Value = ( m_Values[0] - m_startValue ) * ( m_Values[0] - m_startValue );
    for ( int k = 1; k < m_numOfFreeShapes; k++ )
      Value += ( m_Values[k] - m_Values[k - 1] ) * ( m_Values[k] - m_Values[k - 1] );
    Value += ( m_endValue - m_Values[m_numOfFreeShapes - 1] ) * ( m_endValue - m_Values[m_numOfFreeShapes - 1] );

    Value *= m_K;
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    ScopeTimer timer( "evaluateGradient" );

    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size())
      Gradient.resize( Point.size());

    Gradient.setZero();

    evaluateEnergiesAndGradients( Point );

    // compute path energy gradient
    Gradient.head( m_numLocalDofs ) += 2. * ( m_Values[0] - m_startValue ) * m_Gradients[0];

    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      if ( k > 0 )
        Gradient.segment( k * m_numLocalDofs, m_numLocalDofs ) +=
            2. * ( m_Values[k] - m_Values[k - 1] ) * m_Gradients[k];
      if ( k < m_numOfFreeShapes - 1 )
        Gradient.segment( k * m_numLocalDofs, m_numLocalDofs ) -=
            2. * ( m_Values[k + 1] - m_Values[k] ) * m_Gradients[k];
    }

    Gradient.tail( m_numLocalDofs ) -=
        2. * ( m_endValue - m_Values[m_numOfFreeShapes - 1] ) * m_Gradients[m_numOfFreeShapes - 1];

    Gradient *= m_K;
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> HessOp( const VectorType &Point ) const override {
    ScopeTimer timer( "HessOp" );

    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "TangentPointDifferenceEnergy::HessOp: wrong number of dofs!" );

    evaluateEnergiesAndGradients( Point );

    // Build block operators
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( m_numOfFreeShapes ); // Main diagonal
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> B( m_numOfFreeShapes ); // Offdiagonal

#pragma omp parallel for
    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      A[k] = std::make_unique<RankOneOperator<ConfiguratorType>>( 4 * m_K * m_Gradients[k], m_Gradients[k] );

      if ( k < m_numOfFreeShapes - 1 ) {
        B[k] = std::make_unique<RankOneOperator<ConfiguratorType> >( -2. * m_K * m_Gradients[k], m_Gradients[k + 1] );
      }
    }

    return std::make_unique<SymmetricTriadiagonalBlockOperator<ConfiguratorType>>( std::move( A ), std::move( B ));
  }

  VectorType stepEnergies(const VectorType &Point) const {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::stepEnergies: wrong number of dofs!" );

    evaluateEnergiesAndGradients( Point );
    VectorType output( m_numOfFreeShapes + 1 );

    output[0] = ( m_Values[0] - m_startValue ) * ( m_Values[0] - m_startValue );
    for ( int k = 1; k < m_numOfFreeShapes; k++ )
      output[k] = ( m_Values[k] - m_Values[k - 1] ) * ( m_Values[k] - m_Values[k - 1] );
    output[m_numOfFreeShapes] = ( m_endValue - m_Values[m_numOfFreeShapes - 1] ) *
                                ( m_endValue - m_Values[m_numOfFreeShapes - 1] );

    return output;
  }

  void resetCache() const {
    m_oldPoint.setZero();
    // for ( int k = 0; k < m_numOfFreeShapes; k++ )
    //   m_TPE[k]->resetCache();
  }
  
protected:
  void evaluateEnergiesAndGradients(const VectorType &Point) const {
    ScopeTimer timer( "evaluateEnergiesAndGradients" );

    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateTPE: wrong number of dofs!" );

    if ( ( m_oldPoint - Point ).norm() < 1.e-8 )
      return;
#pragma omp parallel for
    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      m_E[k]->evaluateGradient( Point.segment( k * m_numLocalDofs, m_numLocalDofs ), m_Gradients[k] );
      m_E[k]->evaluate( Point.segment( k * m_numLocalDofs, m_numLocalDofs ), m_Values[k] );
    }

    m_oldPoint = Point;
  }
};
