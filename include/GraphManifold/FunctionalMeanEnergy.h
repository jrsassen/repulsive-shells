#pragma once

#include <goast/Core.h>

#include <goast/Optimization/LinearOperator.h>
#include <goast/Optimization/Functionals.h>

template<typename ConfiguratorType=DefaultConfigurator>
class FunctionalMeanEnergy : public ObjectiveFunctional<ConfiguratorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const ObjectiveFunctional<ConfiguratorType> &m_F;

  const std::vector<VectorType> &m_Shapes;
  const VectorType &m_Weights;
  VectorType m_Values;
  int m_numShapes;

  const int m_numLocalDofs;

  // F cache
  mutable RealType m_Value;
  mutable VectorType m_Gradient;

public:
  FunctionalMeanEnergy( const ObjectiveFunctional<ConfiguratorType> &F,
                        const std::vector<VectorType> &Shapes,
                        const VectorType &Weights ) : m_F( F ),
                                                      m_Shapes( Shapes ),
                                                      m_Weights( Weights ),
                                                      m_numShapes( Shapes.size() ),
                                                      m_numLocalDofs( Shapes[0].size() ) {
    assert( Weights.size() == m_numShapes && "FunctionalMeanEnergy: wrong number of weights" );

    m_Gradient = VectorType::Zero( m_numLocalDofs );
    m_Values = VectorType::Zero( m_numShapes );

    for ( int k = 0; k < m_numShapes; k++ )
      m_Values[k] = F( m_Shapes[k] );

    std::cout << " .. FME::m_values = " << m_Values.transpose() << std::endl;
  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "FunctionalMeanEnergy::evaluate: wrong number of dofs!" );

    m_Value = m_F( Point );

    Value = ( m_Values.array() - m_Value ).square().matrix().dot( m_Weights );
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "FunctionalMeanEnergy::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size())
      Gradient.resize( Point.size());

    Gradient.setZero();

    m_Value = m_F( Point );
    m_Gradient = m_F.grad( Point );

    // compute energy gradient
    Gradient = -2. * ( m_Values.array() - m_Value ).matrix().dot( m_Weights ) * m_Gradient;
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> HessOp( const VectorType &Point ) const override {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "FunctionalMeanEnergy::HessOp: wrong number of dofs!" );

    m_Value = m_F( Point );
    m_Gradient = m_F.grad( Point );

    auto gradProduct = std::make_unique<RankOneOperator<ConfiguratorType>>( 2 * m_Weights.sum() * m_Gradient,
                                                                            m_Gradient );

    //! \todo Add Hessian if available

    return gradProduct;
  }

};

