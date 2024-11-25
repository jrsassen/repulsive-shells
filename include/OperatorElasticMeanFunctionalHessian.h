#pragma once

#include <LinearCombinationOperator.h>
#include <goast/Core.h>
#include <goast/Optimization/LinearOperator.h>


template<typename ConfiguratorType>
class OperatorElasticMeanFunctionalHessian final : public MapToLinOp<ConfiguratorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const DeformationBase<ConfiguratorType> &m_W;
  const VectorType &m_shapes, m_alpha;
  const unsigned int m_numOfShapes;
  std::vector<Eigen::Ref<const VectorType> > m_argRefs;

public:
  OperatorElasticMeanFunctionalHessian( const DeformationBase<ConfiguratorType> &W, const VectorType &shapes,
                                        const VectorType alpha, const unsigned int numOfShapes )
          : m_W( W ), m_shapes( shapes ), m_alpha( alpha ), m_numOfShapes( numOfShapes ) {
    if ( shapes.size() % numOfShapes != 0 )
      throw std::length_error( "ElasticMeanFunctional: wrong number of dof!" );
    if ( alpha.size() % numOfShapes != 0 )
      throw std::length_error( "ElasticMeanFunctional: wrong number of alphas!" );

    // Create references for this different shapes bec. of convenience
    m_argRefs.reserve( numOfShapes );

    const int numLocalDofs = m_shapes.size() / m_numOfShapes;
    for ( int k = 0; k < numOfShapes; k++ )
      m_argRefs.push_back( m_shapes.segment( k * numLocalDofs, numLocalDofs ));
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    if ( Point.size() != m_argRefs[0].size())
      throw std::length_error( "OperatorElasticMeanFunctionalHessian::apply: wrong size of dofs!" );

    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( m_numOfShapes );

#pragma omp parallel for
    for ( int k = 0; k < m_numOfShapes; k++ ) {
      TripletListType localTripletList;

      m_W.pushTripletsDefHessian( m_argRefs[k], Point, localTripletList, 0, 0, m_alpha[k] );

      MatrixType localA( m_argRefs[0].size(), m_argRefs[0].size() );
      localA.setFromTriplets( localTripletList.begin(), localTripletList.end());

      A[k] = std::make_unique<MatrixOperator<ConfiguratorType>>( std::move( localA ));
    }

    return std::make_unique<LinearCombinationOperator<ConfiguratorType>>( m_alpha, std::move( A ));
  }

};
