#pragma once

#include <utility>

#include <goast/Core.h>
#include <goast/Optimization/LinearOperator.h>


template<typename ConfiguratorType=DefaultConfigurator>
class OperatorPathEnergyHessian : public MapToLinOp<ConfiguratorType>,
                                  public TimedClass<OperatorPathEnergyHessian<ConfiguratorType>> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  using typename TimedClass<OperatorPathEnergyHessian<ConfiguratorType>>::ScopeTimer;

  const DeformationBase<ConfiguratorType> &m_W;
  const VectorType &m_start, m_end;
  const int m_K;
  const int m_numDOF;
  std::vector<int> m_mask;
  std::vector<int> m_localMask;

public:
  OperatorPathEnergyHessian( const DeformationBase<ConfiguratorType> &W, int K, const VectorType &start,
                             const VectorType &end, std::vector<int> localMask ) : m_W( W ), m_start( start ),
                                                                                   m_end( end ), m_K( K ),
                                                                                   m_localMask( std::move( localMask )),
                                                                                   m_numDOF( start.size()) {
    fillPathMask( m_K - 1, m_start.size(), m_localMask, m_mask );

//    m_localSolver.cholmod().print = 0;
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const {
    ScopeTimer timer("evaluate");

    int numOfFreeShapes = m_K - 1;

    if ( Point.size() % numOfFreeShapes != 0 )
      throw std::length_error( "DiscretePathEnergyHessian::apply: wrong number of dofs!" );

    if ( Point.size() / numOfFreeShapes != m_start.size())
      throw std::length_error( "DiscretePathEnergyHessian::apply: wrong size of dofs!" );

    std::vector<Eigen::Ref<const VectorType> > argRefs;
    argRefs.reserve( numOfFreeShapes );

    for ( int k = 0; k < numOfFreeShapes; k++ )
      argRefs.push_back( Point.segment( k * m_numDOF, m_numDOF ));

    // Build block operators
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( numOfFreeShapes ); // Main diagonal
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> B( numOfFreeShapes ); // Offdiagonal

//    std::cerr << " -- Assembling matrices... " << std::endl;

#pragma omp parallel for
    for ( int k = 0; k < numOfFreeShapes; k++ ) {
      TripletListType tripletListA, tripletListB;

      if ( k == 0 )
        m_W.pushTripletsDefHessian( m_start, argRefs[k], tripletListA, 0, 0, m_K );
      else
        m_W.pushTripletsDefHessian( argRefs[k - 1], argRefs[k], tripletListA, 0, 0, m_K );

      if ( k == numOfFreeShapes - 1 )
        m_W.pushTripletsUndefHessian( argRefs[k], m_end, tripletListA, 0, 0, m_K );
      else
        m_W.pushTripletsUndefHessian( argRefs[k], argRefs[k + 1], tripletListA, 0, 0, m_K );

      MatrixType localA;

      localA.resize( m_numDOF, m_numDOF );
      localA.setZero();
      localA.reserve(tripletListA.size());
      localA.setFromTriplets( tripletListA.begin(), tripletListA.end());

      A[k] = std::make_unique<MatrixOperator<ConfiguratorType>>( std::move( localA ));

//      applyMaskToSymmetricMatrix<MatrixType>( m_localMask, B[k] );

      if ( k < numOfFreeShapes - 1 ) {
        m_W.pushTripletsMixedHessian( argRefs[k], argRefs[k + 1], tripletListB, 0, 0, false, m_K );

        MatrixType localB;
        localB.resize( m_numDOF, m_numDOF );
        localB.setZero();
        localB.reserve(tripletListB.size());
        localB.setFromTriplets( tripletListB.begin(), tripletListB.end());
//        applyMaskToMatrix<MatrixType>( m_localMask, B[k], false );

        B[k] = std::make_unique<MatrixOperator<ConfiguratorType>>( std::move( localB ));
      }
    }

    return std::make_unique<SymmetricTriadiagonalBlockOperator<ConfiguratorType>>( std::move( A ), std::move( B ));
  }
};