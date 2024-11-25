#pragma once

#include <goast/Optimization/Functionals.h>
#include <goast/Optimization/LinearOperator.h>

#include "Optimization/Preconditioners.h"

template<typename ConfiguratorType=DefaultConfigurator>
class L2PathMetric : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const BaseOp<VectorType, MatrixType> &m_Metric;
  int m_K;

public:
  L2PathMetric( const BaseOp<VectorType, MatrixType> &Metric, int K )
          : m_Metric( Metric ), m_K( K ) {}

  // Arg =  (s_1, ..., s_{K-1}) are intermediate shapes only, where s_k = (x_k, y_k, z_k)
  void apply( const VectorType &Arg, MatrixType &Dest ) const override {
    int numOfFreeShapes = m_K - 1;

    if ( Arg.size() % numOfFreeShapes != 0 )
      throw std::length_error( "H1PathMetric::apply: wrong number of dofs!" );

    if (( Dest.cols() != Arg.size()) || ( Dest.rows() != Arg.size()))
      Dest.resize( Arg.size(), Arg.size());

    // fill triplet lists
    TripletListType tripletList;
    pushTriplets( Arg, tripletList );

    Dest.setFromTriplets( tripletList.cbegin(), tripletList.cend());
  }

  // fill triplets
  // Arg =  (s_1, ..., s_{K-1}) are intermediate shapes only, where s_k = (x_k, y_k, z_k)
  void pushTriplets( const VectorType &Arg, TripletListType &tripletList ) const override {
    int numOfFreeShapes = m_K - 1;
    if ( Arg.size() % numOfFreeShapes != 0 )
      throw std::length_error( "H1PathMetric::pushTriplets: wrong number of dofs!" );

    // bring into more convenient form
    std::vector<Eigen::Ref<const VectorType> > argRefs;
    argRefs.reserve( numOfFreeShapes );

    const int numLocalDofs = Arg.size() / numOfFreeShapes;
    for ( int k = 0; k < numOfFreeShapes; k++ )
      argRefs.push_back( Arg.segment( k * numLocalDofs, numLocalDofs ));

    // compute path energy Hessian on diagonal
//#pragma omp parallel for default(none) shared(numOfFreeShapes, argRefs, numLocalDofs, tripletList)
    for ( int k = 0; k < numOfFreeShapes; k++ ) {
      m_Metric.pushTriplets( argRefs[k], tripletList, 1., k * numLocalDofs, k * numLocalDofs ); // m_K *
    }
  }

};


template<typename ConfiguratorType=DefaultConfigurator>
class OperatorL2PathMetric : public MapToLinOp<ConfiguratorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const MapToLinOp<ConfiguratorType> &m_Metric;
  int m_K;

public:
  OperatorL2PathMetric( const MapToLinOp<ConfiguratorType> &Metric, int K )
          : m_Metric( Metric ), m_K( K ) {
#ifndef NDEBUG
    std::cout << "Warning: OperatorL2PathMetric lacks scaling!" << std::endl;
#endif
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    const int numLocalDofs = Point.size() / ( m_K - 1 );

    // Build block operators

    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( m_K - 1 ); // Main diagonal

    for ( int k = 0; k < m_K - 1; k++ ) {
      A[k] = m_Metric( Point.segment( k * numLocalDofs, numLocalDofs ));
    }

    return std::make_unique<BlockDiagonalOperator<ConfiguratorType>>( std::move( A ) );
  }
};


template<typename ConfiguratorType=DefaultConfigurator>
class InverseOperatorL2PathMetric : public MapToLinOp<ConfiguratorType>,
                                    public TimedClass<InverseOperatorL2PathMetric<ConfiguratorType>> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  using typename TimedClass<InverseOperatorL2PathMetric<ConfiguratorType>>::ScopeTimer;

  const MapToLinOp<ConfiguratorType> &m_Metric;
  const std::vector<int> &m_fixedVariables;
  int m_K;
  const RealType m_scaleFactor;

public:
  InverseOperatorL2PathMetric( const MapToLinOp<ConfiguratorType> &Metric, int K,
                               const std::vector<int> &fixedVariables,
                               RealType scaleFactor = 1. )
          : m_Metric( Metric ), m_K( K ), m_fixedVariables( fixedVariables ), m_scaleFactor(scaleFactor) {}

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    const int numLocalDofs = Point.size() / ( m_K - 1 );

    // Build block operators
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( m_K - 1 ); // Main diagonal

#pragma omp parallel for
    for ( int k = 0; k < m_K - 1; k++ ) {
      MatrixType M;
      {
        ScopeTimer timer("AssembleBlocks");
        auto localMetric = m_Metric( Point.segment( k * numLocalDofs, numLocalDofs ));
        localMetric->assembleTransformationMatrix( M );
        M *= m_K * m_scaleFactor;
        applyMaskToSymmetricMatrix( m_fixedVariables, M );
      }

      {
        ScopeTimer timer("FactorizeBlocks");
        A[k] = std::make_unique<CholeskyPreconditioner<ConfiguratorType >>( M );
      }
    }

    return std::make_unique<BlockDiagonalOperator<ConfiguratorType>>( std::move( A ) );
  }
};