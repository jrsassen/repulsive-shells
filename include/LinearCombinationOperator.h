#pragma once

#include <goast/Core.h>
#include <goast/Optimization/LinearOperator.h>

template<typename ConfiguratorType>
class LinearCombinationOperator final : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> m_A;

  VectorType m_weights;

public:
  LinearCombinationOperator( VectorType weights,
                             std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> &&A )
          : m_weights(std::move( weights )) {
    m_A.swap( A );
  }

//  SummedOperator( VectorType &&a, VectorType &&b ) : m_a( a ), m_b( b ) {  }

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( this->rows());
    Dest.setZero();

//#pragma omp parallel
    for ( int i = 0; i < m_A.size(); i++ )
      Dest += m_weights[i] * ( *m_A[i] )( Arg );
  }

  void applyTransposed( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( this->cols());
    Dest.setZero();

//#pragma omp parallel
    for ( int i = 0; i < m_A.size(); i++ )
      Dest += m_weights[i] * ( *m_A[i] ).T( Arg );
  }

  int rows() const override {
    return m_A[0]->rows();
  }

  int cols() const override {
    return m_A[0]->cols();
  }

  void assembleTransformationMatrix( SparseMatrixType &T, bool transposed ) const override {
    T.resize( m_A[0]->rows(),  m_A[0]->cols());
    T.setZero();

//#pragma omp parallel
    for ( int i = 0; i < m_A.size(); i++ ) {
      SparseMatrixType mat;
      m_A[i]->assembleTransformationMatrix( mat, transposed );
      T += m_weights[i] * mat;
    }
  }

  void assembleTransformationMatrix( FullMatrixType &T, bool transposed ) const override {
    T.resize( m_A[0]->rows(),  m_A[0]->cols());
    T.setZero();

//#pragma omp parallel
    for ( int i = 0; i < m_A.size(); i++ ) {
      FullMatrixType mat;
      m_A[i]->assembleTransformationMatrix( mat, transposed );
      T += m_weights[i] * mat;
    }
  }
};