#pragma once

#include <goast/Optimization/LinearOperator.h>

template<typename ConfiguratorType=DefaultConfigurator>
class MatrixOperatorMapWrapper : public MapToLinOp<ConfiguratorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const BaseOp<VectorType, MatrixType> &m_F;

public:
  explicit MatrixOperatorMapWrapper( const BaseOp<VectorType, MatrixType> &F ) : m_F( F ) {}

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    MatrixType localMat;

    m_F.apply( Point, localMat );

    return std::make_unique<MatrixOperator<ConfiguratorType>>( std::move( localMat ) );
  }
};
