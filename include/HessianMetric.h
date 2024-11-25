#pragma once

#include <goast/Core.h>

template<typename ConfiguratorType=DefaultConfigurator>
class HessianMetric
    : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const DeformationBase<ConfiguratorType> &m_W;
  const std::vector<int> &m_boundaryMask;

public:
  explicit HessianMetric( const DeformationBase<ConfiguratorType> &W, const std::vector<int> &boundaryMask ) : m_W( W ),
                                                                                                               m_boundaryMask(
                                                                                                                       boundaryMask ) {}

  void apply( const VectorType &Arg, MatrixType &Dest ) const override {
    Dest.resize( Arg.size(), Arg.size());
    Dest.setZero();

    m_W.applyDefHessian( Arg, Arg, Dest );
//    applyMaskToSymmetricMatrix(m_boundaryMask, Dest);
  }

  void pushTriplets( const VectorType &Arg, TripletListType &Dest ) const override {
    m_W.pushTripletsDefHessian( Arg, Arg, Dest, 0, 0 );
//    for ( auto &triplet : Dest ) {
//      if ( std::find( m_boundaryMask.begin(), m_boundaryMask.end(), triplet.row()) != m_boundaryMask.end() ||
//           std::find( m_boundaryMask.begin(), m_boundaryMask.end(), triplet.col()) != m_boundaryMask.end())
//        triplet = TripletType(triplet.row(), triplet.col(), 0.);
//    }
//    for ( int idx : m_boundaryMask )
//      Dest.emplace_back( idx, idx, 1. );
  }

  void pushTriplets( const VectorType &Arg, TripletListType &Dest, RealType factor, int rowOffset,
                     int colOffset ) const override {
    m_W.pushTripletsDefHessian( Arg, Arg, Dest, rowOffset, colOffset, 1. );
  }

};

template<typename ConfiguratorType=DefaultConfigurator>
class OperatorHessianMetric : public MapToLinOp<ConfiguratorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const DeformationBase<ConfiguratorType> &m_W;
  const std::vector<int> &m_boundaryMask;
  const RealType m_scaleFactor;

public:
  explicit OperatorHessianMetric( const DeformationBase<ConfiguratorType> &W,
                                  const std::vector<int> &boundaryMask,
                                  RealType scaleFactor = 1. ) : m_W( W ), m_boundaryMask( boundaryMask ),
                                                                m_scaleFactor( scaleFactor ) {}

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    MatrixType localMat;
    localMat.resize( Point.size(), Point.size());
    localMat.setZero();

    m_W.applyDefHessian( Point, Point, localMat, m_scaleFactor );

    return std::make_unique<MatrixOperator<ConfiguratorType>>( std::move( localMat ));
  }
};