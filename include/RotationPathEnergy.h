#pragma once

#include <goast/Core.h>
#include <goast/Optimization/LinearOperator.h>
#include <goast/Optimization/Functionals.h>

template<typename ConfiguratorType=DefaultConfigurator>
class RotationHessianOperator : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using VecType = typename ConfiguratorType::VecType;
  using TripletListType = std::vector<TripletType>;

  const int m_numVertices;
  const std::vector<int> m_barycenterIndices;
  const RealType m_factor;
  const VectorType m_other;

public:
  explicit RotationHessianOperator( const int numVertices, std::vector<int> barycenterIndices, RealType factor,
                                    VectorType other )
          : m_numVertices( numVertices ), m_barycenterIndices( std::move( barycenterIndices )), m_factor( factor ),
            m_other( std::move( other )) {}

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( 3 * m_numVertices );
    Dest.setZero();

    VecType p, q, t;
    for ( int i: m_barycenterIndices ) {
      for ( int j: m_barycenterIndices ) {
        getXYZCoord( m_other, p, i );
        getXYZCoord( m_other, q, j );
        getXYZCoord( Arg, t, i );

        VecType localGrad = q.crossProduct( t ).crossProduct( p );

        for ( int d: { 0, 1, 2 } )
          Dest[d * m_numVertices + j] += localGrad[d];
      }
    }

  }

//  void apply( const FullMatrixType &Arg, FullMatrixType &Dest ) const override {
//    Dest = m_A * Arg;
//  }
//
//  void apply( const SparseMatrixType &Arg, SparseMatrixType &Dest ) const override {
//    Dest = m_A * Arg;
//  }

  void applyTransposed( const VectorType &Arg, VectorType &Dest ) const override {
    apply(Arg, Dest);
//    Dest *= -1.;
  }

//  void applyTransposed( const FullMatrixType &Arg, FullMatrixType &Dest ) const override {
//    Dest = m_A.transpose() * Arg;
//  }
//
//  void applyTransposed( const SparseMatrixType &Arg, SparseMatrixType &Dest ) const override {
//    Dest = m_A.transpose() * Arg;
//  }

  int rows() const override {
    return 3 * m_numVertices;
  }

  int cols() const override {
    return 3 * m_numVertices;
  }

//  void assembleTransformationMatrix( SparseMatrixType &T, bool transposed ) const override {
//    TripletListType tripletList;
//    for ( int i = 0; i < m_numVertices; i++ ) {
//      for ( int j = 0; j < m_numVertices; j++ ) {
//        for ( int d: { 0, 1, 2 } ) {
//          tripletList.emplace_back( i * 3 + d, j * 3 + d, m_factor );
//        }
//      }
//    }
//    T.resize( 3 * m_numVertices, 3 * m_numVertices );
//    T.setFromTriplets( tripletList.begin(), tripletList.end());
//  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class ZeroOperator : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using VecType = typename ConfiguratorType::VecType;
  using TripletListType = std::vector<TripletType>;

  const int m_rows, m_cols;


public:
  explicit ZeroOperator( int rows, int cols ) : m_rows( rows ), m_cols( cols ) {}

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( m_rows );
    Dest.setZero();

  }

//  void apply( const FullMatrixType &Arg, FullMatrixType &Dest ) const override {
//    Dest = m_A * Arg;
//  }
//
//  void apply( const SparseMatrixType &Arg, SparseMatrixType &Dest ) const override {
//    Dest = m_A * Arg;
//  }

  void applyTransposed( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize(m_cols);
    Dest.setZero();
//    Dest *= -1.;
  }

//  void applyTransposed( const FullMatrixType &Arg, FullMatrixType &Dest ) const override {
//    Dest = m_A.transpose() * Arg;
//  }
//
//  void applyTransposed( const SparseMatrixType &Arg, SparseMatrixType &Dest ) const override {
//    Dest = m_A.transpose() * Arg;
//  }

  int rows() const override {
    return m_rows;
  }

  int cols() const override {
    return m_cols;
  }

  void assembleTransformationMatrix( SparseMatrixType &T, bool transposed ) const override {
    if ( transposed )
      T.resize( m_cols, m_rows );
    else
      T.resize( m_rows, m_cols );
    T.setZero();
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class RotationPathEnergy : public ObjectiveFunctional<ConfiguratorType> {
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const MeshTopologySaver &m_Topology;
  const int m_numVertices;

  const VectorType &m_start, m_end;
  const int m_K;
  const int m_numOfFreeShapes;
  const int m_numLocalDofs;

  std::vector<int> m_momentIndices;

  mutable VectorType m_oldPoint;
  mutable std::vector<VecType> m_Moments;
public:
  RotationPathEnergy( const MeshTopologySaver &Topology,
                      int K,
                      const VectorType &start,
                      const VectorType &end ) : m_Topology( Topology ),
                                                m_numVertices( Topology.getNumVertices()),
                                                m_start( start ),
                                                m_end( end ),
                                                m_K( K ),
                                                m_numOfFreeShapes( K - 1 ),
                                                m_numLocalDofs( start.size()) {

    m_Moments.resize( m_K);
    m_oldPoint = VectorType::Zero( m_numOfFreeShapes * 3 * m_numVertices );

    m_momentIndices.resize( m_numVertices, 0 );
    std::iota( m_momentIndices.begin(), m_momentIndices.end(), 0 );
  }

  RotationPathEnergy( const MeshTopologySaver &Topology,
                      std::vector<int> barycenterIndices,
                      const int K,
                      const VectorType &start,
                      const VectorType &end ) : m_Topology( Topology ),
                                                m_numVertices( Topology.getNumVertices()),
                                                m_start( start ),
                                                m_end( end ),
                                                m_K( K ),
                                                m_numOfFreeShapes( K - 1 ),
                                                m_numLocalDofs( start.size()),
                                                m_momentIndices( std::move( barycenterIndices )) {
    m_Moments.resize( m_K );
    m_oldPoint = VectorType::Zero( m_numOfFreeShapes * 3 * m_numVertices );
  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::evaluate: wrong number of dofs!" );

    evaluateMoments( Point );

    Value = 0;
    for ( int k = 0; k < m_K; k++ )
      Value += m_Moments[k].squaredNorm();

    Value *= m_K;
  }

  VectorType stepEnergies( const VectorType &Point ) const {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::stepEnergies: wrong number of dofs!" );

    evaluateMoments( Point );
    VectorType output( m_K );
    for ( int k = 0; k < m_K; k++ )
      output[k] = m_Moments[k].squaredNorm();

    return output;
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size() )
      Gradient.resize( Point.size() );

    Gradient.setZero();

    evaluateMoments( Point );

    // compute path energy gradient
    VecType p, q, localGrad;
    for ( int i: m_momentIndices ) {
      getXYZCoord( m_start, p, i );

      localGrad = m_Moments[0].crossProduct( p );

      for ( int d: { 0, 1, 2 } )
        Gradient[d * m_numVertices + i] -= localGrad[d];
    }

    for ( int k = 1; k < m_K - 1; k++ ) {
      for ( int i: m_momentIndices ) {
        getXYZCoord( Point.segment( ( k - 1 ) * m_numLocalDofs, m_numLocalDofs ), p, i );
        getXYZCoord( Point.segment( k * m_numLocalDofs, m_numLocalDofs ), q, i );

        localGrad = m_Moments[k].crossProduct( q );
        for ( int d: { 0, 1, 2 } )
          Gradient[( k - 1 ) * m_numLocalDofs + d * m_numVertices + i] += localGrad[d];

        localGrad = p.crossProduct( m_Moments[k] );
        for ( int d: { 0, 1, 2 } )
          Gradient[k * m_numLocalDofs + d * m_numVertices + i] += localGrad[d];
      }
    }

    for ( int i: m_momentIndices ) {
      getXYZCoord( m_end, q, i );

      localGrad = q.crossProduct( m_Moments[m_K - 1] );
      for ( int d: { 0, 1, 2 } )
        Gradient[( m_numOfFreeShapes - 1 ) * m_numLocalDofs + d * m_numVertices + i] -= localGrad[d];
    }

    Gradient *= 2. * m_K / m_momentIndices.size();
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> HessOp( const VectorType &Point ) const override {
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( m_numOfFreeShapes ); // Main diagonal
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> B( m_numOfFreeShapes );

    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      if (k == 0) {
        A[k] = std::make_unique<RotationHessianOperator<ConfiguratorType>>( m_numVertices, m_momentIndices,
                                                                            4. * m_K / m_momentIndices.size() / m_momentIndices.size(), m_start );
      }

      if (k == m_numOfFreeShapes - 1) {
        A[k] = std::make_unique<RotationHessianOperator<ConfiguratorType>>( m_numVertices, m_momentIndices,
                                                                            -4. * m_K / m_momentIndices.size() / m_momentIndices.size(), m_end );
      }

      if (k != 0 && k != m_numOfFreeShapes - 1) {
        auto leftOperator = std::make_unique<RotationHessianOperator<ConfiguratorType>>( m_numVertices, m_momentIndices,
                                                                                         4. * m_K / m_momentIndices.size() / m_momentIndices.size(),
                                                                                         Point.segment(( k - 1 ) * m_numLocalDofs, m_numLocalDofs ) );
        auto rightOperator = std::make_unique<RotationHessianOperator<ConfiguratorType>>( m_numVertices,
                                                                                          m_momentIndices,
                                                                                          -4. * m_K / m_momentIndices.size() / m_momentIndices.size(),
                                                                                          Point.segment(( k + 1 ) * m_numLocalDofs, m_numLocalDofs ) );

        A[k] = std::make_unique<SummedOperator<ConfiguratorType>>( std::move( leftOperator ),
                                                                   std::move( rightOperator ));
      }

      if ( k < m_numOfFreeShapes - 1 ) {
        B[k] = std::make_unique<ZeroOperator<ConfiguratorType>>( 3 * m_numVertices, 3 * m_numVertices );
      }
    }

    return std::make_unique<SymmetricTriadiagonalBlockOperator<ConfiguratorType>>( std::move( A ), std::move( B ));
  }

protected:
  void evaluateMoments( const VectorType &Point ) const {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::evaluateMoments: wrong number of dofs!" );

    if (( m_oldPoint - Point ).norm() < 1.e-8 )
      return;

#pragma omp parallel for
    for ( int k = 0; k < m_K; k++ ) {
      m_Moments[k].setZero();

      VecType p, q;
      for ( int i : m_momentIndices ) {
        if ( k == 0 )
          getXYZCoord( m_start, p, i );
        else
          getXYZCoord( Point.segment( (k-1) * m_numLocalDofs, m_numLocalDofs ), p, i );

        if ( k == m_K -1 )
          getXYZCoord( m_end, q, i );
        else
          getXYZCoord( Point.segment(k * m_numLocalDofs, m_numLocalDofs), q, i );

        m_Moments[k] += q.crossProduct(p);
      }

      m_Moments[k] /= m_momentIndices.size();
    }
  }

};