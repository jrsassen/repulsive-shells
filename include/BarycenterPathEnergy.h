#pragma once

#include <goast/Core.h>
#include <goast/Optimization/LinearOperator.h>
#include <goast/Optimization/Functionals.h>

template<typename ConfiguratorType=DefaultConfigurator>
class BarycenterHessianOperator : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const int m_numVertices;
  const std::vector<int> m_barycenterIndices;
  const RealType m_factor;

public:
  explicit BarycenterHessianOperator( const int numVertices, std::vector<int> barycenterIndices, RealType factor )
          : m_numVertices( numVertices ), m_barycenterIndices( std::move( barycenterIndices )), m_factor( factor ) {}

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( 3 * m_numVertices );

    for ( int d: { 0, 1, 2 } ) {
      RealType value = 0;
      for ( int i : m_barycenterIndices ) {
        value += Arg[d * m_numVertices + i];
      }
      value *= m_factor;
      for ( int j : m_barycenterIndices ) {
        Dest[d * m_numVertices + j] = value;
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

  void assembleTransformationMatrix( SparseMatrixType &T, bool transposed ) const override {
    TripletListType tripletList;
    for ( int i = 0; i < m_numVertices; i++ ) {
      for ( int j = 0; j < m_numVertices; j++ ) {
        for ( int d: { 0, 1, 2 } ) {
          tripletList.emplace_back( i + d * m_numVertices, j + d * m_numVertices, m_factor );
        }
      }
    }
    T.resize( 3 * m_numVertices, 3 * m_numVertices );
    T.setFromTriplets( tripletList.begin(), tripletList.end());
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class BarycenterPathEnergy : public ObjectiveFunctional<ConfiguratorType> {
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

  VecType m_startBarycenter, m_endBarycenter;

  std::vector<int> m_barycenterIndices;


  mutable VectorType m_oldPoint;
  mutable std::vector<VecType> m_Barycenters;
public:
  BarycenterPathEnergy( const MeshTopologySaver &Topology,
                        int K,
                        const VectorType &start,
                        const VectorType &end ) : m_Topology( Topology ),
                                                  m_numVertices( Topology.getNumVertices()),
                                                  m_start( start ),
                                                  m_end( end ),
                                                  m_K( K ),
                                                  m_numOfFreeShapes( K - 1 ),
                                                  m_numLocalDofs( start.size()) {

    m_Barycenters.resize( m_numOfFreeShapes);
    m_oldPoint = VectorType::Zero( m_numOfFreeShapes * 3 * m_numVertices );

    m_barycenterIndices.resize( m_numVertices, 0);
    std::iota( m_barycenterIndices.begin(), m_barycenterIndices.end(), 0);

    VecType p;
    for ( int i : m_barycenterIndices ) {
      getXYZCoord( start, p, i );
      m_startBarycenter += p;
      getXYZCoord( end, p, i );
      m_endBarycenter += p;
    }
    m_startBarycenter /= m_barycenterIndices.size();
    m_endBarycenter /= m_barycenterIndices.size();

  }

  BarycenterPathEnergy( const MeshTopologySaver &Topology,
                        std::vector<int> barycenterIndices,
                        int K,
                        const VectorType &start,
                        const VectorType &end ) : m_Topology( Topology ),
                                                  m_numVertices( Topology.getNumVertices()),
                                                  m_barycenterIndices(std::move(barycenterIndices)),
                                                  m_start( start ),
                                                  m_end( end ),
                                                  m_K( K ),
                                                  m_numOfFreeShapes( K - 1 ),
                                                  m_numLocalDofs( start.size()) {

    m_Barycenters.resize( m_numOfFreeShapes);
    m_oldPoint = VectorType::Zero( m_numOfFreeShapes * 3 * m_numVertices );

    VecType p;
    for ( int i : m_barycenterIndices ) {
      getXYZCoord( start, p, i );
      m_startBarycenter += p;
      getXYZCoord( end, p, i );
      m_endBarycenter += p;
    }
    m_startBarycenter /= m_barycenterIndices.size();
    m_endBarycenter /= m_barycenterIndices.size();

  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::evaluate: wrong number of dofs!" );

    evaluateBarycenters( Point );

    Value = ( m_Barycenters[0] - m_startBarycenter ).squaredNorm();
    for ( int k = 1; k < m_numOfFreeShapes; k++ )
      Value += ( m_Barycenters[k] - m_Barycenters[k - 1] ).squaredNorm();
    Value += ( m_endBarycenter - m_Barycenters[m_numOfFreeShapes - 1] ).squaredNorm();

    Value *= m_K;
  }

  VectorType stepEnergies(const VectorType &Point) const {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::stepEnergies: wrong number of dofs!" );

    evaluateBarycenters( Point );
    VectorType output(m_numOfFreeShapes+1);
    output[0] = ( m_Barycenters[0] - m_startBarycenter ).squaredNorm();
    for ( int k = 1; k < m_numOfFreeShapes; k++ )
      output[k] = ( m_Barycenters[k] - m_Barycenters[k - 1] ).squaredNorm();
    output[m_numOfFreeShapes] = ( m_endBarycenter - m_Barycenters[m_numOfFreeShapes - 1] ).squaredNorm();

    return output;
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size())
      Gradient.resize( Point.size());

    Gradient.setZero();

    evaluateBarycenters( Point );

    // compute path energy gradient
    VecType diff = m_Barycenters[0] - m_startBarycenter;
    for ( int i : m_barycenterIndices )
      for (int d : {0,1,2})
        Gradient[d * m_numVertices + i] += diff[d];

    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      if ( k > 0 ) {
        diff = m_Barycenters[k] - m_Barycenters[k - 1];
        for ( int i : m_barycenterIndices )
          for (int d : {0,1,2})
            Gradient[k * m_numLocalDofs + d * m_numVertices + i] += diff[d];
      }

      if ( k < m_numOfFreeShapes - 1 ) {
        diff = m_Barycenters[k + 1] - m_Barycenters[k];
        for ( int i : m_barycenterIndices )
          for (int d : {0,1,2})
            Gradient[k * m_numLocalDofs + d * m_numVertices + i] -= diff[d];
      }
    }

    diff = m_endBarycenter - m_Barycenters[m_numOfFreeShapes - 1];
    for ( int i : m_barycenterIndices )
      for (int d : {0,1,2})
        Gradient[(m_numOfFreeShapes -1) * m_numLocalDofs + d * m_numVertices + i] -= diff[d];

    Gradient *= 2. * m_K / m_barycenterIndices.size();
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> HessOp( const VectorType &Point ) const override {
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> A( m_numOfFreeShapes ); // Main diagonal
    std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> B( m_numOfFreeShapes );

    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      A[k] = std::make_unique<BarycenterHessianOperator<ConfiguratorType> >( m_numVertices, m_barycenterIndices,
                                                                             4. * m_K / m_barycenterIndices.size() / m_barycenterIndices.size() );

      if ( k < m_numOfFreeShapes - 1 ) {
        B[k] = std::make_unique<BarycenterHessianOperator<ConfiguratorType> >( m_numVertices, m_barycenterIndices,
                                                                               -2. * m_K / m_barycenterIndices.size() /
                                                                               m_barycenterIndices.size());
      }
    }

    return std::make_unique<SymmetricTriadiagonalBlockOperator<ConfiguratorType>>( std::move( A ), std::move( B ));
  }

protected:
  void evaluateBarycenters( const VectorType &Point ) const {
    if ( Point.size() != m_numLocalDofs * m_numOfFreeShapes )
      throw std::length_error( "BarycenterPathEnergy::evaluateMoments: wrong number of dofs!" );

    if (( m_oldPoint - Point ).norm() < 1.e-8 )
      return;

#pragma omp parallel for
    for ( int k = 0; k < m_numOfFreeShapes; k++ ) {
      m_Barycenters[k].setZero();

      VecType p;
      for ( int i : m_barycenterIndices ) {
        getXYZCoord( Point.segment(k * m_numLocalDofs, m_numLocalDofs), p, i );
        m_Barycenters[k] += p;
      }

      m_Barycenters[k] /= m_barycenterIndices.size();
    }
  }

};

template<typename ConfiguratorType=DefaultConfigurator>
class BarycenterPenalty : public ObjectiveFunctional<ConfiguratorType> {
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const MeshTopologySaver &m_Topology;
  const int m_numVertices;

  const VecType &m_target;

  std::vector<int> m_barycenterIndices;
public:
  BarycenterPenalty( const MeshTopologySaver &Topology,

                     const VecType &target ) : m_Topology( Topology ),
                                               m_numVertices( Topology.getNumVertices()),
                                               m_target( target ) {
    m_barycenterIndices.resize( m_numVertices, 0 );
    std::iota( m_barycenterIndices.begin(), m_barycenterIndices.end(), 0 );

  }

  BarycenterPenalty( const MeshTopologySaver &Topology,
                     std::vector<int> barycenterIndices,

                     const VecType &target ) : m_Topology( Topology ),
                                               m_numVertices( Topology.getNumVertices()),
                                               m_barycenterIndices( std::move( barycenterIndices )),
                                               m_target( target ) {
  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    assert (( Point.size() != 3 * m_Topology.getNumVertices()) &&
            "BarycenterPenalty::evaluate: wrong number of dofs!" );

    VecType bary = evaluateBarycenter( Point );

    Value = ( bary - m_target ).squaredNorm() / 2.;
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    assert (( Point.size() != 3 * m_Topology.getNumVertices()) &&
            "BarycenterPenalty::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size())
      Gradient.resize( Point.size());

    Gradient.setZero();

    VecType bary = evaluateBarycenter( Point );

    // compute path energy gradient
    VecType diff = bary - m_target;
    for ( int i : m_barycenterIndices )
      for (int d : {0,1,2})
        Gradient[d * m_numVertices + i] += diff[d];

    Gradient *= 1. / m_barycenterIndices.size();
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> HessOp( const VectorType &Point ) const override {
    assert (( Point.size() != 3 * m_Topology.getNumVertices()) &&
            "BarycenterPenalty::HessOp: wrong number of dofs!" );

    return std::make_unique<BarycenterHessianOperator<ConfiguratorType> >( m_numVertices, m_barycenterIndices,
                                                                           1. / m_barycenterIndices.size() / m_barycenterIndices.size() );
  }

protected:
  VecType evaluateBarycenter( const VectorType &Point ) const {
    assert (( Point.size() != 3 * m_Topology.getNumVertices()) &&
            "BarycenterPenalty::evaluateBarycenter: wrong number of dofs!" );

    VecType bary, p;
    for ( int i: m_barycenterIndices ) {
      getXYZCoord( Point, p, i );
      bary += p;
    }
    bary /= m_barycenterIndices.size();

    return bary;
  }

};