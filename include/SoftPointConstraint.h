#pragma once

#include <goast/Core.h>

template<typename ConfiguratorType=DefaultConfigurator>
class PointPositionPenalty : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::RealType> {
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const MeshTopologySaver &m_Topology;
  const std::vector<int> &m_constrainedVertices;
  const VectorType &m_referenceGeometry;

public:
  explicit PointPositionPenalty( const MeshTopologySaver &Topology,
                                 const std::vector<int> &constrainedVertices,
                                 const VectorType &referenceGeometry ) : m_Topology( Topology ),
                                                                         m_constrainedVertices( constrainedVertices ),
                                                                         m_referenceGeometry( referenceGeometry ) {


  }

  void apply( const VectorType &Arg, RealType &Dest ) const {
    assert( Arg.size() == 3 * m_Topology.getNumVertices() && "PointPositionPenalty::apply: Wrong size of input!" );

    Dest = 0.;

//    VectorType nodalAreas;
//    computeNodalAreas<ConfiguratorType>( m_Topology, Arg, nodalAreas );

    for ( int i: m_constrainedVertices ) {
      VecType p, q;
      getXYZCoord( Arg, p, i );
      getXYZCoord( m_referenceGeometry, q, i );
      VecType displacement = p - q;

      Dest += displacement.squaredNorm() / 2.;
    }
  }
};


template<typename ConfiguratorType=DefaultConfigurator>
class PointPositionPenaltyDerivative
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::VectorType> {
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const MeshTopologySaver &m_Topology;
  const std::vector<int> &m_constrainedVertices;
  const VectorType &m_referenceGeometry;

public:
  explicit PointPositionPenaltyDerivative( const MeshTopologySaver &Topology,
                                           const std::vector<int> &constrainedVertices,
                                           const VectorType &referenceGeometry ) : m_Topology( Topology ),
                                                                                   m_constrainedVertices(
                                                                                           constrainedVertices ),
                                                                                   m_referenceGeometry(
                                                                                           referenceGeometry ) {


  }

  void apply( const VectorType &Arg, VectorType &Dest ) const {
    assert( Arg.size() == 3 * m_Topology.getNumVertices() &&
            "PointPositionPenaltyDerivative::apply: Wrong size of input!" );

    Dest.resize( 3 * m_Topology.getNumVertices());
    Dest.setZero();

    for ( int i: m_constrainedVertices ) {
      VecType p, q;
      getXYZCoord( Arg, p, i );
      getXYZCoord( m_referenceGeometry, q, i );
      VecType localGradient = p - q;

      addXYZCoord( Dest, localGradient, i );
    }

  }
};


template<typename ConfiguratorType=DefaultConfigurator>
class PointPositionPenaltyHessian
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const int m_numVertices;
  const MeshTopologySaver &m_Topology;
  const std::vector<int> &m_constrainedVertices;
  const VectorType &m_referenceGeometry;

public:
  explicit PointPositionPenaltyHessian( const MeshTopologySaver &Topology,
                                        const std::vector<int> &constrainedVertices,
                                        const VectorType &referenceGeometry )
          : m_Topology( Topology ), m_numVertices( Topology.getNumVertices()),
            m_constrainedVertices( constrainedVertices ), m_referenceGeometry( referenceGeometry ) {


  }

  void apply( const VectorType &Arg, SparseMatrixType &Dest ) const {
    assert( Arg.size() == 3 * m_numVertices && "PointPositionPenaltyHessian::apply: Wrong size of input!" );

    Dest.resize( 3 * m_numVertices, 3 * m_numVertices );
    Dest.setZero();

    TripletListType globalTripletList;

    for ( int i: m_constrainedVertices ) {
      for ( int k: { 0, 1, 2 } )
        globalTripletList.emplace_back( k * m_numVertices + i, k * m_numVertices + i, 1. );
    }

    Dest.setFromTriplets( globalTripletList.begin(), globalTripletList.end());
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class PointPositionPenaltyHessianOperator : public MapToLinOp<ConfiguratorType> {
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const int m_numVertices;
  const MeshTopologySaver &m_Topology;
  const std::vector<int> &m_constrainedVertices;
  const VectorType &m_referenceGeometry;

public:
  explicit PointPositionPenaltyHessianOperator( const MeshTopologySaver &Topology,
                                                const std::vector<int> &constrainedVertices,
                                                const VectorType &referenceGeometry )
          : m_Topology( Topology ), m_numVertices( Topology.getNumVertices()),
            m_constrainedVertices( constrainedVertices ), m_referenceGeometry( referenceGeometry ) {


  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A( 3 * m_numVertices, 3 * m_numVertices );
    A.setZero();

    TripletListType tripletList;
    for ( int i: m_constrainedVertices ) {
      for ( int k: { 0, 1, 2 } )
        tripletList.emplace_back( k * m_numVertices + i, k * m_numVertices + i, 1. );
    }
    A.setFromTriplets( tripletList.begin(), tripletList.end());

    return std::make_unique<MatrixOperator<ConfiguratorType>>( std::move( A ));
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class TrackingPathEnergy : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::RealType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;


  const std::vector<int> &m_constrainedVertices;

  const VectorType &m_start, m_end;
  int m_K;
  int m_numOfFreeShapes;

public:
  TrackingPathEnergy( const std::vector<int> &constrainedVertices, int K, const VectorType &start,
                      const VectorType &end )
          : m_start( start ), m_end( end ), m_K( K ), m_constrainedVertices( constrainedVertices ),
            m_numOfFreeShapes( K - 1 ) {}

  // Arg =  (s_1, ..., s_{K-1}) are intermediate shapes only, where s_k = (x_k, y_k, z_k)
  void apply( const VectorType &Arg, RealType &Dest ) const {
    if ( Arg.size() % m_numOfFreeShapes != 0 )
      throw std::length_error( "TrackingPathEnergy::apply: wrong number of dofs!" );

    if ( Arg.size() / m_numOfFreeShapes != m_start.size())
      throw std::length_error( "TrackingPathEnergy::apply: wrong size of dofs!" );

    VectorType singleEnergies;
    evaluateSingleEnergies( Arg, singleEnergies );

    Dest = m_K * singleEnergies.sum();
  }

  void evaluateSingleEnergies( const VectorType &Arg, VectorType &Energies ) const {
    const int numLocalDofs = Arg.size() / m_numOfFreeShapes;

    if ( Arg.size() % m_numOfFreeShapes != 0 )
      throw std::length_error( "TrackingPathEnergy::evaluateSingleEnergies: wrong number of dofs!" );

    if ( Arg.size() / m_numOfFreeShapes != m_start.size())
      throw std::length_error( "TrackingPathEnergy::evaluateSingleEnergies: wrong size of dofs!" );

    Energies.resize( m_K );
    Energies.setZero();

    for ( int i: m_constrainedVertices ) {
      VecType p, q;

      // 0
      getXYZCoord( m_start, p, i );
      getXYZCoord( Arg.head( numLocalDofs ), q, i );
      Energies[0] += ( q - p ).squaredNorm();

      for ( int k = 1; k < m_numOfFreeShapes; k++ ) {
        getXYZCoord( Arg.segment(( k - 1 ) * numLocalDofs, numLocalDofs ), p, i );
        getXYZCoord( Arg.segment( k * numLocalDofs, numLocalDofs ), q, i );
        Energies[k] += ( q - p ).squaredNorm();
      }

      getXYZCoord( Arg.tail( numLocalDofs ), p, i );
      getXYZCoord( m_end, q, i );
      Energies[m_numOfFreeShapes] += ( q - p ).squaredNorm();
    }

    Energies /= 2.;
  }

};


template<typename ConfiguratorType=DefaultConfigurator>
class TrackingPathEnergyGradient
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;

  const std::vector<int> &m_constrainedVertices;

  const VectorType &m_start, m_end;
  int m_K;
  int m_numOfFreeShapes;

public:
  TrackingPathEnergyGradient( const std::vector<int> &constrainedVertices,
                              int K, const VectorType &start, const VectorType &end )
          : m_start( start ), m_end( end ), m_K( K ), m_constrainedVertices( constrainedVertices ),
            m_numOfFreeShapes( K - 1 ) {}


  // Arg =  (s_1, ..., s_{K-1}) are intermediate shapes only, where s_k = (x_k, y_k, z_k)
  void apply( const VectorType &Arg, VectorType &Dest ) const {
    const int numLocalDofs = Arg.size() / m_numOfFreeShapes;
    const int numVertices = numLocalDofs / 3;

    if ( Arg.size() % m_numOfFreeShapes != 0 )
      throw std::length_error( "TrackingPathEnergyGradient::apply: wrong number of dofs!" );

    if ( Arg.size() / m_numOfFreeShapes != m_start.size())
      throw std::length_error( "TrackingPathEnergyGradient::apply: wrong size of dofs!" );

    if ( Dest.size() != Arg.size())
      Dest.resize( Arg.size());

    Dest.setZero();

    for ( int i: m_constrainedVertices ) {
      VecType p, q, displacement;

      // 0
      getXYZCoord( m_start, p, i );
      getXYZCoord( Arg.segment( 0 * numLocalDofs, numLocalDofs ), q, i );
      displacement = q - p;

      for ( int j = 0; j < 3; j++ )
        Dest[0 * numLocalDofs + j * numVertices + i] += displacement[j];

      for ( int k = 1; k < m_numOfFreeShapes; k++ ) {
        getXYZCoord( Arg.segment(( k - 1 ) * numLocalDofs, numLocalDofs ), p, i );
        getXYZCoord( Arg.segment( k * numLocalDofs, numLocalDofs ), q, i );
        displacement = q - p;

        for ( int j = 0; j < 3; j++ ) {
          Dest[( k - 1 ) * numLocalDofs + j * numVertices + i] -= displacement[j];
          Dest[k * numLocalDofs + j * numVertices + i] += displacement[j];
        }
      }

      getXYZCoord( Arg.segment(( m_numOfFreeShapes - 1 ) * numLocalDofs, numLocalDofs ), p, i );
      getXYZCoord( m_end, q, i );
      displacement = q - p;
      for ( int j = 0; j < 3; j++ )
        Dest[( m_numOfFreeShapes - 1 ) * numLocalDofs + j * numVertices + i] -= displacement[j];
    }

    Dest *= m_K;
  }

};
template<typename ConfiguratorType=DefaultConfigurator>
class TrackingPathEnergyHessian
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const std::vector<int> &m_constrainedVertices;

  const VectorType &m_start, m_end;
  int m_K;
  int m_numOfFreeShapes;

  MatrixType m_Hess;

public:
  TrackingPathEnergyHessian( const std::vector<int> &constrainedVertices,
                                      int K, const VectorType &start, const VectorType &end )
          : m_start( start ), m_end( end ), m_K( K ),m_constrainedVertices( constrainedVertices ), m_numOfFreeShapes( K - 1 ) {

    TripletListType tripletList;

    const int numLocalDofs = start.size(); ;
    const int numVertices = numLocalDofs / 3;

    m_Hess.resize( m_numOfFreeShapes * numLocalDofs, m_numOfFreeShapes * numLocalDofs);
    m_Hess.setZero();
    for ( int i: m_constrainedVertices ) {
      for ( int j: { 0, 1, 2 } )
        tripletList.emplace_back( 0 * numLocalDofs + j * numVertices + i, 0 * numLocalDofs + j * numVertices + i, m_K );

      for ( int k = 1; k < m_numOfFreeShapes; k++ ) {
        for ( int j = 0; j < 3; j++ ) {
          tripletList.emplace_back(( k - 1 ) * numLocalDofs + j * numVertices + i,
                                   ( k - 1 ) * numLocalDofs + j * numVertices + i, m_K );
          tripletList.emplace_back(( k - 1 ) * numLocalDofs + j * numVertices + i,
                                   k * numLocalDofs + j * numVertices + i, m_K );
          tripletList.emplace_back( k * numLocalDofs + j * numVertices + i,
                                    ( k - 1 ) * numLocalDofs + j * numVertices + i, m_K );
          tripletList.emplace_back( k * numLocalDofs + j * numVertices + i,
                                    k * numLocalDofs + j * numVertices + i,
                                    m_K );
        }
      }

      for ( int j: { 0, 1, 2 } )
        tripletList.emplace_back(( m_numOfFreeShapes - 1 ) * numLocalDofs + j * numVertices + i,
                                 ( m_numOfFreeShapes - 1 ) * numLocalDofs + j * numVertices + i, m_K );
    }
    m_Hess.setFromTriplets( tripletList.begin(), tripletList.end());
  }


  // Arg =  (s_1, ..., s_{K-1}) are intermediate shapes only, where s_k = (x_k, y_k, z_k)
  void apply( const VectorType &Arg, MatrixType &Dest ) const {
    const int numLocalDofs = Arg.size() / m_numOfFreeShapes;
    const int numVertices = numLocalDofs / 3;

    if ( Arg.size() % m_numOfFreeShapes != 0 )
      throw std::length_error( "DifferencePathEnergyGradient::apply: wrong number of dofs!" );

    if ( Arg.size() / m_numOfFreeShapes != m_start.size())
      throw std::length_error( "DifferencePathEnergyGradient::apply: wrong size of dofs!" );

    if ( Dest.rows() != Arg.size() || Dest.cols() != Arg.size())
      Dest.resize( Arg.size(), Arg.size());

    Dest = m_Hess;
  }

};

template<typename ConfiguratorType=DefaultConfigurator>
class TrackingPathEnergyHessianOperator : public MapToLinOp<ConfiguratorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const std::vector<int> &m_constrainedVertices;

  const VectorType &m_start, m_end;
  int m_K;
  int m_numOfFreeShapes;

  MatrixType m_Hess;

public:
  TrackingPathEnergyHessianOperator( const std::vector<int> &constrainedVertices,
                                     int K, const VectorType &start, const VectorType &end )
          : m_start( start ), m_end( end ), m_K( K ), m_constrainedVertices( constrainedVertices ),
            m_numOfFreeShapes( K - 1 ) {

    TripletListType tripletList;

    const int numLocalDofs = start.size();;
    const int numVertices = numLocalDofs / 3;

    m_Hess.resize( m_numOfFreeShapes * numLocalDofs, m_numOfFreeShapes * numLocalDofs);
    m_Hess.setZero();
    for ( int i: m_constrainedVertices ) {
      for ( int j: { 0, 1, 2 } )
        tripletList.emplace_back( 0 * numLocalDofs + j * numVertices + i, 0 * numLocalDofs + j * numVertices + i, m_K );

      for ( int k = 1; k < m_numOfFreeShapes; k++ ) {
        for ( int j = 0; j < 3; j++ ) {
          tripletList.emplace_back(( k - 1 ) * numLocalDofs + j * numVertices + i,
                                   ( k - 1 ) * numLocalDofs + j * numVertices + i, m_K );
          tripletList.emplace_back(( k - 1 ) * numLocalDofs + j * numVertices + i,
                                   k * numLocalDofs + j * numVertices + i, m_K );
          tripletList.emplace_back( k * numLocalDofs + j * numVertices + i,
                                    ( k - 1 ) * numLocalDofs + j * numVertices + i, m_K );
          tripletList.emplace_back( k * numLocalDofs + j * numVertices + i,
                                    k * numLocalDofs + j * numVertices + i,
                                    m_K );
        }
      }

      for ( int j: { 0, 1, 2 } )
        tripletList.emplace_back(( m_numOfFreeShapes - 1 ) * numLocalDofs + j * numVertices + i,
                                 ( m_numOfFreeShapes - 1 ) * numLocalDofs + j * numVertices + i, m_K );
    }
    m_Hess.setFromTriplets( tripletList.begin(), tripletList.end());
  }

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
      return std::make_unique<MatrixOperator<ConfiguratorType>>( m_Hess );
  }

};