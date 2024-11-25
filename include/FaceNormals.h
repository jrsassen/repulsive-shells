#pragma once

#include <goast/Core.h>

template<typename ConfiguratorType=DefaultConfigurator>
class FaceNormals : public BaseOp<typename ConfiguratorType::VectorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;
  using VecType = typename ConfiguratorType::VecType;

  const MeshTopologySaver &m_Topology;

public:
  explicit FaceNormals( const MeshTopologySaver &Topology ) : m_Topology( Topology ) {}

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    if ( Arg.size() != 3 * m_Topology.getNumVertices())
      throw std::length_error( "bla" );

    Dest.resize( 3 * m_Topology.getNumFaces());
    Dest.setZero();

    for ( int faceIdx = 0; faceIdx < m_Topology.getNumFaces(); faceIdx++ ) {
      std::array<int, 3> Vertices = { m_Topology.getNodeOfTriangle( faceIdx, 0 ),
                                      m_Topology.getNodeOfTriangle( faceIdx, 1 ),
                                      m_Topology.getNodeOfTriangle( faceIdx, 2 ) };
      std::array<VecType, 3> Points;
      getXYZCoord( Arg, Points[0], Vertices[0] );
      getXYZCoord( Arg, Points[1], Vertices[1] );
      getXYZCoord( Arg, Points[2], Vertices[2] );

      VecType Normal;
      getNormal( Points[0], Points[1], Points[2], Normal );

      setXYZCoord( Dest, Normal, faceIdx );
    }
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class FaceNormalsDerivative
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  const MeshTopologySaver &m_Topology;

public:
  explicit FaceNormalsDerivative( const MeshTopologySaver &Topology ) : m_Topology( Topology ) {}

  void apply( const VectorType &Arg, MatrixType &Dest ) const override {
    if ( Arg.size() != 3 * m_Topology.getNumVertices())
      throw std::length_error( "bla" );

    Dest.resize( 3 * m_Topology.getNumFaces(), 3 * m_Topology.getNumVertices());
    Dest.setZero();

    TripletListType tripletList;
    tripletList.reserve( 3 * 9 * m_Topology.getNumFaces());


    auto localToGlobal = [&]( int k, int l, const MatType &localMatrix ) {
      for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
          tripletList.emplace_back( i * m_Topology.getNumFaces() + k,
                                    j * m_Topology.getNumVertices() + l,
                                    localMatrix( i, j ));
    };


    for ( int faceIdx = 0; faceIdx < m_Topology.getNumFaces(); faceIdx++ ) {
      std::array<int, 3> Vertices = { m_Topology.getNodeOfTriangle( faceIdx, 0 ),
                                      m_Topology.getNodeOfTriangle( faceIdx, 1 ),
                                      m_Topology.getNodeOfTriangle( faceIdx, 2 ) };
      std::array<VecType, 3> Points;
      getXYZCoord( Arg, Points[0], Vertices[0] );
      getXYZCoord( Arg, Points[1], Vertices[1] );
      getXYZCoord( Arg, Points[2], Vertices[2] );

      MatType localDerivative;

      getNormalGradientPk( Points[0], Points[1], Points[2], localDerivative );
      localToGlobal( faceIdx, Vertices[2], localDerivative );

      getNormalGradientPk( Points[1], Points[2], Points[0], localDerivative );
      localToGlobal( faceIdx, Vertices[0], localDerivative );

      getNormalGradientPk( Points[2], Points[0], Points[1], localDerivative );
      localToGlobal( faceIdx, Vertices[1], localDerivative );
    }

    Dest.setFromTriplets( tripletList.begin(), tripletList.end());
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class FaceNormalsHessian
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::TensorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TensorType = typename ConfiguratorType::TensorType;
  using TripletListType = std::vector<TripletType>;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  const MeshTopologySaver &m_Topology;

public:
  explicit FaceNormalsHessian( const MeshTopologySaver &Topology ) : m_Topology( Topology ) {}

  void apply( const VectorType &Arg, TensorType &Dest ) const override {
    if ( Arg.size() != 3 * m_Topology.getNumVertices())
      throw std::length_error( "bla" );

    Dest.resize( 3 * m_Topology.getNumFaces(), 3 * m_Topology.getNumVertices(), 3 * m_Topology.getNumVertices());
    Dest.setZero();

    std::vector<TripletListType> tripletList( 3 * m_Topology.getNumFaces());
//    tripletList.reserve( 3 * 9 * m_Topology.getNumFaces());

    for ( int faceIdx = 0; faceIdx < m_Topology.getNumFaces(); faceIdx++ ) {
      std::array<int, 3> Vertices = { m_Topology.getNodeOfTriangle( faceIdx, 0 ),
                                      m_Topology.getNodeOfTriangle( faceIdx, 1 ),
                                      m_Topology.getNodeOfTriangle( faceIdx, 2 ) };
      std::array<VecType, 3> Points;
      getXYZCoord( Arg, Points[0], Vertices[0] );
      getXYZCoord( Arg, Points[1], Vertices[1] );
      getXYZCoord( Arg, Points[2], Vertices[2] );

      MatType localDerivative;
      GenericTensor<Eigen::MatrixXd> localHessian;

      getNormalHessian( Points[0], Points[1], Points[2], localHessian );
      for ( int n: { 0, 1, 2 } ) // Coordinate of face normal
        for ( int v1: { 0, 1, 2 } ) // first vertex of face
          for ( int v2: { 0, 1, 2 } ) // second vertex of face
            for ( int i: { 0, 1, 2 } ) // entry of first vertex
              for ( int j: { 0, 1, 2 } ) // entry of second vertex
                tripletList[n * m_Topology.getNumFaces() + faceIdx].emplace_back(
                        i * m_Topology.getNumVertices() + Vertices[v1],
                        j * m_Topology.getNumVertices() + Vertices[v2],
                        localHessian( n, i * 3 + v1, j * 3 + v2 )
                );
    }

    Dest.setFromTriplets( tripletList );
  }
};