#pragma once

#include <goast/Core/BaseOpInterface.h>
#include <goast/Core/Topology.h>
#include <goast/Core/TriangleGeometry.h>

#include <Eigen/SPQRSupport> ///\todo Include in GOAST solvers
#include "FaceNormals.h"

template<typename ConfiguratorType=DefaultConfigurator>
class ProjectionProlongationOperator : public BaseOp<typename ConfiguratorType::VectorType> {
public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;
  using VecType = typename ConfiguratorType::VecType;

  const MeshTopologySaver &m_refTopology;
  const VectorType &m_refGeometry;

  const MeshTopologySaver &m_Topology;

  MatrixType m_vertexMatrix;
  MatrixType m_normalMatrix;
  TripletListType m_vertexTriplets;
  TripletListType m_normalTriplets;

  const FaceNormals<ConfiguratorType> m_N;

public:
  ProjectionProlongationOperator( const MeshTopologySaver &refTopology, const VectorType &refGeometry,
                                  const MeshTopologySaver &Topology, const VectorType &Geometry )
          : m_refTopology( refTopology ), m_refGeometry( refGeometry ), m_Topology( Topology ), m_N(refTopology) {
    // Set up matrices


    m_vertexMatrix.resize( 3 * Topology.getNumVertices(), 3 * refTopology.getNumVertices());
    m_vertexMatrix.setZero();
    m_normalMatrix.resize( 3 * Topology.getNumVertices(), 3 * refTopology.getNumFaces());
    m_normalMatrix.setZero();

    const int refNumVertices = refTopology.getNumVertices();
    const int refNumFaces = refTopology.getNumFaces();
    const int numVertices = Topology.getNumVertices();

    const VectorType refFaceNormals = m_N( refGeometry );

    VecType p;
    ELEMENT_TYPE elementType;
    int elementIdx;

    for ( int vertexIdx = 0; vertexIdx < Topology.getNumVertices(); vertexIdx++ ) {
      getXYZCoord( Geometry, p, vertexIdx );

      std::tie( elementType, elementIdx ) = findClosestElement( p );

      if ( elementType == FACE ) {
        std::array<int, 3> Vertices = { refTopology.getNodeOfTriangle( elementIdx, 0 ),
                                        refTopology.getNodeOfTriangle( elementIdx, 1 ),
                                        refTopology.getNodeOfTriangle( elementIdx, 2 ) };
        std::array<VecType, 3> Points;
        getXYZCoord( refGeometry, Points[0], Vertices[0] );
        getXYZCoord( refGeometry, Points[1], Vertices[1] );
        getXYZCoord( refGeometry, Points[2], Vertices[2] );

        VecType Normal;
        getXYZCoord( refFaceNormals, Normal, elementIdx );

        RealType dist = dotProduct( Normal, p - Points[0] );
        VecType cPoint = p - Normal * dist;
        VecType bary = getBarycentricCoordinates( cPoint, Points[0], Points[1], Points[2] );

        VecType rPoint = Points[0] * bary[0] + Points[1] * bary[1] + Points[2] * bary[2];

        for ( int i : { 0, 1, 2 } ) {
          for ( int j : { 0, 1, 2 } ) {
            m_vertexTriplets.emplace_back( vertexIdx + i * numVertices, Vertices[j] + i * refNumVertices, bary[j] );
          }

          m_normalTriplets.emplace_back( vertexIdx + i * numVertices, elementIdx + i * refNumFaces, dist );
        }
      }
      else if ( elementType == EDGE ) {
        // End points of edge
        std::array<int, 2> Vertices = { refTopology.getAdjacentNodeOfEdge( elementIdx, 0 ),
                                        refTopology.getAdjacentNodeOfEdge( elementIdx, 1 ) };

        std::array<VecType, 2> Points;
        getXYZCoord( refGeometry, Points[0], Vertices[0] );
        getXYZCoord( refGeometry, Points[1], Vertices[1] );

        VecType edgeVec = Points[1] - Points[0]; // Edge vectors
        VecType dirVec = p - Points[0]; // Vector to test point

        RealType coeff = dotProduct( edgeVec, dirVec ) / edgeVec.normSqr();

        // Compute closest point
        VecType cPoint = Points[0] + edgeVec * coeff;
        RealType dist = (cPoint - p).norm();

        // Normals of adjacent triangles

        int f0 = refTopology.getAdjacentTriangleOfEdge( elementIdx, 0 );
        int f1 = refTopology.getAdjacentTriangleOfEdge( elementIdx, 1 );


        if ( f1 == -1 ) {
          for ( const int i: { 0, 1, 2 } ) {
            m_vertexTriplets.emplace_back( vertexIdx + i * numVertices, Vertices[0] + i * refNumVertices, 1 - coeff );
            m_vertexTriplets.emplace_back( vertexIdx + i * numVertices, Vertices[1] + i * refNumVertices, coeff );

            m_normalTriplets.emplace_back( vertexIdx + i * numVertices, f0 + i * refNumFaces, dist );
          }
        }
        else {
          VecType Normal_0, Normal_1;
          getXYZCoord( refFaceNormals, Normal_0, f0 );
          getXYZCoord( refFaceNormals, Normal_1, f1 );

          // Direction
          VecType Direction = ( p - cPoint ).normalized();

          // Compute intrinsic representation of direction
          MatrixType localNormalMatrix( 3, 2 );
          VectorType eDir( 3 );
          for ( int i: { 0, 1, 2 } ) {
            localNormalMatrix.coeffRef( i, 0 ) = Normal_0[i];
            localNormalMatrix.coeffRef( i, 1 ) = Normal_1[i];
            eDir[i] = Direction[i];
          }

          if ( dist <= std::numeric_limits<RealType>::epsilon() * 10 )
            eDir.setZero();

          Eigen::SPQR<MatrixType> nSolver( localNormalMatrix );
          VectorType edgeWeights = nSolver.solve( eDir );

          for ( const int i: { 0, 1, 2 } ) {
            m_vertexTriplets.emplace_back( vertexIdx + i * numVertices, Vertices[0] + i * refNumVertices, 1 - coeff );
            m_vertexTriplets.emplace_back( vertexIdx + i * numVertices, Vertices[1] + i * refNumVertices, coeff );

            m_normalTriplets.emplace_back( vertexIdx + i * numVertices, f0 + i * refNumFaces, dist * edgeWeights[0] );
            m_normalTriplets.emplace_back( vertexIdx + i * numVertices, f1 + i * refNumFaces, dist * edgeWeights[1] );
          }
        }
      }
      else {
        // Point
        VecType vPoint;
        getXYZCoord( m_refGeometry, vPoint, elementIdx );

        // Direction and distance
        RealType dist = (vPoint - p).norm();
        VecType Direction = (p - vPoint).normalized();
        VectorType eDir( 3 );
        for ( int i : { 0, 1, 2 } )
          eDir[i] = Direction[i];
        if (dist <= std::numeric_limits<RealType>::epsilon() * 10)
          eDir.setZero();

        // Collect normals
        std::vector<int> Faces;
        refTopology.getVertex1RingFaces( elementIdx, Faces );

        MatrixType localNormalMatrix( 3, Faces.size());
        for ( int j = 0; j < Faces.size(); j++ ) {
          VecType Normal;
          getXYZCoord( refFaceNormals, Normal, Faces[j] );
          for ( int i : { 0, 1, 2 } )
            localNormalMatrix.coeffRef( i, j ) = Normal[i];
        }

        Eigen::SPQR<MatrixType> nSolver( localNormalMatrix );
        VectorType faceWeights = nSolver.solve( eDir );

        for ( int i : { 0, 1, 2 } ) {
          m_vertexTriplets.emplace_back( vertexIdx + i * numVertices, elementIdx + i * refNumVertices, 1. );


          for ( int j = 0; j < Faces.size(); j++ ) {
            m_normalTriplets.emplace_back( vertexIdx + i * numVertices, Faces[j] + i * refNumFaces,
                                         dist * faceWeights[j] );
          }
        }
      }
    }

    m_vertexMatrix.setFromTriplets( m_vertexTriplets.begin(), m_vertexTriplets.end());
    m_normalMatrix.setFromTriplets( m_normalTriplets.begin(), m_normalTriplets.end());
  }

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    assert( Arg.size() == 3 * m_refTopology.getNumVertices() &&
      "ProjectionProlongationOperator::apply: Wrong size of argument" );

    Dest.resize( 3 * m_Topology.getNumVertices() );
    Dest.setZero();

    const VectorType faceNormals = m_N( Arg );

    Dest = m_vertexMatrix * Arg;
    Dest += m_normalMatrix * faceNormals;
  }

protected:
  enum ELEMENT_TYPE {
    VERTEX, EDGE, FACE
  };

  /**
   * \brief Compute closest element of reference mesh to point
   * \param p 3D-coordinates of query point
   * \return Tuple with type and id of the closest element
   *
   * \todo Switch to something more efficient
   */
  std::tuple<ELEMENT_TYPE, int> findClosestElement( const VecType &p ) {
    RealType Distance = std::numeric_limits<RealType>::infinity();
    ELEMENT_TYPE elementType = FACE;
    int elementId = -1;

    for ( int faceIdx = 0; faceIdx < m_refTopology.getNumFaces(); faceIdx++ ) {
      std::array<int, 3> Vertices = { m_refTopology.getNodeOfTriangle( faceIdx, 0 ),
                                      m_refTopology.getNodeOfTriangle( faceIdx, 1 ),
                                      m_refTopology.getNodeOfTriangle( faceIdx, 2 ) };
      std::array<VecType, 3> Points;
      getXYZCoord( m_refGeometry, Points[0], Vertices[0] );
      getXYZCoord( m_refGeometry, Points[1], Vertices[1] );
      getXYZCoord( m_refGeometry, Points[2], Vertices[2] );

      VecType Normal;
      getNormal( Points[0], Points[1], Points[2], Normal );

      RealType dist = dotProduct( Normal, p - Points[0] );

      VecType cPoint = p - Normal * dist;

      VecType bary = getBarycentricCoordinates( cPoint, Points[0], Points[1], Points[2] );

      if ( std::min( { bary[0], bary[1], bary[2] } ) < 0. ) {
        continue;
      }

      if ( std::max( { bary[0], bary[1], bary[2] } ) > 1. ) {
        continue;
      }


      if ( std::abs( dist ) < Distance ) {
        Distance = std::abs( dist );
        elementId = faceIdx;
        elementType = FACE;
      }
    }

    // Points
    for ( int vertexIdx = 0; vertexIdx < m_refTopology.getNumVertices(); vertexIdx++ ) {
      VecType vPoint;
      getXYZCoord( m_refGeometry, vPoint, vertexIdx );

      RealType dist = (vPoint - p).norm();
      if ( dist < Distance ) {
        Distance = dist;
        elementType = VERTEX;
        elementId = vertexIdx;
      }
    }

    // Edges
    for ( int edgeIdx = 0; edgeIdx < m_refTopology.getNumEdges(); edgeIdx++ ) {
      std::array<int, 2> Vertices = { m_refTopology.getAdjacentNodeOfEdge( edgeIdx, 0 ),
                                      m_refTopology.getAdjacentNodeOfEdge( edgeIdx, 1 ) };

      std::array<VecType, 2> Points;
      getXYZCoord( m_refGeometry, Points[0], Vertices[0] );
      getXYZCoord( m_refGeometry, Points[1], Vertices[1] );

      VecType edgeVec = Points[1] - Points[0]; //
      VecType dirVec = p - Points[0]; // Vector to test point

      RealType coeff = dotProduct( edgeVec, dirVec ) / edgeVec.normSqr();
      if ( coeff > 1. || coeff < 0. )
        continue;

      VecType cPoint = Points[0] + edgeVec * coeff;
      RealType dist = (cPoint - p).norm();

      if ( dist < Distance ) {
        Distance = dist;
        elementType = EDGE;
        elementId = edgeIdx;
      }
    }

//    std::cout << "    Distance: " << Distance << std::endl;

    return std::make_tuple( elementType, elementId );
  }

  VecType getBarycentricCoordinates( const VecType &p, const VecType &a, const VecType &b, const VecType &c ) const {
    VecType retVal;

    VecType v0 = b - a, v1 = c - a, v2 = p - a;
    RealType d00 = dotProduct( v0, v0 );
    RealType d01 = dotProduct( v0, v1 );
    RealType d11 = dotProduct( v1, v1 );
    RealType d20 = dotProduct( v2, v0 );
    RealType d21 = dotProduct( v2, v1 );
    RealType denom = d00 * d11 - d01 * d01;
    retVal[1] = (d11 * d20 - d01 * d21) / denom;
    retVal[2] = (d00 * d21 - d01 * d20) / denom;
    retVal[0] = 1.0 - retVal[1] - retVal[2];
    return retVal;
  }
};
