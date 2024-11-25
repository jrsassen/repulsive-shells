#pragma once

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>
#include "ClusterTree.h"
#include "TangentPointNode.h"

namespace SpookyTPE {

template<typename ConfiguratorType=DefaultConfigurator>
class BarnesHutTangentPointEnergy : public ObjectiveFunctional<ConfiguratorType>,
                                    public TimedClass<BarnesHutTangentPointEnergy<ConfiguratorType>> {
public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  using typename TimedClass<BarnesHutTangentPointEnergy>::ScopeTimer;

  using NodeType = TangentPointNode<ConfiguratorType>;

  const int m_numVertices;
  const int m_alpha, m_beta;
  const RealType m_thetaSquared;
  const RealType m_innerWeight;

  const MeshTopologySaver &m_Topology;
  const int m_numFaces;

  const int m_numLocalDofs;

  // TPE cache
  mutable VectorType m_oldPoint;
  mutable VectorType m_oldGradientPoint;
  mutable RealType m_Value;
  mutable VectorType m_Gradient;

public:
  BarnesHutTangentPointEnergy( const MeshTopologySaver&Topology,
                               int alpha,
                               int beta,
                               RealType innerWeight = 1.,
                               RealType theta = 0.25 ) : m_Topology( Topology ),
                                                                 m_alpha( alpha ),
                                                                 m_beta( beta ),
                                                                 m_innerWeight( innerWeight ),
                                                                 m_thetaSquared( theta * theta ),
                                                                 m_numVertices( Topology.getNumVertices() ),
                                                                 m_numFaces( Topology.getNumFaces() ),
                                                                 m_numLocalDofs( 3 * Topology.getNumVertices() ) {
    m_oldPoint = VectorType::Zero( 3 * m_numVertices );
    m_Gradient = VectorType::Zero( 3 * m_numVertices );
    m_oldGradientPoint = VectorType::Zero( 3 * m_numVertices );
  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    ScopeTimer timer( "evaluate" );
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "BarnesHutTangentPointEnergy::evaluate: wrong number of dofs!" );

    evaluateTangentPointGradients( Point );

    Value = m_innerWeight * m_Value;
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    ScopeTimer timer( "evaluateGradient" );
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size())
      Gradient.resize( Point.size());

    Gradient.setZero();

    evaluateTangentPointGradients( Point );

    Gradient = m_innerWeight * m_Gradient;
  }

  void resetCache() const {
    m_oldPoint.setZero();
    m_oldGradientPoint.setZero();
  }

protected:
  void evaluateTangentPointEnergies( const VectorType &Point ) const {
    ScopeTimer timer( "evaluateTangentPointEnergies" );
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateTPE: wrong number of dofs!" );

    if (( m_oldPoint - Point ).norm() < 1.e-8 )
      return;
    else {
      m_Value = 0.;

      std::vector<VecType> faceNormals( m_numFaces );
      std::vector<VecType> faceLowerBounds( m_numFaces );
      std::vector<VecType> faceUpperBounds( m_numFaces );
      std::vector<VecType> faceBarycenters( m_numFaces );
      VectorType BarycenterVector( 3 * m_numFaces );
      VectorType faceAreas( m_numFaces );
      VectorType faceRadii( m_numFaces );
      VectorType EdgeLengths;
      getEdgeLengths<ConfiguratorType>(m_Topology, Point, EdgeLengths);

      // Precompute areas, normals and barycenters
      {
        ScopeTimer innerTimer( "evaluateTangentPointEnergies::01_Precompute" );
        for ( int fIdx = 0; fIdx < m_numFaces; fIdx++ ) {
          faceAreas[fIdx] = getNormalAndArea<ConfiguratorType>( m_Topology, fIdx, Point, faceNormals[fIdx] );

          for ( int i: { 0, 1, 2 } ) {
            VecType p;
            getXYZCoord( Point, p, m_Topology.getNodeOfTriangle( fIdx, i ));

            for ( int d : { 0, 1, 2 } ) {
              faceLowerBounds[fIdx][d] = std::min( faceLowerBounds[fIdx][d], p[d] );
              faceUpperBounds[fIdx][d] = std::max( faceUpperBounds[fIdx][d], p[d] );
            }

            faceBarycenters[fIdx] += p;
          }
          faceBarycenters[fIdx] /= 3.;

          setXYZCoord( BarycenterVector, faceBarycenters[fIdx], fIdx );

          RealType a = EdgeLengths[m_Topology.getEdgeOfTriangle(fIdx, 0)];
          RealType b = EdgeLengths[m_Topology.getEdgeOfTriangle(fIdx, 1)];
          RealType c = EdgeLengths[m_Topology.getEdgeOfTriangle(fIdx, 2)];

          faceRadii[fIdx] = my_pow( std::max( { a, b, c } ), 2 );
        }
      }

      // Compute cluster tree
      std::unique_ptr<NodeType> ClusterTree;
      {
        ScopeTimer innerTimer( "evaluateTangentPointEnergies::02_ClusterTree" );
        ClusterTree = std::make_unique<LongestAxisTree<NodeType>>(BarycenterVector, 8 );
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointEnergies::03_ComputeGQ" );
        ClusterTree->computeGeometricQuantities( faceBarycenters, faceAreas, faceNormals );
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointEnergies::04_ComputeBH" );
        for ( int f1 = 0; f1 < m_numFaces; f1++ ) {
          RealType f1Value = 0.;

          std::queue<NodeType *> nodeQueue;
          nodeQueue.push( ClusterTree.get() );
          while ( !nodeQueue.empty()) {
            // Get the next node
            NodeType *node = nodeQueue.front();
            // Remove the currently visited node from the queue
            nodeQueue.pop();

            // Bounding box distance
            VecType d;
            for ( int k: { 0, 1, 2 } )
              d[k] = std::max( 0., std::max( faceLowerBounds[f1][k], node->lowerBounds()[k] ) -
                                   std::min( faceUpperBounds[f1][k], node->upperBounds()[k] ));
            RealType R2 = d.squaredNorm();
            RealType h2 = std::max( faceRadii[f1], node->squaredDiameter());

            if ( h2 < m_thetaSquared * R2 ) {
              f1Value += node->Area() * TPEKernel( faceBarycenters[f1], node->Centroid(), faceNormals[f1]);
            }
            else {
              if ( node->Children().empty()) {
                for ( const int &f2: node->Vertices()) {
                  if (f1 == f2)
                    continue;
                  f1Value += faceAreas[f2] * TPEKernel( faceBarycenters[f1], faceBarycenters[f2], faceNormals[f1] );
                }
              }
              else {
                for ( auto &newNode: node->Children())
                  nodeQueue.push( &newNode );
              }
            }
          }

          m_Value += faceAreas[f1] * f1Value;
        }
      }

      m_oldPoint = Point;
    }
  }

  void evaluateTangentPointGradients( const VectorType &Point ) const {
    ScopeTimer timer( "evaluateTangentPointGradients" );
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateTPE: wrong number of dofs!" );

    if (( m_oldGradientPoint - Point ).norm() < 1.e-8 )
      return;
    else {
      m_Value = 0.;
      m_Gradient.setZero();

      std::vector<VecType> faceNormals( m_numFaces );
      std::vector<VecType> faceLowerBounds( m_numFaces );
      std::vector<VecType> faceUpperBounds( m_numFaces );
      std::vector<VecType> faceBarycenters( m_numFaces );
      VectorType BarycenterVector( 3 * m_numFaces );
      VectorType faceAreas( m_numFaces );
      std::vector<VecType> areaGradientsPi( m_numFaces );
      std::vector<VecType> areaGradientsPj( m_numFaces );
      std::vector<VecType> areaGradientsPk( m_numFaces );
      std::vector<VecType> edges( m_numFaces );
      std::vector<MatType> normalGradientsPi( m_numFaces );
      std::vector<MatType> normalGradientsPj( m_numFaces );
      std::vector<MatType> normalGradientsPk( m_numFaces );
      VectorType faceRadii( m_numFaces );
      VectorType EdgeLengths;
      getEdgeLengths<ConfiguratorType>(m_Topology, Point, EdgeLengths);

      // Precompute areas, normals and barycenters
      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::01_Precompute" );
        for ( int fIdx = 0; fIdx < m_numFaces; fIdx++ ) {
          // get indices of vertices
          std::array<int, 3> verts{};
          std::array<VecType, 3> nodes;
          for ( int j: { 0, 1, 2 } ) {
            verts[j] = m_Topology.getNodeOfTriangle( fIdx, j );
            getXYZCoord<VectorType, VecType>( Point, nodes[j], verts[j] );
          }

          // Areas and area gradients
          faceAreas[fIdx] = getNormalAndArea<ConfiguratorType>( m_Topology, fIdx, Point, faceNormals[fIdx] );
//          getAreaGradients<ConfiguratorType>( m_Topology, Point, faceAreas, areaGradientsPi, areaGradientsPj,
//                                              areaGradientsPk, edges );
          areaGradientsPk[fIdx].makeCrossProduct( faceNormals[fIdx], nodes[1] - nodes[0]);
          areaGradientsPk[fIdx] /= 2.;
          areaGradientsPi[fIdx].makeCrossProduct( faceNormals[fIdx], nodes[2] - nodes[1]);
          areaGradientsPi[fIdx] /= 2.;
          areaGradientsPj[fIdx].makeCrossProduct( faceNormals[fIdx], nodes[0] - nodes[2]);
          areaGradientsPj[fIdx] /= 2.;

          // Normal gradients
          getNormalGradientPk<RealType>( nodes[0], nodes[1], nodes[2], normalGradientsPk[fIdx] );
          getNormalGradientPk<RealType>( nodes[2], nodes[0], nodes[1], normalGradientsPj[fIdx] );
          getNormalGradientPk<RealType>( nodes[1], nodes[2], nodes[0], normalGradientsPi[fIdx] );

          // Barycenters
          for ( int i: { 0, 1, 2 } ) {
            faceBarycenters[fIdx] += nodes[i];

            for ( int d: { 0, 1, 2 } ) {
              faceLowerBounds[fIdx][d] = std::min( faceLowerBounds[fIdx][d], nodes[i][d] );
              faceUpperBounds[fIdx][d] = std::max( faceUpperBounds[fIdx][d], nodes[i][d] );
            }
          }
          faceBarycenters[fIdx] /= 3.;
          setXYZCoord( BarycenterVector, faceBarycenters[fIdx], fIdx );

          RealType a = EdgeLengths[m_Topology.getEdgeOfTriangle( fIdx, 0 )];
          RealType b = EdgeLengths[m_Topology.getEdgeOfTriangle( fIdx, 1 )];
          RealType c = EdgeLengths[m_Topology.getEdgeOfTriangle( fIdx, 2 )];

          faceRadii[fIdx] = my_pow( std::max( { a, b, c } ), 2 );
        }
      }

      // Compute cluster tree
      std::unique_ptr<NodeType> ClusterTree;
      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::02_ClusterTree" );
        ClusterTree = std::make_unique<LongestAxisTree<NodeType>>( BarycenterVector, 8 );
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::03_ComputeGQ" );
        ClusterTree->computeGeometricQuantities( faceBarycenters, faceAreas, faceNormals );
      }

      std::vector<RealType> faceAreaGradients( m_numFaces, 0. );
      std::vector<VecType> faceNormalGradients( m_numFaces );
      std::vector<VecType> faceCenterGradients( m_numFaces );

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::04_ComputeBH" );

        for ( int f1 = 0; f1 < m_numFaces; f1++ ) {
          RealType f1Value = 0.;

          std::queue<NodeType *> nodeQueue;
          nodeQueue.push( ClusterTree.get() );
          while ( !nodeQueue.empty()) {
            // Get the next node
            NodeType *node = nodeQueue.front();
            // Remove the currently visited node from the queue
            nodeQueue.pop();

            // Bounding box distance
            VecType d;
            for ( int k: { 0, 1, 2 } )
              d[k] = std::max( 0., std::max( faceLowerBounds[f1][k], node->lowerBounds()[k] ) -
                                   std::min( faceUpperBounds[f1][k], node->upperBounds()[k] ) );
            RealType R2 = d.squaredNorm();
            RealType h2 = std::max( faceRadii[f1], node->squaredDiameter() );

            if ( h2 < m_thetaSquared * R2 ) {
              std::array<VecType, 3> DK;
              RealType localValue = DTPEKernel( faceBarycenters[f1], node->Centroid(), faceNormals[f1], DK );
              f1Value += node->Area() * localValue;

              RealType factor = faceAreas[f1] * node->Area();
              faceCenterGradients[f1] += factor * DK[0];
              node->CentroidDerivativeFactor() += factor * DK[1];
              faceNormalGradients[f1] += factor * DK[2];
              node->AreaDerivativeFactor() += localValue * faceAreas[f1];
              // f1Value += node->Area() * TPEKernel( faceBarycenters[f1], node->Centroid(), faceNormals[f1]);
            }
            else {
              if ( node->Children().empty()) {
                for ( const int &f2: node->Vertices()) {
                  if (f1 == f2)
                    continue;

                  std::array<VecType, 3> DK;
                  RealType localValue = DTPEKernel( faceBarycenters[f1], faceBarycenters[f2], faceNormals[f1], DK );
                  f1Value += faceAreas[f2] * localValue;

                  // Gradient of TPEKernel
                  RealType factor = faceAreas[f1] * faceAreas[f2];

                  faceCenterGradients[f1] += factor * DK[0];
                  faceCenterGradients[f2] += factor * DK[1];

                  faceNormalGradients[f1] += factor * DK[2];

                  // Gradient of faceAreas[f2]
                  faceAreaGradients[f2] += localValue * faceAreas[f1];
                }
              }
              else {
                for ( auto &newNode: node->Children())
                  nodeQueue.push( &newNode );
              }
            }
          }

          // Gradient of faceAreas[f1]
          faceAreaGradients[f1] += f1Value;

          m_Value += faceAreas[f1] * f1Value;
        }
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::05_Collect" );
        ClusterTree->collectFaceDerivatives( faceBarycenters, faceAreas, faceNormals,
          faceAreaGradients, faceNormalGradients, faceCenterGradients );
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::06_Assemble" );

        // Add to global gradient
        for ( int fIdx = 0; fIdx < m_numFaces; fIdx++ ) {
          std::array<VecType, 3> faceGradients;

          // Area
          faceGradients[0] += faceAreaGradients[fIdx] * areaGradientsPi[fIdx];
          faceGradients[1] += faceAreaGradients[fIdx] * areaGradientsPj[fIdx];
          faceGradients[2] += faceAreaGradients[fIdx] * areaGradientsPk[fIdx];

          // Barycenter
          for ( int i: { 0, 1, 2 } )
            faceGradients[i] += faceCenterGradients[fIdx]  / 3.;

          // Normal
          faceGradients[0] += faceNormalGradients[fIdx] * normalGradientsPi[fIdx];
          faceGradients[1] += faceNormalGradients[fIdx] * normalGradientsPj[fIdx];
          faceGradients[2] += faceNormalGradients[fIdx] * normalGradientsPk[fIdx];

          // Add to global gradient
          for ( int i: { 0, 1, 2 } )
            addXYZCoord( m_Gradient, faceGradients[i], m_Topology.getNodeOfTriangle( fIdx, i ));
        }
      }


      m_oldGradientPoint = Point;
    }
  }

  RealType TPEKernel( const VecType &a, const VecType &b, const VecType &n ) const {
//      ScopeTimer timer( "TPEKernel" );
    VecType offset = a - b;
    return my_pow( dotProduct( n, offset ), m_alpha ) / my_pow( offset.norm(), m_beta );
  }

  RealType DTPEKernel( const VecType &a, const VecType &b, const VecType &n, std::array<VecType, 3> &DK ) const {
//    ScopeTimer timer( "DTPEKernel" );
    VecType offset = a - b;

    RealType DP =  dotProduct( n, offset );
    RealType aPart = my_pow( DP, m_alpha - 1 );
    RealType bPart = my_pow( offset.norm(), m_beta );

    DK[0] = (m_alpha * aPart) * n - (m_beta * aPart * DP / offset.squaredNorm()) * offset;
    DK[0] /= bPart;

    DK[1] = -1. * DK[0];

    DK[2] = (m_alpha * aPart / bPart) * offset ;

    return aPart * DP / bPart;
  }

  static RealType my_pow( RealType x, int n ) {
    RealType r = 1.0;

    while ( n > 0 ) {
      r *= x;
      --n;
    }

    return r;
  }
};
}