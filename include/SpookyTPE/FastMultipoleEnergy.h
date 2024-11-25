#pragma once

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>

#include "ClusterTree.h"
#include "BarnesHutEnergy.h"

#pragma omp declare reduction (vtv_add : std::vector<DefaultConfigurator::VecType> : std::transform(omp_in.cbegin(), omp_in.cend(), omp_out.cbegin(), omp_out.begin(), std::plus<>{}))
#pragma omp declare reduction (rtv_add : std::vector<DefaultConfigurator::RealType> : std::transform(omp_in.cbegin(), omp_in.cend(), omp_out.cbegin(), omp_out.begin(), std::plus<>{}))
#pragma omp declare reduction (vtav_add : std::vector<std::array<DefaultConfigurator::VecType, 3>> : std::transform(omp_in.cbegin(), omp_in.cend(), omp_out.cbegin(), omp_out.begin(), [](const std::array<DefaultConfigurator::VecType, 3> &a, const std::array<DefaultConfigurator::VecType, 3> &b){std::array<DefaultConfigurator::VecType, 3> ret; ret[0]=a[0]+b[0]; ret[1]=a[1]+b[1]; ret[2]=a[2]+b[2]; return ret; }))


namespace SpookyTPE {
template<typename ConfiguratorType=DefaultConfigurator>
class FastMultipoleTangentPointEnergy : public ObjectiveFunctional<ConfiguratorType>,
                                        public TimedClass<FastMultipoleTangentPointEnergy<ConfiguratorType>> {
public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  using typename TimedClass<FastMultipoleTangentPointEnergy<ConfiguratorType>>::ScopeTimer;

  using NodeType = TangentPointNode<ConfiguratorType>;
  using NodePair = std::pair<NodeType *, NodeType *>;

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
  mutable std::unique_ptr<NodeType> m_ClusterTree;

public:
  FastMultipoleTangentPointEnergy( const MeshTopologySaver &Topology,
                                   const int alpha,
                                   const int beta,
                                   RealType innerWeight = 1.,
                                   RealType theta = 0.25 ) : m_numVertices( Topology.getNumVertices() ),
                                                             m_alpha( alpha ),
                                                             m_beta( beta ),
                                                             m_thetaSquared( theta * theta ),
                                                             m_innerWeight( innerWeight ),
                                                             m_Topology( Topology ),
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

    if ( !m_ClusterTree ) {
      std::vector<VecType> faceBarycenters( m_numFaces );
      VectorType BarycenterVector( 3 * m_numFaces );
      for ( int fIdx = 0; fIdx < m_numFaces; fIdx++ ) {
        // get indices of vertices
        std::array<int, 3> verts{};
        std::array<VecType, 3> nodes;
        for ( int j: { 0, 1, 2 } ) {
          verts[j] = m_Topology.getNodeOfTriangle( fIdx, j );
          getXYZCoord<VectorType, VecType>( Point, nodes[j], verts[j] );
        }

        for ( int i: { 0, 1, 2 } ) {
          faceBarycenters[fIdx] += nodes[i];
        }
        faceBarycenters[fIdx] /= 3.;
        setXYZCoord( BarycenterVector, faceBarycenters[fIdx], fIdx );
      }

      {
        ScopeTimer innerTimer( "evaluateGradient::ClusterTree" );
        m_ClusterTree = std::make_unique<LongestAxisTree<NodeType>>( BarycenterVector, 8 );
      }
    }

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

    std::vector<VecType> faceBarycenters( m_numFaces );
    VectorType BarycenterVector( 3 * m_numFaces );
    for ( int fIdx = 0; fIdx < m_numFaces; fIdx++ ) {
      // get indices of vertices
      std::array<int, 3> verts{};
      std::array<VecType, 3> nodes;
      for ( int j: { 0, 1, 2 } ) {
        verts[j] = m_Topology.getNodeOfTriangle( fIdx, j );
        getXYZCoord<VectorType, VecType>( Point, nodes[j], verts[j] );
      }

      for ( int i: { 0, 1, 2 } ) {
        faceBarycenters[fIdx] += nodes[i];
      }
      faceBarycenters[fIdx] /= 3.;
      setXYZCoord( BarycenterVector, faceBarycenters[fIdx], fIdx );
    }

    {
      ScopeTimer innerTimer( "evaluateGradient::ClusterTree" );
      m_ClusterTree = std::make_unique<LongestAxisTree<NodeType>>( BarycenterVector, 8 );
    }

    evaluateTangentPointGradients( Point );

    Gradient = m_innerWeight * m_Gradient;
  }

  void resetCache() const {
    m_oldPoint.setZero();
    m_oldGradientPoint.setZero();
    m_ClusterTree.reset();
  }

protected:

  void evaluateTangentPointGradients( const VectorType &Point ) const {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateTPE: wrong number of dofs!" );

    if (( m_oldGradientPoint - Point ).norm() < 1.e-8 )
      return;
    else {
      ScopeTimer timer( "evaluateTangentPointGradients" );
      m_Value = 0.;
      m_Gradient.setZero();

      std::vector<VecType> faceNormals( m_numFaces );
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

      // Precompute areas, normals, barycenters and their gradients
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
          }
          faceBarycenters[fIdx] /= 3.;
          setXYZCoord( BarycenterVector, faceBarycenters[fIdx], fIdx );
        }
      }

      // Compute cluster tree
//      {
//        ScopeTimer innerTimer( "evaluateTangentPointGradients::02_ClusterTree" );
//        m_ClusterTree = std::make_unique<LongestAxisTree<NodeType>>( BarycenterVector, 8 );
//      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::03_ComputeGQ" );
        m_ClusterTree->computeGeometricQuantities( faceBarycenters, faceAreas, faceNormals );
      }

      std::vector<NodePair> FarFieldNodes, NearFieldNodes;

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::04_DetermineFaNF" );

        std::queue<NodePair> pairQueue;
        pairQueue.emplace( m_ClusterTree.get(), m_ClusterTree.get());

        while ( !pairQueue.empty()) {
          // Get the next node
          NodeType *n1, *n2;
          std::tie(n1, n2) = pairQueue.front();

          // Remove the currently visited node from the queue
          pairQueue.pop();

          // Bounding box distance
          VecType d;
          for ( int k: { 0, 1, 2 } )
            d[k] = std::max( 0., std::max( n1->lowerBounds()[k], n2->lowerBounds()[k] ) -
                                 std::min( n1->upperBounds()[k], n2->upperBounds()[k] ));

          RealType R2 = d.squaredNorm();
          RealType h2 = std::max( n1->squaredDiameter(), n2->squaredDiameter());

          if ( h2 < m_thetaSquared * R2 ) {
            FarFieldNodes.emplace_back(n1, n2);
          }
          else {
            if ( n1->Children().empty() && n2->Children().empty()) {
              NearFieldNodes.emplace_back(n1, n2);
            }
            else if ( n1->Children().empty()) {
              for ( auto &newNode2: n2->Children())
                pairQueue.emplace( n1, &newNode2 );
            }
            else if ( n2->Children().empty()) {
              for ( auto &newNode1: n1->Children())
                pairQueue.emplace( &newNode1, n2 );
            }
            else {
              for ( auto &newNode1: n1->Children())
                for ( auto &newNode2: n2->Children())
                  pairQueue.emplace( &newNode1, &newNode2 );
            }
          }
        }
      }

      std::vector<RealType> faceAreaGradients( m_numFaces, 0. );
      std::vector<VecType> faceNormalGradients( m_numFaces );
      std::vector<VecType> faceCenterGradients( m_numFaces );
      std::vector<std::array<VecType, 3>> faceNodalGradients( m_numFaces );
      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::05_ComputeFF" );

        for ( auto &nodes: FarFieldNodes ) {
          NodeType *&n1 = nodes.first;
          NodeType *&n2 = nodes.second;

          std::array<VecType, 3> DK;
          RealType localValue = DTPEKernel( n1->Centroid(), n2->Centroid(), n1->Normal(), DK );

          RealType factor = n1->Area() * n2->Area();
          n1->CentroidDerivativeFactor() += factor * DK[0];
          n2->CentroidDerivativeFactor() += factor * DK[1];
          n1->NormalDerivativeFactor() += factor * DK[2];

          n2->AreaDerivativeFactor() += localValue * n1->Area();
          n1->AreaDerivativeFactor() += localValue * n2->Area();

          m_Value += factor * localValue;
        }
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::06_ComputeNF" );
#pragma omp parallel for default(shared) reduction(rtv_add:faceAreaGradients) reduction(vtv_add:faceNormalGradients) reduction(vtv_add:faceCenterGradients) reduction(+:m_Value)
        for ( long i = 0; i < NearFieldNodes.size(); i++ ) {
          if ( faceAreaGradients.empty() ) {
            faceAreaGradients.resize( m_numFaces, 0 );
            faceNormalGradients.resize( m_numFaces );
            faceCenterGradients.resize( m_numFaces );
            faceNodalGradients.resize( m_numFaces );
          }
          NodeType *&n1 = NearFieldNodes[i].first;
          NodeType *&n2 = NearFieldNodes[i].second;

          for ( const int &f1: n1->Vertices()) {
            RealType f1Value = 0.;
            for ( const int &f2: n2->Vertices()) {
              if ( f1 == f2 )
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

            // Gradient of faceAreas[f1]
            faceAreaGradients[f1] += f1Value;
            m_Value += faceAreas[f1] * f1Value;
          }
        }
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::07_Collect" );
        m_ClusterTree->collectFaceDerivatives( faceBarycenters, faceAreas, faceNormals, faceAreaGradients, faceNormalGradients, faceCenterGradients );
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::08_Assemble" );

        // Add to global gradient
        for ( int fIdx = 0; fIdx < m_numFaces; fIdx++ ) {
          std::array<VecType, 3> &faceGradients = faceNodalGradients[fIdx];

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
      m_oldPoint = Point;
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