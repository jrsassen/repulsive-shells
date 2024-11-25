#pragma once

#include <stack>

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>

#include <PQP.h>

#include "ClusterTree.h"
#include "BarnesHutEnergy.h"

namespace SpookyTPE {

template<typename ConfiguratorType=DefaultConfigurator>
class VirtualTriangleHierarchy {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  std::array<VecType, 3> &m_V;
  std::array<VecType, 3> m_VCoordinates;
  VecType m_Barycenter, m_BarycenterCoordinates;
  RealType m_AreaFactor;
  RealType m_SquaredDiameter;

  int m_Level = 0;
//  int m_ChildId = -1;

  std::stack<int> m_ChildIds;
  int m_FormerChild = -1;

public:
  VirtualTriangleHierarchy( std::array<VecType, 3> &V,
                            VecType Barycenter,
                            RealType SquaredDiameter ) :
      m_V( V ), m_Barycenter( Barycenter ), m_BarycenterCoordinates( 1. / 3. ), m_AreaFactor( 1. ),
      m_SquaredDiameter( SquaredDiameter ) {
    m_ChildIds.push( -1 );
    for ( int d: { 0, 1, 2 } )
      m_VCoordinates[d][d] = 1.;
  }

  void toParent() {
    if ( m_Level > 0 ) {
      int &c = m_ChildIds.top();
      m_FormerChild = c;

      m_AreaFactor *= 4.;
      m_SquaredDiameter *= 4.;

      if (( 0 <= c ) && ( c <= 2 )) {
        for ( int i: { 0, 1, 2 } )
          if ( i != c ) {
            m_V[i] = m_V[c] + 2. * ( m_V[i] - m_V[c] );
            m_VCoordinates[i] = m_VCoordinates[c] + 2. * ( m_VCoordinates[i] - m_VCoordinates[c] );
          }

        m_Barycenter = m_V[c] + 2. * ( m_Barycenter - m_V[c] );
        m_BarycenterCoordinates = m_VCoordinates[c] + 2. * ( m_BarycenterCoordinates - m_VCoordinates[c] );
      }
      else if ( c == 3 ) {
        for ( int i: { 0, 1, 2 } ) {
          m_V[i] = m_Barycenter - 2. * ( m_V[i] - m_Barycenter );
          m_VCoordinates[i] = m_BarycenterCoordinates - 2. * ( m_VCoordinates[i] - m_BarycenterCoordinates );
        }
      }

      m_ChildIds.pop();
      m_Level--;
    }
  }

  void toChild( int c ) {
    assert( c >= 0 && c < 4 && "VirtualTriangleHierarchy::toChild: Invalid Child ID" );
    m_FormerChild = m_ChildIds.top();
    m_ChildIds.push( c );
    m_Level++;

    m_AreaFactor *= 0.25;
    m_SquaredDiameter *= 0.25;

    if ( ( 0 <= c ) && ( c <= 2 ) ) {
      for ( int i: { 0, 1, 2 } )
        if ( i != c ) {
          m_V[i] = m_V[c] + 0.5 * ( m_V[i] - m_V[c] );
          m_VCoordinates[i] = m_VCoordinates[c] + 0.5 * ( m_VCoordinates[i] - m_VCoordinates[c] );
        }

      m_Barycenter = m_V[c] + 0.5 * ( m_Barycenter - m_V[c] );
      m_BarycenterCoordinates = m_VCoordinates[c] + 0.5 * ( m_BarycenterCoordinates - m_VCoordinates[c] );
    }
    else if ( c == 3 ) {
      for ( int i: { 0, 1, 2 } ) {
        m_V[i] = m_Barycenter - 0.5 * ( m_V[i] - m_Barycenter );
        m_VCoordinates[i] = m_BarycenterCoordinates - 0.5 * ( m_VCoordinates[i] - m_BarycenterCoordinates );
      }
    }
  }
  const VecType &Barycenter() const {
    return m_Barycenter;
  }

  const VecType &BarycenterCoordinates() const {
    return m_BarycenterCoordinates;
  }

  const RealType &AreaFactor() const {
    return m_AreaFactor;
  }

  const RealType &SquaredDiameter() const {
    return m_SquaredDiameter;
  }

  int Level() const {
    return m_Level;
  }

  const int &CurrentChild() const {
    return m_ChildIds.top();
  }

  const int &FormerChild() const {
    return m_FormerChild;
  }
};

template<typename ConfiguratorType=DefaultConfigurator>
class AdaptiveFastMultipoleTangentPointEnergy :
    public ObjectiveFunctional<ConfiguratorType>,
    public TimedClass<AdaptiveFastMultipoleTangentPointEnergy<ConfiguratorType>> {
public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  using typename TimedClass<AdaptiveFastMultipoleTangentPointEnergy<ConfiguratorType>>::ScopeTimer;

  using NodeType = TangentPointNode<ConfiguratorType>;
  using NodePair = std::pair<NodeType *, NodeType *>;

  const int m_numVertices;
  const int m_alpha, m_beta;
  const RealType m_thetaSquared;
  const RealType m_thetaNearSquared;
  const RealType m_innerWeight;
  const int m_maxRefinementLevel;

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
  AdaptiveFastMultipoleTangentPointEnergy( const MeshTopologySaver &Topology,
                                           int alpha,
                                           int beta,
                                           RealType innerWeight = 1.,
                                           RealType theta = 0.25,
                                           RealType thetaNear = 10.,
                                           int maxLevel = 10 ) : m_numVertices( Topology.getNumVertices() ),
                                                                 m_alpha( alpha ),
                                                                 m_beta( beta ),
                                                                 m_thetaSquared( theta * theta ),
                                                                 m_thetaNearSquared( thetaNear * thetaNear ),
                                                                 m_innerWeight( innerWeight ),
                                                                 m_maxRefinementLevel( maxLevel ),
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

    Value = m_Value;
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

    Gradient = m_Gradient;
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
      VectorType squaredFaceDiameters( m_numFaces );
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

          // Barycenters and bounding boxes
          for ( int i: { 0, 1, 2 } ) {
            faceBarycenters[fIdx] += nodes[i];

            for ( int d: { 0, 1, 2 } ) {
              faceLowerBounds[fIdx][d] = std::min( faceLowerBounds[fIdx][d], nodes[i][d] );
              faceUpperBounds[fIdx][d] = std::max( faceUpperBounds[fIdx][d], nodes[i][d] );
            }
          }
          faceBarycenters[fIdx] /= 3.;
          setXYZCoord( BarycenterVector, faceBarycenters[fIdx], fIdx );

          // (Squared) radii of triangles
          RealType a = EdgeLengths[m_Topology.getEdgeOfTriangle( fIdx, 0 )];
          RealType b = EdgeLengths[m_Topology.getEdgeOfTriangle( fIdx, 1 )];
          RealType c = EdgeLengths[m_Topology.getEdgeOfTriangle( fIdx, 2 )];

          squaredFaceDiameters[fIdx] = my_pow( std::max( { a, b, c } ), 2 );
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

      std::vector<RealType> faceAreaGradients( m_numFaces, 0. );
      std::vector<VecType> faceNormalGradients( m_numFaces );
      std::vector<VecType> faceCenterGradients( m_numFaces );
      std::vector<std::array<VecType, 3>> faceNodalGradients( m_numFaces );

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::04_ComputeFM" );

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
            // ScopeTimer innerInnerTimer( "evaluateTangentPointGradients::04_ComputeFM_FField" );
            // Far field computation for block clusters fulfilling the MAC
            std::array<VecType, 3> DK;
            RealType localValue = DTPEKernel( n1->Centroid(), n2->Centroid(), n1->Normal(), DK );

            RealType factor = n1->Area() * n2->Area();
            n1->CentroidDerivativeFactor() += factor * DK[0];
            n2->CentroidDerivativeFactor() += factor * DK[1];
            n1->NormalDerivativeFactor() += factor * DK[2];

            n2->AreaDerivativeFactor() += localValue *  n1->Area();
            n1->AreaDerivativeFactor() += localValue *  n2->Area();

            m_Value += factor * localValue;
          }
          else {
            if ( n1->Children().empty() && n2->Children().empty()) {
              // ScopeTimer innerInnerTimer( "evaluateTangentPointGradients::04_ComputeFM_NField" );
              // Near field computations
              for ( const int &f1: n1->Vertices()) {
                RealType f1Value = 0.;
                for ( const int &f2: n2->Vertices()) {
                  if ( f1 >= f2 )
                    continue; // exploit symmetry

                  // Collect incident vertices
                  int v1_1 = m_Topology.getNodeOfTriangle( f1, 0 );
                  int v1_2 = m_Topology.getNodeOfTriangle( f1, 1 );
                  int v1_3 = m_Topology.getNodeOfTriangle( f1, 2 );
                  int v2_1 = m_Topology.getNodeOfTriangle( f2, 0 );
                  int v2_2 = m_Topology.getNodeOfTriangle( f2, 1 );
                  int v2_3 = m_Topology.getNodeOfTriangle( f2, 2 );

                  // check if triangles are neighbors
                  if ( v1_1 != v2_1 && v1_1 != v2_2 && v1_1 != v2_3 &&
                       v1_2 != v2_1 && v1_2 != v2_2 && v1_2 != v2_3 &&
                       v1_3 != v2_1 && v1_3 != v2_2 && v1_3 != v2_3 ) {
                    // Extract vertex positions in format usable by PQP
                    std::array<VecType, 3> V1, V2;

                    for ( int i: { 0, 1, 2 } ) {
                      getXYZCoord( Point, V1[i], m_Topology.getNodeOfTriangle( f1, i ));
                      getXYZCoord( Point, V2[i], m_Topology.getNodeOfTriangle( f2, i ));
                    }

                    const RealType *pV1[3] = { V1[0].data(), V1[1].data(), V1[2].data() };
                    const RealType *pV2[3] = { V2[0].data(), V2[1].data(), V2[2].data() };

                    // Compute triangle-triangle distance
                    VecType p, q; // closest points, actually not needed but produced by PQP
                    RealType dT = TriDist( p.data(), q.data(), pV1, pV2 );

                    // Finish if there is an intersection
                    if ( dT == 0. ) {
                      std::cerr << "AdaptiveFastMultipoleTangentPointEnergy: intersection detected" << std::endl;
                      m_Value = std::numeric_limits<RealType>::infinity();
                      m_Gradient.setZero();

                      m_oldGradientPoint = Point;
                      m_oldPoint = Point;
                      return;
                    }

                    if ( std::max( squaredFaceDiameters[f1], squaredFaceDiameters[f2] ) < m_thetaNearSquared * dT * dT ) {
                      // Triangles fullfill the relaxed MAC -> compute TPE (+derivative) using standard midpoint scheme
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

                      localValue = DTPEKernel( faceBarycenters[f2], faceBarycenters[f1], faceNormals[f2], DK );

                      f1Value += faceAreas[f2] * localValue;

                      // Gradient of TPEKernel
                      faceCenterGradients[f1] += factor * DK[1];
                      faceCenterGradients[f2] += factor * DK[0];
                      faceNormalGradients[f2] += factor * DK[2];

                      // Gradient of faceAreas[f2]
                      faceAreaGradients[f2] += localValue * faceAreas[f1];
                    }
                    else {
                      // ScopeTimer inner3Timer( "evaluateTangentPointGradients::04_ComputeFM_NField_A" );
                      // Triangles do not fulfill the relaxed MAC -> create and traverse a hierarchical subdivision
                      // Create virtual hierarchy
                      VirtualTriangleHierarchy S( V1, faceBarycenters[f1], squaredFaceDiameters[f1] );
                      VirtualTriangleHierarchy T( V2, faceBarycenters[f2], squaredFaceDiameters[f2] );

                      // Helper function to evaluate to energy at the current levels of the virtual hierarchy
                      // (for not having to write it twice)
                      auto evaluateEnergy = [&]() {
                        std::array<VecType, 3> DK;
                        RealType localValue = S.AreaFactor() * T.AreaFactor() *
                                              DTPEKernel( S.Barycenter(), T.Barycenter(), faceNormals[f1], DK );

                        f1Value += faceAreas[f2] * localValue;

                        // Gradient of TPEKernel
                        RealType factor = S.AreaFactor() * faceAreas[f1] * T.AreaFactor() * faceAreas[f2];

                        for ( int i: { 0, 1, 2 } )
                          faceNodalGradients[f1][i] += factor * S.BarycenterCoordinates()[i] * DK[0];
                        for ( int i: { 0, 1, 2 } )
                          faceNodalGradients[f2][i] += factor * T.BarycenterCoordinates()[i] * DK[1];

                        faceNormalGradients[f1] += factor * DK[2];

                        // Gradient of faceAreas[f2]
                        faceAreaGradients[f2] += localValue * faceAreas[f1];

                        localValue = S.AreaFactor() * T.AreaFactor() *
                                     DTPEKernel( T.Barycenter(), S.Barycenter(), faceNormals[f2], DK );

                        f1Value += faceAreas[f2] * localValue;

                        // Gradient of TPEKernel
                        for ( int i: { 0, 1, 2 } )
                          faceNodalGradients[f1][i] += factor * S.BarycenterCoordinates()[i] * DK[1];
                        for ( int i: { 0, 1, 2 } )
                          faceNodalGradients[f2][i] += factor * T.BarycenterCoordinates()[i] * DK[0];

                        faceNormalGradients[f2] += factor * DK[2];

                        // Gradient of faceAreas[f2]
                        faceAreaGradients[f2] += localValue * faceAreas[f1];
                      };

                      // Traverse hierarchy
                      bool from_above = true;

                      S.toChild( 0 );
                      T.toChild( 0 );

                      while ( true ) {
                        if ( from_above ) {
                          if ( S.Level() >= m_maxRefinementLevel ) {
//                            std::cerr << " ...... reached max level " << std::endl;
                            // If at lowest level and inadmissable then we just compute the energy and move up.
                            evaluateEnergy();
                            S.toParent();
                            T.toParent();
                            from_above = false;
                          }
                          else {
                            // If not at lowest level, then we have to check for admissability.
                            dT = TriDist( p.data(), q.data(), pV1, pV2 );

                            if ( std::max( S.SquaredDiameter(), T.SquaredDiameter() ) < m_thetaNearSquared * dT * dT ) {
                              // We compute energy, go to parent, and prepare the next child of the parent.
                              evaluateEnergy();
                              S.toParent();
                              T.toParent();
                              from_above = false;
                            }
                            else {
                              // If inadmissible, we go a level deeper.
                              S.toChild( 0 );
                              T.toChild( 0 );
                              from_above = true;
                            }
                          }
                        }
                        else {
                          // If we come from below, we have to find the next pair of simplices to visit.
                          int S_k = S.FormerChild();
                          int T_k = T.FormerChild();

                          if ( T_k < 3 ) {
                            S.toChild( S_k );
                            T.toChild( T_k + 1 );
                            from_above = true;
                          }
                          else {
                            if ( S_k < 3 ) {
                              S.toChild( S_k + 1 );
                              T.toChild( 0 );
                              from_above = true;
                            }
                            else {
                              // No further unvisited children. Either move up or break.
                              if ( S.Level() == 0 ) {
                                break;
                              }

                              S.toParent();
                              T.toParent();
                              from_above = false;
                            }
                          }

                        } // if( from_above )
                      }
                      // dT = TriDist( p.data(), q.data(), pV1, pV2 );
                      // std::max( faceRadii[f1], faceRadii[f2] ) < m_thetaFarSquared * dT * dT
                    }
                  }
                  else {
                    // Triangles are neighbors -> compute TPE (+derivative) using standard midpoint scheme
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

                    localValue = DTPEKernel( faceBarycenters[f2], faceBarycenters[f1], faceNormals[f2], DK );

                    f1Value += faceAreas[f2] * localValue;

                    // Gradient of TPEKernel
                    faceCenterGradients[f1] += factor * DK[1];
                    faceCenterGradients[f2] += factor * DK[0];
                    faceNormalGradients[f2] += factor * DK[2];

                    // Gradient of faceAreas[f2]
                    faceAreaGradients[f2] += localValue * faceAreas[f1];
                  }
                }

                m_Value += faceAreas[f1] * f1Value;
                // Gradient of faceAreas[f1]
                faceAreaGradients[f1] += f1Value;
}
            }
            else if ( n1->Children().empty() ) {
              for ( auto &newNode2: n2->Children() )
                pairQueue.emplace( n1, &newNode2 );
            }
            else if ( n2->Children().empty() ) {
              for ( auto &newNode1: n1->Children() )
                pairQueue.emplace( &newNode1, n2 );
            }
            else {
              for ( auto &newNode1: n1->Children() )
                for ( auto &newNode2: n2->Children() )
                  pairQueue.emplace( &newNode1, &newNode2 );
            }
          }
        }
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::05_Collect" );
        // Backpropagation of partial deriatives w.r.t. cluster properties to partial derivatives w.r.t. face properties
        m_ClusterTree->collectFaceDerivatives( faceBarycenters, faceAreas, faceNormals,
                                             faceAreaGradients, faceNormalGradients, faceCenterGradients );
      }

      {
        ScopeTimer innerTimer( "evaluateTangentPointGradients::06_Assemble" );

        // Assemble gradient w.r.t. nodal positions from partial derivatives w.r.t. face properties
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
            addXYZCoord( m_Gradient, m_innerWeight * faceGradients[i], m_Topology.getNodeOfTriangle( fIdx, i ) );
        }
      }

      // Weighting of energy
      m_Value *= m_innerWeight;

      // Cache point of evaluation
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
