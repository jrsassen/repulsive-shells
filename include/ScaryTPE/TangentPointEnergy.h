#pragma once

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>

#include <Repulsion.h>


namespace ScaryTPE {
template<typename ConfiguratorType=DefaultConfigurator>
class TangentPointEnergy : public ObjectiveFunctional<ConfiguratorType> {
public:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const int m_numVertices;
  const RealType m_alpha, m_beta;
  const RealType m_theta;
  const RealType m_thetaNear;
  const RealType m_innerWeight;
  const bool m_useAdaptivity;
  const int m_maxRefinementLevel;

  const MeshTopologySaver &m_Topology;
  std::vector<int> m_Faces;
  int m_numFaces;

  const int m_numLocalDofs;

  // TPE cache
  mutable VectorType m_oldPoint;
  mutable RealType m_Value;
  mutable VectorType m_Gradient;

public:
  TangentPointEnergy( const MeshTopologySaver &Topology,
                      const RealType alpha,
                      const RealType beta,
                      const RealType innerWeight = 1.,
                      const bool useAdaptivity = false,
                      const RealType theta = 0.5,
                      const RealType thetaNear = 10.,
                      const int maxLevel = 10 ) : m_numVertices( Topology.getNumVertices() ),
                                                  m_alpha( alpha ),
                                                  m_beta( beta ),
                                                  m_theta( theta ),
                                                  m_thetaNear( thetaNear ),
                                                  m_innerWeight( innerWeight ),
                                                  m_useAdaptivity( useAdaptivity ),
                                                  m_maxRefinementLevel( maxLevel ),
                                                  m_Topology( Topology ),
                                                  m_numLocalDofs( 3 * Topology.getNumVertices() ) {
    m_oldPoint = VectorType::Zero( 3 * m_numVertices );
    m_Gradient = VectorType::Zero( 3 * m_numVertices );

    // Topology for Repulsion library
    m_Faces.resize( 3 * m_Topology.getNumFaces());
    for ( int i = 0; i < m_Topology.getNumFaces(); i++ ) {
      for ( const int d: { 0, 1, 2 } )
        m_Faces[i * 3 + d] = m_Topology.getNodeOfTriangle( i, d );
    }
    m_numFaces = m_Topology.getNumFaces();
  }

  void evaluate( const VectorType &Point, RealType &Value ) const override {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluate: wrong number of dofs!" );

    evaluateTangentPointEnergiesAndGradients( Point );

    Value = m_Value;
  }

  void evaluateGradient( const VectorType &Point, VectorType &Gradient ) const override {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateGradient: wrong number of dofs!" );

    if ( Gradient.size() != Point.size() )
      Gradient.resize( Point.size() );

    Gradient.setZero();

    evaluateTangentPointEnergiesAndGradients( Point );

    Gradient = m_Gradient;
  }

  void resetCache() const {
    m_oldPoint.setZero();
  }

protected:
  void evaluateTangentPointEnergiesAndGradients( const VectorType &Point ) const {
    if ( Point.size() != m_numLocalDofs )
      throw std::length_error( "TangentPointDifferenceEnergy::evaluateTPE: wrong number of dofs!" );

    if ( ( m_oldPoint - Point ).norm() < 1.e-8 )
      return;

    // X.Y.Z. -> XYZ.
    VecType p;
    std::vector<double> Rvertices( 3 * m_numVertices );
    for ( int i = 0; i < m_numVertices; i++ ) {
      getXYZCoord( Point, p, i );
      for ( int d: { 0, 1, 2 } )
        Rvertices[i * 3 + d] = p[d];
    }

    // Create mesh
    auto mesh = std::make_unique<Repulsion::SimplicialMesh<2, 3, RealType, int, RealType, RealType>>(
      Rvertices.data(),
      m_numVertices,
      m_Faces.data(),
      m_Topology.getNumFaces(),
      1 );

    mesh->cluster_tree_settings.split_threshold = 2;
    mesh->cluster_tree_settings.thread_count = 1;
    mesh->block_cluster_tree_settings.far_field_separation_parameter = m_theta;
    mesh->block_cluster_tree_settings.near_field_separation_parameter = m_thetaNear;
    mesh->block_cluster_tree_settings.threads_available = 1;

    Repulsion::AdaptivitySettings adaptivity_settings = Repulsion::AdaptivityDefaultSettings;
    adaptivity_settings.theta = m_thetaNear;
    adaptivity_settings.max_level = m_maxRefinementLevel;

    std::unique_ptr<Repulsion::Energy_FMM<2, 3, 0, RealType, int, RealType, RealType>> tpe;

    if ( m_useAdaptivity )
      tpe = std::make_unique<Repulsion::TangentPointEnergy_FMM_Adaptive<2, 3, 0, RealType, int, RealType, RealType>>(
        m_alpha, m_beta, m_innerWeight, adaptivity_settings );
    else
      tpe = std::make_unique<Repulsion::TangentPointEnergy_FMM<2, 3, 0, RealType, int, RealType, RealType>>(
        m_alpha, m_beta, m_innerWeight );

    // Evaluate gradient and energy
    FullMatrixType GSGrad = FullMatrixType::Zero( 3, m_numVertices );
    m_Value = tpe->Differential( *mesh, GSGrad.data(), false );

    // XYZ. -> X.Y.Z.
    for ( int d: { 0, 1, 2 } )
      m_Gradient.segment( d * m_numVertices, m_numVertices ) = GSGrad.row( d );

    m_oldPoint = Point;
  }
};
}