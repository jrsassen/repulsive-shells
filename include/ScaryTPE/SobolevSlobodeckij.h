#pragma once

#include <goast/Core.h>
#include <goast/Optimization/Functionals.h>

#include <Repulsion.h>

namespace ScaryTPE {
template<typename ConfiguratorType=DefaultConfigurator>
class SurfaceSobolevSlobodeckijOperator : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  std::unique_ptr<Repulsion::SimplicialMesh<2, 3, RealType, int, RealType, RealType>> mesh;
  std::unique_ptr<Repulsion::Metric_FMM<2, 3, 0, RealType, int, RealType, RealType>> metric;
  bool m_lowerOrder;
  bool m_higherOrder;

  const int m_numVertices;

public:
  SurfaceSobolevSlobodeckijOperator( const MeshTopologySaver &Topology,
                                     const VectorType &Geometry,
                                     const RealType alpha,
                                     const RealType beta,
                                     const bool useLowerOrder = false,
                                     const bool useHigherOrder = true,
                                     const RealType innerWeight = 1.,
                                     const bool useAdaptivity = false,
                                     const RealType theta = 0.5,
                                     const RealType thetaNear = 10.,
                                     const int maxLevel = 10 ) : m_lowerOrder( useLowerOrder ),
                                                                 m_higherOrder( useHigherOrder ),
                                                                 m_numVertices( Topology.getNumVertices() ) {
    if ( useLowerOrder || useHigherOrder ) {
      VecType p;

      // Topology for Repulsion library
      std::vector<int> Rfaces( 3 * Topology.getNumFaces());
      std::vector<double> Rvertices( 3 * m_numVertices );

      for ( int i = 0; i < Topology.getNumVertices(); i++ ) {
        getXYZCoord( Geometry, p, i );
        for ( const int d: { 0, 1, 2 } )
          Rvertices[i * 3 + d] = p[d];
      }

      for ( int i = 0; i < Topology.getNumFaces(); i++ ) {
        for ( const int d: { 0, 1, 2 } )
          Rfaces[i * 3 + d] = Topology.getNodeOfTriangle( i, d );
      }

      int thread_count = 1;

      mesh = std::make_unique<Repulsion::SimplicialMesh<2, 3, RealType, int, RealType, RealType>>( Rvertices.data(),
                                                                                                   m_numVertices,
                                                                                                   Rfaces.data(),
                                                                                                   Topology.getNumFaces(),
                                                                                                   thread_count );

      mesh->cluster_tree_settings.split_threshold = 2;
      mesh->cluster_tree_settings.thread_count = 1;
      mesh->block_cluster_tree_settings.far_field_separation_parameter = theta;
      mesh->block_cluster_tree_settings.near_field_separation_parameter = thetaNear;
      mesh->block_cluster_tree_settings.threads_available = 1;

      Repulsion::AdaptivitySettings adaptivity_settings = Repulsion::AdaptivityDefaultSettings;
      adaptivity_settings.theta = thetaNear;
      adaptivity_settings.max_level = maxLevel;

      if ( m_lowerOrder || m_higherOrder ) {
        if ( useAdaptivity )
          metric = std::make_unique<Repulsion::TangentPointMetric_FMM_Adaptive<2, 3, 0, RealType, int, RealType, RealType>>(
                  mesh->GetBlockClusterTree(), alpha, beta, innerWeight, adaptivity_settings );
        else
          metric = std::make_unique<Repulsion::TangentPointMetric_FMM<2, 3, 0, RealType, int, RealType, RealType>>(
                  mesh->GetBlockClusterTree(), alpha, beta, innerWeight );
      }
    }
  }

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    assert( Arg.size() == 3 * m_numVertices &&
            "SurfaceSobolevSlobodeckijOperator(): wrong size of Point" );

    std::vector<double> Rin( 3 * m_numVertices );
    std::vector<double> Rout( 3 * m_numVertices, 0. );
    Dest.resize( 3 * m_numVertices );

    // Col major to row major
    for ( int i = 0; i < m_numVertices; i++ )
      for ( const int d: { 0, 1, 2 } )
        Rin[i * 3 + d] = Arg[d * m_numVertices + i];

    if ( m_lowerOrder )
      metric->Multiply_DenseMatrix( 1., Rin.data(), 1., Rout.data(), 3, Repulsion::KernelType::LowOrder );
    if ( m_higherOrder )
      metric->Multiply_DenseMatrix( 1., Rin.data(), 1., Rout.data(), 3, Repulsion::KernelType::HighOrder );

    // Row major to col major
    for ( int i = 0; i < m_numVertices; i++ )
      for ( const int d: { 0, 1, 2 } )
        Dest[d * m_numVertices + i] = Rout[i * 3 + d];
  }

  int rows() const override {
    return 3 * m_numVertices;
  }

  int cols() const override {
    return 3 * m_numVertices;
  }

};

template<typename ConfiguratorType=DefaultConfigurator>
class SurfaceSobolevSlobodeckijOperatorMap  : public MapToLinOp<ConfiguratorType> {
protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const RealType m_alpha, m_beta;
  const MeshTopologySaver &m_Topology;
  const RealType m_innerWeight;
  const bool m_lowerOrder = false;
  const bool m_higherOrder = true;
  const bool m_useAdaptivity;
  const RealType m_theta, m_thetaNear;
  const int m_maxRefinementLevel;

public:
  SurfaceSobolevSlobodeckijOperatorMap( const MeshTopologySaver &Topology,
                                        RealType alpha,
                                        RealType beta,
                                        const bool useLowerOrder = false,
                                        const bool useHigherOrder = true,
                                        RealType innerWeight = 1.,
                                        const bool useAdaptivity = false,
                                        const RealType theta = 0.5,
                                        const RealType thetaNear = 10.,
                                        const int maxLevel = 10 ) : m_alpha( alpha ),
                                                                    m_beta( beta ),
                                                                    m_Topology( Topology ),
                                                                    m_innerWeight( innerWeight ),
                                                                    m_lowerOrder( useLowerOrder ),
                                                                    m_higherOrder( useHigherOrder ),
                                                                    m_useAdaptivity( useAdaptivity ),
                                                                    m_theta( theta ),
                                                                    m_thetaNear( thetaNear ),
                                                                    m_maxRefinementLevel( maxLevel ) {}

  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    return std::make_unique<SurfaceSobolevSlobodeckijOperator<ConfiguratorType>>( m_Topology, Point, m_alpha, m_beta,
                                                                                  m_lowerOrder, m_higherOrder,
                                                                                  m_innerWeight, m_useAdaptivity,
                                                                                  m_theta, m_thetaNear,
                                                                                  m_maxRefinementLevel );
  }
};
}