#include <MeshIO.h>
#include <utility>

#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <goast/Core.h>
#include <goast/NRIC.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>
#include <goast/external/vtkIO.h>

#include "ScaryTPE/TangentPointEnergy.h"
#include "ScaryTPE/SobolevSlobodeckij.h"

#include "SpookyTPE/FastMultipoleEnergy.h"
#include "SpookyTPE/AdaptiveEnergy.h"

#include "Optimization/TrustRegionNewton.h"
#include "Optimization/LineSearchNewtonCG.h"
#include "Optimization/PreconditionedGD.h"

#include "NRICReferenceMembraneFunctional.h"
#include "NRICReferenceBendingFunctional.h"
#include "SoftPointConstraint.h"
#include "HessianMetric.h"
#include "MatrixOperatorMapWrapper.h"

using VectorType = DefaultConfigurator::VectorType;
using MatrixType = DefaultConfigurator::SparseMatrixType;
using VecType = DefaultConfigurator::VecType;
using RealType = DefaultConfigurator::RealType;

using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator> >;

template<typename VectorType>
VectorType edgeToNodeVector( const MeshTopologySaver &Topol, const VectorType &edgeValues, bool twoEdgeValues = true ) {
  VectorType nodalValues( Topol.getNumVertices() );
  nodalValues.setZero();
  for ( int edgeIdx = 0; edgeIdx < Topol.getNumEdges(); edgeIdx++ ) {
    int v0 = Topol.getAdjacentNodeOfEdge( edgeIdx, 0 );
    int v1 = Topol.getAdjacentNodeOfEdge( edgeIdx, 1 );

    nodalValues[v0] += edgeValues[edgeIdx] / 2;
    nodalValues[v1] += edgeValues[edgeIdx] / 2;

    if ( twoEdgeValues ) {
      nodalValues[v0] += edgeValues[Topol.getNumEdges() + edgeIdx] / 2;
      nodalValues[v1] += edgeValues[Topol.getNumEdges() + edgeIdx] / 2;
    }
  }
  return nodalValues;
}

template<class RandAccessIter>
double median( RandAccessIter begin, RandAccessIter end ) {
  std::size_t size = end - begin;
  std::size_t middleIdx = size / 2;
  RandAccessIter target = begin + middleIdx;
  std::nth_element( begin, target, end );

  if ( size % 2 != 0 ) { // Odd number of elements
    return *target;
  }

  // Even number of elements
  double a = *target;
  RandAccessIter targetNeighbor = target - 1;
  std::nth_element( begin, targetNeighbor, end );
  return ( a + *targetNeighbor ) / 2.0;
}

int main( int argc, char *argv[] ) {
  //region Config
  enum TPEType {
    SPOOKY,
    SCARY
  };

  std::unordered_map<std::string, TPEType> const TPETypeTable = {
    { "Spooky", TPEType::SPOOKY },
    { "Scary",  TPEType::SCARY },
  };

  enum OptimizationType {
    PRECONDITIONED,
    NEWTON,
    TRUSTREGION
  };

  std::unordered_map<std::string, OptimizationType> const optTypeTable = {
    { "Preconditioned", OptimizationType::PRECONDITIONED },
    { "TrustRegion",    OptimizationType::TRUSTREGION },
    { "Newton",         OptimizationType::NEWTON },
  };

  enum MetricType {
    POINCARE_DISK,
    FLAT_TORUS
  };

  std::unordered_map<std::string, MetricType> const metricTypeTable = {
    { "PoincareDisk", MetricType::POINCARE_DISK },
    { "Flat",         MetricType::FLAT_TORUS },
  };


  struct {
    std::string outputFolder = "./";
    bool timestampOutput = true;
    std::string outputFilePrefix;

    std::string File;
    std::string initFile;
    std::vector<int> dirichletVertices{ 8, 56 };

    MetricType Metric = POINCARE_DISK;
    RealType hyperbolicRadius = 1.;
    RealType scaleFactor = 1.;
    RealType metricFactor = 1.;

    struct {
      int alpha = 6;
      int beta = 12;
      TPEType Type = SCARY;
      RealType innerWeight = 1.;
      RealType theta = 0.5;
      RealType thetaNear = 10.;
      bool useLowerSSMTerm = false;
      bool useHigherSSMTerm = false;
      bool useAdaptivity = true;
    } TPE;

    struct {
      RealType membraneWeight = 1.;
      RealType bendingWeight = 1.;
      RealType elasticWeight = 1.;
      RealType tpeWeight = 1.e-3;
      RealType dirichletWeight = 1.e-6;
    } Energy;

    struct {
      int maxNumIterations = 50000;
      RealType minStepsize = 1.e-12;
      RealType maxStepsize = 10.;

      OptimizationType Type = NEWTON;
    } Optimization;
  } Config;
  //endregion

  //region Read config
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.File = config["Data"]["File"].as<std::string>();
    Config.initFile = config["Data"]["initFile"].as<std::string>();
    Config.dirichletVertices = config["Data"]["dirichletVertices"].as<std::vector<int>>();

    Config.scaleFactor = config["Data"]["scaleFactor"].as<RealType>();
    Config.metricFactor = config["Data"]["metricFactor"].as<RealType>();

    auto metric_it = metricTypeTable.find( config["Data"]["metricType"].as<std::string>());
    if ( metric_it != metricTypeTable.end())
      Config.Metric = metric_it->second;
    else
      throw std::runtime_error( "Invalid Metric::Type in Config." );

    Config.outputFilePrefix = config["Output"]["outputFilePrefix"].as<std::string>();
    Config.outputFolder = config["Output"]["outputFolder"].as<std::string>();
    Config.timestampOutput = config["Output"]["timestampOutput"].as<bool>();

    Config.TPE.alpha = config["TPE"]["alpha"].as<int>();
    Config.TPE.beta = config["TPE"]["beta"].as<int>();
    Config.TPE.useAdaptivity = config["TPE"]["useAdaptivity"].as<bool>();
    Config.TPE.innerWeight = config["TPE"]["innerWeight"].as<RealType>();
    Config.TPE.theta = config["TPE"]["theta"].as<RealType>();
    Config.TPE.thetaNear = config["TPE"]["thetaNear"].as<RealType>();
    Config.TPE.useLowerSSMTerm = config["TPE"]["useLowerSSMTerm"].as<bool>();
    Config.TPE.useHigherSSMTerm = config["TPE"]["useHigherSSMTerm"].as<bool>();

    if ( auto ttype_it = TPETypeTable.find( config["TPE"]["Type"].as<std::string>()); ttype_it != TPETypeTable.end())
      Config.TPE.Type = ttype_it->second;
    else
      throw std::runtime_error( "Invalid TPE::Type in Config." );

    Config.Energy.membraneWeight = config["Energy"]["membraneWeight"].as<RealType>();
    Config.Energy.bendingWeight = config["Energy"]["bendingWeight"].as<RealType>();
    Config.Energy.elasticWeight = config["Energy"]["elasticWeight"].as<RealType>();
    Config.Energy.tpeWeight = config["Energy"]["tpeWeight"].as<RealType>();
    Config.Energy.dirichletWeight = config["Energy"]["dirichletWeight"].as<RealType>();

    Config.Optimization.maxNumIterations = config["Optimization"]["maxNumIterations"].as<int>();
    Config.Optimization.minStepsize = config["Optimization"]["minStepsize"].as<RealType>();
    Config.Optimization.maxStepsize = config["Optimization"]["maxStepsize"].as<RealType>();

    auto type_it = optTypeTable.find( config["Optimization"]["Type"].as<std::string>());
    if ( type_it != optTypeTable.end())
      Config.Optimization.Type = type_it->second;
    else
      throw std::runtime_error( "Invalid Optimization::Type in Config." );
  }
  //endregion

  //region Output path
  if ( Config.outputFolder.compare( Config.outputFolder.length() - 1, 1, "/" ) != 0 )
    Config.outputFolder += "/";

  std::string execName( argv[0] );
  execName = execName.substr( execName.find_last_of( '/' ) + 1 );

  if ( Config.timestampOutput ) {
    std::time_t t = std::time( nullptr );
    std::stringstream ss;
    ss << Config.outputFolder;
    ss << std::put_time( std::localtime( &t ), "%Y%m%d_%H%M%S" );
    ss << "_" << execName;
    ss << "/";
    Config.outputFolder = ss.str();
    boost::filesystem::create_directory( Config.outputFolder );
  }

  std::string outputPrefix = Config.outputFolder + Config.outputFilePrefix;

  if ( argc == 2 ) {
    boost::filesystem::copy_file( argv[1], Config.outputFolder + "_parameters.conf",
                                  boost::filesystem::copy_options::overwrite_existing );
  }
  //endregion

  //region Logging
  std::ofstream logFile;
  logFile.open( Config.outputFolder + "/_output.log" );

  std::ostream output_cout( std::cout.rdbuf());
  std::ostream output_cerr( std::cerr.rdbuf());

  using TeeDevice = boost::iostreams::tee_device<std::ofstream, std::ostream>;
  using TeeStream = boost::iostreams::stream<TeeDevice>;
  TeeDevice tee_cout( logFile, output_cout );
  TeeDevice tee_cerr( logFile, output_cerr );

  TeeStream split_cout( tee_cout );
  TeeStream split_cerr( tee_cerr );

  std::cout.rdbuf( split_cout.rdbuf());
  std::cerr.rdbuf( split_cerr.rdbuf());
  //endregion

  //region Setup
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

  // Read meshes
  TriMesh Mesh, initMesh;
  OpenMesh::IO::Options ropt;
  ropt += OpenMesh::IO::Options::VertexTexCoord;
  initMesh.request_vertex_texcoords2D();
  Mesh.request_vertex_texcoords2D();
  if ( !OpenMesh::IO::read_mesh( Mesh, Config.File, ropt ))
    throw std::runtime_error( "Failed to read file: " + Config.File );
  if ( !OpenMesh::IO::read_mesh( initMesh, Config.initFile, ropt ))
    throw std::runtime_error( "Failed to read file: " + Config.initFile );

  // Topology of the mesh
  MeshTopologySaver Topology( Mesh );
  int numVertices = Topology.getNumVertices();
  int numEdges = Topology.getNumEdges();

  // Geometry of the meshes
  VectorType Geometry, initGeometry;

  getGeometry( Mesh, Geometry );
  getGeometry( initMesh, initGeometry );

  // Scale mesehs
  if ( (Geometry - initGeometry).norm() < 1.e-8 )
    initGeometry *= Config.scaleFactor;
  Geometry *= Config.scaleFactor;

  // NRIC
  NRICMap<DefaultConfigurator> Z( Topology );
  VectorType NRIC_End = Z( initGeometry );
  //endregion

  //region Basic energies
  // Shell energy + Metrics
  ShellDeformationType W( Topology,  Config.Energy.membraneWeight, Config.Energy.bendingWeight );

  HessianMetric<DefaultConfigurator> ElasticMetric( W, Config.dirichletVertices );
  OperatorHessianMetric<DefaultConfigurator> opElasticMetric( W, Config.dirichletVertices );

  std::cout << " .. W = " << W( Geometry, initGeometry ) << std::endl;

  // Tangent-Point Energy + Fractional Sobolev metric
  ScaryTPE::TangentPointEnergy<DefaultConfigurator> scTPE( Topology,
                                                           Config.TPE.alpha,
                                                           Config.TPE.beta,
                                                           Config.TPE.innerWeight,
                                                           Config.TPE.useAdaptivity,
                                                           Config.TPE.theta,
                                                           Config.TPE.thetaNear );

  SpookyTPE::FastMultipoleTangentPointEnergy<DefaultConfigurator> spTPE( Topology,
                                                                         Config.TPE.alpha,
                                                                         Config.TPE.beta,
                                                                         Config.TPE.innerWeight,
                                                                         Config.TPE.theta );

  SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy<DefaultConfigurator> adspTPE( Topology,
                                                                                   Config.TPE.alpha,
                                                                                   Config.TPE.beta,
                                                                                   Config.TPE.innerWeight,
                                                                                   Config.TPE.theta,
                                                                                   Config.TPE.thetaNear );
  ObjectiveFunctional<DefaultConfigurator> *chosenTPE = nullptr;

  if ( Config.TPE.Type == SCARY ) {
    chosenTPE = &scTPE;
  }
  else if ( Config.TPE.Type == SPOOKY ) {
    if ( Config.TPE.useAdaptivity ) {
      chosenTPE = &adspTPE;
    }
    else {
      chosenTPE = &spTPE;
    }
  }

  ObjectiveWrapper TPE( *chosenTPE );
  ObjectiveGradientWrapper TPG( *chosenTPE );

  ScaryTPE::SurfaceSobolevSlobodeckijOperatorMap<DefaultConfigurator> SSMop( Topology,
                                                                             Config.TPE.alpha, Config.TPE.beta,
                                                                             Config.TPE.useLowerSSMTerm,
                                                                             Config.TPE.useHigherSSMTerm,
                                                                             Config.TPE.innerWeight,
                                                                             Config.TPE.useAdaptivity );
  //endregion

  //region Hyperbolic metric
  VectorType prescribedNRIC = Z( Geometry );
  if ( Config.Metric == POINCARE_DISK ) {
    std::cout << " .. Computing hyperbolic metric..." << std::endl;
    auto hdist = [Config]( const VecType &p, const VecType &q ) {
      RealType temp = 2 * ( p - q ).normSqr() * Config.hyperbolicRadius * Config.hyperbolicRadius;
      temp /= Config.hyperbolicRadius * Config.hyperbolicRadius - p.normSqr();
      temp /= Config.hyperbolicRadius * Config.hyperbolicRadius - q.normSqr();
      return std::acosh( 1 + temp );
    };

    std::vector<int> edgelengthMask;
    for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {
      VecType p, q;
      int e = Topology.getAdjacentNodeOfEdge( edgeIdx, 0 );
      getXYZCoord( Geometry, p, e );
      e = Topology.getAdjacentNodeOfEdge( edgeIdx, 1 );
      getXYZCoord( Geometry, q, e );

      prescribedNRIC[edgeIdx] = hdist( p, q );
      //      std::cout << " .. edge " << edgeIdx << " = " << hdist( p, q ) << std::endl;
      edgelengthMask.push_back( edgeIdx );
    }
  }
  else if (Config.Metric == FLAT_TORUS ) {
    RealType l_v = Config.metricFactor;
    RealType l_h = Config.metricFactor;
    RealType l_d = std::sqrt( l_v * l_v + l_h * l_h );

    prescribedNRIC.tail( numEdges ).array() = 0.;

    std::cout << "Mesh.has_vertex_texcoords1D = " << Mesh.has_vertex_texcoords1D() << std::endl;
    std::cout << "Mesh.has_vertex_texcoords2D = " << Mesh.has_vertex_texcoords2D() << std::endl;
    std::cout << "Mesh.has_vertex_texcoords3D = " << Mesh.has_vertex_texcoords3D() << std::endl;

    for ( int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++ ) {
      prescribedNRIC[numEdges + edgeIdx] = 0.;

      VecType p, q;
      int e = Topology.getAdjacentNodeOfEdge( edgeIdx, 0 );
      getXYZCoord( Geometry, p, e );

      auto &uv1 = Mesh.texcoord2D( initMesh.vertex_handle( e ) );

      e = Topology.getAdjacentNodeOfEdge( edgeIdx, 1 );
      getXYZCoord( Geometry, q, e );

      auto &uv2 = Mesh.texcoord2D( initMesh.vertex_handle( e ) );

      if ( std::fabs( uv1[0] - uv2[0] ) < 1.e-8 ) {
        prescribedNRIC[edgeIdx] = l_v;
      }
      else if ( std::fabs( uv1[1] - uv2[1] ) < 1.e-8 ) {
        prescribedNRIC[edgeIdx] = l_h;
      }
      else {
        prescribedNRIC[edgeIdx] = l_d;
      }
    }
  }

#ifdef GOAST_WITH_VTK
  std::map<std::string, VectorType> WireframeData;
  WireframeData["EdgeLengths"] = prescribedNRIC.head( numEdges );
  WireframeData["DihedralAngles"] = prescribedNRIC.tail( numEdges );
  saveWireframeAsVTP<VectorType>( Topology, Geometry, outputPrefix + "Metric.vtp", WireframeData );
#endif
  // Sanity check
  VectorType triangleInequalityValues = TriangleInequalityTripleOp<DefaultConfigurator>( Topology )( prescribedNRIC );

  if ( triangleInequalityValues.minCoeff() <= 0 ) {
    std::cerr << "WARNING: triangle inequality violated!" << std::endl;
  }

  std::cout << " .. prescribedNRIC - NRIC_End = "
      << ( prescribedNRIC.head( numEdges ) - NRIC_End.head( numEdges ) ).norm()
      << std::endl;
  std::cout << " .. prescribedNRIC - NRIC_End = "
      << ( prescribedNRIC.head( numEdges ) - NRIC_End.head( numEdges ) ).norm() / NRIC_End.head( numEdges ).norm()
      << std::endl;
  //endregion

  //region Functionals
  // Elasticity
  NRICReferenceBendingFunctional<DefaultConfigurator> Fbend( Topology, prescribedNRIC );
  NRICReferenceBendingGradient<DefaultConfigurator> DFbend( Topology, prescribedNRIC );
  NRICReferenceBendingHessian<DefaultConfigurator> D2Fbend( Topology, prescribedNRIC );

  NRICReferenceMembraneFunctional<DefaultConfigurator> Fmem( Topology, prescribedNRIC );
  NRICReferenceMembraneGradient<DefaultConfigurator> DFmem( Topology, prescribedNRIC );
  NRICReferenceMembraneHessian<DefaultConfigurator> D2Fmem( Topology, prescribedNRIC );

  VectorType elasticWeights( 2 );
  elasticWeights << Config.Energy.membraneWeight, Config.Energy.bendingWeight;

  AdditionOp<DefaultConfigurator> FCombElast( elasticWeights, Fmem, Fbend );
  AdditionGradient<DefaultConfigurator> DFCombElast( elasticWeights, DFmem, DFbend );
  AdditionHessian<DefaultConfigurator> D2FCombElast( elasticWeights, D2Fmem, D2Fbend );
  MatrixOperatorMapWrapper<DefaultConfigurator> opD2Felast( D2FCombElast );

  // Tangent-point
  auto &Ftpe = TPE;
  auto &DFtpe = TPG;
  auto &opD2Ftpe = SSMop;

  // Soft Dirichlet penalty
  PointPositionPenalty<DefaultConfigurator> Fdir( Topology, Config.dirichletVertices, Geometry );
  PointPositionPenaltyDerivative<DefaultConfigurator> DFdir( Topology, Config.dirichletVertices, Geometry );
  PointPositionPenaltyHessian<DefaultConfigurator> D2Fdir( Topology, Config.dirichletVertices, Geometry );
  PointPositionPenaltyHessianOperator<DefaultConfigurator> opD2Fdir( Topology, Config.dirichletVertices, Geometry );

  // Combined energy
  VectorType Weights( 3 );
  Weights << Config.Energy.elasticWeight, Config.Energy.tpeWeight, Config.Energy.dirichletWeight;

  AdditionOp<DefaultConfigurator> F( Weights, FCombElast, Ftpe, Fdir );
  AdditionGradient<DefaultConfigurator> DF( Weights, DFCombElast, DFtpe, DFdir );
  // AdditionHessian<DefaultConfigurator> D2F( Weights, D2FCombElast, SSM, D2Fdir );
  LinearlyCombinedMaps<DefaultConfigurator> opD2F( Weights, opD2Felast, opD2Ftpe, opD2Fdir );
  //endregion

  //region Preconditioners
  VectorType regWeights( 2 );
  regWeights << Config.Energy.elasticWeight, Config.Energy.dirichletWeight;
  AdditionHessian<DefaultConfigurator> regMetric( regWeights, ElasticMetric,  D2Fdir );
  LinearlyCombinedMaps<DefaultConfigurator> opRegMetric( regWeights, opElasticMetric,  opD2Fdir );

  std::vector<int> emptyMask;
  LocalizedPreconditioner<DefaultConfigurator, CholeskyPreconditioner> Pre( opRegMetric, emptyMask, 1. );

  VectorType metricWeights( 3 );
  metricWeights << Config.Energy.elasticWeight,  Config.Energy.tpeWeight, Config.Energy.dirichletWeight;
  // AdditionHessian<DefaultConfigurator> Metric( metricWeights, ElasticMetric, SSM, D2Fdir );
  LinearlyCombinedMaps<DefaultConfigurator> opMetric( metricWeights, opElasticMetric, SSMop, opD2Fdir );

  //endregion

  VectorType deformedGeometry = initGeometry;
  if ( (Geometry - initGeometry).norm() < 1.e-8 )
    deformedGeometry.tail( numVertices ) += VectorType::Random( numVertices ) * 0.00001;

//  prescribedNRIC.head( Topology.getNumEdges()).array() = prescribedNRIC.head( Topology.getNumEdges()).maxCoeff();

  VectorType faceAreas = TriangleAreaOp<DefaultConfigurator>( Topology )( prescribedNRIC );
  VectorType interiorAngles = InteriorAngleOp<DefaultConfigurator>( Topology )( prescribedNRIC );
  VectorType SqrEdgeLengths = prescribedNRIC.head(Topology.getNumEdges()).array().square();

  VectorType EdgeAreas = VectorType::Zero(Topology.getNumEdges());
  for ( int edgeIdx = 0; edgeIdx < Topology.getNumEdges(); ++edgeIdx ) {
    EdgeAreas[edgeIdx] += faceAreas[Topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
    EdgeAreas[edgeIdx] += faceAreas[Topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
  }

  VectorType bendingWeights = SqrEdgeLengths.array() / EdgeAreas.array();

  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. minimal face area = " << faceAreas.minCoeff()
      << "; maximal face area = " << faceAreas.maxCoeff() << std::endl;
  std::cout << " .. minimal interior angle = " << interiorAngles.minCoeff()
      << "; maximal interior angle = " << interiorAngles.maxCoeff() << std::endl;
  std::cout << " .. minimal bending weight = " << bendingWeights.minCoeff()
      << "; maximal bending weight = " << bendingWeights.maxCoeff() << std::endl;
  std::cout << "  -----   " << std::endl;
  std::cout << " Weighted:   " << std::endl;
  std::cout << " .. Fmem(s2) = "
      << Fmem( deformedGeometry ) * Config.Energy.membraneWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. Fbend(s2) = "
      << Fbend( deformedGeometry ) * Config.Energy.bendingWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. Ftpe(s2) = "
      << Ftpe( deformedGeometry ) * Config.Energy.tpeWeight << std::endl;
  std::cout << " .. DFmem(s2).norm = "
      << DFmem( deformedGeometry ).norm() * Config.Energy.membraneWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. DFbend(s2).norm = "
      << DFbend( deformedGeometry ).norm() * Config.Energy.bendingWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. DFtpe(s2).norm = "
      << DFtpe( deformedGeometry ).norm() * Config.Energy.tpeWeight << std::endl;
  std::cout << "  -----   " << std::endl;
  std::cout << " Unweighted:   " << std::endl;
  std::cout << " .. Fmem(s2) = " << Fmem( deformedGeometry ) << std::endl;
  std::cout << " .. Fbend(s2) = " << Fbend( deformedGeometry ) << std::endl;
  std::cout << " .. Ftpe(s2) = " << Ftpe( deformedGeometry ) << std::endl;
  std::cout << " .. DFmem(s2).norm = " << DFmem( deformedGeometry ).norm() << std::endl;
  std::cout << " .. DFbend(s2).norm = " << DFbend( deformedGeometry ).norm() << std::endl;
  std::cout << " .. DFtpe(s2).norm = " << DFtpe( deformedGeometry ).norm() << std::endl;

#ifdef GOAST_WITH_VTK
  saveAsVTP<VectorType>( Topology, deformedGeometry, outputPrefix + "undeformed.vtp" );
#endif
  saveAsPLY<VectorType>( Topology, deformedGeometry, outputPrefix + "undeformed.ply" );

  std::cout << " .. elasticWeights = " << elasticWeights.transpose() << std::endl;

  if ( Config.Optimization.Type == TRUSTREGION ) {
    NewOpt::TrustRegionNewton<DefaultConfigurator> Solver( F, DF, opD2F, 1., 100., 1.e-8,
                                                           Config.Optimization.maxNumIterations,
                                                           1000 );
    Solver.setParameter( "minimal_reduction", 1.e-10 );
    Solver.setParameter( "trsolver__maximum_iterations", 250 );
    Solver.setParameter( "preconditioner", NewOpt::TrustRegionNewton<DefaultConfigurator>::PROVIDED );
    Solver.setPreconditioner( Pre );

    if ( Config.Energy.dirichletWeight == 0. )
      Solver.setBoundaryMask( Config.dirichletVertices );

    t_start = std::chrono::high_resolution_clock::now();
    Solver.solve( deformedGeometry, deformedGeometry );
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << " .... Time: " << std::chrono::duration<double>( t_end - t_start ).count() << " seconds." << std::endl;
  }
  else if ( Config.Optimization.Type == NEWTON ) {
    NewOpt::LineSearchNewtonCG<DefaultConfigurator> Solver( F, DF, opD2F, Pre, 1.e-8,
                                                            Config.Optimization.maxNumIterations, SHOW_ALL );

    if ( Config.Energy.dirichletWeight == 0. )
      Solver.setBoundaryMask( Config.dirichletVertices );

    t_start = std::chrono::high_resolution_clock::now();
    Solver.solve( deformedGeometry, deformedGeometry );
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << " .... Time: " << std::chrono::duration<double>( t_end - t_start ).count() << " seconds." << std::endl;
  }
  else if ( Config.Optimization.Type == PRECONDITIONED ) {
    NewOpt::LineSearchNewtonCG<DefaultConfigurator> Solver( F, DF, opMetric, Pre, 1.e-8,
                                                            Config.Optimization.maxNumIterations, SHOW_ALL );

    if ( Config.Energy.dirichletWeight == 0. )
      Solver.setBoundaryMask( Config.dirichletVertices );

    t_start = std::chrono::high_resolution_clock::now();
    Solver.solve( deformedGeometry, deformedGeometry );
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << " .... Time: " << std::chrono::duration<double>( t_end - t_start ).count() << " seconds." << std::endl;
  }

  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << "  -----   " << std::endl;
  std::cout << " Weighted:   " << std::endl;
  std::cout << " .. Fmem(s2) = "
      << Fmem( deformedGeometry ) * Config.Energy.membraneWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. Fbend(s2) = "
      << Fbend( deformedGeometry ) * Config.Energy.bendingWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. Ftpe(s2) = "
      << Ftpe( deformedGeometry ) * Config.Energy.tpeWeight << std::endl;
  std::cout << " .. DFmem(s2).norm = "
      << DFmem( deformedGeometry ).norm() * Config.Energy.membraneWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. DFbend(s2).norm = "
      << DFbend( deformedGeometry ).norm() * Config.Energy.bendingWeight * Config.Energy.elasticWeight << std::endl;
  std::cout << " .. DFtpe(s2).norm = "
      << DFtpe( deformedGeometry ).norm() * Config.Energy.tpeWeight << std::endl;
  std::cout << "  -----   " << std::endl;
  std::cout << " Unweighted:   " << std::endl;
  std::cout << " .. Fmem(s2) = " << Fmem( deformedGeometry ) << std::endl;
  std::cout << " .. Fbend(s2) = " << Fbend( deformedGeometry ) << std::endl;
  std::cout << " .. Ftpe(s2) = " << Ftpe( deformedGeometry ) << std::endl;
  std::cout << " .. DFmem(s2).norm = " << DFmem( deformedGeometry ).norm() << std::endl;
  std::cout << " .. DFbend(s2).norm = " << DFbend( deformedGeometry ).norm() << std::endl;
  std::cout << " .. DFtpe(s2).norm = " << DFtpe( deformedGeometry ).norm() << std::endl;

  VectorType EdgeLengthError = (Z( deformedGeometry ).segment( 0, numEdges ) - prescribedNRIC.segment( 0, numEdges )).array().abs();
  VectorType relEdgeLengthError = EdgeLengthError.array() /  prescribedNRIC.segment( 0, numEdges ).array();

  std::cout << " .. abs l error = "
      << EdgeLengthError.minCoeff() << " / "
      << median( EdgeLengthError.begin(), EdgeLengthError.end() ) << " / "
      << EdgeLengthError.mean() << " / "
      << EdgeLengthError.maxCoeff()
      << std::endl;

  std::cout << " .. rel l error = "
      << relEdgeLengthError.minCoeff() << " / "
      << median( relEdgeLengthError.begin(), relEdgeLengthError.end() ) << " / "
      << relEdgeLengthError.mean() << " / "
      << relEdgeLengthError.maxCoeff()
      << std::endl;

#ifdef GOAST_WITH_VTK
  std::map<std::string, VectorType> colorings;
  colorings["IsometryError"] = edgeToNodeVector<VectorType>(
    Topology, Z( deformedGeometry ).segment( 0, numEdges ) - prescribedNRIC.segment( 0, numEdges ),
    false
  );
  colorings["RelIsometryError"] = edgeToNodeVector<VectorType>(
    Topology,
    ( Z( deformedGeometry ).segment( 0, numEdges ) - prescribedNRIC.segment( 0, numEdges ) ).array() /
    prescribedNRIC.segment( 0, numEdges ).array(), false
  );
  saveAsVTP<VectorType>( Topology, deformedGeometry, outputPrefix + "deformed.vtp", colorings );
#endif
  saveAsPLY<VectorType>( Topology, deformedGeometry, outputPrefix + "deformed.ply" );


  std::cout << " - outputPrefix: " << outputPrefix << std::endl;

  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
