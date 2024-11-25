#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <goast/Core.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>
#include <goast/Optimization.h>
#include <goast/external/vtkIO.h>


#include "GraphManifold/DifferencePathEnergy.h"

#include "ScaryTPE/TangentPointEnergy.h"
#include "ScaryTPE/TPObstacleEnergy.h"
#include "ScaryTPE/SobolevSlobodeckij.h"

#include "SpookyTPE/FastMultipoleEnergy.h"
#include "SpookyTPE/AdaptiveEnergy.h"

#include "Optimization/TrustRegionNewton.h"
#include "Optimization/LineSearchNewtonCG.h"

#include "PathMetrics.h"
#include "SoftPointConstraint.h"
#include "CombinedDeformation.h"
#include "OperatorPathEnergyHessian.h"
#include "HessianMetric.h"
#include "BarycenterPathEnergy.h"
#include "RotationPathEnergy.h"

#include "MeshIO.h"

#pragma omp declare reduction (merge : std::vector<DefaultConfigurator::TripletType> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

using VectorType = DefaultConfigurator::VectorType;
using VecType = DefaultConfigurator::VecType;
using RealType = DefaultConfigurator::RealType;
using MatrixType = DefaultConfigurator::SparseMatrixType;

using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator> >;


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
    GRADIENT_DESCENT,
    BFGS,
    PRECONDITIONED,
    NEWTONCG,
    TRUSTREGION
  };

  std::unordered_map<std::string, OptimizationType> const optTypeTable = {
    { "GradientDescent", OptimizationType::GRADIENT_DESCENT },
    { "BFGS",            OptimizationType::BFGS },
    { "Preconditioned",  OptimizationType::PRECONDITIONED },
    { "NewtonCG",        OptimizationType::NEWTONCG },
    { "TrustRegion",     OptimizationType::TRUSTREGION },
  };

  enum PreconditionerType {
    ELASTIC_HESSIAN,
    L2_ELASTIC_METRIC,
    L2_COMBINED_METRIC,
    ELASTIC_HESSIAN_AND_L2_SSM,
    L2_SSM,
    ELASTIC_HESSIAN_AND_REDUCED,
  };

  std::unordered_map<std::string, PreconditionerType> const PreconditionerTable = {
    { "ElasticHessian",                  PreconditionerType::ELASTIC_HESSIAN },
    { "L2ElasticMetric",                 PreconditionerType::L2_ELASTIC_METRIC },
    { "L2CombinedMetric",                PreconditionerType::L2_COMBINED_METRIC },
    { "ElasticHessianAndL2SSM",          PreconditionerType::ELASTIC_HESSIAN_AND_L2_SSM },
    { "L2SoboSlobo",                     PreconditionerType::L2_SSM },
    { "ElasticHessianAndReducedHessian", PreconditionerType::ELASTIC_HESSIAN_AND_REDUCED },
  };

  struct {
    std::string outputFolder = "./";
    bool timestampOutput = true;
    std::string outputFilePrefix;

    std::string startFile;
    std::string endFile;
    std::string obstacleFile;
    std::vector<std::string> initFiles;
    std::vector<int> dirichletVertices;

    int numSteps = 6;
    int numLevels = 1;

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
      bool useObstacleAdaptivity = true;
    } TPE;

    struct {
      RealType bendingWeight = 1.;
      RealType elasticWeight = 1.;
      RealType tpeWeight = 1.e-3;
      RealType dirichletWeight = 1.;
      RealType obstacleWeight = 0.;
      RealType barycenterWeight = 0.;
      RealType rotationWeight = 0.;
    } Energy;

    struct {
      int maxNumIterations = 50000;
      RealType minStepsize = 1.e-12;
      RealType maxStepsize = 10.;
      RealType minReduction = 1.e-14;

      OptimizationType Type = GRADIENT_DESCENT;
      PreconditionerType Preconditioner = ELASTIC_HESSIAN;
    } Optimization;
  } Config;
  //endregion Config

  //region Read config
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.startFile = config["Data"]["startFile"].as<std::string>();
    Config.endFile = config["Data"]["endFile"].as<std::string>();
    if ( config["Data"]["obstacleFile"] )
      Config.obstacleFile = config["Data"]["obstacleFile"].as<std::string>();

    Config.numSteps = config["numSteps"].as<int>();
    Config.numLevels = config["numLevels"].as<int>();

    if ( config["Data"]["initFile"] ) {
      if ( config["Data"]["initFile"].IsSequence()) {
        Config.initFiles = config["Data"]["initFile"].as<std::vector<std::string>>();
      }
      else {
        Config.initFiles.resize( Config.numSteps - 1, config["Data"]["initFile"].as<std::string>() );
      }
    }

    Config.dirichletVertices = config["Data"]["dirichletVertices"].as<std::vector<int>>();

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

    Config.Energy.bendingWeight = config["Energy"]["bendingWeight"].as<RealType>();
    Config.Energy.elasticWeight = config["Energy"]["elasticWeight"].as<RealType>();
    Config.Energy.tpeWeight = config["Energy"]["tpeWeight"].as<RealType>();
    Config.Energy.dirichletWeight = config["Energy"]["dirichletWeight"].as<RealType>();
    if ( config["Energy"]["obstacleWeight"] )
      Config.Energy.obstacleWeight = config["Energy"]["obstacleWeight"].as<RealType>();
    if ( config["Energy"]["barycenterWeight"] )
      Config.Energy.barycenterWeight = config["Energy"]["barycenterWeight"].as<RealType>();
    if ( config["Energy"]["rotationWeight"] )
      Config.Energy.rotationWeight = config["Energy"]["rotationWeight"].as<RealType>();

    Config.Optimization.maxNumIterations = config["Optimization"]["maxNumIterations"].as<int>();
    Config.Optimization.minStepsize = config["Optimization"]["minStepsize"].as<RealType>();
    Config.Optimization.maxStepsize = config["Optimization"]["maxStepsize"].as<RealType>();
    Config.Optimization.minReduction = config["Optimization"]["minReduction"].as<RealType>();

    auto otype_it = optTypeTable.find( config["Optimization"]["Type"].as<std::string>());
    if ( otype_it != optTypeTable.end())
      Config.Optimization.Type = otype_it->second;
    else
      throw std::runtime_error( "Invalid Optimization::Type in Config." );

    auto precond_it = PreconditionerTable.find( config["Optimization"]["Preconditioner"].as<std::string>());
    if ( precond_it != PreconditionerTable.end())
      Config.Optimization.Preconditioner = precond_it->second;
    else
      throw std::runtime_error( "Invalid Optimization::Preconditioner in Config." );
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

  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

  // Read meshes
  TriMesh startMesh, endMesh, obstacleMesh;
  std::vector<TriMesh> initMeshes;
  if ( !OpenMesh::IO::read_mesh( startMesh, Config.startFile ))
    throw std::runtime_error( "Failed to read file: " + Config.startFile );
  if ( !OpenMesh::IO::read_mesh( endMesh, Config.endFile ))
    throw std::runtime_error( "Failed to read file: '" + Config.endFile + "'" );

  if (!Config.obstacleFile.empty()) {
    if ( !OpenMesh::IO::read_mesh( obstacleMesh, Config.obstacleFile ))
      throw std::runtime_error( "Failed to read file: " + Config.obstacleFile );

    OpenMesh::IO::write_mesh( obstacleMesh, outputPrefix + "obstacle.ply" );
  }

  if ( !Config.initFiles.empty()) {
    initMeshes.resize( Config.numSteps - 1 );
    for ( int k = 0; k < Config.numSteps - 1; k++ )
      if ( !OpenMesh::IO::read_mesh( initMeshes[k], Config.initFiles[k] ))
        throw std::runtime_error( "Failed to read file: " + Config.initFiles[k] );
  }
  else {
    initMeshes.resize( Config.numSteps - 1, startMesh );
  }

  //region Setup
  // Topology of the mesh
  MeshTopologySaver Topology( startMesh );
  MeshTopologySaver obstacleTopology( obstacleMesh );
  int numVertices = Topology.getNumVertices();

  std::cout << " .. numVertices = " << numVertices << std::endl;

  // Geometry of the mesh
  VectorType Vertices_Start, Vertices_End, Vertices_Obstacle;
  std::vector<VectorType> Vertices_Init( Config.numSteps - 1 );
  getGeometry( startMesh, Vertices_Start );
  getGeometry( endMesh, Vertices_End );
  getGeometry( obstacleMesh, Vertices_Obstacle );
  for ( int k = 0; k < Config.numSteps - 1; k++ )
    getGeometry( initMeshes[k], Vertices_Init[k] );

  // Registration
  std::vector<int> dirichletIndices = Config.dirichletVertices;
  std::vector<int> nonDirichletIndices;
  const auto numDirichletVertices = Config.dirichletVertices.size();
  Config.dirichletVertices.resize( 3 * numDirichletVertices );
  for ( int i = 0; i < numDirichletVertices; i++ ) {
    Config.dirichletVertices[numDirichletVertices + i] = numVertices + Config.dirichletVertices[i];
    Config.dirichletVertices[2 * numDirichletVertices + i] = 2 * numVertices + Config.dirichletVertices[i];
  }
  for ( int vertexIdx = 0; vertexIdx < numVertices; vertexIdx++ ) {
    if ( std::find( dirichletIndices.begin(), dirichletIndices.end(), vertexIdx ) == dirichletIndices.end() )
      nonDirichletIndices.push_back( vertexIdx );
  }

  std::cout << " .. dirichletIndices = {";
  for ( auto idx: dirichletIndices )
    std::cout << idx << ", ";
  std::cout << "}" << std::endl;

  std::vector<int> fixedVariables;
  fillPathMask( Config.numSteps - 1, 3 * numVertices, Config.dirichletVertices, fixedVariables );
  //endregion

  //region Basic energies
  // Elastic energy
  NonlinearMembraneDeformation<DefaultConfigurator> Wmem( Topology, 1. );
  SimpleBendingDeformation<DefaultConfigurator> Wbend( Topology, 1. );
  CombinedDeformation<DefaultConfigurator> Welast( Wmem, Config.Energy.bendingWeight, Wbend );

  std::cout << " .. Wmem = " << Wmem( Vertices_Start, Vertices_End ) << std::endl;
  std::cout << " .. Wbend = " << Wbend( Vertices_Start, Vertices_End ) << std::endl;
  std::cout << " .. W = " << Welast( Vertices_Start, Vertices_End ) << std::endl;

  // Elastic metric
  HessianMetric<DefaultConfigurator> ElasticMetric( Welast, Config.dirichletVertices );
  OperatorHessianMetric<DefaultConfigurator> opElasticMetric( Welast, Config.dirichletVertices );

  // Tangent Point Energy and SobolevSlobodeckijMetric
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
                                                                             Config.TPE.alpha,
                                                                             Config.TPE.beta,
                                                                             Config.TPE.useLowerSSMTerm,
                                                                             Config.TPE.useHigherSSMTerm,
                                                                             Config.TPE.innerWeight,
                                                                             Config.TPE.useAdaptivity );
  //endregion

  VectorType Path;
  for ( int level = 0; level < Config.numLevels; level++ ) {
    std::cout << std::endl;
    std::cout << " --- Level " << level << " --- " << std::endl;
    // Refine in time
    if ( level > 0 )
      Config.numSteps *= 2;

    // Create Dirichlet mask along time
    fillPathMask( Config.numSteps - 1, 3 * numVertices, Config.dirichletVertices, fixedVariables );

    //region Path energies
    VectorType Weights( 6 );
    Weights << Config.Energy.elasticWeight, Config.Energy.barycenterWeight, Config.Energy.dirichletWeight,
        Config.Energy.obstacleWeight, Config.Energy.tpeWeight, Config.Energy.rotationWeight;

    // Elasticity
    DiscretePathEnergy<DefaultConfigurator> Emem( Wmem, Config.numSteps, Vertices_Start, Vertices_End );
    DiscretePathEnergy<DefaultConfigurator> Ebend( Wbend, Config.numSteps, Vertices_Start, Vertices_End );

    DiscretePathEnergy<DefaultConfigurator> Eelast( Welast, Config.numSteps, Vertices_Start, Vertices_End );
    DiscretePathEnergyGradient<DefaultConfigurator> DEelast( Welast, Config.numSteps, Vertices_Start, Vertices_End );
    DiscretePathEnergyHessian<DefaultConfigurator> D2Eelast( Welast, Config.numSteps, Vertices_Start, Vertices_End );
    OperatorPathEnergyHessian<DefaultConfigurator> D2Eop( Welast, Config.numSteps, Vertices_Start, Vertices_End,
                                                          Config.dirichletVertices );


    // TPE
    std::unique_ptr<ObjectiveFunctional<DefaultConfigurator>> chosenTPDE;
    // ObjectiveFunctional<DefaultConfigurator> *chosenTPDE = &spTPDEn;

    if ( Config.TPE.Type == SCARY ) {
      chosenTPDE = std::make_unique<DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy>>(
        Config.numSteps,
        Vertices_Start,
        Vertices_End,
        Topology,
        Config.TPE.alpha,
        Config.TPE.beta,
        Config.TPE.innerWeight,
        Config.TPE.useAdaptivity,
        Config.TPE.theta,
        Config.TPE.thetaNear
      );
    }
    else if ( Config.TPE.Type == SPOOKY ) {
      if ( Config.TPE.useAdaptivity ) {
        chosenTPDE = std::make_unique<DifferencePathEnergy<DefaultConfigurator, SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy>>(
          Config.numSteps,
          Vertices_Start,
          Vertices_End,
          Topology,
          Config.TPE.alpha,
          Config.TPE.beta,
          Config.TPE.innerWeight,
          Config.TPE.theta,
          Config.TPE.thetaNear
        );
      }
      else {
        chosenTPDE = std::make_unique<DifferencePathEnergy<DefaultConfigurator, SpookyTPE::FastMultipoleTangentPointEnergy>>(
          Config.numSteps,
          Vertices_Start,
          Vertices_End,
          Topology,
          Config.TPE.alpha,
          Config.TPE.beta,
          Config.TPE.innerWeight,
          Config.TPE.theta
        );
      }
    }

    ObjectiveWrapper Etpe( *chosenTPDE );
    ObjectiveGradientWrapper DEtpe( *chosenTPDE );
    ObjectiveHessianWrapper RD2Etpe( *chosenTPDE );
    ObjectiveHessianOperatorWrapper opRD2Etpe( *chosenTPDE );

    // RBM
    TrackingPathEnergy<DefaultConfigurator> Edir( dirichletIndices, Config.numSteps, Vertices_Start, Vertices_End );
    TrackingPathEnergyGradient<DefaultConfigurator> DEdir( dirichletIndices, Config.numSteps, Vertices_Start,
                                                           Vertices_End );
    TrackingPathEnergyHessian<DefaultConfigurator> D2Edir( dirichletIndices, Config.numSteps, Vertices_Start,
                                                           Vertices_End );
    TrackingPathEnergyHessianOperator<DefaultConfigurator> opD2Edir( dirichletIndices, Config.numSteps, Vertices_Start,
                                                                     Vertices_End );

    // Obstacle (only available in scary)
    ScaryTPE::TangentPointObstacleEnergy<DefaultConfigurator> Etpoe( Topology, obstacleTopology, Vertices_Obstacle,
                                                                     Config.TPE.alpha, Config.TPE.beta,
                                                                     Config.TPE.innerWeight, Config.TPE.useObstacleAdaptivity,
                                                                     Config.TPE.theta, Config.TPE.thetaNear );

    DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointObstacleEnergy> TPODE(
      Config.numSteps, Vertices_Start, Vertices_End, Topology, obstacleTopology, Vertices_Obstacle, Config.TPE.alpha,
      Config.TPE.beta, Config.TPE.innerWeight, Config.TPE.useObstacleAdaptivity, Config.TPE.theta, Config.TPE.thetaNear,
      5
    );

    ObjectiveWrapper Eobs( TPODE );
    ObjectiveGradientWrapper DEobs( TPODE );
    ObjectiveHessianWrapper D2Eobs( TPODE );
    ObjectiveHessianOperatorWrapper opD2Eobs( TPODE );

    // Barycenter
    BarycenterPathEnergy BPE( Topology, nonDirichletIndices, Config.numSteps, Vertices_Start, Vertices_End );

    ObjectiveWrapper Ebary( BPE );
    ObjectiveGradientWrapper DEbary( BPE );
    ObjectiveHessianWrapper D2Ebary( BPE );
    ObjectiveHessianOperatorWrapper opD2Ebary( BPE );

    // Rotation
    RotationPathEnergy RPE( Topology, nonDirichletIndices, Config.numSteps, Vertices_Start, Vertices_End );

    ObjectiveWrapper Erot( RPE );
    ObjectiveGradientWrapper DErot( RPE );
    ObjectiveHessianWrapper D2Erot( RPE );
    ObjectiveHessianOperatorWrapper opD2Erot( RPE );

    AdditionOp<DefaultConfigurator> E( Weights, Eelast, Ebary, Edir, Eobs, Etpe, Erot );
    AdditionGradient<DefaultConfigurator> DE( Weights, DEelast, DEbary, DEdir, DEobs, DEtpe, DErot );
    //endregion

    //region 2nd order quadratic models (= Hessian approximations)
    // Elastic metric for path
    L2PathMetric<DefaultConfigurator> L2EM( ElasticMetric, Config.numSteps );
    OperatorL2PathMetric<DefaultConfigurator> opL2EM( opElasticMetric, Config.numSteps );

    // Sobolev-Slobodeckij metric for path
    OperatorL2PathMetric<DefaultConfigurator> opL2SSM( SSMop, Config.numSteps );

    // Tracking metric (a constant matrix since tracking term is quadratic, hence Hessian of penalty to fixed positions is used)
    PointPositionPenaltyHessian<DefaultConfigurator> D2Fdir( Topology, dirichletIndices, Vertices_Start );
    PointPositionPenaltyHessianOperator<DefaultConfigurator> opD2Fdir( Topology, dirichletIndices, Vertices_Start );

    // L2-in-time Elastic + L2-in-time Hs + Dirichlet
    // AdditionHessian<DefaultConfigurator> L2comb( Weights, L2EM, L2SSM, D2Fdir ); -- matrix SSM not available anymore
    LinearlyCombinedMaps<DefaultConfigurator> opL2comb( Weights, opL2EM, opD2Ebary, opD2Edir, opD2Eobs, opL2SSM );

    // Elastic Path Energy Hessian + L2-in-time Hs + Dirichlet
    // AdditionHessian<DefaultConfigurator> EH_L2SSM( Weights, D2Eelast, L2SSM, D2Fdir ); -- matrix SSM not available anymore
    LinearlyCombinedMaps<DefaultConfigurator> opEH_L2SSM( Weights, D2Eop, opD2Ebary, opD2Edir, opD2Eobs, opL2SSM );

    // Elastic Path Energy Hessian + approximate TPE Hessian + Dirichlet
    LinearlyCombinedMaps<DefaultConfigurator> opEH_RD2Etp( Weights, D2Eop, opD2Ebary, opD2Edir, opD2Eobs, opRD2Etpe );
    AdditionHessian<DefaultConfigurator> EH_RD2Etp( Weights, D2Eelast, D2Ebary, D2Edir, D2Eobs, RD2Etpe );

    // Local regularized metric, i.e. elastic metric + Hessian of point constraint energy
    LinearlyCombinedMaps<DefaultConfigurator> regMetric( Weights, opElasticMetric, opD2Fdir );

    // Dirichlet boundary for preconditioners if no soft penalty is used
    std::vector<int> preconditionerMask;
    if ( Config.Energy.dirichletWeight == 0. )
      preconditionerMask = Config.dirichletVertices;

    InverseOperatorL2PathMetric<DefaultConfigurator> Pre( regMetric, Config.numSteps, preconditionerMask );
    //endregion


    //region Initial path
//    const RealType tau = 1. / Config.numSteps;
    if ( level == 0 ) {
      Path.resize(( Config.numSteps - 1 ) * 3 * numVertices );
      if ( Config.initFiles.empty()) {
        for ( int k = 0; k < Config.numSteps - 1; k++ )
          Path.segment( k * 3 * numVertices, 3 * numVertices ) =
                  k < Config.numSteps / 2 ? Vertices_Start : Vertices_End;
      }
      else {
        for ( int k = 0; k < Config.numSteps - 1; k++ )
          Path.segment( k * 3 * numVertices, 3 * numVertices ) = Vertices_Init[k];
      }
    }
    else {
      VectorType newPath(( Config.numSteps - 1 ) * 3 * numVertices );
      for ( int k = 0; k < (Config.numSteps / 2) - 1; k++ ) {
        newPath.segment( 2 * k * 3 * numVertices, 3 * numVertices ) = Path.segment( k * 3 * numVertices,
                                                                                    3 * numVertices );
        newPath.segment(( 2 * k + 1 ) * 3 * numVertices, 3 * numVertices ) = Path.segment( k * 3 * numVertices,
                                                                                           3 * numVertices );
      }
      newPath.tail( 3 * numVertices ) = Path.tail( 3 * numVertices );
      Path.resize(( Config.numSteps - 1 ) * 3 * numVertices );
      Path = newPath;
    }

    {
      std::cout << " .. Profiling: " << std::endl;
      // -- Energy --
      t_start = std::chrono::high_resolution_clock::now();
      RealType Eval = E( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Energy evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Eelast( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Elastic: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Etpe( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... TPE: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Edir( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... RBM: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Eobs( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Obstacle: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Ebary( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Barycenter: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;;

      t_start = std::chrono::high_resolution_clock::now();
      Eval = Erot( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Rotation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // -- Gradient --
      t_start = std::chrono::high_resolution_clock::now();
      VectorType DEval = DE( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Gradient evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEelast( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Elastic: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEtpe( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... TPE: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEdir( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... RBM: "
      << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEobs( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Obstacle: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DEbary( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Barycenter: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      DEval = DErot( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Rotation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // -- Hessian --
      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval = opEH_RD2Etp( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Hessian (operator) evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // D2Eop, opRD2Etpe, opD2Edir
      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_elast = D2Eop( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Elastic: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_tpe = opRD2Etpe( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... TPE: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_dir = opD2Edir( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... RBM: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_obs = opD2Eobs( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Obstacle: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      auto D2Eval_bary = opD2Ebary( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " ...... Barycenter: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;


      t_start = std::chrono::high_resolution_clock::now();
      VectorType Hv = ( *D2Eval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Hessian * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Hv = ( *D2Eval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Hessian * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      // -- Preconditioner --
      t_start = std::chrono::high_resolution_clock::now();
      auto Preval = Pre( Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Preconditioner (operator) evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      VectorType Pv = ( *Preval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Pre * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;

      t_start = std::chrono::high_resolution_clock::now();
      Pv = ( *Preval )( DEval );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .... Pre * grad evaluation: "
                << std::chrono::duration<double, std::milli>( t_end - t_start ).count() << "ms" << std::endl;
    }


    std::cout << " ................................................ " << std::endl;
    std::cout << std::scientific << std::setprecision( 6 );
    std::cout << " .. TPE          = ( ";
    std::cout << TPE( Vertices_Start ) << " ";
    for ( int k = 0; k < Config.numSteps - 1; k++ )
      std::cout << TPE( Path.segment( k * 3 * numVertices, 3 * numVertices )) << " ";
    std::cout << TPE( Vertices_End ) << " )" << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. E            = " << E( Path ) << std::endl;
    std::cout << " .. Eelast       = " << Eelast( Path ) << std::endl;
    std::cout << " .. Etpe         = " << Etpe( Path ) << std::endl;
    std::cout << " .. Eobs         = " << Eobs( Path ) << std::endl;
    std::cout << " .. Edir         = " << Edir( Path ) << std::endl;
    std::cout << " .. Ebary        = " << Ebary( Path ) << std::endl;
    std::cout << " .. Erot         = " << Erot( Path ) << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. DE.norm      = " << DE( Path ).norm() << std::endl;
    std::cout << " .. DEelast.norm = " << DEelast( Path ).norm() << std::endl;
    std::cout << " .. DEtpe.norm   = " << DEtpe( Path ).norm() << std::endl;
    std::cout << " .. DEobs.norm   = " << DEobs( Path ).norm() << std::endl;
    std::cout << " .. DEdir.norm   = " << DEdir( Path ).norm() << std::endl;
    std::cout << " .. DEbary.norm  = " << DEbary( Path ).norm() << std::endl;
    std::cout << " .. DErot.norm   = " << DErot( Path ).norm() << std::endl;
    std::cout << " ................................................ " << std::endl;
    VectorType pathGradient = -DE( Path );
    applyMaskToVector( fixedVariables, pathGradient );

    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_Start );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_init_0.vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_init_0.ply" );
    }
    for ( int k = 0; k < Config.numSteps - 1; k++ ) {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = pathGradient.segment( k * 3 * numVertices, 3 * numVertices );
      data["TPG"] = -TPG( Path.segment( k * 3 * numVertices, 3 * numVertices ));
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( k + 1 ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( k + 1 ) + ".ply" );
    }
    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_Start );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( Config.numSteps ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_init_" + std::to_string( Config.numSteps ) + ".ply" );
    }

    //endregion

    //region Optimization
    if ( Config.Optimization.Type == GRADIENT_DESCENT ) {
      GradientDescent<DefaultConfigurator> Solver( E, DE, Config.Optimization.maxNumIterations, 1.e-8, ARMIJO, SHOW_ALL,
                                                   0.1, Config.Optimization.minStepsize,
                                                   Config.Optimization.maxStepsize );

      Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == BFGS ) {
      QuasiNewtonBFGS<DefaultConfigurator> Solver( E, DE, Config.Optimization.maxNumIterations, 1.e-8, ARMIJO, 50,
                                                   SHOW_ALL, 0.1, Config.Optimization.minStepsize,
                                                   Config.Optimization.maxStepsize );

      Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == PRECONDITIONED ) {
      ObjectiveHessian<DefaultConfigurator> *Preconditioner = nullptr;

      if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN ) {
        Preconditioner = &D2Eelast;
      }
      else if ( Config.Optimization.Preconditioner == L2_ELASTIC_METRIC ) {
        Preconditioner = &L2EM;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_REDUCED ) {
        Preconditioner = &EH_RD2Etp;
      }

      LineSearchNewton<DefaultConfigurator> Solver( E, DE, *Preconditioner, 1.e-8, Config.Optimization.maxNumIterations,
                                                    SHOW_ALL );
      Solver.setParameter( "minimal_stepsize", Config.Optimization.minStepsize );
      Solver.setParameter( "maximal_stepsize", Config.Optimization.maxStepsize );
      Solver.setParameter( "tau_increase", 5. );
      Solver.setParameter( "reduced_direction", false );

      Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == NEWTONCG ) {
      MapToLinOp<DefaultConfigurator> *Hess = nullptr;

      if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN ) {
        Hess = &D2Eop;
      }
      else if ( Config.Optimization.Preconditioner == L2_ELASTIC_METRIC ) {
        Hess = &opL2EM;
      }
      else if ( Config.Optimization.Preconditioner == L2_COMBINED_METRIC ) {
        Hess = &opL2comb;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_L2_SSM ) {
        Hess = &opEH_L2SSM;
      }
      else if ( Config.Optimization.Preconditioner == L2_SSM ) {
        Hess = &opL2SSM;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_REDUCED ) {
        Hess = &opEH_RD2Etp;
      }

      NewOpt::LineSearchNewtonCG<DefaultConfigurator> Solver( E, DE, *Hess, Pre, 1.e-8,
                                                           Config.Optimization.maxNumIterations,
                                                           SHOW_ALL );
      Solver.setParameter( "cg_iterations", 2000 );
      Solver.setBoundaryMask( Config.dirichletVertices );
      Solver.setParameter( "minimal_stepsize", Config.Optimization.minStepsize );
      Solver.setParameter( "maximal_stepsize", Config.Optimization.maxStepsize );

      if ( Config.Energy.dirichletWeight == 0. )
        Solver.setBoundaryMask( fixedVariables );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == TRUSTREGION ) {
      MapToLinOp<DefaultConfigurator> *Hess = nullptr;

      if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN ) {
        Hess = &D2Eop;
      }
      else if ( Config.Optimization.Preconditioner == L2_ELASTIC_METRIC ) {
        Hess = &opL2EM;
      }
      else if ( Config.Optimization.Preconditioner == L2_COMBINED_METRIC ) {
        Hess = &opL2comb;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_L2_SSM ) {
        Hess = &opEH_L2SSM;
      }
      else if ( Config.Optimization.Preconditioner == L2_SSM ) {
        Hess = &opL2SSM;
      }
      else if ( Config.Optimization.Preconditioner == ELASTIC_HESSIAN_AND_REDUCED ) {
        Hess = &opEH_RD2Etp;
      }

      NewOpt::SteihaugCGMethod<DefaultConfigurator>::resetTimers();
      InverseOperatorL2PathMetric<DefaultConfigurator>::resetTimers();

      if ( Config.TPE.Type == SCARY )
        DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy>::resetTimers();
      else if ( Config.TPE.Type == SPOOKY )
        if ( Config.TPE.useAdaptivity )
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy>::resetTimers();
        else
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::FastMultipoleTangentPointEnergy>::resetTimers();

      DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointObstacleEnergy>::resetTimers();

      DiscretePathEnergy<DefaultConfigurator>::resetTimers();
      DiscretePathEnergyGradient<DefaultConfigurator>::resetTimers();
      DiscretePathEnergyHessian<DefaultConfigurator>::resetTimers();
      OperatorPathEnergyHessian<DefaultConfigurator>::resetTimers();

      NewOpt::TrustRegionNewton<DefaultConfigurator> Solver( E, DE, *Hess, 1., 100., 1e-8,
                                                          Config.Optimization.maxNumIterations,
                                                          5000 );
      Solver.setParameter( "minimal_reduction", Config.Optimization.minReduction );
      Solver.setParameter( "trsolver__maximum_iterations", 250 );
      Solver.setParameter( "preconditioner", NewOpt::TrustRegionNewton<DefaultConfigurator>::PROVIDED );
      Solver.setPreconditioner( Pre );

      if ( Config.Energy.dirichletWeight == 0. )
        Solver.setBoundaryMask( fixedVariables );

      std::function callbackFct = [&]( int i, const VectorType &x, const RealType &F, const VectorType &grad_F ) {
        if ( i % 100 == 0 ) {
          saveAsPLY<VectorType>( Topology, Vertices_Start,
                                 outputPrefix + "comb_level_" + std::to_string( level ) + "_I" + std::to_string( i ) +
                                 "_0.ply" );

          for ( int k = 0; k < Config.numSteps - 1; k++ ) {
            saveAsPLY<VectorType>( Topology, x.segment( k * 3 * numVertices, 3 * numVertices ),
                                   outputPrefix + "comb_level_" + std::to_string( level ) + "_I" + std::to_string( i ) +
                                   "_" +
                                   std::to_string( k + 1 ) + ".ply" );
          }

          saveAsPLY<VectorType>( Topology, Vertices_End,
                                 outputPrefix + "comb_level_" + std::to_string( level ) + "_I" + std::to_string( i ) +
                                 "_" +
                                 std::to_string( Config.numSteps ) + ".ply" );
        }
      };

//      Solver.addCallbackFunction( callbackFct );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( Path, Path );
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << " .. Total time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;


      std::cout << std::scientific << std::setprecision( 6 ) << " .. Result: "
                << Config.numSteps << ","
                << numVertices << ","
                << Solver.Status().Iteration << ","
                << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count() << ","
                << Solver.Status().additionalIterations.at( "Subproblem" ) << ","
                << Solver.Status().additionalTimings.at( "Subproblem" ) << ","
                << Solver.Status().additionalTimings.at( "Preconditioner" ) << ","
                << Solver.Status().additionalTimings.at( "Evaluation" ) << ","
                << std::endl;


      std::cout << " ................................................ " << std::endl;

      NewOpt::SteihaugCGMethod<DefaultConfigurator>::printTimings();
      InverseOperatorL2PathMetric<DefaultConfigurator>::printTimings();

      if ( Config.TPE.Type == SCARY )
        DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy>::printTimings();
      else if ( Config.TPE.Type == SPOOKY )
        if ( Config.TPE.useAdaptivity )
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy>::printTimings();
        else
          DifferencePathEnergy<DefaultConfigurator, SpookyTPE::FastMultipoleTangentPointEnergy>::printTimings();

      DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointObstacleEnergy>::printTimings();

      DiscretePathEnergy<DefaultConfigurator>::printTimings();
      DiscretePathEnergyGradient<DefaultConfigurator>::printTimings();
      DiscretePathEnergyHessian<DefaultConfigurator>::printTimings();
      OperatorPathEnergyHessian<DefaultConfigurator>::printTimings();
    }
    //endregion

    //region Evaluation and output
    std::cout << " ................................................ " << std::endl;
    std::cout << std::scientific << std::setprecision( 6 ) << std::endl;
    std::cout << " .. TPE          = ( ";
    std::cout << TPE( Vertices_Start ) << " ";
    for ( int k = 0; k < Config.numSteps - 1; k++ )
      std::cout << TPE( Path.segment( k * 3 * numVertices, 3 * numVertices )) << " ";
    std::cout << TPE( Vertices_End ) << " )" << std::endl;

    VectorType pathEnergies, pathEnergies_elast, pathEnergies_tpe, pathEnergies_mem, pathEnergies_bend,
        pathEnergies_dir, pathEnergies_obs, pathEnergies_rot, pathEnergies_bary;
    Eelast.evaluateSingleEnergies( Path, pathEnergies_elast );
    Emem.evaluateSingleEnergies( Path, pathEnergies_mem );
    Ebend.evaluateSingleEnergies( Path, pathEnergies_bend );

    if ( Config.TPE.Type == SCARY )
      pathEnergies_tpe = dynamic_cast<DifferencePathEnergy<DefaultConfigurator, ScaryTPE::TangentPointEnergy> *>(
        chosenTPDE.get())->stepEnergies( Path );
    else if ( Config.TPE.Type == SPOOKY )
      if ( Config.TPE.useAdaptivity )
        pathEnergies_tpe = dynamic_cast<DifferencePathEnergy<DefaultConfigurator,
          SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy> *>(chosenTPDE.get())->stepEnergies( Path );
      else
        pathEnergies_tpe = dynamic_cast<DifferencePathEnergy<DefaultConfigurator,
          SpookyTPE::FastMultipoleTangentPointEnergy> *>(chosenTPDE.get())->stepEnergies( Path );

    pathEnergies_obs = TPODE.stepEnergies( Path );

    // Eelast, Ebary, Edir, Eobs, Etpe, Erot
    Edir.evaluateSingleEnergies( Path, pathEnergies_dir );
    pathEnergies_bary = BPE.stepEnergies( Path );
    pathEnergies_rot = RPE.stepEnergies( Path );
    pathEnergies = Config.Energy.elasticWeight * pathEnergies_elast +
                   Config.Energy.tpeWeight * pathEnergies_tpe + Config.Energy.obstacleWeight * pathEnergies_obs +
                   Config.Energy.dirichletWeight * pathEnergies_dir +
                   Config.Energy.barycenterWeight * pathEnergies_bary + Config.Energy.rotationWeight * pathEnergies_rot;

    std::cout << " ................................................ " << std::endl;
    std::cout << " .. E            = " << pathEnergies.transpose() << std::endl;
    std::cout << " .. Eelast       = " << pathEnergies_elast.transpose() << std::endl;
    std::cout << " .. Emem         = " << pathEnergies_mem.transpose() << std::endl;
    std::cout << " .. Ebend        = " << pathEnergies_bend.transpose() << std::endl;
    std::cout << " .. Etpe         = " << pathEnergies_tpe.transpose() << std::endl;
    std::cout << " .. Eobs         = " << pathEnergies_obs.transpose() << std::endl;
    std::cout << " .. Edir         = " << pathEnergies_dir.transpose() << std::endl;
    std::cout << " .. Ebary        = " << pathEnergies_bary.transpose() << std::endl;
    std::cout << " .. Erot         = " << pathEnergies_rot.transpose() << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. E            = " << E( Path ) << std::endl;
    std::cout << " .. Eelast       = " << Eelast( Path ) << std::endl;
    std::cout << " .. Etpe         = " << Etpe( Path ) << std::endl;
    std::cout << " .. Eobs         = " << Eobs( Path ) << std::endl;
    std::cout << " .. Edir         = " << Edir( Path ) << std::endl;
    std::cout << " .. Ebary        = " << Ebary( Path ) << std::endl;
    std::cout << " .. Erot         = " << Erot( Path ) << std::endl;
    std::cout << " ................................................ " << std::endl;
    std::cout << " .. DE.norm      = " << DE( Path ).norm() << std::endl;
    std::cout << " .. DEelast.norm = " << DEelast( Path ).norm() << std::endl;
    std::cout << " .. DEtpe.norm   = " << DEtpe( Path ).norm() << std::endl;
    std::cout << " .. DEobs.norm   = " << DEobs( Path ).norm() << std::endl;
    std::cout << " .. DEdir.norm   = " << DEdir( Path ).norm() << std::endl;
    std::cout << " .. DEbary.norm  = " << DEbary( Path ).norm() << std::endl;
    std::cout << " .. DErot.norm   = " << DErot( Path ).norm() << std::endl;
    pathGradient = -DE( Path );
    applyMaskToVector( fixedVariables, pathGradient );

    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_Start );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_0.vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_Start, outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_0.ply" );
    }
    for ( int k = 0; k < Config.numSteps - 1; k++ ) {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = pathGradient.segment( k * 3 * numVertices, 3 * numVertices );
      data["TPG"] = -TPG( Path.segment( k * 3 * numVertices, 3 * numVertices ));
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( k + 1 ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Path.segment( k * 3 * numVertices, 3 * numVertices ),
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( k + 1 ) + ".ply" );
    }
    {
#ifdef GOAST_WITH_VTK
      std::map<std::string, VectorType> data;
      data["Gradient"] = VectorType::Zero( 3 * numVertices );
      data["TPG"] = -TPG( Vertices_End );
      applyMaskToVector( Config.dirichletVertices, data["TPG"] );
      saveAsVTP<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( Config.numSteps ) + ".vtp", data );
#endif
      saveAsPLY<VectorType>( Topology, Vertices_End,
                             outputPrefix + "comb_level_" + std::to_string( level ) +"_curve_" + std::to_string( Config.numSteps ) + ".ply" );
    }
    //endregion
  }
  std::cout << " ................................................ " << std::endl;
  std::cout << " - outputPrefix: " << outputPrefix << std::endl;
  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
