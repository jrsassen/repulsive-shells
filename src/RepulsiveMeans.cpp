#include <utility>
#include <iostream>
#include <fstream>

#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <goast/Core.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>

#include "SpookyTPE/AdaptiveEnergy.h"
#include "SpookyTPE/FastMultipoleEnergy.h"

#include "ScaryTPE/TangentPointEnergy.h"
#include "ScaryTPE/SobolevSlobodeckij.h"

#include "Optimization/TrustRegionNewton.h"
#include "Optimization/LineSearchNewtonCG.h"

#include "GraphManifold/FunctionalMeanEnergy.h"

#include "PathMetrics.h"
#include "MeshIO.h"
#include "HessianMetric.h"
#include "OperatorElasticMeanFunctionalHessian.h"

using VectorType = DefaultConfigurator::VectorType;
using MatrixType = DefaultConfigurator::SparseMatrixType;
using VecType = DefaultConfigurator::VecType;
using RealType = DefaultConfigurator::RealType;

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
    NEWTONCG,
    TRUSTREGION
  };

  std::unordered_map<std::string, OptimizationType> const optTypeTable = {
    { "NewtonCG",    OptimizationType::NEWTONCG },
    { "TrustRegion", OptimizationType::TRUSTREGION },
  };

  enum PreconditionerType {
    ELASTIC_HESSIAN,
    ELASTIC_METRIC,
    COMBINED_METRIC,
    ELASTIC_HESSIAN_AND_SSM,
  };

  std::unordered_map<std::string, PreconditionerType> const PreconditionerTable = {
    { "ElasticHessian",       PreconditionerType::ELASTIC_HESSIAN },
    { "ElasticMetric",        PreconditionerType::ELASTIC_METRIC },
    { "CombinedMetric",       PreconditionerType::COMBINED_METRIC },
    { "ElasticHessianAndSSM", PreconditionerType::ELASTIC_HESSIAN_AND_SSM },
  };

  struct {
    std::string outputFolder = "./";
    bool timestampOutput = true;
    std::string outputFilePrefix;

    std::vector<std::string> inputFiles;
    std::vector<int> dirichletVertices;
    std::string initFile;

    struct {
      int maxNumIterations = 50000;
      RealType minStepsize = 1.e-12;
      RealType maxStepsize = 10.;

      OptimizationType Type = NEWTONCG;
      PreconditionerType Preconditioner = ELASTIC_HESSIAN;
    } Optimization;

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
      RealType bendingWeight = 1.;
      RealType elasticWeight = 1.;
      RealType tpeWeight = 1.e-3;
    } Energy;
  } Config;
  //endregion

  //region Read config
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.inputFiles = config["Data"]["inputFiles"].as<std::vector<std::string>>();
    Config.initFile = config["Data"]["initFile"].as<std::string>();
    Config.dirichletVertices = config["Data"]["dirichletVertices"].as<std::vector<int>>();

    Config.outputFilePrefix = config["Output"]["outputFilePrefix"].as<std::string>();
    Config.outputFolder = config["Output"]["outputFolder"].as<std::string>();
    Config.timestampOutput = config["Output"]["timestampOutput"].as<bool>();

    Config.Energy.bendingWeight = config["Energy"]["bendingWeight"].as<RealType>();
    Config.Energy.elasticWeight = config["Energy"]["elasticWeight"].as<RealType>();
    Config.Energy.tpeWeight = config["Energy"]["tpeWeight"].as<RealType>();

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

    Config.Optimization.maxNumIterations = config["Optimization"]["maxNumIterations"].as<int>();
    Config.Optimization.minStepsize = config["Optimization"]["minStepsize"].as<RealType>();
    Config.Optimization.maxStepsize = config["Optimization"]["maxStepsize"].as<RealType>();

    auto type_it = optTypeTable.find( config["Optimization"]["Type"].as<std::string>());
    if ( type_it != optTypeTable.end())
      Config.Optimization.Type = type_it->second;
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

  int numInputShapes = Config.inputFiles.size();

  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

  VectorType combinedVertices;
  std::vector<VectorType> Vertices;

  // Read first mesh
  TriMesh Mesh;
  if ( !OpenMesh::IO::read_mesh( Mesh, Config.inputFiles[0] ))
    throw std::runtime_error( "Failed to read file: " + Config.inputFiles[0] );

  // Topology of the mesh
  MeshTopologySaver Topology( Mesh );
  int numVertices = Topology.getNumVertices();
  std::cout << " .. numVertices = " << numVertices << std::endl;

  Vertices.resize( numInputShapes );
  combinedVertices.resize( numInputShapes * 3 * numVertices );
  for ( int i = 0; i < numInputShapes; i++ ) {
    TriMesh inputMesh;
    if ( !OpenMesh::IO::read_mesh( inputMesh, Config.inputFiles[i] ))
      throw std::runtime_error( "Failed to read file: " + Config.inputFiles[i] );

    getGeometry( inputMesh, Vertices[i] );

    combinedVertices.segment( i * 3 * numVertices, 3 * numVertices ) = Vertices[i];

    saveAsPLY<VectorType>( Topology, Vertices[i], outputPrefix + "input_" + std::to_string( i ) + ".ply" );
  }


  TriMesh initMesh;
  if ( !OpenMesh::IO::read_mesh( initMesh, Config.initFile ))
    throw std::runtime_error( "Failed to read file: " + Config.initFile );

  const auto numDirichletVertices = Config.dirichletVertices.size();
  Config.dirichletVertices.resize( 3 * numDirichletVertices );
  for ( int i = 0; i < numDirichletVertices; i++ ) {
    Config.dirichletVertices[numDirichletVertices + i] = numVertices + Config.dirichletVertices[i];
    Config.dirichletVertices[2 * numDirichletVertices + i] = 2 * numVertices + Config.dirichletVertices[i];
  }

  // Elastic energy
  NonlinearMembraneDeformation<DefaultConfigurator> Wmem( Topology, 1. );
  SimpleBendingDeformation<DefaultConfigurator> Wbend( Topology, 1. );
  ShellDeformationType W( Topology, Config.Energy.bendingWeight );

  // Tangent-point energy
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

  for ( int k = 0; k < numInputShapes; k++ ) {
    std::cout << " .. TPE(i" << k << ") = " << TPE( Vertices[k] ) << std::endl;
  }

  // Functionals
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<RealType> distribution(1.,1.);

  // Initialization
  VectorType initShape;
  getGeometry( initMesh, initShape );
  saveAsPLY<VectorType>( Topology, initShape, outputPrefix + "init.ply" );

  HessianMetric<DefaultConfigurator> ElasticMetric( W, Config.dirichletVertices );
  OperatorHessianMetric<DefaultConfigurator> opElasticMetric( W, Config.dirichletVertices );
  LocalizedPreconditioner<DefaultConfigurator, CholeskyPreconditioner> Pre( opElasticMetric, Config.dirichletVertices,
                                                                            1. / Config.Energy.elasticWeight );

  VectorType Weights( 2 );
  Weights << Config.Energy.elasticWeight, Config.Energy.tpeWeight;

  VectorType metricWeights( 2 );
  metricWeights << Config.Energy.elasticWeight, Config.Energy.tpeWeight;
  AddedMaps<DefaultConfigurator> opMetric( opElasticMetric, SSMop, Config.Energy.elasticWeight,
                                           Config.Energy.tpeWeight );

  int numSteps = 2;
  std::vector<VectorType> barycentricWeights;
  for ( int x = 0; x < numSteps + 1; x++ ) {
    RealType X = x * 1. / numSteps;
    for ( int y = 0; y < numSteps - x + 1; y++ ) {
      RealType Y = y * 1. / numSteps;
      RealType Z = 1 - X - Y;

      VectorType localWeights = VectorType::Zero( numInputShapes );
      localWeights[0] = X;
      localWeights[1] = Y;
      localWeights[2] = Z;

      barycentricWeights.push_back( localWeights );

      std::cout << "Coordinates: " << X << " - " << Y << " - " << Z << std::endl;
    }
  }

  std::ofstream out( outputPrefix + "coordinates.csv" );
  out << "id,w_0,w_1,w_2" << std::endl;
  for ( int i = 0; i < barycentricWeights.size(); i++ ) {
    VectorType localWeights = barycentricWeights[i];
    out << i << "," << localWeights[0] << "," << localWeights[1] << "," << localWeights[2] << std::endl;
  }
  out.close();


//#pragma omp parallel for private(t_start, t_end)
  for (int i = 0; i < barycentricWeights.size(); i++) {
    std::cout << "i = " << i << std::endl;
    VectorType elasticWeights = barycentricWeights[i];

    ElasticMeanFunctional<DefaultConfigurator> Felast( W, combinedVertices, elasticWeights, numInputShapes );
    ElasticMeanFunctionalGradient<DefaultConfigurator> DFelast( W, combinedVertices, elasticWeights, numInputShapes );
    ElasticMeanFunctionalHessian<DefaultConfigurator> D2Felast( W, combinedVertices, elasticWeights, numInputShapes );
    OperatorElasticMeanFunctionalHessian<DefaultConfigurator> opD2Felast( W, combinedVertices, elasticWeights, numInputShapes );

    VectorType meanShape = Vertices[0];

    if ( Config.Optimization.Type == TRUSTREGION ) {
      NewOpt::TrustRegionNewton<DefaultConfigurator> Solver( Felast, DFelast, opD2Felast, 1., Config.Optimization.maxStepsize, 1.e-8,
                                                          Config.Optimization.maxNumIterations,
                                                          1000 );
      Solver.setParameter( "minimal_reduction", 1.e-12 );
//    Solver.setParameter( "trsolver__maximum_iterations", 250 );
      Solver.setParameter( "preconditioner", NewOpt::TrustRegionNewton<DefaultConfigurator>::PROVIDED );
      Solver.setPreconditioner( Pre );

//      Solver.setParameter("print_level", 0);

      Solver.setBoundaryMask( Config.dirichletVertices );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( initShape, meanShape );
      t_end = std::chrono::high_resolution_clock::now();

      std::cout << " .... Time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
                << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == NEWTONCG ) {
      LocalizedPreconditioner<DefaultConfigurator, CholeskyPreconditioner> Pre( opElasticMetric,
                                                                                Config.dirichletVertices,
                                                                                1. );
      NewOpt::LineSearchNewtonCG<DefaultConfigurator> Solver( Felast, DFelast, opElasticMetric, Pre, 1.e-8,
                                                              Config.Optimization.maxNumIterations,
                                                              SHOW_TERMINATION_INFO );

      Solver.setBoundaryMask( Config.dirichletVertices );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( initShape, meanShape );
      t_end = std::chrono::high_resolution_clock::now();

      std::cout << " .... Time: " << std::chrono::duration<double>( t_end - t_start ).count() << " seconds."
          << std::endl;
    }

    RealType tpeValue = TPE( meanShape );

#pragma omp critical
    {
      std::cout << " .. Weights (" << i << ") = " << elasticWeights.transpose() << std::endl;
      std::cout << " .. TPE(" << i << ") = " << tpeValue << std::endl;
      if ( tpeValue == std::numeric_limits<RealType>::infinity() )
        std::cout << " ---- INTERSECTION ----" << std::endl;
      saveAsPLY<VectorType>( Topology, meanShape, outputPrefix + "mean_" + std::to_string( i ) + ".ply" );
    }


    std::cout << " .. Initializing TangentPointMeanEnergy..." << std::endl;
    FunctionalMeanEnergy TPME( *chosenTPE, Vertices, elasticWeights );

    ObjectiveWrapper Ftpe( TPME );
    ObjectiveGradientWrapper DFtpe( TPME );
    ObjectiveHessianOperatorWrapper opRD2Ftpe( TPME );

    AdditionOp<DefaultConfigurator> F( Weights, Felast, Ftpe );
    AdditionGradient<DefaultConfigurator> DF( Weights, DFelast, DFtpe );
    AddedMaps<DefaultConfigurator> opD2F( opD2Felast, opRD2Ftpe, Config.Energy.elasticWeight,
                                          Config.Energy.tpeWeight );

    std::cout << " .. Starting optimization..." << std::endl;
    if ( Config.Optimization.Type == TRUSTREGION ) {
      NewOpt::TrustRegionNewton<DefaultConfigurator> Solver( F, DF, opD2F, 1., Config.Optimization.maxStepsize,
                                                             1.e-8,
                                                             Config.Optimization.maxNumIterations,
                                                             1000 );
      Solver.setParameter( "minimal_reduction", 1.e-10 );
      //    Solver.setParameter( "trsolver__maximum_iterations", 250 );
      Solver.setParameter( "preconditioner", NewOpt::TrustRegionNewton<DefaultConfigurator>::PROVIDED );
      Solver.setPreconditioner( Pre );

      //      Solver.setParameter("print_level", 0);

      Solver.setBoundaryMask( Config.dirichletVertices );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( initShape, meanShape );
      t_end = std::chrono::high_resolution_clock::now();

      std::cout << " .... Time: " << std::chrono::duration<double, std::ratio<1>>( t_end - t_start ).count()
          << " seconds." << std::endl;
    }
    else if ( Config.Optimization.Type == NEWTONCG ) {
      NewOpt::LineSearchNewtonCG<DefaultConfigurator> Solver( F, DF, opMetric, Pre, 1.e-8,
                                                              Config.Optimization.maxNumIterations, SHOW_ALL );

      Solver.setBoundaryMask( Config.dirichletVertices );

      t_start = std::chrono::high_resolution_clock::now();
      Solver.solve( initShape, meanShape );
      t_end = std::chrono::high_resolution_clock::now();

      std::cout << " .... Time: " << std::chrono::duration<double>( t_end - t_start ).count()
          << " seconds." << std::endl;
    }

    saveAsPLY<VectorType>( Topology, meanShape, outputPrefix + "tpe_mean_" + std::to_string( i ) + ".ply" );


  }

  std::cout << " - outputPrefix: " << outputPrefix << std::endl;

  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
