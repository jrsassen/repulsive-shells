#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <utility>

#include <goast/Core.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>
#include <goast/external/vtkIO.h>
#include <SpookyTPE/FastMultipoleEnergy.h>

#include "ScaryTPE/TangentPointEnergy.h"
#include "SpookyTPE/AdaptiveEnergy.h"

#include "Optimization/LineSearchNewtonCG.h"
#include "Optimization/Newton.h"

#include "MeshIO.h"
#include "BarycenterPathEnergy.h"

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

  struct {
    std::string outputFolder = "./";
    bool timestampOutput = true;
    std::string outputFilePrefix;

    std::string startFile;
    std::string secondFile;
    std::string initFile;
    std::vector<int> dirichletVertices;

    int numSteps = 6;

    struct {
      int maxNumIterations = 50000;
      RealType minStepsize = 1.e-12;
      RealType maxStepsize = 10.;
    } Optimization;

    struct {
      int alpha = 6;
      int beta = 12;
      TPEType Type = SCARY;
      RealType innerWeight = 1.;
      RealType theta = 0.25;
      RealType thetaNear = 10.;
      bool useAdaptivity = true;
      bool useObstacleAdaptivity = true;
    } TPE;

    struct {
      RealType bendingWeight = 1.;
      RealType elasticWeight = 1.;
      RealType tpeWeight = 1.e-3;
      RealType dirichletWeight = 0.;
      RealType barycenterWeight = 1.;
    } Energy;
  } Config;
  //endregion

  //region Read config
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.startFile = config["Data"]["startFile"].as<std::string>();
    Config.secondFile = config["Data"]["secondFile"].as<std::string>();
    if ( config["Data"]["initFile"] )
      Config.initFile = config["Data"]["initFile"].as<std::string>();
    Config.dirichletVertices = config["Data"]["dirichletVertices"].as<std::vector<int>>();

    Config.outputFilePrefix = config["Output"]["outputFilePrefix"].as<std::string>();
    Config.outputFolder = config["Output"]["outputFolder"].as<std::string>();
    Config.timestampOutput = config["Output"]["timestampOutput"].as<bool>();

    Config.numSteps = config["numSteps"].as<int>();

    Config.TPE.alpha = config["TPE"]["alpha"].as<int>();
    Config.TPE.beta = config["TPE"]["beta"].as<int>();
    Config.TPE.useAdaptivity = config["TPE"]["useAdaptivity"].as<bool>();
    Config.TPE.innerWeight = config["TPE"]["innerWeight"].as<RealType>();
    Config.TPE.theta = config["TPE"]["theta"].as<RealType>();
    Config.TPE.thetaNear = config["TPE"]["thetaNear"].as<RealType>();

    if ( auto ttype_it = TPETypeTable.find( config["TPE"]["Type"].as<std::string>()); ttype_it != TPETypeTable.end())
      Config.TPE.Type = ttype_it->second;
    else
      throw std::runtime_error( "Invalid TPE::Type in Config." );

    Config.Energy.bendingWeight = config["Energy"]["bendingWeight"].as<RealType>();
    Config.Energy.elasticWeight = config["Energy"]["elasticWeight"].as<RealType>();
    Config.Energy.tpeWeight = config["Energy"]["tpeWeight"].as<RealType>();
    Config.Energy.barycenterWeight = config["Energy"]["barycenterWeight"].as<RealType>();
    Config.Energy.dirichletWeight = config["Energy"]["dirichletWeight"].as<RealType>();

    Config.Optimization.maxNumIterations = config["Optimization"]["maxNumIterations"].as<int>();
    Config.Optimization.minStepsize = config["Optimization"]["minStepsize"].as<RealType>();
    Config.Optimization.maxStepsize = config["Optimization"]["maxStepsize"].as<RealType>();
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
  TriMesh startMesh, secondMesh, initMesh;
  if ( !OpenMesh::IO::read_mesh( startMesh, Config.startFile ))
    throw std::runtime_error( "Failed to read file: " + Config.startFile );
  if ( !OpenMesh::IO::read_mesh( secondMesh, Config.secondFile ))
    throw std::runtime_error( "Failed to read file: " + Config.secondFile );
  if ( !Config.initFile.empty()) {
    if ( !OpenMesh::IO::read_mesh( initMesh, Config.initFile ))
      throw std::runtime_error( "Failed to read file: " + Config.initFile );
  }
  else {
    initMesh = startMesh;
  }

  // Topology of the mesh
  MeshTopologySaver Topology( startMesh );
  int numVertices = Topology.getNumVertices();

  // Geometry of the mesh
  VectorType Vertices_Start, Vertices_Second, Vertices_Init;
  getGeometry( startMesh, Vertices_Start );
  getGeometry( secondMesh, Vertices_Second );
  getGeometry( initMesh, Vertices_Init );

  std::cout << " .. numVertices = " << numVertices << std::endl;

  std::vector<int> dirichletIndices, nonDirichletIndices;

  dirichletIndices = Config.dirichletVertices;
  const auto numDirichletVertices = Config.dirichletVertices.size();
  Config.dirichletVertices.resize( 3 * numDirichletVertices );
  for ( int i = 0; i < numDirichletVertices; i++ ) {
    Config.dirichletVertices[numDirichletVertices + i] = numVertices + Config.dirichletVertices[i];
    Config.dirichletVertices[2 * numDirichletVertices + i] = 2 * numVertices + Config.dirichletVertices[i];
  }

  for ( int vertexIdx = 0; vertexIdx < numVertices; vertexIdx++ ) {
    if ( std::find( dirichletIndices.begin(), dirichletIndices.end(), vertexIdx ) == dirichletIndices.end())
      nonDirichletIndices.push_back( vertexIdx );
  }

  ShellDeformationType W( Topology, Config.Energy.bendingWeight );

  std::cout << " .. W = " << W( Vertices_Start, Vertices_Second ) << std::endl;

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

  std::cout << " .. TPE(1) = " << TPE( Vertices_Start ) << std::endl;
  std::cout << " .. TPE(2) = " << TPE( Vertices_Second ) << std::endl;

  VectorType s0 = Vertices_Start, s1 = Vertices_Second, s2 = Vertices_Second;

  saveAsPLY<VectorType>( Topology, s0, outputPrefix + "elast_exp_0.ply" );
  saveAsPLY<VectorType>( Topology, s1, outputPrefix + "elast_exp_1.ply" );

  for ( int t = 2; t <= Config.numSteps; t++ ) {
    std::cout << std::endl;
    std::cout << " .. Step " << t << ": " << std::endl;

    if ( !Config.initFile.empty())
      s2 = Vertices_Init;

    Exp2Energy<DefaultConfigurator> F( W, s0, s1 );
    Exp2Gradient<DefaultConfigurator> DF( W, s0, s1 );

    // NewOpt::NewtonMethod<DefaultConfigurator> Solver( F, DF, invDF, Config.Optimization.maxNumIterations, 1e-8,
    //                                                   NEWTON_OPTIMAL, SHOW_ALL, 0.1,
    //                                                   Config.Optimization.minStepsize,
    //                                                   Config.Optimization.maxStepsize );
    NewtonMethod<DefaultConfigurator> Solver( F, DF, Config.Optimization.maxNumIterations, 1e-8,
        NEWTON_OPTIMAL, SHOW_ALL, 0.1,
        Config.Optimization.minStepsize,
        Config.Optimization.maxStepsize );
    Solver.setBoundaryMask( Config.dirichletVertices );

    t_start = std::chrono::high_resolution_clock::now();
    Solver.solve( s2, s2 );
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << " .... Time: " << std::chrono::duration<double>( t_end - t_start ).count() << " seconds." << std::endl;
    saveAsPLY<VectorType>( Topology, s2, outputPrefix + "elast_exp_" + std::to_string(t) + ".ply" );

    s0 = s1;
    s1 = s2;
  }

  std::cout << " - outputPrefix: " << outputPrefix << std::endl;

  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
