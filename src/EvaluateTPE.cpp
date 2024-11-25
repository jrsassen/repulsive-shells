#include <chrono>
#include <thread>

#include <yaml-cpp/yaml.h>
#include <goast/Core.h>
#include <goast/GeodesicCalculus.h>
#include <goast/DiscreteShells.h>

#include "ScaryTPE/TangentPointEnergy.h"

#include "SpookyTPE/AllPairsEnergy.h"
#include "SpookyTPE/BarnesHutEnergy.h"
#include "SpookyTPE/FastMultipoleEnergy.h"
#include "SpookyTPE/AdaptiveEnergy.h"

using VectorType = DefaultConfigurator::VectorType;
using VecType = DefaultConfigurator::VecType;
using RealType = DefaultConfigurator::RealType;
using MatrixType = DefaultConfigurator::SparseMatrixType;

using ShellDeformationType = ShellDeformation<DefaultConfigurator, NonlinearMembraneDeformation<DefaultConfigurator>, SimpleBendingDeformation<DefaultConfigurator> >;

int main( int argc, char *argv[] ) {
  // Config
  struct {
    std::string MeshFile = "../data/finger0.ply";

    struct {
      int alpha = 6;
      int beta = 12;
      RealType theta = 0.5;
      RealType thetaNear = 10.;
    } Energy;
  } Config;

  //region Read config
  if ( argc == 2 ) {
    YAML::Node config = YAML::LoadFile( argv[1] );

    Config.MeshFile = config["Data"]["startFile"].as<std::string>();

    Config.Energy.alpha = config["TPE"]["alpha"].as<int>();
    Config.Energy.beta = config["TPE"]["beta"].as<int>();
    Config.Energy.theta = config["TPE"]["theta"].as<RealType>();
    Config.Energy.thetaNear = config["TPE"]["thetaNear"].as<RealType>();
  }
  //endregion

  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_end = std::chrono::high_resolution_clock::now();

  // Read meshes
  TriMesh Mesh;
  if ( !OpenMesh::IO::read_mesh( Mesh, Config.MeshFile ))
    throw std::runtime_error( "Failed to read file: " + Config.MeshFile );


  //region Setup
  // Topology of the mesh
  MeshTopologySaver Topology( Mesh );
  int numVertices = Topology.getNumVertices();
  int numFaces = Topology.getNumFaces();

  std::cout << " .. numVertices = " << numVertices << std::endl;
  std::cout << " .. numFaces = " << numFaces << std::endl;

  // Geometry of the mesh
  VectorType Geometry;
  getGeometry( Mesh, Geometry );
  //endregion

  //region Energies
  // Tangent Point Energy and SobolevSlobodeckijMetric
  ScaryTPE::TangentPointEnergy<DefaultConfigurator> TPE( Topology, Config.Energy.alpha, Config.Energy.beta, 1., false,
                                                         Config.Energy.theta, Config.Energy.thetaNear );
  ScaryTPE::TangentPointEnergy<DefaultConfigurator> adaptiveTPE( Topology, Config.Energy.alpha, Config.Energy.beta, 1.,
                                                                 true,
                                                                 Config.Energy.theta,
                                                                 Config.Energy.thetaNear );

  SpookyTPE::AllPairsTangentPointEnergy<DefaultConfigurator> spAPTPE( Topology, Config.Energy.alpha, Config.Energy.beta,
                                                                      1. );
  SpookyTPE::BarnesHutTangentPointEnergy<DefaultConfigurator> spBHTPE( Topology, Config.Energy.alpha,
                                                                       Config.Energy.beta, 1.,
                                                                       Config.Energy.theta );
  SpookyTPE::FastMultipoleTangentPointEnergy<DefaultConfigurator> spFMTPE( Topology, Config.Energy.alpha,
                                                                           Config.Energy.beta, 1.,
                                                                           Config.Energy.theta );
  SpookyTPE::AdaptiveFastMultipoleTangentPointEnergy<DefaultConfigurator> spADTPE( Topology,
                                                                                   Config.Energy.alpha,
                                                                                   Config.Energy.beta,
                                                                                   1.,
                                                                                   Config.Energy.theta,
                                                                                   Config.Energy.thetaNear );


  //endregion

  int numEval = 5;

  RealType Value;
  VectorType Gradient;

  std::this_thread::sleep_for( std::chrono::seconds( 1 ) );

//  ObjectiveWrapper<DefaultConfigurator> J( spAPTPE );
//  ObjectiveGradientWrapper<DefaultConfigurator> DJ( spAPTPE );
//
//  ScalarValuedDerivativeTester<DefaultConfigurator>( J, DJ, 1.e-6 ).plotRandomDirections( Geometry, 10,
//                                                                                          "testAP" );
//  ObjectiveWrapper<DefaultConfigurator> JBH( spBHTPE );
//  ObjectiveGradientWrapper<DefaultConfigurator> DJBH( spBHTPE );
//
//  ScalarValuedDerivativeTester<DefaultConfigurator>( JBH, DJBH, 1.e-6 ).plotRandomDirections( Geometry, 10,
//                                                                                          "testBH" );
//  ObjectiveWrapper<DefaultConfigurator> JFM( spFMTPE );
//  ObjectiveGradientWrapper<DefaultConfigurator> DJFM( spFMTPE );
//
//  ScalarValuedDerivativeTester<DefaultConfigurator>( JFM, DJFM, 1.e-6 ).plotRandomDirections( Geometry, 10,"testFM" );

//  ObjectiveWrapper<DefaultConfigurator> JAD( spADTPE );
//  ObjectiveGradientWrapper<DefaultConfigurator> DJAD( spADTPE );
//
//  ScalarValuedDerivativeTester<DefaultConfigurator>( JAD, DJAD, 1.e-6 ).plotRandomDirections( Geometry, 100, "testAD" );

//  ObjectiveWrapper<DefaultConfigurator> JCP( spCPTPE );
//  ObjectiveGradientWrapper<DefaultConfigurator> DJCP( spCPTPE );
//
//  ScalarValuedDerivativeTester<DefaultConfigurator>( JCP, DJCP, 1.e-6 ).plotRandomDirections( Geometry, 10,"testCP" );

//  ObjectiveWrapper<DefaultConfigurator> JHen( TPE );
//  ObjectiveGradientWrapper<DefaultConfigurator> DJHen( TPE );
//
//  ScalarValuedDerivativeTester<DefaultConfigurator>( JHen, DJHen, 1.e-6 ).plotRandomDirections( Geometry, 10, "testHen" );

  std::cout << std::endl;
  std::cout
      << " ----------------------------------- "
      << "SpookyTPE"
      << " ----------------------------------- "
      << std::endl;

  t_start = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < numEval; i++ ) {
    spAPTPE.resetCache();
    Value = spAPTPE( Geometry );
    Gradient = spAPTPE.grad( Geometry );
  }
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. Internal all-pairs energy, Value = " << Value << std::endl;
  std::cout << " .. Internal all-pairs energy, Gradient.norm = " << Gradient.norm() << std::endl;
  std::cout << std::fixed << std::setprecision( 2 );
  std::cout << " .... Time for energy: "
            << std::chrono::duration<double, std::milli>( t_end - t_start ).count() / numEval
            << "ms." << std::endl;
  std::cout << std::endl;
  //  spAPTPE.printTimings();

  t_start = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < numEval; i++ ) {
    spBHTPE.resetCache();
    Value = spBHTPE( Geometry );
    Gradient = spBHTPE.grad( Geometry );
  }
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. Internal Barnes-Hut energy, Value = " << Value << std::endl;
  std::cout << " .. Internal Barnes-Hut energy, Gradient.norm = " << Gradient.norm() << std::endl;
  std::cout << std::fixed << std::setprecision( 2 );
  std::cout << " .... Time for energy: "
            << std::chrono::duration<double, std::milli>( t_end - t_start ).count() / numEval
            << "ms." << std::endl;
  std::cout << std::endl;
  // spBHTPE.printTimings();

  t_start = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < numEval; i++ ) {
    spFMTPE.resetCache();
    Value = spFMTPE( Geometry );
    Gradient = spFMTPE.grad( Geometry );
  }
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. Internal Fast Multipole energy, Value = " << Value << std::endl;
  std::cout << " .. Internal Fast Multipole energy, Gradient.norm = " << Gradient.norm() << std::endl;
  std::cout << std::fixed << std::setprecision( 2 );
  std::cout << " .... Time for energy: "
            << std::chrono::duration<double, std::milli>( t_end - t_start ).count() / numEval
            << "ms." << std::endl;
  std::cout << std::endl;
  // spFMTPE.printTimings();

  t_start = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < numEval; i++ ) {
    spADTPE.resetCache();
    Value = spADTPE( Geometry );
    Gradient = spADTPE.grad( Geometry );
  }
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. Internal adaptive energy, Value = " << Value << std::endl;
  std::cout << " .. Internal adaptive energy, Gradient.norm = " << Gradient.norm() << std::endl;
  std::cout << std::fixed << std::setprecision( 2 );
  std::cout << " .... Time for energy: "
            << std::chrono::duration<double, std::milli>( t_end - t_start ).count() / numEval
            << "ms." << std::endl;
  std::cout << std::endl;
  // spADTPE.printTimings();

  std::cout
      << " ----------------------------------- "
      << "ScaryTPE"
      << " ----------------------------------- "
      << std::endl;
  t_start = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < numEval; i++ ) {
    TPE.resetCache();
    Value = TPE( Geometry );
    Gradient = TPE.grad( Geometry );
  }
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. Repulsor non-adaptive energy, Value = " << Value << std::endl;
  std::cout << " .. Repulsor non-adaptive energy, Gradient.norm = " << Gradient.norm() << std::endl;
  std::cout << std::fixed << std::setprecision( 2 );
  std::cout << " .... Time for energy: "
            << std::chrono::duration<double, std::milli>( t_end - t_start ).count() / numEval
            << "ms." << std::endl;
  std::cout << std::endl;


  t_start = std::chrono::high_resolution_clock::now();
  for ( int i = 0; i < numEval; i++ ) {
    adaptiveTPE.resetCache();
    Value = adaptiveTPE( Geometry );
    Gradient = adaptiveTPE.grad( Geometry );
  }
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::scientific << std::setprecision( 6 );
  std::cout << " .. Repulsor adaptive energy, Value = " << Value << std::endl;
  std::cout << " .. Repulsor adaptive energy, Gradient.norm = " << Gradient.norm() << std::endl;
  std::cout << std::fixed << std::setprecision( 2 );
  std::cout << " .... Time for energy: "
            << std::chrono::duration<double, std::milli>( t_end - t_start ).count() / numEval
            << "ms." << std::endl;
  std::cout << std::endl;

}
