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


template<typename ConfiguratorType>
class ShermanMorrisonOperator final : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;


  std::unique_ptr<LinearOperator<ConfiguratorType>> m_Ainv;

  VectorType m_u,m_v;
  VectorType m_Ainv_u;
  RealType m_denominator;
  const int m_dim;

public:
  explicit ShermanMorrisonOperator( std::unique_ptr<LinearOperator<ConfiguratorType>> &&Ainv,
                                    const VectorType &u, const VectorType &v,
                                    const std::vector<int> &fixedVariables ) : m_u( u ), m_v( v ),
                                                                               m_Ainv( std::move( Ainv ) ),
                                                                               m_dim( m_Ainv->rows() ) {
    // Assemble system matrix
    applyMaskToVector( fixedVariables, m_v );
    applyMaskToVector( fixedVariables, m_u );

    m_Ainv_u = ( *m_Ainv )( m_u );

    m_denominator = 1 + m_v.dot( m_Ainv_u );
  }


  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest.resize( Arg.size());
    Dest = ( *m_Ainv )( Arg );

    Dest -= m_Ainv_u * m_v.dot( Dest ) / m_denominator;
  }

  int rows() const override {
    return m_dim;
  }

  int cols() const override {
    return m_dim;
  }
};

template<typename ConfiguratorType>
class InverseMatrixOperator final : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;



  Eigen::UmfPackLU<SparseMatrixType> m_Solver;
  SparseMatrixType m_localA;
  const int m_dim;

public:
  explicit InverseMatrixOperator( SparseMatrixType &&A, const std::vector<int> &fixedVariables )
          : m_dim( A.rows()) {
    assert( A.rows() == A.cols() && "InverseMatrixOperator: Operator has to be quadratic" );
    m_localA.swap(A);

    // Assemble system matrix
    applyMaskToMatrix( fixedVariables, m_localA );

    // Prepare solver
    m_Solver.compute( m_localA );
  }


  void apply( const VectorType &Arg, VectorType &Dest ) const override {
    Dest = m_Solver.solve( Arg );
  }

  int rows() const override {
    return m_dim;
  }

  int cols() const override {
    return m_dim;
  }
};

template<typename ConfiguratorType>
class DifferenceExp2Energy final : public BaseOp<typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;

  const BaseOp<VectorType, RealType> &m_V;
  const BaseOp<VectorType, VectorType> &m_DV;
  const VectorType &_shape0;
  const VectorType &_shape1;
  VectorType _constVecPart;
  RealType _constRealPart;
  int _numDofs;

public:
  DifferenceExp2Energy( const BaseOp<VectorType, RealType> &V,
                        const BaseOp<VectorType, VectorType> &DV,
                        const VectorType &shape0,
                        const VectorType &shape1 ) :
          m_V( V ), m_DV( DV ), _shape0( shape0 ), _shape1( shape1 ), _numDofs( shape0.size()) {

    _constVecPart.resize( _numDofs );
    m_DV.apply( _shape1, _constVecPart );

    _constRealPart = 4 * m_V( _shape1 ) - 2 * m_V( _shape0 );
  }


  //! The vertex positions of S_2 are given as argument.
  void apply( const VectorType &shape2, VectorType &Dest ) const override {
    if ( shape2.size() != _numDofs )
      throw std::length_error( "Exp2Energy::apply(): arg has wrong size!" );
    if ( Dest.size() != _numDofs )
      Dest.resize( _numDofs );

    // add constant partial
    Dest = _constVecPart * ( _constRealPart - 2 * m_V( shape2 ));
  }
};




template<typename ConfiguratorType>
class BarycenterExp2Energy final : public BaseOp<typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;

  const VectorType &m_shape0;
  const VectorType &m_shape1;
  VecType m_bary0, m_bary1;
  int m_numVertices;

  std::vector<int> m_barycenterIndices;

public:
  BarycenterExp2Energy( const VectorType &shape0,
                        const VectorType &shape1,
                        const std::vector<int> &barycenterIndices ) : m_shape0( shape0 ), m_shape1( shape1 ),
                                                                      m_numVertices( shape0.size() / 3 ),
                                                                      m_barycenterIndices( barycenterIndices ) {
//    m_barycenterIndices.resize( m_numVertices, 0 );
//    std::iota( m_barycenterIndices.begin(), m_barycenterIndices.end(), 0 );

    VecType p;
    for ( int i: m_barycenterIndices ) {
      getXYZCoord( shape0, p, i );
      m_bary0 += p;
      getXYZCoord( shape1, p, i );
      m_bary1 += p;
    }
  }


  //! The vertex positions of S_2 are given as argument.
  void apply( const VectorType &shape2, VectorType &Dest ) const override {
    if ( shape2.size() != 3 * m_numVertices )
      throw std::length_error( "BarycenterExp2Energy::apply(): arg has wrong size!" );
    if ( Dest.size() != 3 * m_numVertices )
      Dest.resize( 3 * m_numVertices );
    Dest.setZero();

    VecType p, bary2;
    for ( int i: m_barycenterIndices ) {
      getXYZCoord( shape2, p, i );
      bary2 += p;
    }

    VecType diff0 = m_bary1 - m_bary0;
    VecType diff1 = bary2 - m_bary1;
    for ( const int i: m_barycenterIndices ) {
      for ( int d: { 0, 1, 2 } ) {
        Dest[d * m_numVertices + i] += diff0[d];
        Dest[d * m_numVertices + i] -= diff1[d];
      }
    }

    Dest *= 4. / m_barycenterIndices.size();
  }
};

template<typename ConfiguratorType>
class DirichletExp2Energy final : public BaseOp<typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;

  const VectorType &m_shape0;
  const VectorType &m_shape1;
  VecType m_bary0, m_bary1;
  int m_numVertices;

  const std::vector<int> &m_constrainedVertices;

public:
  DirichletExp2Energy( const VectorType &shape0,
                       const VectorType &shape1,
                       const std::vector<int> &constrainedVertices ) : m_shape0( shape0 ), m_shape1( shape1 ),
                                                                       m_numVertices( shape0.size() / 3 ),
                                                                       m_constrainedVertices( constrainedVertices ) {
//    m_barycenterIndices.resize( m_numVertices, 0 );
//    std::iota( m_barycenterIndices.begin(), m_barycenterIndices.end(), 0 );

  }


  //! The vertex positions of S_2 are given as argument.
  void apply( const VectorType &shape2, VectorType &Dest ) const override {
    if ( shape2.size() != 3 * m_numVertices )
      throw std::length_error( "BarycenterExp2Energy::apply(): arg has wrong size!" );
    if ( Dest.size() != 3 * m_numVertices )
      Dest.resize( 3 * m_numVertices );
    Dest.setZero();

    Dest.setZero();

    for ( int i: m_constrainedVertices ) {
      VecType p, q, r, displacement;

      // 0
      getXYZCoord( m_shape0, p, i );
      getXYZCoord( m_shape1, q, i );
      getXYZCoord( shape2, r, i );
      displacement = (q - p) - (r - q);

      for ( int j = 0; j < 3; j++ )
        Dest[j * m_numVertices + i] += displacement[j];
    }
  }
};


template<typename ConfiguratorType>
class InverseCombinedExp2Gradient final : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const BaseOp<VectorType, RealType> &m_V;
  const BaseOp<VectorType, VectorType> &m_DV;
  const VectorType &_shape1;
  const VectorType _constVecPart;
  const RealType m_elasticWeight, m_tpeWeight,m_barycenterWeight,m_dirichletWeight;
  const std::vector<int> &m_fixedVariables;
  const std::vector<int> &m_barycenterIndices;
  const int m_numVertices;
  VectorType m_oneX, m_oneY, m_oneZ;

public:
  InverseCombinedExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                               const BaseOp<VectorType, RealType> &V,
                               const BaseOp<VectorType, VectorType> &DV,
                               const VectorType &shape0,
                               const VectorType &shape1,
                               const std::vector<int> &fixedVariables,
                               const std::vector<int> &barycenterIndices,
                               RealType elasticWeight = 1.,
                               RealType tpeWeight = 1.,
                               RealType dirichletWeight = 0.,
                               RealType barycenterWeight = 1. ) : m_W( W ), m_V( V ), m_DV( DV ), _shape1( shape1 ),
                                                                  _constVecPart( -2. * tpeWeight * DV( shape1 ) ),
                                                                  m_elasticWeight( elasticWeight ),
                                                                  m_tpeWeight( tpeWeight ),
                                                                  m_barycenterWeight( barycenterWeight ),
                                                                  m_dirichletWeight( dirichletWeight ),
                                                                  m_fixedVariables( fixedVariables ),
                                                                  m_barycenterIndices( barycenterIndices ),
                                                                  m_numVertices( _shape1.size() / 3 ),
                                                                  m_oneX( 3 * m_numVertices ),
                                                                  m_oneY( 3 * m_numVertices ),
                                                                  m_oneZ( 3 * m_numVertices ) {
    m_oneX.setZero();
    m_oneY.setZero();
    m_oneZ.setZero();

    for ( int i: m_barycenterIndices ) {
      m_oneX[i] = 1.;
      m_oneY[m_numVertices + i] = 1.;
      m_oneZ[2 * m_numVertices + i] = 1.;
    }
  }


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );
    A *= m_elasticWeight;

    std::vector<int> localFixed;

    if (m_fixedVariables.empty()) {
      for (int i = 0; i < A.rows();i++) {
        A.coeffRef(i,i) += 1.e-6;
      }
    }
    else {
      if (m_dirichletWeight == 0.) {
        localFixed = m_fixedVariables;
      }
      else {
        for (int i : m_fixedVariables) {
          A.coeffRef(i,i) -= m_dirichletWeight;
        }
      }
    }

    VectorType changingPart = m_DV( Point );

    // Inverse of elastic part
    auto Ainv = std::make_unique<InverseMatrixOperator<ConfiguratorType>>( std::move( A ), localFixed );

    // Sherman-Morrison for TPE
    auto SMtpe = std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( Ainv ), _constVecPart, changingPart, localFixed );

    // Sherman-Morrison for each component of barycenter path energy
    auto SMbX = std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( SMtpe ), -m_barycenterWeight * 4. / m_barycenterIndices.size() * m_oneX, m_oneX, localFixed );
    auto SMbY = std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( SMbX ), -m_barycenterWeight * 4. / m_barycenterIndices.size() * m_oneY, m_oneY, localFixed );
    return std::make_unique<ShermanMorrisonOperator<ConfiguratorType>>(
      std::move( SMbY ), -m_barycenterWeight * 4. / m_barycenterIndices.size() * m_oneZ, m_oneZ, localFixed );
  }

};


template<typename ConfiguratorType>
class CombinedExp2Gradient final : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const BaseOp<VectorType, RealType> &m_V;
  const BaseOp<VectorType, VectorType> &m_DV;
  const VectorType &_shape1;
  VectorType _constVecPart;
  const RealType m_elasticWeight, m_tpeWeight,m_barycenterWeight,m_dirichletWeight;
  const std::vector<int> &m_fixedVariables;
  const std::vector<int> &m_barycenterIndices;
  const int m_numVertices;
public:
  CombinedExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                               const BaseOp<VectorType, RealType> &V,
                               const BaseOp<VectorType, VectorType> &DV,
                               const VectorType &shape0,
                               const VectorType &shape1,
                               const std::vector<int> &fixedVariables,
                               const std::vector<int> &barycenterIndices,
                               RealType elasticWeight = 1.,
                               RealType tpeWeight = 1.,
                              RealType dirichletWeight = 0.,
                               RealType barycenterWeight = 1. ) :
          m_W( W ), m_V( V ), m_DV( DV ), _shape1( shape1 ),
          _constVecPart( -2. * tpeWeight * DV( shape1 )), m_elasticWeight( elasticWeight ), m_tpeWeight( tpeWeight ), m_barycenterWeight(barycenterWeight),m_dirichletWeight(dirichletWeight),
          m_fixedVariables( fixedVariables ),
          m_barycenterIndices( barycenterIndices ), m_numVertices(_shape1.size() / 3)  {
    applyMaskToVector( m_fixedVariables, _constVecPart );
  }


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );
    A *= m_elasticWeight;



    VectorType changingPart = m_DV( Point );

    std::vector<int> localFixed;

    if (m_fixedVariables.empty()) {
      for (int i = 0; i < A.rows();i++) {
        A.coeffRef(i,i) += 1.e-6;
      }
    }
    else {
      if (m_dirichletWeight == 0.) {
        applyMaskToMatrix( m_fixedVariables, A );
        applyMaskToVector( m_fixedVariables, changingPart );
      }
      else {
        for (int i : m_fixedVariables) {
          A.coeffRef(i,i) -= m_dirichletWeight;
        }
      }
    }


    std::unique_ptr<LinearOperator<ConfiguratorType>> energyOp = std::make_unique<MatrixOperator<ConfiguratorType>>( std::move(A) );
    std::unique_ptr<LinearOperator<ConfiguratorType>> diffOp = std::make_unique<RankOneOperator<ConfiguratorType>>( _constVecPart, changingPart );
    std::unique_ptr<LinearOperator<ConfiguratorType>> baryOp = std::make_unique<BarycenterHessianOperator<ConfiguratorType>>( m_numVertices, m_barycenterIndices, -m_barycenterWeight * 4. / m_barycenterIndices.size() );

    VectorType Weights(3);
    Weights << 1,1,1;
    return std::make_unique<LinearlyCombinedOperator<ConfiguratorType>>( Weights, std::move( energyOp ), std::move( diffOp ), std::move( baryOp ));
  }

};


template<typename ConfiguratorType>
class OperatorExp2Gradient final : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const VectorType &_shape1;
  const std::vector<int> &m_fixedVariables;

public:
  OperatorExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                        const VectorType &shape0,
                        const VectorType &shape1,
                        const std::vector<int> &fixedVariables) :
          m_W( W ), _shape1( shape1 ), m_fixedVariables( fixedVariables ) {
  }


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );

    applyMaskToMatrix( m_fixedVariables, A );

    return std::make_unique<MatrixOperator<ConfiguratorType>>( std::move(A) );
  }

};

template<typename ConfiguratorType>
class InverseExp2Gradient : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const DeformationBase<ConfiguratorType> &m_W;
  const VectorType &_shape1;
  const std::vector<int> &m_fixedVariables;

public:
  InverseExp2Gradient( const DeformationBase<ConfiguratorType> &W,
                               const VectorType &shape0,
                               const VectorType &shape1,
                               const std::vector<int> &fixedVariables ) :
          m_W( W ), _shape1( shape1 ),
          m_fixedVariables( fixedVariables ) {}


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    SparseMatrixType A;
    m_W.applyMixedHessian( _shape1, Point, A, false );

    return std::make_unique<InverseMatrixOperator<ConfiguratorType>>( std::move( A ), m_fixedVariables );
  }

};


int main( int argc, char *argv[] ) {
  //region Config
  enum TPEType {
    SPOOKY,
    SCARY
  };

  std::unordered_map<std::string, TPEType> const TPETypeTable = {
    { "Spooky", TPEType::SPOOKY },
    { "Scary",    TPEType::SCARY },
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
      RealType theta = 0.5;
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

  saveAsPLY<VectorType>( Topology, s0, outputPrefix + "comb_exp_0.ply" );
  saveAsPLY<VectorType>( Topology, s1, outputPrefix + "comb_exp_1.ply" );

  for ( int t = 2; t <= Config.numSteps; t++ ) {
    std::cout << std::endl;
    std::cout << " .. Step " << t << ": " << std::endl;

    if ( !Config.initFile.empty())
      s2 = Vertices_Init;

    // Vector-Valued Equation (confusingly called energy)
    Exp2Energy<DefaultConfigurator> Felast( W, s0, s1 );
    DifferenceExp2Energy<DefaultConfigurator> Ftpe( TPE, TPG, s0, s1 );
    BarycenterExp2Energy<DefaultConfigurator> Fbary( s0, s1, nonDirichletIndices );
    DirichletExp2Energy<DefaultConfigurator> Fdir( s0, s1, dirichletIndices );

    VectorType Weights( 4 );
    Weights << Config.Energy.elasticWeight, Config.Energy.tpeWeight, Config.Energy.dirichletWeight,
        Config.Energy.barycenterWeight;

    AdditionGradient<DefaultConfigurator> F( Weights, Felast, Ftpe, Fdir, Fbary );

    // Derivatives (matrix valued) and its inverse
    CombinedExp2Gradient<DefaultConfigurator> DF( W, TPE, TPG, s0, s1,
                                                  Config.dirichletVertices, nonDirichletIndices,
                                                  Config.Energy.elasticWeight,
                                                  Config.Energy.tpeWeight,
                                                  Config.Energy.dirichletWeight,
                                                  Config.Energy.barycenterWeight );

    InverseCombinedExp2Gradient<DefaultConfigurator> invDF( W, TPE, TPG, s0, s1,
                                                            Config.dirichletVertices, nonDirichletIndices,
                                                            Config.Energy.elasticWeight,
                                                            Config.Energy.tpeWeight,
                                                            Config.Energy.dirichletWeight,
                                                            Config.Energy.barycenterWeight );

    NewOpt::NewtonMethod<DefaultConfigurator> Solver( F, DF, invDF, Config.Optimization.maxNumIterations, 1e-8,
                                                      NEWTON_OPTIMAL, SHOW_ALL, 0.1,
                                                      Config.Optimization.minStepsize,
                                                      Config.Optimization.maxStepsize );
    if ( Config.Energy.dirichletWeight == 0. )
      Solver.setBoundaryMask( Config.dirichletVertices );

    t_start = std::chrono::high_resolution_clock::now();
    Solver.solve( s2, s2 );
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << " .... Time: " << std::chrono::duration<double, std::ratio<1> >( t_end - t_start ).count()
              << " seconds." << std::endl;
    saveAsPLY<VectorType>( Topology, s2, outputPrefix + "comb_exp_" + std::to_string(t) + ".ply" );

    s0 = s1;
    s1 = s2;
  }

  std::cout << " - outputPrefix: " << outputPrefix << std::endl;

  std::cout.rdbuf( output_cout.rdbuf());
  std::cerr.rdbuf( output_cerr.rdbuf());
}
