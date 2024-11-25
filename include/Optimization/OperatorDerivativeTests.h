#pragma once

#include <iostream>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <random>

#include <goast/Core.h>
#include <goast/Optimization/LinearOperator.h>

template<typename ConfiguratorType=DefaultConfigurator>
class VectorValuedOperatorDerivativeTester {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;

  const BaseOp<VectorType, VectorType> &m_F;
  const MapToLinOp<ConfiguratorType> &m_DF;

  const int _dimOfRange;
  RealType _stepSize;
  int _numSteps;

public:
  VectorValuedOperatorDerivativeTester( const BaseOp<VectorType, VectorType> &F,
                                        const MapToLinOp<ConfiguratorType> &DF,
                                        const RealType stepSize,
                                        int dimOfRange = -1 )
          : m_F( F ),
            m_DF( DF ),
            _dimOfRange( dimOfRange ),
            _stepSize( stepSize ),
            _numSteps( 50 ) {}

  //
  void plotRandomDirections( const VectorType &testPoint, int numOfRandomDir, const std::string& saveNameStem,
                             const bool testNonNegEntriesOfHessianOnly = true ) const {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;

    auto JacobiOp = m_DF(testPoint);

    VectorType timeSteps( _numSteps ), energies( _numSteps ), derivs( _numSteps );
    VectorType outerDirection( dimRange ), innerDirection( numDofs ), gradientWRTOuterDirection( numDofs );

    std::srand( std::time( 0 )); // use current time as seed for random generator
    //VectorType randApproxDeriv ( numOfRandomDir );
    std::vector<int> directionOuterIndices( numOfRandomDir ), directionInnerIndices( numOfRandomDir );

    for ( int i = 0; i < numOfRandomDir; i++ ) {

      outerDirection.setZero();
      innerDirection.setZero();

      // outer index means, we test the ith component of  F = (f_1, ..., f_m), for 1 <= i <= m
      directionOuterIndices[i] = std::rand() % dimRange;
      outerDirection[directionOuterIndices[i]] = 1.;
      // inner index means, we test the jth entry of the component f_i
      directionInnerIndices[i] = std::rand() % numDofs;
      innerDirection[directionInnerIndices[i]] = 1.;

      // F = (f_1, ..., f_m)  =>  \nabla f_i = DF^T e_i (gradient of f_i is ith row of DF)
      RealType gateauxDerivative = (*JacobiOp)(innerDirection).dot(outerDirection);

      std::ostringstream saveName;
      saveName << saveNameStem << "_" << directionOuterIndices[i] << "_" << directionInnerIndices[i] << ".png";

      if ( testNonNegEntriesOfHessianOnly && ( std::abs( gateauxDerivative ) < 1.e-10 )) {
        i--;
        continue;
      }

      VectorType tempVec( dimRange );
      m_F.apply( testPoint, tempVec );
      RealType initialEnergy = tempVec.dot( outerDirection );

      for ( int j = 0; j < _numSteps; j++ ) {
        timeSteps[j] = ( j - ( _numSteps / 2 )) * _stepSize;
        VectorType shiftedPoint = testPoint + timeSteps[j] * innerDirection;
        m_F.apply( shiftedPoint, tempVec );
        energies[j] = tempVec.dot( outerDirection );
        derivs[j] = initialEnergy + timeSteps[j] * gateauxDerivative;
      }

      generatePNG( timeSteps, energies, derivs, saveName.str());

    }

  }

  //
  void plotAllDirections( const VectorType &testPoint, const std::string saveNameStem ) const {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;

    auto JacobiOp = m_DF(testPoint);

    VectorType timeSteps( _numSteps ), energies( _numSteps ), derivs( _numSteps );
    VectorType outerDirection( dimRange ), innerDirection( numDofs ), gradientWRTOuterDirection( numDofs );

    for ( int k = 0; k < dimRange; k++ ) {

      outerDirection.setZero();
      outerDirection[k] = 1.;

      for ( int i = 0; i < numDofs; i++ ) {
        innerDirection.setZero();
        innerDirection[i] = 1.;

        // F = (f_1, ..., f_n)  =>  \nabla f_i = DF^T e_i
        RealType gateauxDerivative = (*JacobiOp)(innerDirection).dot(outerDirection);

        VectorType tempVec( dimRange );
        m_F.apply( testPoint, tempVec );
        RealType initialEnergy = tempVec.dot( outerDirection );

        for ( int j = 0; j < _numSteps; j++ ) {
          timeSteps[j] = ( j - ( _numSteps / 2 )) * _stepSize;
          VectorType shiftedPoint = testPoint + timeSteps[j] * innerDirection;
          m_F.apply( shiftedPoint, tempVec );
          energies[j] = tempVec.dot( outerDirection );
          derivs[j] = initialEnergy + timeSteps[j] * gateauxDerivative;
        }

        std::ostringstream saveName;
        saveName << saveNameStem << "_" << k << "_" << i << ".png";
        generatePNG( timeSteps, energies, derivs, saveName.str());
      }

    }

  }

  //
  void
  plotSingleDirection( const VectorType &testPoint, const VectorType &innerDirection, const VectorType &outerDirection,
                       const std::string saveName ) const {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;

    if ( outerDirection.size() != dimRange )
      throw BasicException( "VectorValuedDerivativeTester::plotSingleDirections: outerDir has wrong size!" );

    if ( innerDirection.size() != numDofs )
      throw BasicException( "VectorValuedDerivativeTester::plotSingleDirections: innerDir has wrong size!" );

    auto JacobiOp = m_DF(testPoint);

    VectorType timeSteps( _numSteps ), energies( _numSteps ), derivs( _numSteps );

    RealType gateauxDerivative = (*JacobiOp)(innerDirection).dot(outerDirection);

    VectorType tempVec( dimRange );
    m_F.apply( testPoint, tempVec );
    RealType initialEnergy = tempVec.dot( outerDirection );

    for ( int j = 0; j < _numSteps; j++ ) {
      timeSteps[j] = ( j - ( _numSteps / 2 )) * _stepSize;
      VectorType shiftedPoint = testPoint + timeSteps[j] * innerDirection;
      m_F.apply( shiftedPoint, tempVec );
      energies[j] = tempVec.dot( outerDirection );
      derivs[j] = initialEnergy + timeSteps[j] * gateauxDerivative;
    }

    generatePNG( timeSteps, energies, derivs, saveName );

  }

  //
  void plotSingleDirection( const VectorType &testPoint, int innerDirection, int outerDirection,
                            const std::string saveName ) const {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;

    if ( !( outerDirection < dimRange ))
      throw BasicException( "VectorValuedDerivativeTester::plotSingleDirections: outerDir too large!" );

    if ( !( innerDirection < numDofs ))
      throw BasicException( "VectorValuedDerivativeTester::plotSingleDirections: innerDir too large!" );

    VectorType innerDir( dimRange ), outerDir( numDofs );
    innerDir.setZero();
    innerDir[innerDirection] = 1.;
    outerDir.setZero();
    outerDir[outerDirection] = 1.;
    plotSingleDirection( testPoint, innerDir, outerDir, saveName );
  }

  //
  RealType computeDiffQuotient( const VectorType &testPoint, const VectorType &outerDir, const VectorType &innerDir,
                                RealType tau ) const {

    // evalualte DF(m)
    int numDofs = innerDir.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;

    VectorType derivative( dimRange ), derivativeShifted( dimRange );
    m_F.apply( testPoint, derivative );

    //evaluate DF(m + h testDirection )
    VectorType shiftedPoint = testPoint + tau * innerDir;
    m_F.apply( shiftedPoint, derivativeShifted );
    return ( derivativeShifted.dot( outerDir ) - derivative.dot( outerDir )) / tau;
  }

  //
  void testSingleDirection( const VectorType &testPoint, const VectorType &outerDir, const VectorType &innerDir,
                            RealType tau = -1. ) const {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;

    RealType stepSize = ( tau < 0 ) ? _stepSize : tau;
    auto JacobiOp = m_DF(testPoint);
    RealType gateaux = (*JacobiOp)(innerDir).dot(outerDir);
    std::cerr << std::abs(( computeDiffQuotient( testPoint, outerDir, innerDir, stepSize ) - gateaux ) / gateaux )
              << std::endl;

  }

  //
  void testAllDirections( const VectorType &testPoint, bool detailedOutput = false, RealType tau = -1. ) const {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;
    RealType stepSize = ( tau < 0 ) ? _stepSize : tau;

    //evaluate DF and compute DF^T
    VectorType outerDir( dimRange ), innerDir( numDofs );

    auto JacobiOp = m_DF(testPoint);

    // run over outer directions

    for ( int i = 0; i < dimRange; i++ ) {
      VectorType error( numDofs ), diffQuot( numDofs ), resDeriv( numDofs );
      std::vector<std::tuple<int, int>> directionIndices( numDofs );
      outerDir.setZero();
      outerDir[i] = 1.;
      VectorType gradientWRTOuterDirection( numDofs );
      JacobiOp->applyTransposed(outerDir, gradientWRTOuterDirection);
      // run over inner directions
      for ( int j = 0; j < numDofs; j++ ) {
        innerDir.setZero();
        innerDir[j] = 1.;
        RealType gateaux = gradientWRTOuterDirection.dot( innerDir );
        resDeriv[j] = gateaux;
        diffQuot[j] = computeDiffQuotient( testPoint, outerDir, innerDir, stepSize );
        error[j] = std::abs(( diffQuot[j] - resDeriv[j] ) / resDeriv[j] );
        directionIndices[j] = std::make_tuple(i, j);
      }
      if ( detailedOutput ) {
        printVector( directionIndices, 10 );
        printVector( diffQuot, 10 );
        printVector( resDeriv, 10 );
      }
      printVector( error, 10 );
      std::cerr << std::endl;
    }
  }

  //
  void testRandomDirections( const VectorType &testPoint, unsigned numOfRandomDir, bool detailedOutput = false,
                             RealType tau = -1.,
                             bool testNonNegEntriesOfHessianOnly = true ) {

    int numDofs = testPoint.size();
    int dimRange = _dimOfRange < 0 ? testPoint.size() : _dimOfRange;
    std::srand( std::time( 0 )); // use current time as seed for random generator
    RealType stepSize = ( tau < 0 ) ? _stepSize : tau;

    VectorType outerDir( dimRange ), innerDir( numDofs ), error( numOfRandomDir );
    auto JacobiOp = m_DF(testPoint);
    RealType maxError = 0.;

    // diffQuotient
    VectorType randApproxDeriv( numOfRandomDir );

    VectorType diffQuot( numOfRandomDir ), resDeriv( numOfRandomDir );
    std::vector<std::string> directionIndices( numOfRandomDir );

    for ( unsigned i = 0; i < numOfRandomDir; ++i ) {
      int outerIdx = std::rand() % dimRange;
      int innerIdx = std::rand() % numDofs;
      outerDir.setZero();
      innerDir.setZero();
      outerDir[outerIdx] = 1.;
      innerDir[innerIdx] = 1.;

      RealType gateaux = (*JacobiOp)(innerDir).dot(outerDir);
//      RealType gateaux = (*JacobiOp)(outerDir).dot(innerDir);

      RealType diffQuotient = computeDiffQuotient( testPoint, outerDir, innerDir, stepSize );

      resDeriv[i] = gateaux;
      diffQuot[i] = diffQuotient;
      directionIndices[i] = std::to_string( innerIdx ) + "," + std::to_string( outerIdx );

      if ( testNonNegEntriesOfHessianOnly && std::abs( gateaux ) < 1.e-10 && std::abs( diffQuotient ) < 1.e-10 ) {
        i--;
        continue;
      }

      RealType denom = std::max( std::abs( diffQuotient ), std::abs( gateaux ));
      error[i] = denom > 1.e-10 ? std::abs(( diffQuotient - gateaux ) / denom ) : 0;
      if ( error[i] > maxError )
        maxError = error[i];
    }

    if ( detailedOutput ) {
      printVector( directionIndices, 10 );
      printVector( diffQuot, 10 );
      printVector( resDeriv, 10 );
    }

    printVector( error, 10 );
    std::cout << "max. error = " << maxError << std::endl << std::endl;
  }

  void setNumSteps( int Steps ) {
    _numSteps = Steps;
  }

  void setStepSize( RealType Stepsize ) {
    _stepSize = Stepsize;
  }

};

