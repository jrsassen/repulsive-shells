// This file is part of GOAST, a C++ library for variational methods in Geometry Processing
//
// Copyright (C) 2020 Behrend Heeren & Josua Sassen, University of Bonn <goast@ins.uni-bonn.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
/**
 * \file
 * \brief Hyperelastic membrane energy (for fixed reference domain given by edge lengths) and derivatives.
 * \author Heeren, Sassen
 */
#pragma once


//== INCLUDES =================================================================
#include <goast/Core/Auxiliary.h>
#include <goast/Core/LocalMeshGeometry.h>
#include <goast/Core/Topology.h>
#include <goast/NRIC/TrigonometryOperators.h>
#include <goast/DiscreteShells/AuxiliaryFunctions.h>

/**
 * \brief Hyperelastic membrane energy for fixed reference domain given by edge lengths
 * \author Heeren
 *
 * Same as NonlinearMembraneEnergy for fixed reference domain and free deformed shape.
 */
template<typename ConfiguratorType=DefaultConfigurator>
class NRICReferenceMembraneFunctional
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::RealType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  const MeshTopologySaver &_topology;
  VectorType _refFaceAreas, _refSqrEdgeLengths;
  VectorType _muWeights, _lambdaWeights;

public:

  NRICReferenceMembraneFunctional( const MeshTopologySaver &topology,
                                   const VectorType &ReferenceEdgeLengths,
                                   RealType Mu = 1.,
                                   RealType Lambda = 1. )
          : _topology( topology ),
            _refSqrEdgeLengths( ReferenceEdgeLengths.array().square()),
            _muWeights( VectorType::Constant( topology.getNumEdges(), Mu )),
            _lambdaWeights( VectorType::Constant( topology.getNumEdges(), Lambda )) {
    _refFaceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceEdgeLengths );
  }

  NRICReferenceMembraneFunctional( const MeshTopologySaver &topology,
                                   const VectorType &ReferenceEdgeLengths,
                                   const VectorType &MuWeights,
                                   const VectorType &LambdaWeights )
          : _topology( topology ),
            _refSqrEdgeLengths( ReferenceEdgeLengths.array().square()),
            _muWeights( MuWeights ),
            _lambdaWeights( LambdaWeights ) {
    _refFaceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceEdgeLengths );
  }

  void getLocalEnergies( const VectorType &Arg, VectorType &Dest ) const {
    if ( Arg.size() != 3 * _topology.getNumVertices())
      throw std::length_error( "NonlinearMembraneFunctional::getLocalEnergies(): sizes dont match!" );

    Dest.resize( _topology.getNumFaces());
    Dest.setZero();

    // precompute quantities of argument
    VectorType sqrFaceAreas;
    std::vector<VecType> dotProducts;
    getDotProductsAndSquaredFaceAreas<ConfiguratorType>( _topology, Arg, dotProducts, sqrFaceAreas );

    // run over all faces and comupte local contributions  
    for ( int faceIdx = 0; faceIdx < _topology.getNumFaces(); ++faceIdx ) {
      RealType volRef( _refFaceAreas[faceIdx] );
      // trace term = -1. *  \sum_{i =0,1,2} <e_{i+1}, e_{i+2}> |\bar e_i|^2, where \bar e_i are reference edges
      // note the signs! This is since we have dotProducts[faceIdx] = { <e_1, -e_2>, <-e_2, e_0>, <e_0, e_1> }
      RealType traceTerm( dotProducts[faceIdx][0] * _refSqrEdgeLengths[_topology.getEdgeOfTriangle( faceIdx, 0 )] +
                          dotProducts[faceIdx][1] * _refSqrEdgeLengths[_topology.getEdgeOfTriangle( faceIdx, 1 )] -
                          dotProducts[faceIdx][2] * _refSqrEdgeLengths[_topology.getEdgeOfTriangle( faceIdx, 2 )] );
      //
      RealType mu = _muWeights[faceIdx];
      RealType lambdaQuarter = _lambdaWeights[faceIdx] / 4.;
      Dest[faceIdx] = ( mu / 8. * traceTerm + lambdaQuarter * sqrFaceAreas[faceIdx] ) / volRef -
                      (( 0.5 * mu + lambdaQuarter ) * std::log( sqrFaceAreas[faceIdx] / ( volRef * volRef )) + mu +
                       lambdaQuarter ) * volRef;
    }
  }

  // energy evaluation
  void apply( const VectorType &Arg, RealType &Dest ) const override {
    if ( Arg.size() != 3 * _topology.getNumVertices())
      throw std::length_error( "NonlinearMembraneFunctional::apply(): sizes dont match!" );

    VectorType localEnergies;
    getLocalEnergies( Arg, localEnergies );
    Dest = localEnergies.sum();
  }

  void setMuWeights( const VectorType &muWeights ) {
    if ( muWeights.size() != _topology.getNumFaces())
      throw std::length_error( "NonlinearMembraneFunctional::setMuWeights(): sizes do not match!" );
    _muWeights = muWeights;
  }

  void setLambdaWeights( const VectorType &lambdaWeights ) {
    if ( lambdaWeights.size() != _topology.getNumFaces())
      throw std::length_error( "NonlinearMembraneFunctional:setLambdaWeights(): sizes do not match!" );
    _lambdaWeights = lambdaWeights;
  }

};

//! \brief First derivative of NonlinearMembraneFunctional (for fixed reference domain given by edge lengths)
//! \author Heeren
template<typename ConfiguratorType=DefaultConfigurator>
class NRICReferenceMembraneGradient
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::VectorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  const MeshTopologySaver &_topology;
  VectorType _muWeights, _lambdaWeights;
  VectorType _refFaceAreas, _refFactors;

public:
  NRICReferenceMembraneGradient( const MeshTopologySaver &topology,
                                 const VectorType &ReferenceEdgeLengths,
                                 RealType Mu = 1.,
                                 RealType Lambda = 1. )
          : _topology( topology ),
            _muWeights( Eigen::MatrixXd::Constant( topology.getNumEdges(), 1, Mu )),
            _lambdaWeights( Eigen::MatrixXd::Constant( topology.getNumEdges(), 1, Lambda )) {
    _refFaceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceEdgeLengths );

    _refFactors.resize( 3 * topology.getNumFaces());
    for ( int faceIdx = 0; faceIdx < topology.getNumFaces(); faceIdx++ ) {
      std::array<RealType, 3> lengths = { ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 0 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 1 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 2 )] };

      std::array<RealType, 3> sqLengths = { lengths[0] * lengths[0], lengths[1] * lengths[1], lengths[2] * lengths[2] };

      int a, b;

      // Opposite of c
      for ( int c: { 0, 1, 2 } ) {
        a = ( c + 1 ) % 3;
        b = ( c + 2 ) % 3;

        _refFactors[3 * faceIdx + c] = -( sqLengths[a] + sqLengths[b] - sqLengths[c] ) / ( 2 * _refFaceAreas[faceIdx] );
      }
    }
  }

  NRICReferenceMembraneGradient( const MeshTopologySaver &topology,
                                 const VectorType &ReferenceEdgeLengths,
                                 const VectorType &MuWeights,
                                 const VectorType &LambdaWeights )
          : _topology( topology ),
            _muWeights( MuWeights ),
            _lambdaWeights( LambdaWeights ) {
    _refFaceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceEdgeLengths );

    _refFactors.resize( 3 * topology.getNumFaces());
    for ( int faceIdx = 0; faceIdx < topology.getNumFaces(); faceIdx++ ) {
      std::array<RealType, 3> lengths = { ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 0 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 1 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 2 )] };

      std::array<RealType, 3> sqLengths = { lengths[0] * lengths[0], lengths[1] * lengths[1], lengths[2] * lengths[2] };

      int a, b;

      // Opposite of c
      for ( int c: { 0, 1, 2 } ) {
        a = ( c + 1 ) % 3;
        b = ( c + 2 ) % 3;

        _refFactors[3 * faceIdx + c] = -( sqLengths[a] + sqLengths[b] - sqLengths[c] ) / ( 2 * _refFaceAreas[faceIdx] );
      }
    }
  }

  //
  void apply( const VectorType &Argument, VectorType &Dest ) const override {

    int numV = _topology.getNumVertices();
    if ( Argument.size() != 3 * numV )
      throw BasicException( "NonlinearMembraneGradient::apply(): wrong size of dofs!" );
    if ( Dest.size() != 3 * numV )
      Dest.resize( 3 * numV );
    Dest.setZero();

    // get face areas and area gradients
    VectorType faceAreas;
    std::vector<VecType> gradPi, gradPj, gradPk, gradPl;
    getAreaGradients<ConfiguratorType>( _topology, Argument, faceAreas, gradPi, gradPj, gradPk, gradPl );

    // membrane contribution     
    for ( int faceIdx = 0; faceIdx < _topology.getNumFaces(); ++faceIdx ) {
      std::vector<int> nodesIdx( 3 );
      for ( int j = 0; j < 3; j++ )
        nodesIdx[j] = _topology.getNodeOfTriangle( faceIdx, j );

      RealType muHalfPlusLambdaQuarter = 0.5 * _muWeights[faceIdx] + 0.25 * _lambdaWeights[faceIdx];
      RealType factor = 2. * ( 0.25 * _lambdaWeights[faceIdx] * faceAreas[faceIdx] / _refFaceAreas[faceIdx] -
                               muHalfPlusLambdaQuarter * _refFaceAreas[faceIdx] / faceAreas[faceIdx] );
      for ( int j = 0; j < 3; j++ ) {
        // det part
        Dest[j * numV + nodesIdx[0]] += factor * gradPi[faceIdx][j];
        Dest[j * numV + nodesIdx[1]] += factor * gradPj[faceIdx][j];
        Dest[j * numV + nodesIdx[2]] += factor * gradPk[faceIdx][j];

        // trace part
        for ( int i = 0; i < 3; i++ ) {
          int nextIdx = ( i + 1 ) % 3;
          int prevIdx = ( i + 2 ) % 3;
          Dest[j * numV + nodesIdx[i]] += 0.25 * _muWeights[faceIdx] *
                                          ( _refFactors[3 * faceIdx + prevIdx] * gradPl[3 * faceIdx + prevIdx][j] -
                                            _refFactors[3 * faceIdx + nextIdx] * gradPl[3 * faceIdx + nextIdx][j] );
        }
      }
    }
  }

  //
  void setMuWeights( const VectorType &muWeights ) {
    if ( muWeights.size() != _topology.getNumFaces())
      throw BasicException( "NonlinearMembraneGradient::setMuWeights(): sizes do not match!" );
    _muWeights = muWeights;
  }

  //
  void setLambdaWeights( const VectorType &lambdaWeights ) {
    if ( lambdaWeights.size() != _topology.getNumFaces())
      throw BasicException( "NonlinearMembraneGradient:setLambdaWeights(): sizes do not match!" );
    _lambdaWeights = lambdaWeights;
  }
};

//! \brief Second derivative of NonlinearMembraneFunctional (for fixed reference domain given by edge lengths)
//! \author Heeren
template<typename ConfiguratorType=DefaultConfigurator>
class NRICReferenceMembraneHessian
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::SparseMatrixType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using MatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  using TripletListType = std::vector<TripletType>;

  const MeshTopologySaver &_topology;
  RealType _factor;
  VectorType _muWeights, _lambdaWeights;
  VectorType _refFaceAreas, _refFactors;

  mutable int _rowOffset, _colOffset;

public:
  NRICReferenceMembraneHessian( const MeshTopologySaver &topology,
                                const VectorType &ReferenceEdgeLengths,
                                const RealType Factor = 1.,
                                int rowOffset = 0,
                                int colOffset = 0,
                                RealType Mu = 1.,
                                RealType Lambda = 1. )
          : _topology( topology ),
            _factor( Factor ),
            _muWeights( Eigen::MatrixXd::Constant( topology.getNumEdges(), 1, Mu )),
            _lambdaWeights( Eigen::MatrixXd::Constant( topology.getNumEdges(), 1, Lambda )),
            _rowOffset( rowOffset ),
            _colOffset( colOffset ) {
    _refFaceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceEdgeLengths );

    _refFactors.resize( 3 * topology.getNumFaces());
    for ( int faceIdx = 0; faceIdx < topology.getNumFaces(); faceIdx++ ) {
      std::array<RealType, 3> lengths = { ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 0 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 1 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 2 )] };

      std::array<RealType, 3> sqLengths = { lengths[0] * lengths[0], lengths[1] * lengths[1], lengths[2] * lengths[2] };

      int a, b;

      // Opposite of c
      for ( int c: { 0, 1, 2 } ) {
        a = ( c + 1 ) % 3;
        b = ( c + 2 ) % 3;

        _refFactors[3 * faceIdx + c] = -( sqLengths[a] + sqLengths[b] - sqLengths[c] ) / ( 2 * _refFaceAreas[faceIdx] );
      }
    }
  }

  NRICReferenceMembraneHessian( const MeshTopologySaver &topology,
                                const VectorType &ReferenceEdgeLengths,
                                const VectorType &MuWeights,
                                const VectorType &LambdaWeights,
                                const RealType Factor = 1.,
                                int rowOffset = 0,
                                int colOffset = 0 )
          : _topology( topology ),
            _factor( Factor ),
            _muWeights( MuWeights ),
            _lambdaWeights( LambdaWeights ),
            _rowOffset( rowOffset ),
            _colOffset( colOffset ) {
    _refFaceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceEdgeLengths );

    _refFactors.resize( 3 * topology.getNumFaces());
    for ( int faceIdx = 0; faceIdx < topology.getNumFaces(); faceIdx++ ) {
      std::array<RealType, 3> lengths = { ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 0 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 1 )],
                                          ReferenceEdgeLengths[topology.getEdgeOfTriangle( faceIdx, 2 )] };

      std::array<RealType, 3> sqLengths = { lengths[0] * lengths[0], lengths[1] * lengths[1], lengths[2] * lengths[2] };

      int a, b;

      // Opposite of c
      for ( int c: { 0, 1, 2 } ) {
        a = ( c + 1 ) % 3;
        b = ( c + 2 ) % 3;

        _refFactors[3 * faceIdx + c] = -( sqLengths[a] + sqLengths[b] - sqLengths[c] ) / ( 2 * _refFaceAreas[faceIdx] );
      }
    }
  }

  void setRowOffset( int rowOffset ) const {
    _rowOffset = rowOffset;
  }

  void setColOffset( int colOffset ) const {
    _colOffset = colOffset;
  }

  //
  void apply( const VectorType &Argument, MatrixType &Dest ) const override {
    assembleHessian( Argument, Dest );
  }

  // assmeble Hessian matrix via triplet list
  void assembleHessian( const VectorType &Argument, MatrixType &Hessian ) const {
    int dofs = 3 * _topology.getNumVertices();
    if (( Hessian.rows() != dofs ) || ( Hessian.cols() != dofs ))
      Hessian.resize( dofs, dofs );
    Hessian.setZero();

    // set up triplet list
    TripletListType tripletList;
    // per face we have 3 active vertices, i.e. 9 combinations each producing a 3x3-matrix
    tripletList.reserve( 9 * 9 * _topology.getNumFaces());

    pushTriplets( Argument, tripletList );

    // fill matrix from triplets
    Hessian.setFromTriplets( tripletList.cbegin(), tripletList.cend());
  }

  //
  void pushTriplets( const VectorType &Argument, TripletListType &tripletList ) const override {

    if ( Argument.size() != 3 * _topology.getNumVertices())
      throw BasicException( "NonlinearMembraneHessian::pushTriplets(): argument has wrong size!" );

    // run over all faces
    for ( int faceIdx = 0; faceIdx < _topology.getNumFaces(); ++faceIdx ) {

      std::vector<int> nodesIdx( 3 );
      std::vector<VecType> nodes( 3 );
      VecType temp;
      for ( int j = 0; j < 3; j++ )
        nodesIdx[j] = _topology.getNodeOfTriangle( faceIdx, j );
      //! get deformed quantities
      for ( int j = 0; j < 3; j++ )
        getXYZCoord<VectorType, VecType>( Argument, nodes[j], nodesIdx[j] );
      RealType volDefSqr = getAreaSqr<RealType>( nodes[0], nodes[1], nodes[2] );
      RealType volDef = std::sqrt( volDefSqr );

      VecType traceFactors;
      for ( int i = 0; i < 3; i++ )
        traceFactors[i] = -0.25 * _muWeights[faceIdx] * _refFactors[3 * faceIdx + i];
      RealType muHalfPlusLambdaQuarter = 0.5 * _muWeights[faceIdx] + 0.25 * _lambdaWeights[faceIdx];
      RealType mixedFactor = 0.5 * _lambdaWeights[faceIdx] / _refFaceAreas[faceIdx] +
                             2. * muHalfPlusLambdaQuarter * _refFaceAreas[faceIdx] / volDefSqr;
      RealType areaFactor = 0.5 * _lambdaWeights[faceIdx] * volDef / _refFaceAreas[faceIdx] -
                            2. * muHalfPlusLambdaQuarter * _refFaceAreas[faceIdx] / volDef;

      // precompute area gradients
      std::vector<VecType> gradArea( 3 );
      for ( int i = 0; i < 3; i++ )
        getAreaGradient( nodes[( i + 1 ) % 3], nodes[( i + 2 ) % 3], nodes[i], gradArea[i] );

      // compute local matrices
      MatType tensorProduct, H, auxMat;

      // i==j
      for ( int i = 0; i < 3; i++ ) {
        getHessAreaKK( nodes[( i + 1 ) % 3], nodes[( i + 2 ) % 3], nodes[i], auxMat );
        tensorProduct.makeTensorProduct( gradArea[i], gradArea[i] );
        getWeightedMatrixSum( areaFactor, auxMat, mixedFactor, tensorProduct, H );
        H.addToDiagonal( traceFactors[( i + 1 ) % 3] + traceFactors[( i + 2 ) % 3] );
        localToGlobal( tripletList, nodesIdx[i], nodesIdx[i], H );
      }

      // i!=j
      for ( int i = 0; i < 3; i++ ) {
        getHessAreaIK( nodes[i], nodes[( i + 1 ) % 3], nodes[( i + 2 ) % 3], auxMat );
        tensorProduct.makeTensorProduct( gradArea[i], gradArea[( i + 2 ) % 3] );
        getWeightedMatrixSum( areaFactor, auxMat, mixedFactor, tensorProduct, H );
        H.addToDiagonal( -traceFactors[( i + 1 ) % 3] );
        localToGlobal( tripletList, nodesIdx[i], nodesIdx[( i + 2 ) % 3], H );
      }

    }
  }

  //
  void setMuWeights( const VectorType &muWeights ) {
    if ( muWeights.size() != _topology.getNumFaces())
      throw BasicException( "NonlinearMembraneHessian::setMuWeights(): sizes do not match!" );
    _muWeights = muWeights;
  }

  //
  void setLambdaWeights( const VectorType &lambdaWeights ) {
    if ( lambdaWeights.size() != _topology.getNumFaces())
      throw BasicException( "NonlinearMembraneHessian:setLambdaWeights(): sizes do not match!" );
    _lambdaWeights = lambdaWeights;
  }

protected:
  void localToGlobal( TripletListType &tripletList, int k, int l, const MatType &localMatrix ) const {
    int numV = _topology.getNumVertices();
    for ( int i = 0; i < 3; i++ )
      for ( int j = 0; j < 3; j++ )
        tripletList.push_back(
                TripletType( _rowOffset + i * numV + k, _colOffset + j * numV + l, _factor * localMatrix( i, j )));

    if ( k != l ) {
      for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
          tripletList.push_back(
                  TripletType( _rowOffset + i * numV + l, _colOffset + j * numV + k, _factor * localMatrix( j, i )));
    }
  }

};
