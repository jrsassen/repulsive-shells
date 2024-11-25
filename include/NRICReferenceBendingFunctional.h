// This file is part of GOAST, a C++ library for variational methods in Geometry Processing
//
// Copyright (C) 2020 Behrend Heeren & Josua Sassen, University of Bonn <goast@ins.uni-bonn.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
/**
 * \file
 * \brief Original Discrete Shell's bending energy (for fixed reference domain given as NRIC) and derivatives.
 * \author Heeren
 * \cite GrHiDeSc03
 */
#pragma once

//== INCLUDES =================================================================
#include <goast/Core/Auxiliary.h>
#include <goast/Core/LocalMeshGeometry.h>
#include <goast/Core/Topology.h>
#include <goast/DiscreteShells/AuxiliaryFunctions.h>

/**
 * \brief Simple bending energy with fixed reference domain given as NRIC (cf SimpleBendingEnergy with ActiveShellIsDeformed = true)
 * \author Heeren
 *
 * Realization of \f[ F[x] = \sum_{e \in E} \frac{ (\theta_e - \theta_e[x])^2 }{ d_e }\cdot l_e^2 \, ,\f]
 * where \f$ theta_e \f$ is the dihedral angle, \f$  l_e \f$  the edge length and  \f$  d_e \f$  the edge area at some edge e.
 *
 * All quantities of the reference domain (i.e. dihedral angles, squared edge lengths and edge areas)
 * are computed once in the constructor and stored in corresponding vectors.
 *
 * For a function evaluation \f$  F[x] \f$  we only need to compute the deformed dihedral angles \f$  (\theta_e[x])_e \f$  then.
 *
 * It is possible to define varying edge weights (default value is one for all edge weights)
 */
template<typename ConfiguratorType=DefaultConfigurator>
class NRICReferenceBendingFunctional
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::RealType> {

protected:
  using RealType = typename ConfiguratorType::RealType;

  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  const MeshTopologySaver &_topology;
  VectorType _refDihedralAngles, _refSqrEdgeLengths, _refEdgeAreas;
  RealType _weight;
  VectorType _edgeWeights;

public:
  NRICReferenceBendingFunctional( const MeshTopologySaver &topology,
                                  const VectorType &ReferenceNRIC,
                                  RealType Weight = 1. ) : _topology( topology ),
                                                           _weight( Weight ),
                                                           _edgeWeights(
                                                             VectorType::Constant( topology.getNumEdges(), 1.0 ) ) {
    _refDihedralAngles = ReferenceNRIC.tail( topology.getNumEdges() );
    _refSqrEdgeLengths = ReferenceNRIC.head( topology.getNumEdges() ).array().square();

    _refEdgeAreas.resize( topology.getNumEdges() );
    _refEdgeAreas.setZero();
    VectorType faceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceNRIC );
    for ( int edgeIdx = 0; edgeIdx < topology.getNumEdges(); ++edgeIdx ) {
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
    }
  }

  NRICReferenceBendingFunctional( const MeshTopologySaver &topology,
                                  const VectorType &ReferenceNRIC,
                                  const VectorType &EdgeWeights,
                                  RealType Weight = 1. ) : _topology( topology ),
                                                           _weight( Weight ),
                                                           _edgeWeights( EdgeWeights ) {
    if ( EdgeWeights.size() != topology.getNumEdges() )
      throw std::length_error( "NRICReferenceBendingFunctional::NRICReferenceBendingFunctional(): "
        "wrong size of edge weights vector!" );
    _refDihedralAngles = ReferenceNRIC.tail( topology.getNumEdges() );
    _refSqrEdgeLengths = ReferenceNRIC.head( topology.getNumEdges() ).array().square();

    _refEdgeAreas.resize( topology.getNumEdges() );
    _refEdgeAreas.setZero();
    VectorType faceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceNRIC );
    for ( int edgeIdx = 0; edgeIdx < topology.getNumEdges(); ++edgeIdx ) {
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
    }
  }

  //
  void setEdgeWeights( const VectorType &edgeWeights ) {
    if ( edgeWeights.size() != _topology.getNumEdges())
      throw std::length_error( "NRICReferenceBendingFunctional::setEdgeWeights(): sizes do not match!" );
    _edgeWeights = edgeWeights;
  }

  // energy evaluation
  void getLocalEnergies( const VectorType &Argument, VectorType &Dest ) const {

    if ( Argument.size() != 3 * _topology.getNumVertices())
      throw std::length_error( "NRICReferenceBendingFunctional::getLocalEnergies(): argument has wrong size!" );
    Dest.resize( _topology.getNumEdges());
    Dest.setZero();

    for ( int edgeIdx = 0; edgeIdx < _topology.getNumEdges(); ++edgeIdx ) {

      if ( !( _topology.isEdgeValid( edgeIdx )))
        continue;

      int pi( _topology.getAdjacentNodeOfEdge( edgeIdx, 0 )),
              pj( _topology.getAdjacentNodeOfEdge( edgeIdx, 1 )),
              pk( _topology.getOppositeNodeOfEdge( edgeIdx, 0 )),
              pl( _topology.getOppositeNodeOfEdge( edgeIdx, 1 ));

      // no bending at boundary edges
      if ( std::min( pl, pk ) < 0 )
        continue;

      // get deformed geometry
      VecType Pi, Pj, Pk, Pl;
      getXYZCoord<VectorType, VecType>( Argument, Pi, pi );
      getXYZCoord<VectorType, VecType>( Argument, Pj, pj );
      getXYZCoord<VectorType, VecType>( Argument, Pk, pk );
      getXYZCoord<VectorType, VecType>( Argument, Pl, pl );

      // compute deformed dihedral angle and energy
      RealType delTheta = getDihedralAngle( Pi, Pj, Pk, Pl ) - _refDihedralAngles[edgeIdx];
      Dest[edgeIdx] = _weight * _edgeWeights[edgeIdx] * delTheta * delTheta * _refSqrEdgeLengths[edgeIdx] /
                      _refEdgeAreas[edgeIdx];

#ifdef DEBUGMODE
      if( std::isnan( Dest[0] ) ){
          std::cerr << "NaN in simple bending functional in edge " << edgeIdx << "! " << std::endl;
          if( hasNanEntries(Argument) )
            std::cerr << "Argument has NaN entries! " << std::endl;
          throw BasicException("NRICReferenceBendingEnergy::getLocalEnergies(): NaN Error!");
      }
#endif
    }
  }

  // energy evaluation
  void apply( const VectorType &Argument, RealType &Dest ) const override {

    if ( Argument.size() != 3 * _topology.getNumVertices())
      throw std::length_error( "NRICReferenceBendingFunctional::apply(): argument has wrong size!" );

    // get local energy contributions and sum up
    VectorType localEnergies;
    getLocalEnergies( Argument, localEnergies );
    Dest = localEnergies.sum();
  }

  void setWeight( double Eta ) {
    _weight = Eta;
  }

};

//! \brief First derivative of NRICReferenceBendingFunctional
//! \author Heeren
template<typename ConfiguratorType=DefaultConfigurator>
class NRICReferenceBendingGradient
        : public BaseOp<typename ConfiguratorType::VectorType, typename ConfiguratorType::VectorType> {

  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using MatType = typename ConfiguratorType::MatType;

  const MeshTopologySaver &_topology;
  VectorType _refDihedralAngles, _refSqrEdgeLengths, _refEdgeAreas;
  mutable VectorType _defFaceArea, _defEdgeLengths;
  mutable std::vector<VecType> _defFaceNormal;
  RealType _weight;
  VectorType _edgeWeights;

public:
  NRICReferenceBendingGradient( const MeshTopologySaver &topology,
                                const VectorType &ReferenceNRIC,
                                RealType Weight = 1. ) : _topology( topology ),
                                                         _weight( Weight ),
                                                         _edgeWeights(
                                                           VectorType::Constant( topology.getNumEdges(), 1.0 ) ) {
    _refDihedralAngles = ReferenceNRIC.tail(topology.getNumEdges());
    _refSqrEdgeLengths = ReferenceNRIC.head(topology.getNumEdges()).array().square();

    _refEdgeAreas.resize(topology.getNumEdges());
    _refEdgeAreas.setZero();
    VectorType faceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceNRIC );
    for ( int edgeIdx = 0; edgeIdx < topology.getNumEdges(); ++edgeIdx ) {
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
    }
  }

  NRICReferenceBendingGradient( const MeshTopologySaver &topology,
                                const VectorType &ReferenceNRIC,
                                const VectorType &EdgeWeights,
                                RealType Weight = 1. ) : _topology( topology ),
                                                         _weight( Weight ),
                                                         _edgeWeights( EdgeWeights ) {
    if ( EdgeWeights.size() != topology.getNumEdges())
      throw std::length_error( "NRICReferenceBendingGradient::NRICReferenceBendingGradient(): "
                               "wrong size of edge weights vector!" );
    _refDihedralAngles = ReferenceNRIC.tail(topology.getNumEdges());
    _refSqrEdgeLengths = ReferenceNRIC.head(topology.getNumEdges()).array().square();

    _refEdgeAreas.resize(topology.getNumEdges());
    _refEdgeAreas.setZero();
    VectorType faceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceNRIC );
    for ( int edgeIdx = 0; edgeIdx < topology.getNumEdges(); ++edgeIdx ) {
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
    }
  }

  //
  void setEdgeWeights( const VectorType &edgeWeights ) {
    if ( edgeWeights.size() != _topology.getNumEdges())
      throw BasicException( "NRICReferenceBendingGradient::setEdgeWeights(): sizes do not match!" );
    _edgeWeights = edgeWeights;
  }

  void apply( const VectorType &Argument, VectorType &Dest ) const override {

    if ( Argument.size() != 3 * _topology.getNumVertices()) {
      std::cerr << "size of arg = " << Argument.size() << " vs. num pf dofs = " << 3 * _topology.getNumVertices()
                << std::endl;
      throw std::length_error( "NRICReferenceBendingGradient::apply(): argument has wrong size!" );
    }

    if ( Dest.size() != Argument.size())
      Dest.resize( Argument.size());
    Dest.setZero();

    // cache deformed normals and face areas
    _defFaceArea.resize( _topology.getNumFaces());
    _defFaceNormal.resize( _topology.getNumFaces());
    for ( int i = 0; i < _topology.getNumFaces(); i++ )
      _defFaceArea[i] = getNormalAndArea<ConfiguratorType>( _topology, i, Argument, _defFaceNormal[i] );

    // cache deformed edge lengths
    _defEdgeLengths.resize( _topology.getNumEdges());
    for ( int i = 0; i < _topology.getNumEdges(); i++ )
      _defEdgeLengths[i] = getEdgeLength<ConfiguratorType>( _topology, i, Argument );

    // run over all edges
    for ( int edgeIdx = 0; edgeIdx < _topology.getNumEdges(); ++edgeIdx ) {

      if ( !( _topology.isEdgeValid( edgeIdx )))
        continue;

      int pi( _topology.getAdjacentNodeOfEdge( edgeIdx, 0 )),
              pj( _topology.getAdjacentNodeOfEdge( edgeIdx, 1 )),
              pk( _topology.getOppositeNodeOfEdge( edgeIdx, 0 )),
              pl( _topology.getOppositeNodeOfEdge( edgeIdx, 1 ));

      // no bending at boundary edges
      if ( std::min( pl, pk ) < 0 )
        continue;

      //! now get the deformed values
      VecType Pi, Pj, Pk, Pl;
      getXYZCoord<VectorType, VecType>( Argument, Pi, pi );
      getXYZCoord<VectorType, VecType>( Argument, Pj, pj );
      getXYZCoord<VectorType, VecType>( Argument, Pk, pk );
      getXYZCoord<VectorType, VecType>( Argument, Pl, pl );

      // compute weighted differnce of dihedral angles
      RealType delTheta = _refDihedralAngles[edgeIdx] - getDihedralAngle( Pi, Pj, Pk, Pl );
      delTheta *= -2. * _weight * _edgeWeights[edgeIdx] * _refSqrEdgeLengths[edgeIdx] / _refEdgeAreas[edgeIdx];

      // compute first derivatives of dihedral angle
      VecType thetak, thetal, thetai, thetaj;
      getThetaGradK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), edgeIdx, thetak );
      getThetaGradK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), edgeIdx, thetal );
      getThetaGradI( edgeIdx, Pi, Pj, Pk, Pl, thetai );
      getThetaGradJ( edgeIdx, Pi, Pj, Pk, Pl, thetaj );

      // assemble in global vector
      for ( int i = 0; i < 3; i++ ) {
        Dest[i * _topology.getNumVertices() + pi] += delTheta * thetai[i];
        Dest[i * _topology.getNumVertices() + pj] += delTheta * thetaj[i];
        Dest[i * _topology.getNumVertices() + pk] += delTheta * thetak[i];
        Dest[i * _topology.getNumVertices() + pl] += delTheta * thetal[i];
      }


#ifdef DEBUGMODE
      if( hasNanEntries( Dest ) ){
  std::cerr << "NaN in simple bending gradient deformed in edge " << edgeIdx << "! " << std::endl;
        if( hasNanEntries(Argument) )
    std::cerr << "Argument has NaN entries! " << std::endl;
        throw BasicException("NRICReferenceBendingGradientDef::apply(): NaN Error!");
      }
#endif

    }
  }

  void setWeight( double Eta ) {
    _weight = Eta;
  }

protected:
  void getThetaGradK( int faceIdx, int edgeIdx, VecType &grad ) const {
    grad = _defFaceNormal[faceIdx];
    grad *= -0.5 * _defEdgeLengths[edgeIdx] / _defFaceArea[faceIdx];
  }

  void getThetaGradILeftPart( int faceIdx, int edgeIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk,
                              VecType &grad ) const {
    VecType e( Pj - Pi ), d( Pk - Pj );
    getThetaGradK( faceIdx, edgeIdx, grad );
    grad *= dotProduct( d, e ) / dotProduct( e, e );
  }

  void getThetaGradJLeftPart( int faceIdx, int edgeIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk,
                              VecType &grad ) const {
    VecType e( Pj - Pi ), a( Pi - Pk );
    getThetaGradK( faceIdx, edgeIdx, grad );
    grad *= dotProduct( a, e ) / dotProduct( e, e );
  }

  void getThetaGradI( int edgeIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, const VecType &Pl,
                      VecType &grad ) const {
    VecType temp;
    getThetaGradILeftPart( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), edgeIdx, Pi, Pj, Pk, grad );
    getThetaGradILeftPart( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), edgeIdx, Pi, Pj, Pl, temp );
    grad += temp;
  }

  void getThetaGradJ( int edgeIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, const VecType &Pl,
                      VecType &grad ) const {
    VecType temp;
    getThetaGradJLeftPart( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), edgeIdx, Pi, Pj, Pk, grad );
    getThetaGradJLeftPart( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), edgeIdx, Pi, Pj, Pl, temp );
    grad += temp;
  }

};


//! \brief Second derivative of NRICReferenceBendingFunctional
//! \author Heeren
//! TODO can be optimized further!!
template<typename ConfiguratorType=DefaultConfigurator>
class NRICReferenceBendingHessian
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
  // reference quantities that are precomputed once
  VectorType _refDihedralAngles, _refSqrEdgeLengths, _refEdgeAreas;

  // deformed quantities that are precomputed in each apply call 
  mutable VectorType _defFaceArea;
  mutable std::vector<VecType> _defFaceNormal;
  mutable VecType _edge, _thetak, _thetal;
  mutable MatType _edgeCrossOp, _edgeReflectionOp;
  mutable RealType _defEdgeLengthSqr, _defEdgeLength;

  VectorType _edgeWeights;
  RealType _factor;
  mutable int _rowOffset, _colOffset;


public:
  NRICReferenceBendingHessian( const MeshTopologySaver &topology,
                               const VectorType &ReferenceNRIC,
                               const RealType Weight = 1.,
                               int rowOffset = 0,
                               int colOffset = 0 ) : _topology( topology ),
                                                     _edgeWeights( VectorType::Constant( topology.getNumEdges(), 1.0 ) ),
                                                     _factor( Weight ),
                                                     _rowOffset( rowOffset ),
                                                     _colOffset( colOffset ) {
    _refDihedralAngles = ReferenceNRIC.tail(topology.getNumEdges());
    _refSqrEdgeLengths = ReferenceNRIC.head(topology.getNumEdges()).array().square();

    _refEdgeAreas.resize(topology.getNumEdges());
    _refEdgeAreas.setZero();
    VectorType faceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceNRIC );
    for ( int edgeIdx = 0; edgeIdx < topology.getNumEdges(); ++edgeIdx ) {
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
    }
  }

  NRICReferenceBendingHessian( const MeshTopologySaver &topology,
                               const VectorType &ReferenceNRIC,
                               const VectorType &EdgeWeights,
                               const RealType Weight = 1.,
                               int rowOffset = 0,
                               int colOffset = 0 ) : _topology( topology ), _edgeWeights( EdgeWeights ),
                                                     _factor( Weight ),
                                                     _rowOffset( rowOffset ), _colOffset( colOffset ) {
    _refDihedralAngles = ReferenceNRIC.tail(topology.getNumEdges());
    _refSqrEdgeLengths = ReferenceNRIC.head(topology.getNumEdges()).array().square();

    _refEdgeAreas.resize(topology.getNumEdges());
    _refEdgeAreas.setZero();
    VectorType faceAreas = TriangleAreaOp<ConfiguratorType>( topology )( ReferenceNRIC );
    for ( int edgeIdx = 0; edgeIdx < topology.getNumEdges(); ++edgeIdx ) {
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 0 )];
      _refEdgeAreas[edgeIdx] += faceAreas[topology.getAdjacentTriangleOfEdge( edgeIdx, 1 )];
    }
  }


  void setRowOffset( int rowOffset ) const {
    _rowOffset = rowOffset;
  }

  void setColOffset( int colOffset ) const {
    _colOffset = colOffset;
  }

  //
  void setEdgeWeights( const VectorType &edgeWeights ) {
    if ( edgeWeights.size() != _topology.getNumEdges())
      throw std::length_error( "NRICReferenceBendingHessian::setEdgeWeights(): sizes do not match!" );
    _edgeWeights = edgeWeights;
  }

  //
  void apply( const VectorType &defShell, MatrixType &Dest ) const override {
    assembleHessian( defShell, Dest );
  }

  // assmeble Hessian matrix via triplet list
  void assembleHessian( const VectorType &defShell, MatrixType &Hessian ) const {
    int dofs = 3 * _topology.getNumVertices();
    if (( Hessian.rows() != dofs ) || ( Hessian.cols() != dofs ))
      Hessian.resize( dofs, dofs );
    Hessian.setZero();

    // set up triplet list
    TripletListType tripletList;
    // per edge we have 4 active vertices, i.e. 16 combinations each producing a 3x3-matrix
    tripletList.reserve( 16 * 9 * _topology.getNumEdges());
    // fill matrix from triplets
    pushTriplets( defShell, tripletList );
    Hessian.setFromTriplets( tripletList.cbegin(), tripletList.cend());
  }

  // fill triplets
  void pushTriplets( const VectorType &defShell, TripletListType &tripletList ) const override {

    if ( defShell.size() != 3 * _topology.getNumVertices()) {
      std::cerr << "size of def = " << defShell.size() << " vs. num of dofs = " << 3 * _topology.getNumVertices()
                << std::endl;
      throw std::length_error( "NRICReferenceBendingHessian::pushTriplets(): sizes dont match!" );
    }

    // cache deformed normals and face areas
    _defFaceArea.resize( _topology.getNumFaces());
    _defFaceNormal.resize( _topology.getNumFaces());
    for ( int i = 0; i < _topology.getNumFaces(); i++ )
      _defFaceArea[i] = getNormalAndArea<ConfiguratorType>( _topology, i, defShell, _defFaceNormal[i] );

    // run over all edges and fill triplets
    for ( int edgeIdx = 0; edgeIdx < _topology.getNumEdges(); ++edgeIdx ) {

      if ( !( _topology.isEdgeValid( edgeIdx )))
        continue;

      int pi( _topology.getAdjacentNodeOfEdge( edgeIdx, 0 )),
              pj( _topology.getAdjacentNodeOfEdge( edgeIdx, 1 )),
              pk( _topology.getOppositeNodeOfEdge( edgeIdx, 0 )),
              pl( _topology.getOppositeNodeOfEdge( edgeIdx, 1 ));

      // no bending at boundary edges
      if ( std::min( pl, pk ) < 0 )
        continue;

      //! get defomed quantities
      VecType Pi, Pj, Pk, Pl, temp;
      getXYZCoord<VectorType, VecType>( defShell, Pi, pi );
      getXYZCoord<VectorType, VecType>( defShell, Pj, pj );
      getXYZCoord<VectorType, VecType>( defShell, Pk, pk );
      getXYZCoord<VectorType, VecType>( defShell, Pl, pl );

      // precompute edge quantities
      _edge = Pj - Pi;
      VecType edgeNormalized( _edge );
      getCrossOp( _edge, _edgeCrossOp );
      edgeNormalized.normalize();
      getReflection( edgeNormalized, _edgeReflectionOp );
      _defEdgeLengthSqr = dotProduct( _edge, _edge );
      _defEdgeLength = std::sqrt( _defEdgeLengthSqr );

      // compute difference in dihedral angles
      RealType factor = 2. * _refSqrEdgeLengths[edgeIdx] / _refEdgeAreas[edgeIdx];
      RealType delThetaDouble = factor * ( getDihedralAngle( Pi, Pj, Pk, Pl ) - _refDihedralAngles[edgeIdx] );

      // compute first derivatives of dihedral angle
      VecType thetai, thetaj;
      getThetaGradK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), _thetak );
      getThetaGradK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), _thetal );
      getThetaGradI( edgeIdx, Pi, Pj, Pk, Pl, thetai );
      getThetaGradJ( edgeIdx, Pi, Pj, Pk, Pl, thetaj );

      // now compute second derivatives of dihedral angle
      MatType tensorProduct, H, aux;

      //kk
      getHessThetaKK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), Pi, Pj, Pk, aux, 1.0 );
      tensorProduct.makeTensorProduct( _thetak, _thetak );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, aux, H );
      localToGlobal( tripletList, pk, pk, H, _edgeWeights[edgeIdx] );

      //ik & ki (Hki = Hik)
      MatType HessIKijk;
      getHessThetaIK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), Pi, Pj, Pk, HessIKijk );
      tensorProduct.makeTensorProduct( thetai, _thetak );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, HessIKijk, H );
      localToGlobal( tripletList, pi, pk, H, _edgeWeights[edgeIdx] );

      //jk & kj (Hkj = Hjk)
      MatType HessJKijk;
      getHessThetaJK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), Pi, Pj, Pk, HessJKijk );
      tensorProduct.makeTensorProduct( thetaj, _thetak );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, HessJKijk, H );
      localToGlobal( tripletList, pj, pk, H, _edgeWeights[edgeIdx] );

      //ll
      getHessThetaKK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), Pj, Pi, Pl, aux, -1.0 );
      tensorProduct.makeTensorProduct( _thetal, _thetal );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, aux, H );
      localToGlobal( tripletList, pl, pl, H, _edgeWeights[edgeIdx] );

      //il & li (Hli = Hil)
      getHessThetaJK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), Pj, Pi, Pl, aux );
      tensorProduct.makeTensorProduct( thetai, _thetal );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, aux, H );
      localToGlobal( tripletList, pi, pl, H, _edgeWeights[edgeIdx] );

      //jl & lj (Hlj = Hjl)
      MatType HessIKjil;
      getHessThetaIK( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), Pj, Pi, Pl, HessIKjil );
      tensorProduct.makeTensorProduct( thetaj, _thetal );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, HessIKjil, H );
      localToGlobal( tripletList, pj, pl, H, _edgeWeights[edgeIdx] );

      //kl/lk: Hkl = 0 and Hlk = 0
      tensorProduct.makeTensorProduct( _thetak, _thetal );
      tensorProduct *= factor;
      localToGlobal( tripletList, pk, pl, tensorProduct, _edgeWeights[edgeIdx] );

      //ii  
      VecType PkPj( Pk - Pj ), PlPj( Pl - Pj );
      getPartialHessThetaII( _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 ), Pi, Pj, Pl, PkPj, PlPj, aux );
      aux.addMultiple( HessIKijk, dotProduct( PkPj, _edge ) / _defEdgeLengthSqr );
      tensorProduct.makeTensorProduct( thetai, thetai );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, aux, H );
      localToGlobal( tripletList, pi, pi, H, _edgeWeights[edgeIdx] );

      //jj    
      VecType PkPi( Pk - Pi ), PlPi( Pl - Pi );
      getPartialHessThetaJJ( _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 ), Pi, Pj, Pk, PkPi, PlPi, aux );
      aux.addMultiple( HessIKjil, -1. * dotProduct( PlPi, _edge ) / _defEdgeLengthSqr );
      tensorProduct.makeTensorProduct( thetaj, thetaj );
      getWeightedMatrixSum( factor, tensorProduct, delThetaDouble, aux, H );
      localToGlobal( tripletList, pj, pj, H, _edgeWeights[edgeIdx] );

      //ij & ji (Hij = Hji)
      getHessThetaJI( Pi, Pj, Pk, Pl, HessIKjil, HessJKijk, H );
      H *= delThetaDouble;
      tensorProduct.makeTensorProduct( thetai, thetaj );
      H.addMultiple( tensorProduct, factor );
      localToGlobal( tripletList, pi, pj, H, _edgeWeights[edgeIdx] );
    }
  }

  void setWeight( double Eta ) {
    _factor = Eta;
  }

protected:
  void localToGlobal( TripletListType &tripletList, int k, int l, const MatType &localMatrix, RealType weight ) const {
    int numV = _topology.getNumVertices();
    for ( int i = 0; i < 3; i++ )
      for ( int j = 0; j < 3; j++ )
        tripletList.push_back( TripletType( _rowOffset + i * numV + k, _colOffset + j * numV + l,
                                            _factor * weight * localMatrix( i, j )));

    if ( k != l ) {
      for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ )
          tripletList.push_back( TripletType( _rowOffset + i * numV + l, _colOffset + j * numV + k,
                                              _factor * weight * localMatrix( j, i )));
    }
  }

//////////////////////////////////////////////////////////////

  void getThetaGradK( int faceIdx, VecType &grad ) const {
    grad = _defFaceNormal[faceIdx];
    grad *= -0.5 * _defEdgeLength / _defFaceArea[faceIdx];
  }

  void getThetaGradI( int edgeIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, const VecType &Pl,
                      VecType &grad ) const {
    grad.setZero();
    int fk = _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 );
    int fl = _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 );
    grad.addMultiple( _defFaceNormal[fk], -0.5 * dotProduct( Pk - Pj, _edge ) / ( _defEdgeLength * _defFaceArea[fk] ));
    grad.addMultiple( _defFaceNormal[fl], -0.5 * dotProduct( Pl - Pj, _edge ) / ( _defEdgeLength * _defFaceArea[fl] ));
  }

  void getThetaGradJ( int edgeIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, const VecType &Pl,
                      VecType &grad ) const {
    grad.setZero();
    int fk = _topology.getAdjacentTriangleOfEdge( edgeIdx, 0 );
    int fl = _topology.getAdjacentTriangleOfEdge( edgeIdx, 1 );
    grad.addMultiple( _defFaceNormal[fk], -0.5 * dotProduct( Pi - Pk, _edge ) / ( _defEdgeLength * _defFaceArea[fk] ));
    grad.addMultiple( _defFaceNormal[fl], -0.5 * dotProduct( Pi - Pl, _edge ) / ( _defEdgeLength * _defFaceArea[fl] ));
  }

//////////////////////////////////////////////////////////////

  void getAreaGradK( int faceIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, VecType &grad ) const {
    VecType a( Pi - Pk ), d( Pk - Pj ), e( Pj - Pi );
    RealType temp1( -0.25 * dotProduct( e, a ) / _defFaceArea[faceIdx] ), temp2(
            0.25 * dotProduct( e, d ) / _defFaceArea[faceIdx] );
    getWeightedVectorSum( temp1, d, temp2, a, grad );
  }

///////////////////////////////////////////////////////////////

  void getHessThetaKK( int faceIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, MatType &Hkk,
                       RealType factor ) const {
    RealType areaSqr = _defFaceArea[faceIdx] * _defFaceArea[faceIdx];
    VecType gradArea;
    getAreaGradK( faceIdx, Pi, Pj, Pk, gradArea );
    MatType auxMat;
    auxMat.makeTensorProduct( gradArea, _defFaceNormal[faceIdx] );
    getWeightedMatrixSum( factor * _defEdgeLength / ( 4. * areaSqr ), _edgeCrossOp, _defEdgeLength / areaSqr, auxMat,
                          Hkk );
  }

  void getHessThetaIK( int faceIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, MatType &Hik,
                       RealType normalFactor = 1. ) const {

    RealType area = _defFaceArea[faceIdx];
    RealType areaSqr = area * area;

    VecType e( Pj - Pi ), d( Pk - Pj ), gradArea;
    getAreaGradK( faceIdx, Pj, Pk, Pi, gradArea );

    MatType mat1, mat2, mat3;
    mat1.makeTensorProduct( e, _defFaceNormal[faceIdx] );
    getCrossOp( d, mat2 );
    getWeightedMatrixSum( normalFactor / ( 2. * area * _defEdgeLength ), mat1, _defEdgeLength / ( 4. * areaSqr ), mat2,
                          mat3 );

    mat1.makeTensorProduct( gradArea, _defFaceNormal[faceIdx] );
    getWeightedMatrixSum( 1., mat3, normalFactor * _defEdgeLength / areaSqr, mat1, Hik );
  }

  void getHessThetaJK( int faceIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, MatType &Hjk ) const {

    RealType area = _defFaceArea[faceIdx];
    RealType areaSqr = area * area;

    VecType e( Pi - Pj ), a( Pi - Pk ), gradArea;
    getAreaGradK( faceIdx, Pk, Pi, Pj, gradArea );

    MatType mat1, mat2, mat3;
    mat1.makeTensorProduct( e, _defFaceNormal[faceIdx] );
    getCrossOp( a, mat2 );
    getWeightedMatrixSum( 1. / ( 2. * area * _defEdgeLength ), mat1, _defEdgeLength / ( 4. * areaSqr ), mat2, mat3 );

    mat1.makeTensorProduct( gradArea, _defFaceNormal[faceIdx] );
    getWeightedMatrixSum( 1., mat3, _defEdgeLength / areaSqr, mat1, Hjk );
  }

//
  void getPartialHessThetaII( int faceIdx, const VecType &Pi, const VecType &Pj, const VecType &Pl, const VecType &PkPj,
                              const VecType &PlPj, MatType &Aux ) const {
    VecType temp;
    MatType tempMat;
    Aux.setZero();

    _edgeReflectionOp.mult( PkPj, temp );
    tempMat.makeTensorProduct( temp, _thetak );
    Aux.addMultiple( tempMat, -1. / _defEdgeLengthSqr );

    _edgeReflectionOp.mult( PlPj, temp );
    tempMat.makeTensorProduct( temp, _thetal );
    Aux.addMultiple( tempMat, -1. / _defEdgeLengthSqr );

    getHessThetaIK( faceIdx, Pi, Pj, Pl, tempMat, -1. );
    Aux.addMultiple( tempMat, -1. * dotProduct( PlPj, _edge ) / _defEdgeLengthSqr );
  }

//
  void getPartialHessThetaJJ( int faceIdx, const VecType &Pi, const VecType &Pj, const VecType &Pk, const VecType &PkPi,
                              const VecType &PlPi, MatType &Aux ) const {
    VecType temp;
    MatType tempMat;
    Aux.setZero();

    _edgeReflectionOp.mult( PlPi, temp );
    tempMat.makeTensorProduct( temp, _thetal );
    Aux.addMultiple( tempMat, -1. / _defEdgeLengthSqr );

    _edgeReflectionOp.mult( PkPi, temp );
    tempMat.makeTensorProduct( temp, _thetak );
    Aux.addMultiple( tempMat, -1. / _defEdgeLengthSqr );

    getHessThetaIK( faceIdx, Pj, Pi, Pk, tempMat, -1. );
    Aux.addMultiple( tempMat, dotProduct( PkPi, _edge ) / _defEdgeLengthSqr );
  }

//
  void getHessThetaJI( const VecType &Pi, const VecType &Pj, const VecType &Pk, const VecType &Pl,
                       const MatType &HessIKjil, const MatType &HessJKijk, MatType &Hji ) const {
    VecType d( Pk - Pj ), c( Pj - Pl ), grad;
    VecType diff( d - _edge ), sum( c + _edge );

    getWeightedVectorSum( dotProduct( _edge, d ), _thetak, -1. * dotProduct( _edge, c ), _thetal, grad );

    // Hess part
    getWeightedMatrixSumTransposed( dotProduct( _edge, d ) / _defEdgeLengthSqr, HessJKijk,
                                    -1. * dotProduct( _edge, c ) / _defEdgeLengthSqr, HessIKjil, Hji );

    MatType tensorProduct;
    tensorProduct.makeTensorProduct( grad, _edge );
    Hji.addMultiple( tensorProduct, -2. / ( _defEdgeLengthSqr * _defEdgeLengthSqr ));

    tensorProduct.makeTensorProduct( _thetak, diff );
    Hji.addMultiple( tensorProduct, 1. / _defEdgeLengthSqr );
    tensorProduct.makeTensorProduct( _thetal, sum );
    Hji.addMultiple( tensorProduct, -1. / _defEdgeLengthSqr );
  }

};
