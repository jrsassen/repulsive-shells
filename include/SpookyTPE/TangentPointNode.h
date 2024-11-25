#pragma once

#include <goast/Core.h>
#include "ClusterTree.h"

namespace SpookyTPE {
template<typename ConfiguratorType=DefaultConfigurator>
class TangentPointNode : public ClusterTreeNode<ConfiguratorType>,
                         public TreeNodeBase<TangentPointNode<ConfiguratorType>> {
protected:
  //  typedef typename BaseNodeType::ConfiguratorType ConfiguratorType;
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  VecType m_Centroid;
  RealType m_Area;
  RealType m_squaredRadius;
  VecType m_Normal;

  RealType m_AreaDerivativeFactor;
  VecType m_CentroidDerivativeFactor;
  VecType m_NormalDerivativeFactor;

  using ClusterTreeNode<ConfiguratorType>::m_lowerBounds;
  using ClusterTreeNode<ConfiguratorType>::m_upperBounds;

public:
  // Resolve ambiguous inheritance
  using TreeNodeBase<TangentPointNode<ConfiguratorType>>::Children;
  using TreeNodeBase<TangentPointNode<ConfiguratorType>>::addChild;
  using TreeNodeBase<TangentPointNode<ConfiguratorType>>::removeChild;

  // Inherit constructors from underlying node type
  using ClusterTreeNode<ConfiguratorType>::ClusterTreeNode;

  void computeGeometricQuantities( const std::vector<VecType> &faceBarycenters,
                                   const VectorType &faceAreas,
                                   const std::vector<VecType> &faceNormals ) {
    m_AreaDerivativeFactor = 0;
    m_CentroidDerivativeFactor.setZero();
    m_NormalDerivativeFactor.setZero();

    m_Area = 0;
    m_Normal.setZero();
    m_Centroid.setZero();
    if ( Children().empty() ) {
      // Leaf node
      for ( const auto &fIdx: this->Vertices() ) {
        m_Area += faceAreas[fIdx];
        m_Normal += faceAreas[fIdx] * faceNormals[fIdx];
        m_Centroid += faceAreas[fIdx] * faceBarycenters[fIdx];
      }

      m_Normal /= m_Area;
      m_Centroid /= m_Area;
    }
    else {
      // Recursive call to children and summing their contributions
      for ( auto &child: Children() ) {
        child.computeGeometricQuantities( faceBarycenters, faceAreas, faceNormals );

        m_Area += child.Area();
        m_Normal += child.Area() * child.Normal();
        m_Centroid += child.Area() * child.Centroid();
      }

      m_Normal /= m_Area;
      m_Centroid /= m_Area;
    }

    //    m_Normal.normalize();

    // Diameter (or Radius??) of bounding box
    m_squaredRadius = 0;
    // for ( const int d: { 0, 1, 2 } ) {
    //   RealType mid = m_Centroid[d];
    //   RealType delta_max = std::fabs( m_upperBounds[d] - mid );
    //   RealType delta_min = std::fabs( mid - m_lowerBounds[d] );
    //   m_squaredRadius += ( delta_min <= delta_max ) ? delta_max * delta_max : delta_min * delta_min;
    // }
    for ( const int d: { 0, 1, 2 } ) {
      RealType delta = m_upperBounds[d] - m_lowerBounds[d];
      m_squaredRadius += delta * delta;
    }
  }

  void collectFaceDerivatives( const std::vector<VecType> &faceBarycenters,
                               const VectorType &faceAreas,
                               const std::vector<VecType> &faceNormals,
                               std::vector<RealType> &faceAreaGradients,
                               std::vector<VecType> &faceNormalGradients,
                               std::vector<VecType> &faceCenterGradients ) {
    if ( Children().empty() ) {
      // Leaf node
      if ( m_AreaDerivativeFactor != 0 ) {
        for ( const auto &fIdx: this->Vertices() ) {
          faceAreaGradients[fIdx] += m_AreaDerivativeFactor;
        }
      }

      if ( m_CentroidDerivativeFactor.norm() != 0 ) {
        for ( const auto &fIdx: this->Vertices() ) {
          faceCenterGradients[fIdx] += faceAreas[fIdx] / m_Area * m_CentroidDerivativeFactor;
          VecType localVec = ( faceBarycenters[fIdx] - m_Centroid ) / m_Area;
          faceAreaGradients[fIdx] += dotProduct( localVec,
                                                 m_CentroidDerivativeFactor );
        }
      }

      if ( m_NormalDerivativeFactor.norm() != 0 ) {
        for ( const auto &fIdx: this->Vertices() ) {
          faceNormalGradients[fIdx] += faceAreas[fIdx] / m_Area * m_NormalDerivativeFactor;
          VecType localVec = ( faceNormals[fIdx] - m_Normal ) / m_Area;
          faceAreaGradients[fIdx] += dotProduct( localVec, m_NormalDerivativeFactor );
        }
      }
    }
    else {
      for ( auto &child: Children() ) {
        if ( m_AreaDerivativeFactor != 0 ) {
          child.AreaDerivativeFactor() += m_AreaDerivativeFactor;
        }

        if ( m_CentroidDerivativeFactor.norm() != 0 ) {
          // derivative w.r.t. centroid
          child.CentroidDerivativeFactor() += child.Area() / m_Area * m_CentroidDerivativeFactor;

          // derivative w.r.t area
          VecType localVec = ( child.Centroid() - m_Centroid ) / m_Area;
          child.AreaDerivativeFactor() += dotProduct( localVec,
                                                      m_CentroidDerivativeFactor );
        }

        if ( m_NormalDerivativeFactor.norm() != 0 ) {
          // derivative w.r.t. centroid
          child.NormalDerivativeFactor() += child.Area() / m_Area * m_NormalDerivativeFactor;

          // derivative w.r.t area
          // m_Area * child.Normal() - m_Normal * m_Area * 1
          VecType localVec = ( child.Normal() - m_Normal ) / m_Area;
          child.AreaDerivativeFactor() += dotProduct( localVec, m_NormalDerivativeFactor );
        }

        child.collectFaceDerivatives( faceBarycenters, faceAreas, faceNormals,
                                      faceAreaGradients, faceNormalGradients, faceCenterGradients );
      }
    }
  }


  const RealType &Area() const {
    return m_Area;
  }

  const VecType &Normal() const {
    return m_Normal;
  }

  const VecType &Centroid() const {
    return m_Centroid;
  }

  const RealType &squaredDiameter() const {
    return m_squaredRadius;
  }

  RealType &AreaDerivativeFactor() {
    return m_AreaDerivativeFactor;
  }

  VecType &NormalDerivativeFactor() {
    return m_NormalDerivativeFactor;
  }

  VecType &CentroidDerivativeFactor() {
    return m_CentroidDerivativeFactor;
  }
};
}
