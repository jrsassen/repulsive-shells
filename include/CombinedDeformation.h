#pragma once

template<typename ConfiguratorType=DefaultConfigurator>
class CombinedDeformation : public DeformationBase<ConfiguratorType> {

protected:
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using TripletType = typename ConfiguratorType::TripletType;
  using TripletListType = std::vector<TripletType>;

  const DeformationBase<ConfiguratorType> &m_Wa;
  const DeformationBase<ConfiguratorType> &m_Wb;

  RealType m_bendWeight, m_memWeight;

public:
  CombinedDeformation( const DeformationBase<ConfiguratorType> &Wa, RealType bendWeight,
                       const DeformationBase<ConfiguratorType> &Wb ) : m_Wa( Wa ), m_Wb( Wb ),
                                                                       m_bendWeight( bendWeight ),
                                                                       m_memWeight( 1. ) {}

  CombinedDeformation( RealType memWeight, const DeformationBase<ConfiguratorType> &Wa, RealType bendWeight,
                       const DeformationBase<ConfiguratorType> &Wb ) : m_Wa( Wa ), m_Wb( Wb ),
                                                                       m_bendWeight( bendWeight ),
                                                                       m_memWeight( memWeight ) {}

  void applyEnergy( const VectorType &UndeformedGeom, const VectorType &DeformedGeom, RealType &Dest ) const override {
    RealType memEnergy, bendEnergy;
    m_Wa.applyEnergy( UndeformedGeom, DeformedGeom, memEnergy );
    m_Wb.applyEnergy( UndeformedGeom, DeformedGeom, bendEnergy );
    Dest = m_memWeight * memEnergy + m_bendWeight * bendEnergy;
  }

  void applyUndefGradient( const VectorType &UndeformedGeom, const VectorType &DeformedGeom,
                           VectorType &Dest ) const override {
    VectorType memGrad, bendGrad;
    m_Wa.applyUndefGradient( UndeformedGeom, DeformedGeom, memGrad );
    m_Wb.applyUndefGradient( UndeformedGeom, DeformedGeom, bendGrad );
    Dest = m_memWeight * memGrad + m_bendWeight * bendGrad;
  }

  void applyDefGradient( const VectorType &UndeformedGeom, const VectorType &DeformedGeom,
                         VectorType &Dest ) const override {
    VectorType memGrad, bendGrad;
    m_Wa.applyDefGradient( UndeformedGeom, DeformedGeom, memGrad );
    m_Wb.applyDefGradient( UndeformedGeom, DeformedGeom, bendGrad );
    Dest = m_memWeight * memGrad + m_bendWeight * bendGrad;
  }

  void pushTripletsDefHessian( const VectorType &UndeformedGeom, const VectorType &DeformedGeom,
                               TripletListType &triplets, int rowOffset, int colOffset,
                               RealType factor = 1.0 ) const override {
    //TODO parallelize?!
    m_Wa.pushTripletsDefHessian( UndeformedGeom, DeformedGeom, triplets, rowOffset, colOffset, m_memWeight * factor );
    m_Wb.pushTripletsDefHessian( UndeformedGeom, DeformedGeom, triplets, rowOffset, colOffset, m_bendWeight * factor );
  }

  void pushTripletsUndefHessian( const VectorType &UndeformedGeom, const VectorType &DeformedGeom,
                                 TripletListType &triplets, int rowOffset, int colOffset,
                                 RealType factor = 1.0 ) const override {
    //TODO parallelize?!
    m_Wa.pushTripletsUndefHessian( UndeformedGeom, DeformedGeom, triplets, rowOffset, colOffset, m_memWeight * factor );
    m_Wb.pushTripletsUndefHessian( UndeformedGeom, DeformedGeom, triplets, rowOffset, colOffset,
                                   m_bendWeight * factor );
  }

  // mixed second derivative of deformation energy E[S_1, S_2], i.e. if "FirstDerivWRTDef" we have D_1 D_2 E[.,.], otherwise D_2 D_1 E[.,.]
  void pushTripletsMixedHessian( const VectorType &UndeformedGeom, const VectorType &DeformedGeom,
                                 TripletListType &triplets, int rowOffset, int colOffset, const bool FirstDerivWRTDef,
                                 RealType factor = 1.0 ) const override {
    //TODO parallelize?!
    m_Wa.pushTripletsMixedHessian( UndeformedGeom, DeformedGeom, triplets, rowOffset, colOffset, FirstDerivWRTDef,
                                   m_memWeight * factor );
    m_Wb.pushTripletsMixedHessian( UndeformedGeom, DeformedGeom, triplets, rowOffset, colOffset, FirstDerivWRTDef,
                                   m_bendWeight * factor );
  }

  int numOfNonZeroHessianEntries() const override {
    return m_Wa.numOfNonZeroHessianEntries() + m_Wb.numOfNonZeroHessianEntries();
  }

  RealType getBendingWeight() const override {
    return m_bendWeight;
  }

};