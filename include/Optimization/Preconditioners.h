#pragma once

#include <goast/Optimization/LinearOperator.h>

//template<typename ConfiguratorType=DefaultConfigurator>
//class DiagonalPreconditioner : public LinearOperator<ConfiguratorType> {
//  using VectorType = typename ConfiguratorType::VectorType;
//  using RealType = typename ConfiguratorType::RealType;
//  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
//  using FullMatrixType = typename ConfiguratorType::FullMatrixType;
//
//  const RealType m_diagonalCutoff = 1.e-12;
//
//  VectorType m_Dinv;
//
//public:
//  explicit DiagonalPreconditioner( const LinearOperator<ConfiguratorType> &A ) {
//    assert( A.rows() == A.cols() && "DiagonalPreconditioner: Operator has to be quadratic" );
//
//    m_Dinv.resize(A.rows());
//    m_Dinv.setConstant(1.);
//  }
//
//  void apply( const VectorType &Arg, VectorType &Dest ) const override {
//    assert( Arg.size() == m_Dinv.size() &&
//            "DiagonalPreconditioner::apply: Wrong size of Arg." );
//
//    Dest.resize( m_Dinv.size());
//
//    for ( int i = 0; i < m_Dinv.size(); i++ )
//      Dest[i] = Arg[i] * m_Dinv[i];
//  }
//
//  int rows() const override {
//    return m_Dinv.size();
//  }
//
//  int cols() const override {
//    return m_Dinv.size();
//  }
//};

template<typename ConfiguratorType=DefaultConfigurator>
class DiagonalPreconditioner : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  VectorType m_Dinv;

public:
  explicit DiagonalPreconditioner( const LinearOperator<ConfiguratorType> &A ) {
    assert( A.rows() == A.cols() && "DiagonalPreconditioner: Operator has to be quadratic" );
    m_Dinv.resize( A.rows());

    if ( typeid( A ) == typeid( SymmetricTriadiagonalBlockOperator<ConfiguratorType> )) {
      const std::vector<std::unique_ptr<LinearOperator<ConfiguratorType>>> &blocks = dynamic_cast<const SymmetricTriadiagonalBlockOperator<ConfiguratorType> &>(A).getMainDiagonalBlocks();

      const int blockSize = blocks[0]->rows();

      for ( int i = 0; i < blocks.size(); i++ ) {
        SparseMatrixType localMat;
        blocks[i]->assembleTransformationMatrix( localMat );
        m_Dinv.segment( i * blockSize, blockSize ) = localMat.diagonal();
      }
    }
    else {
      SparseMatrixType M;
      A.assembleTransformationMatrix( M );

      m_Dinv = M.diagonal();
    }

    for ( int i = 0; i < m_Dinv.size(); i++ ) {
      if ( m_Dinv[i] > 100 * std::numeric_limits<RealType>::epsilon())
        m_Dinv[i] = 1. / m_Dinv[i];
      else
        m_Dinv[i] = 1.;
    }

  }

  void apply( const VectorType &Arg, VectorType &Dest ) const override {
//    Dest = Arg;

    Dest.resize( m_Dinv.size());
    for ( int i = 0; i < m_Dinv.size(); i++ )
      Dest[i] = Arg[i] * m_Dinv[i];

//    std::cout << " .. Dest - Arg = " << std::scientific << (Dest - Arg).norm() << std::endl;
  }

  int rows() const override {
    return m_Dinv.size();
  }

  int cols() const override {
    return m_Dinv.size();
  }
};


template<typename ConfiguratorType=DefaultConfigurator>
class IncompleteCholeskyPreconditioner : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  Eigen::IncompleteCholesky<RealType, Eigen::Lower, Eigen::AMDOrdering<typename SparseMatrixType::StorageIndex>> m_Solver;
  const int m_dim;

public:
  explicit IncompleteCholeskyPreconditioner( const LinearOperator<ConfiguratorType> &A,
                                             const std::vector<int> &fixedVariables ) : m_dim( A.rows()) {
    assert( A.rows() == A.cols() && "IncompleteCholeskyPreconditioner: Operator has to be quadratic" );

    // Assemble system matrix
    SparseMatrixType M;
    A.assembleTransformationMatrix( M );

    applyMaskToSymmetricMatrix( fixedVariables, M );

    // Prepare solver
    m_Solver.compute( M );
  }

  explicit IncompleteCholeskyPreconditioner( const SparseMatrixType &M )
          : m_dim( M.rows()) {
    assert( M.rows() == M.cols() && "IncompleteCholeskyPreconditioner: Matrix has to be quadratic" );

    // Prepare solver
    m_Solver.compute( M );
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
template<typename ConfiguratorType=DefaultConfigurator>
class IncompleteLUPreconditioner : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  Eigen::IncompleteLUT<RealType> m_Solver;
  const int m_dim;

public:
  explicit IncompleteLUPreconditioner( const LinearOperator<ConfiguratorType> &A,
                                             const std::vector<int> &fixedVariables ) : m_dim( A.rows()) {
    assert( A.rows() == A.cols() && "IncompleteCholeskyPreconditioner: Operator has to be quadratic" );

    // Assemble system matrix
    SparseMatrixType M;
    A.assembleTransformationMatrix( M );

    applyMaskToSymmetricMatrix( fixedVariables, M );

    // Prepare solver
    m_Solver.compute( M );
  }

  explicit IncompleteLUPreconditioner( const SparseMatrixType &M, const std::vector<int> &fixedVariables )
          : m_dim( M.rows()) {
    assert( M.rows() == M.cols() && "IncompleteCholeskyPreconditioner: Operator has to be quadratic" );

    applyMaskToSymmetricMatrix( fixedVariables, M );

    // Prepare solver
    m_Solver.compute( M );
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

template<typename ConfiguratorType=DefaultConfigurator>
class CholeskyPreconditioner : public LinearOperator<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;


  RealType m_dir_beta = 1e-6;
  RealType m_tau_factor = 5.;

  Eigen::CholmodSupernodalLLT<SparseMatrixType> m_Solver;
  const int m_dim;

public:
  explicit CholeskyPreconditioner( const LinearOperator<ConfiguratorType> &A, const std::vector<int> &fixedVariables )
          : m_dim( A.rows()) {
    assert( A.rows() == A.cols() && "IncompleteCholeskyPreconditioner: Operator has to be quadratic" );

    // Assemble system matrix
    SparseMatrixType M;
    A.assembleTransformationMatrix( M );

    applyMaskToSymmetricMatrix( fixedVariables, M );

    // Prepare solver
    m_Solver.analyzePattern( M );
    m_Solver.cholmod().print = 0;

    VectorType diagonal = M.diagonal();
    RealType min_diag = diagonal.minCoeff();
    RealType tau_k;

    if ( min_diag > -100. * std::numeric_limits<RealType>::epsilon())
      tau_k = 0;
    else
      tau_k = -min_diag + m_dir_beta; // std::max( -min_diag + _dir_beta, tau_k / _tau_factor ); //

    while ( true ) {
      m_Solver.setShift( tau_k );
      m_Solver.factorize( M );

      if ( m_Solver.info() == Eigen::Success )
        break;

      tau_k = std::max( m_tau_factor * tau_k, m_dir_beta );
    }

//    std::cerr << " .... tau = " << std::scientific << tau_k << std::endl;
  }

  explicit CholeskyPreconditioner( const SparseMatrixType &M ) : m_dim(
          M.rows()) {
    assert( M.rows() == M.cols() && "IncompleteCholeskyPreconditioner: Operator has to be quadratic" );

    // Prepare solver
    m_Solver.analyzePattern( M );
    m_Solver.cholmod().print = 0;

    VectorType diagonal = M.diagonal();
    RealType min_diag = diagonal.minCoeff();
    RealType tau_k;

    if ( min_diag > -100. * std::numeric_limits<RealType>::epsilon())
      tau_k = 0;
    else
      tau_k = -min_diag + m_dir_beta; // std::max( -min_diag + _dir_beta, tau_k / _tau_factor ); //

    while ( true ) {
      m_Solver.setShift( tau_k );
      m_Solver.factorize( M );

      if ( m_Solver.info() == Eigen::Success )
        break;

      tau_k = std::max( m_tau_factor * tau_k, m_dir_beta );
    }
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


template<typename ConfiguratorType, template<typename C> class PreconditionerType>
class LocalizedPreconditioner : public MapToLinOp<ConfiguratorType> {
  using VectorType = typename ConfiguratorType::VectorType;
  using RealType = typename ConfiguratorType::RealType;
  using SparseMatrixType = typename ConfiguratorType::SparseMatrixType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  const MapToLinOp<ConfiguratorType>  &m_F;
  const std::vector<int> &m_fixedVariables;
  RealType m_W = 1.;

public:
  LocalizedPreconditioner( const MapToLinOp<ConfiguratorType> &F, const std::vector<int> &fixedVariables ) : m_F( F ),
                                                                                                             m_fixedVariables(
                                                                                                                     fixedVariables ) {}

  LocalizedPreconditioner( const MapToLinOp<ConfiguratorType> &F, const std::vector<int> &fixedVariables, RealType W )
          : m_F( F ), m_fixedVariables( fixedVariables ), m_W( W ) {}


  std::unique_ptr<LinearOperator<ConfiguratorType>> operator()( const VectorType &Point ) const override {
    auto L = m_F( Point );

    auto unscaled = std::make_unique<PreconditionerType<ConfiguratorType>>( *L, m_fixedVariables );

    return std::make_unique<ScaledOperator<ConfiguratorType>>( m_W, std::move( unscaled ));
  }

};