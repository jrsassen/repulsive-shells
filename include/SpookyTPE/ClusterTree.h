/**
 * \brief Header containing data structures and basic algorithms for cluster trees
 * \author Sassen
 */
#pragma once

#include <utility>
#include <vector>

#include <goast/Core.h>

namespace SpookyTPE {

/**
 * \brief Base class for all tree nodes
 * \author Sassen
 * \tparam Derived The type of the derived node
 *
 * This is the base from which all tree nodes inherit. It provides the API for the essential structure of a tree.
 */
template<typename Derived>
class TreeNodeBase {
protected:
  std::vector<Derived> m_Children;

public:
  TreeNodeBase() = default;

  virtual ~TreeNodeBase() = default;

  /**
   * \return Vector of all child nodes
   */
  std::vector<Derived> &Children() {
    return m_Children;
  }

  /**
   * \return Constant vector of all child nodes
   */
  const std::vector<Derived> &Children() const {
    return m_Children;
  }

  /**
   * Add a new child node
   * \param child New node to be added
   */
  void addChild( const Derived &child ) {
    //! \todo Check if child actually can be a child
    //! \todo Use efficient move semantics
    m_Children.push_back( child );
  }

  /**
   * Remove a child node
   * \param childId Position of the node to be removed among all children of this node
   */
  void removeChild( const int childId ) {
    m_Children.erase( m_Children.begin() + childId );
  }
};

/**
 * \brief Tree nodes for hierarchical decompositions into cluster trees
 * \author Sassen
 * \tparam ConfiguratorType Container with datatypes
 */
template<typename ConfiguratorType=DefaultConfigurator>
class ClusterTreeNode : virtual public TreeNodeBase<ClusterTreeNode<ConfiguratorType>> {
protected:
//  using ConfiguratorType = ConfiguratorType;
  using RealType = typename ConfiguratorType::RealType;
  using VectorType = typename ConfiguratorType::VectorType;
  using VecType = typename ConfiguratorType::VecType;
  using FullMatrixType = typename ConfiguratorType::FullMatrixType;

  std::vector<int> m_Vertices;
  VecType m_lowerBounds;
  VecType m_upperBounds;


public:
  int numGlobalVertices; // \todo Figure out how to do this properly!

  /**
   * \brief Construct node containing given vertices without computing its bounding box
   * \param Vertices A vector containing the indices of the vertices
   */
  explicit ClusterTreeNode( std::vector<int> Vertices ) : m_Vertices( std::move( Vertices )),
                                                          m_lowerBounds( std::numeric_limits<RealType>::infinity()),
                                                          m_upperBounds( -std::numeric_limits<RealType>::infinity()),
                                                          numGlobalVertices( -1 ) {}

  /**
   * \brief Construct node containing given vertices with computing its bounding box from the given vertex positions
   * \param Vertices A vector containing the indices of the vertices
   * \param Geometry The vertex positions
   */
  ClusterTreeNode( std::vector<int> Vertices, const VectorType &Geometry ) :
      m_Vertices( std::move( Vertices )),
      m_lowerBounds( std::numeric_limits<RealType>::infinity()),
      m_upperBounds( -std::numeric_limits<RealType>::infinity()),
      numGlobalVertices( Geometry.size() / 3 ) {
    setBoundsFromGeometry( Geometry );
  }

  /**
   * \return Constant reference to vector with indices of all vertices in the cluster
   */
  const std::vector<int> &Vertices() const {
    return m_Vertices;
  }

  /**
   * \return Modifiable reference to vector with indices of all vertices in the cluster
   */
  std::vector<int> &Vertices() {
    return m_Vertices;
  }

  /**
   * \return Number of vertices in this cluster
   */
  int numVertices() const {
    return m_Vertices.size();
  }

  /**
   * \return Lower bounds of the box associated with this node
   */
  const VecType &lowerBounds() const {
    return m_lowerBounds;
  }

  /**
   * \return Upper bounds of the box associated with this node
   */
  const VecType &upperBounds() const {
    return m_upperBounds;
  }

  /**
   * Construct bounding box of cluster from nodal positions
   * \param Geometry nodal positions to use
   */
  void setBoundsFromGeometry( const VectorType &Geometry ) {
    for ( const int &vertexIdx: m_Vertices ) {
      VecType coord;
      getXYZCoord( Geometry, coord, vertexIdx );
      for ( int i: { 0, 1, 2 } ) {
        m_lowerBounds[i] = std::min( m_lowerBounds[i], coord[i] );
        m_upperBounds[i] = std::max( m_upperBounds[i], coord[i] );
      }
    }
  }

  /**
   * \brief Check validity of cluster tree
   * \return Validity of cluster tree
   *
   * This method checks, recursively for all clusters, that the sets of vertices of the child clusters are disjoint
   * and that their union is exactly the set of vertices of that cluster.
   *
   * \note If compiled in debugging mode this will print the occurring errors to stderr.
   */
  bool valid() const {
    // Nothing to do if this is a leaf
    if ( this->Children().empty())
      return true;

    // Check if child clusters are disjoint
    std::set<int> childrenVertices;
    for ( const auto &child: this->Children()) {
      std::set<int> childVertices, intersection;
      childVertices.insert( child.Vertices().begin(), child.Vertices().end());
      std::set_intersection( childVertices.begin(), childVertices.end(), childrenVertices.begin(),
                             childrenVertices.end(), std::inserter( intersection, intersection.begin()));
      if ( !intersection.empty()) {
#ifndef NDEBUG
        std::cerr << "ClustertreeNode::valid: Child clusters are not disjoint" << std::endl;
#endif
        return false;
      }

      childrenVertices.insert( child.Vertices().begin(), child.Vertices().end());
    }

    // Check if vertices in this cluster is identical to union of child clusters
    std::set<int> thisVertices;
    thisVertices.insert( Vertices().begin(), Vertices().end());

    if ( childrenVertices != thisVertices ) {
#ifndef NDEBUG
      std::cerr << "ClustertreeNode::valid: Vertices in cluster are not identical to union of child clusters ( "
                << thisVertices.size() << " vs. " << childrenVertices.size() << ")" << std::endl;

#endif

      return false;
    }

    // Recursively check for all children
    for ( const auto &child: this->Children())
      if ( !child.valid())
        return false;

    return true;
  }

protected:
  /**
   * Protected constructor for derived root nodes
   */
  ClusterTreeNode() : m_lowerBounds( std::numeric_limits<RealType>::infinity()),
                      m_upperBounds( -std::numeric_limits<RealType>::infinity()),
                      numGlobalVertices( -1 ) {};
};

/**
 * \brief Construct cluster tree by regular splitting of the bounding box
 * \author Sassen
 * \tparam NodeType The type of nodes which the cluster tree should be made of, has to support storing vertices and bounding boxes
 *
 * This class constructs a cluster tree by splitting the bounding box of each cluster into eight equally sized subboxes
 * until the leaves of the tree have reached a certain number of vertices.
 *
 * \sa ClustertreeNode
 */
template<typename NodeType>
class RegularSplittingTree : public NodeType {
private:
  using RealType = typename NodeType::RealType;
  using VectorType = typename NodeType::VectorType;
  using VecType = typename NodeType::VecType;

  using NodeType::m_Vertices;

  const int m_maxLeafSize = 8;

public:
  /**
   * \brief Construct cluster tree from given vertex positions
   * \param Geometry The vertex positions
   * \param maxLeafSize Maximal number of vertices in a leaf
   */
  explicit RegularSplittingTree( const VectorType &Geometry, const int maxLeafSize = 8 ) : m_maxLeafSize(
      maxLeafSize ) {
    if ( Geometry.size() % 3 != 0 )
      throw std::length_error( "RegularSplittingTree: Provided nodal positions do not have a "
                               "multiple of three entries" );

    // Construct root nodes
    m_Vertices.resize( Geometry.size() / 3 );
    std::iota( m_Vertices.begin(), m_Vertices.end(), 0 );
    this->numGlobalVertices = Geometry.size() / 3;
    this->setBoundsFromGeometry( Geometry );

    std::queue<NodeType *> nodeQueue;
    nodeQueue.push( this );

    while ( !nodeQueue.empty()) {
      // Get the next node
      NodeType *node = nodeQueue.front();
      // Remove the currently visited node from the queue
      nodeQueue.pop();

      // Ignore nodes already small enough
      if ( node->numVertices() <= m_maxLeafSize )
        continue;

      // Determine sub-boxes = 8 equal-size boxes
      std::vector<std::pair<VecType, VecType>> childBoxes;

      // More intuitive names
      const RealType &x_min = node->lowerBounds()[0];
      const RealType &y_min = node->lowerBounds()[1];
      const RealType &z_min = node->lowerBounds()[2];

      const RealType &x_max = node->upperBounds()[0];
      const RealType &y_max = node->upperBounds()[1];
      const RealType &z_max = node->upperBounds()[2];

      const RealType x_mid = ( node->lowerBounds()[0] + node->upperBounds()[0] ) / 2.;
      const RealType y_mid = ( node->lowerBounds()[1] + node->upperBounds()[1] ) / 2.;
      const RealType z_mid = ( node->lowerBounds()[2] + node->upperBounds()[2] ) / 2.;

      childBoxes.emplace_back( VecType( x_min, y_min, z_min ), VecType( x_mid, y_mid, z_mid ));
      childBoxes.emplace_back( VecType( x_mid, y_min, z_min ), VecType( x_max, y_mid, z_mid ));
      childBoxes.emplace_back( VecType( x_min, y_mid, z_min ), VecType( x_mid, y_max, z_mid ));
      childBoxes.emplace_back( VecType( x_mid, y_mid, z_min ), VecType( x_max, y_max, z_mid ));
      childBoxes.emplace_back( VecType( x_min, y_min, z_mid ), VecType( x_mid, y_mid, z_max ));
      childBoxes.emplace_back( VecType( x_mid, y_min, z_mid ), VecType( x_max, y_mid, z_max ));
      childBoxes.emplace_back( VecType( x_min, y_mid, z_mid ), VecType( x_mid, y_max, z_max ));
      childBoxes.emplace_back( VecType( x_mid, y_mid, z_mid ), VecType( x_max, y_max, z_max ));

      // For each vertex determine in which sub-box it lies.
      std::vector<std::vector<int>> vertexAssignments( childBoxes.size());
      for ( const int &vertexIdx: node->Vertices()) {
        VecType Coord;
        getXYZCoord( Geometry, Coord, vertexIdx );
        for ( int i = 0; i < childBoxes.size(); i++ ) {
          // Check if vertex is contained in this new box
          if ( Coord[0] < childBoxes[i].first[0] || Coord[0] > childBoxes[i].second[0] ) continue;
          if ( Coord[1] < childBoxes[i].first[1] || Coord[1] > childBoxes[i].second[1] ) continue;
          if ( Coord[2] < childBoxes[i].first[2] || Coord[2] > childBoxes[i].second[2] ) continue;

          // If we get here, the point was not rejected for this box, so we assign it
          vertexAssignments[i].push_back( vertexIdx );
          break;
        }
      }

      // Construct and add childclusters to node
      for ( const auto &newVertices: vertexAssignments ) {
        if ( !newVertices.empty()) {
          node->addChild( NodeType( newVertices, Geometry ));
        }
      }

      // Add new nodes to queue
      for ( auto &newNode: node->Children())
        nodeQueue.push( &newNode );
    }
  }
};

/**
 * \brief Construct cluster tree by balanced splitting of the bounding box
 * \author Sassen
 * \tparam NodeType The type of nodes which the cluster tree should be made of, has to support storing vertices and bounding boxes
 *
 * This class constructs a cluster tree by splitting the bounding box of each cluster into four subboxes each containing
 * approximately the same number of vertices. This is achieved by first splitting the bounding box along the longest
 * edge such that in each resulting subbox the number of vertices is (approximately) the same and then repeats this on
 * the two subboxes to obtain four subboxes in total. This is done until the leaves of the tree have reached a
 * certain number of vertices.
 *
 * \sa ClustertreeNode
 */
template<typename NodeType>
class Balanced4Tree : public NodeType {
private:
  using RealType = typename NodeType::RealType;
  using VectorType = typename NodeType::VectorType;
  using VecType = typename NodeType::VecType;

  using NodeType::m_Vertices;

  const int m_maxLeafSize;

public:
  /**
   * \brief Construct cluster tree from given vertex positions
   * \param Geometry The vertex positions
   * \param maxLeafSize Maximal number of vertices in a leaf
   */
  explicit Balanced4Tree( const VectorType &Geometry, const int maxLeafSize = 8 ) : m_maxLeafSize( maxLeafSize ) {
    if ( Geometry.size() % 3 != 0 )
      throw std::length_error( "RegularSplittingTree: Provided nodal positions do not have a "
                               "multiple of three entries" );

    // Construct root nodes
    m_Vertices.resize( Geometry.size() / 3 );
    std::iota( m_Vertices.begin(), m_Vertices.end(), 0 );
    this->numGlobalVertices = Geometry.size() / 3;
    this->setBoundsFromGeometry( Geometry );

    std::queue<NodeType *> nodeQueue;
    nodeQueue.push( this );

    while ( !nodeQueue.empty()) {
      // Get the next node
      NodeType *node = nodeQueue.front();
      // Remove the currently visited node from the queue
      nodeQueue.pop();

      // Ignore nodes already small enough
      if ( node->numVertices() <= m_maxLeafSize )
        continue;

      // More intuitive names
      const RealType &x_min = node->lowerBounds()[0];
      const RealType &y_min = node->lowerBounds()[1];
      const RealType &z_min = node->lowerBounds()[2];

      const RealType &x_max = node->upperBounds()[0];
      const RealType &y_max = node->upperBounds()[1];
      const RealType &z_max = node->upperBounds()[2];

      const RealType x_mid = ( node->lowerBounds()[0] + node->upperBounds()[0] ) / 2.;
      const RealType y_mid = ( node->lowerBounds()[1] + node->upperBounds()[1] ) / 2.;
      const RealType z_mid = ( node->lowerBounds()[2] + node->upperBounds()[2] ) / 2.;

      // A. First split along axis along longest edge.
      // Determine longest edge
      int d_max = 0;
      RealType max_length = -std::numeric_limits<RealType>::infinity();

      for ( const int d: { 0, 1, 2 } ) {
        if ( node->upperBounds()[d] - node->lowerBounds()[d] > max_length ) {
          d_max = d;
          max_length = node->upperBounds()[d] - node->lowerBounds()[d];
        }
      }

      // Assemble values along dimension d to compute split
      std::vector<std::pair<int, RealType>> dValues;
      for ( const int &vertexIdx: node->Vertices()) {
        VecType Coord;
        getXYZCoord( Geometry, Coord, vertexIdx );
        dValues.emplace_back( vertexIdx, Coord[d_max] );
      }

      // Split values in lower and upper half
      std::nth_element( dValues.begin(), dValues.begin() + dValues.size() / 2, dValues.end(),
                        []( std::pair<int, RealType> a, std::pair<int, RealType> b ) {
                          return a.second < b.second;
                        } );

      // B. Second splits
      // B.1. Lower Half

      // Determine new bounds
      VecType lowerBounds( std::numeric_limits<RealType>::infinity()),
          upperBounds( -std::numeric_limits<RealType>::infinity());

      for ( int i = 0; i < dValues.size() / 2; i++ ) {
        VecType Coord;
        getXYZCoord( Geometry, Coord, dValues[i].first );
        for ( int d: { 0, 1, 2 } ) {
          lowerBounds[d] = std::min( lowerBounds[d], Coord[d] );
          upperBounds[d] = std::max( upperBounds[d], Coord[d] );
        }
      }

      // Determine new longest edge
      max_length = -std::numeric_limits<RealType>::infinity();

      for ( const int d: { 0, 1, 2 } ) {
        if ( upperBounds[d] - lowerBounds[d] > max_length ) {
          d_max = d;
          max_length = upperBounds[d] - lowerBounds[d];
        }
      }

      // Adopt new coordinates
      for ( int i = 0; i < dValues.size() / 2; i++ ) {
        VecType Coord;
        getXYZCoord( Geometry, Coord, dValues[i].first );
        dValues[i].second = Coord[d_max];
      }

      // Split new values in lower and upper half
      std::nth_element( dValues.begin(), dValues.begin() + dValues.size() / 4, dValues.begin() + dValues.size() / 2,
                        []( std::pair<int, RealType> a, std::pair<int, RealType> b ) {
                          return a.second < b.second;
                        } );

      // B.2. Upper Half

      // Determine new bounds
      lowerBounds = VecType( std::numeric_limits<RealType>::infinity());
      upperBounds = VecType( -std::numeric_limits<RealType>::infinity());

      for ( int i = dValues.size() / 2; i < dValues.size(); i++ ) {
        VecType Coord;
        getXYZCoord( Geometry, Coord, dValues[i].first );
        for ( int d: { 0, 1, 2 } ) {
          lowerBounds[d] = std::min( lowerBounds[d], Coord[d] );
          upperBounds[d] = std::max( upperBounds[d], Coord[d] );
        }
      }

      // Determine new longest edge
      max_length = -std::numeric_limits<RealType>::infinity();

      for ( const int d: { 0, 1, 2 } ) {
        if ( upperBounds[d] - lowerBounds[d] > max_length ) {
          d_max = d;
          max_length = upperBounds[d] - lowerBounds[d];
        }
      }

      // Adopt new coordinates
      for ( int i = dValues.size() / 2; i < dValues.size(); i++ ) {
        VecType Coord;
        getXYZCoord( Geometry, Coord, dValues[i].first );
        dValues[i].second = Coord[d_max];
      }

      // Split new values in lower and upper half
      std::nth_element( dValues.begin() + dValues.size() / 2,
                        dValues.begin() + dValues.size() / 2 + dValues.size() / 4,
                        dValues.end(),
                        []( std::pair<int, RealType> a, std::pair<int, RealType> b ) {
                          return a.second < b.second;
                        } );

      // C. Build new clusters
      std::vector<std::vector<int>> vertexAssignments( 4 );
      for ( int i = 0; i < dValues.size(); i++ ) {
        if ( i < dValues.size() / 4 ) {
          vertexAssignments[0].emplace_back( dValues[i].first );
        } else if ( i < dValues.size() / 2 ) {
          vertexAssignments[1].emplace_back( dValues[i].first );
        } else if ( i < 3 * dValues.size() / 4 ) {
          vertexAssignments[2].emplace_back( dValues[i].first );
        } else {
          vertexAssignments[3].emplace_back( dValues[i].first );
        }
      }

      // Construct and add child clusters to node
      for ( const auto &newVertices: vertexAssignments ) {
        if ( !newVertices.empty()) {
          node->addChild( NodeType( newVertices, Geometry ));
        }
      }

      // Add new nodes to queue
      for ( auto &newNode: node->Children())
        nodeQueue.push( &newNode );

    }
  }
};

template<typename NodeType>
class LongestAxisTree : public NodeType {
private:
  using RealType = typename NodeType::RealType;
  using VectorType = typename NodeType::VectorType;
  using VecType = typename NodeType::VecType;

  using NodeType::m_Vertices;

  const int m_maxLeafSize;

  const bool m_medianSplit = false;

public:
  /**
   * \brief Construct cluster tree from given vertex positions
   * \param Geometry The vertex positions
   * \param maxLeafSize Maximal number of vertices in a leaf
   */
  explicit LongestAxisTree( const VectorType &Geometry, const int maxLeafSize = 8 ) : m_maxLeafSize( maxLeafSize ) {
    if ( Geometry.size() % 3 != 0 )
      throw std::length_error( "RegularSplittingTree: Provided nodal positions do not have a "
                               "multiple of three entries" );

    // Construct root nodes
    m_Vertices.resize( Geometry.size() / 3 );
    std::iota( m_Vertices.begin(), m_Vertices.end(), 0 );
    this->numGlobalVertices = Geometry.size() / 3;
    this->setBoundsFromGeometry( Geometry );

    std::queue<NodeType *> nodeQueue;
    nodeQueue.push( this );

    while ( !nodeQueue.empty()) {
      // Get the next node
      NodeType *node = nodeQueue.front();
      // Remove the currently visited node from the queue
      nodeQueue.pop();

      // Ignore nodes already small enough
      if ( node->numVertices() <= m_maxLeafSize )
        continue;

      // A. First split along axis along longest edge.
      // Determine the longest edge
      int d_max = 0;
      RealType max_length = -std::numeric_limits<RealType>::infinity();

      for ( const int d: { 0, 1, 2 } ) {
        if ( node->upperBounds()[d] - node->lowerBounds()[d] > max_length ) {
          d_max = d;
          max_length = node->upperBounds()[d] - node->lowerBounds()[d];
        }
      }

      std::array<std::vector<int>, 2> vertexAssignments;
      if (m_medianSplit) {
        // Assemble values along dimension d to compute split
        std::vector<std::pair<int, RealType>> dValues;
        for ( const int &vertexIdx: node->Vertices()) {
          VecType Coord;
          getXYZCoord( Geometry, Coord, vertexIdx );
          dValues.emplace_back( vertexIdx, Coord[d_max] );
        }

        // Split values in lower and upper half
        std::nth_element( dValues.begin(), dValues.begin() + dValues.size() / 2, dValues.end(),
                          []( std::pair<int, RealType> a, std::pair<int, RealType> b ) {
                            return a.second < b.second;
                          } );

        // Build new clusters
        for ( int i = 0; i < dValues.size(); i++ ) {
          if ( i < dValues.size() / 2 ) {
            vertexAssignments[0].emplace_back( dValues[i].first );
          }
          else {
            vertexAssignments[1].emplace_back( dValues[i].first );
          }
        }
      }
      else {
        RealType mid = ( node->upperBounds()[d_max] + node->lowerBounds()[d_max] ) / 2.;
        for ( const int &vertexIdx: node->Vertices()) {
          VecType Coord;
          getXYZCoord( Geometry, Coord, vertexIdx );
          if ( Coord[d_max] < mid )
            vertexAssignments[0].emplace_back( vertexIdx );
          else
            vertexAssignments[1].emplace_back( vertexIdx );
        }
      }

      // Construct and add child clusters to node
      for ( const auto &newVertices: vertexAssignments ) {
        if ( !newVertices.empty()) {
          node->addChild( NodeType( newVertices, Geometry ));
        }
      }

      // Add new nodes to queue
      for ( auto &newNode: node->Children())
        nodeQueue.push( &newNode );

    }
  }
};

}