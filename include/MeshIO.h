#pragma once

#include <goast/Core.h>

template<typename VectorType>
void saveAsPLY( const MeshTopologySaver &Topology, const VectorType &Geometry, const std::string &path ) {
  assert( Geometry.size() == 3 * Topology.getNumVertices() && "saveAsPLY/OBJ: Wrong size of Geometry!" );

  TriMesh outMesh( Topology.getGrid());
  setGeometry( outMesh, Geometry );

  OpenMesh::IO::Options wopt = OpenMesh::IO::Options::Default;
  // wopt += OpenMesh::IO::Options::VertexTexCoord;

  if ( !OpenMesh::IO::write_mesh( outMesh, path, wopt, 20 ))
    throw std::runtime_error( "Failed to write file: " + path );
}

template<typename T>
const auto saveAsOBJ = saveAsPLY<T>;