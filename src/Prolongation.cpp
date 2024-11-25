#include <goast/Core.h>

#include "ProjectionProlongation.h"

using VectorType = typename DefaultConfigurator::VectorType;

int main( int argc, char *argv[] ) {
  if ( argc < 4 ) {
    throw std::runtime_error( "Wrong number of arguments" );
  }

  std::string coarseRefPath = argv[1];
  std::string fineRefPath = argv[2];
  std::string inputPath = argv[3];
  std::string outputPath = argv[4];


  TriMesh coarseReferenceMesh, fineReferenceMesh, inputMesh;

  OpenMesh::IO::Options opt = OpenMesh::IO::Options::Default;
  opt += OpenMesh::IO::Options::VertexTexCoord;

  fineReferenceMesh.request_vertex_texcoords2D();

  if ( !OpenMesh::IO::read_mesh( coarseReferenceMesh, coarseRefPath ))
    throw std::runtime_error( "Failed to read file: " + coarseRefPath );
  if ( !OpenMesh::IO::read_mesh( fineReferenceMesh, fineRefPath, opt ))
    throw std::runtime_error( "Failed to read file: " + fineRefPath  );

  if ( !OpenMesh::IO::read_mesh( inputMesh, inputPath ))
    throw std::runtime_error( "Failed to read file: " + coarseRefPath );

  MeshTopologySaver coarseTopology( coarseReferenceMesh );
  MeshTopologySaver fineTopology( fineReferenceMesh );

  VectorType coarseGeometry, fineGeometry, inputGeometry;
  getGeometry( coarseReferenceMesh, coarseGeometry );
  getGeometry( fineReferenceMesh, fineGeometry );

  getGeometry( inputMesh, inputGeometry );

  ProjectionProlongationOperator<DefaultConfigurator> P( coarseTopology, coarseGeometry, fineTopology, fineGeometry );

  VectorType prolongatedGeometry = P( inputGeometry );

  if ( !outputPath.empty()) {
    setGeometry( fineReferenceMesh, prolongatedGeometry );
    if ( !OpenMesh::IO::write_mesh( fineReferenceMesh, outputPath, opt ))
      throw std::runtime_error( "Failed to read file: " + outputPath );
  }
}