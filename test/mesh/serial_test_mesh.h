/* 
用于模拟单一进程环境下的网格划分
*/

#ifndef SERIAL_TEST_MESH_H
#define SERIAL_TEST_MESH_H

#include <tuple>
#include <limits>
#include <cassert>

// A serial mesh used for testing, though the mesh can be instantiated on multiple ranks. Rank 0 holds
// NX x NY x NZ cells and all other ranks hold no cells. This is used to test the special case where
// rank 0 reads in a serial mesh and p (p >= 1) processes are used to partition the mesh into k parts.
//
// This mesh implements the minimum API required by the parallel mesh partitioner.
template<typename R, typename I> // R - vertex coordinate type, I - index type for vertices and cells
class serial_test_mesh
{
public:
  using coordinate_type = R;
  using index_type = I;

public:
  explicit serial_test_mesh(int rank) : NX(10), NY (10), NZ(10), RANK(rank) {}
  serial_test_mesh(I nx, I ny, I nz, int rank) : NX(nx), NY(ny), NZ(nz), RANK(rank) {}

  I num_local_cells() const { return RANK == 0 ? NX * NY * NZ : 0; }//返回本地（当前进程）拥有的网格单元数

  std::tuple<R, R, R, R, R, R> local_bounding_box() const
  {
    if (RANK == 0)
      return std::make_tuple(0, 0, 0, NX, NY, NZ); 
    else
    {
      R min = std::numeric_limits<R>::min();
      R max = std::numeric_limits<R>::max();
      // "empty" bounding box that will never affect the global
      // bounding box obtained via MPI_Allreduce
      return std::make_tuple(max, max, max, min, min, min);
    }
  }

  std::tuple<R, R, R> cell_centroid(I c) const//返回编号为c的单元格的质心坐标
  {
    assert(c >= 0 && c < num_local_cells()); // this will fail on RANK != 0

    I l = c % (NX * NY);
    R half = static_cast<R>(1) / static_cast<R>(2);
    return std::make_tuple((l % NX) + half, (l / NX) + half, (c / (NX * NY)) + half);
  }

private:
  const I NX;
  const I NY;
  const I NZ;
  const int RANK;
};

#endif
