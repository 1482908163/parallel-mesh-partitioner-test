/* 
用于模拟分布式计算环境下的网格划分（MPI）
*/

#ifndef DISTRIBUTED_TEST_MESH_H
#define DISTRIBUTED_TEST_MESH_H

#include <tuple>
#include <cassert>

template<typename R, typename I> 
class distributed_test_mesh
{
public:
  using coordinate_type = R;
  using index_type = I;

public:
  explicit distributed_test_mesh(int rank) : NX(10), NY (10), NZ(10), RANK(rank) {}
  distributed_test_mesh(I nx, I ny, I nz, int rank) : NX(nx), NY(ny), NZ(nz), RANK(rank) {}

  I num_local_cells() const { return NX * NY * NZ; }

  std::tuple<R, R, R, R, R, R> local_bounding_box() const
  { return std::make_tuple(RANK * NX, 0, 0, (RANK + 1) * NX, NY, NZ); }

  std::tuple<R, R, R> cell_centroid(I c) const
  {
    assert(c >= 0 && c < num_local_cells());

    I l = c % (NX * NY);
    R half = static_cast<R>(1) / static_cast<R>(2);
    return std::make_tuple(RANK * NX + (l % NX) + half, (l / NX) + half, (c / (NX * NY)) + half);
  }

private:
  const I NX;
  const I NY;
  const I NZ;
  const int RANK;
};

#endif
