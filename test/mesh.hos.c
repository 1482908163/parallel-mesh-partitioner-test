/* 
该文件提供了如何使用 partition 函数的示例代码。它展示了如何在MPI环境中进行网格划分，包括：

序列化网格的划分：使用 serial_test_mesh 类创建一个简单的本地网格，并使用 partition 函数进行划分。
分布式网格的划分：使用 distributed_test_mesh 类创建一个分布式网格，并在MPI环境中进行划分。

该文件还展示了如何使用MPI进行初始化、划分和输出划分结果到文件。
*/

#include <vector>
#include <iterator>
#include <tuple>
#include <iostream>
#include <fstream>
//#include <cstdio> 
#include <chrono>

#include <mpi.h>

#include "test_hilbert_curve.h"
#include "serial_test_mesh.h"
#include "distributed_test_mesh.h"
#include "mesh_partitioner.h"

#include "hthread_host.h"

int main (int argc, char* argv[])
{
  int k = 64; // the default number of parts to partition into
  if (argc > 1) k = std::atoi(argv[1]);

  MPI_Init(NULL, NULL);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // //串行网格划分
  // serial_test_mesh<float, int> smesh(rank);
  // std::vector<float> sweights(smesh.num_local_cells(), 1.0);
  // std::vector<int> serial_output;
  // pmp::partition(smesh, k, sweights.begin(), std::back_inserter(serial_output), MPI_COMM_WORLD);
  // if (rank == 0)
  // {
  //   char serial_buff[100];
  //   std::snprintf(serial_buff, sizeof(serial_buff), "PartitionSerial_%d-%d.txt", size, k); //使用安全字符串格式化函数，把格式化的内容写入 serial_buff,防溢出
  //   std::ofstream serial_file(serial_buff);//用上面格式化出来的文件名创建一个输出文件流 serial_file
  //   serial_file << "x     y     z     p" << std::endl;
  //   for (int c = 0; c < smesh.num_local_cells(); ++c)
  //   {
  //     std::tuple<float, float, float> centroid = smesh.cell_centroid(c); 
  //     serial_file << std::get<0>(centroid) << " " << std::get<1>(centroid) << " ";
  //     serial_file << std::get<2>(centroid) << " " << serial_output[c] << std::endl;
  //   }
  // }
  
  //分布式网格划分
  distributed_test_mesh<double, int> dmesh(10, 10, 10, rank);
  std::vector<double> dweights(dmesh.num_local_cells(), 1.0);
  auto t0 = std::chrono::system_clock::now();
  std::vector<int> output;
  pmp::partition(dmesh, k, dweights.begin(), std::back_inserter(output), MPI_COMM_WORLD);
  auto t1 = std::chrono::system_clock::now();
  std::cout << "time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

  // output the partition info
  char buff[100];
  std::snprintf(buff, sizeof(buff), "PartitionDistributed_%d-%d_%d.txt", size, k, rank); 
  std::ofstream file(buff);
  file << "x     y     z     p" << std::endl;
  for (int c = 0; c < dmesh.num_local_cells(); ++c)
  {
    std::tuple<double, double, double> centroid = dmesh.cell_centroid(c); 
    file << std::get<0>(centroid) << " " << std::get<1>(centroid) << " " << std::get<2>(centroid) << " " << output[c] << std::endl;
  }

  MPI_Finalize();

  return 0;
}

