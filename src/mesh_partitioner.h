/* 
partition 函数模板，该函数使用MPI将网格划分为 k 部分，初始网格分布在 p 个进程中。
partition 函数接受一个网格类，该类需要具有特定方法：num_local_cells，local_bounding_box 和 cell_centroid
*/

#ifndef MESH_PARTITIONER_H
#define MESH_PARTITIONER_H 

#include <mpi.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <iterator>
#include <numeric>
#include <cassert>

#include "hilbert_curve.h"
#include "mpi_datatype_traits.h"
#include "mpi_utils.h"

#include "hthread_host.h"

namespace pmp {

// policy classes for handling constant vs custom cell weights
template<typename WT> // weight_type
struct constant_weights //常量权重的策略类
{
  using weight_type = WT; //别名，等同于typedef WT weight_type;

  static constexpr bool is_constant_weight = true;//constexpr 表示该值在编译时已知，并且该常量的值不能被修改
  //静态变量是类级别的变量，而非实例级别的变量。无论你创建多少个类的实例，这个静态变量只有一个副本，所有实例都会共享这个副本。

  explicit/* 显示的 */ constant_weights(WT local_weight) : m_local_weight(local_weight) {}//构造函数

  weight_type local_weight() const { return m_local_weight; }

private:
  const WT m_local_weight;
  //常量（const），意味着它在对象构造时初始化后不能更改。这个成员保存了传入的局部权重值。
};

template<typename ConstRandAccItr>//常量随机访问迭代器类型
struct custom_weights//自定义权重的策略类
{
  using weight_type = typename std::iterator_traits<ConstRandAccItr>::value_type;//代表了迭代器类型 ConstRandAccItr 所指向的元素类型
  //typename 只有在“嵌套依赖类型”里才需要，直接写类型名不用。
  static constexpr bool is_constant_weight = false;

  custom_weights(ConstRandAccItr wit, std::size_t size) : m_wit(wit), m_size(size) {}

  //计算并返回所有权重的和
  weight_type local_weight() const { return std::accumulate(m_wit, m_wit + m_size, 0); }

  ConstRandAccItr weights_begin() const { return m_wit; }

  weight_type weight(std::size_t i) const { return *(m_wit + i); }

private:
  // note that these weights could belong to either the local cells or
  // those cells associated a particular bin (after MPI communications) 
  const ConstRandAccItr m_wit;
  const std::size_t m_size;
};

// function objects used to populate data sending to the specified rank
/* 在网格数据和空间填充曲线（SFC）之间建立联系。它通过空间填充曲线将每个网格单元的坐标映射为索引，
并按进程（rank）进行数据的处理和分配。最终，它将这些计算出的索引填充到目标迭代器中， */
template<typename SFC, typename MSH>
struct sfc_index_generator
{
  sfc_index_generator(const SFC& sfc, const MSH& mesh, const int* cell_bins, int rank_to_bin_offset)
    : m_sfc(&sfc), m_mesh(&mesh), m_cell_bins(cell_bins), m_rank_to_bin_offset(rank_to_bin_offset) {}
    /*  const SFC& sfc：一个对 SFC 对象的常量引用，表示空间填充曲线。
        const MSH& mesh：一个对网格对象的常量引用，表示网格。
        const int* cell_bins：一个整数数组指针，表示每个单元格的 bin 分配（bin assignment）。cell_bins 数组指定每个单元格属于哪个 bin。
        int rank_to_bin_offset：表示进程 rank 与 bin 之间的偏移量，用于映射 rank 到正确的 bin。 */

  template<typename ForwardItr>
  void operator()(int rank, ForwardItr it) const
  {//类对象通过重载 operator()，可以像函数一样被使用
    for (typename MSH::index_type c = 0; c < m_mesh->num_local_cells()/* 网格中的局部单元格数量 */; ++c)
      if (m_cell_bins[c] == (rank + m_rank_to_bin_offset))
      {
        auto centroid = m_mesh->cell_centroid(c);//获取该单元格的质心（centroid）
        *it++ = m_sfc->index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid));
      }
  }

private:
  const SFC* m_sfc;
  const MSH* m_mesh;
  const int* m_cell_bins; // cells' bin assignments
  const int  m_rank_to_bin_offset; // mapps ranks to bins for a particular phase
};

template<typename ConstRandAccItr/* 权重数据迭代器 */, typename MSH>
struct weight_populator//权重迁移
{
  weight_populator(ConstRandAccItr wit, const MSH& mesh, const int* cell_bins, int rank_to_bin_offset)
    : m_wit(wit), m_mesh(&mesh), m_cell_bins(cell_bins), m_rank_to_bin_offset(rank_to_bin_offset) {}

  template<typename ForwardItr>
  void operator()(int rank, ForwardItr it) const
  {
    for (typename MSH::index_type c = 0; c < m_mesh->num_local_cells(); ++c)
      if (m_cell_bins[c] == (rank + m_rank_to_bin_offset)) 
        *it++ = *(m_wit + c); // weights are ordered the same way as the local cells of the mesh
  }

private:
  const ConstRandAccItr m_wit;
  const MSH* m_mesh;
  const int* m_cell_bins; // cells' bin assignments
  const int  m_rank_to_bin_offset; // mapps ranks to bins for a particular phase
};

template<typename IndexType, typename WP> // WP - weight policy权重策略
struct sfc_partitioner//将网格单元（cells）根据空间填充曲线（SFC）索引和权重策略进行分区（partition）
{
  sfc_partitioner(const std::int64_t* cell_sfc_indices, IndexType cell_count,
                  const IndexType* rank_offsets, int part_begin, double part_size,
                  double remaining_capacity, const WP& weight_policy)
    : m_cell_count(cell_count), m_cell_sfc_indices(cell_sfc_indices), m_rank_offsets(rank_offsets),
      m_part_begin(part_begin), m_part_size(part_size), m_remaining_capacity(remaining_capacity),
      m_weight_policy(&weight_policy), m_sorted_indices(cell_count)
  { 
    //按照 cell_sfc_indices 排序，使得 m_sorted_indices 中的索引顺序与 SFC 索引升序对应，方便后续分区
    for (IndexType i = 0 ; i < cell_count; ++i) m_sorted_indices[i] = i;
    std::sort(m_sorted_indices.begin(), m_sorted_indices.end(), [=](IndexType a, IndexType b)
              { return m_cell_sfc_indices[a] < m_cell_sfc_indices[b]; });
  }

  template<typename RandAccItr>
  void operator()(int rank, RandAccItr it) const
  {
    //确定本 rank 处理的单元格区间：用 rank_offsets 查区间
    IndexType rank_begin = m_rank_offsets[rank];
    IndexType rank_end = m_rank_offsets[rank + 1];
    for (IndexType i = 0; i < m_cell_count; ++i)
    {
      IndexType id = m_sorted_indices[i];
      if (id >= rank_begin && id < rank_end)//判断当前单元格是否属于本 rank
      {//A. 常量权重策略
        if constexpr(m_weight_policy->is_constant_weight)/* if constexpr：编译期判断，只有在权重策略是“常量权重”时，这段代码才会被启用。 */
          *(it + id - rank_begin) = i < m_remaining_capacity ? m_part_begin :
                                    (m_part_begin + static_cast<int>((i - m_remaining_capacity) / m_part_size) + 1);
      //B. 自定义权重策略
        else
        {
          // assume weights from the weight policy are in the same
          // order as cells assigned to this bin
          double weight = static_cast<double>(m_weight_policy->weight(id));
          assert(weight > 0);
          if (weight <= m_remaining_capacity)
            m_remaining_capacity -= weight;
          else
          {
            // partition numbers need to be continuous, so even if
            // the weight is so huge that it exceeds m_part_size
            // it is still assigned to the immediate next part, i.e.,
            // there is no skipping of part numbers
            m_remaining_capacity += (m_part_size - weight);
            ++m_part_begin;
          }

          // TODO: round-off errors might cause m_part_begin to pass
          // TODO: the allowed maximum - cap to be below m_part_end?
          *(it + id - rank_begin) = m_part_begin;
        }
      }
    }
  }

private:
  IndexType           m_cell_count;       // 分区中单元格的数量
  const std::int64_t* m_cell_sfc_indices; // 每个单元格的 SFC 索引数组
  const IndexType*    m_rank_offsets;     // 每个rank的分区起止
  mutable int         m_part_begin;
  mutable double      m_part_size;
  mutable double      m_remaining_capacity; // leftover capacity of the part "part_begin" for this bin
  const WP*           m_weight_policy;
  std::vector<IndexType> m_sorted_indices;  //SFC索引升序排序的单元格索引
};


// MSH - mesh, WP - weight policy, SFC - space filling curve
template<typename MSH, typename WP, typename OutputItr,
         template<typename> class SFC = sfc::hilbert_curve_3d>
void partition_impl(const MSH& mesh, int k, const WP& weight_policy, OutputItr oit, MPI_Comm comm)
{
  using R = typename MSH::coordinate_type;
  using I = typename MSH::index_type;
  using W = typename WP::weight_type;
  static_assert(std::is_arithmetic_v<R>, "wrong coordinate_type");
  static_assert(std::is_integral_v<I>, "wrong index_type");
  static_assert(std::is_arithmetic_v<W>, "wrong weight_type");
  static_assert((std::is_signed_v<I> && sizeof(I) <= sizeof(std::int64_t)) ||
                (std::is_unsigned_v<I> && sizeof(I) < sizeof(std::int64_t)), "index_type exceeds the range of int64_t");

  //dsp参数
  int retc;
  char *devProgram = "./mesh.dev.dat";
  int nthreads = 4;
  //根据进程数确定bin 数
  int num_processes, rank;
  MPI_Comm_size(comm, &num_processes);
  MPI_Comm_rank(comm, &rank);

  int bin_depth = 1, num_bins = 8;
  while (num_bins < num_processes) { num_bins <<= 3; bin_depth++; }
  assert(bin_depth <= 10); // bin_depth 不超过10，防止 num_bins 溢出int

  // 计算改线程位于第几组dsp簇的第几个
  int clusters_per_node = 4;
  int node_id = rank / clusters_per_node; // 计算节点 ID
  int cluster_id = rank % clusters_per_node;

  // 设备初始化
  retc = hthread_dev_open(cluster_id);
  retc = hthread_dat_load(cluster_id, devProgram);

  //有效线程数
  int availThreads = hthread_get_avail_threads(cluster_id);

  // 每个进程先算自己本地网格的坐标范围，再用 MPI 归约出全局边界，初始化 SFC 曲线。
  R min[3], max[3];
  std::tuple<R, R, R, R, R, R> lbbox = mesh.local_bounding_box();
  R lmin[3] = {std::get<0>(lbbox), std::get<1>(lbbox), std::get<2>(lbbox)};
  R lmax[3] = {std::get<3>(lbbox), std::get<4>(lbbox), std::get<5>(lbbox)};
  MPI_Allreduce(lmin, min, 3, mpi_datatype_v<R>, MPI_MIN, comm);
  MPI_Allreduce(lmax, max, 3, mpi_datatype_v<R>, MPI_MAX, comm);
  SFC<R> sfc(min[0], min[1], min[2], max[0], max[1], max[2]);

  // 根据每个单元格的质心,把单元格关联到SFC bin
  I num_local_cells = mesh.num_local_cells();
  std::vector<int> associated_bins(num_local_cells);
  for (I c = 0; c < num_local_cells; ++c)
  {
    std::tuple<R, R, R> centroid = mesh.cell_centroid(c);
    std::int64_t coarse_index = sfc.index(std::get<0>(centroid), std::get<1>(centroid), std::get<2>(centroid),
                                          bin_depth);
    assert(coarse_index >= 0 && coarse_index < num_bins);
    associated_bins[c] = static_cast<int>(coarse_index);
  }

  // 统计全体单元格的权重总和（并行求和）
  W total_weight, local_weight = weight_policy.local_weight();
  MPI_Allreduce(&local_weight, &total_weight, 1, mpi_datatype_v<W>, MPI_SUM, comm);

  // 循环主遍历：一次处理 num_processes 个 bin（每个进程负责一个 bin）。大数据量时分多轮，每轮只搬一批。
  std::vector<int> results(num_local_cells);
  std::vector<I> send_scheme(num_processes + 1); // need one extra space as it will be transferred to offsets later
  std::vector<I> recv_scheme(num_processes + 1); // need one extra space as it will be transferred to offsets later
  std::vector<W> bin_weights(num_processes);
  W phase_weight, prev_phase_weight = 0;
  for(int bin_begin = 0; bin_begin < num_bins; bin_begin += num_processes)
  {
    int bin_end = bin_begin + num_processes;
    if (bin_end > num_bins) bin_end = num_bins;//计算本轮要处理的 bin 范围，[bin_begin, bin_end)

    std::fill(send_scheme.begin(), send_scheme.end(), 0);
    std::fill(recv_scheme.begin(), recv_scheme.end(), 0);//清空发送/接收方案数组，初始化为 0。
    
    // // 分配 DSP 全局内存
    // I *d_send = (I *)hthread_malloc(cluster_id,(num_processes + 1)*sizeof(I), HT_MEM_RW);
    // for (int i = 0; i < num_processes + 1; ++i)
    //     d_send[i]=0;
    // 准备传 DSP 的参数列表
    
    uint64_t args[6];
    args[0] = num_local_cells;
    args[1] = bin_begin;
    args[2] = bin_end;
    args[3] = rank;
    args[4] = (uint64_t)(uintptr_t)associated_bins.data();
    args[5] = (uint64_t)(uintptr_t)send_scheme.data();

    
    // 启动 DSP 线程组并计时
    int tid = hthread_group_create(cluster_id,nthreads,"tran_cell", 4, 2, args);
    hthread_group_wait(tid);

    // // 从 DSP 内存拷回结果
    // for (int i = 0; i < num_processes + 1; ++i)
    //     send_scheme[i] = d_send[i];
 
    // 用 MPI_Alltoall 让每个进程把要发送的数据量告诉所有其它进程，并收到每个进程发来的数据量。
    MPI_Alltoall(send_scheme.data() + 1, 1, mpi_datatype_v<I>, recv_scheme.data() + 1, 1, mpi_datatype_v<I>, comm);

    //将“每个进程要发的数据量”累加成偏移量（offset）
    int num_send_processes = 0, num_recv_processes = 0;
    assert(send_scheme[0] == 0 && recv_scheme[0] == 0);
    for (std::size_t i = 1; i < send_scheme.size(); ++i)//遍历所有进程对应的通信方案（下标1到N）。
    {
      if (send_scheme[i] > 0) num_send_processes++;
      if (recv_scheme[i] > 0) num_recv_processes++;
      send_scheme[i] += send_scheme[i - 1];
      recv_scheme[i] += recv_scheme[i - 1];
    }

    // SFC 索引在进程间的点对点通信
    std::vector<std::int64_t> send_buffer(send_scheme[num_processes]);
    std::vector<std::int64_t> recv_buffer(recv_scheme[num_processes]);//分配发送缓冲区和接收缓冲区,大小分别为最后一个偏移量
    utils::point_to_point_communication( send_scheme.begin(), recv_scheme.begin(), send_scheme.size(),
                                         send_buffer.data(), recv_buffer.data(),
                                         sfc_index_generator(sfc, mesh, associated_bins.data(), bin_begin), comm );

    if constexpr (weight_policy.is_constant_weight)
    {
      // 所有进程都拿到全局各 bin 的权重（cell数量）
      MPI_Allgather(recv_scheme.data() + num_processes, 1, mpi_datatype_v<I>, bin_weights.data(), 1, mpi_datatype_v<W>, comm);

      //prev_bin_weight：比当前 rank 小的所有进程的 bin 的权重和（用于确定自己是全局第几个cell）。
      //phase_weight：本阶段所有进程加起来的总权重。
      W prev_bin_weight = prev_phase_weight;
      phase_weight = 0;
      for (int p = 0; p < num_processes; ++p)
      {
        W weight = bin_weights[p];
        if (p < rank) prev_bin_weight += weight;
        phase_weight += weight;
      }
      // 计算每个分区应承担的权重（part_size），以及当前进程负责 cell 的初始分区和剩余空间（residual）
      static_assert(std::is_integral_v<W> || (std::is_floating_point_v<W> && sizeof(W) <= sizeof(double)));
      double part_size = static_cast<double>(total_weight) / k;
      int init_part = static_cast<double>(prev_bin_weight) / part_size;
      double residual = (init_part + 1 ) * part_size - static_cast<double>(prev_bin_weight);

      //将分配好的分区方案通过 SFC 索引/权重，回传或通知到其他进程。
      utils::point_to_point_communication( recv_scheme.begin(), send_scheme.begin(), recv_scheme.size(),
                                           recv_buffer.data(), send_buffer.data(),
                                           sfc_partitioner(recv_buffer.data(), static_cast<I>(recv_buffer.size()),
                                                           recv_scheme.data(), init_part, part_size, residual,
                                                           // note that this is a new constant_weights policy that
                                                           // does not really matter, however
                                                           constant_weights(recv_buffer.size())),
                                           comm );
    }
    else
    {
      // 点对点通信，先把每个单元格的自定义权重分发到目标进程
      std::vector<W> weights_send_buffer(send_scheme[num_processes]);
      std::vector<W> weights_recv_buffer(recv_scheme[num_processes]);
      utils::point_to_point_communication( send_scheme.begin(), recv_scheme.begin(), send_scheme.size(),
                                           weights_send_buffer.data(), weights_recv_buffer.data(),
                                           weight_populator(weight_policy.weights_begin(), mesh, associated_bins.data(), bin_begin),
                                           comm );

      //MPI_Allgather 汇总所有进程的 bin 权重到 bin_weights，用于全局分区
      W bin_weight = std::accumulate(weights_recv_buffer.begin(), weights_recv_buffer.end(), 0);
      MPI_Allgather(&bin_weight, 1, mpi_datatype_v<W>, bin_weights.data(), 1, mpi_datatype_v<W>, comm);

      //累计自己的 bin 在全局是第几个（prev_bin_weight），并统计总权重（phase_weight）
      W prev_bin_weight = prev_phase_weight;
      phase_weight = 0;
      for (int p = 0; p < num_processes; ++p)
      {
        W weight = bin_weights[p];
        if (p < rank) prev_bin_weight += weight;
        phase_weight += weight;
      }
      //计算每个分区应承担的权重、当前进程的分区编号、以及当前分区还剩下多少权重空间
      static_assert(std::is_integral_v<W> || (std::is_floating_point_v<W> && sizeof(W) <= sizeof(double)));
      double part_size = static_cast<double>(total_weight) / k;
      int init_part = static_cast<double>(prev_bin_weight) / part_size;
      double residual = (init_part + 1 ) * part_size - static_cast<double>(prev_bin_weight);

      // 以每个cell的实际权重为基础，再做一次分区信息的分发
      utils::point_to_point_communication( recv_scheme.begin(), send_scheme.begin(), recv_scheme.size(),
                                           recv_buffer.data(), send_buffer.data(),
                                           sfc_partitioner(recv_buffer.data(), static_cast<I>(recv_buffer.size()),
                                                           recv_scheme.data(), init_part, part_size, residual,
                                                           // note that this is a new custom_weights policy working with
                                                           // weights assoicated to the bin that this rank is responsible for
                                                           custom_weights(weights_recv_buffer.begin(), weights_recv_buffer.size())),
                                           comm );
    }

    //将分配好的分区编号，根据 bin 对应关系逐个写回本地 cell 的最终结果数组
    for (int p = 0; p < num_processes; ++p)
    {
      I count = send_scheme[p + 1] - send_scheme[p];
      if (count > 0)
      {
        I index = send_scheme[p];
        for (I c = 0; c < num_local_cells; ++c)
          if (associated_bins[c] == (static_cast<int>(p) + bin_begin)) // map back ranks to bins
            results[c] = send_buffer[index++];
      }
    }

    prev_phase_weight += phase_weight;
  }

  for (I c = 0; c < num_local_cells; ++c) oit = results[c]; 
  
  retc = hthread_dat_unload(cluster_id); 
  hthread_dev_close(cluster_id);
}

// version with no custom weights: k is the desired number of parts to partition into
template<typename MSH, typename OutputItr>
void partition(const MSH& mesh, int k, OutputItr it, MPI_Comm comm)
{ partition_impl(mesh, k, constant_weights(mesh.num_local_cells()), it, comm); }

// version with custom cell weights
template<typename MSH, typename ConstRandAccItr, typename OutputItr>
void partition(const MSH& mesh, int k, ConstRandAccItr wit, OutputItr oit, MPI_Comm comm)
{ partition_impl(mesh, k, custom_weights(wit, mesh.num_local_cells()), oit, comm); }

} // namespace pmp

#endif
