#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"

// DSP 入口函数，供 host 调用
__global__
void tran_cell(uint64_t num_local_cells, uint64_t bin_begin, uint64_t bin_end, uint64_t rank, int *associated_bins,int *d_send) {
    // 获取线程 ID 和线程总数
    uint64_t thread_id = get_thread_id();
    uint64_t threads_count = get_group_size();
    uint64_t chunk = num_local_cells / threads_count;
    uint64_t start = thread_id * chunk;
    uint64_t end   = (thread_id == threads_count-1) ? num_local_cells : start + chunk;

    hthread_printf("[DSP %lu] [thread_id %lu] [start: %lu - end: %lu]\n", rank, start, end);

    // 遍历本进程的所有单元格（cell），查它属于哪个 bin
    for (uint64_t c = start; c < end; ++c)
    {
      uint64_t bin_index = associated_bins[c];
      if (bin_index >= bin_begin && bin_index < bin_end)  
        d_send[bin_index - bin_begin + 1]++;         
    }
}
