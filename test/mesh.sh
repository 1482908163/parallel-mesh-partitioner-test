#!/bin/bash

#SBATCH --job-name=mesh # 程序的作业名
#SBATCH --exclusive  # 不共享节点
#SBATCH --output=test_%j.out     # 标准输出文件
#SBATCH --error=test_%j.err      # 错误输出文件
#SBATCH --time=00:02:00               # 最长运行时间 (HH:MM:SS)
#SBATCH --partition=mt_module         # 使用的分区 (更新为 mt_module)            
#SBATCH --nodes=4                     # 请求的节点数
#SBATCH --ntasks-per-node=4         # 每个节点上的任务数 (进程数)
#SBATCH --ntasks=16                    # 请求的任务数 (总核数)
#SBATCH --exclude=cn7833,cn7836 

#export LD_LIBRARY_PATH=/vol8/home/hnu_test04/cjz/aarch64-linux-gnu:$LD_LIBRARY_PATH

yhrun -p mt_module --mpi=pmix  ./mesh.hos
