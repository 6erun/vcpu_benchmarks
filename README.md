# vcpu_benchmarks

## Overview

Scripts to measure and compare performance of different libvirt configurations of vcpu.
The goal is to find out how much performance we can get from different vcpu pinning and topology configurations, and how it compares to the performance of the host.

## NUMA nodes

Non-uniform memory access (NUMA) systems are server platforms with more than one system bus. These platforms can utilize multiple processors on a single motherboard, and all processors can access all the memory on the board. When a processor accesses memory that does not lie within its own node (remote memory), data must be transferred over the NUMA connection at a rate that is slower than it would be when accessing local memory. Thus, memory access times are not uniform and depend on the location (proximity) of the memory and the node from which it is accessed.

On servers with many CPU cores, they are clustered in NUMA (non-uniform memory access) nodes. It solves problem with UMA bottleneck (memory access through single system bus), but introduces latency when accessing other's node memory.

NUMA node is CPU + local RAM for low latency. To access other node's memory CPU has to ask another CPU and memory controller and that adds latency (The Remote Hop Cost).
You can pin a process to specific NUMA node using `numactl`.

NUMA nodes are not always equal to host sockets.