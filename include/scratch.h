// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "boost_dynamic_bitset_fwd.h"
// #include "boost/dynamic_bitset.hpp"
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "tsl/sparse_map.h"

#include "neighbor.h"
#include "concurrent_queue.h"
#include "pq.h"
#include "aligned_file_reader.h"

// In-mem index related limits
#define GRAPH_SLACK_FACTOR 1.3

// SSD Index related limits
#define MAX_GRAPH_DEGREE 512
#define SECTOR_LEN (size_t)4096
#define MAX_N_SECTOR_READS 128

namespace diskann
{

//
// Scratch space for in-memory index based search
//
// 用于在搜索过程中存储和管理各种临时数据和中间结果，优化搜索和过滤邻居的操作，并在搜索过程中提供一些临时存储空间。
template <typename T> class InMemQueryScratch
{
  public:
    ~InMemQueryScratch();
    // REFACTOR TODO: move all parameters to a new class.
    /* 
    search_l：搜索时使用的邻居数量 L。
    indexing_l：索引构建时使用的邻居数量 L。
    r：搜索半径，用于搜索过程中确定候选邻居。
    maxc：max_candidate_size，用来限制pool的大小
    */
    InMemQueryScratch(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim, size_t aligned_dim,
                      size_t alignment_factor, bool init_pq_scratch = false);
    void resize_for_new_L(uint32_t new_search_l);
    void clear();

    inline uint32_t get_L()
    {
        return _L;
    }
    inline uint32_t get_R()
    {
        return _R;
    }
    inline uint32_t get_maxc()
    {
        return _maxc;
    }
    inline T *aligned_query()
    {
        return _aligned_query;
    }
    inline PQScratch<T> *pq_scratch()
    {
        return _pq_scratch;
    }
    inline std::vector<Neighbor> &pool()
    {
        return _pool;
    }
    inline NeighborPriorityQueue &best_l_nodes()
    {
        return _best_l_nodes;
    }
    inline NeighborPriorityQueue &final_filtered_nodes()
    {
        return _final_filtered_nodes;
    }
    inline std::vector<float> &occlude_factor()
    {
        return _occlude_factor;
    }
    inline tsl::robin_set<uint32_t> &inserted_into_pool_rs()
    {
        return _inserted_into_pool_rs;
    }
    inline boost::dynamic_bitset<> &inserted_into_pool_bs()
    {
        return *_inserted_into_pool_bs;
    }
    inline tsl::robin_set<uint32_t> &expanded_into_pool_rs()
    {
        return _expanded_into_pool_rs;
    }
    inline boost::dynamic_bitset<> &expanded_into_pool_bs()
    {
        return *_expanded_into_pool_bs;
    }
    inline std::vector<uint32_t> &id_scratch()
    {
        return _id_scratch;
    }
    inline std::vector<float> &dist_scratch()
    {
        return _dist_scratch;
    }
    inline tsl::robin_set<uint32_t> &expanded_nodes_set()
    {
        return _expanded_nodes_set;
    }
    inline std::vector<Neighbor> &expanded_nodes_vec()
    {
        return _expanded_nghrs_vec;
    }
    inline std::vector<uint32_t> &occlude_list_output()
    {
        return _occlude_list_output;
    }

  private:
    uint32_t _L;
    uint32_t _R;
    uint32_t _maxc;

    T *_aligned_query = nullptr;

    PQScratch<T> *_pq_scratch = nullptr;

    // _pool stores all neighbors explored from best_L_nodes.
    // Usually around L+R, but could be higher.
    // Initialized to 3L+R for some slack, expands as needed.
    // 存储_best_l_nodes的node，以及从这些node扩展的node，最后是从pool来裁剪节点的邻居
    std::vector<Neighbor> _pool;

    // _best_l_nodes is reserved for storing best L entries
    // Underlying storage is L+1 to support inserts
    /*
    _best_l_nodes 是一个邻居优先级队列（NeighborPriorityQueue）。
    用于存储最近的 L 个节点。
    该队列具有固定大小（L+1），以支持插入操作。
    */
    NeighborPriorityQueue _best_l_nodes;

    // 存储最终符合filter的答案
    NeighborPriorityQueue _final_filtered_nodes;

    // _occlude_factor.size() >= pool.size() in occlude_list function
    // _pool is clipped to maxc in occlude_list before affecting _occlude_factor
    // _occlude_factor is initialized to maxc size
    // 用于存储影响节点被插入到 _pool 的因子
    std::vector<float> _occlude_factor;

    // Capacity initialized to 20L
    // 用于存储已经插入到 _pool 中的节点 ID
    tsl::robin_set<uint32_t> _inserted_into_pool_rs;

    // Use a pointer here to allow for forward declaration of dynamic_bitset
    // in public headers to avoid making boost a dependency for clients
    // of DiskANN.
    // 用于记录哪些节点已经被插入到了搜索过程中的邻居池 _pool 中，从而避免重复插入相同的节点。
    boost::dynamic_bitset<> *_inserted_into_pool_bs;

    tsl::robin_set<uint32_t> _expanded_into_pool_rs;
    boost::dynamic_bitset<> *_expanded_into_pool_bs;

    // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    std::vector<uint32_t> _id_scratch; // 存储节点 ID

    // _dist_scratch must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    // _dist_scratch should be at least the size of id_scratch
    std::vector<float> _dist_scratch; // 存储距离，和_id_scratch一起用，存储通过邻居节点扩展的节点信息

    //  Buffers used in process delete, capacity increases as needed
    tsl::robin_set<uint32_t> _expanded_nodes_set; // 用于存储已扩展的节点。
    std::vector<Neighbor> _expanded_nghrs_vec; // 存储邻居的向量，用于存储已扩展节点的邻居信息。
    std::vector<uint32_t> _occlude_list_output; // 用于存储已扩展的节点。
};

//
// Scratch space for SSD index based search
//

template <typename T> class SSDQueryScratch
{
  public:
    T *coord_scratch = nullptr; // MUST BE AT LEAST [sizeof(T) * data_dim]

    char *sector_scratch = nullptr; // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    size_t sector_idx = 0;          // index of next [SECTOR_LEN] scratch to use

    T *aligned_query_T = nullptr;

    PQScratch<T> *_pq_scratch;

    tsl::robin_set<size_t> visited;
    NeighborPriorityQueue retset; // == best_L_nodes
    std::vector<Neighbor> full_retset; // == _pool

    SSDQueryScratch(size_t aligned_dim, size_t visited_reserve);
    ~SSDQueryScratch();

    void reset();
};

template <typename T> class SSDThreadData
{
  public:
    SSDQueryScratch<T> scratch;
    IOContext ctx;

    SSDThreadData(size_t aligned_dim, size_t visited_reserve);
    void clear();
};

//
// Class to avoid the hassle of pushing and popping the query scratch.
// 高效地管理临时空间，确保临时空间在需要时可以分配给计算任务，并在不需要时可以正确地释放，以避免内存泄漏和频繁的内存分配开销。
//
template <typename T> class ScratchStoreManager
{
  public:
    ScratchStoreManager(ConcurrentQueue<T *> &query_scratch) : _scratch_pool(query_scratch)
    {
        _scratch = query_scratch.pop();
        while (_scratch == nullptr)
        {
            query_scratch.wait_for_push_notify();
            _scratch = query_scratch.pop();
        }
    }
    T *scratch_space()
    {
        return _scratch;
    }

    ~ScratchStoreManager()
    {
        _scratch->clear();
        _scratch_pool.push(_scratch);
        _scratch_pool.push_notify_all();
    }

    void destroy()
    {
        while (!_scratch_pool.empty())
        {
            auto scratch = _scratch_pool.pop();
            while (scratch == nullptr)
            {
                _scratch_pool.wait_for_push_notify();
                scratch = _scratch_pool.pop();
            }
            delete scratch;
        }
    }

  private:
    T *_scratch;
    ConcurrentQueue<T *> &_scratch_pool;
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
};
} // namespace diskann
