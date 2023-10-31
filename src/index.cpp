// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <type_traits>
#include <omp.h>

#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "boost/dynamic_bitset.hpp"

// #include <iostream>
// #include <type_traits>
// #include <string>
// #include <cstdint>

#include "memory_mapper.h"
#include "timer.h"
#include "windows_customizations.h"
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif
#include "index.h"
#define MAX_POINTS_FOR_USING_BITSET 10000000


namespace diskann
{
// Initialize an index with metric m, load the data of type T with filename
// (bin), and initialize max_points
template <typename T, typename TagT, typename LabelT>
Index<T, TagT, LabelT>::Index(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
                              const IndexWriteParameters &indexParams, const uint32_t initial_search_list_size,
                              const uint32_t search_threads, const bool enable_tags, const bool concurrent_consolidate,
                              const bool pq_dist_build, const size_t num_pq_chunks, const bool use_opq)
    : Index(m, dim, max_points, dynamic_index, enable_tags, concurrent_consolidate, pq_dist_build, num_pq_chunks,
            use_opq, indexParams.num_frozen_points)
{
    if (dynamic_index)
    {
        this->enable_delete();
    }
    _indexingQueueSize = indexParams.search_list_size;
    _indexingRange = indexParams.max_degree;
    _indexingMaxC = indexParams.max_occlusion_size;
    _indexingAlpha = indexParams.alpha;
    _filterIndexingQueueSize = indexParams.filter_list_size;

    uint32_t num_threads_indx = indexParams.num_threads;
    uint32_t num_scratch_spaces = search_threads + num_threads_indx;

    initialize_query_scratch(num_scratch_spaces, initial_search_list_size, _indexingQueueSize, _indexingRange,
                             _indexingMaxC, dim);
}

template <typename T, typename TagT, typename LabelT>
Index<T, TagT, LabelT>::Index(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
                              const bool enable_tags, const bool concurrent_consolidate, const bool pq_dist_build,
                              const size_t num_pq_chunks, const bool use_opq, const size_t num_frozen_pts,
                              const bool init_data_store) // num_pq_chunks是PQ_bytes
                                                          // init_data_store默认为true
    : _dist_metric(m), _dim(dim), _max_points(max_points), _num_frozen_pts(num_frozen_pts),
      _dynamic_index(dynamic_index), _enable_tags(enable_tags), _indexingMaxC(DEFAULT_MAXC), _query_scratch(nullptr),
      _pq_dist(pq_dist_build), _use_opq(use_opq), _num_pq_chunks(num_pq_chunks),
      _delete_set(new tsl::robin_set<uint32_t>), _conc_consolidate(concurrent_consolidate)
{
    if (dynamic_index && !enable_tags)
    {
        throw ANNException("ERROR: Dynamic Indexing must have tags enabled.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_pq_dist)
    {
        if (dynamic_index)
            throw ANNException("ERROR: Dynamic Indexing not supported with PQ distance based "
                               "index construction",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        if (m == diskann::Metric::INNER_PRODUCT)
            throw ANNException("ERROR: Inner product metrics not yet supported "
                               "with PQ distance "
                               "base index",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (dynamic_index && _num_frozen_pts == 0)
    {
        _num_frozen_pts = 1;
    }
    // Sanity check. While logically it is correct, max_points = 0 causes
    // downstream problems.
    if (_max_points == 0)
    {
        _max_points = 1;
    }
    const size_t total_internal_points = _max_points + _num_frozen_pts;

    if (_pq_dist)
    {
        if (_num_pq_chunks > _dim)
            throw diskann::ANNException("ERROR: num_pq_chunks > dim", -1, __FUNCSIG__, __FILE__, __LINE__);
        alloc_aligned(((void **)&_pq_data), total_internal_points * _num_pq_chunks * sizeof(char), 8 * sizeof(char));
        std::memset(_pq_data, 0, total_internal_points * _num_pq_chunks * sizeof(char));
    }

    _start = (uint32_t)_max_points;

    _final_graph.resize(total_internal_points);

    if (init_data_store)
    {
        // Issue #374: data_store is injected from index factory. Keeping this for backward compatibility.
        // distance is owned by data_store
        if (m == diskann::Metric::COSINE && std::is_floating_point<T>::value)
        {
            // This is safe because T is float inside the if block.
            this->_distance.reset((Distance<T> *)new AVXNormalizedCosineDistanceFloat());
            this->_normalize_vecs = true;
            diskann::cout << "Normalizing vectors and using L2 for cosine "
                             "AVXNormalizedCosineDistanceFloat()."
                          << std::endl;
        }
        else
        {
            this->_distance.reset((Distance<T> *)get_distance_function<T>(m));
        }
        // Note: moved this to factory, keeping this for backward compatibility.
        _data_store =
            std::make_unique<diskann::InMemDataStore<T>>((location_t)total_internal_points, _dim, this->_distance);
    }

    _locks = std::vector<non_recursive_mutex>(total_internal_points);

    if (enable_tags)
    {
        _location_to_tag.reserve(total_internal_points);
        _tag_to_location.reserve(total_internal_points);
    }
}

template <typename T, typename TagT, typename LabelT>
Index<T, TagT, LabelT>::Index(const IndexConfig &index_config, std::unique_ptr<AbstractDataStore<T>> data_store)
    : Index(index_config.metric, index_config.dimension, index_config.max_points, index_config.dynamic_index,
            index_config.enable_tags, index_config.concurrent_consolidate, index_config.pq_dist_build,
            index_config.num_pq_chunks, index_config.use_opq, index_config.num_frozen_pts, false) // 调用第二个构造函数
{

    _data_store = std::move(data_store);
    _distance.reset(_data_store->get_dist_fn());

    // enable delete by default for dynamic index
    if (_dynamic_index)
    {
        this->enable_delete();
    }
    if (_dynamic_index && index_config.index_write_params != nullptr)
    {
        _indexingQueueSize = index_config.index_write_params->search_list_size;
        _indexingRange = index_config.index_write_params->max_degree;
        _indexingMaxC = index_config.index_write_params->max_occlusion_size;
        _indexingAlpha = index_config.index_write_params->alpha;
        _filterIndexingQueueSize = index_config.index_write_params->filter_list_size;

        uint32_t num_threads_indx = index_config.index_write_params->num_threads;
        uint32_t num_scratch_spaces = index_config.search_threads + num_threads_indx;

        initialize_query_scratch(num_scratch_spaces, index_config.initial_search_list_size, _indexingQueueSize,
                                 _indexingRange, _indexingMaxC, _data_store->get_dims());
    }
}

template <typename T, typename TagT, typename LabelT> Index<T, TagT, LabelT>::~Index()
{
    // Ensure that no other activity is happening before dtor()
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    for (auto &lock : _locks)
    {
        LockGuard lg(lock);
    }

    // if (this->_distance != nullptr)
    //{
    //     delete this->_distance;
    //     this->_distance = nullptr;
    // }
    // REFACTOR

    if (_opt_graph != nullptr)
    {
        delete[] _opt_graph;
    }

    if (!_query_scratch.empty())
    {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        manager.destroy();
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l,
                                                      uint32_t r, uint32_t maxc, size_t dim)
{
    std::cout << "Init query scratch, use PQ:" << _pq_dist << ", # PQ chunks:" << _num_pq_chunks << std::endl;
    for (uint32_t i = 0; i < num_threads; i++)
    {
        auto scratch = new InMemQueryScratch<T>(search_l, indexing_l, r, maxc, dim, _data_store->get_aligned_dim(),
                                                _data_store->get_alignment_factor(), _pq_dist);
        _query_scratch.push(scratch);
    }
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::save_tags(std::string tags_file)
{
    if (!_enable_tags)
    {
        diskann::cout << "Not saving tags as they are not enabled." << std::endl;
        return 0;
    }
    size_t tag_bytes_written;
    TagT *tag_data = new TagT[_nd + _num_frozen_pts];
    for (uint32_t i = 0; i < _nd; i++)
    {
        TagT tag;
        if (_location_to_tag.try_get(i, tag))
        {
            tag_data[i] = tag;
        }
        else
        {
            // catering to future when tagT can be any type.
            std::memset((char *)&tag_data[i], 0, sizeof(TagT));
        }
    }
    if (_num_frozen_pts > 0)
    {
        std::memset((char *)&tag_data[_start], 0, sizeof(TagT) * _num_frozen_pts);
    }
    try
    {
        tag_bytes_written = save_bin<TagT>(tags_file, tag_data, _nd + _num_frozen_pts, 1);
    }
    catch (std::system_error &e)
    {
        throw FileException(tags_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }
    delete[] tag_data;
    return tag_bytes_written;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::save_data(std::string data_file)
{
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
    // location limit.
    return _data_store->save(data_file, (location_t)(_nd + _num_frozen_pts));
}

// save the graph index on a file as an adjacency list. For each point,
// first store the number of neighbors, and then the neighbor list (each as
// 4 byte uint32_t)
template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::save_graph(std::string graph_file)
{
    std::ofstream out;
    open_file_to_write(out, graph_file);

    size_t file_offset = 0; // we will use this if we want
    out.seekp(file_offset, out.beg);
    size_t index_size = 24; // number of bytes written
    uint32_t max_degree = 0;
    out.write((char *)&index_size, sizeof(uint64_t));           // 8
    out.write((char *)&_max_observed_degree, sizeof(uint32_t)); // 12
    uint32_t ep_u32 = _start;
    out.write((char *)&ep_u32, sizeof(uint32_t));        // 16
    out.write((char *)&_num_frozen_pts, sizeof(size_t)); // 24
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
    // location limit.
    for (uint32_t i = 0; i < _nd + _num_frozen_pts; i++)
    {
        uint32_t GK = (uint32_t)_final_graph[i].size();
        out.write((char *)&GK, sizeof(uint32_t));
        out.write((char *)_final_graph[i].data(), GK * sizeof(uint32_t));
        max_degree = _final_graph[i].size() > max_degree ? (uint32_t)_final_graph[i].size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    out.close();
    return index_size; // number of bytes written
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::save_delete_list(const std::string &filename)
{
    if (_delete_set->size() == 0)
    {
        return 0;
    }
    std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(_delete_set->size());
    uint32_t i = 0;
    for (auto &del : *_delete_set)
    {
        delete_list[i++] = del;
    }
    return save_bin<uint32_t>(filename, delete_list.get(), _delete_set->size(), 1);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save(const char *filename, bool compact_before_save)
{
    
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::save(const char *filename, std::string label_filename, bool compact_before_save)
{
    diskann::Timer timer;

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    if (compact_before_save)
    {
        compact_data();
        compact_frozen_point();
    }
    else
    {
        if (!_data_compacted)
        {
            throw ANNException("Index save for non-compacted index is not yet implemented", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        }
    }

    // graph reorder
    int w = 5;
    uint32_t u32_nd_ = _nd + _num_frozen_pts;
    auto& projection_graph_ = _final_graph;

    // /*
    // Find entry node for gorder
    unsigned int seed_node = _start;
    std::cout << "In gorder, vertices: "  << u32_nd_ << ", start: " << _start << std::endl;
    // Create table of in-degrees
    std::unordered_map<unsigned int,std::vector<unsigned int> > indegree_table;
    for (unsigned int node = 0; node < u32_nd_; node++){
        // unsigned int* node_edges = dataNodeLinks(node);
        unsigned int* node_edges = projection_graph_[node].data();
        for (int m = 0; m < projection_graph_[node].size(); m++){
            if (node_edges[m] != node){
                // this was a bug (i think):
                // indegree_table[node].push_back(node_edges[m]);
                indegree_table[node_edges[m]].push_back(node);
            }
        }
    }

    std::cout << "starting reordering ..." << std::endl;
    // do the actual gorder
    GorderPriorityQueue<unsigned int> Q(u32_nd_);
    std::vector<unsigned int> P(u32_nd_, 0);
    Q.increment(seed_node);
    P[0] = Q.pop();

    // for i = 1 to N:
    for (int i = 1; i < u32_nd_; i++){
        unsigned int v_e = P[i-1];
        // ve = P[i-1] # the newest node in window
        // for each node u in out-edges of ve:
        unsigned int* v_e_edges = projection_graph_[v_e].data();
        for (int m = 0; m < projection_graph_[v_e].size(); m++){
            if (v_e_edges[m] != v_e){
                // if u in Q, increment priority of u
                Q.increment(v_e_edges[m]);
            }
        }

        // for each node u in in-edges of ve: 
        for (unsigned int& u : indegree_table[v_e]){
            // if u in Q, increment priority of u
            Q.increment(u);
            // for each node v in out-edges of u:
            unsigned int* u_edges = projection_graph_[u].data();
            for (int m = 0; m < projection_graph_[u].size(); m++){
                if (u_edges[m] != u){
                    // if v in Q, increment priority of v
                    Q.increment(u_edges[m]);
                }
            }
        }

        if (i > w+1){
            unsigned int v_b = P[i-w-1];
            // for each node u in out-edges of vb:
            unsigned int* v_b_edges = projection_graph_[v_b].data();
            for (int m = 0; m < projection_graph_[v_b].size(); m++){
                if (v_b_edges[m] != v_b){
                    // if u in Q, decrement priority of u
                    Q.decrement(v_b_edges[m]);
                }
            }
            // for each node u in in-edges of vb:
            for (unsigned int& u : indegree_table[v_b]){
                // if u in Q, increment priority of u
                Q.increment(u); // this could be a bug?
                // for each node v in out-edges of u:
                unsigned int* u_edges = projection_graph_[u].data();
                for (int m = 0; m < projection_graph_[u].size(); m++){
                    if (u_edges[m] != u){
                        // if v in Q, decrement priority of v
                        Q.decrement(u_edges[m]);
                    }
                }
            }
        }
        P[i] = Q.pop();
        // add progress, no change line, flush in same line
        if (i % 100000 == 0){
            std::cout << "\r" << float(100*i) / u32_nd_ << "% processed." << std::flush;
        }
    }
    // */

    std::vector<uint32_t> raw_to_new_id;
    raw_to_new_id.resize(u32_nd_);
    _new_to_raw_id.resize(u32_nd_);

    // use reorder
    for (int n = 0; n < u32_nd_; n++){
        raw_to_new_id[P[n]] = n;
        _new_to_raw_id[n] = P[n];
    }

    // not use reorder
    // for (int n = 0; n < u32_nd_; n++){
    //     _new_to_raw_id[n] = n;
    //     raw_to_new_id[n] = n;
    // }

    std::cout << "Saving index to " << std::string(filename) << std::endl;
    if (!_save_as_one_file)
    {
        std::ofstream ofs;

        // build CSR
        std::cout << "Processing CSR ..." << std::endl;
        std::vector<uint32_t> csr_offset, csr_data;
        csr_offset.push_back(0);
        for (uint32_t i = 0; i < u32_nd_; i++) {
            auto& raw_neighbors = projection_graph_[_new_to_raw_id[i]];
            for (uint32_t j = 0; j < raw_neighbors.size(); j++) {
                csr_data.push_back(raw_to_new_id[raw_neighbors[j]]);
            }
            csr_offset.push_back(csr_data.size());
        }

        // write CSR to file
        std::string csr_offset_file = std::string(filename) + "_csr_offset.bin";
        std::string csr_data_file = std::string(filename) + "_csr_data.bin";
        uint32_t csr_offset_size = (uint32_t)csr_offset.size(), csr_data_size = (uint32_t)csr_data.size();
        ofs.open(csr_offset_file, std::ios::binary);
        ofs.write((char *)&csr_offset_size, sizeof(uint32_t));
        ofs.write((char *)csr_offset.data(), csr_offset.size() * sizeof(uint32_t));
        ofs.close();
        ofs.open(csr_data_file, std::ios::binary);
        ofs.write((char *)&csr_data_size, sizeof(uint32_t));
        ofs.write((char *)csr_data.data(), csr_data.size() * sizeof(uint32_t));
        ofs.write((char *)&_start, sizeof(uint32_t));
        ofs.close();

        // write new_to_raw_id to file
        std::string new_to_raw_file = std::string(filename) + "_new_to_raw.txt";
        ofs.open(new_to_raw_file);
        for (size_t i = 0; i < _new_to_raw_id.size(); i++) {
            ofs << _new_to_raw_id[i] << std::endl;
        }
        ofs.close();

        // rewrite label file
        uint32_t num_points = 0, num_labels = 0;
        std::cout << "label_file: " << label_filename << std::endl;
        std::string rewrite_label_file = std::string(filename) + "_label.bin";
        parse_label_file_neurips23(label_filename, num_points, num_labels, rewrite_label_file);
        int threshold = 90;

        // process label_to_medoid
        std::cout << "Processing label_to_medoid ..." << std::endl;
        _label_to_medoid_id.resize(num_labels);
        uint32_t num_cands = 25;
        for (auto curr_label = 0; curr_label < num_labels; curr_label++) // 遍历标签，找到该标签的中心点
        {
            uint32_t best_medoid_count = std::numeric_limits<uint32_t>::max();
            // auto &curr_label = *itr;
            uint32_t best_medoid = _start;
            auto& start = _label_to_pts_offset[curr_label];
            auto& end = _label_to_pts_offset[curr_label + 1];
            if (end - start == 0) {
                _label_to_medoid_id[curr_label] = best_medoid;
                _medoid_counts[best_medoid]++;
                continue;
            }

            // auto labeled_points = _label_to_pts[curr_label]; // 满足f标签的所有point
            for (uint32_t cnd = 0; cnd < num_cands; cnd++)     // 从Pf里抽样得到的num_cands个points
            {
                uint32_t cur_cnd = _label_to_pts_data[rand() % (end - start) + start];
                uint32_t cur_cnt = std::numeric_limits<uint32_t>::max();
                if (_medoid_counts.find(cur_cnd) == _medoid_counts.end())
                {
                    _medoid_counts[cur_cnd] = 0;
                    cur_cnt = 0;
                }
                else
                {
                    cur_cnt = _medoid_counts[cur_cnd];
                }
                if (cur_cnt < best_medoid_count) // 出现次数最少的point
                {
                    best_medoid_count = cur_cnt;
                    best_medoid = cur_cnd;
                }
            }
            _label_to_medoid_id[curr_label] = best_medoid; // f标签的中心点就是p*，另外将p*出现的次数+1
            _medoid_counts[best_medoid]++;
        }
        
        // write label_to_medoid to file
        if (_label_to_medoid_id.size() > 0)
        {
            std::ofstream medoid_writer(std::string(filename) + "_labels_to_medoids.txt");
            if (medoid_writer.fail())
            {
                throw diskann::ANNException(std::string("Failed to open file 8") + filename, -1);
            }
            for (auto i = 0; i < _label_to_medoid_id.size(); i++)
            {
                medoid_writer << i << ", " << raw_to_new_id[_label_to_medoid_id[i]] << std::endl;
            }
            medoid_writer.close();
        }

        // process label_to_hash
        std::cout << "Processing label and pts hash ..." << std::endl;
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> dist(1, 100);
        _label_to_hash.resize(num_labels);
        for (size_t i = 0; i < num_labels; i++) {
            int64_t val = 0;
            for (size_t j = 0; j < 64; j++) {
                val = val | ((dist(rng) > threshold) << j);
            }
            _label_to_hash[i] = val;        
        }

        // process pts_to_hash
        for (size_t i = 0; i < num_points; i++) {
            int64_t val = 0;
            for (auto j = _pts_to_labels_hash_and_offset[2*i + 1]; j < _pts_to_labels_hash_and_offset[2*i + 3]; ++j) {
                val = val | _label_to_hash[_pts_to_labels_data[j]];
            }
            _pts_to_labels_hash_and_offset[2*i] = val;        
        }

        // write labels_to_hash to file
        std::string labels_to_hash_file = std::string(filename) + "_labels_to_hash.txt";
        ofs.open(labels_to_hash_file);
        for (size_t i = 0; i < _label_to_hash.size(); i++) {
            ofs << _label_to_hash[i] << std::endl;
        }
        ofs.close();

        // write pts_to_hash to file
        std::string pts_to_hash_file = std::string(filename) + "_pts_to_hash.txt";
        ofs.open(pts_to_hash_file);
        for (size_t i = 0; i < num_points; i++) {
            ofs << _pts_to_labels_hash_and_offset[2*(_new_to_raw_id[i])] << std::endl;
        }
        ofs.close();

        int npts = (int)_nd, ndims = (int)(_data_store->get_dims()), aligned_dim = _data_store->get_aligned_dim();
        std::cout << "Writing data vectors, n = " << npts << ", dim = " << ndims << std::endl;
        ofs.open(std::string(filename) + ".data", std::ios::binary);
        T* tmp_vec = new T[aligned_dim];
        size_t bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
        ofs.write((char *)&npts, sizeof(int));
        ofs.write((char *)&ndims, sizeof(int));
        for (size_t i = 0; i < npts; i++)
        {
            _data_store->get_vector(_new_to_raw_id[i], tmp_vec);
            ofs.write((char *)tmp_vec, ndims * sizeof(T));
        }
        ofs.close();
        delete[] tmp_vec;

        std::string graph_file = std::string(filename);
        std::string tags_file = std::string(filename) + ".tags";
        std::string data_file = std::string(filename) + ".data";
        std::string delete_list_file = std::string(filename) + ".del";

        delete_file(tags_file);
        save_tags(tags_file);
        delete_file(delete_list_file);
        save_delete_list(delete_list_file);
    }
    else
    {
        diskann::cout << "Save index in a single file currently not supported. "
                         "Not saving the index."
                      << std::endl;
    }

    // If frozen points were temporarily compacted to _nd, move back to _max_points.
    reposition_frozen_point_to_end();
    diskann::cout << "Time taken for transform and save: " << timer.elapsed() / 1000000.0 << "s." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_tags(AlignedFileReader &reader)
{
#else
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_tags(const std::string tag_filename)
{
    if (_enable_tags && !file_exists(tag_filename))
    {
        diskann::cerr << "Tag file " << tag_filename << " does not exist!" << std::endl;
        throw diskann::ANNException("Tag file " + tag_filename + " does not exist!", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }
#endif
    if (!_enable_tags)
    {
        diskann::cout << "Tags not loaded as tags not enabled." << std::endl;
        return 0;
    }

    size_t file_dim, file_num_points;
    TagT *tag_data;
#ifdef EXEC_ENV_OLS
    load_bin<TagT>(reader, tag_data, file_num_points, file_dim);
#else
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points, file_dim);
#endif

    if (file_dim != 1)
    {
        std::stringstream stream;
        stream << "ERROR: Found " << file_dim << " dimensions for tags,"
               << "but tag file must have 1 dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        delete[] tag_data;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const size_t num_data_points = file_num_points - _num_frozen_pts;
    _location_to_tag.reserve(num_data_points);
    _tag_to_location.reserve(num_data_points);
    for (uint32_t i = 0; i < (uint32_t)num_data_points; i++)
    {
        TagT tag = *(tag_data + i);
        if (_delete_set->find(i) == _delete_set->end())
        {
            _location_to_tag.set(i, tag);
            _tag_to_location[tag] = i;
        }
    }
    diskann::cout << "Tags loaded." << std::endl;
    delete[] tag_data;
    return file_num_points;
}

template <typename T, typename TagT, typename LabelT>
#ifdef EXEC_ENV_OLS
size_t Index<T, TagT, LabelT>::load_data(AlignedFileReader &reader)
{
#else
size_t Index<T, TagT, LabelT>::load_data(std::string filename)
{
#endif
    size_t file_dim, file_num_points;
#ifdef EXEC_ENV_OLS
    diskann::get_bin_metadata(reader, file_num_points, file_dim);
#else
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);
#endif

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > _max_points + _num_frozen_pts)
    {
        // update and tag lock acquired in load() before calling load_data
        resize(file_num_points - _num_frozen_pts);
    }

#ifdef EXEC_ENV_OLS
    // REFACTOR TODO: Must figure out how to support aligned reader in a clean manner.
    copy_aligned_data_from_file<T>(reader, _data, file_num_points, file_dim, _data_store->get_aligned_dim());
#else
    _data_store->load(filename); // offset == 0.
#endif
    return file_num_points;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_delete_set(AlignedFileReader &reader)
{
#else
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_delete_set(const std::string &filename)
{
#endif
    std::unique_ptr<uint32_t[]> delete_list;
    size_t npts, ndim;

#ifdef EXEC_ENV_OLS
    diskann::load_bin<uint32_t>(reader, delete_list, npts, ndim);
#else
    diskann::load_bin<uint32_t>(filename, delete_list, npts, ndim);
#endif
    assert(ndim == 1);
    for (uint32_t i = 0; i < npts; i++)
    {
        _delete_set->insert(delete_list[i]);
    }
    return npts;
}

// load the index from file and update the max_degree, cur (navigating
// node loc), and _final_graph (adjacency list)
template <typename T, typename TagT, typename LabelT>
#ifdef EXEC_ENV_OLS
void Index<T, TagT, LabelT>::load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l)
{
#else
void Index<T, TagT, LabelT>::load(const char *filename, uint32_t num_threads, uint32_t search_l)
{
#endif
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    _has_built = true;

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;
    uint32_t label_num_pts = 0, num_labels = 0;
#ifndef EXEC_ENV_OLS
    std::string mem_index_file(filename);
    std::string labels_file = mem_index_file + "_label.bin";
    std::string labels_to_medoids = mem_index_file + "_labels_to_medoids.txt";
    std::string labels_map_file = mem_index_file + "_labels_map.txt";
#endif
    if (!_save_as_one_file)
    {
        // For DLVS Store, we will not support saving the index in multiple
        // files.
#ifndef EXEC_ENV_OLS
        std::string data_file = std::string(filename) + ".data";
        std::string tags_file = std::string(filename) + ".tags";
        std::string delete_set_file = std::string(filename) + ".del";
        // std::string graph_file = std::string(filename);

        std::cout << "Load vector data from " << data_file << std::endl;
        data_file_num_pts = load_data(data_file);
        if (file_exists(delete_set_file))
        {
            load_delete_set(delete_set_file);
        }

        // if (_enable_tags)
        // {
        //     tags_file_num_pts = load_tags(tags_file);
        // }
        // graph_num_pts = load_graph(graph_file, data_file_num_pts);

        std::cout << "Load Graph CSR" << std::endl;
        std::string csr_offset_file = std::string(filename) + "_csr_offset.bin";
        std::string csr_data_file = std::string(filename) + "_csr_data.bin";

        // 读取 csr_offset_file 文件
        std::ifstream csr_offset_reader(csr_offset_file, std::ios::binary);
        uint32_t csr_offset_size = 0;
        csr_offset_reader.read(reinterpret_cast<char*>(&csr_offset_size), sizeof(csr_offset_size));
        _csr_offset.resize(csr_offset_size);
        csr_offset_reader.read(reinterpret_cast<char*>(_csr_offset.data()), csr_offset_size * sizeof(_csr_offset[0]));
        csr_offset_reader.close();

        // 读取 csr_data_file 文件
        std::ifstream csr_data_reader(csr_data_file, std::ios::binary);
        uint32_t csr_data_size = 0;
        csr_data_reader.read(reinterpret_cast<char*>(&csr_data_size), sizeof(csr_data_size));
        _csr_data.resize(csr_data_size);
        csr_data_reader.read(reinterpret_cast<char*>(_csr_data.data()), csr_data_size * sizeof(_csr_data[0]));
        csr_data_reader.read(reinterpret_cast<char*>(&_start), sizeof(_start));
        csr_data_reader.close();

        graph_num_pts = csr_offset_size - 1;
        _max_range_of_loaded_graph = 0;
        for (size_t i=0; i<csr_offset_size-1; ++i) {
            if (_max_range_of_loaded_graph < _csr_offset[i+1] - _csr_offset[i])
                _max_range_of_loaded_graph = _csr_offset[i+1] - _csr_offset[i];
        }

        std::string new_to_raw_file = std::string(filename) + "_new_to_raw.txt";
        std::ifstream new_to_raw_reader(new_to_raw_file);
        _new_to_raw_id.resize(_csr_offset.size() - 1);
        for (size_t i = 0; i < _new_to_raw_id.size(); i++) {
            new_to_raw_reader >> _new_to_raw_id[i];
        }
        new_to_raw_reader.close();
#endif
    }
    else
    {
        diskann::cout << "Single index file saving/loading support not yet "
                         "enabled. Not loading the index."
                      << std::endl;
        return;
    }

    if (data_file_num_pts != graph_num_pts || (data_file_num_pts != tags_file_num_pts && _enable_tags))
    {
        std::stringstream stream;
        stream << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
               << graph_num_pts << " from graph, and " << tags_file_num_pts
               << " tags, with num_frozen_pts being set to " << _num_frozen_pts << " in constructor." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
#ifndef EXEC_ENV_OLS
    if (file_exists(labels_file))
    {
        std::cout << "Reading label file: " << labels_file << std::endl;
        std::string empty_str = "";
        parse_label_file_neurips23(labels_file, label_num_pts, num_labels, empty_str);
        assert(label_num_pts == data_file_num_pts);
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            std::string line, token;
            uint32_t line_cnt = 0;

            _label_to_medoid_id.resize(num_labels);

            while (std::getline(medoid_stream, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t medoid = 0;
                LabelT label;
                while (std::getline(iss, token, ','))
                {
                    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                    uint32_t token_as_num = std::stoul(token);
                    if (cnt == 0)
                        label = (LabelT)token_as_num;
                    else
                        medoid = token_as_num;
                    cnt++;
                }
                _label_to_medoid_id[label] = medoid;
                line_cnt++;
            }
        }

        std::string universal_label_file(filename);
        universal_label_file += "_universal_label.txt";
        if (file_exists(universal_label_file))
        {
            std::ifstream universal_label_reader(universal_label_file);
            universal_label_reader >> _universal_label;
            _use_universal_label = true;
            universal_label_reader.close();
        }
    }

    std::string labels_to_hash_file = std::string(filename) + "_labels_to_hash.txt";
    std::ifstream labels_to_hash_reader(labels_to_hash_file);
    int64_t val = 0;
    while (labels_to_hash_reader >> val) {
        _label_to_hash.push_back(val);
    }
        
    std::string pts_to_hash_file = std::string(filename) + "_pts_to_hash.txt";
    std::ifstream pts_to_hash_reader(pts_to_hash_file);
    int i = 0;
    while (pts_to_hash_reader >> val) {
        _pts_to_labels_hash_and_offset[2*i] = val;
        ++i;
    }

#endif
    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }

    reposition_frozen_point_to_end();
    diskann::cout << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd << " _start: " << _start
                  << " size(_location_to_tag): " << _location_to_tag.size()
                  << " size(_tag_to_location):" << _tag_to_location.size() << " Max points: " << _max_points
                  << std::endl;

    visited_list_pool_ = new VisitedListPool(2*num_threads, _nd);
}

#ifndef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::get_graph_num_frozen_points(const std::string &graph_file)
{
    size_t expected_file_size;
    uint32_t max_observed_degree, start;
    size_t file_frozen_pts;

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);

    in.open(graph_file, std::ios::binary);
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&max_observed_degree, sizeof(uint32_t));
    in.read((char *)&start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));

    return file_frozen_pts;
}
#endif

#ifdef EXEC_ENV_OLS
template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_graph(AlignedFileReader &reader, size_t expected_num_points)
{
#else

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::load_graph(std::string filename, size_t expected_num_points)
{
#endif
    size_t expected_file_size;
    size_t file_frozen_pts;

#ifdef EXEC_ENV_OLS
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((size_t *)header.get());
    _max_observed_degree = *((uint32_t *)(header.get() + sizeof(size_t)));
    _start = *((uint32_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t)));
    file_frozen_pts = *((size_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));
#else

    size_t file_offset = 0; // will need this for single file format support
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&_max_observed_degree, sizeof(uint32_t));
    in.read((char *)&_start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));
    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

#endif
    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << _start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (file_frozen_pts != _num_frozen_pts)
    {
        std::stringstream stream;
        if (file_frozen_pts == 1)
        {
            stream << "ERROR: When loading index, detected dynamic index, but "
                      "constructor asks for static index. Exitting."
                   << std::endl;
        }
        else
        {
            stream << "ERROR: When loading index, detected static index, but "
                      "constructor asks for dynamic index. Exitting."
                   << std::endl;
        }
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

#ifdef EXEC_ENV_OLS
    diskann::cout << "Loading vamana graph from reader..." << std::flush;
#else
    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;
#endif

    const size_t expected_max_points = expected_num_points - file_frozen_pts;

    // If user provides more points than max_points
    // resize the _final_graph to the larger size.
    if (_max_points < expected_max_points)
    {
        diskann::cout << "Number of points in data: " << expected_max_points
                      << " is greater than max_points: " << _max_points
                      << " Setting max points to: " << expected_max_points << std::endl;
        _final_graph.resize(expected_max_points + _num_frozen_pts);
        _max_points = expected_max_points;
    }
#ifdef EXEC_ENV_OLS
    uint32_t nodes_read = 0;
    size_t cc = 0;
    size_t graph_offset = header_size;
    while (nodes_read < expected_num_points)
    {
        uint32_t k;
        read_value(reader, k, graph_offset);
        graph_offset += sizeof(uint32_t);
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        read_array(reader, tmp.data(), k, graph_offset);
        graph_offset += k * sizeof(uint32_t);
        cc += k;
        _final_graph[nodes_read].swap(tmp);
        nodes_read++;
        if (nodes_read % 1000000 == 0)
        {
            diskann::cout << "." << std::flush;
        }
        if (k > _max_range_of_loaded_graph)
        {
            _max_range_of_loaded_graph = k;
        }
    }
#else
    size_t bytes_read = vamana_metadata_size;
    size_t cc = 0;
    uint32_t nodes_read = 0;

    uint32_t err_nodes = 0;
    // std::unordered_map<uint32_t, uint32_t> err_nodes_map; // id, in_degree

    while (bytes_read != expected_file_size)
    {
        uint32_t k;
        in.read((char *)&k, sizeof(uint32_t));

        cc += k;
        ++nodes_read;
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        in.read((char *)tmp.data(), k * sizeof(uint32_t));
        _final_graph[nodes_read - 1].swap(tmp);
        if (k == 0)
        {
            ++err_nodes;
            // diskann::cerr << "ERROR: Point found with no out-neighbors, point#" << nodes_read << std::endl;
            // err_nodes_map.insert({nodes_read - 1, 0});
        }
        bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
        if (nodes_read % 10000000 == 0)
            diskann::cout << "." << std::flush;
        if (k > _max_range_of_loaded_graph)
        {
            _max_range_of_loaded_graph = k;
        }
    }
#endif
    diskann::cerr << "err_nodes :" << err_nodes << std::endl;
    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to "
                  << _start << std::endl;
    return nodes_read;
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::_get_vector_by_tag(TagType &tag, DataType &vec)
{
    try
    {
        TagT tag_val = std::any_cast<TagT>(tag);
        T *vec_val = std::any_cast<T *>(vec);
        return this->get_vector_by_tag(tag_val, vec_val);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing _get_vector_by_tags() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::get_vector_by_tag(TagT &tag, T *vec)
{
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end())
    {
        diskann::cout << "Tag " << tag << " does not exist" << std::endl;
        return -1;
    }

    location_t location = _tag_to_location[tag];
    _data_store->get_vector(location, vec);

    return 0;
}

template <typename T, typename TagT, typename LabelT> uint32_t Index<T, TagT, LabelT>::calculate_entry_point()
{
    //  TODO: need to compute medoid with PQ data too, for now sample at random
    if (_pq_dist)
    {
        size_t r = (size_t)rand() * (size_t)RAND_MAX + (size_t)rand();
        return (uint32_t)(r % (size_t)_nd);
    }

    // TODO: This function does not support multi-threaded calculation of medoid.
    // Must revisit if perf is a concern.
    return _data_store->calculate_medoid();
}

template <typename T, typename TagT, typename LabelT> std::vector<uint32_t> Index<T, TagT, LabelT>::get_init_ids()
{
    std::vector<uint32_t> init_ids;
    init_ids.reserve(1 + _num_frozen_pts);

    init_ids.emplace_back(_start);

    for (uint32_t frozen = (uint32_t)_max_points; frozen < _max_points + _num_frozen_pts; frozen++)
    {
        if (frozen != _start)
        {
            init_ids.emplace_back(frozen);
        }
    }

    return init_ids;
}

// Find whether the query labels are contained, not taking into account universal label
template <typename T, typename TagT, typename LabelT>
bool Index<T, TagT, LabelT>::contain_required_filters(const uint32_t& point_id, const int32_t &label1, const int32_t &label2)
{
    auto& start = _pts_to_labels_hash_and_offset[2*point_id+1];
    auto& end = _pts_to_labels_hash_and_offset[2*point_id+3];
    if (end - start == 0 || label1 > _pts_to_labels_data[end-1] || label1 < _pts_to_labels_data[start])
        return false;
    if (std::binary_search(_pts_to_labels_data + start, _pts_to_labels_data + end, label1) == false)
        return false;
    if (label2 != -1) {
        if (end - start == 0 || label2 > _pts_to_labels_data[end-1] || label2 < _pts_to_labels_data[start])
            return false;
        if (std::binary_search(_pts_to_labels_data + start, _pts_to_labels_data + end, label2) == false)
            return false;
    }
    return true;
}

/*
    为node节点找L个候选邻居（从_start出发，从它的邻居中扩展候选集，再从扩展的邻居中继续扩展），结果存在best_L_nodes
    并将扩展的邻居存在pool中，为后续使用算法2做铺垫
*/
template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::iterate_to_fixed_point(const T *query, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, InMemQueryScratch<T> *scratch,
    bool use_filter, const std::vector<LabelT> &filter_label, bool search_invocation)
{
    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    assert(id_scratch.size() == 0);

    // REFACTOR
    // T *aligned_query = scratch->aligned_query();
    // memcpy(aligned_query, query, _dim * sizeof(T));
    // if (_normalize_vecs)
    //{
    //     normalize((float *)aligned_query, _dim);
    // }

    T *aligned_query = scratch->aligned_query();

    float *query_float = nullptr;
    float *query_rotated = nullptr;
    float *pq_dists = nullptr;
    uint8_t *pq_coord_scratch = nullptr;
    // Intialize PQ related scratch to use PQ based distances
    if (_pq_dist)
    {
        // Get scratch spaces
        PQScratch<T> *pq_query_scratch = scratch->pq_scratch();
        query_float = pq_query_scratch->aligned_query_float;
        query_rotated = pq_query_scratch->rotated_query;
        pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;

        // Copy query vector to float and then to "rotated" query
        for (size_t d = 0; d < _dim; d++)
        {
            query_float[d] = (float)aligned_query[d];
        }
        pq_query_scratch->set(_dim, aligned_query);

        // center the query and rotate if we have a rotation matrix
        _pq_table.preprocess_query(query_rotated);
        _pq_table.populate_chunk_distances(query_rotated, pq_dists);

        pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;
    }

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0)
    {
        throw ANNException("ERROR: Clear scratch space before passing.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            // hopefully using 2X will reduce the number of allocations.
            auto resize_size =
                2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }

    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const std::vector<uint32_t> &ids,
                                                            std::vector<float> &dists_out) {
        diskann::aggregate_coords(ids, this->_pq_data, this->_num_pq_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, ids.size(), this->_num_pq_chunks, pq_dists, dists_out);
    };

    // Initialize the candidate pool with starting points
    float distance;
    for (
        auto id :
        init_ids) // 遍历init_ids，计算与query的距离，得到最近的L个node（init_ids默认只有_start节点，有filter时是多个label_start节点）
    {
        if (id >= _max_points + _num_frozen_pts)
        {
            diskann::cerr << "Out of range loc found as an edge : " << id << std::endl;
            throw diskann::ANNException(std::string("Wrong loc") + std::to_string(id), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }

        if (use_filter)
        {
            // 检查起始节点和query有没有相同的标签，没有则不能将其加入候选集
            // 这一步似乎没有必要啊，因为起始节点肯定有
            // if (!detect_common_filters(id, search_invocation, filter_label)) {
            //     continue;
            // }
        }

        if (is_not_visited(id))
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
            
            if (_pq_dist)
            {
                pq_dist_lookup(pq_coord_scratch, 1, this->_num_pq_chunks, pq_dists, &distance);
            }
            else
            {
                distance = _data_store->get_distance(aligned_query, id);
            }
            Neighbor nn = Neighbor(id, distance);
            best_L_nodes.insert(nn);
        }
    }

    // std::cout << "best_L_nodes size: " << best_L_nodes.size() << std::endl;
    uint32_t hops = 0;
    uint32_t cmps = 0;
    // 通过候选集来扩展邻居集（可能c原本不是候选，但是a->b->c的距离比较近）
    while (best_L_nodes.has_unexpanded_node()) // 如果有邻居没有被扩展
    {
        auto nbr = best_L_nodes.closest_unexpanded(); // 返回该邻居
        auto n = nbr.id;

        // Add node to expanded nodes to create pool for prune later
        if (!search_invocation)
        {
            if (!use_filter)
            {
                expanded_nodes.emplace_back(nbr);
            }
            else
            { // in filter based indexing, the same point might invoke
                // multiple iterate_to_fixed_points, so need to be careful
                // not to add the same item to pool multiple times.
                if (std::find(expanded_nodes.begin(), expanded_nodes.end(), nbr) == expanded_nodes.end())
                {
                    expanded_nodes.emplace_back(nbr);
                }
            }
        }

        // Find which of the nodes in des have not been visited before
        id_scratch.clear();
        dist_scratch.clear();
        {
            if (_dynamic_index)
                _locks[n].lock();
            // std::cout << "neighbor size: " << _final_graph[n].size() << std::endl;
            for (auto id : _final_graph[n]) // 获取最近邻的邻居
            {
                assert(id < _max_points + _num_frozen_pts);

                if (use_filter)
                {
                    // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
                    // 判断p*（p的邻居的邻居）和query 节点的有没有共同的filter label
                    // if (!detect_common_filters(id, search_invocation, filter_label)) {
                    //     continue;
                    // }
                }

                if (is_not_visited(id))
                {
                    id_scratch.push_back(id);
                }
            }

            if (_dynamic_index)
                _locks[n].unlock();
        }

        // Mark nodes visited
        for (auto id : id_scratch)
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
        }

        // Compute distances to unvisited nodes in the expansion
        if (_pq_dist)
        {
            assert(dist_scratch.capacity() >= id_scratch.size());
            compute_dists(id_scratch, dist_scratch);
        }
        else
        {
            assert(dist_scratch.size() == 0);
            for (size_t m = 0; m < id_scratch.size(); ++m)
            {
                uint32_t id = id_scratch[m];

                if (m + 1 < id_scratch.size())
                {
                    auto nextn = id_scratch[m + 1];
                    _data_store->prefetch_vector(nextn);
                }

                dist_scratch.push_back(_data_store->get_distance(aligned_query, id));
            }
        }
        cmps += (uint32_t)id_scratch.size(); // 扩展数量

        // Insert <id, dist> pairs into the pool of candidates
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
        }
    }
    return std::make_pair(hops, cmps);
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::iterate_to_fixed_point(const T *query, const uint32_t& Lsize, NeighborPriorityQueue& final_filtered_nodes, 
                                                                             const int32_t& label1, const int32_t& label2, const uint32_t& threshold_2)
{ 
    _data_store->prefetch_vector(_start);
    _mm_prefetch((char *)(_label_to_medoid_id.data() + label1), _MM_HINT_T0);
    NeighborPriorityQueue best_L_nodes(Lsize);

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    VisitedList *el = visited_list_pool_->getFreeVisitedList();
    vl_type *expanded_array = el->mass;
    vl_type expanded_array_tag = el->curV;

    // start node
    uint32_t id = _start;
    visited_array[id] = visited_array_tag;
    uint32_t cmps = 1;
    best_L_nodes.insert(Neighbor(id, _data_store->get_distance(query, id)));

    // start from filters
    id = _label_to_medoid_id[label1];
    if (visited_array[id] != visited_array_tag) {
        visited_array[id] = visited_array_tag;
        cmps += 1;
        best_L_nodes.insert(Neighbor(id, _data_store->get_distance(query, id)));
    }
    if (label2 != -1) {
        id = _label_to_medoid_id[label2];
        if (visited_array[id] != visited_array_tag) {
            visited_array[id] = visited_array_tag;
            cmps += 1;
            best_L_nodes.insert(Neighbor(id, _data_store->get_distance(query, id)));
        }
    }

    while (best_L_nodes.has_unexpanded_node()) // 如果有邻居没有被扩展
    {
        auto cur = best_L_nodes.closest_unexpanded(); // 返回该邻居
        auto cur_id = cur.id;

        for (auto i = _csr_offset[cur_id]; i < _csr_offset[cur_id + 1]; i++) {
            auto& id = _csr_data[i];
            if (i+1 < _csr_offset[cur_id + 1]) {
                _mm_prefetch((char *)(visited_array + _csr_data[i+1]), _MM_HINT_T0);
                _data_store->prefetch_vector(_csr_data[i+1]);
            }
            if (visited_array[id] != visited_array_tag) {
                visited_array[id] = visited_array_tag;
                cmps += 1;
                best_L_nodes.insert(Neighbor(id, _data_store->get_distance(query, id)));
            }
        }
    }

    auto cmp = [](const Neighbor &a, const Neighbor &b) {
        return a.distance > b.distance; // Compare based on distance (smallest first)
    };
    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp)> pq(cmp);
    for (size_t i=0; i<best_L_nodes.size(); ++i) {
        if (i+1 < best_L_nodes.size())
            _mm_prefetch((char *)(expanded_array + best_L_nodes[i+1].id), _MM_HINT_T0);
        pq.push(best_L_nodes[i]);
        expanded_array[best_L_nodes[i].id] = expanded_array_tag;
    }

    int64_t filter_hash = _label_to_hash[label1];
    if (label2 != -1)
        filter_hash |= _label_to_hash[label2];

    uint32_t cnt = 0;
    while (final_filtered_nodes.size() < final_filtered_nodes.capacity() && pq.empty()==false && cnt < threshold_2)
    {
        auto cur = pq.top(); 
        pq.pop();
        auto cur_id = cur.id;

        if ((~(_pts_to_labels_hash_and_offset[2*cur_id]) & filter_hash) == 0) {
            if (contain_required_filters(cur_id, label1, label2)) {
                final_filtered_nodes.insert(cur);
            }
        }

        for (auto i = _csr_offset[cur_id]; i < _csr_offset[cur_id + 1]; i++) {
            auto& id = _csr_data[i];
            if (i+1 < _csr_offset[cur_id + 1]) {
                _mm_prefetch((char *)(expanded_array + _csr_data[i+1]), _MM_HINT_T0);
                _data_store->prefetch_vector(_csr_data[i+1]);
            }
            if (expanded_array[id] != expanded_array_tag) {
                expanded_array[id] = expanded_array_tag;
                cnt += 1;
                cmps += 1;
                pq.push(Neighbor(id, _data_store->get_distance(query, id)));
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);
    visited_list_pool_->releaseVisitedList(el);
    return std::make_pair(0, cmps);
}

// 为当前节点寻找邻居并利用算法2裁剪，结果存在pruned_list
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::search_for_point_and_prune(int location, uint32_t Lindex,
                                                        std::vector<uint32_t> &pruned_list,
                                                        InMemQueryScratch<T> *scratch, bool use_filter,
                                                        uint32_t filteredLindex)
{
    const std::vector<uint32_t> init_ids = get_init_ids(); // 第一个节点是起点，其余是图中的所有点（除了起点）
    const std::vector<LabelT> unused_filter_label;

    if (!use_filter)
    {
        _data_store->get_vector(location, scratch->aligned_query()); // 把location节点的vector放入aligned_query
        iterate_to_fixed_point(scratch->aligned_query(), Lindex, init_ids, scratch, false, unused_filter_label, false);
    }
    else
    {

    }

    auto &pool = scratch->pool();

    for (uint32_t i = 0; i < pool.size(); i++) // 从pool中删除location节点
    {
        if (pool[i].id == (uint32_t)location)
        {
            pool.erase(pool.begin() + i);
            i--;
        }
    }

    if (pruned_list.size() > 0)
    {
        throw diskann::ANNException("ERROR: non-empty pruned_list passed", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    prune_neighbors(location, pool, pruned_list, scratch); // 利用算法2裁剪pool中的邻居，保存到pruned_list

    // assert(!pruned_list.empty());
    assert(_final_graph.size() == _max_points + _num_frozen_pts);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha,
                                          const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result,
                                          InMemQueryScratch<T> *scratch,
                                          const tsl::robin_set<uint32_t> *const delete_set_ptr)
{
    if (pool.size() == 0)
        return;

    // Truncate pool at maxc and initialize scratch spaces
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(result.size() == 0);
    if (pool.size() > maxc)
        pool.resize(maxc);
    std::vector<float> &occlude_factor = scratch->occlude_factor();
    // occlude_list can be called with the same scratch more than once by
    // search_for_point_and_add_link through inter_insert.
    occlude_factor.clear();
    // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree)
    {
        // used for MIPS, where we store a value of eps in cur_alpha to
        // denote pruned out entries which we can skip in later rounds.
        float eps = cur_alpha + 0.01f;

        for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter)
        {
            if (occlude_factor[iter - pool.begin()] > cur_alpha)
            {
                continue;
            }
            // Set the entry to float::max so that is not considered again
            // p∗
            occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
            // Add the entry to the result if its not been deleted, and doesn't
            // add a self loop
            if (delete_set_ptr == nullptr || delete_set_ptr->find(iter->id) == delete_set_ptr->end())
            {
                if (iter->id != location) // 避免自循环
                {
                    result.push_back(iter->id);
                }
            }
            // α · d(p∗, p′) ≤ d(p, p′)
            // Update occlude factor for points from iter+1 to pool.end()
            for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++)
            {
                auto t = iter2 - pool.begin();
                if (occlude_factor[t] > alpha)
                    continue;

                bool prune_allowed = true;
                if (_filtered_index)
                {

                }
                if (!prune_allowed)
                    continue;
                // p′
                float djk = _data_store->get_distance(iter2->id, iter->id); // d(p∗, p′)
                if (_dist_metric == diskann::Metric::L2 || _dist_metric == diskann::Metric::COSINE)
                {
                    occlude_factor[t] =
                        (djk == 0) ? std::numeric_limits<float>::max()
                                   : std::max(occlude_factor[t],
                                              iter2->distance /
                                                  djk); // d(p, p′)/d(p∗, p′),如果比值大于等于α，则不把p'加入result
                }
                else if (_dist_metric == diskann::Metric::INNER_PRODUCT)
                {
                    // Improvization for flipping max and min dist for MIPS
                    float x = -iter2->distance;
                    float y = -djk;
                    if (y > cur_alpha * x)
                    {
                        occlude_factor[t] = std::max(occlude_factor[t], eps);
                    }
                }
            }
        }
        cur_alpha *= 1.2f;
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool,
                                             std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch)
{
    prune_neighbors(location, pool, _indexingRange, _indexingMaxC, _indexingAlpha, pruned_list, scratch);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                                             const uint32_t max_candidate_size, const float alpha,
                                             std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch)
{
    if (pool.size() == 0)
    {
        // if the pool is empty, behave like a noop
        pruned_list.clear();
        return;
    }

    _max_observed_degree = (std::max)(_max_observed_degree, range);

    // If using _pq_build, over-write the PQ distances with actual distances
    if (_pq_dist)
    {
        for (auto &ngh : pool)
            ngh.distance = _data_store->get_distance(ngh.id, location);
    }

    // sort the pool based on distance to query and prune it with occlude_list
    std::sort(pool.begin(), pool.end());
    pruned_list.clear();
    pruned_list.reserve(range);

    occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, scratch); // 调用算法2裁剪pool中的邻居
    assert(pruned_list.size() <= range);

    if (_saturate_graph && alpha > 1)
    {
        for (const auto &node : pool)
        {
            if (pruned_list.size() >= range)
                break;
            if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) == pruned_list.end()) &&
                node.id != location)
                pruned_list.push_back(node.id);
        }
    }
}

// 将当前节点和它的修剪后的邻居列表插入到索引中
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                                          InMemQueryScratch<T> *scratch)
{
    const auto &src_pool = pruned_list;

    // assert(!src_pool.empty());

    for (auto des : src_pool)
    {
        // des.loc is the loc of the neighbors of n
        assert(des < _max_points + _num_frozen_pts);
        // des_pool contains the neighbors of the neighbors of n
        std::vector<uint32_t> copy_of_neighbors;
        bool prune_needed = false;
        {
            LockGuard guard(_locks[des]);
            auto &des_pool = _final_graph[des];
            if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) // 如果n的邻居已经和n相连
            {
                if (des_pool.size() <
                    (uint64_t)(GRAPH_SLACK_FACTOR *
                               range)) // 如果n和邻居连接后，该邻居的度没用超过阈值，那么就让其与n相连，无需裁剪
                {
                    des_pool.emplace_back(n);
                    prune_needed = false;
                }
                else // 裁剪该邻居
                {
                    copy_of_neighbors.reserve(des_pool.size() + 1);
                    copy_of_neighbors = des_pool;
                    copy_of_neighbors.push_back(n);
                    prune_needed = true;
                }
            }
        } // des lock is released by this point

        if (prune_needed)
        {
            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);

            size_t reserveSize = (size_t)(std::ceil(1.05 * GRAPH_SLACK_FACTOR * range));
            dummy_visited.reserve(reserveSize);
            dummy_pool.reserve(reserveSize);

            for (auto cur_nbr : copy_of_neighbors)
            {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) // 防止重复插入
                {
                    float dist = _data_store->get_distance(des, cur_nbr);
                    dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                    dummy_visited.insert(cur_nbr);
                }
            }
            std::vector<uint32_t> new_out_neighbors;
            prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
            {
                LockGuard guard(_locks[des]);

                _final_graph[des] = new_out_neighbors;
            }
        }
    }
}

// 将当前节点和它的修剪后的邻居列表插入到索引中
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch)
{
    inter_insert(n, pruned_list, _indexingRange, scratch);
}

// 遍历每个节点，构建索引
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::link(const IndexWriteParameters &parameters)
{
    uint32_t num_threads = parameters.num_threads;
    // openMP不同于pthread的地方是，它是根植于编译器的（也要包含头文件omp.h），而不是在各系统平台是做文章。它貌似更偏向于将原来串行化的程序，通过加入一些适当的编译器指令（compiler
    // directive）变成并行执行，从而提高代码运行的速率。
    if (num_threads != 0)
        omp_set_num_threads(num_threads);

    _saturate_graph = parameters.saturate_graph;

    _indexingQueueSize = parameters.search_list_size;
    _filterIndexingQueueSize = parameters.filter_list_size;
    _indexingRange = parameters.max_degree;
    _indexingMaxC = parameters.max_occlusion_size;
    _indexingAlpha = parameters.alpha;

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<uint32_t> visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<uint32_t> visited;
    visit_order.reserve(_nd + _num_frozen_pts);
    for (uint32_t i = 0; i < (uint32_t)_nd; i++)
    {
        visit_order.emplace_back(i);
    }

    // If there are any frozen points, add them all.
    for (uint32_t frozen = (uint32_t)_max_points; frozen < _max_points + _num_frozen_pts; frozen++)
    {
        visit_order.emplace_back(frozen);
    }

    // if there are frozen points, the first such one is set to be the _start
    if (_num_frozen_pts > 0)
        _start = (uint32_t)_max_points;
    else
        _start = calculate_entry_point();

    for (size_t p = 0; p < _nd; p++)
    {
        _final_graph[p].reserve((size_t)(std::ceil(_indexingRange * GRAPH_SLACK_FACTOR * 1.05))); // 分配空间
    }

    diskann::Timer link_timer;

#pragma omp parallel for schedule(dynamic, 2048) // 將for循環的每次迭代分成小的任務，每個任務處理2048次迭代
    for (int64_t node_ctr = 0; node_ctr < (int64_t)(visit_order.size()); node_ctr++)
    {
        auto node = visit_order[node_ctr];

        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch); // 会在销毁的时候把scratch push回去
        auto scratch = manager.scratch_space();

        std::vector<uint32_t> pruned_list; // 存储对当前节点进行搜索并修剪后的结果
        if (_filtered_index)
        {
            search_for_point_and_prune(node, _indexingQueueSize, pruned_list, scratch, _filtered_index,
                                       _filterIndexingQueueSize);
        }
        else
        {
            search_for_point_and_prune(node, _indexingQueueSize, pruned_list, scratch);
            // search_for_point_and_prune(node, _indexingQueueSize, pruned_list, false, scratch);
        }
        {
            LockGuard guard(_locks[node]);
            _final_graph[node].reserve((size_t)(_indexingRange * GRAPH_SLACK_FACTOR * 1.05));
            _final_graph[node] = pruned_list; // 更新节点的邻居
            assert(_final_graph[node].size() <= _indexingRange);
        }

        inter_insert(node, pruned_list, scratch);
        if (node_ctr % 100000 == 0)
        {
            diskann::cout << "\r" << (100.0 * node_ctr) / (visit_order.size()) << "% of index build completed."
                          << std::flush;
        }
    }

    if (_nd > 0)
    {
        diskann::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 2048)
    for (int64_t node_ctr = 0; node_ctr < (int64_t)(visit_order.size()); node_ctr++) // 把节点的度限制在R内
    {
        auto node = visit_order[node_ctr];
        if (_final_graph[node].size() > _indexingRange)
        {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();

            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);
            std::vector<uint32_t> new_out_neighbors;

            for (auto cur_nbr : _final_graph[node])
            {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node)
                {
                    float dist = _data_store->get_distance(node, cur_nbr);
                    dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                    dummy_visited.insert(cur_nbr);
                }
            }
            prune_neighbors(node, dummy_pool, new_out_neighbors, scratch);

            _final_graph[node].clear();
            for (auto id : new_out_neighbors)
                _final_graph[node].emplace_back(id);
        }
    }
    if (_nd > 0)
    {
        diskann::cout << "done. Link time: " << ((double)link_timer.elapsed() / (double)1000000) << "s" << std::endl;
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::prune_all_neighbors(const uint32_t max_degree, const uint32_t max_occlusion_size,
                                                 const float alpha)
{
    const uint32_t range = max_degree;
    const uint32_t maxc = max_occlusion_size;

    _filtered_index = true;

    diskann::Timer timer;
#pragma omp parallel for
    for (int64_t node = 0; node < (int64_t)(_max_points + _num_frozen_pts); node++)
    {
        if ((size_t)node < _nd || (size_t)node >= _max_points)
        {
            if (_final_graph[node].size() > range)
            {
                tsl::robin_set<uint32_t> dummy_visited(0);
                std::vector<Neighbor> dummy_pool(0); // 保存邻居
                std::vector<uint32_t> new_out_neighbors;

                ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
                auto scratch = manager.scratch_space();

                for (auto cur_nbr : _final_graph[node])
                {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node)
                    {
                        float dist = _data_store->get_distance((location_t)node, (location_t)cur_nbr);
                        dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                        dummy_visited.insert(cur_nbr);
                    }
                }

                prune_neighbors((uint32_t)node, dummy_pool, range, maxc, alpha, new_out_neighbors, scratch);
                _final_graph[node].clear();
                for (auto id : new_out_neighbors)
                    _final_graph[node].emplace_back(id);
            }
        }
    }

    diskann::cout << "Prune time : " << timer.elapsed() / 1000 << "ms" << std::endl;
    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _max_points + _num_frozen_pts; i++)
    {
        if (i < _nd || i >= _max_points)
        {
            const std::vector<uint32_t> &pool = _final_graph[i];
            max = (std::max)(max, pool.size());
            min = (std::min)(min, pool.size());
            total += pool.size();
            if (pool.size() < 2)
                cnt++;
        }
    }
    if (min > max)
        min = max;
    if (_nd > 0)
    {
        diskann::cout << "Index built with degree: max:" << max
                      << "  avg:" << (float)total / (float)(_nd + _num_frozen_pts) << "  min:" << min
                      << "  count(deg<2):" << cnt << std::endl;
    }
}

// REFACTOR
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_start_points(const T *data, size_t data_count)
{
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    if (_nd > 0)
        throw ANNException("Can not set starting point for a non-empty index", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (data_count != _num_frozen_pts * _dim)
        throw ANNException("Invalid number of points", -1, __FUNCSIG__, __FILE__, __LINE__);

    //     memcpy(_data + _aligned_dim * _max_points, data, _aligned_dim *
    //     sizeof(T) * _num_frozen_pts);
    for (location_t i = 0; i < _num_frozen_pts; i++)
    {
        _data_store->set_vector((location_t)(i + _max_points), data + i * _dim);
    }
    _has_built = true;
    diskann::cout << "Index start points set: #" << _num_frozen_pts << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_set_start_points_at_random(DataType radius, uint32_t random_seed)
{
    try
    {
        T radius_to_use = std::any_cast<T>(radius);
        this->set_start_points_at_random(radius_to_use, random_seed);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException(
            "Error: bad any cast while performing _set_start_points_at_random() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_start_points_at_random(T radius, uint32_t random_seed)
{
    std::mt19937 gen{random_seed};
    std::normal_distribution<> d{0.0, 1.0};

    std::vector<T> points_data;
    points_data.reserve(_dim * _num_frozen_pts);
    std::vector<double> real_vec(_dim);

    for (size_t frozen_point = 0; frozen_point < _num_frozen_pts; frozen_point++)
    {
        double norm_sq = 0.0;
        for (size_t i = 0; i < _dim; ++i)
        {
            auto r = d(gen);
            real_vec[i] = r;
            norm_sq += r * r;
        }

        const double norm = std::sqrt(norm_sq);
        for (auto iter : real_vec)
            points_data.push_back(static_cast<T>(iter * radius / norm));
    }

    set_start_points(points_data.data(), points_data.size());
}

// 初始化候选集等空间
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build_with_data_populated(const IndexWriteParameters &parameters,
                                                       const std::vector<TagT> &tags)
{
    diskann::cout << "Starting index build with " << _nd << " points... " << std::endl;

    if (_nd < 1)
        throw ANNException("Error: Trying to build an index with 0 points", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (_enable_tags && tags.size() != _nd)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _nd << " points from file,"
               << "but tags vector is of size " << tags.size() << "." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_enable_tags)
    {
        for (size_t i = 0; i < tags.size(); ++i)
        {
            _tag_to_location[tags[i]] = (uint32_t)i;
            _location_to_tag.set(static_cast<uint32_t>(i), tags[i]);
        }
    }

    uint32_t index_R = parameters.max_degree;
    uint32_t num_threads_index = parameters.num_threads;
    uint32_t index_L = parameters.search_list_size;
    uint32_t maxc = parameters.max_occlusion_size;

    if (_query_scratch.size() == 0)
    {
        initialize_query_scratch(5 + num_threads_index, index_L, index_L, index_R, maxc,
                                 _data_store->get_aligned_dim());
    }

    generate_frozen_point();
    link(parameters);

    size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++)
    {
        auto &pool = _final_graph[i];
        max = std::max(max, pool.size());
        min = std::min(min, pool.size());
        total += pool.size();
        if (pool.size() < 2)
            cnt++;
    }
    diskann::cout << "Index built with degree: max:" << max << "  avg:" << (float)total / (float)(_nd + _num_frozen_pts)
                  << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

    _max_observed_degree = std::max((uint32_t)max, _max_observed_degree);
    _has_built = true;
}
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_build(const DataType &data, const size_t num_points_to_load,
                                    const IndexWriteParameters &parameters, TagVector &tags)
{
    try
    {
        this->build(std::any_cast<const T *>(data), num_points_to_load, parameters,
                    tags.get<const std::vector<TagT>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast in while building index. " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error" + std::string(e.what()), -1);
    }
}
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const T *data, const size_t num_points_to_load,
                                   const IndexWriteParameters &parameters, const std::vector<TagT> &tags)
{
    if (num_points_to_load == 0)
    {
        throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_pq_dist)
    {
        throw ANNException("ERROR: DO not use this build interface with PQ distance", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    {
        // 有timed字面值的锁对应没有timed字面值的锁多了两种操作（1. 可以指定请求锁等待的超时时间；2.
        // 可以指定请求锁一直到某一个时刻）
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock); // 读写锁
        _nd = num_points_to_load;

        _data_store->populate_data(data, (location_t)num_points_to_load);

        // REFACTOR
        // memcpy((char *)_data, (char *)data, _aligned_dim * _nd * sizeof(T));
        // if (_normalize_vecs)
        //{
        //     for (size_t i = 0; i < num_points_to_load; i++)
        //     {
        //         normalize(_data + _aligned_dim * i, _aligned_dim);
        //     }
        // }
    }

    build_with_data_populated(parameters, tags);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const char *filename, const size_t num_points_to_load,
                                   const IndexWriteParameters &parameters, const std::vector<TagT> &tags)
{
    // idealy this should call build_filtered_index based on params passed

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    // error checks
    if (num_points_to_load == 0)
        throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: Data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr)
    {
        throw diskann::ANNException("Can not build with an empty file", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::get_bin_metadata(filename, file_num_points, file_dim);
    if (file_num_points > _max_points)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has " << file_num_points
               << " points, but "
               << "index can support only " << _max_points << " points as specified in constructor." << std::endl;

        if (_pq_dist)
            aligned_free(_pq_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (num_points_to_load > file_num_points)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has only "
               << file_num_points << " points." << std::endl;

        if (_pq_dist)
            aligned_free(_pq_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;

        if (_pq_dist)
            aligned_free(_pq_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_pq_dist)
    {
        double p_val = std::min(1.0, ((double)MAX_PQ_TRAINING_SET_SIZE / (double)file_num_points));

        std::string suffix = _use_opq ? "_opq" : "_pq";
        suffix += std::to_string(_num_pq_chunks); // _num_pq_chunks == build_PQ_bytes 也就是分成多少个子空间
        auto pq_pivots_file = std::string(filename) + suffix + "_pivots.bin";
        auto pq_compressed_file = std::string(filename) + suffix + "_compressed.bin";
        generate_quantized_data<T>(std::string(filename), pq_pivots_file, pq_compressed_file, _dist_metric, p_val,
                                   _num_pq_chunks, _use_opq);

        copy_aligned_data_from_file<uint8_t>(pq_compressed_file.c_str(), _pq_data, file_num_points, _num_pq_chunks,
                                             _num_pq_chunks); // 从文件中加载数据到内存
#ifdef EXEC_ENV_OLS
        throw ANNException("load_pq_centroid_bin should not be called when "
                           "EXEC_ENV_OLS is defined.",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
#else
        _pq_table.load_pq_centroid_bin(pq_pivots_file.c_str(),
                                       _num_pq_chunks); // 将保存在文件中的 PQ 质心数据加载到 _pq_table 对象
#endif
    }

    _data_store->populate_data(filename, 0U); // 填充数据
    diskann::cout << "Using only first " << num_points_to_load << " from file.. " << std::endl;

    {
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        _nd = num_points_to_load;
    }
    build_with_data_populated(parameters, tags);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const char *filename, const size_t num_points_to_load,
                                   const IndexWriteParameters &parameters, const char *tag_filename)
{
    std::vector<TagT> tags;

    if (_enable_tags)
    {
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        if (tag_filename == nullptr)
        {
            throw ANNException("Tag filename is null, while _enable_tags is set", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        else
        {
            if (file_exists(tag_filename))
            {
                diskann::cout << "Loading tags from " << tag_filename << " for vamana index build" << std::endl;
                TagT *tag_data = nullptr;
                size_t npts, ndim;
                diskann::load_bin(tag_filename, tag_data, npts, ndim);
                if (npts < num_points_to_load)
                {
                    std::stringstream sstream;
                    sstream << "Loaded " << npts << " tags, insufficient to populate tags for " << num_points_to_load
                            << "  points to load";
                    throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
                }
                for (size_t i = 0; i < num_points_to_load; i++)
                {
                    tags.push_back(tag_data[i]);
                }
                delete[] tag_data;
            }
            else
            {
                throw diskann::ANNException(std::string("Tag file") + tag_filename + " does not exist", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
            }
        }
    }
    build(filename, num_points_to_load, parameters, tags);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const std::string &data_file, const size_t num_points_to_load,
                                   IndexBuildParams &build_params)
{

}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const std::string &data_file, const size_t num_points_to_load,
                                   IndexBuildParams &build_params, const std::string &label_file)
{
    std::string labels_file_to_use = build_params.save_path_prefix + "_label_formatted.txt";
    std::string mem_labels_int_map_file = build_params.save_path_prefix + "_labels_map.txt";

    size_t points_to_load = num_points_to_load == 0 ? _max_points : num_points_to_load;

    auto s = std::chrono::high_resolution_clock::now();
    this->build(data_file.c_str(), points_to_load, build_params.index_write_params);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    std::cout << "Graph Indexing time: " << diff.count() << "\n";
}

template <typename T, typename TagT, typename LabelT>
std::unordered_map<std::string, LabelT> Index<T, TagT, LabelT>::load_label_map(const std::string &labels_map_file)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::ifstream map_reader(labels_map_file);
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = (LabelT)std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename TagT, typename LabelT>
LabelT Index<T, TagT, LabelT>::get_converted_label(const std::string &raw_label)
{
    if (_label_map.find(raw_label) != _label_map.end())
    {
        return _label_map[raw_label];
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::parse_label_file_neurips23(const std::string &label_file, uint32_t &num_points, uint32_t &num_labels, std::string& rewrite_filename)
{
    // Format of Label txt file: filters with comma separators

    int64_t nrow;
    int64_t ncol;
    int64_t nnz; // 稀疏矩阵中的非零元素的总数
    std::ifstream infile(label_file, std::ios::binary);
    infile.read((char *)&nrow, sizeof(int64_t));
    infile.read((char *)&ncol, sizeof(int64_t));
    infile.read((char *)&nnz, sizeof(int64_t));

    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file 10") + label_file, -1);
    }

    num_points = (uint32_t)nrow;
    _pts_to_labels_hash_and_offset = new int64_t[2*nrow + 2];
    int64_t *pts_to_labels_offset = new int64_t[nrow + 1];
    infile.read((char *)pts_to_labels_offset, (nrow + 1) * sizeof(int64_t));

    _pts_to_labels_data = new uint32_t[nnz];
    infile.read((char *)_pts_to_labels_data, nnz * sizeof(uint32_t));

    for (auto i = 0; i < num_points; ++i) {
        _pts_to_labels_hash_and_offset[2*i + 1] = pts_to_labels_offset[i];
        std::sort(_pts_to_labels_data + pts_to_labels_offset[i], _pts_to_labels_data + pts_to_labels_offset[i + 1]);
    }
    _pts_to_labels_hash_and_offset[2*num_points + 1] = nnz;

    num_labels = 0;
    for (auto i = 0; i < nnz; ++i)
        num_labels = (num_labels > _pts_to_labels_data[i]) ? num_labels : (_pts_to_labels_data[i]+1);
    
    std::vector<std::vector<uint32_t>> label_to_pts(num_labels);
    for (auto i = 0; i < num_points; ++i) {
        for (auto j = pts_to_labels_offset[i]; j < pts_to_labels_offset[i + 1]; ++j) {
            label_to_pts[_pts_to_labels_data[j]].push_back(i);
        }
    }

    _label_to_pts_offset = new int64_t[num_labels + 1];
    _label_to_pts_data = new uint32_t[nnz];
    _label_to_pts_offset[0] = 0;
    for (auto i = 0; i < num_labels; ++i) {
        _label_to_pts_offset[i + 1] = _label_to_pts_offset[i] + label_to_pts[i].size();
        for (auto j = _label_to_pts_offset[i]; j < _label_to_pts_offset[i + 1]; ++j) {
            _label_to_pts_data[j] = label_to_pts[i][j - _label_to_pts_offset[i]];
        }
    }
    
    if (rewrite_filename != "")
    {
        std::cout << "rewriting label file .." << std::endl;
        int64_t *new_indptr = new int64_t[nrow + 1];
        int32_t *new_indices = new int32_t[nnz];

        new_indptr[0] = 0;
        for (int64_t i = 0; i < nrow; ++i)
        {
            auto start = pts_to_labels_offset[_new_to_raw_id[i]];
            auto end = pts_to_labels_offset[_new_to_raw_id[i] + 1];
            new_indptr[i + 1] = new_indptr[i] + end - start;
            for (int64_t j = new_indptr[i]; j < new_indptr[i + 1]; ++j) {
                new_indices[j] = _pts_to_labels_data[start + j - new_indptr[i]];
            }
        }

        std::ofstream rewrite_file(rewrite_filename, std::ios::binary);
        rewrite_file.write((char *)&nrow, sizeof(int64_t));
        rewrite_file.write((char *)&ncol, sizeof(int64_t));
        rewrite_file.write((char *)&nnz, sizeof(int64_t));
        rewrite_file.write((char *)new_indptr, (nrow + 1) * sizeof(int64_t));
        rewrite_file.write((char *)new_indices, nnz * sizeof(int32_t));
        rewrite_file.close();
        delete[] new_indptr;
        delete[] new_indices;
    }

    delete[] pts_to_labels_offset;
    diskann::cout << "Identified " << num_labels << " distinct label(s)" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::set_universal_label(const LabelT &label)
{
    _use_universal_label = true;
    _universal_label = label;
}

// 找中心点，存在_label_to_medoid_id{label, id}，然后构建索引
template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build_filtered_index(const char *filename, const std::string &label_file,
                                                  const size_t num_points_to_load, IndexWriteParameters &parameters,
                                                  const std::vector<TagT> &tags)
{
    
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::_search(const DataType &query, const size_t& K, const uint32_t& L, 
                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                            const int32_t& label1, const int32_t& label2, std::any &indices, float *distances)
{
    std::vector<LabelT> raw_labels;
    raw_labels.push_back(label1);
    if (label2 != -1) {
        raw_labels.push_back(label2);
    }

    try
    {
        // std::cerr << "label1: " << label1 << ", " << "label2: " << label2 << "\n";
        auto typed_query = std::any_cast<const T *>(query);
        if (typeid(uint32_t *) == indices.type())
        {
            auto u32_ptr = std::any_cast<uint32_t *>(indices);
            // return this->search(typed_query, K, L, threshold_1, threshold_2, label1, label2, u32_ptr, distances);
            return this->search_with_filters(typed_query, raw_labels, K,
                                            L, u32_ptr, distances, -1, threshold_1, threshold_2);
        }
        else if (typeid(uint64_t *) == indices.type())
        {
            auto u64_ptr = std::any_cast<uint64_t *>(indices);
            // return this->search(typed_query, K, L, threshold_1, threshold_2, label1, label2, u64_ptr, distances);
            return this->search_with_filters(typed_query, raw_labels, K,
                                            L, u64_ptr, distances, -1, threshold_1, threshold_2);
        }
        else
        {
            throw ANNException("Error: indices type can only be uint64_t or uint32_t.", -1);
        }
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while searching. " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search(const T *query, const size_t& K, const uint32_t& L, 
                                                             const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                             const int32_t& label1, const int32_t& label2, IdType *indices, float *distances)
{
    auto& start1 = _label_to_pts_offset[label1];
    auto& end1 = _label_to_pts_offset[label1 + 1];
    int64_t start2, end2;
    auto min_cmps = end1-start1;
    auto best_label_start = start1, best_label_end = end1;
    LabelT the_other_label;
    
    if (label2 != -1) {
        start2 = _label_to_pts_offset[label2];
        end2 = _label_to_pts_offset[label2 + 1];
        min_cmps = (end2 - start2 < min_cmps) ? end2 - start2 : min_cmps;
        best_label_start = (min_cmps == end1 - start1) ? start1 : start2;
        best_label_end = (min_cmps == end1 - start1) ? end1 : end2;
        the_other_label = (min_cmps == end1 - start1) ? label2 : label1;
    }

    NeighborPriorityQueue final_filtered_nodes(K);
    std::pair<uint32_t, uint32_t> retval;
    if (min_cmps < threshold_1) { 
        if (label2 == -1) {
            for (auto i=best_label_start; i<best_label_end; ++i) {
                auto &id = _label_to_pts_data[i];
                if (i + 1 < best_label_end) {
                    auto nextn = _label_to_pts_data[i+1];
                    _data_store->prefetch_vector(nextn);
                }
                float distance = _data_store->get_distance(query, id);
                final_filtered_nodes.insert(Neighbor(id, distance));
            }
        } else {
            int64_t filter_hash = _label_to_hash[the_other_label];                
            for (auto i=best_label_start; i<best_label_end; ++i) {
                auto &id = _label_to_pts_data[i];
                if (i + 1 < best_label_end) {
                    auto next_id = _label_to_pts_data[i+1];
                    _mm_prefetch((char *)(_pts_to_labels_hash_and_offset + 2 * next_id), _MM_HINT_T0);
                    _data_store->prefetch_vector(next_id);
                }

                if (~(_pts_to_labels_hash_and_offset[2*id]) & filter_hash) {
                    continue;
                }
                if (contain_required_filters(id, the_other_label, -1)==false)
                    continue;
                float distance = _data_store->get_distance(query, id);
                final_filtered_nodes.insert(Neighbor(id, distance));
            }
        }
        retval = {0, min_cmps};
    } else {
        retval = iterate_to_fixed_point(query, L, final_filtered_nodes, label1, label2, threshold_2);
    }

    size_t pos = 0;
    for (size_t i = 0; i < final_filtered_nodes.size(); ++i)
    {
        if (final_filtered_nodes[i].id < _max_points)
        {   
            // use reorder
            indices[pos] = (IdType)(_new_to_raw_id[final_filtered_nodes[i].id]);
            // not use reorder
            // indices[pos] = (IdType)(final_filtered_nodes[i].id);
            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                // DLVS expects negative distances
                distances[pos] = final_filtered_nodes[i].distance;
#else
                distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * final_filtered_nodes[i].distance
                                                                                : final_filtered_nodes[i].distance;
#endif
            }
            pos++;
        }
        if (pos == K)
            break;
    }

    return retval;
}

template <typename T, typename TagT, typename LabelT>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::_search_with_filters(const DataType &query,
                                                                           const std::string &raw_label, const size_t K,
                                                                           const uint32_t L, std::any &indices,
                                                                           float *distances, int8_t best_method,
                                                                           const uint32_t threshold_1, const uint32_t threshold_2)
{
    auto converted_label = this->get_converted_label(raw_label);
    if (typeid(uint64_t *) == indices.type())
    {
        auto ptr = std::any_cast<uint64_t *>(indices);
        return this->search_with_filters(std::any_cast<T *>(query), converted_label, K, L, ptr, distances, best_method, threshold_1, threshold_2);
    }
    else if (typeid(uint32_t *) == indices.type())
    {
        auto ptr = std::any_cast<uint32_t *>(indices);
        return this->search_with_filters(std::any_cast<T *>(query), converted_label, K, L, ptr, distances, best_method, threshold_1, threshold_2);
    }
    else
    {
        throw ANNException("Error: Id type can only be uint64_t or uint32_t.", -1);
    }
}

template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_filters(const T *query, const LabelT &filter_label,
                                                                          const size_t K, const uint32_t L,
                                                                          IdType *indices, float *distances, int8_t best_method,
                                                                          const uint32_t threshold_1, const uint32_t threshold_2)
{
    return {0, 0};
}

template <typename T, typename TagT, typename LabelT>
template <typename IdType>
std::pair<uint32_t, uint32_t> Index<T, TagT, LabelT>::search_with_filters(const T *query,
                                                                          const std::vector<LabelT> &filter_labels,
                                                                          const size_t K, const uint32_t L,
                                                                          IdType *indices, float *distances, int8_t best_method,
                                                                          const uint32_t threshold_1, const uint32_t threshold_2)
{
    return {0, 0};
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::_search_with_tags(const DataType &query, const uint64_t K, const uint32_t L,
                                                 const TagType &tags, float *distances, DataVector &res_vectors)
{
    try
    {
        return this->search_with_tags(std::any_cast<const T *>(query), K, L, std::any_cast<TagT *>(tags), distances,
                                      res_vectors.get<std::vector<T *>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing _search_with_tags() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::search_with_tags(const T *query, const uint64_t K, const uint32_t L, TagT *tags,
                                                float *distances, std::vector<T *> &res_vectors)
{
    if (K > (uint64_t)L)
    {
        throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    if (L > scratch->get_L())
    {
        diskann::cout << "Attempting to expand query scratch_space. Was created "
                      << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
        scratch->resize_for_new_L(L);
        diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
    }

    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);

    const std::vector<uint32_t> init_ids = get_init_ids();
    const std::vector<LabelT> unused_filter_label;

    _distance->preprocess_query(query, _data_store->get_dims(), scratch->aligned_query());
    iterate_to_fixed_point(scratch->aligned_query(), L, init_ids, scratch, false, unused_filter_label, true);

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    assert(best_L_nodes.size() <= L);

    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        auto node = best_L_nodes[i];

        TagT tag;
        if (_location_to_tag.try_get(node.id, tag))
        {
            tags[pos] = tag;

            if (res_vectors.size() > 0)
            {
                _data_store->get_vector(node.id, res_vectors[pos]);
            }

            if (distances != nullptr)
            {
#ifdef EXEC_ENV_OLS
                distances[pos] = node.distance; // DLVS expects negative distances
#else
                distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * node.distance : node.distance;
#endif
            }
            pos++;
            // If res_vectors.size() < k, clip at the value.
            if (pos == K || pos == res_vectors.size())
                break;
        }
    }

    return pos;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::get_num_points()
{
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    return _nd;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::get_max_points()
{
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    return _max_points;
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::generate_frozen_point()
{
    if (_num_frozen_pts == 0)
        return;

    if (_num_frozen_pts > 1)
    {
        throw ANNException("More than one frozen point not supported in generate_frozen_point", -1, __FUNCSIG__,
                           __FILE__, __LINE__);
    }

    if (_nd == 0)
    {
        throw ANNException("ERROR: Can not pick a frozen point since nd=0", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    size_t res = calculate_entry_point();

    if (_pq_dist)
    {
        // copy the PQ data corresponding to the point returned by
        // calculate_entry_point
        memcpy(_pq_data + _max_points * _num_pq_chunks, _pq_data + res * _num_pq_chunks,
               _num_pq_chunks * DIV_ROUND_UP(NUM_PQ_BITS, 8));
    }
    else
    {
        _data_store->copy_vectors((location_t)res, (location_t)_max_points, 1);
    }
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::enable_delete()
{
    assert(_enable_tags);

    if (!_enable_tags)
    {
        diskann::cerr << "Tags must be instantiated for deletions" << std::endl;
        return -2;
    }

    if (this->_deletes_enabled)
    {
        return 0;
    }

    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    if (_data_compacted)
    {
        for (uint32_t slot = (uint32_t)_nd; slot < _max_points; ++slot)
        {
            _empty_slots.insert(slot);
        }
    }
    this->_deletes_enabled = true;
    return 0;
}

template <typename T, typename TagT, typename LabelT>
inline void Index<T, TagT, LabelT>::process_delete(const tsl::robin_set<uint32_t> &old_delete_set, size_t loc,
                                                   const uint32_t range, const uint32_t maxc, const float alpha,
                                                   InMemQueryScratch<T> *scratch)
{
    tsl::robin_set<uint32_t> &expanded_nodes_set = scratch->expanded_nodes_set();
    std::vector<Neighbor> &expanded_nghrs_vec = scratch->expanded_nodes_vec();

    // If this condition were not true, deadlock could result
    assert(old_delete_set.find((uint32_t)loc) == old_delete_set.end());

    std::vector<uint32_t> adj_list;
    {
        // Acquire and release lock[loc] before acquiring locks for neighbors
        std::unique_lock<non_recursive_mutex> adj_list_lock;
        if (_conc_consolidate)
            adj_list_lock = std::unique_lock<non_recursive_mutex>(_locks[loc]);
        adj_list = _final_graph[loc];
    }

    bool modify = false;
    for (auto ngh : adj_list)
    {
        if (old_delete_set.find(ngh) == old_delete_set.end())
        {
            expanded_nodes_set.insert(ngh);
        }
        else
        {
            modify = true;

            std::unique_lock<non_recursive_mutex> ngh_lock;
            if (_conc_consolidate)
                ngh_lock = std::unique_lock<non_recursive_mutex>(_locks[ngh]);
            for (auto j : _final_graph[ngh])
                if (j != loc && old_delete_set.find(j) == old_delete_set.end())
                    expanded_nodes_set.insert(j);
        }
    }

    if (modify)
    {
        if (expanded_nodes_set.size() <= range)
        {
            std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
            _final_graph[loc].clear();
            for (auto &ngh : expanded_nodes_set)
                _final_graph[loc].push_back(ngh);
        }
        else
        {
            // Create a pool of Neighbor candidates from the expanded_nodes_set
            expanded_nghrs_vec.reserve(expanded_nodes_set.size());
            for (auto &ngh : expanded_nodes_set)
            {
                expanded_nghrs_vec.emplace_back(ngh, _data_store->get_distance((location_t)loc, (location_t)ngh));
            }
            std::sort(expanded_nghrs_vec.begin(), expanded_nghrs_vec.end());
            std::vector<uint32_t> &occlude_list_output = scratch->occlude_list_output();
            occlude_list((uint32_t)loc, expanded_nghrs_vec, alpha, range, maxc, occlude_list_output, scratch,
                         &old_delete_set);
            std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
            _final_graph[loc] = occlude_list_output;
        }
    }
}

// Returns number of live points left after consolidation
template <typename T, typename TagT, typename LabelT>
consolidation_report Index<T, TagT, LabelT>::consolidate_deletes(const IndexWriteParameters &params)
{
    if (!_enable_tags)
        throw diskann::ANNException("Point tag array not instantiated", -1, __FUNCSIG__, __FILE__, __LINE__);

    {
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
        if (_empty_slots.size() + _nd != _max_points)
        {
            std::string err = "#empty slots + nd != max points";
            diskann::cerr << err << std::endl;
            throw ANNException(err, -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        if (_location_to_tag.size() + _delete_set->size() != _nd)
        {
            diskann::cerr << "Error: _location_to_tag.size (" << _location_to_tag.size() << ")  + _delete_set->size ("
                          << _delete_set->size() << ") != _nd(" << _nd << ") ";
            return consolidation_report(diskann::consolidation_report::status_code::INCONSISTENT_COUNT_ERROR, 0, 0, 0,
                                        0, 0, 0, 0);
        }

        if (_location_to_tag.size() != _tag_to_location.size())
        {
            throw diskann::ANNException("_location_to_tag and _tag_to_location not of same size", -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
        }
    }

    std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock, std::defer_lock);
    if (!_conc_consolidate)
        update_lock.lock();

    std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock, std::defer_lock);
    if (!cl.try_lock())
    {
        diskann::cerr << "Consildate delete function failed to acquire consolidate lock" << std::endl;
        return consolidation_report(diskann::consolidation_report::status_code::LOCK_FAIL, 0, 0, 0, 0, 0, 0, 0);
    }

    diskann::cout << "Starting consolidate_deletes... ";

    std::unique_ptr<tsl::robin_set<uint32_t>> old_delete_set(new tsl::robin_set<uint32_t>);
    {
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        std::swap(_delete_set, old_delete_set);
    }

    if (old_delete_set->find(_start) != old_delete_set->end())
    {
        throw diskann::ANNException("ERROR: start node has been deleted", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint32_t range = params.max_degree;
    const uint32_t maxc = params.max_occlusion_size;
    const float alpha = params.alpha;
    const uint32_t num_threads = params.num_threads == 0 ? omp_get_num_threads() : params.num_threads;

    uint32_t num_calls_to_process_delete = 0;
    diskann::Timer timer;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192) reduction(+ : num_calls_to_process_delete)
    for (int64_t loc = 0; loc < (int64_t)_max_points; loc++)
    {
        if (old_delete_set->find((uint32_t)loc) == old_delete_set->end() && !_empty_slots.is_in_set((uint32_t)loc))
        {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();
            process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
            num_calls_to_process_delete += 1;
        }
    }
    for (int64_t loc = _max_points; loc < (int64_t)(_max_points + _num_frozen_pts); loc++)
    {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
        num_calls_to_process_delete += 1;
    }

    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    size_t ret_nd = release_locations(*old_delete_set);
    size_t max_points = _max_points;
    size_t empty_slots_size = _empty_slots.size();

    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
    size_t delete_set_size = _delete_set->size();
    size_t old_delete_set_size = old_delete_set->size();

    if (!_conc_consolidate)
    {
        update_lock.unlock();
    }

    double duration = timer.elapsed() / 1000000.0;
    diskann::cout << " done in " << duration << " seconds." << std::endl;
    return consolidation_report(diskann::consolidation_report::status_code::SUCCESS, ret_nd, max_points,
                                empty_slots_size, old_delete_set_size, delete_set_size, num_calls_to_process_delete,
                                duration);
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::compact_frozen_point()
{
    if (_nd < _max_points && _num_frozen_pts > 0)
    {
        reposition_points((uint32_t)_max_points, (uint32_t)_nd, (uint32_t)_num_frozen_pts);
        _start = (uint32_t)_nd;
    }
}

// Should be called after acquiring _update_lock
template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::compact_data()
{
    if (!_dynamic_index)
        throw ANNException("Can not compact a non-dynamic index", -1, __FUNCSIG__, __FILE__, __LINE__);

    if (_data_compacted)
    {
        diskann::cerr << "Warning! Calling compact_data() when _data_compacted is true!" << std::endl;
        return;
    }

    if (_delete_set->size() > 0)
    {
        throw ANNException("Can not compact data when index has non-empty _delete_set of "
                           "size: " +
                               std::to_string(_delete_set->size()),
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::Timer timer;

    std::vector<uint32_t> new_location = std::vector<uint32_t>(_max_points + _num_frozen_pts, UINT32_MAX);

    uint32_t new_counter = 0;
    std::set<uint32_t> empty_locations;
    for (uint32_t old_location = 0; old_location < _max_points; old_location++)
    {
        if (_location_to_tag.contains(old_location))
        {
            new_location[old_location] = new_counter;
            new_counter++;
        }
        else
        {
            empty_locations.insert(old_location);
        }
    }
    for (uint32_t old_location = (uint32_t)_max_points; old_location < _max_points + _num_frozen_pts; old_location++)
    {
        new_location[old_location] = old_location;
    }

    // If start node is removed, throw an exception
    if (_start < _max_points && !_location_to_tag.contains(_start))
    {
        throw diskann::ANNException("ERROR: Start node deleted.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    size_t num_dangling = 0;
    for (uint32_t old = 0; old < _max_points + _num_frozen_pts; ++old)
    {
        std::vector<uint32_t> new_adj_list;

        if ((new_location[old] < _max_points) // If point continues to exist
            || (old >= _max_points && old < _max_points + _num_frozen_pts))
        {
            new_adj_list.reserve(_final_graph[old].size());
            for (auto ngh_iter : _final_graph[old])
            {
                if (empty_locations.find(ngh_iter) != empty_locations.end())
                {
                    ++num_dangling;
                    diskann::cerr << "Error in compact_data(). _final_graph[" << old << "] has neighbor " << ngh_iter
                                  << " which is a location not associated with any tag." << std::endl;
                }
                else
                {
                    new_adj_list.push_back(new_location[ngh_iter]);
                }
            }
            _final_graph[old].swap(new_adj_list);

            // Move the data and adj list to the correct position
            if (new_location[old] != old)
            {
                assert(new_location[old] < old);
                _final_graph[new_location[old]].swap(_final_graph[old]);

                _data_store->copy_vectors(old, new_location[old], 1);
            }
        }
        else
        {
            _final_graph[old].clear();
        }
    }
    diskann::cerr << "#dangling references after data compaction: " << num_dangling << std::endl;

    _tag_to_location.clear();
    for (auto pos = _location_to_tag.find_first(); pos.is_valid(); pos = _location_to_tag.find_next(pos))
    {
        const auto tag = _location_to_tag.get(pos);
        _tag_to_location[tag] = new_location[pos._key];
    }
    _location_to_tag.clear();
    for (const auto &iter : _tag_to_location)
    {
        _location_to_tag.set(iter.second, iter.first);
    }

    for (size_t old = _nd; old < _max_points; ++old)
    {
        _final_graph[old].clear();
    }
    _empty_slots.clear();
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }
    _data_compacted = true;
    diskann::cout << "Time taken for compact_data: " << timer.elapsed() / 1000000. << "s." << std::endl;
}

//
// Caller must hold unique _tag_lock and _delete_lock before calling this
//
template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::reserve_location()
{
    if (_nd >= _max_points)
    {
        return -1;
    }
    uint32_t location;
    if (_data_compacted && _empty_slots.is_empty())
    {
        // This code path is encountered when enable_delete hasn't been
        // called yet, so no points have been deleted and _empty_slots
        // hasn't been filled in. In that case, just keep assigning
        // consecutive locations.
        location = (uint32_t)_nd;
    }
    else
    {
        assert(_empty_slots.size() != 0);
        assert(_empty_slots.size() + _nd == _max_points);

        location = _empty_slots.pop_any();
        _delete_set->erase(location);
    }

    ++_nd;
    return location;
}

template <typename T, typename TagT, typename LabelT> size_t Index<T, TagT, LabelT>::release_location(int location)
{
    if (_empty_slots.is_in_set(location))
        throw ANNException("Trying to release location, but location already in empty slots", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    _empty_slots.insert(location);

    _nd--;
    return _nd;
}

template <typename T, typename TagT, typename LabelT>
size_t Index<T, TagT, LabelT>::release_locations(const tsl::robin_set<uint32_t> &locations)
{
    for (auto location : locations)
    {
        if (_empty_slots.is_in_set(location))
            throw ANNException("Trying to release location, but location "
                               "already in empty slots",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        _empty_slots.insert(location);

        _nd--;
    }

    if (_empty_slots.size() + _nd != _max_points)
        throw ANNException("#empty slots + nd != max points", -1, __FUNCSIG__, __FILE__, __LINE__);

    return _nd;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                               uint32_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    // Update pointers to the moved nodes. Note: the computation is correct even
    // when new_location_start < old_location_start given the C++ uint32_t
    // integer arithmetic rules.
    const uint32_t location_delta = new_location_start - old_location_start;

    for (uint32_t i = 0; i < _max_points + _num_frozen_pts; i++)
        for (auto &loc : _final_graph[i])
            if (loc >= old_location_start && loc < old_location_start + num_locations)
                loc += location_delta;

    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    // Move the adjacency lists. Make sure that overlapping ranges are handled
    // correctly.
    if (new_location_start < old_location_start)
    {
        // New location before the old location: copy the entries in order
        // to avoid modifying locations that are yet to be copied.
        for (uint32_t loc_offset = 0; loc_offset < num_locations; loc_offset++)
        {
            assert(_final_graph[new_location_start + loc_offset].empty());
            _final_graph[new_location_start + loc_offset].swap(_final_graph[old_location_start + loc_offset]);
        }

        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // Old location after the new location: copy from the end of the range
        // to avoid modifying locations that are yet to be copied.
        for (uint32_t loc_offset = num_locations; loc_offset > 0; loc_offset--)
        {
            assert(_final_graph[new_location_start + loc_offset - 1u].empty());
            _final_graph[new_location_start + loc_offset - 1u].swap(_final_graph[old_location_start + loc_offset - 1u]);
        }

        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }
    _data_store->move_vectors(old_location_start, new_location_start, num_locations);
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::reposition_frozen_point_to_end()
{
    if (_num_frozen_pts == 0)
        return;

    if (_nd == _max_points)
    {
        diskann::cout << "Not repositioning frozen point as it is already at the end." << std::endl;
        return;
    }

    reposition_points((uint32_t)_nd, (uint32_t)_max_points, (uint32_t)_num_frozen_pts);
    _start = (uint32_t)_max_points;
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::resize(size_t new_max_points)
{
    const size_t new_internal_points = new_max_points + _num_frozen_pts;
    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() == 0); // should not resize if there are empty slots.

    _data_store->resize((location_t)new_internal_points);
    _final_graph.resize(new_internal_points);
    _locks = std::vector<non_recursive_mutex>(new_internal_points);

    if (_num_frozen_pts != 0)
    {
        reposition_points((uint32_t)_max_points, (uint32_t)new_max_points, (uint32_t)_num_frozen_pts);
        _start = (uint32_t)new_max_points;
    }

    _max_points = new_max_points;
    _empty_slots.reserve(_max_points);
    for (auto i = _nd; i < _max_points; i++)
    {
        _empty_slots.insert((uint32_t)i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Resizing took: " << std::chrono::duration<double>(stop - start).count() << "s" << std::endl;
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::_insert_point(const DataType &point, const TagType tag)
{
    try
    {
        return this->insert_point(std::any_cast<const T *>(point), std::any_cast<const TagT>(tag));
    }
    catch (const std::bad_any_cast &anycast_e)
    {
        throw new ANNException("Error:Trying to insert invalid data type" + std::string(anycast_e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw new ANNException("Error:" + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
int Index<T, TagT, LabelT>::insert_point(const T *point, const TagT tag)
{
    assert(_has_built);
    if (tag == static_cast<TagT>(0))
    {
        throw diskann::ANNException("Do not insert point with tag 0. That is "
                                    "reserved for points hidden "
                                    "from the user.",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::shared_lock<std::shared_timed_mutex> shared_ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

    // Find a vacant location in the data array to insert the new point
    auto location = reserve_location();
    if (location == -1)
    {
#if EXPAND_IF_FULL
        dl.unlock();
        tl.unlock();
        shared_ul.unlock();

        {
            std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
            tl.lock();
            dl.lock();

            if (_nd >= _max_points)
            {
                auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
                resize(new_max_points);
            }

            dl.unlock();
            tl.unlock();
            ul.unlock();
        }

        shared_ul.lock();
        tl.lock();
        dl.lock();

        location = reserve_location();
        if (location == -1)
        {
            throw diskann::ANNException("Cannot reserve location even after "
                                        "expanding graph. Terminating.",
                                        -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#else
        return -1;
#endif
    }
    dl.unlock();

    // Insert tag and mapping to location
    if (_enable_tags)
    {
        if (_tag_to_location.find(tag) != _tag_to_location.end())
        {
            release_location(location);
            return -1;
        }

        _tag_to_location[tag] = location;
        _location_to_tag.set(location, tag);
    }
    tl.unlock();

    _data_store->set_vector(location, point);

    // Find and add appropriate graph edges
    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    auto scratch = manager.scratch_space();
    std::vector<uint32_t> pruned_list;
    if (_filtered_index)
    {
        search_for_point_and_prune(location, _indexingQueueSize, pruned_list, scratch, true, _filterIndexingQueueSize);
    }
    else
    {
        search_for_point_and_prune(location, _indexingQueueSize, pruned_list, scratch);
    }
    {
        std::shared_lock<std::shared_timed_mutex> tlock(_tag_lock, std::defer_lock);
        if (_conc_consolidate)
            tlock.lock();

        LockGuard guard(_locks[location]);
        _final_graph[location].clear();
        _final_graph[location].reserve((size_t)(_indexingRange * GRAPH_SLACK_FACTOR * 1.05));

        for (auto link : pruned_list)
        {
            if (_conc_consolidate)
                if (!_location_to_tag.contains(link))
                    continue;
            _final_graph[location].emplace_back(link);
        }
        assert(_final_graph[location].size() <= _indexingRange);

        if (_conc_consolidate)
            tlock.unlock();
    }

    inter_insert(location, pruned_list, scratch);

    return 0;
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::_lazy_delete(const TagType &tag)
{
    try
    {
        return lazy_delete(std::any_cast<const TagT>(tag));
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException(std::string("Error: ") + e.what(), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_lazy_delete(TagVector &tags, TagVector &failed_tags)
{
    try
    {
        this->lazy_delete(tags.get<const std::vector<TagT>>(), failed_tags.get<std::vector<TagT>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad any cast while performing _lazy_delete() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT> int Index<T, TagT, LabelT>::lazy_delete(const TagT &tag)
{
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    _data_compacted = false;

    if (_tag_to_location.find(tag) == _tag_to_location.end())
    {
        diskann::cerr << "Delete tag not found " << tag << std::endl;
        return -1;
    }
    assert(_tag_to_location[tag] < _max_points);

    const auto location = _tag_to_location[tag];
    _delete_set->insert(location);
    _location_to_tag.erase(location);
    _tag_to_location.erase(tag);

    return 0;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::lazy_delete(const std::vector<TagT> &tags, std::vector<TagT> &failed_tags)
{
    if (failed_tags.size() > 0)
    {
        throw ANNException("failed_tags should be passed as an empty list", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
    _data_compacted = false;

    for (auto tag : tags)
    {
        if (_tag_to_location.find(tag) == _tag_to_location.end())
        {
            failed_tags.push_back(tag);
        }
        else
        {
            const auto location = _tag_to_location[tag];
            _delete_set->insert(location);
            _location_to_tag.erase(location);
            _tag_to_location.erase(tag);
        }
    }
}

template <typename T, typename TagT, typename LabelT> bool Index<T, TagT, LabelT>::is_index_saved()
{
    return _is_saved;
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_get_active_tags(TagRobinSet &active_tags)
{
    try
    {
        this->get_active_tags(active_tags.get<tsl::robin_set<TagT>>());
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException("Error: bad_any cast while performing _get_active_tags() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error :" + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::get_active_tags(tsl::robin_set<TagT> &active_tags)
{
    active_tags.clear();
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    for (auto iter : _tag_to_location)
    {
        active_tags.insert(iter.first);
    }
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::print_status()
{
    std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
    std::shared_lock<std::shared_timed_mutex> cl(_consolidate_lock);
    std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
    std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);

    diskann::cout << "------------------- Index object: " << (uint64_t)this << " -------------------" << std::endl;
    diskann::cout << "Number of points: " << _nd << std::endl;
    diskann::cout << "Graph size: " << _final_graph.size() << std::endl;
    diskann::cout << "Location to tag size: " << _location_to_tag.size() << std::endl;
    diskann::cout << "Tag to location size: " << _tag_to_location.size() << std::endl;
    diskann::cout << "Number of empty slots: " << _empty_slots.size() << std::endl;
    diskann::cout << std::boolalpha << "Data compacted: " << this->_data_compacted << std::endl;
    diskann::cout << "---------------------------------------------------------"
                     "------------"
                  << std::endl;
}

template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::count_nodes_at_bfs_levels()
{
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

    boost::dynamic_bitset<> visited(_max_points + _num_frozen_pts);

    size_t MAX_BFS_LEVELS = 32;
    auto bfs_sets = new tsl::robin_set<uint32_t>[MAX_BFS_LEVELS];

    bfs_sets[0].insert(_start);
    visited.set(_start);

    for (uint32_t i = (uint32_t)_max_points; i < _max_points + _num_frozen_pts; ++i)
    {
        if (i != _start)
        {
            bfs_sets[0].insert(i);
            visited.set(i);
        }
    }

    for (size_t l = 0; l < MAX_BFS_LEVELS - 1; ++l)
    {
        diskann::cout << "Number of nodes at BFS level " << l << " is " << bfs_sets[l].size() << std::endl;
        if (bfs_sets[l].size() == 0)
            break;
        for (auto node : bfs_sets[l])
        {
            for (auto nghbr : _final_graph[node])
            {
                if (!visited.test(nghbr))
                {
                    visited.set(nghbr);
                    bfs_sets[l + 1].insert(nghbr);
                }
            }
        }
    }

    delete[] bfs_sets;
}

// REFACTOR: This should be an OptimizedDataStore class, dummy impl here for
// compiling sake template <typename T, typename TagT, typename LabelT> void
// Index<T, TagT, LabelT>::optimize_index_layout()
//{ // use after build or load
//}

// REFACTOR: This should be an OptimizedDataStore class
template <typename T, typename TagT, typename LabelT> void Index<T, TagT, LabelT>::optimize_index_layout()
{ // use after build or load
    if (_dynamic_index)
    {
        throw diskann::ANNException("Optimize_index_layout not implemented for dyanmic indices", -1, __FUNCSIG__,
                                    __FILE__, __LINE__);
    }

    float *cur_vec = new float[_data_store->get_aligned_dim()];
    std::memset(cur_vec, 0, _data_store->get_aligned_dim() * sizeof(float));
    _data_len = (_data_store->get_aligned_dim() + 1) * sizeof(float);
    _neighbor_len = (_max_observed_degree + 1) * sizeof(uint32_t);
    _node_size = _data_len + _neighbor_len;
    _opt_graph = new char[_node_size * _nd];
    DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *)_data_store->get_dist_fn();
    for (uint32_t i = 0; i < _nd; i++)
    {
        char *cur_node_offset = _opt_graph + i * _node_size;
        _data_store->get_vector(i, (T *)cur_vec);
        float cur_norm = dist_fast->norm((T *)cur_vec, (uint32_t)_data_store->get_aligned_dim());
        std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
        std::memcpy(cur_node_offset + sizeof(float), cur_vec, _data_len - sizeof(float));

        cur_node_offset += _data_len;
        uint32_t k = (uint32_t)_final_graph[i].size();
        std::memcpy(cur_node_offset, &k, sizeof(uint32_t));
        std::memcpy(cur_node_offset + sizeof(uint32_t), _final_graph[i].data(), k * sizeof(uint32_t));
        std::vector<uint32_t>().swap(_final_graph[i]);
    }
    _final_graph.clear();
    _final_graph.shrink_to_fit();
    delete[] cur_vec;
}

//  REFACTOR: once optimized layout becomes its own Data+Graph store, we should
//  just invoke regular search
// template <typename T, typename TagT, typename LabelT>
// void Index<T, TagT, LabelT>::search_with_optimized_layout(const T *query,
// size_t K, size_t L, uint32_t *indices)
//{
//}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::_search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices)
{
    try
    {
        return this->search_with_optimized_layout(std::any_cast<const T *>(query), K, L, indices);
    }
    catch (const std::bad_any_cast &e)
    {
        throw ANNException(
            "Error: bad any cast while performing _search_with_optimized_layout() " + std::string(e.what()), -1);
    }
    catch (const std::exception &e)
    {
        throw ANNException("Error: " + std::string(e.what()), -1);
    }
}

template <typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::search_with_optimized_layout(const T *query, size_t K, size_t L, uint32_t *indices)
{
    DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *)_data_store->get_dist_fn();

    NeighborPriorityQueue retset(L);
    std::vector<uint32_t> init_ids(L);

    boost::dynamic_bitset<> flags{_nd, 0};
    uint32_t tmp_l = 0;
    uint32_t *neighbors = (uint32_t *)(_opt_graph + _node_size * _start + _data_len);
    uint32_t MaxM_ep = *neighbors;
    neighbors++;

    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++)
    {
        init_ids[tmp_l] = neighbors[tmp_l];
        flags[init_ids[tmp_l]] = true;
    }

    while (tmp_l < L)
    {
        uint32_t id = rand() % _nd;
        if (flags[id])
            continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
    }

    for (uint32_t i = 0; i < init_ids.size(); i++)
    {
        uint32_t id = init_ids[i];
        if (id >= _nd)
            continue;
        _mm_prefetch(_opt_graph + _node_size * id, _MM_HINT_T0);
    }
    L = 0;
    for (uint32_t i = 0; i < init_ids.size(); i++)
    {
        uint32_t id = init_ids[i];
        if (id >= _nd)
            continue;
        T *x = (T *)(_opt_graph + _node_size * id);
        float norm_x = *x;
        x++;
        float dist = dist_fast->compare(x, query, norm_x, (uint32_t)_data_store->get_aligned_dim());
        retset.insert(Neighbor(id, dist));
        flags[id] = true;
        L++;
    }

    while (retset.has_unexpanded_node())
    {
        auto nbr = retset.closest_unexpanded();
        auto n = nbr.id;
        _mm_prefetch(_opt_graph + _node_size * n + _data_len, _MM_HINT_T0);
        neighbors = (uint32_t *)(_opt_graph + _node_size * n + _data_len);
        uint32_t MaxM = *neighbors;
        neighbors++;
        for (uint32_t m = 0; m < MaxM; ++m)
            _mm_prefetch(_opt_graph + _node_size * neighbors[m], _MM_HINT_T0);
        for (uint32_t m = 0; m < MaxM; ++m)
        {
            uint32_t id = neighbors[m];
            if (flags[id])
                continue;
            flags[id] = 1;
            T *data = (T *)(_opt_graph + _node_size * id);
            float norm = *data;
            data++;
            float dist = dist_fast->compare(query, data, norm, (uint32_t)_data_store->get_aligned_dim());
            Neighbor nn(id, dist);
            retset.insert(nn);
        }
    }

    for (size_t i = 0; i < K; i++)
    {
        indices[i] = retset[i].id;
    }
}

/*  Internals of the library */
template <typename T, typename TagT, typename LabelT> const float Index<T, TagT, LabelT>::INDEX_GROWTH_FACTOR = 1.5f;

// EXPORTS
template DISKANN_DLLEXPORT class Index<float, int32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint32_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, int64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<float, uint64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint64_t, uint32_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint64_t, uint32_t>;
// Label with short int 2 byte
template DISKANN_DLLEXPORT class Index<float, int32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint32_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, int64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, int64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, int64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<float, uint64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<int8_t, uint64_t, uint16_t>;
template DISKANN_DLLEXPORT class Index<uint8_t, uint64_t, uint16_t>;

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search<uint64_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search<uint32_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search<uint64_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search<uint32_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search<uint64_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search<uint32_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search<uint64_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search<uint32_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search<uint64_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search<uint32_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search<uint64_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search<uint32_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search<uint64_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search<uint32_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search<uint64_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search<uint32_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search<uint64_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search<uint32_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search<uint64_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search<uint32_t>(const float *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search<uint64_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search<uint32_t>(const uint8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search<uint64_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint64_t *indices, float *distances);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search<uint32_t>(const int8_t *query, const size_t& K, const uint32_t& L, 
                                                                                                            const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                                                                            const int32_t& label1, const int32_t& label2, uint32_t *indices, float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
// TagT==uint32_t
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<float, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<uint8_t, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> Index<int8_t, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances, int8_t best_method, const uint32_t threshold_1, const uint32_t threshold_2);
} // namespace diskann