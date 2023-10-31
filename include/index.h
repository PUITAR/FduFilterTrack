// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common_includes.h"

#ifdef EXEC_ENV_OLS
#include "aligned_file_reader.h"
#endif

#include "distance.h"
#include "locking.h"
#include "natural_number_map.h"
#include "natural_number_set.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "in_mem_data_store.h"
#include "abstract_index.h"

#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0
#define DEFAULT_MAXC 750

typedef unsigned short int vl_type;
#include <string.h>
#include <deque>
#include <mutex>

class VisitedList {
   public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
#define POOLLIKELY(x) __builtin_expect(x, 1)
#define POOLUNLIKELY(x) __builtin_expect(x, 0)
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

   public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++) pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock<std::mutex> lock(poolguard);
            if (POOLLIKELY(pool.size() > 0)) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock<std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};

template<typename node_id_t>
class GorderPriorityQueue{

	typedef std::unordered_map<node_id_t, int> map_t;

	struct Node {
		node_id_t key;
		int priority;
	};

	std::vector<Node> list;
	map_t index_table; // map: key -> index in list

	inline void swap(int i, int j){
		Node tmp = list[i];
		list[i] = list[j];
		list[j] = tmp;
		index_table[list[i].key] = i;
		index_table[list[j].key] = j;
	}


	public:
	GorderPriorityQueue(const std::vector<node_id_t>& nodes){
		for (int i = 0; i < nodes.size(); i++){
			list.push_back({nodes[i],0});
			index_table[nodes[i]] = i;
		}
	}

	GorderPriorityQueue(size_t N){
		for (unsigned int i = 0; i < N; i++){
			list.push_back({i,0});
			index_table[i] = i;
		}
	}


	void print(){
		for(int i = 0; i < list.size(); i++){
			std::cout<<"("<<list[i].key<<":"<<list[i].priority<<")"<<" ";
		}
		std::cout<<std::endl;
	}


	static bool compare(const Node &a, const Node &b){
		return (a.priority < b.priority);
	}


	void increment(node_id_t key){
		typename map_t::const_iterator i = index_table.find(key);
		if (i == index_table.end()){
			return;
		}
		// int new_index = list.size()-1;
		// while((new_index > 0) && (list[new_index].priority > list[i->second].priority)){
		// 	new_index--;
		// }

		auto it = std::upper_bound(list.begin(), list.end(), list[i->second], compare);
		size_t new_index = it - list.begin() - 1; // possible bug
		// new_index points to the right-most element with same priority as key
		// i.e. priority equal to "list[i->second].priority" (i.e. the current priority)
		swap(i->second, new_index);
		list[new_index].priority++;
	}

	void decrement(node_id_t key){
		typename map_t::const_iterator i = index_table.find(key);
		if (i == index_table.end()){
			return;
		}
		// int new_index = list.size()-1;
		// while((new_index > 0) && (list[new_index].priority >= list[i->second].priority)){
		// 	new_index--;
		// }
		// new_index++;
		// i shoudl do this better but am pressed for time now
		auto it = std::lower_bound(list.begin(), list.end(), list[i->second], compare);
		size_t new_index = it - list.begin(); // POSSIBLE BUG
		// while((new_index > list.size()) && (list[new_index].priority == list[i->second].priority)){
		// 	new_index++;
		// }
		// new_index--;
		// new_index points to the right-most element with same priority as key

		swap(i->second, new_index);
		list[new_index].priority--;
	}
	
	node_id_t pop(){
		Node max = list.back();
		list.pop_back();
		index_table.erase(max.key);
		return max.key;
	}

	size_t size(){
		return list.size();
	}

};

namespace diskann
{

inline double estimate_ram_usage(size_t size, uint32_t dim, uint32_t datasize, uint32_t degree)
{
    double size_of_data = ((double)size) * ROUND_UP(dim, 8) * datasize;
    double size_of_graph = ((double)size) * degree * sizeof(uint32_t) * GRAPH_SLACK_FACTOR;
    double size_of_locks = ((double)size) * sizeof(non_recursive_mutex);
    double size_of_outer_vector = ((double)size) * sizeof(ptrdiff_t);

    return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector);
}

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t> class Index : public AbstractIndex
{
    /**************************************************************************
     *
     * Public functions acquire one or more of _update_lock, _consolidate_lock,
     * _tag_lock, _delete_lock before calling protected functions which DO NOT
     * acquire these locks. They might acquire locks on _locks[i]
     *
     **************************************************************************/

  public:
    // Constructor for Bulk operations and for creating the index object solely
    // for loading a prexisting index.
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim, const size_t max_points = 1, const bool dynamic_index = false,
                            const bool enable_tags = false, const bool concurrent_consolidate = false,
                            const bool pq_dist_build = false, const size_t num_pq_chunks = 0,
                            const bool use_opq = false, const size_t num_frozen_pts = 0,
                            const bool init_data_store = true);

    // Constructor for incremental index
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim, const size_t max_points, const bool dynamic_index,
                            const IndexWriteParameters &indexParameters, const uint32_t initial_search_list_size,
                            const uint32_t search_threads, const bool enable_tags = false,
                            const bool concurrent_consolidate = false, const bool pq_dist_build = false,
                            const size_t num_pq_chunks = 0, const bool use_opq = false);

    DISKANN_DLLEXPORT Index(const IndexConfig &index_config, std::unique_ptr<AbstractDataStore<T>> data_store
                            /* std::unique_ptr<AbstractGraphStore> graph_store*/);

    DISKANN_DLLEXPORT ~Index();

    // Saves graph, data, metadata and associated tags.
    DISKANN_DLLEXPORT void save(const char *filename, std::string label_filename, bool compact_before_save = false);
    DISKANN_DLLEXPORT void save(const char *filename, bool compact_before_save = false);

    // Load functions
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l);
#else
    // Reads the number of frozen points from graph's metadata file section.
    DISKANN_DLLEXPORT static size_t get_graph_num_frozen_points(const std::string &graph_file);

    DISKANN_DLLEXPORT void load(const char *index_file, uint32_t num_threads, uint32_t search_l);
#endif

    // get some private variables
    DISKANN_DLLEXPORT size_t get_num_points();
    DISKANN_DLLEXPORT size_t get_max_points();

    // DISKANN_DLLEXPORT bool detect_common_filters(uint32_t point_id, bool search_invocation,
    //                                              const std::vector<LabelT> &incoming_labels);
    DISKANN_DLLEXPORT bool contain_required_filters(const uint32_t& point_id, const int32_t &label1, const int32_t &label2);

    // Batch build from a file. Optionally pass tags vector.
    DISKANN_DLLEXPORT void build(const char *filename, const size_t num_points_to_load,
                                 const IndexWriteParameters &parameters,
                                 const std::vector<TagT> &tags = std::vector<TagT>());

    // Batch build from a file. Optionally pass tags file.
    DISKANN_DLLEXPORT void build(const char *filename, const size_t num_points_to_load,
                                 const IndexWriteParameters &parameters, const char *tag_filename);

    // Batch build from a data array, which must pad vectors to aligned_dim
    DISKANN_DLLEXPORT void build(const T *data, const size_t num_points_to_load, const IndexWriteParameters &parameters,
                                 const std::vector<TagT> &tags);

    DISKANN_DLLEXPORT void build(const std::string &data_file, const size_t num_points_to_load,
                                 IndexBuildParams &build_params);
    DISKANN_DLLEXPORT void build(const std::string &data_file, const size_t num_points_to_load,
                                 IndexBuildParams &build_params, const std::string& label_file);
                                 
    // Filtered Support
    DISKANN_DLLEXPORT void build_filtered_index(const char *filename, const std::string &label_file,
                                                const size_t num_points_to_load, IndexWriteParameters &parameters,
                                                const std::vector<TagT> &tags = std::vector<TagT>());

    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

    // Get converted integer label from string to int map (_label_map)
    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &raw_label);

    // Set starting point of an index before inserting any points incrementally.
    // The data count should be equal to _num_frozen_pts * _aligned_dim.
    DISKANN_DLLEXPORT void set_start_points(const T *data, size_t data_count);
    // Set starting points to random points on a sphere of certain radius.
    // A fixed random seed can be specified for scenarios where it's important
    // to have higher consistency between index builds.
    DISKANN_DLLEXPORT void set_start_points_at_random(T radius, uint32_t random_seed = 0);

    // For FastL2 search on a static index, we interleave the data with graph
    DISKANN_DLLEXPORT void optimize_index_layout();

    // For FastL2 search on optimized layout
    DISKANN_DLLEXPORT void search_with_optimized_layout(const T *query, size_t K, size_t L, uint32_t *indices);

    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    template <typename IDType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(const T *query, const size_t& K, const uint32_t& L, 
                                                           const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                           const int32_t& label1, const int32_t& label2, IDType *indices, float *distances = nullptr);

    // Initialize space for res_vectors before calling.
    DISKANN_DLLEXPORT size_t search_with_tags(const T *query, const uint64_t K, const uint32_t L, TagT *tags,
                                              float *distances, std::vector<T *> &res_vectors);

    // Filter support search
    template <typename IndexType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search_with_filters(const T *query, const LabelT &filter_label,
                                                                        const size_t K, const uint32_t L,
                                                                        IndexType *indices, float *distances, int8_t best_method,
                                                                        const uint32_t threshold_1, const uint32_t threshold_2);

    template <typename IndexType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search_with_filters(const T *query,
                                                                        const std::vector<LabelT> &filter_labels,
                                                                        const size_t K, const uint32_t L,
                                                                        IndexType *indices, float *distances, int8_t best_method,
                                                                        const uint32_t threshold_1, const uint32_t threshold_2);

    // Will fail if tag already in the index or if tag=0.
    DISKANN_DLLEXPORT int insert_point(const T *point, const TagT tag);

    // call this before issuing deletions to sets relevant flags
    DISKANN_DLLEXPORT int enable_delete();

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK.
    DISKANN_DLLEXPORT int lazy_delete(const TagT &tag);

    // Record deleted points now and restructure graph later. Add to failed_tags
    // if tag not found.
    DISKANN_DLLEXPORT void lazy_delete(const std::vector<TagT> &tags, std::vector<TagT> &failed_tags);

    // Call after a series of lazy deletions
    // Returns number of live points left after consolidation
    // If _conc_consolidates is set in the ctor, then this call can be invoked
    // alongside inserts and lazy deletes, else it acquires _update_lock
    DISKANN_DLLEXPORT consolidation_report consolidate_deletes(const IndexWriteParameters &parameters);

    DISKANN_DLLEXPORT void prune_all_neighbors(const uint32_t max_degree, const uint32_t max_occlusion,
                                               const float alpha);

    DISKANN_DLLEXPORT bool is_index_saved();

    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    DISKANN_DLLEXPORT void reposition_frozen_point_to_end();
    DISKANN_DLLEXPORT void reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                             uint32_t num_locations);

    // DISKANN_DLLEXPORT void save_index_as_one_file(bool flag);

    DISKANN_DLLEXPORT void get_active_tags(tsl::robin_set<TagT> &active_tags);

    // memory should be allocated for vec before calling this function
    DISKANN_DLLEXPORT int get_vector_by_tag(TagT &tag, T *vec);

    DISKANN_DLLEXPORT void print_status();

    DISKANN_DLLEXPORT void count_nodes_at_bfs_levels();

    // This variable MUST be updated if the number of entries in the metadata
    // change.
    DISKANN_DLLEXPORT static const int METADATA_ROWS = 5;

    // ********************************
    //
    // Internals of the library
    //
    // ********************************

  protected:
    // overload of abstract index virtual methods
    virtual void _build(const DataType &data, const size_t num_points_to_load, const IndexWriteParameters &parameters,
                        TagVector &tags) override;

    virtual std::pair<uint32_t, uint32_t> _search(const DataType &query, const size_t& K, const uint32_t& L, const uint32_t& threshold_1, const uint32_t& threshold_2, 
                                                  const int32_t& label1, const int32_t& label2, std::any &indices, float *distances = nullptr) override;
    virtual std::pair<uint32_t, uint32_t> _search_with_filters(const DataType &query,
                                                               const std::string &filter_label_raw, const size_t K,
                                                               const uint32_t L, std::any &indices,
                                                               float *distances, int8_t best_method,
                                                               const uint32_t threshold_1, const uint32_t threshold_2) override;

    virtual int _insert_point(const DataType &data_point, const TagType tag) override;

    virtual int _lazy_delete(const TagType &tag) override;

    virtual void _lazy_delete(TagVector &tags, TagVector &failed_tags) override;

    virtual void _get_active_tags(TagRobinSet &active_tags) override;

    virtual void _set_start_points_at_random(DataType radius, uint32_t random_seed = 0) override;

    virtual int _get_vector_by_tag(TagType &tag, DataType &vec) override;

    virtual void _search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices) override;

    virtual size_t _search_with_tags(const DataType &query, const uint64_t K, const uint32_t L, const TagType &tags,
                                     float *distances, DataVector &res_vectors) override;

    // No copy/assign.
    Index(const Index<T, TagT, LabelT> &) = delete;
    Index<T, TagT, LabelT> &operator=(const Index<T, TagT, LabelT> &) = delete;

    // Use after _data and _nd have been populated
    // Acquire exclusive _update_lock before calling
    void build_with_data_populated(const IndexWriteParameters &parameters, const std::vector<TagT> &tags);

    // generates 1 frozen point that will never be deleted from the graph
    // This is not visible to the user
    void generate_frozen_point();

    // determines navigating node of the graph by calculating medoid of datafopt
    uint32_t calculate_entry_point();

    void parse_label_file_neurips23(const std::string &label_file, uint32_t &num_points, uint32_t &num_labels, std::string& rewrite_filename);

    std::unordered_map<std::string, LabelT> load_label_map(const std::string &map_file);

    // Returns the locations of start point and frozen points suitable for use
    // with iterate_to_fixed_point.
    std::vector<uint32_t> get_init_ids();

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(const T *node_coords, const uint32_t Lindex,
                                                         const std::vector<uint32_t> &init_ids,
                                                         InMemQueryScratch<T> *scratch, bool use_filter,
                                                         const std::vector<LabelT> &filters, bool search_invocation);

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(const T *query, const uint32_t& Lsize, NeighborPriorityQueue& final_filter_nodes, 
                                                         const int32_t& label1, const int32_t& label2, const uint32_t& threshold_2);

    void search_for_point_and_prune(int location, uint32_t Lindex, std::vector<uint32_t> &pruned_list,
                                    InMemQueryScratch<T> *scratch, bool use_filter = false,
                                    uint32_t filteredLindex = 0);

    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
                         InMemQueryScratch<T> *scratch);

    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                         const uint32_t max_candidate_size, const float alpha, std::vector<uint32_t> &pruned_list,
                         InMemQueryScratch<T> *scratch);

    // Prunes candidates in @pool to a shorter list @result
    // @pool must be sorted before calling
    void occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha, const uint32_t degree,
                      const uint32_t maxc, std::vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
                      const tsl::robin_set<uint32_t> *const delete_set_ptr = nullptr);

    // add reverse links from all the visited nodes to node n.
    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                      InMemQueryScratch<T> *scratch);

    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch);

    // Acquire exclusive _update_lock before calling
    void link(const IndexWriteParameters &parameters);

    // Acquire exclusive _tag_lock and _delete_lock before calling
    int reserve_location();

    // Acquire exclusive _tag_lock before calling
    size_t release_location(int location);
    size_t release_locations(const tsl::robin_set<uint32_t> &locations);

    // Resize the index when no slots are left for insertion.
    // Acquire exclusive _update_lock and _tag_lock before calling.
    void resize(size_t new_max_points);

    // Acquire unique lock on _update_lock, _consolidate_lock, _tag_lock
    // and _delete_lock before calling these functions.
    // Renumber nodes, update tag and location maps and compact the
    // graph, mode = _consolidated_order in case of lazy deletion and
    // _compacted_order in case of eager deletion
    DISKANN_DLLEXPORT void compact_data();
    DISKANN_DLLEXPORT void compact_frozen_point();

    // Remove deleted nodes from adjacency list of node loc
    // Replace removed neighbors with second order neighbors.
    // Also acquires _locks[i] for i = loc and out-neighbors of loc.
    void process_delete(const tsl::robin_set<uint32_t> &old_delete_set, size_t loc, const uint32_t range,
                        const uint32_t maxc, const float alpha, InMemQueryScratch<T> *scratch);

    void initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l, uint32_t r,
                                  uint32_t maxc, size_t dim);

    // Do not call without acquiring appropriate locks
    // call public member functions save and load to invoke these.
    DISKANN_DLLEXPORT size_t save_graph(std::string filename);
    DISKANN_DLLEXPORT size_t save_data(std::string filename);
    DISKANN_DLLEXPORT size_t save_tags(std::string filename);
    DISKANN_DLLEXPORT size_t save_delete_list(const std::string &filename);
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT size_t load_graph(AlignedFileReader &reader, size_t expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_tags(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_delete_set(AlignedFileReader &reader);
#else
    DISKANN_DLLEXPORT size_t load_graph(const std::string filename, size_t expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(std::string filename0);
    DISKANN_DLLEXPORT size_t load_tags(const std::string tag_file_name);
    DISKANN_DLLEXPORT size_t load_delete_set(const std::string &filename);
#endif

  private:
    // Distance functions
    Metric _dist_metric = diskann::L2;
    std::shared_ptr<Distance<T>> _distance;

    // Data
    std::unique_ptr<AbstractDataStore<T>> _data_store;
    char *_opt_graph = nullptr;

    // new to raw id mapping
    std::vector<uint32_t> _new_to_raw_id;

    // Graph related data structures
    std::vector<std::vector<uint32_t>> _final_graph;
    std::vector<uint32_t> _csr_offset, _csr_data;

    T *_data = nullptr; // coordinates of all base points
    // Dimensions
    size_t _dim = 0;
    size_t _nd = 0;         // number of active points i.e. existing in the graph
    size_t _max_points = 0; // total number of points in given data set

    // _num_frozen_pts is the number of points which are used as initial
    // candidates when iterating to closest point(s). These are not visible
    // externally and won't be returned by search. At least 1 frozen point is
    // needed for a dynamic index. The frozen points have consecutive locations.
    // See also _start below.
    size_t _num_frozen_pts = 0;
    size_t _max_range_of_loaded_graph = 0;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    uint32_t _max_observed_degree = 0;
    // Start point of the search. When _num_frozen_pts is greater than zero,
    // this is the location of the first frozen point. Otherwise, this is a
    // location of one of the points in index.
    uint32_t _start = 0;

    bool _has_built = false;
    bool _saturate_graph = false;
    bool _save_as_one_file = false; // plan to support in next version
    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _normalize_vecs = false; // Using normalied L2 for cosine.
    bool _deletes_enabled = false;

    // Filter Support

    bool _filtered_index = false;
    int64_t *_pts_to_labels_hash_and_offset;
    uint32_t *_pts_to_labels_data;
    int64_t *_label_to_pts_offset;
    uint32_t *_label_to_pts_data;
    std::string _labels_file;
    std::vector<uint32_t> _label_to_medoid_id;
    std::unordered_map<uint32_t, uint32_t> _medoid_counts;
    bool _use_universal_label = false;
    uint64_t _universal_label = -1;
    uint32_t _filterIndexingQueueSize;
    std::unordered_map<std::string, LabelT> _label_map;
    std::vector<int64_t> _label_to_hash;

    // Indexing parameters
    uint32_t _indexingQueueSize; // L候选集大小
    uint32_t _indexingRange;     // R最大度
    uint32_t _indexingMaxC;      // max_candidate_size，用来控制pool的大小
    float _indexingAlpha;

    // Query scratch data structures
    ConcurrentQueue<InMemQueryScratch<T> *> _query_scratch;

    // Flags for PQ based distance calculation
    bool _pq_dist = false;
    bool _use_opq = false;
    size_t _num_pq_chunks = 0;
    uint8_t *_pq_data = nullptr;
    bool _pq_generated = false;
    FixedChunkPQTable _pq_table;

    //
    // Data structures, locks and flags for dynamic indexing and tags
    //

    // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
    // _location_to_tag does not resolve a location, infer that it was deleted.
    tsl::sparse_map<TagT, uint32_t> _tag_to_location;
    natural_number_map<uint32_t, TagT> _location_to_tag;

    // _empty_slots has unallocated slots and those freed by consolidate_delete.
    // _delete_set has locations marked deleted by lazy_delete. Will not be
    // immediately available for insert. consolidate_delete will release these
    // slots to _empty_slots.
    natural_number_set<uint32_t> _empty_slots;
    std::unique_ptr<tsl::robin_set<uint32_t>> _delete_set;

    bool _data_compacted = true;    // true if data has been compacted
    bool _is_saved = false;         // Checking if the index is already saved.
    bool _conc_consolidate = false; // use _lock while searching

    // Acquire locks in the order below when acquiring multiple locks
    std::shared_timed_mutex // RW mutex between save/load (exclusive lock) and
        _update_lock;       // search/inserts/deletes/consolidate (shared lock)
    std::shared_timed_mutex // Ensure only one consolidate or compact_data is
        _consolidate_lock;  // ever active
    std::shared_timed_mutex // RW lock for _tag_to_location,
        _tag_lock;          // _location_to_tag, _empty_slots, _nd, _max_points
    std::shared_timed_mutex // RW Lock on _delete_set and _data_compacted
        _delete_lock;       // variable

    // Per node lock, cardinality=_max_points
    std::vector<non_recursive_mutex> _locks;

    static const float INDEX_GROWTH_FACTOR;

    VisitedListPool *visited_list_pool_{nullptr};
};
} // namespace diskann
