// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

namespace po = boost::program_options;


template <class DT>
auto load_filter_index(std::string index_path, diskann::Metric m, size_t num_points, uint32_t num_threads, uint32_t dimensions, uint32_t L) {
    auto index = new diskann::Index<DT>(m, dimensions, num_points,
                                    false, // not a dynamic_index
                                    false, // no enable_tags/ids
                                    false, // no concurrent_consolidate,
                                    false, // pq_dist_build
                                    0,     // num_pq_chunks
                                    false, // use_opq = false
                                    0);    // num_frozen_points
    index->load(index_path.c_str(), num_threads, L);
    std::cout << "Index loaded" << std::endl;
    return index;
}


template <class DT>
auto batch_search(DT* queries, const uint64_t num_queries, const uint64_t knn, 
                 const uint64_t complexity, const uint32_t num_threads,

                 // 这些是新加的参数，其中query_filters是一个二维vector，你看看怎么处理接口方便
                 int32_t* query_filters_offset, int32_t* query_filters_data, uint32_t threshold_1, uint32_t threshold_2,
                 
                 // 以下这些是给cpp使用的参数，不用放进去python接口里
                 diskann::Index<DT>* index, std::vector<uint32_t>& cmp_stats, std::vector<float>& latency_stats)
{
    // 这部分就在StaticMemoryIndex<DT>::batch_search里，不用动的
    const uint32_t _num_threads = num_threads != 0 ? num_threads : omp_get_num_threads();
    omp_set_num_threads(static_cast<int32_t>(_num_threads));
    
    // 这部分是给python用的，其实就在StaticMemoryIndex<DT>::batch_search里，不用动的
    // py::array_t<StaticIdType> ids({num_queries, knn});
    // py::array_t<float> dists({num_queries, knn});

    // 这部分是给cpp跑测试用的，python不用用到
    std::vector<uint32_t> ids;
    std::vector<float> dists;
    ids.resize(num_queries * knn);
    dists.resize(num_queries * knn);

    // 这个给cpp测试用的，python的不用放这部分
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)num_queries; i++)
    {
        auto qs = std::chrono::high_resolution_clock::now();
        int32_t query_filter1 = query_filters_data[query_filters_offset[i]], query_filter2 = -1;
        if (query_filters_offset[i+1] - query_filters_offset[i] == 2)
            query_filter2 = query_filters_data[query_filters_offset[i]+1];
        auto retval =
            index->search(queries + i* 192, knn, complexity, threshold_1, threshold_2, query_filter1, query_filter2, 
                                                ids.data() + i * knn, dists.data() + i * knn);
        auto qe = std::chrono::high_resolution_clock::now();
        cmp_stats[i] = retval.second;
        std::chrono::duration<double> diff = qe - qs;
        latency_stats[i] = (float)(diff.count() * 1000000);
    }

    // 这个是给python用的，复制进StaticMemoryIndex<DT>::batch_search就行
// #pragma omp parallel for schedule(dynamic, 1) default(none) shared(num_queries, queries, knn, complexity, ids, dists)
//     for (int64_t i = 0; i < (int64_t)num_queries; i++)
//     {
//         int32_t query_filter1 = query_filters_data[query_filters_offset[i]], query_filter2 = -1;
//         if (query_filters_offset[i+1] - query_filters_offset[i] == 2)
//             query_filter2 = query_filters_data[query_filters_offset[i]+1];
//         _index.search_with_filters_neurips23(queries.data(i), query_filter1, query_filter2, knn, complexity, 
//                                              ids.mutable_data(i), dists.mutable_data(i), -1, threshold_1, threshold_2);
//     }

    return std::make_pair(ids, dists);
}


std::string query_filters_file;
template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const float fail_if_recall_below, const uint32_t threshold_1, const uint32_t threshold_2)
{ // recall_at == K(k top)
    std::vector<std::vector<LabelT>> query_filters;
    bool filtered_search = true;

    if (query_filters_file != "")
        query_filters = read_file_to_vector_of_num<LabelT>(query_filters_file);
    else
        filtered_search = false;

    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num,
        gt_dim; // gt_num == query_num，gt_dim是K' top，也就是真实的最邻近节点的个数（不是向量的维度）
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);
    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }
    
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);
    // auto index = load_filter_index(index_path, num_threads, query_dim, *(std::max_element(Lvec.begin(), Lvec.end())));
    auto index = load_filter_index<T>(index_path, metric, num_frozen_pts, num_threads, query_dim, *(std::max_element(Lvec.begin(), Lvec.end())));

    if (metric == diskann::FAST_L2)
        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout << "Threshold_1: " << threshold_1 << ", threshold_2: " << threshold_2 << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats; // 从候选集的邻居中扩展得到的节点数量,也是比较次数
    if (not tags)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        // read a vector best_method from file
        // std::vector<int8_t> best_methods;
        // std::ifstream in("../data/yfcc100M/best_methods_40000.txt");
        // std::string line;
        // while (getline(in, line)) {
        //     std::stringstream ss(line);
        //     int temp;
        //     ss >> temp;
        //     best_methods.push_back((int8_t)temp);
        // }

        int32_t* query_filter_offset = new int32_t[query_num+1]();
        int32_t* query_filter_data = new int32_t[2*query_num];
        for (uint32_t i = 0; i < query_num; i++) {
            for (uint32_t j = 0; j < query_filters[i].size(); j++) {
                query_filter_data[query_filter_offset[i]+j] = query_filters[i][j];
            }
            query_filter_offset[i+1] = query_filter_offset[i] + query_filters[i].size();
        }

        auto s = std::chrono::high_resolution_clock::now();
        auto res = batch_search<T>(query, query_num, recall_at, L, num_threads, query_filter_offset, query_filter_data, 
                           threshold_1, threshold_2, index, cmp_stats, latency_stats);
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        query_result_ids[test_id] = res.first;
        query_result_dists[test_id] = res.second;

        delete[] query_filter_offset;
        delete[] query_filter_data;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0.0) / (float)query_num;

        if (tags)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }

    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, filter_label, label_type;
    uint32_t num_threads, K, threshold_1, threshold_2;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION); // 查询时 候选集大小
        required_configs.add_options()("threshold_1", po::value<uint32_t>(&threshold_1)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("threshold_2", po::value<uint32_t>(&threshold_2)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()(
            "query_filters_file", po::value<std::string>(&query_filters_file)->default_value(std::string("")),
            program_options_utils::FILTERS_FILE_DESCRIPTION); // 得是int型的，因此需要人工或者用程序转换一下
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);

        // Output controls
        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified "
                                      "recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in "
                                      "the output table");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs).add(output_controls);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    if (dynamic && not tags)
    {
        std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
        return -1;
    }

    if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0)
    {
        std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    // 搜索的时候不压缩向量
    // try
    // {
        if (query_filters_file != "" && label_type == "uint")
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t, uint32_t>(metric, index_path_prefix, result_path, query_file,
                                                             gt_file, num_threads, K, print_all_recalls, Lvec, dynamic,
                                                             tags, show_qps_per_thread, fail_if_recall_below,
                                                             threshold_1, threshold_2);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t, uint32_t>(metric, index_path_prefix, result_path, query_file,
                                                              gt_file, num_threads, K, print_all_recalls, Lvec, dynamic,
                                                              tags, show_qps_per_thread, fail_if_recall_below,
                                                              threshold_1, threshold_2);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float, uint32_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                            num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                            show_qps_per_thread, fail_if_recall_below,
                                                            threshold_1, threshold_2);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                   num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                   show_qps_per_thread, fail_if_recall_below,
                                                   threshold_1, threshold_2);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, fail_if_recall_below,
                                                    threshold_1, threshold_2);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, fail_if_recall_below,
                                                  threshold_1, threshold_2);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
    // }
    // catch (std::exception &e)
    // {
    //     std::cout << std::string(e.what()) << std::endl;
    //     diskann::cerr << "Index search failed." << std::endl;
    //     return -1;
    // }
}