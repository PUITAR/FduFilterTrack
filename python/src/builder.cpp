// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "builder.h"
#include "common.h"
#include "disk_utils.h"
#include "index.h"
#include "parameters.h"

#include <iostream>

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace diskannpy
{
template <typename DT>
void build_disk_index(const diskann::Metric metric, const std::string &data_file_path,
                      const std::string &index_prefix_path, const uint32_t complexity, const uint32_t graph_degree,
                      const double final_index_ram_limit, const double indexing_ram_budget, const uint32_t num_threads,
                      const uint32_t pq_disk_bytes)
{
    std::string params = std::to_string(graph_degree) + " " + std::to_string(complexity) + " " +
                         std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) + " " +
                         std::to_string(num_threads);
    if (pq_disk_bytes > 0)
        params = params + " " + std::to_string(pq_disk_bytes);
    diskann::build_disk_index<DT>(data_file_path.c_str(), index_prefix_path.c_str(), params.c_str(), metric);
}

template void build_disk_index<float>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                      double, double, uint32_t, uint32_t);

template void build_disk_index<uint8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                        double, double, uint32_t, uint32_t);
template void build_disk_index<int8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                       double, double, uint32_t, uint32_t);

template <typename T, typename TagT, typename LabelT>
void build_memory_index(const diskann::Metric metric, const std::string &vector_bin_path,
                        const std::string &index_output_path, const uint32_t graph_degree, const uint32_t complexity,
                        const float alpha, const uint32_t num_threads, const bool use_pq_build,
                        const size_t num_pq_bytes, const bool use_opq, std::string &label_file, const uint32_t filter_complexity,
                        const bool use_tags)
{
    std::cerr << "#######################\n";
    // diskann::IndexWriteParameters index_build_params = diskann::IndexWriteParametersBuilder(complexity, graph_degree)
    //                                                        .with_filter_list_size(filter_complexity)
    //                                                        .with_alpha(alpha)
    //                                                        .with_saturate_graph(false)
    //                                                        .with_num_threads(num_threads)
    //                                                        .build();
    // size_t data_num, data_dim;
    // diskann::get_bin_metadata(vector_bin_path, data_num, data_dim);
    // diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, use_tags, use_tags, false, use_pq_build,
    //                                       num_pq_bytes, use_opq);

    // if (use_tags)
    // {
    //     const std::string tags_file = index_output_path + ".tags";
    //     if (!file_exists(tags_file))
    //     {
    //         throw std::runtime_error("tags file not found at expected path: " + tags_file);
    //     }
    //     TagT *tags_data;
    //     size_t tag_dims = 1;
    //     diskann::load_bin(tags_file, tags_data, data_num, tag_dims);
    //     std::vector<TagT> tags(tags_data, tags_data + data_num);
    //     index.build(vector_bin_path.c_str(), data_num, index_build_params, tags);
    // }
    // else
    // {
    //     index.build(vector_bin_path.c_str(), data_num, index_build_params);
    // }

    // index.save(index_output_path.c_str());
    std::string data_type = "uint8";
    std::string label_type = "uint";
    // std::string label_file = "/home/";

    
    diskann::cout << "Starting index build with graph_degree: " << graph_degree << "  Lbuild: " << complexity << "  alpha: " << alpha
                    << "  #threads: " << num_threads << std::endl;

    std::cerr << "label_file 0 :" + label_file + "\n";
    size_t data_num, data_dim;
    diskann::get_bin_metadata(vector_bin_path, data_num, data_dim);
    std::cerr << "label_file 1 :" + label_file + "\n";

    auto config = diskann::IndexConfigBuilder()
                        .with_metric(metric)
                        .with_dimension(data_dim)
                        .with_max_points(data_num)
                        .with_data_load_store_strategy(diskann::MEMORY)
                        .with_data_type(data_type)
                        .with_label_type(label_type)
                        .is_dynamic_index(false)
                        .is_enable_tags(false)
                        .is_use_opq(false)
                        .is_pq_dist_build(false)
                        .with_num_pq_chunks(0)
                        .build();

    std::cerr << "label_file 2 :" + label_file + "\n";
    auto index_build_params = diskann::IndexWriteParametersBuilder(complexity, graph_degree)  
                                    .with_filter_list_size(0) // 过滤的L
                                    .with_alpha(alpha)
                                    .with_saturate_graph(false) // 是否为饱和图，即每个node的邻居要==graph_degree
                                    .with_num_threads(num_threads)
                                    .build();
    std::cerr << "label_file 3 :" + label_file + "\n";

    auto build_params = diskann::IndexBuildParamsBuilder(index_build_params)
                            .with_universal_label("")
                            .with_label_file(label_file)
                            .with_save_path_prefix(index_output_path)
                            .build();
    std::cerr << "label_file 4 :" + label_file + "\n";
    auto index_factory = diskann::IndexFactory(config);
    std::cerr << "label_file 5 :" + label_file + "\n";
    // 后面应该调用build_in_memory_index()
    auto index = index_factory.create_instance(); 
    std::cerr << "label_file 6 :" + label_file + "\n";
    index->build(vector_bin_path, data_num, build_params, label_file); // 调用的是Index类的build函数
    std::cerr << "label_file 7 :" + label_file + "\n";
    
    index->save(index_output_path.c_str(), label_file); // 存了graph和data，search的时候要加载这两个
    std::cerr << "label_file 8 :" + label_file + "\n";
    // 如果是filter，还存了universal_label.txt labels_to_medoids.txt labels.txt
    // labels_to_medoids.txt的格式是{label, id} labels.txt的格式是{label1, label2(int)}
    index.reset();
}

template void build_memory_index<float>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                        float, uint32_t, bool, size_t, bool, std::string &, uint32_t, bool);

template void build_memory_index<int8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                         float, uint32_t, bool, size_t, bool, std::string &, uint32_t, bool);

template void build_memory_index<uint8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                          float, uint32_t, bool, size_t, bool, std::string &, uint32_t, bool);

} // namespace diskannpy
