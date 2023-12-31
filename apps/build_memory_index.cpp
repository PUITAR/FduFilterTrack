// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
int build_in_memory_index(const diskann::Metric &metric, const std::string &data_path, const uint32_t R,
                          const uint32_t L, const float alpha, const std::string &save_path, const uint32_t num_threads,
                          const bool use_pq_build, const size_t num_pq_bytes, const bool use_opq,
                          const std::string &label_file, const std::string &universal_label, const uint32_t Lf)
{
    diskann::IndexWriteParameters paras = diskann::IndexWriteParametersBuilder(L, R)
                                              .with_filter_list_size(Lf)
                                              .with_alpha(alpha)
                                              .with_saturate_graph(false)
                                              .with_num_threads(num_threads)
                                              .build();
    std::string labels_file_to_use = save_path + "_label_formatted.txt";
    std::string mem_labels_int_map_file = save_path + "_labels_map.txt";

    size_t data_num, data_dim;
    diskann::get_bin_metadata(data_path, data_num, data_dim);

    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, false, false, false, use_pq_build, num_pq_bytes,
                                          use_opq);
    auto s = std::chrono::high_resolution_clock::now();
    if (label_file == "")
    {
        index.build(data_path.c_str(), data_num, paras);
    }
    else
    {
        convert_labels_string_to_int(label_file, labels_file_to_use, mem_labels_int_map_file, universal_label);
        if (universal_label != "")
        {
            LabelT unv_label_as_num = 0;
            index.set_universal_label(unv_label_as_num);
        }
        index.build_filtered_index(data_path.c_str(), labels_file_to_use, data_num, paras);
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
    index.save(save_path.c_str());
    if (label_file != "")
        std::remove(labels_file_to_use.c_str());
    return 0;
}


int build_in_memory_index_for_python(std::string data_path, std::string index_path_prefix, std::string label_file, 
                                     uint32_t num_threads, uint32_t R, uint32_t L, float alpha) 
{
    try
    {
        std::string data_type = "uint8";
        std::string dist_fn = "l2";
        std::string label_type = "uint";
        
        diskann::Metric metric;
        if (dist_fn == std::string("mips"))
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
        else
        {
            std::cout << "Unsupported distance function. Currently only L2/ Inner "
                        "Product/Cosine are supported."
                    << std::endl;
            return -1;
        }

        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

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

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)  
                                      .with_filter_list_size(0) // 过滤的L
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false) // 是否为饱和图，即每个node的邻居要==R
                                      .with_num_threads(num_threads)
                                      .build();

        auto build_params = diskann::IndexBuildParamsBuilder(index_build_params)
                                .with_universal_label("")
                                .with_label_file(label_file)
                                .with_save_path_prefix(index_path_prefix)
                                .build();
        auto index_factory = diskann::IndexFactory(config);
        // 后面应该调用build_in_memory_index()
        auto index = index_factory.create_instance(); 
        index->build(data_path, data_num, build_params, label_file); // 调用的是Index类的build函数
        
        index->save(index_path_prefix.c_str(), label_file); // 存了graph和data，search的时候要加载这两个
        // 如果是filter，还存了universal_label.txt labels_to_medoids.txt labels.txt
        // labels_to_medoids.txt的格式是{label, id} labels.txt的格式是{label1, label2(int)}
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }
    return 0;
}


int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION); // Path prefix to the index, e.g. '/mnt/data/my_ann_index
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH); // Input data file in bin format. 向量

        // Optional parameters
        po::options_description optional_configs("Optional");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->default_value("uint8"),
                                       program_options_utils::DATA_TYPE_DESCRIPTION); // int8, uint8, float
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->default_value("l2"),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION); // 距离函数
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE); // 出度，要从L里筛选出R个节点作为邻居
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY); // 候选集大小
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA); // 论文中的α值
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES); // 使用PQ压缩后的空间维度(子空间数量)
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ); // 是否使用OPQ压缩技术(OPQ 是对 PQ 的改进)
        // 根据标签文件中的信息，将节点按照其标签进行分类和组织，得到过滤索引
        // 在定位到与查询条件匹配的组后，我们只需要在这个组中进行实际的最近邻搜索，而不需要在整个数据集中搜索。这样可以大大加快搜索过程

        // 指定标签文件的路径，用于图形构建（例如，一个标签文件可以定义每个节点的含义，篮球、足球等）
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE); 
        // 通用标签，例如其他节点是篮球、足球，某个节点可能是UNIVERSAL_LABEL，表示什么都适用
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL); 

        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD); // 过滤的候选集L
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION); // 标签的数据类型

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    return build_in_memory_index_for_python(data_path, index_path_prefix, label_file, num_threads, R, L, alpha);
}
