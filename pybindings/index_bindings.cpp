// index bindings of bipartite_index for python

#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>
#include <unistd.h>

#include "index.h"
#include "index_factory.h"

namespace py = pybind11;


class FilterDiskANN {
   public:
    FilterDiskANN(const diskann::Metric& m, const std::string &index_prefix, const size_t& num_points, 
                  const size_t& dimensions, const uint32_t& num_threads, const uint32_t& L) {
        _index = new diskann::Index<uint8_t>(m, dimensions, num_points,
                                        false, // not a dynamic_index
                                        false, // no enable_tags/ids
                                        false, // no concurrent_consolidate,
                                        false, // pq_dist_build
                                        0,     // num_pq_chunks
                                        false, // use_opq = false
                                        0);    // num_frozen_points
        _index->load(index_prefix.c_str(), num_threads, L);
        std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    }

    ~FilterDiskANN() { 
        delete _index;
    }

    void Search(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &queries_input, 
                py::array_t<int, py::array::c_style> &query_filters_offset_input, 
                py::array_t<int, py::array::c_style> &query_filters_data_input,
                const uint64_t num_queries, const uint64_t knn, const uint64_t L, 
                const uint32_t num_threads, uint32_t threshold_1, uint32_t threshold_2,
                py::array_t<uint32_t, py::array::c_style> &res_id) {
        
        auto queries = queries_input.unchecked();
        auto query_filters_offset = query_filters_offset_input.unchecked();
        auto query_filters_data = query_filters_data_input.unchecked();
        auto res = res_id.mutable_unchecked();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)num_queries; i++)
        {
            
            auto offset_i = query_filters_offset[i];
            auto offset_i1 = query_filters_offset[i+1];
            int32_t query_filter1 = query_filters_data[offset_i], query_filter2 = -1;
            if (offset_i1 - offset_i == 2)
                query_filter2 = query_filters_data[offset_i + 1];
            // std::cout << "(" << i << " " << query_filter1 << " " << query_filter2 << ") ";
            _index->search(&queries(i, 0), knn, L, threshold_1, threshold_2, query_filter1, query_filter2, &res(i, 0), nullptr);
        }
        // std::cout << "finish in cpp"  << std::endl;
    }

   private:
    diskann::Index<uint8_t, uint32_t, uint32_t>* _index;
};


void Build(const std::string& data_path, const std::string& index_path_prefix, const std::string& label_file, 
           const uint32_t& num_threads, const uint32_t& R, const uint32_t& L, const float& alpha) {

        std::string data_type = "uint8";
        std::string label_type = "uint";
        diskann::Metric metric = diskann::Metric::L2;

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
        auto index = index_factory.create_instance(); 
        index->build(data_path, data_num, build_params, label_file); // 调用的是Index类的build函数 
        index->save(index_path_prefix.c_str(), label_file); // 存了graph和data，search的时候要加载这两个
}

PYBIND11_MODULE(filterdiskann, m) {
    m.doc() = "pybind11 filterdiskann plugin";  // optional module docstring
    // enumerate...
    py::enum_<diskann::Metric>(m, "Metric")
        .value("L2", diskann::Metric::L2)
        .value("IP", diskann::Metric::INNER_PRODUCT)
        .value("COSINE", diskann::Metric::COSINE)
        .export_values();
    py::class_<FilterDiskANN>(m, "FilterDiskANN")
        .def(py::init<const diskann::Metric&, const std::string&, const size_t&, 
                      const size_t&, const uint32_t&, const uint32_t&>())
        .def("search", &FilterDiskANN::Search);
    m.def("build", &Build);
}
