#include "data_store.h"
#include "util.h"

void datapoint_have_to_query_dist()
{
    std::ofstream outputFile("datapoint(have)_to_query_dist.log");

    // 重定向输出到文件
    std::streambuf *originalOutput = std::cout.rdbuf(); // 保存原始输出流
    std::cout.rdbuf(outputFile.rdbuf());
    uint8_t *data = nullptr;
    size_t npt_num, pt_dim;
    copy_aligned_data_from_file("/home/mr/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin.crop_nb_10000000", data,
                                npt_num, pt_dim);

    uint8_t *query = nullptr;
    size_t query_num, query_dim;
    copy_aligned_data_from_file("/home/mr/big-ann-benchmarks/data/yfcc100M/query.public.100K.u8bin", query, query_num,
                                query_dim);
    std::unordered_map<uint32_t, std::vector<uint32_t>> label_to_points;
    MetaData data_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/base.metadata.10M.spmat");
    MetaData query_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat");

    for (uint32_t point_id = 0; point_id < npt_num;
         ++point_id) // 倒排索引，遍历每个节点的label，将其插入到{label, ids}中
    {
        for (auto label : data_metadata._pts_to_labels[point_id])
        {
            label_to_points[label].emplace_back(point_id);
        }
    }
    uint32_t id = 91846 - 1;
    uint8_t *query_ = query + id * pt_dim;

    auto &labels = query_metadata._pts_to_labels[id];
    std::unordered_set<uint32_t> unique_ids;
    for (auto &label : labels)
    {
        unique_ids.insert(label_to_points[label].begin(), label_to_points[label].end());
    }
    double min_ = 10000000;
    for (uint32_t id_ : unique_ids)
    {
        double inner_product = 0.0;
        for (size_t j = 0; j < pt_dim; ++j)
        {
            // inner_product += data[j + id.first * pt_dim] * query[j + i * pt_dim];
            double diff = data[j + id_ * pt_dim] - query[j + id * pt_dim];
            inner_product += diff * diff;
        }
        std::cout << "#point: " << id_ + 1 << " to query " << id + 1 << " dist: " << std::sqrt(inner_product)
                  << std::endl;
        min_ = std::min(std::sqrt(inner_product), min_);
    }
    std::cout << "min dist: " << min_ << std::endl;
}

void analyse_discmp()
{
    float *dis_cmp = nullptr;
    size_t npt_num, pt_dim;
    copy_aligned_data_from_file<float>("/home/mr/diskann_index/diskann_1__1000_dists_float.bin", dis_cmp, npt_num,
                                       pt_dim);
    float zero_cmp = 0;
    for (uint32_t i = 0; i < npt_num; ++i)
    {
        int j;
        for (j = 0; j < pt_dim - 1; ++j)
        {
            // std::cout << dis_cmp[i * pt_dim + j] << ", ";
            zero_cmp += dis_cmp[i * pt_dim + j] == 0;
        }
        // std::cout << dis_cmp[i * pt_dim + j] << std::endl;
    }
    std::cout << "zero: " << zero_cmp / (npt_num * pt_dim);
}

void query_gthavenum_and_datahavenum()
{
    // std::ofstream outputFile("query_gthavenum_and_datahavenum.log");

    // 重定向输出到文件
    // std::streambuf *originalOutput = std::cout.rdbuf(); // 保存原始输出流
    // std::cout.rdbuf(outputFile.rdbuf());                // 将标准输出流重定向到文件

    std::ifstream input_file("query_gthavenum_and_gtnonum.log");
    std::string line;
    std::unordered_map<uint32_t, int> ids_have;
    std::regex pattern("query: (\\d+).*have (\\d+)");

    while (std::getline(input_file, line))
    {
        // 使用正则表达式搜索匹配
        std::smatch matches;

        if (std::regex_search(line, matches, pattern))
        {
            // 提取 "query:" 后面的数字
            std::string queryNumber = matches[1];

            // 提取 "have" 后面的数字
            std::string haveNumber = matches[2];
            ids_have.insert({std::stol(queryNumber) - 1, std::stoi(haveNumber)});
        }
    }

    uint8_t *data = nullptr;
    size_t npt_num, pt_dim;
    copy_aligned_data_from_file<uint8_t>("/home/mr/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin.crop_nb_10000000",
                                         data, npt_num, pt_dim);

    std::unordered_map<uint32_t, std::vector<uint32_t>> label_to_points;
    MetaData data_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/base.metadata.10M.spmat");
    MetaData query_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat");

    for (uint32_t point_id = 0; point_id < npt_num;
         ++point_id) // 倒排索引，遍历每个节点的label，将其插入到{label, ids}中
    {
        for (auto label : data_metadata._pts_to_labels[point_id])
        {
            label_to_points[label].emplace_back(point_id);
        }
    }

    for (auto &id_have : ids_have)
    {
        uint32_t id = id_have.first;
        int have = id_have.second;
        auto &labels = query_metadata._pts_to_labels[id];
        std::unordered_set<uint32_t> unique_ids;
        for (auto &label : labels)
        {
            unique_ids.insert(label_to_points[label].begin(), label_to_points[label].end());
        }
        std::cout << "#query: " << id + 1 << " gt have " << have << " data have: " << unique_ids.size() << std::endl;
    }
}

void query_gthavenum_and_gtnonum()
{
    std::ofstream outputFile("query_gthavenum_and_gtnonum.log");

    // 重定向输出到文件
    std::streambuf *originalOutput = std::cout.rdbuf(); // 保存原始输出流
    std::cout.rdbuf(outputFile.rdbuf());                // 将标准输出流重定向到文件

    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr; // 比较次数，即读取了多少个点才获取到这个结果的
    size_t gt_num, gt_dim;
    load_truthset("/home/mr/big-ann-benchmarks/data/yfcc100M/GT.public.ibin", gt_ids, gt_dists, gt_num, gt_dim);

    uint8_t *data = nullptr;
    size_t npt_num, pt_dim;
    copy_aligned_data_from_file<uint8_t>("/home/mr/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin.crop_nb_10000000",
                                         data, npt_num, pt_dim);
    std::unordered_map<uint32_t, std::vector<uint32_t>> label_to_points;
    MetaData data_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/base.metadata.10M.spmat");
    MetaData query_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat");

    uint32_t total_have = 0, total_no = 0;
    for (size_t i = 0; i < gt_num; ++i)
    {
        uint32_t have_ = 0, no_ = 0;
        std::cout << "#query: " << i + 1;

        for (int k = 0; k < gt_dim; ++k)
        {
            uint32_t id = gt_ids[i * gt_dim + k];
            std::vector<uint32_t> intersection;

            // 使用 std::set_intersection 计算交集
            std::set_intersection(data_metadata._pts_to_labels[id].begin(), data_metadata._pts_to_labels[id].end(),
                                  query_metadata._pts_to_labels[i].begin(), query_metadata._pts_to_labels[i].end(),
                                  std::back_inserter(intersection));
            if (intersection.size())
            {
                ++have_;
            }
            else
            {
                ++no_;
            }
        }
        total_have += have_;
        total_no += no_;
        std::cout << " have " << have_ << " no " << no_ << std::endl;
    }
    std::cout << "total have " << total_have << " total no " << total_no << std::endl;

    delete[] gt_ids, gt_dists;
}

void gtpoint_to_query_dist()
{
    std::ofstream outputFile("gtpoint_to_query_dist.log");

    // 重定向输出到文件
    std::streambuf *originalOutput = std::cout.rdbuf(); // 保存原始输出流
    std::cout.rdbuf(outputFile.rdbuf());                // 将标准输出流重定向到文件

    // 孤立点是否在groundtruth里：先收集有哪些point，然后看每个point在几个query的groundtruth里
    std::ifstream error_log("error.log");

    // 定义一个正则表达式，用于匹配井号后面的数字
    std::regex pattern("#(\\d+)");

    // 用于存储匹配到的数字的容器
    std::smatch matches;

    std::string line;

    std::unordered_map<uint32_t, uint32_t> ids;
    while (std::getline(error_log, line))
    {
        // 在每行中搜索匹配项
        if (regex_search(line, matches, pattern))
        {
            // matches[1] 包含匹配到的数字
            std::string number_str = matches[1].str();
            // std::cout << std::stol(number_str) - 1 << std::endl;
            ids.insert({std::stol(number_str) - 1, 0});
        }
    }

    error_log.close();

    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr; // 比较次数，即读取了多少个点才获取到这个结果的
    size_t gt_num, gt_dim;
    load_truthset("/home/mr/big-ann-benchmarks/data/yfcc100M/GT.public.ibin", gt_ids, gt_dists, gt_num, gt_dim);

    uint8_t *data = nullptr;
    size_t npt_num, pt_dim;
    copy_aligned_data_from_file<uint8_t>("/home/mr/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin.crop_nb_10000000",
                                         data, npt_num, pt_dim);

    uint8_t *query = nullptr;
    size_t query_num, query_dim;
    copy_aligned_data_from_file<uint8_t>("/home/mr/big-ann-benchmarks/data/yfcc100M/query.public.100K.u8bin", query,
                                         query_num, query_dim);

    MetaData data_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/base.metadata.10M.spmat");
    MetaData query_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat");
    for (size_t i = 0; i < gt_num; ++i)
    {
        for (int k = 0; k < gt_dim; ++k)
        {
            uint32_t id = gt_ids[i * gt_dim + k];
            double inner_product = 0.0;
            for (size_t j = 0; j < pt_dim; ++j)
            {
                // inner_product += data[j + id * pt_dim] * query[j + i * pt_dim];
                double diff = data[j + id * pt_dim] - query[j + i * pt_dim];
                inner_product += diff * diff;
            }
            std::cout << "#point: " << id + 1 << " to query " << i + 1 << " dist: " << std::sqrt(inner_product)
                      << std::endl;
            // std::cout << "#point: " << id + 1 << " to query " << i + 1 << " innerproduct: " << inner_product
            //           << std::endl;
        }
    }

    delete[] gt_ids, gt_dists;
}

void gtpoint_to_query_have_or_not()
{
    std::ofstream outputFile("gtpoint_to_query_have_or_not.log");

    // 重定向输出到文件
    std::streambuf *originalOutput = std::cout.rdbuf(); // 保存原始输出流
    std::cout.rdbuf(outputFile.rdbuf());                // 将标准输出流重定向到文件

    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr; // 比较次数，即读取了多少个点才获取到这个结果的
    size_t gt_num, gt_dim;
    load_truthset("/home/mr/big-ann-benchmarks/data/yfcc100M/GT.public.ibin", gt_ids, gt_dists, gt_num, gt_dim);

    MetaData data_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/base.metadata.10M.spmat");
    MetaData query_metadata("/home/mr/big-ann-benchmarks/data/yfcc100M/query.metadata.public.100K.spmat");

    for (size_t i = 0; i < gt_num; ++i)
    {
        for (int k = 0; k < gt_dim; ++k)
        {
            uint32_t id = gt_ids[i * gt_dim + k];
            std::vector<uint32_t> intersection;
            auto &labels = data_metadata._pts_to_labels[id];
            // 使用 std::set_intersection 计算交集
            std::set_intersection(data_metadata._pts_to_labels[id].begin(), data_metadata._pts_to_labels[id].end(),
                                  query_metadata._pts_to_labels[i].begin(), query_metadata._pts_to_labels[i].end(),
                                  std::back_inserter(intersection));

            if (intersection.size())
            {
                std::cout << "#point: " << id + 1 << " to query " << i + 1 << " have" << std::endl;
            }
            else
            {
                std::cout << "#point: " << id + 1 << " to query " << i + 1 << " no" << std::endl;
            }
        }
    }
    delete[] gt_ids, gt_dists;
}


void test()
{
    uint32_t t1 = 0b1101101000011;
    uint32_t t2 = 0b0110001010100;
    int res = __builtin_popcount(t1 ^ t2);
    std::cout << res;
}
int main()
{

    return 0;
}