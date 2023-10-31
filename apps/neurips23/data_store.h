#include <bits/stdc++.h>
class MetaData
{
  public:
    MetaData(std::string path)
    {
        uint32_t num = 0;
        std::ifstream f(path, std::ios::binary);
        f.read((char *)&nrow, sizeof(int64_t));
        f.read((char *)&ncol, sizeof(int64_t));
        f.read((char *)&nnz, sizeof(int64_t));

        std::cout << nrow << " " << ncol << " " << nnz << std::endl;

        indptr = new int64_t[nrow + 1];
        f.read((char *)indptr, (nrow + 1) * sizeof(int64_t));

        indices = new int32_t[nnz];
        f.read((char *)indices, nnz * sizeof(int32_t));
        _pts_to_labels.resize(nrow, std::vector<uint32_t>());
        // data = new float[nnz];
        // f.read((char *)data, nnz * sizeof(float));
        // for (int64_t i = 0; i < nrow; ++i)
        // {
        //     int64_t start = indptr[i];
        //     int64_t end = indptr[i + 1];
        //     if(start == end) {
        //         ++num;
        //     }
        // }
        // std::cout << num;
        uint64_t line_cnt = 0;
        for (int64_t i = 0; i < nrow; ++i)
        {
            int64_t start = indptr[i];
            int64_t end = indptr[i + 1];
            std::vector<uint32_t> lbls(0);
            for (int64_t j = start; j < end; ++j)
            {
                int64_t col = indices[j];
                lbls.push_back(col);
            }
            std::sort(lbls.begin(), lbls.end());
            _pts_to_labels[line_cnt] = lbls;
            ++line_cnt;
        }
        // for (int64_t i = 0; i < nnz; ++i)
        // {
        //     if (indices[i] < 0 || indices[i] >= ncol)
        //     {
        //         std::cerr << "Assertion failed: indices[i] < 0 || indices[i] >= ncol" << std::endl;
        //         exit(1);
        //     }
        // }
        f.close();
    }
    ~MetaData()
    {
        delete[] indptr, indices, data;
    }

  public:
    int64_t nrow;
    int64_t ncol;
    int64_t nnz; // 稀疏矩阵中的非零元素的总数
    const int64_t nb_M = 10;
    const int64_t nb = 10e6 * nb_M;
    const int64_t d = 192;
    const int64_t nq = 100000;
    int64_t *indptr;
    int32_t *indices;
    float *data;
    std::vector<std::vector<uint32_t>> _pts_to_labels;
};