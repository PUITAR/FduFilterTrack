#include <bits/stdc++.h>
inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim)
{
    std::ifstream reader(bin_file, std::ios::binary);
    std::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;

    int npts_i32 = 100 * 1000, dim_i32 = 192;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    std::cout << "npts_i32: " << npts_i32 << " dim_i32:" << dim_i32 << std::endl;
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "... " << std::endl;

    int truthset_type = 1; // 1 means truthset has ids and distances, 2 means
                           // only ids, -1 is error

    ids = new uint32_t[npts * dim];
    reader.read((char *)ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1)
    {
        dists = new float[npts * dim];
        reader.read((char *)dists, npts * dim * sizeof(float));
    }
}

template <typename T>
inline void copy_aligned_data_from_file(const char *bin_file, T *&data, size_t &npts, size_t &dim)
{

    std::ifstream reader;
    reader.exceptions(std::ios::badbit | std::ios::failbit);
    reader.open(bin_file, std::ios::binary);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;
    data = new T[npts * dim * sizeof(T)];
    reader.read((char *)data, npts * dim * sizeof(T));
}