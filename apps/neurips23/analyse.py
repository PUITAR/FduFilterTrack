import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import Counter


class DataSet:
    def __init__(self, data_fn, meta_fn):
        self.data_fn = data_fn
        self.meta_fn = meta_fn
        self.data = self.get_dateset()
        self.meta = self.get_dataset_metadata()
        self.n = 0
        self.d = 0

    def get_dataset_metadata(self):
        with open(self.meta_fn, "rb") as f:
            sizes = np.fromfile(f, dtype="int64", count=3)
            nrow, ncol, nnz = sizes
            indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
            assert nnz == indptr[-1]
            indices = np.fromfile(f, dtype="int32", count=nnz)
            assert np.all(indices >= 0) and np.all(indices < ncol)
            data = np.fromfile(f, dtype="float32", count=nnz)
            return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))

    def get_dateset(self):
        n, d = map(int, np.fromfile(self.data_fn, dtype="uint32", count=2))
        self.n = n
        self.d = d
        f = open(self.data_fn, "rb")
        f.seek(4 + 4)
        return np.fromfile(f, dtype="uint8", count=n * d).reshape(n, d)

    def csr_get_row_indices(self, i):
        """get the non-0 column indices for row i in matrix m"""
        m = self.meta
        return m.indices[m.indptr[i] : m.indptr[i + 1]]

    # 统计label分布（tag为1的有多少个point)；
    def labels_to_points(self, draw=False, log=True):
        _res = []
        m = self.meta
        pic_name = "labels_to_points"
        for i in range(m.shape[0]):
            labels = self.csr_get_row_indices(i)
            _res.extend(labels)

        if draw:
            res = Counter(_res)
            labels = list(res.keys())
            num_points = list(res.values())
            if log:
                num_points = np.log10(num_points)
                plt.yscale('log')
                pic_name += "_log"
            
            # 创建条形图
            plt.vlines(
                x=labels,
                ymin=0,
                ymax=num_points,
                colors="lightblue",
                linestyles="solid",
            )
            pic_name += ".png"
            # 添加标签和标题
            plt.xlabel("Label")
            plt.ylabel("Number of vectors")
            # 添加图例
            plt.legend()
            plt.savefig(pic_name)
            plt.clf()

    # 统计一下label数量的种类数（10个label的point有多少个...）
    def labels_num_to_points(self, draw=True, log=True):
        _res = []
        m = self.meta
        pic_name = "labels_num_to_points"
        for i in range(m.shape[0]):
            labels = self.csr_get_row_indices(i)
            num = len(labels)
            _res.append(num)

        if draw:
            res = Counter(_res)
            labels_num = list(res.keys())
            num_points = list(res.values())
            if log:
                num_points = np.log10(num_points)
                plt.yscale('log')
                pic_name += "_log"
            
            plt.vlines(
                x=labels_num,
                ymin=0,
                ymax=num_points,
                colors="lightblue",
                linestyles="solid",
            )

            # 添加标签和标题
            plt.xlabel("Number of Label")
            plt.ylabel("Number of vectors")

            # 添加图例
            pic_name += ".png"
            plt.legend()
            plt.savefig(pic_name)
            plt.clf()
            # plt.hist(labels_num, weights=num_points, edgecolor="k")

            # # 添加标签和标题
            # plt.xlabel("Number of Label")
            # plt.ylabel("Number of vectors")

            # # 添加图例
            # plt.legend()
            # plt.savefig("test.png")
            # plt.clf()

    # 统计一下label组合的种类数（{1,3}的有多少个...)；
    def labels_types_to_points(self, draw=True, log=True):
        _res = []
        m = self.meta
        pic_name = "labels_types_to_points"
        for i in range(m.shape[0]):
            labels = self.csr_get_row_indices(i)
            tuple_labels = tuple(labels)
            _res.append(tuple_labels)

        if draw:
            res = Counter(_res)
            # 提取组合和对应的数量
            combinations = list(res.keys())
            num_points = list(res.values())
            if log:
                num_points = np.log10(num_points)
                plt.yscale('log')
                pic_name += "_log"
                
            
            plt.vlines(
                x=np.arange(len(combinations)),
                ymin=0,
                ymax=num_points,
                colors="lightblue",
                linestyles="solid",
            )
            pic_name += ".png"
            # 添加标签和标题
            plt.xlabel("Kinds of Label")
            plt.ylabel("Number of vectors")

            # 添加图例
            plt.legend()
            plt.savefig(pic_name)
            plt.clf()

    def embedding_label(self):
        m = self.meta
        res = []
        for i in range(m.shape[0]):
            labels = self.csr_get_row_indices(i)
            if len(labels) == 0:
                labels = np.append(labels, 0)
            binary_numbers = np.array([bin(label)[2:] for label in labels], dtype=int)
            result = np.bitwise_or.reduce(binary_numbers)
            res.append(result)
        
        res = np.array(res)
        new_data = np.c_[self.data, res]
        print(new_data.shape)

if __name__ == "__main__":
    data_set = DataSet(
        "/home/mr/big-ann-benchmarks/data/yfcc100M/base.10M.u8bin.crop_nb_10000000",
        "/home/mr/big-ann-benchmarks/data/yfcc100M/base.metadata.10M.spmat",
    )
    # data_set.labels_to_points()
    # data_set.labels_num_to_points()
    data_set.labels_types_to_points()

    # data_set.embedding_label()
