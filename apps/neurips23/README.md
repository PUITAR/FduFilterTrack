debug的时候，要这样编译：
cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j  这样才能用gdb访问到其他cpp文件
正式运行的时候，要这样编译（否则会很慢）：
cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j

# 以下所有命令都在 build 文件夹下执行
mkdir ../results/final

# 构建索引
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
nohup apps/build_memory_index -R 32 -L 100 --alpha 1.2 --data_type uint8 --dist_fn l2 --label_file ../data/yfcc100M/base.metadata.10M.spmat --index_path_prefix ../results/final/index --data_path ../data/yfcc100M/base.10M.u8bin.crop_nb_10000000 --num_threads 16 >> ../results/final/build.log 

# 执行查询
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
nohup apps/search_memory_index --search_list 100 --threshold_1 20000 --threshold_2 40000 --data_type uint8 --dist_fn l2 --index_path_prefix ../results/final/index --result_path ../results/final/output --query_file ../data/yfcc100M/query.public.100K.u8bin --recall_at 10 --query_filters_file ../data/yfcc100M/query.metadata.public.100K.spmat --label_type uint --gt_file ../data/yfcc100M/GT.public.ibin --num_threads 16 >> ../results/final/search.log 2>>../results/final/error.log

# vtune分析
cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j
/var/lib/docker/ANNS/FilterDiskann/apps/search_memory_index
--search_list 100 --threshold_1 20000 --threshold_2 40000 --data_type uint8 --dist_fn l2 --index_path_prefix /var/lib/docker/ANNS/FilterDiskann/results/final/index --result_path /var/lib/docker/ANNS/FilterDiskann/results/final/output --query_file /var/lib/docker/ANNS/FilterDiskann/data/yfcc100M/query.public.100K.u8bin --recall_at 10 --query_filters_file /var/lib/docker/ANNS/FilterDiskann/data/yfcc100M/query.metadata.public.100K.spmat --label_type uint --gt_file /var/lib/docker/ANNS/FilterDiskann/data/yfcc100M/GT.public.ibin --num_threads 16

# 构建索引测试
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
nohup apps/build_memory_index -R 32 -L 100 --alpha 1.2 --data_type uint8 --dist_fn l2 --label_file ../data/yfcc100M/base.metadata.10M.spmat --index_path_prefix ../results/tmp/index --data_path ../data/yfcc100M/base.10M.u8bin.crop_nb_10000000 --num_threads 80 >> ../results/tmp/build.log 

# 执行查询测试
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
nohup apps/search_memory_index --search_list 100 --threshold_1 20000 --threshold_2 40000 --data_type uint8 --dist_fn l2 --index_path_prefix ../results/tmp/index --result_path ../results/tmp/output --query_file ../data/yfcc100M/query.public.100K.u8bin --recall_at 10 --query_filters_file ../data/yfcc100M/query.metadata.public.100K.spmat --label_type uint --gt_file ../data/yfcc100M/GT.public.ibin --num_threads 16 >> ../results/tmp/search.log 2>>../results/tmp/error.log
