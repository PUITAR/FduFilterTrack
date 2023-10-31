# 找不到K个point的原因

算法是从对应的label_center（假设为start节点）出发进行search的，但是存在一个问题，只有start与query有共同label，start的neighbor point与query没有共同label，或者最多只有一两个neighbor point与query有共同label，如此一来影响了QPS和recall

假设query_label = {1, 3}, start的label = {1, 10, 19, 30, 74}。在构建索引的时候，起初start的邻居可能都是label == 1的节点，但是到后面可能会被其他label的节点所替换掉，这样就导致start的邻居节点的label里没有{1}，所以query无法找到更多的point来扩展候选集，只有一个start节点。



# 想法

1.保证neighor point的label多样性-->尽可能保证邻居节点数量==节点的label数量
2.将label进行组合，降低distinct label
3.label_center的选择：
	3.1 默认从包含label的所有节点中遍历25个节点，从这25个中挑选一个作为center --> idea:增加遍历节点数量？
	3.2 idea:挑选center时，尽量挑选label较少的，在构建索引时，也以该原则寻找邻居（因为在搜索过程中，如果搜索到的节点的label太多，那么其neighbor point与query无关的可能性也就越大）

4.build filter-index，但是修改search算法

5.build index，但是搜索的时候要搜索到label的，直到满足K个

6.先聚类，再filter建图

7.把label嵌入至向量(AjiaoB / AbingB)

8.以label数量为种类建图，数量多的先建图