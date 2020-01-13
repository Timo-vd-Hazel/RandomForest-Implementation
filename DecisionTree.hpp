#ifndef DECISIONTREE_HPP_
#define DECISIONTREE_HPP_

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <limits>
#include <array>


/**
 * @brief: This class contain's a single decision tree capable of classifying elements within a dataset.
 * @author: Timo van den Hazel & Daan Sperber
 */
class DecisionTree {
public:
	/**
	 * @brief: DecisionTree constructor
	 * @param:
	 * - is_regression: contains booleans that tells whether a feature is labeled for classification or regression.
	 * - data: all the data within the data set (where the data is formatted in std::strings, integers or doubles).
	 * - number_of_features: the number of features which are sampled and used to create one single decision tree.
	 * - minimum_number_of_leaves: minimum number of data rows required to cause another split.
	 * - depth: integer that describes the depth of the tree, a higher depth means an increased number of splits (which could result in overfitting!).
	 */
	DecisionTree(std::vector<bool>& is_regression, const std::vector<std::vector<int16_t>>& data, int16_t number_of_features, int16_t minimum_number_of_leaves, int16_t depth);

	DecisionTree(const DecisionTree& rhs);

	/**
	 * @brief: DecisionTree destructor
	 */
	virtual ~DecisionTree();

	/**
	 * @brief: ?!?!
	 */
	void find_best_split();

	/**
	 * @brief: ?!?!
	 */
	int16_t calculate_second_lowest_value(const std::vector<int16_t>& v, int16_t lowest_value);

	/**
	 *
	 */
	int16_t predict(const std::vector<int16_t>& sample_to_predict);




private:


	std::vector<bool> is_regression;
	std::vector<std::vector<int16_t>> data;
	int16_t number_of_features;
	int16_t minimum_number_of_leaves;
	int16_t depth;

	int16_t current_splitting_feature;

	std::map<int16_t, DecisionTree*> list_of_child_nodes;

	std::vector<DecisionTree> reg_list_of_child_nodes;
	int16_t if_regression_best_value_to_split_on = 0;

	bool is_leaf_node;
	std::map<uint16_t, uint16_t> count_of_classifications;
};

#endif /* DECISIONTREE_HPP_ */
