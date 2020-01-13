#ifndef RANDOMFOREST_HPP_
#define RANDOMFOREST_HPP_

/**
 * @PUBLIC INCLUDES
 */
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "DecisionTree.hpp"

/**
 * @brief: This class contain's the decision tree's and bundles them so they can be used to classify (predict) values for certain data sets.
 * @author: Timo van den Hazel & Daan Sperber
 */
class RandomForest {
public:
	/**
	 * @brief: RandomForest constructor
	 * @param:
	 * - is_regression: contains booleans that tells whether a feature is labeled for classification or regression.
	 * - data: all the data within the data set (where the data is formatted in std::strings, integers or doubles).
	 * - number_of_trees: the number of uncorrelated tree's that will be generated to form the random forest.
	 * - sample_size: the amount of data samples used to generate the decision tree's, the lower this number the higher uncorrelation between the deicision tree's.
	 * - number_of_features: the number of features which are sampled and used to create one single decision tree.
	 * - minimum_number_of_leaves: minimum number of data rows required to cause another split.
	 * - depth: integer that describes the depth of the tree, a higher depth means an increased number of splits (which could result in overfitting!).
	 */
	RandomForest(std::vector<bool>& is_regression, std::vector<std::vector<int16_t>>& data, int64_t number_of_trees, int16_t number_of_features, int16_t sample_size, int16_t minimum_number_of_leaves, int16_t depth);

	/**
	 * @brief: RandomForest destructor
	 */
	virtual ~RandomForest();

	/**
	 * @brief: constructs the individuel decision tree's and append them to the vector named trees.
	 */
	bool create_forest();

	/**
	 * @brief generates a random subset of features based on the maximum number of features.
	 */
	void generate_random_features(std::vector<std::vector<std::int16_t>>& output);

	/**
	 * @brief generates a random subset of the data based on the sample size.
	 */
	void generate_sample_data(const std::vector<std::vector<int16_t>>& feature_trimmed_data, std::vector<std::vector<int16_t>>& output);


	int16_t predict(const std::vector<int16_t>& sample);


private:
	std::vector<bool> is_regression;
	std::vector<std::vector<int16_t>> data;
	int64_t number_of_trees;
	int16_t number_of_features;
	int16_t sample_size;
	int16_t minimum_number_of_leaves;
	int16_t depth;

	std::vector<DecisionTree*> trees;


	unsigned previous_seed = 0;
};

#endif /* RANDOMFOREST_HPP_ */
