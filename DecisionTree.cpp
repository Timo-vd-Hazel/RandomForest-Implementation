#include "DecisionTree.hpp"


DecisionTree::DecisionTree(const DecisionTree& rhs) :
	is_regression(rhs.is_regression),
	data(rhs.data),
	number_of_features(rhs.number_of_features),
	minimum_number_of_leaves(rhs.minimum_number_of_leaves),
	depth(rhs.depth),
	current_splitting_feature(rhs.current_splitting_feature),
	list_of_child_nodes(rhs.list_of_child_nodes),
	reg_list_of_child_nodes(rhs.reg_list_of_child_nodes),
	is_leaf_node(rhs.is_leaf_node),
	count_of_classifications(rhs.count_of_classifications)
{

}

DecisionTree::DecisionTree(std::vector<bool>& is_regression, const std::vector<std::vector<int16_t>>& data, int16_t number_of_features, int16_t minimum_number_of_leaves, int16_t depth) : is_regression(is_regression), data(data), number_of_features(number_of_features), minimum_number_of_leaves(minimum_number_of_leaves), depth(depth), is_leaf_node(true)
{
	find_best_split();

	if(is_leaf_node)
	{

		for(uint64_t i = 0; i < data.size(); ++i)
		{
			uint16_t current_class = data.at(i).at(data.at(i).size() - 1);

			if(count_of_classifications.find(current_class) == count_of_classifications.end())
			{
				count_of_classifications.insert(std::make_pair(current_class, 1));
			} else {
				count_of_classifications[current_class] += 1;
			}
		}
	}
}

DecisionTree::~DecisionTree()
{
}

void DecisionTree::find_best_split()
{
	// The vector *list_of_classifications* contains all the classifications available (which are located in the final row of the dataset).
	std::vector<int16_t> list_of_classifications;
	for(uint16_t i = 0; i < data.size(); ++i) {
		list_of_classifications.push_back(data.at(i).at((uint16_t)data.at(i).size() - 1));
	}

	// The vector *unique_classes* contains a unique list of the classifications where one copy of each unique class is stored.
	std::vector<int16_t> unique_classes (list_of_classifications);
	sort( unique_classes.begin(), unique_classes.end() );
	unique_classes.erase( unique(unique_classes.begin(), unique_classes.end() ), unique_classes.end() );

	// This variable holds the gini for the entire data set, note that it is only correct once the for loop below has finished!
	double current_gini_dataset = 1;

	// Gini formula = 1 - P(class a)^2 - P(class b)^2 - ... - P(class n)^2
	// In essence this is all this for loop is doing, for all the unique classes calculate the probability, square it and subtract it from the *current_gini_dataset*
	for(uint64_t i = 0; i < unique_classes.size(); ++i)
	{
		double probability = std::count(list_of_classifications.begin(), list_of_classifications.end(), unique_classes.at(i)) / (double)list_of_classifications.size();
		current_gini_dataset -= pow(probability, 2.0);
	}

	// If the gini is extremely low (we decided on 0.05 for now) it is almost in any case a leaf node because all the classes are of the same value.
	// When this occurs we don't have to continue and look for a split because we declare it as a leaf node. We then return the function.
	if(current_gini_dataset <= 0.05) {
		std::cout << "There is no need to split any further, we've arrived at a leaf node!" << std::endl;
		std::cout << "" << std::endl;
		is_leaf_node = true;
		return;
	}

	// (semi-)Global variables to keep all the information stored we are going to need troughout the algorithm.
	double best_possible_gain = 0;
	current_splitting_feature = 0;
	std::vector<int16_t> if_classification_unique_feature_variables;

	for(uint16_t i = 0; i < number_of_features; ++i)
	{
		// We check whether the current feature has continuous values or not, if the statement is false the values are labeled.
		if(is_regression.at(i) == false)
		{
			// The vector *feature_variables* contains all the variables of this specific feature. No elements are left out meaning there probably are duplicates within this vector.
			// The vector *unique_feature_variables* contains all the unique elements of a feature. For example:
			// The feature weather contains "sunny", "rainy" and "windy" this list will contain those three elements.
			std::vector<int16_t> feature_variables;
			for(uint16_t j = 0; j < data.size(); ++j) {
				feature_variables.push_back(data.at(j).at(i));
			}

			// The vector *unique_feature_variables* contains a unique list of all the different variables in the current dataset.
			std::vector<int16_t> unique_feature_variables(feature_variables);
			sort( unique_feature_variables.begin(), unique_feature_variables.end() );
			unique_feature_variables.erase( unique(unique_feature_variables.begin(), unique_feature_variables.end() ), unique_feature_variables.end() );

			std::vector<double> gini_feature_vector;
			std::vector<uint16_t> sum_of_all_values;

			for(uint64_t j = 0; j < unique_feature_variables.size(); ++j)
			{
				int16_t number_of_classes[unique_classes.size()] = {0};

				for(uint64_t all_f = 0; all_f < feature_variables.size(); ++all_f)
				{
					if(feature_variables.at(all_f) == unique_feature_variables.at(j))
					{
						++number_of_classes[list_of_classifications.at(all_f)];
					}
				}

				double gini = 1.0;
				uint16_t sum = 0;

				for(uint64_t c = 0; c < unique_classes.size(); ++c)
				{
					sum += number_of_classes[c];
				}
				sum_of_all_values.push_back(sum);
				for(uint64_t g_f = 0; g_f < unique_classes.size(); ++g_f)
				{
					if(sum != 0)
					{
						double probability = number_of_classes[g_f] / (double) sum;
						gini -= pow(probability, 2.0);
					} else {
					}
				}

				gini_feature_vector.push_back(gini);
			}

			// Calculate the gain of splitting on this feature:
			// Gain = 1 - P(class a) * Gini(class a) - P(class b) * Gini(class b) - ... - P(class n) * Gini(class n)
			double current_gain = current_gini_dataset;
			for(uint64_t j = 0; j < gini_feature_vector.size(); ++j)
			{
				double Probability = sum_of_all_values.at(j) / (double)feature_variables.size();
				double Gini_coefficient = Probability * (double)gini_feature_vector.at(j);
				current_gain -= Gini_coefficient;
			}

			// Check whether this split is the best possible split by checking if the gain is higher than any previous splits
			// If this is true, set the best possible split to the current split and add the current feature id
			if(sum_of_all_values.size() != 1 && current_gain > best_possible_gain)
			{
				best_possible_gain = current_gain;
				current_splitting_feature = i;

				if_classification_unique_feature_variables.clear();
				for(uint64_t k = 0; k < unique_feature_variables.size(); ++k)
				{
					if_classification_unique_feature_variables.push_back(unique_feature_variables.at(k));
				}
			}
		}


		// If the feature has continuous variables we can't use the algorithm above, instead use the one below
		else
		{
			// The vector *feature_variables* contains all the variables of this specific feature. No elements are left out meaning there probably are duplicates within this vector.
			std::vector<int16_t> feature_variables;
			for(uint16_t j = 0; j < data.size(); ++j) {
				feature_variables.push_back(data.at(j).at(i));
			}

			std::map<uint16_t, uint16_t> g_number_of_classes;
			std::map<uint16_t, uint16_t> l_number_of_classes;

			for(uint64_t all_c = 0; all_c < unique_classes.size(); ++all_c)
			{
				g_number_of_classes.insert(std::make_pair(unique_classes.at(all_c), 0));
				l_number_of_classes.insert(std::make_pair(unique_classes.at(all_c), 0));
			}

			for(uint64_t all_f = 0; all_f < list_of_classifications.size(); ++all_f)
			{
				g_number_of_classes[list_of_classifications.at(all_f)] += 1;
			}

			// We then determine the lowest value and remove (basically subtract) the classifications of that lowest value(s)
			// Afterwards we add those classification values to the *lesser_than_split*
			int16_t lowest_value = std::numeric_limits<int16_t>::min();
			for(uint64_t j = 0; j <= feature_variables.size() - 1; ++j)
			{
				// This function calculates the second lowest value which we use to iterate trough the data from lowest to highest value.
				int16_t current_lowest_var = calculate_second_lowest_value(feature_variables, lowest_value);

				// Because there can be multiple variables of the same value (e.g. 2 occurs multiple times)
				// We have to find all instances of 2 and subtract / add all classifications to the other table.
				for(uint64_t k = 0; k < feature_variables.size(); ++k)
				{
					if(current_lowest_var == feature_variables.at(k))
					{
						g_number_of_classes[list_of_classifications.at(k)] -= 1;
						l_number_of_classes[list_of_classifications.at(k)] += 1;
					}
				}

				// These variables simply count the total number of data samples in the splits.
				// These variables are useful for calculating the Gain and the Gini.
				int16_t g_total_classifications = 0;
				int16_t l_total_classifications = 0;
				for(auto const& c : g_number_of_classes)
				{
					g_total_classifications += c.second;
				}
				for(auto const& c : l_number_of_classes)
				{
					l_total_classifications += c.second;
				}


				// Gini formula = 1 - P(class a)^2 - P(class b)^2 - ... - P(class n)^2
				// In essence this is all this for loop is doing, for all the unique classes calculate the probability, square it and subtract it from the correct gini (either g or l).
				double g_gini = 1;
				double l_gini = 1;

				for(auto const& c : g_number_of_classes)
				{
					if(g_total_classifications != 0 ) {
						double g_probability = c.second / (double) g_total_classifications;
						g_gini -= pow(g_probability, 2.0);
					}
				}

				for(auto const& c : l_number_of_classes)
				{
					if(l_total_classifications != 0 ) {
						double l_probability = c.second / (double) l_total_classifications;
						l_gini -= pow(l_probability, 2.0);
					}
				}

				// Calculate the gain of splitting on this specific variable for this feature:
				// Gain = 1 - P(class a) * Gini(class a) - P(class b) * Gini(class b)
				double current_gain = current_gini_dataset - ((g_total_classifications / (double)list_of_classifications.size()) * g_gini) - ((l_total_classifications / (double)list_of_classifications.size()) * l_gini);


				// Check whether this split is the best possible split by checking if the gain is higher than any previous splits
				// If this is true, set the best possible split to the current split and add the current feature id
				if(current_gain > best_possible_gain)
				{
					best_possible_gain = current_gain;
					current_splitting_feature = i;
					if_regression_best_value_to_split_on = current_lowest_var;
				}

				// Because we have to check iteratively from lowest to highest value we have to set the current lowest value to the new lowest value so the loops won't break.
				lowest_value = current_lowest_var;
			}
		}
	}


	if(best_possible_gain <= 0.000005)
	{
		std::cout << "Unfortunately, there is no good split possible so we declare this as a leaf node!" << std::endl;
		std::cout << "The best possible gain would've been: " << best_possible_gain << std::endl;
		std::cout << "" << std::endl;
		is_leaf_node = true;
		return;
	}

	unique_classes = std::vector<int16_t>();
	list_of_classifications = std::vector<int16_t>();

	is_leaf_node = false;

	// Create new child nodes:
	if(depth > 0)
	{
		if(is_regression.at(current_splitting_feature) == true)
		{
			std::vector<std::vector<int16_t>> g_temp_data;
			std::vector<std::vector<int16_t>> l_temp_data;

			for(uint64_t j = 0; j < data.size(); ++j)
			{
				if(data.at(j).at(current_splitting_feature) <= if_regression_best_value_to_split_on)
				{
					l_temp_data.push_back(data.at(j));
				} else {
					g_temp_data.push_back(data.at(j));
				}
			}

			DecisionTree t1(is_regression, g_temp_data, number_of_features, minimum_number_of_leaves, depth - 1);
			DecisionTree t2(is_regression, l_temp_data, number_of_features, minimum_number_of_leaves, depth - 1);
			reg_list_of_child_nodes.push_back(t1);
			reg_list_of_child_nodes.push_back(t2);
		} else {
			for(uint64_t i = 0; i < if_classification_unique_feature_variables.size(); ++i)
			{
				std::vector<std::vector<int16_t>> temp_data;
				for(uint64_t j = 0; j < data.size(); ++j)
				{
					if(data.at(j).at(current_splitting_feature) == if_classification_unique_feature_variables.at(i))
					{
						temp_data.push_back(data.at(j));
					}
				}

				list_of_child_nodes.insert({if_classification_unique_feature_variables.at(i), new DecisionTree(is_regression, temp_data, number_of_features, minimum_number_of_leaves, depth - 1)});
			}
		}
	}

}

int16_t DecisionTree::predict(const std::vector<int16_t>& sample_to_predict)
{
	if(is_leaf_node)
	{
		uint16_t max_n = 0;
		uint16_t classi = 0;

		for(auto g : count_of_classifications)
		{
			if(g.second > max_n)
			{
				max_n = g.second;
				classi = g.first;
			}
		}

		return classi;


	} else {
		if(reg_list_of_child_nodes.size() == 0)
		{

			if(list_of_child_nodes[sample_to_predict.at(current_splitting_feature)] != 0)
			{
				return list_of_child_nodes[sample_to_predict.at(current_splitting_feature)]->predict(sample_to_predict);
			} else {
				std::cout << "Unfortunately this tree doesn't know how to continue from here as it doesn't know it's value!" << std::endl;
				throw 69;
			}
		} else {
			if(sample_to_predict.at(current_splitting_feature) < if_regression_best_value_to_split_on)
			{
				return reg_list_of_child_nodes.at(1).predict(sample_to_predict);
			} else {
				return reg_list_of_child_nodes.at(0).predict(sample_to_predict);
			}
		}
	}
}

int16_t DecisionTree::calculate_second_lowest_value(const std::vector<int16_t>& v, int16_t lowest_value)
{
	int16_t second_lowest_value = std::numeric_limits<int16_t>::max();
	for(uint64_t i = 0; i < v.size(); ++i)
	{
		if(second_lowest_value > v.at(i) && lowest_value < v.at(i)) {second_lowest_value = v.at(i);}
	}

	return second_lowest_value;
}
