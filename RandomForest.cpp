#include "RandomForest.hpp"

RandomForest::RandomForest(std::vector<bool>& is_regression, std::vector<std::vector<int16_t>>& data, int64_t number_of_trees, int16_t number_of_features, int16_t sample_size, int16_t minimum_number_of_leaves, int16_t depth) : is_regression(is_regression), data(data), number_of_trees(number_of_trees), number_of_features(number_of_features), sample_size(sample_size), minimum_number_of_leaves(minimum_number_of_leaves), depth(depth)
{
	try{
		create_forest();
	} catch(std::string& exception) {
		std::cout << "An error has occured while creating the decision tree's: " << exception << std::endl;
	}
}

RandomForest::~RandomForest()
{

}

bool RandomForest::create_forest()
{

	for(int64_t i = 0; i < number_of_trees; ++i)
	{
		std::vector<std::vector<std::int16_t>> subset_of_features = data;
		generate_random_features(subset_of_features);

		std::vector<std::vector<int16_t>> subset_of_data;
		generate_sample_data(subset_of_features, subset_of_data);

		trees.push_back(new DecisionTree(is_regression, subset_of_data, subset_of_data.at(0).size() - 1, minimum_number_of_leaves, depth));
	}

	return true;
}

int16_t RandomForest::predict(const std::vector<int16_t>& sample)
{
    std::map<int16_t, uint64_t> counters;

    for(int64_t i = 0; i < number_of_trees; ++i)
    {
        int16_t x;
        try
        {
            x = trees.at(i)->predict(sample);
        }
        catch(int e)
        {
            continue;
        }

        if(counters.find(x) != counters.end())
        {
            ++counters.at(x);
        }
        else
        {
            counters.insert({x,1});
        }
    }

    uint64_t max_classifier = 0;
    uint64_t max_val = 0;
    uint64_t sum = 0;

    for(auto c : counters)
	{
		if(c.second > max_val)
		{
			max_val = c.second;
			max_classifier = c.first;
		}
		sum += c.second;
	}

    return max_classifier;
}

void RandomForest::generate_random_features(std::vector<std::vector<std::int16_t>>& output)
{
	// This vector *features_indices* contains all indices (starting from 0) equal to the size of the feature set in integer format.
	std::vector<uint16_t> features_indices(output.at(0).size() - 1);
    std::iota(features_indices.begin(), features_indices.end(), 0);


    // The random device and generator are used to randomize the vector with indices (using std::shuffle).
    // This results in a randomized list of indices which we can use to select our features.

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    while(seed == previous_seed)
    {
    	seed = std::chrono::system_clock::now().time_since_epoch().count();
    }

    previous_seed = seed;

    std::shuffle(features_indices.begin(), features_indices.end(), std::default_random_engine(seed));


    // In this for loop the indices are read and used to delete the corresponding feature from the set of features.
    // The function stops whenever the maximum number of random features are reached.
    for(uint64_t i = 0; i < (output.at(0).size() - 1 - uint64_t(number_of_features)); ++i)
    {
    	for(uint64_t j = 0; j < output.size(); ++j)
    	{
    		output.at(j).at(features_indices.at(i)) = 0;
    	}
    }
}

void RandomForest::generate_sample_data(const std::vector<std::vector<int16_t>>& feature_trimmed_data, std::vector<std::vector<int16_t>>& output)
{
	// The vector *subset_of_data* is the output vector where all randomly selected samples will be stored and returned.
	;

	// This vector *data_indices* contains all indices (starting from 0) equal to the size of the sample set in integer format.
	std::vector<uint16_t> data_indices(feature_trimmed_data.size());
    std::iota(data_indices.begin(), data_indices.end(), 0);

    // The random device and generator are used to randomize the vector with indices (using std::shuffle).
    // This results in a randomized list of indices which we can use to select our samples.
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    while(seed == previous_seed)
	{
		seed = std::chrono::system_clock::now().time_since_epoch().count();
	}

    previous_seed = seed;

    std::shuffle(data_indices.begin(), data_indices.end(), std::default_random_engine(seed));

    // In this for loop the indices are read and used to take the corresponding sample from the set of samples.
    // The function stops whenever the maximum number of samples is reached.
    for(uint64_t i = 0; i < uint64_t(sample_size); ++i)
    {
    	output.push_back(feature_trimmed_data.at(data_indices.at(i)));
    }
}
