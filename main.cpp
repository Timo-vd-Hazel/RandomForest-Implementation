#include "DecisionTree.hpp"
#include "RandomForest.hpp"
#include "DataParser.hpp"

#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
	DataParser data_parser;
	data_parser.parse();

	std::ofstream outputFile;
	outputFile.open("output.txt");

	const uint16_t n_runs = 60;

	outputFile << "n_trees, n_features, n_samples, n_correct, n_incorrect\n";
	for(uint16_t n_t = 1; n_t < n_runs; n_t+=1)
	{
		for(uint16_t n_f = 1; n_f < 15; n_f += 2)
		{
			for(uint16_t n_s = 1; n_s < n_runs * 2; n_s += 20)
			{
				RandomForest r(data_parser.feature_labels, data_parser.parsed_data, n_t, n_f, n_s, 0, 10000);

				std::vector<std::vector<std::string>> test_data_string;
				std::vector<std::vector<int16_t>> test_data;
				std::string filePath = "stripped_test_data.txt";

				data_parser.loadFileData(filePath, test_data_string);
				data_parser.mapStringsToInt16_t(test_data_string, test_data);


				uint64_t P = 0,N = 0;
				for(auto row : test_data)
				{
					int16_t score = r.predict(row);
					(score == row.at(row.size()-1)) ? ++P : ++N;
				}

				outputFile << n_t << ", " << n_f << ", " << n_s << ", " << P << ", " << N << std::endl;
			}
		}
	}

	outputFile.close();
	return 0;
}
