#ifndef DATAPARSER_HPP_
#define DATAPARSER_HPP_


#include <string>
#include <vector>
#include <map>

class DataParser {
public:
	DataParser(std::string filePath = "stripped_data.txt");
	virtual ~DataParser();

	void parse();
	void setFilePath(const std::string& filePath_);
	void loadFileData(const std::string& filePath, std::vector<std::vector<std::string>>& output);
	void mapStringsToInt16_t(const std::vector<std::vector< std::string>>& raw_data, std::vector<std::vector<int16_t>>& output);
	std::string lookupMappingasString(uint64_t value);


	/*
	 * True => regression, False => Classification
	 */
	std::vector<bool> feature_labels;
	std::vector<std::vector<int16_t>> parsed_data;

private:
	std::string filePath;
	std::map<const std::string, uint64_t> classification_mapping_to_int;


	void determineFeatureRegressOrClassifyLabels(const std::vector<std::vector< std::string>>& raw_data, std::vector<bool>& output) ;


};


#endif /* DATAPARSER_HPP_ */
