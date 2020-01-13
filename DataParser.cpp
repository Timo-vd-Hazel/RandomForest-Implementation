#include "DataParser.hpp"

#include <iostream>
#include <sstream>
#include <fstream>

DataParser::DataParser(std::string filePath) : filePath(filePath){}
DataParser::~DataParser(){}



void DataParser::parse()
{
	std::vector<std::vector<std::string>> raw_data;

	loadFileData(filePath, raw_data);
	determineFeatureRegressOrClassifyLabels(raw_data, feature_labels);
	mapStringsToInt16_t(raw_data, parsed_data);

}

void DataParser::setFilePath(const std::string& filePath_)
{
	filePath = filePath_;
}


void DataParser::loadFileData(const std::string& filePath, std::vector<std::vector<std::string>>& output)
{
	std::ifstream file(filePath);

	while (file.good())
	{
		std::string line_string;
		getline(file, line_string);
		std::stringstream string_stream(line_string);
		std::vector<std::string> line;
		std::string word;

		while(getline(string_stream, word, ','))
		{
			line.push_back(word);
		}

		output.push_back(line);
	}
}



void DataParser::determineFeatureRegressOrClassifyLabels(const std::vector<std::vector<std::string>>& raw_data, std::vector<bool>& output)
{
	const std::string numbers = "0123456789 ";
	for(std::string raw_string : raw_data.at(0))
	{
		bool is_number = true;
		//If raw_string exclusively contains numbers => regression, else => classification
		for(char raw_char : raw_string)
		{
			if(numbers.find(raw_char) == std::string::npos)
			{
				is_number = false;
				break;
			}
		}
		output.push_back(is_number);
	}
}


void DataParser::mapStringsToInt16_t(const std::vector<std::vector<std::string>>& raw_data, std::vector<std::vector<int16_t>>& output)
{
    uint16_t ocurences = 0;

    for(uint16_t row = 0; row < raw_data.size(); ++row)
    {
        std::vector<int16_t> temp_v;

        for(uint16_t cell = 0; cell < raw_data.at(row).size(); ++cell)
        {
            const std::string value = raw_data.at(row).at(cell);

            if(feature_labels.at(cell))
            {
                temp_v.push_back(std::stoi(value));
            }
            else
            {
                auto mapped_value_ptr = classification_mapping_to_int.find(value);
                if(mapped_value_ptr != classification_mapping_to_int.end())
                {
                    temp_v.push_back(mapped_value_ptr->second);
                }
                else
                {
                	std::cout << "Mapping " << value << " to " << ocurences << std::endl;
                    classification_mapping_to_int.insert({value, ocurences});
                    temp_v.push_back(ocurences);
                    ++ocurences;
                }
            }
        }

        if(temp_v.size() > 0)
        {
            output.push_back(temp_v);
        }
    }
}

std::string DataParser::lookupMappingasString(uint64_t value)
{
	for(auto x : classification_mapping_to_int)
	{
		if(x.second == value)
		{
			return x.first;
		}
	}

	return "";
}
