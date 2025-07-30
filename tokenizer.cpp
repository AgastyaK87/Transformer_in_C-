#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>


Tokenizer::Tokenizer(const std::string& vocab_path)
{
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open())
    {
        throw std::runtime_error("Could not open vocabulary file" + vocab_path);
    }

    std::string line;
    int id = 0;
    while (std::getline(vocab_file, line))
    {
        //remove potential carraige return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        word_to_id_[line] = id;
        id_to_word_[id] = line;
        id++;
    }
    std::cout << "Tokenizer loaded with " << word_to_id_.size() << " tokens." << std::endl;
}

std::vector<int> Tokenizer::encode(const std::string& text)
{
    std::vector<int> ids;
    std::stringstream ss(text);
    std::string word;

    //cls token at beginning
    ids.push_back(word_to_id_["[CLS]"]);

    while (ss>>word)
    {
        if (word_to_id_.count(word))
        {
            ids.push_back(word_to_id_[word]);
        } else
        {
            ids.push_back(word_to_id_["[UNK]"]);
        }
    }

    ids.push_back(word_to_id_["[SEP]"]);

    return ids;
}

std::string Tokenizer::decode(const std::vector<int> & ids)
{
    std::string text = "";
    for (int id : ids)
    {
        if (id_to_word_.count(id))
        {
            if (id_to_word_[id] != "[PAD]" && id_to_word_[id] != "[CLS]" && id_to_word_[id] != "[SEP]") {
                text += id_to_word_[id] + " ";
            }
        }
    }
    if (!text.empty()) {
        text.pop_back();
    }
    return text;

}

int Tokenizer::get_token_id(const std::string& word) const {
    if (word_to_id_.count(word)) {
        return word_to_id_.at(word);
    }
    return word_to_id_.at("[UNK]");
}
