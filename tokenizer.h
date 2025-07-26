
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <map>


class Tokenizer
{
public:
    Tokenizer(const std::string& vocab_path);
    //text prompt = vector of token ids
    std::vector<int> encode(const std::string& text);

    //vector of token ids = text string
    std::string decode(const std::vector<int> & ids);

private:

    //lookup tables
    std::map<std::string, int> word_to_id_;
    std::map<int, std::string> id_to_word_;
};


#endif //TOKENIZER_H
