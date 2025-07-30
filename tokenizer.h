#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <map>

class Tokenizer {
public: // <-- Must be public
    Tokenizer(const std::string& vocab_path);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);

    // This declaration must be exactly as written below, including "const"
    [[nodiscard]] int get_token_id(const std::string& word) const;

    std::map<std::string, int> word_to_id_;
    std::map<int, std::string> id_to_word_;
};

#endif //TOKENIZER_H