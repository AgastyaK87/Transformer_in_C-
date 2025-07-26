#include <iostream>
#include "tokenizer.h"

int main() {
    try {
        // 1. Initialize the tokenizer with the vocab file
        Tokenizer tokenizer("vocab.txt");

        // 2. Define a test sentence
        std::string test_prompt = "find all . log files";

        // 3. Encode the sentence
        std::vector<int> token_ids = tokenizer.encode(test_prompt);

        std::cout << "\nOriginal prompt: '" << test_prompt << "'" << std::endl;
        std::cout << "Encoded IDs: ";
        for (int id : token_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // 4. Decode the IDs back to a string
        std::string decoded_prompt = tokenizer.decode(token_ids);
        std::cout << "Decoded prompt: '" << decoded_prompt << "'" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}