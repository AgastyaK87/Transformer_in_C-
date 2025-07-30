#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include "tokenizer.h"
#include "input_layer.h"
#include "encoder.h"
#include "decoder.h"
#include "utils.h"

// The final linear layer that comes after the decoder
class FinalLinearLayer {
public:
    FinalLinearLayer(int d_model, int vocab_size, std::ifstream& weight_file) {
        weights_ = Eigen::MatrixXf(d_model, vocab_size);
        load_matrix(weights_, weight_file);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& x) {
        return x * weights_;
    }
private:
    Eigen::MatrixXf weights_;
};


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"<your prompt>\"" << std::endl;
        return 1;
    }
    std::string user_prompt = argv[1];

    // --- Hyperparameters ---
    const int VOCAB_SIZE = 30522;
    const int D_MODEL = 512;
    const int MAX_SEQ_LEN = 256;
    const int N_HEADS = 8;
    const int D_FF = 2048;
    const int N_LAYERS = 6;
    // --- NEW: A shorter limit for easier testing ---
    const int MAX_GENERATION_LENGTH = 50;

    try {
        // --- 1. Load Everything ---
        std::ifstream weight_file("model_weights.bin", std::ios::binary);
        if (!weight_file.is_open()) {
            std::cerr << "ERROR: Could not open model_weights.bin" << std::endl;
            return 1;
        }

        Tokenizer tokenizer("vocab.txt");
        InputLayer input_layer(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, weight_file);
        Encoder encoder(N_LAYERS, D_MODEL, N_HEADS, D_FF, weight_file);
        Decoder decoder(N_LAYERS, D_MODEL, N_HEADS, D_FF, weight_file);
        FinalLinearLayer final_layer(D_MODEL, VOCAB_SIZE, weight_file);

        weight_file.close();

        // --- 2. Encode the User's Prompt ---
        std::vector<int> source_tokens = tokenizer.encode(user_prompt);
        Eigen::MatrixXf source_matrix = input_layer.forward(source_tokens);
        Eigen::MatrixXf encoder_output = encoder.forward(source_matrix);

        // --- 3. The Generation (Decoding) Loop ---
        std::vector<int> target_tokens;
        // Use the public getter method to be safe
        target_tokens.push_back(tokenizer.get_token_id("[CLS]"));

        std::cout << "Generating response (max " << MAX_GENERATION_LENGTH << " tokens)..." << std::endl;

        for (int i = 0; i < MAX_GENERATION_LENGTH; ++i) {
            Eigen::MatrixXf target_matrix = input_layer.forward(target_tokens);
            Eigen::MatrixXf decoder_output = decoder.forward(target_matrix, encoder_output);
            Eigen::MatrixXf logits = final_layer.forward(decoder_output);

            Eigen::MatrixXf::Index max_col;
            // We only care about the prediction for the *last* token in the sequence
            logits.row(logits.rows() - 1).maxCoeff(&max_col);
            int predicted_token_id = max_col -1;

            // --- NEW: Debug print to see the model's prediction at each step ---
            std::cout << "Step " << i << ": Predicted token ID: " << predicted_token_id
                      << " (" << tokenizer.decode({predicted_token_id}) << ")" << std::endl;

            if (predicted_token_id == tokenizer.get_token_id("[SEP]")) {
                std::cout << "Model predicted end-of-sequence token." << std::endl;
                break;
            }

            target_tokens.push_back(predicted_token_id);
        }

        // --- 4. De-tokenize and Print the Final Output ---
        std::string final_json = tokenizer.decode(target_tokens);

        std::cout << "\n--- Generated Output ---" << std::endl;
        std::cout << final_json << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}