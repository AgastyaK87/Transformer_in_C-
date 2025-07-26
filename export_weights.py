import torch
import torch.nn as nn
import numpy as np
from train import TransformerModel, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS

def export_weights_to_binary(model, filename= "model_weights.bin"):
    #loads pytorch model and exports weights to binary
    print(f"Opening binary file for writing: {filename}")
    with open(filename, "wb") as f:
        #model.state_dict() = dict with all learnable params (weight,bias)
        state_dict = model.state_dict()
        print("exporting weights in this order:")

        #read and write weights in the exact same order

        for name,param in state_dict.items():
            print(f"{name} with shape {list(param.shape)}")

            #tensor converted to numpy array of float32
            data = param.cpu().numpy().astype(np.float32)

            #numpy to raw bytes
            f.write(data.tobytes())

    print(f"\nSuccessfully exported all weights to {filename}")


if __name__ == "__main__":

    #recreate transformer model
    model = TransformerModel()

    #Load train weights from pth file into model struct
    try:
        model.load_state_dict(torch.load('transformer_os_llm.pth'))
        print("Successfully loaded trained weights from transformer_os_llm.pth")
    except FileNotFoundError:
        print("ERROR: Could not find 'transformer_os_llm.pth'. Make sure you have trained the model first.")
        exit()


    #eval mode of model.
    model.eval()

    #export funct
    export_weights_to_binary(model)
