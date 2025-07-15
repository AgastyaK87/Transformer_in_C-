# Use a more common CMake version for better compatibility
cmake_minimum_required(VERSION 3.20)

# Define your project name
project(Transformer_in_C_)

# Set the C++ standard to a more modern version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the Eigen library directory
# Make sure the folder name 'eigen-3.4.0' is correct
include_directories(eigen-3.4.0)

# Create the executable from ONLY your .cpp source files
add_executable(Transformer_in_C_ main.cpp
        token_embedding.h
        token_embedding.cpp
        input_layer.h
        input_layer.cpp)