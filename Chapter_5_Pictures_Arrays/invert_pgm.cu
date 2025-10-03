// invert_pgm.cu
// A minimal end-to-end CUDA program:
// 1) Load 8-bit grayscale PGM (P5) into host memory
// 2) Copy to GPU and invert pixels: out[i] = 255 - in[i]
// 3) Save result as PGM
// 4) Optional: --verify runs a second inversion on GPU and checks equality with original

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>

// --------------------------- Error handling --------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err__));     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// --------------------------- PGM I/O (P5, 8-bit) ----------------------------

// Skip comments in PGM files (lines starting with '#')
// This is a simple implementation that skips lines until it finds a non-comment line.
static void skip_comments(std::istream& in) {
    // Skip lines that start with optional whitespace followed by '#'
    while (true) {
        in >> std::ws; // consume whitespace
        if (in.peek() == '#') {
            std::string line;
            std::getline(in, line);
        } else {
            break;
        }
    }
}

// Read PGM header and pixel data
// Returns true on success, false on failure
static bool read_pgm(const std::string& path, int& width, int& height, std::vector<unsigned char>& pixels) {
    std::ifstream f(path, std::ios::binary);
    // Check if file opened successfully
    if (!f) {
        std::cerr << "Error: cannot open file: " << path << "\n";
        return false;
    }

    std::string magic;
    // Read the magic number (should be "P5" for binary PGM)
    f >> magic;
    // Check if the magic number is valid
    if (magic != "P5") {
        // P5 is the magic number for binary PGM files
        std::cerr << "Error: only P5 (binary) PGM is supported. Found: " << magic << "\n";
        return false;
    }
    // Read width, height, and maxval
    skip_comments(f);
    f >> width;
    // Check if width is valid
    skip_comments(f);
    f >> height;
    // Check if height is valid
    skip_comments(f);
    // Read maxval (should be 255 for 8-bit grayscale)
    int maxval = 0;
    f >> maxval;

    // Check if maxval is valid
    if (!f.good() || width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
        std::cerr << "Error: invalid PGM header (width/height/maxval)\n";
        return false;
    }

    // Consume single whitespace character after header before pixel data
    f.get();
    // Check if we are at the end of the file
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    // Resize pixel vector to hold all pixel data
    pixels.resize(n);
    // Read pixel data into the vector
    f.read(reinterpret_cast<char*>(pixels.data()), n);
    // Check if we read the expected number of bytes
    if (!f) {
        // If we didn't read enough bytes, report an error
        std::cerr << "Error: truncated PGM data\n";
        return false;
    }
    return true;
}

// Write PGM header and pixel data
// Returns true on success, false on failure
static bool write_pgm(const std::string& path, int width, int height, const std::vector<unsigned char>& pixels) {
    // Check if width and height are valid
    std::ofstream f(path, std::ios::binary);
    // Check if file opened successfully
    if (!f) {
        std::cerr << "Error: cannot open for write: " << path << "\n";
        return false;
    }
    // Write PGM header
    f << "P5\n" << width << " " << height << "\n255\n";
    // Check if we wrote the header successfully
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    // Ensure pixel vector has the correct size
    f.write(reinterpret_cast<const char*>(pixels.data()), n);
    return static_cast<bool>(f);
}

// --------------------------- CUDA kernel ------------------------------------
// Invert pixel values: out[i] = 255 - in[i]
__global__ void invert_kernel(const unsigned char* __restrict__ in,
                              unsigned char* __restrict__ out,
                              int width, int height)
{
    // Calculate global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    // Check if the thread is within image bounds
    if (x >= width || y >= height) return;

    // Calculate linear index in the 1D pixel array
    int idx = y * width + x;
    // Invert pixel value: out[i] = 255 - in[i]
    unsigned char p = in[idx];
    // Ensure pixel value is within 0-255 range
    out[idx] = static_cast<unsigned char>(255 - p);
}

// --------------------------- Utility ----------------------------------------

// Compare two byte vectors for equality
// Returns true if they have the same size and all bytes are equal
static bool equal_bytes(const std::vector<unsigned char>& a,
                        const std::vector<unsigned char>& b)
{
    // Check if sizes are equal
    if (a.size() != b.size()) return false;
    // Compare each byte
    for (size_t i = 0; i < a.size(); ++i) {
        // If any byte differs, return false
        if (a[i] != b[i]) return false;
    }
    return true;
}

// Print usage instructions
// Displays how to run the program and what arguments it accepts
static void print_usage(const char* prog) {
    // Print usage instructions to stderr
    std::cerr << "Usage:\n"
              << "  " << prog << " <input.pgm> <output.pgm> [--verify]\n\n"
              << "Notes:\n"
              << "  - Supports 8-bit grayscale PGM (P5) only.\n"
              << "  - --verify: runs a second inversion on the GPU and checks equality with input.\n";
}

// --------------------------- Main -------------------------------------------

int main(int argc, char** argv) {
    // Check if enough arguments are provided
    if (argc < 3) {
        // If not, print usage instructions and exit
        print_usage(argv[0]);
        std::cerr << "Error: not enough arguments.\n";
        std::cerr << "Expected: <input.pgm> <output.pgm> [--verify]\n";
        std::cerr << "Got: " << (argc - 1) << " arguments.\n";
        std::cerr << "Example: " << argv[0] << " input.pgm output.pgm --verify\n";
        std::cerr << "Note: PGM must be 8-bit grayscale (P5 format).\n";
        std::cerr << "Note: --verify runs a second inversion on the GPU and checks equality with input.\n";
        // Exit with failure status
        std::cerr << "Exiting...\n";
        return EXIT_FAILURE;
    }
    // Parse command line arguments
    // Expecting: <input.pgm> <output.pgm> [--verify]
    // Input PGM file path
    // Output PGM file path
    // Optional: --verify flag to check double inversion
    // If --verify is provided, it will run a second inversion and check equality with the original input
    const std::string in_path  = argv[1];
    // Output PGM file path
    const std::string out_path = argv[2];
    // Check if --verify flag is provided
    const bool do_verify = (argc >= 4 && std::string(argv[3]) == "--verify");
    // If --verify is provided, print a message indicating verification will be performed
    int width = 0, height = 0;
    // Read input PGM file
    // Initialize host vector to hold pixel data
    // This will hold the pixel values read from the input PGM file
    std::vector<unsigned char> h_in;
    // Attempt to read the PGM file into host memory
    // If reading fails, print an error message and exit
    if (!read_pgm(in_path, width, height, h_in)) {
        // If reading the PGM file fails, print an error message and exit
        std::cerr << "Error: failed to read input PGM: " << in_path << "\n";
        std::cerr << "Ensure the file exists and is a valid 8-bit grayscale PGM (P5) file.\n";
        std::cerr << "Exiting...\n";
        return EXIT_FAILURE;
    }

    // Check if the input PGM file has valid dimensions
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    // If width or height is zero, print an error message and exit
    std::vector<unsigned char> h_out(n);
    // If the input PGM file has no pixels, print an error message and exit
    unsigned char *d_in = nullptr, *d_out = nullptr;
    // Allocate device memory for input and output pixel data
    const size_t nBytes = n * sizeof(unsigned char);
    // If the number of pixels is zero, print an error message and exit
    CUDA_CHECK(cudaMalloc(&d_in,  nBytes));
    // Allocate device memory for output pixel data
    CUDA_CHECK(cudaMalloc(&d_out, nBytes));
    // Copy input pixel data from host to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), nBytes, cudaMemcpyHostToDevice));
    // Check if the copy was successful
    dim3 block(16, 16);
    // Define block size for CUDA kernel
    dim3 grid( (width  + block.x - 1) / block.x,    
               (height + block.y - 1) / block.y );
    // Define grid size for CUDA kernel
    // Launch the CUDA kernel to invert pixel values
    // The kernel will run on the GPU and perform the inversion operation
    // Each thread will handle one pixel, inverting its value
    // The kernel will run in parallel across the grid of blocks
    // Each block will contain multiple threads, and each thread will process one pixel
    invert_kernel<<<grid, block>>>(d_in, d_out, width, height);
    // Check for any errors during kernel launch
    // If there are errors, print an error message and exit
    CUDA_CHECK(cudaGetLastError());
    // Check for any errors after kernel execution
    // If there are errors, print an error message and exit
    // This ensures that the kernel executed successfully before proceeding
    // Copy the output pixel data from device to host
    // This will copy the inverted pixel values back to the host memory
    // The output pixel data will be stored in the h_out vector
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, nBytes, cudaMemcpyDeviceToHost));
    // Check if the copy was successful
    // If the copy was not successful, print an error message and exit
    if (!write_pgm(out_path, width, height, h_out)) {
        // If writing the output PGM file fails, print an error message and exit
        // This ensures that the output PGM file was written successfully
        // If the output PGM file could not be written, print an error message and exit
        std::cerr << "Error: failed to write output PGM\n";
        // Free device memory before exiting
        // This ensures that the device memory is freed before exiting
        // If the output PGM file could not be written, print an error message and exit
        CUDA_CHECK(cudaFree(d_in));
        // Free device memory for output buffer
        // If the output PGM file could not be written, print an error message and exit
        CUDA_CHECK(cudaFree(d_out));
        return EXIT_FAILURE;
    }

    if (do_verify) {
        // Run second inversion on GPU: out2 = invert(out1)
        std::vector<unsigned char> h_out2(n);
        // Reuse d_in as input buffer for the second pass
        CUDA_CHECK(cudaMemcpy(d_in, h_out.data(), nBytes, cudaMemcpyHostToDevice));
        // Launch the kernel again to invert the already inverted image
        invert_kernel<<<grid, block>>>(d_in, d_out, width, height);
        // Check for any errors during the second kernel launch
        CUDA_CHECK(cudaGetLastError());
        // Copy the second output back to host memory
        CUDA_CHECK(cudaMemcpy(h_out2.data(), d_out, nBytes, cudaMemcpyDeviceToHost));
        // Verify if double inversion matches original input
        if (equal_bytes(h_in, h_out2)) {
            std::cout << "[VERIFY] PASS: double inversion matches original.\n";
        } else {
            std::cout << "[VERIFY] FAIL: double inversion does not match original.\n";
        }
    }
    // Free device memory for input and output buffers
    CUDA_CHECK(cudaFree(d_in));
    // Free device memory for output buffer
    CUDA_CHECK(cudaFree(d_out));
    // Print success message with output file path and dimensions
    std::cout << "Wrote: " << out_path << " (" << width << "x" << height << ")\n";
    return EXIT_SUCCESS;
}
