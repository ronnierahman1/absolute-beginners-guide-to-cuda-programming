
// gradient_pgm.cpp
// Minimal PGM (P5, 8-bit) gradient generator.
// Generates a grayscale gradient image in PGM format.
// Usage: ./gradient_pgm <width> <height> <mode:h|v> <output.pgm>
//   <width>  - width of the image in pixels
//   <height> - height of the image in pixels
//   <mode>   - 'h' for horizontal gradient, 'v' for vertical gradient
//   <output.pgm> - output file name

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

/**
 * @brief Write grayscale pixel data to a binary PGM (P5) file.
 *
 * @param path   Output file path
 * @param w      Image width in pixels
 * @param h      Image height in pixels
 * @param pixels Vector of grayscale pixel values (0-255), size = w * h
 */
static void write_pgm(const std::string& path, int w, int h, const std::vector<unsigned char>& pixels) {
    std::ofstream f(path, std::ios::binary); // Open file in binary mode
    if (!f) {
        std::cerr << "Error: cannot open for write: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    // Write PGM header
    f << "P5\n" << w << " " << h << "\n255\n";
    // Write pixel data
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (!f) {
        std::cerr << "Error: failed to write PGM data\n";
        std::exit(EXIT_FAILURE);
    }
}


/**
 * @brief Program entry point. Generates a grayscale gradient image in PGM format.
 *
 * Usage: ./gradient_pgm <width> <height> <mode:h|v> <output.pgm>
 *   <width>  - width of the image in pixels
 *   <height> - height of the image in pixels
 *   <mode>   - 'h' for horizontal gradient, 'v' for vertical gradient
 *   <output.pgm> - output file name
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int main(int argc, char** argv) {
    // Check for correct number of arguments
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <mode:h|v> <output.pgm>\n"
                  << "  mode h = horizontal gradient (left→right)\n"
                  << "  mode v = vertical gradient (top→bottom)\n";
        return EXIT_FAILURE;
    }

    // Parse command-line arguments
    const int  width  = std::atoi(argv[1]);   ///< Image width in pixels
    const int  height = std::atoi(argv[2]);   ///< Image height in pixels
    const char mode   = argv[3][0];           ///< 'h' for horizontal, 'v' for vertical gradient
    const std::string out_path = argv[4];     ///< Output file path

    // Validate arguments
    if (width <= 0 || height <= 0 || (mode != 'h' && mode != 'v')) {
        std::cerr << "Error: invalid arguments.\n";
        return EXIT_FAILURE;
    }

    // Allocate image buffer (width * height pixels)
    // Image buffer: stores grayscale values for each pixel
    std::vector<unsigned char> img(static_cast<size_t>(width) * height);

    if (mode == 'h') {
        // Horizontal gradient: intensity increases from left to right
        // For each row, set pixel value based on column index
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Scale x into [0,255] for each column
                unsigned char val = static_cast<unsigned char>((255.0 * x) / (width - 1));
                img[static_cast<size_t>(y) * width + x] = val;
            }
        }
    } else { // mode == 'v'
        // Vertical gradient: intensity increases from top to bottom
        // For each column, set pixel value based on row index
        for (int y = 0; y < height; ++y) {
            // Scale y into [0,255] for each row
            unsigned char val = static_cast<unsigned char>((255.0 * y) / (height - 1));
            for (int x = 0; x < width; ++x) {
                img[static_cast<size_t>(y) * width + x] = val;
            }
        }
    }

    // Write the image to a PGM file
    write_pgm(out_path, width, height, img);
    std::cout << "Wrote: " << out_path << " (" << width << "x" << height << ", mode=" << mode << ")\n";
    return EXIT_SUCCESS;
}
