#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include "kmeans.h"

// Helper function to read data from a CSV file (space or comma separated)
std::vector<Point> read_data(const std::string& filename) {
    std::vector<Point> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> coords;
        std::string value;
        while (std::getline(iss, value, ',')) { // Also handle space by replacing commas with spaces? Let's assume comma separated
            // Remove any whitespace
            value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
            if (!value.empty()) {
                try {
                    coords.push_back(std::stod(value));
                } catch (const std::exception&) {
                    // Skip invalid entries
                }
            }
        }
        // If we didn't get any values from comma separation, try space
        if (coords.empty()) {
            std::istringstream iss2(line);
            double val;
            while (iss2 >> val) {
                coords.push_back(val);
            }
        }
        if (!coords.empty()) {
            data.emplace_back(coords);
        }
    }
    return data;
}

// Helper function to write clustered data to a file
void write_clustered_data(const std::string& filename, 
                         const std::vector<Point>& data, 
                         const std::vector<int>& assignments) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    for (size_t i = 0; i < data.size(); ++i) {
        file << std::fixed << std::setprecision(6);
        for (size_t j = 0; j < data[i].coords.size(); ++j) {
            file << data[i].coords[j];
            if (j < data[i].coords.size() - 1) file << ",";
        }
        file << "," << assignments[i] << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        // Default parameters
        std::string input_file = "data.txt";
        std::string output_file = "clustered_data.txt";
        int k = 3;
        int max_iters = 100;

        // Simple command line argument parsing
        if (argc > 1) input_file = argv[1];
        if (argc > 2) output_file = argv[2];
        if (argc > 3) k = std::stoi(argv[3]);
        if (argc > 4) max_iters = std::stoi(argv[4]);

        std::cout << "Reading data from: " << input_file << std::endl;
        std::vector<Point> data = read_data(input_file);
        std::cout << "Read " << data.size() << " points with dimension " 
                  << (data.empty() ? 0 : data[0].coords.size()) << std::endl;

        std::cout << "Running K-means with k=" << k << ", max_iters=" << max_iters << std::endl;
        KMeans kmeans(k, max_iters);
        kmeans.fit(data);

        const std::vector<int>& assignments = kmeans.get_assignments();
        const auto& centroids = kmeans.get_centroids();

        std::cout << "Clustering complete!" << std::endl;
        std::cout << "Centroids:" << std::endl;
        for (size_t i = 0; i < centroids.size(); ++i) {
            std::cout << "  Cluster " << i << ": [";
            for (size_t j = 0; j < centroids[i].coords.size(); ++j) {
                std::cout << std::fixed << std::setprecision(4) << centroids[i].coords[j];
                if (j < centroids[i].coords.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Count points per cluster
        std::vector<int> cluster_counts(k, 0);
        for (int assignment : assignments) {
            if (assignment >= 0 && assignment < k) {
                cluster_counts[assignment]++;
            }
        }
        std::cout << "Cluster sizes:" << std::endl;
        for (int i = 0; i < k; ++i) {
            std::cout << "  Cluster " << i << ": " << cluster_counts[i] << " points" << std::endl;
        }

        std::cout << "Writing clustered data to: " << output_file << std::endl;
        write_clustered_data(output_file, data, assignments);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}