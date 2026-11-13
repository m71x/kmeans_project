#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>

struct Point {
    std::vector<double> coords;
    // No cluster_id stored here

    Point() = default;
    explicit Point(const std::vector<double>& c) : coords(c) {}
    
    // Euclidean distance squared to another point
    double distance_squared(const Point& other) const {
        double sum = 0.0;
        for (size_t i = 0; i < coords.size(); ++i) {
            double diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    // Euclidean distance
    double distance(const Point& other) const {
        return std::sqrt(distance_squared(other));
    }
};

class KMeans {
public:
    KMeans(int k, int max_iters = 100, int random_seed = 42)
        : k_(k), max_iters_(max_iters), random_seed_(random_seed) {
        if (k <= 0) {
            throw std::invalid_argument("Number of clusters must be positive");
        }
        if (max_iters <= 0) {
            throw std::invalid_argument("Max iterations must be positive");
        }
    }

    // Fit the model to the data
    void fit(const std::vector<Point>& data) {
        if (data.empty()) {
            throw std::invalid_argument("Data cannot be empty");
        }
        if (static_cast<size_t>(k_) > data.size()) {
            throw std::invalid_argument("Number of clusters cannot exceed number of data points");
        }

        // Initialize centroids randomly
        initialize_centroids(data);

        bool converged = false;
        int iter = 0;
        assignments_.clear();
        assignments_.resize(data.size(), -1);

        while (!converged && iter < max_iters_) {
            // Assign each point to the nearest centroid
            assign_clusters(data, assignments_);

            // Update centroids
            std::vector<Point> old_centroids = centroids_;
            update_centroids(data, assignments_);

            // Check for convergence (if centroids don't change)
            converged = true;
            for (size_t i = 0; i < centroids_.size(); ++i) {
                if (centroids_[i].distance_squared(old_centroids[i]) > 1e-6) {
                    converged = false;
                    break;
                }
            }
            ++iter;
        }
    }

    // Predict cluster indices for new data
    std::vector<int> predict(const std::vector<Point>& data) const {
        std::vector<int> labels;
        labels.reserve(data.size());
        for (const auto& point : data) {
            labels.push_back(get_nearest_centroid(point));
        }
        return labels;
    }

    // Get cluster assignments for the last fitted data
    const std::vector<int>& get_assignments() const {
        return assignments_;
    }

    // Get the current centroids
    const std::vector<Point>& get_centroids() const {
        return centroids_;
    }

private:
    int k_;
    int max_iters_;
    int random_seed_;
    std::vector<Point> centroids_;
    std::vector<int> assignments_; // assignments for the last data passed to fit

    // Initialize centroids using random sampling from data points
    void initialize_centroids(const std::vector<Point>& data) {
        std::mt19937 gen(random_seed_);
        std::uniform_int_distribution<size_t> dis(0, data.size() - 1);

        std::vector<bool> used(data.size(), false);
        centroids_.clear();
        centroids_.reserve(k_);

        // Select first centroid randomly
        size_t idx = dis(gen);
        centroids_.push_back(data[idx]);
        used[idx] = true;

        // For remaining centroids, pick random distinct points
        while (centroids_.size() < static_cast<size_t>(k_)) {
            idx = dis(gen);
            if (!used[idx]) {
                centroids_.push_back(data[idx]);
                used[idx] = true;
            }
        }
    }

    // Assign each point to the nearest centroid, store results in assignments
    void assign_clusters(const std::vector<Point>& data, std::vector<int>& assignments) {
        for (size_t i = 0; i < data.size(); ++i) {
            assignments[i] = get_nearest_centroid(data[i]);
        }
    }

    // Update centroids as the mean of all points assigned to each cluster
    void update_centroids(const std::vector<Point>& data, const std::vector<int>& assignments) {
        // Reset centroids to zero and count points per cluster
        std::vector<std::vector<double>> sums(k_, std::vector<double>(data[0].coords.size(), 0.0));
        std::vector<int> counts(k_, 0);

        // Accumulate sums and counts
        for (size_t i = 0; i < data.size(); ++i) {
            int cluster = assignments[i];
            if (cluster >= 0 && cluster < k_) {
                for (size_t j = 0; j < data[i].coords.size(); ++j) {
                    sums[cluster][j] += data[i].coords[j];
                }
                counts[cluster]++;
            }
        }

        // Compute means
        for (int k = 0; k < k_; ++k) {
            if (counts[k] > 0) {
                for (size_t j = 0; j < sums[k].size(); ++j) {
                    centroids_[k].coords[j] = sums[k][j] / counts[k];
                }
            }
            // If a cluster has no points, we leave its centroid unchanged (or could reinitialize)
        }
    }

    // Find the index of the nearest centroid to a point
    int get_nearest_centroid(const Point& point) const {
        double min_dist = std::numeric_limits<double>::max();
        int best_idx = -1;

        for (size_t i = 0; i < centroids_.size(); ++i) {
            double dist = point.distance_squared(centroids_[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = static_cast<int>(i);
            }
        }
        return best_idx;
    }
};

#endif // KMEANS_H