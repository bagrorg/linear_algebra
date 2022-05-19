#pragma once

#include "../Matrix/Matrix.h"
#include <set>

namespace LinearAlgebra::Algorithms {
    template<typename U> requires(Arithmetic<U>)
    class SynkhornProcess {
    public:
        SynkhornProcess(const LinearAlgebra::Matrix<U> &m, double eps = 0.00001) {
            // We assume that `m` is square matrix
            if (!m.isSquare()) {
                throw std::runtime_error("Synkhorn Process currently available only for sqared matricies");
            }
            doublyStochasticMatrix = m;

            // Iterating while matrix is not doubly stochastic
            while (!doublyStochastic(doublyStochasticMatrix, eps)) {
                // If now loss is better then we updating it
                if (iterations == 0 || bestLoss > iterationLoss) {
                    bestLostIteration = iterations;
                    bestLoss = iterationLoss;
                }

                // If we didn't update loss so long then we assume that we diverged
                if (iterations - bestLostIteration > divergenceIterationsDist)
                    throw std::runtime_error("Synkhorn process diverged");

                // Evaluating composition r(c(M))
                c();
                r();
                iterations += 1;
            }
        }

        size_t getIterations() const {
            return iterations;
        }

        Matrix<U> getMatrix() const {
            return doublyStochasticMatrix;
        }

    private:
        void r() {
            // Evaluating sums of all rows
            std::vector<U> D1_diag(doublyStochasticMatrix.getCols(), 0);
            for (size_t row = 0; row < doublyStochasticMatrix.getCols(); row++) {
                for (size_t col = 0; col < doublyStochasticMatrix.getRows(); col++) {
                    D1_diag[row] += doublyStochasticMatrix[row][col];
                }
            }

            // Taking an inverse of sums
            for (size_t i = 0; i < D1_diag.size(); i++) {
                D1_diag[i] = 1.0 / D1_diag[i];
            }

            // Evaluating `r` step of Synkhorn process
            Matrix<U> D1(D1_diag);
            doublyStochasticMatrix = D1 * doublyStochasticMatrix;
        }

        void c() {
            // Evaluating sums of all cols
            std::vector<U> D2_diag(doublyStochasticMatrix.getCols(), 0);
            for (size_t col = 0; col < doublyStochasticMatrix.getCols(); col++) {
                for (size_t row = 0; row < doublyStochasticMatrix.getRows(); row++) {
                    D2_diag[col] += doublyStochasticMatrix[row][col];
                }
            }
            
            // Taking an inverse of sums
            for (size_t i = 0; i < D2_diag.size(); i++) {
                D2_diag[i] = 1.0 / D2_diag[i];
            }
            
            // Evaluating `c` step of Synkhorn process
            Matrix<U> D2(D2_diag);
            doublyStochasticMatrix = doublyStochasticMatrix * D2;
        }

        bool doublyStochastic(const LinearAlgebra::Matrix<U> &m, double eps) {
            iterationLoss = 0;
            return rowlyStochasticconst(m, eps) && columnlyStochasticconst(m, eps);
        }

        bool rowlyStochasticconst (const LinearAlgebra::Matrix<U> &m, double eps) {
            // Evaluating sums of all rows to check is matrix stochastic
            // Also calculating loss
            bool ret = true;
            for (size_t row = 0; row < m.getRows(); row++) {
                U sum = 0;
                for (size_t col = 0; col < m.getCols(); col++) {
                    sum += m[row][col];
                }

                iterationLoss += std::abs(sum - 1);

                if (std::abs(sum - 1) > eps) {
                    ret = false;
                }
            }

            return ret;
        }

        bool columnlyStochasticconst (const LinearAlgebra::Matrix<U> &m, double eps) {
            // Evaluating sums of all cols to check is matrix stochastic
            // Also calculating loss
            bool ret = true;
            for (size_t col = 0; col < m.getCols(); col++) {
                U sum = 0;
                for (size_t row = 0; row < m.getRows(); row++) {
                    sum += m[row][col];
                }

                iterationLoss += std::abs(sum - 1);

                if (std::abs(sum - 1) > eps) {
                    ret = false;
                }
            }

            return ret;
        }


        // Loss here is sum of | r_i - 1 |, where r_i is the sum of i'th row, 
        // and | c_j - 1 |, where c_j is the sum of j'th col, for all i and j.
        // This need to check divergence.
        double iterationLoss = 0;

        double bestLoss = -1;
        size_t bestLostIteration = 0;
        size_t iterations = 0;
        Matrix<U> doublyStochasticMatrix;
        const size_t divergenceIterationsDist = 100;
    };

    namespace Experiments {

    namespace Detail {
    void randomlyZeroMatrix(LinearAlgebra::Matrix<double> &m, size_t numToZero) {
        std::vector<size_t> nonZeroRow(m.getRows(), m.getCols()), nonZeroCol(m.getCols(), m.getRows());
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> rDist(0,m.getRows() - 1);
        std::uniform_int_distribution<std::mt19937::result_type> cDist(0,m.getCols() - 1);
        std::set<std::pair<size_t, size_t>> elemsToZero;    // TODO: is working?

        while (elemsToZero.size() < numToZero) {
            size_t r = rDist(rng);
            size_t c = cDist(rng);

            if (nonZeroCol[c] == 1 || nonZeroRow[r] == 1) {     // To avoid situation of zero column or zero row
                continue;
            }
            nonZeroCol[c]--;
            nonZeroRow[r]--;

            elemsToZero.insert({r, c});
        }

        for (auto &e: elemsToZero) {
            m[e.first][e.second] = 0;
        }
    }
    }

    void randomExperimentSynkhorn() {
        std::vector<size_t> sizes = {2, 3, 3, 4, 5, 6, 7, 8, 9, 10};
        size_t iterations_per_size = 10000;

        std::cout << "Sizes experiments: " << std::endl;
        for (size_t size: sizes) {
            size_t algo_iterations = 0;
            for (size_t i = 0; i < iterations_per_size; i++) {
                LinearAlgebra::Matrix<double> m = LinearAlgebra::Matrix<double>::randomGenerate(0.5, 2, size, size);
                LinearAlgebra::Algorithms::SynkhornProcess<double> sp(m);
                algo_iterations += sp.getIterations();
            }
            std::cout << "SIZE " << size << ": ~"
                    << static_cast<double>(algo_iterations) / static_cast<double>(iterations_per_size) << " iterations"
                    << std::endl;
        }
    }

    void randomExperimentWithZeroesSynkhorn(const std::vector<size_t> &zerosCnts, const std::vector<size_t> &sizes) {
        size_t iterations_cnt = 1000;

        std::cout << "Non-negative experiments: " << std::endl;
        for (size_t size: sizes) {
            for (size_t zeros: zerosCnts) {
                bool diverged = false;
                size_t algo_iterations = 0;
                
                for (size_t i = 0; i < iterations_cnt; i++) {
                    LinearAlgebra::Matrix<double> m = LinearAlgebra::Matrix<double>::randomGenerate(0.01, 5, size, size);
                    Detail::randomlyZeroMatrix(m, zeros);
                    try {
                        LinearAlgebra::Algorithms::SynkhornProcess<double> sp(m);
                        algo_iterations += sp.getIterations();
                    } catch (std::runtime_error &e) {
                        std::cout << e.what() << std::endl;
                        diverged = true;
                        break;
                    }
                }
                if (!diverged)
                    std::cout << "ZEROS " << zeros << ", SIZE " << size << ": ~"
                        << static_cast<double>(algo_iterations) / static_cast<double>(iterations_cnt) << " iterations"
                        << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void randomExperimentWithZeroesAndAddingSynkhorn(const std::vector<size_t> &zerosCnts, const std::vector<size_t> &sizes, const Matrix<double> &eps) {
        size_t iterations_cnt = 1000;

        std::cout << "Non-negative experiments with epsilon: " << std::endl;
        for (size_t size: sizes) {
            for (size_t zeros: zerosCnts) {
                bool diverged = false;
                size_t algo_iterations = 0;
                size_t algo_iterations_eps = 0;
                
                for (size_t i = 0; i < iterations_cnt; i++) {
                    LinearAlgebra::Matrix<double> m = LinearAlgebra::Matrix<double>::randomGenerate(0.01, 5, size, size);
                    Detail::randomlyZeroMatrix(m, zeros);
                    try {
                        LinearAlgebra::Algorithms::SynkhornProcess<double> sp(m);
                        algo_iterations += sp.getIterations();
                    } catch (std::runtime_error &e) {
                        std::cout << e.what() << std::endl;
                        diverged = true;
                        break;
                    }

                    m += eps;

                    LinearAlgebra::Algorithms::SynkhornProcess<double> sp(m);
                    algo_iterations_eps += sp.getIterations();                
                }
                if (!diverged)
                    std::cout << "ZEROS " << zeros << ", SIZE " << size << ": ~"
                        << static_cast<double>(algo_iterations) / static_cast<double>(iterations_cnt) << " / "
                        << static_cast<double>(algo_iterations_eps) / static_cast<double>(iterations_cnt) << " iterations"
                        << std::endl;
            }
            std::cout << std::endl;
        }
    }

    }
}


