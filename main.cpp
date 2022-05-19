#include <iostream>
#include "LinearAlgebra/Matrix/Matrix.h"
#include "LinearAlgebra/SynkhornProcess/SynkhornProcess.h"
#include <set>



int main() {
    LinearAlgebra::Algorithms::Experiments::randomExperimentSynkhorn();
    std::cout << "\n-----------------\n";
    LinearAlgebra::Algorithms::Experiments::randomExperimentWithZeroesSynkhorn({1, 2, 3, 4, 5, 6, 7}, {5});
    LinearAlgebra::Algorithms::Experiments::randomExperimentWithZeroesSynkhorn({1, 2, 3, 4, 5, 6, 7}, {6, 7, 8, 9});
    LinearAlgebra::Algorithms::Experiments::randomExperimentWithZeroesAndAddingSynkhorn({1, 2, 3, 4, 5, 6, 7}, {5}, LinearAlgebra::Matrix<double>(0.0001, 5, 5));
}