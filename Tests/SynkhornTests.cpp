#include <gtest/gtest.h>
#include "../LinearAlgebra/Matrix/Matrix.h"
#include "../LinearAlgebra/SynkhornProcess/SynkhornProcess.h"

using namespace LinearAlgebra;

bool doublyStochastic(const Matrix<double> &m, double eps = 0.00001) {
    std::vector<double> row_sums(m.getRows(), 0);
    std::vector<double> col_sums(m.getCols(), 0);

    for (int i = 0; i < m.getCols(); i++) {
        for (int j = 0; j < m.getRows(); j++) {
            row_sums[j] += m[i][j];
            col_sums[i] += m[i][j];
        }
    }

    for (int i = 0; i < m.getRows(); i++) {
        if (std::abs(row_sums[i] - 1) > eps || std::abs(col_sums[i] - 1) > eps) {
            return false;
        }
    }

    return true;
}

TEST(SynkhornTests, EasyTest) {
    // I_5 matrix
    LinearAlgebra::Matrix<double> m({1, 1, 1, 1, 1});

    Algorithms::SynkhornProcess<double> sp(m);
    ASSERT_EQ(sp.getMatrix(), m);
    ASSERT_EQ(sp.getIterations(), 0);
}

TEST(SynkhornTests, OneStepTest) {
    // 2 * I_6 matrix
    LinearAlgebra::Matrix<double> m({2, 2, 2, 2, 2, 2});

    Algorithms::SynkhornProcess<double> sp(m);
    ASSERT_TRUE(doublyStochastic(sp.getMatrix()));
    ASSERT_EQ(sp.getIterations(), 1);
}

TEST(SynkhornTests, MatrixTest1) {
    // 2 * I_6 matrix
    LinearAlgebra::Matrix<double> m = {
        {1, 2, 1, 9.1, 2.2, 2.22, 1.1},
        {3.2, 1.1, 1.99, 3.1, 0.2, 1.22, 0.1},
        {0.1, 2.1, 1.2, 2.1, 2.3, 2.32, 0.1},
        {9.9, 2.1, 1.2, 3.1, 1.2, 0.22, 1.1},
        {0.1, 2.1, 1.2, 2.1, 2.3, 2.32, 0.1},
        {1, 1.2, 1.11, 9.00001, 6.2, 1.72, 1.231},
        {9.9, 2.1, 1.2, 3.1, 1.2, 0.22, 1.1},
    };

    Algorithms::SynkhornProcess<double> sp(m);
    ASSERT_TRUE(doublyStochastic(sp.getMatrix()));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}