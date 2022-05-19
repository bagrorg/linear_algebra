#include <gtest/gtest.h>
#include "../LinearAlgebra/Matrix/Matrix.h"

TEST(MatrixTest, CreationZeroInt) {
    LinearAlgebra::Matrix<int> m_int(10, 10);
    ASSERT_EQ(m_int.getRows(), 10);
    ASSERT_EQ(m_int.getCols(), 10);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            ASSERT_EQ(m_int[i][j], 0);
        }
    }
}

TEST(MatrixTest, CreationZeroFloat) {
    LinearAlgebra::Matrix<float> m_float(10, 10);
    ASSERT_EQ(m_float.getRows(), 10);
    ASSERT_EQ(m_float.getCols(), 10);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            ASSERT_EQ(m_float[i][j], 0);
        }
    }
}

TEST(MatrixTest, CreationWithConst) {
    LinearAlgebra::Matrix<int> m(1, 3, 3);
    ASSERT_EQ(m.getRows(), 3);
    ASSERT_EQ(m.getCols(), 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_EQ(m[i][j], 1);
        }
    }
}

TEST(MatrixTest, CreationRowsAndColsOrder) {
    LinearAlgebra::Matrix<float> m(5, 6);
    ASSERT_EQ(m.getRows(), 5);
    ASSERT_EQ(m.getCols(), 6);
}

TEST(MatrixTest, CreationQuadratic) {
    LinearAlgebra::Matrix<int> m(5);
    ASSERT_EQ(m.getRows(), 5);
    ASSERT_EQ(m.getCols(), 5);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            ASSERT_EQ(m[i][j], 0);
        }
    }
}

TEST(MatrixTest, CreationDefault) {
    LinearAlgebra::Matrix<float> m;
    ASSERT_EQ(m.getRows(), 1);
    ASSERT_EQ(m.getCols(), 1);
    ASSERT_EQ(m[0][0], 0);
}

TEST(MatrixTest, CreationInitList) {
    LinearAlgebra::Matrix<int> m = {
            {1, 2, 3},
            {-1, -2, -3}
    };
    ASSERT_EQ(m.getRows(), 2);
    ASSERT_EQ(m.getCols(), 3);

    ASSERT_EQ(m[0][0], 1);
    ASSERT_EQ(m[0][1], 2);
    ASSERT_EQ(m[0][2], 3);

    ASSERT_EQ(m[1][0], -1);
    ASSERT_EQ(m[1][1], -2);
    ASSERT_EQ(m[1][2], -3);
}

TEST(MatrixTest, CreationCopy) {
    LinearAlgebra::Matrix<int> m1 = {
            {1, 2, 3},
            {-1, -2, -3}
    };

    LinearAlgebra::Matrix<int> m2 = {
            {1, 2, 3},
            {-1, -2, -3}
    };

    m1[0][0] = 0;

    ASSERT_EQ(m2.getRows(), 2);
    ASSERT_EQ(m2.getCols(), 3);

    ASSERT_EQ(m2[0][0], 1);
    ASSERT_EQ(m2[0][1], 2);
    ASSERT_EQ(m2[0][2], 3);

    ASSERT_EQ(m2[1][0], -1);
    ASSERT_EQ(m2[1][1], -2);
    ASSERT_EQ(m2[1][2], -3);
}

TEST(MatrixTest, AccessOperator) {
    LinearAlgebra::Matrix<int> m1 = {
            {1, 2, 3},
            {-1, -2, -3}
    };

    m1[0][0] = 0;

    ASSERT_EQ(m1[0][0], 0);
}

TEST(MatrixTest, MatrixAdd) {
    LinearAlgebra::Matrix<int> m1 = {
            {1, 2, 3},
            {-1, -2, -3}
    };

    LinearAlgebra::Matrix<int> m2 = {
            {-1, -2, -3},
            {1, 2, 3}
    };

    auto m3 = m1 + m2;
    m1 += m2;

    ASSERT_EQ(m1.getCols(), 3);
    ASSERT_EQ(m1.getRows(), 2);
    ASSERT_EQ(m3.getCols(), 3);
    ASSERT_EQ(m3.getRows(), 2);

    for (int i = 0; i < m1.getRows(); i++) {
        for (int j = 0; j < m1.getCols(); j++) {
            ASSERT_EQ(m1[i][j], 0);
            ASSERT_EQ(m3[i][j], 0);
        }
    }
}

TEST(MatrixTest, MatrixSub) {
    LinearAlgebra::Matrix<int> m1 = {
            {1, 2, 3},
            {-1, -2, -3}
    };

    LinearAlgebra::Matrix<int> m2 = {
            {1, 2, 3},
            {-1, -2, -3}
    };

    auto m3 = m1 - m2;
    m1 -= m2;

    ASSERT_EQ(m1.getCols(), 3);
    ASSERT_EQ(m1.getRows(), 2);
    ASSERT_EQ(m3.getCols(), 3);
    ASSERT_EQ(m3.getRows(), 2);

    for (int i = 0; i < m1.getRows(); i++) {
        for (int j = 0; j < m1.getCols(); j++) {
            ASSERT_EQ(m1[i][j], 0);
            ASSERT_EQ(m3[i][j], 0);
        }
    }
}

TEST(MatrixTest, MatrixMul) {
    LinearAlgebra::Matrix<int> m1 = {
            {1, 2, 3},
            {1, 2, 3},
            {1, 3, 2},
            {3, 2, 1},
            {1, 1, 1}
    };

    LinearAlgebra::Matrix<int> m2 = {
            {1, 2, 3},
            {-1, -2, -3},
            {1, 1, 1}
    };

    LinearAlgebra::Matrix<int> ans = {
            {2,	 1,	 0},
            {2,	 1,	 0},
            {0,	-2,	-4},
            {2,	 3,	 4},
            {1,	 1,	 1}
    };



    auto m3 = m1 * m2;
    m1 *= m2;

    ASSERT_EQ(m1.getCols(), 3);
    ASSERT_EQ(m1.getRows(), 5);
    ASSERT_EQ(m3.getCols(), 3);
    ASSERT_EQ(m3.getRows(), 5);

    for (int i = 0; i < m1.getRows(); i++) {
        for (int j = 0; j < m1.getCols(); j++) {
            ASSERT_EQ(m1[i][j], ans[i][j]);
            ASSERT_EQ(m3[i][j], ans[i][j]);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}