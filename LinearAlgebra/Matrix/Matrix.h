#pragma once

#include <type_traits>
#include <vector>
#include <iosfwd>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <concepts>
#include <random>

namespace LinearAlgebra {
    template<typename T>
    concept Arithmetic = std::is_arithmetic_v<T> && std::totally_ordered<T>;

    template<typename T> requires(Arithmetic<T>)
    class Matrix {
    public:
        Matrix() : Matrix(1) {}
        Matrix(T val, std::size_t r, std::size_t c);
        Matrix(std::size_t r, std::size_t c) : Matrix(0, r, c) {}
        explicit Matrix(std::size_t n) : Matrix(0, n, n) {}
        explicit Matrix(const std::vector<std::vector<T>> &data);
        explicit Matrix(const std::vector<T> &diag);
        Matrix(std::initializer_list<std::vector<T>> l) : Matrix(std::vector<std::vector<T>>(l)) {}
        Matrix(const Matrix &m) : _rows(m._rows), _cols(m._cols), _data(m._data) {}
        ~Matrix() = default;

        void swap(Matrix &m);

        std::size_t getRows() const;
        std::size_t getCols() const;

        void set(std::size_t i, std::size_t j, T val);
        T get(std::size_t i, std::size_t j) const;

        Matrix<T> operator+(Matrix const &m);
        Matrix<T> operator-(Matrix const &m);
        Matrix<T> operator*(Matrix const &m);

        std::vector<T> &operator[](std::size_t index);
        const std::vector<T> &operator[](std::size_t index) const;

        Matrix<T> &operator+=(Matrix const &m);
        Matrix<T> &operator-=(Matrix const &m);
        Matrix<T> &operator*=(Matrix const &m);
        Matrix<T> &operator*=(T m);
        Matrix<T> operator*(T m);
        Matrix<T> &operator=(Matrix m);

        bool operator==(Matrix const &m) const;
        bool operator!=(Matrix const &m) const;

        std::vector<T> getRow(size_t id);
        std::vector<T> getCol(size_t id);

        bool isSquare() const;

        static Matrix<T> randomGenerate(T l, T r, std::size_t n, std::size_t m);
    protected:
        std::size_t _rows;
        std::size_t _cols;
        std::vector<std::vector<T>> _data;
    };

    template<typename T> requires(Arithmetic<T>)
    bool Matrix<T>::isSquare() const {
        return _rows == _cols;
    }

    template<typename T> requires(Arithmetic<T>)
    std::vector<T> Matrix<T>::getRow(size_t id) {
        if (id > _rows) {
            throw std::runtime_error("Row index is greater than size");
        }
        return _data[id];
    }

    template<typename T> requires(Arithmetic<T>)
    std::vector<T> Matrix<T>::getCol(size_t id) {
        if (id > _cols) {
            throw std::runtime_error("Column index is greater than size");
        }
        std::vector<T> ret(_rows);
        for (int i = 0; i < _rows; i++) {
            ret[i] = _data[i][id];
        }
        return ret;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T>::Matrix(const std::vector<T> &diag) : Matrix(diag.size()) {
        for (int i = 0; i < diag.size(); i++) {
            _data[i][i] = diag[i];
        }
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T>::Matrix(const std::vector<std::vector<T>> &data) {
        _rows = data.size();
        _cols = data[0].size();
        _data = data;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T>::Matrix(T val, std::size_t r, std::size_t c) {
        _data.assign(r, std::vector<T>(c, val));
        _rows = r;
        _cols = c;
    }

    template<typename T> requires(Arithmetic<T>)
    void Matrix<T>::swap(Matrix &m) {
        std::swap(_rows, m._rows);
        std::swap(_cols, m._cols);
        std::swap(_data, m._data);
    }

    template<typename T> requires(Arithmetic<T>)
    std::size_t Matrix<T>::getRows() const {
        return _rows;
    }

    template<typename T> requires(Arithmetic<T>)
    std::size_t Matrix<T>::getCols() const {
        return _cols;
    }

    template<typename T> requires(Arithmetic<T>)
    void Matrix<T>::set(std::size_t i, std::size_t j, T val) {
        if (i > _rows || j > _cols) {
            throw std::runtime_error("Wrong index to set");
        }
        _data[i][j] = val;
    }

    template<typename T> requires(Arithmetic<T>)
    T Matrix<T>::get(std::size_t i, std::size_t j) const {
        if (i > _rows || j > _cols) {
            throw std::runtime_error("Wrong index to set");
        }
        return _data[i][j];
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> Matrix<T>::operator+(const Matrix &m) {
        if (m._rows != _rows || m._cols != _cols) {
            throw std::runtime_error("Wrong matrices sizes");
        }
        Matrix<T> ret(*this);
        ret += m;
        return ret;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> Matrix<T>::operator-(const Matrix &m) {
        if (m._rows != _rows || m._cols != _cols) {
            throw std::runtime_error("Wrong matrices sizes");
        }
        Matrix<T> ret(*this);
        ret -= m;
        return ret;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> Matrix<T>::operator*(const Matrix &m) {
        if (m._rows != _cols) {
            throw std::runtime_error("Wrong matrices sizes");
        }
        Matrix<T> ret(_rows, m._cols);

        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < m._cols; j++) {
                for (size_t k = 0; k < _cols; k++) {
                    ret[i][j] += _data[i][k] * m[k][j];
                }
            }
        }

        return ret;
    }

    template<typename T> requires(Arithmetic<T>)
    std::vector<T> &Matrix<T>::operator[](std::size_t index) {
        return _data[index];
    }

    template<typename T> requires(Arithmetic<T>)
    const std::vector<T> &Matrix<T>::operator[](std::size_t index) const {
        return _data[index];
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> &Matrix<T>::operator+=(const Matrix &m) {
        if (m._rows != _rows || m._cols != _cols) {
            throw std::runtime_error("Wrong matrices sizes");
        }
        for (size_t i = 0; i < m._rows; i++) {
            for (size_t j = 0; j < m._cols; j++) {
                _data[i][j] += m._data[i][j];
            }
        }

        return *this;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> &Matrix<T>::operator-=(const Matrix &m) {
        if (m._rows != _rows || m._cols != _cols) {
            throw std::runtime_error("Wrong matrices sizes");
        }
        for (size_t i = 0; i < m._rows; i++) {
            for (size_t j = 0; j < m._cols; j++) {
                _data[i][j] -= m._data[i][j];
            }
        }

        return *this;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> &Matrix<T>::operator*=(const Matrix &m) {
        if (m._rows != _cols) {
            throw std::runtime_error("Wrong matrices sizes");
        }
        return *this = *this * m;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> &Matrix<T>::operator*=(T m) {
        for (std::vector<T> &r: _data) {
            for (T &e: _data) {
                e *= m;
            }
        }
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> Matrix<T>::operator*(T m) {
        Matrix<T> ret(*this);
        ret *= m;
        return ret;
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> &Matrix<T>::operator=(Matrix m) {
        swap(m);
        return *this;
    }

    template<typename T> requires(Arithmetic<T>)
    bool Matrix<T>::operator==(const Matrix &m) const {
        if (m._cols != _cols || m._rows != _rows) {
            return false;
        }

        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++) {
                if (_data[i][j] != m._data[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    template<typename T> requires(Arithmetic<T>)
    bool Matrix<T>::operator!=(const Matrix &m) const {
        return !(*this == m);
    }

    template<typename T> requires(Arithmetic<T>)
    Matrix<T> Matrix<T>::randomGenerate(T l, T r, std::size_t n, std::size_t m) {
        Matrix<T> ret(n, m);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(l, r);
        for (int i = 0; i < ret._rows; i++) {
            for (int j = 0; j < ret._cols; j++) {
                ret[i][j] = dis(gen);
            }
        }
        return ret;
    }

    template <typename U> requires(Arithmetic<U>)
    std::istream &operator>>(std::istream &in, Matrix<U> &m) {
        size_t r, c;
        in >> r >> c;
        Matrix<U> ret(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                in >> ret[i][j];
            }
        }
        m.swap(ret);
        return in;
    }

    template <typename U> requires(Arithmetic<U>)
    std::ostream &operator<<(std::ostream &out, const Matrix<U> &m) {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                out << m[i][j] << ' ';
            }
            out << std::endl;
        }
        return out;
    }

}