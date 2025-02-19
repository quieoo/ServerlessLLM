#pragma once

#include <vector>
#include <shared_mutex>
#include <stdexcept>
#include <iostream>

template <typename T>
class ConcurrentArray {
private:
    std::vector<T> array;
    mutable std::shared_mutex mtx;

public:
    ConcurrentArray() {}

    explicit ConcurrentArray(size_t size) {
        array.resize(size);
    }

    ConcurrentArray(size_t size, const T& default_value) {
        array.resize(size, default_value);
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        return array.size();
    }

    T get(size_t index) const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        if (index >= array.size()) {
            throw std::out_of_range("Index out of range");
        }
        return array[index];
    }

    void set(size_t index, const T& value) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        if (index >= array.size()) {
            throw std::out_of_range("Index out of range");
        }
        array[index] = value;
    }

    void push_back(const T& value) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        array.push_back(value);
    }

    void pop_back() {
        std::unique_lock<std::shared_mutex> lock(mtx);
        if (!array.empty()) {
            array.pop_back();
        }
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(mtx);
        array.clear();
    }

    void print() const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        for (const auto& item : array) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
};
