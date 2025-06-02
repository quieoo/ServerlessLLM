
#pragma once

#include <sstream>
#include <utility>
#include <cstddef>
#include <string>
#include <vector>
#include <cuda_runtime.h>
// Function to print the binary array in hexadecimal format
void PrintBinaryArrayInHex(const unsigned char* data, size_t size);

std::string toHex(const std::vector<uint8_t>& data);
std::vector<uint8_t> fromHex(const std::string& hexStr);
void* allocateAlignedPinnedMemory(size_t size, size_t alignment);
void freeAlignedPinnedMemory(void* ptr);

template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& sep) {
    if (vec.empty()) return "";

    std::ostringstream oss;
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        oss << vec[i] << sep;
    }
    oss << vec.back();
    return oss.str();
}

cudaError_t cuda_safe_move(void* free_region_base_addr, void* data_addr,
                           size_t data_size);

struct BipartEdge {
  size_t right_idx;
  int64_t weight;
};
using namespace std;

vector<pair<int, int>> MinWeightBipartiteMatching(const vector<vector<BipartEdge>>& left_edges, int& total_weight);
vector<pair<int, int>> MaxWeightBMatchingWithBoost(const vector<vector<BipartEdge>>& left_edges);

struct CanSplitResult{
  bool success;
  std::vector<size_t> to_left_sizes;
  std::vector<size_t> to_right_sizes;
};

CanSplitResult CanSplit(size_t left_total_size, size_t right_total_size, std::vector<size_t> request_sizes);


#include <cstddef>
#include <iterator>

template <typename T>
class DoublyLinkedList {
private:
    struct Node {
        T data;
        Node* prev;
        Node* next;
        Node(const T& value) : data(value), prev(nullptr), next(nullptr) {}
    };

    Node* head;
    Node* tail;
    size_t size;

public:
    // 构造函数
    DoublyLinkedList() : head(nullptr), tail(nullptr), size(0) {}

    // 析构函数
    ~DoublyLinkedList() {
        clear();
    }

    // 拷贝构造函数
    DoublyLinkedList(const DoublyLinkedList& other) : head(nullptr), tail(nullptr), size(0) {
        for (const auto& val : other) {
            push_back(val);
        }
    }

    // 拷贝赋值运算符
    DoublyLinkedList& operator=(const DoublyLinkedList& other) {
        if (this != &other) {
            clear();
            for (const auto& val : other) {
                push_back(val);
            }
        }
        return *this;
    }

    // 迭代器类
    class Iterator {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        Iterator(Node* node = nullptr) : current(node) {}

        reference operator*() const { return current->data; }
        pointer operator->() const { return &current->data; }

        Iterator& operator++() {
            current = current->next;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        Iterator& operator--() {
            current = current ? current->prev : nullptr;
            return *this;
        }

        Iterator operator--(int) {
            Iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const { return current == other.current; }
        bool operator!=(const Iterator& other) const { return current != other.current; }

    private:
        Node* current;
        friend class DoublyLinkedList;
    };

    // 常量迭代器类
    class ConstIterator {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        ConstIterator(const Node* node = nullptr) : current(node) {}

        reference operator*() const { return current->data; }
        pointer operator->() const { return &current->data; }

        ConstIterator& operator++() {
            current = current->next;
            return *this;
        }

        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        ConstIterator& operator--() {
            current = current ? current->prev : nullptr;
            return *this;
        }

        ConstIterator operator--(int) {
            ConstIterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const ConstIterator& other) const { return current == other.current; }
        bool operator!=(const ConstIterator& other) const { return current != other.current; }

    private:
        const Node* current;
        friend class DoublyLinkedList;
    };

    // 迭代器相关函数
    Iterator begin() { return Iterator(head); }
    Iterator end() { return Iterator(nullptr); }
    ConstIterator begin() const { return ConstIterator(head); }
    ConstIterator end() const { return ConstIterator(nullptr); }
    ConstIterator cbegin() const { return ConstIterator(head); }
    ConstIterator cend() const { return ConstIterator(nullptr); }

    // 容量相关
    bool empty() const { return size == 0; }
    size_t getSize() const { return size; }

    // 元素访问
    T& front() { return head->data; }
    const T& front() const { return head->data; }
    T& back() { return tail->data; }
    const T& back() const { return tail->data; }

    // 修改器
    void push_front(const T& value) {
        insert(begin(), value);
    }

    void push_back(const T& value) {
        insert(end(), value);
    }

    void pop_front() {
        if (!empty()) {
            erase(begin());
        }
    }

    void pop_back() {
        if (!empty()) {
            erase(--end());
        }
    }

    Iterator insert(Iterator pos, const T& value) {
        Node* newNode = new Node(value);
        Node* current = pos.current;

        if (current == nullptr) { // 插入到尾部
            if (tail == nullptr) { // 空链表
                head = tail = newNode;
            } else {
                tail->next = newNode;
                newNode->prev = tail;
                tail = newNode;
            }
        } else {
            newNode->next = current;
            newNode->prev = current->prev;

            if (current->prev) {
                current->prev->next = newNode;
            } else { // 插入到头部
                head = newNode;
            }

            current->prev = newNode;
        }

        ++size;
        return Iterator(newNode);
    }

    Iterator erase(Iterator pos) {
        if (pos == end() || empty()) {
            return end();
        }

        Node* current = pos.current;
        Node* nextNode = current->next;

        if (current->prev) {
            current->prev->next = current->next;
        } else { // 删除头节点
            head = current->next;
        }

        if (current->next) {
            current->next->prev = current->prev;
        } else { // 删除尾节点
            tail = current->prev;
        }

        delete current;
        --size;

        return Iterator(nextNode);
    }

    void clear() {
        while (!empty()) {
            pop_front();
        }
    }

    // 将新值插入当前迭代器之后，并返回新插入元素的迭代器
    Iterator insert_after(Iterator pos, const T& value) {
      Node* newNode = new Node(value);
      Node* current = pos.current;

      if (current == nullptr) {  // 如果当前迭代器指向末尾
        if (tail == nullptr) {   // 空链表
          head = tail = newNode;
        } else {
          tail->next = newNode;
          newNode->prev = tail;
          tail = newNode;
        }
      } else {
        newNode->next = current->next;
        newNode->prev = current;

        if (current->next) {
          current->next->prev = newNode;
        } else {  // 插入到尾部
          tail = newNode;
        }

        current->next = newNode;
      }

      ++size;
      return Iterator(newNode);
    }

    std::vector<T> toVector() const {
      std::vector<T> vec;
      for (const auto& val : *this) {
        vec.push_back(val);
      }
      return vec;
    }
};