#pragma once
#include <iostream>
#include <list>
#include <optional>
#include <unordered_map>

template <typename K, typename V>
class LRUCache {
 private:
  struct Node {
    K key;
    V value;
    Node(K k, V v) : key(k), value(v) {}
  };

  int capacity;
  std::unordered_map<K, typename std::list<Node>::iterator> cache;
  std::list<Node> lruList;

  void moveToHead(typename std::list<Node>::iterator it);

 public:
  LRUCache(int cap);

  std::optional<V> get(
      const K& key);  // Use std::optional for better error handling

  bool contains(const K& key) const;

  int put(const K& key, const V& value);

  K getLRUKey() const;
  V getLRUValue() const;
  

  void remove(const K& key);

  K getandRemoveLRUKey();
  V getandRemoveLRUValue();

  int getLRUKeySize() const;

  void getLRUList() {
    for (auto it = lruList.begin(); it != lruList.end(); ++it) {
      std::cout << it->key << " " << it->value << std::endl;
    }
  }
};

template <typename K, typename V>
LRUCache<K, V>::LRUCache(int cap) : capacity(cap) {}

template <typename K, typename V>
void LRUCache<K, V>::moveToHead(typename std::list<Node>::iterator it) {
  lruList.splice(lruList.begin(), lruList, it);
}

template <typename K, typename V>
std::optional<V> LRUCache<K, V>::get(const K& key) {
  if (cache.find(key) != cache.end()) {
    auto it = cache[key];
    moveToHead(it);
    return it->value;
  }
  return std::nullopt;  // Use std::nullopt to indicate a missing key
}

template <typename K, typename V>
bool LRUCache<K, V>::contains(const K& key) const {
  return cache.find(key) != cache.end();
}

template <typename K, typename V>
int LRUCache<K, V>::put(const K& key, const V& value) {
  if (cache.find(key) != cache.end()) {
    auto it = cache[key];
    it->value = value;
    moveToHead(it);
  } else {
    if (cache.size() == capacity) {
      K lruKey = lruList.back().key;
      lruList.pop_back();
      cache.erase(lruKey);
      std::cerr << "ERROR: LRUCache is full, removing key: " << lruKey
                << std::endl;  // Error logging
      return -1;
    }
    lruList.push_front(Node(key, value));
    cache[key] = lruList.begin();
  }
  return 0;
}

template <typename K, typename V>
K LRUCache<K, V>::getLRUKey() const {
  if (!lruList.empty()) {
    return lruList.back().key;
  }
  return K();  // Return a default-constructed key if the list is empty
}

template <typename K, typename V>
V LRUCache<K, V>::getLRUValue() const {
  if (!lruList.empty()) {
    return lruList.back().value;
  }
  return V();  // Return a default-constructed key if the list is empty
}

template <typename K, typename V>
void LRUCache<K, V>::remove(const K& key) {
  auto it = cache.find(key);
  if (it != cache.end()) {
    lruList.erase(it->second);  // Erase node from list
    cache.erase(key);           // Remove from map
  }
}

template <typename K, typename V>
K LRUCache<K, V>::getandRemoveLRUKey() {
  if (!lruList.empty()) {
    K lruKey = lruList.back().key;
    lruList.pop_back();
    cache.erase(lruKey);
    return lruKey;
  }
  return K();  // Return default key if the list is empty
}

template <typename K, typename V>
V LRUCache<K, V>::getandRemoveLRUValue() {
  if (!lruList.empty()) {
    V lruValue = lruList.back().value;
    K lruKey = lruList.back().key;
    lruList.pop_back();
    cache.erase(lruKey);
    return lruValue;
  }
  return V();  // Return default value if the list is empty
}

template <typename K, typename V>
int LRUCache<K, V>::getLRUKeySize() const {
  return lruList.size();
}
