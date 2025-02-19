#include <cstring>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

struct handle {
    char reserve[64];
};

// 将二进制数据转换为Hex字符串
std::string toHex(const std::vector<uint8_t>& data) {
    std::stringstream ss;
    for (uint8_t byte : data) {
        ss << std::setw(2) << std::setfill('0') << std::hex << (int)byte;
    }
    return ss.str();
}

// 从Hex字符串恢复为二进制数据
std::vector<uint8_t> fromHex(const std::string& hexStr) {
    std::vector<uint8_t> data;
    for (size_t i = 0; i < hexStr.length(); i += 2) {
        uint8_t byte = (std::stoi(hexStr.substr(i, 2), nullptr, 16));
        data.push_back(byte);
    }
    return data;
}

int main() {
    // create response string
    std::string mock_ret;
    handle h;
    h.reserve[0] = 'a';
    std::string handle_str = std::string(reinterpret_cast<const char*>(&h), sizeof(handle));
    std::cout << "handle_str length: " << handle_str.size() << std::endl;

    // 将handle_str转换为Hex编码并添加到mock_ret
    mock_ret = toHex(std::vector<uint8_t>(handle_str.begin(), handle_str.end()));
    std::cout << "mock_ret length after handle_str: " << mock_ret.size() << std::endl;

    int mock_device_id = 1;
    std::vector<size_t> inputs;
    inputs.push_back(mock_device_id);
    for (int i = 0; i < 10; i++) {
        inputs.push_back(i);
    }
    std::string response_str(reinterpret_cast<const char*>(inputs.data()), inputs.size() * sizeof(size_t));
    std::cout << "response_str length: " << response_str.size() << std::endl;

    // 将response_str转换为Hex编码并追加到mock_ret
    mock_ret += toHex(std::vector<uint8_t>(response_str.begin(), response_str.end()));

    std::cout << "mock_ret length after response_str: " << mock_ret.size() << std::endl;

    // 解码过程

    // 从Hex字符串恢复为二进制数据
    std::vector<uint8_t> serialized_data = fromHex(mock_ret);
    std::cout << "serialized_data length: " << serialized_data.size() << std::endl;
    std::vector<size_t> decoded_offsets;
    handle restored_handle;  // 这里使用结构体对象，而不是指针
    size_t restored_device_id;
    int pos = 0;

    // 反序列化handle结构体
    std::memcpy(&restored_handle, serialized_data.data() + pos, sizeof(handle));
    pos += sizeof(handle);

    // 反序列化设备ID
    restored_device_id = *reinterpret_cast<const size_t*>(serialized_data.data() + pos);
    pos += sizeof(size_t);

    // 反序列化偏移量
    while (pos < serialized_data.size()) {
        size_t offset = *reinterpret_cast<const size_t*>(serialized_data.data() + pos);
        decoded_offsets.push_back(offset);
        pos += sizeof(size_t);
    }

    // 输出结果
    for (size_t i = 0; i < decoded_offsets.size(); i++) {
        std::cout << decoded_offsets[i] << std::endl;
    }
    std::cout << "restored_device_id: " << restored_device_id << std::endl;

    return 0;
}
