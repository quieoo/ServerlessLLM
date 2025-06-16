#pragma once

#include <iostream>
#include <sstream>

// 日志级别枚举
enum LogLevel {
    INFO,
    WARNING,
    ERROR,
    METRIC,
};

// 日志类
class Logger {
public:
    // 构造函数，接收日志级别
    Logger(LogLevel level) : level(level) {}

    // 重载 operator<<，用于输出信息
    template <typename T>
    Logger& operator<<(const T& message) {
        stream << message;  // 将传入的消息追加到流中
        return *this;  // 返回自身，以支持链式调用
    }

    // 输出日志时，调用这个方法将最终的日志信息输出
    ~Logger() {
        if (level == METRIC) {
            std::cout << stream.str() << std::endl;
        } else {
            // 暂时屏蔽INFO日志的输出
            // if (level != INFO) {
                std::cout << "[" << getLevelString() << "] " << stream.str() << std::endl;
            // }
        }
    }

private:
    // 根据日志级别返回相应的字符串
    std::string getLevelString() const {
        switch (level) {
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    LogLevel level;
    std::ostringstream stream;  // 使用字符串流来保存传入的消息
};

// 宏，用于创建日志对象并指定日志级别
#define LOG(level) Logger(level)
