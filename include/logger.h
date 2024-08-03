#ifndef LOGGER_H
#define LOGGER_H
#include <chrono>
#include <cstdio>
#include <ctime>

#ifdef LOG_TO_FILE
inline FILE *get_log_stream() {
    static FILE *log_file = nullptr;
    if (log_file == nullptr) {
        log_file = fopen("app.log", "a");
        if (log_file == nullptr) {
            // If we can't open the file, fall back to stdout
            return stdout;
        }
    }
    return log_file;
}

#define LOG_OUTPUT_STREAM get_log_stream()
#else
#define LOG_OUTPUT_STREAM stdout
#endif

inline int64_t get_unix_timestamp() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch())
        .count();
}

#define LOG(level, fmt, ...)                                                   \
    do {                                                                       \
        time_t timestamp = get_unix_timestamp();                               \
        struct tm *timeinfo = localtime(&timestamp);                           \
        char time_str[20];                                                     \
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", timeinfo);   \
        fprintf(LOG_OUTPUT_STREAM, "[%s] %s: " fmt "\n", time_str, level,      \
                ##__VA_ARGS__);                                                \
        fflush(LOG_OUTPUT_STREAM);                                             \
    } while (0)

#define LOG_DEBUG(fmt, ...) LOG("DEBUG", fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG("INFO", fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG("WARN", fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) LOG("ERROR", fmt, ##__VA_ARGS__)

#endif // LOGGER_H
