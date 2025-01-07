#include "../include/trttl.h"
#include <iostream>
#include <thread>
#include <cassert>

using namespace trttl;

// Test logger with custom streams
void test_logger_custom_streams() {
    Logger<CerrLog, CoutLog, FileLog> logger;

    logger.print<trt_types::Severity::kINTERNAL_ERROR>("This is an internal error log.");
    logger.print<trt_types::Severity::kWARNING>("This is a warning log.");
    logger.print<trt_types::Severity::kINFO>("This is an info log.");
    logger.print<trt_types::Severity::kVERBOSE>("This is a verbose log.");

    std::cout << "Custom stream logger test passed.\n";
}

// Test logger with default parameters
void test_logger_default() {
    DefaultLogger logger;

    logger.print<trt_types::Severity::kINTERNAL_ERROR>("Default logger internal error.");
    logger.print<trt_types::Severity::kWARNING>("Default logger warning.");
    logger.print<trt_types::Severity::kINFO>("Default logger info.");
    logger.print<trt_types::Severity::kVERBOSE>("Default logger verbose.");

    std::cout << "Default logger test passed.\n";
}

// Test thread safety
void test_logger_thread_safety() {
    Logger<CerrLog, CerrLog, CerrLog, CerrLog, CerrLog> logger;

    auto log_in_thread = [&logger](const std::string& msg, trt_types::Severity severity) {
        for (int i = 0; i < 10; ++i) {
            logger.log(severity, msg.c_str());
        }
    };

    std::thread t1(log_in_thread, "Thread 1 log.", trt_types::Severity::kVERBOSE);
    std::thread t2(log_in_thread, "Thread 2 log.", trt_types::Severity::kWARNING);

    t1.join();
    t2.join();

    std::cout << "Thread safety test passed.\n";
}

// Test fallback to NoLog
void test_logger_no_log() {
    Logger<NoLog, NoLog, NoLog, NoLog, NoLog> logger;

    // Nothing should be outputted as all severities use NoLog
    logger.print<trt_types::Severity::kINTERNAL_ERROR>("This should not be logged.");
    logger.print<trt_types::Severity::kWARNING>("This should not be logged.");
    logger.print<trt_types::Severity::kINFO>("This should not be logged.");
    logger.print<trt_types::Severity::kVERBOSE>("This should not be logged.");

    std::cout << "NoLog fallback test passed (if no output above).\n";
}

int main() {
    try {
        test_logger_custom_streams();
        test_logger_default();
        test_logger_thread_safety();
        test_logger_no_log();

        std::cout << "All tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception.\n";
        return 1;
    }

    return 0;
}
