#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "util/cexpr_utils.hpp"
#include "util/trt_types.hpp"
#include <NvInfer.h>
#include <source_location>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
#include <tuple>
#include <mutex>

namespace trttl {

/*!
* Log stream wrapper class - returns stream to write to.
* Uses compile-time polymorphism via wonders of CRTP.
* LogStream object must encapsulate & manage underlying resources.
*
* @tparam Derived - CRTP
*/
template<typename Derived>
class LogStream{
public:
    /*!
    * Returns stream to log to. 
    */
    std::ostream&& get(){
        return std::move(static_cast<Derived*>(this)->get_impl());
    }
};

class CoutLog : public LogStream<CoutLog>{
public:
    std::ostream&& get_impl(){
        return std::move(std::cout);
    }
};

class CerrLog : public LogStream<CerrLog>{
public:
    std::ostream&& get_impl(){
        return std::move(std::cerr);
    }
};

class FileLog : public LogStream<FileLog>{
private:
    std::ofstream fout;

public:
    FileLog() {
        fout.open("trt.log", std::ios::out);
        if (!fout.is_open()) {
            throw std::ios_base::failure("Failed to open log file!");
        }
    }

    std::ostream& get_impl(){
        return fout;
    }

    ~FileLog() {
        if (fout.is_open()) {
            fout.close();
        }
    }
};

/*!
* Special case for not writing.
*/
class NoLog final : public LogStream<NoLog>{
private:
    std::ostream cnull{0};
public:
    std::ostream& get_impl(){
        return cnull;
    }
};

/*!
* Concept for classes derived from LogStream.
*/
template <typename T>
concept DerivedFromLogStream = std::derived_from<T, LogStream<T>>;

/*!
* Thread-safe logger class.
* Template interfaces:
* - Provide LogStreams up to log-level of interest using appropriate template param,
*   rest will be assumed `NoLog`.
* NOTE: By default will assume all LogStreams as `NoLog` - won't produce output!
*
* LogStream objects are initialized only-once and stored for logger lifetime. 
*/
template <DerivedFromLogStream LogStreamINTERNAL_ERROR = NoLog, 
          DerivedFromLogStream LogStreamERROR = NoLog, 
          DerivedFromLogStream LogStreamWARNING = NoLog, 
          DerivedFromLogStream LogStreamINFO = NoLog, 
          DerivedFromLogStream LogStreamVERBOSE = NoLog>
class Logger : public nvinfer1::ILogger {
private:
    static std::mutex mtx;                                    /*!< Mutex for thread safety.*/
    static constexpr std::array<const char*, 5> lookup = {    /*!< Static lookup-table for log level prefixes.*/
        "[IE]", "[E]", "[W]", "[I]", "[V]"
    };

    std::tuple<LogStreamINTERNAL_ERROR, 
               LogStreamERROR, 
               LogStreamWARNING, 
               LogStreamINFO, 
               LogStreamVERBOSE> log_streams;                 /*!< `LogStream` objects container.*/

public:
    /*!
    * Log function.
    * Attaches prefix, timestamp and source_location.
    */
    template<trt_types::Severity severity>
    void print(const char* msg, const std::source_location location = 
             std::source_location::current()
    ) {
        const auto i = cexpr_utils::to_underlying<trt_types::Severity>(severity);
        std::lock_guard<std::mutex> lock(mtx);
        auto&& stream = std::get<i>(log_streams).get();

        const auto now = std::chrono::system_clock::now();
        const std::time_t time = std::chrono::system_clock::to_time_t(now);

        stream << lookup[i] << " " 
               << std::put_time(std::localtime(&time), "%FT%TZ")
               << location.file_name() << "("
               << location.line() << ":"
               << location.column() << ") `"
               << location.function_name() << "`: "
               << msg << std::endl;
    }

    /*! 
    * Just for TRT C++ API comaptibility.
    */ 
    void log(trt_types::Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case trt_types::Severity::kINTERNAL_ERROR:
                print<trt_types::Severity::kINTERNAL_ERROR>(msg);
                break;
            case trt_types::Severity::kERROR:
                print<trt_types::Severity::kERROR>(msg);
                break;
            case trt_types::Severity::kWARNING:
                print<trt_types::Severity::kWARNING>(msg);
                break;
            case trt_types::Severity::kINFO:
                print<trt_types::Severity::kINFO>(msg);
                break;
            case trt_types::Severity::kVERBOSE:
                print<trt_types::Severity::kVERBOSE>(msg);
                break;
            default:
                break;
        }
    }
};

template <
    DerivedFromLogStream LogStreamINTERNAL_ERROR,
    DerivedFromLogStream LogStreamERROR,
    DerivedFromLogStream LogStreamWARNING,
    DerivedFromLogStream LogStreamINFO,
    DerivedFromLogStream LogStreamVERBOSE>
std::mutex Logger<LogStreamINTERNAL_ERROR, LogStreamERROR, LogStreamWARNING, LogStreamINFO, LogStreamVERBOSE>::mtx;

/*!
* Convinience naming for logger with default LogStreams setup.
*/
using DefaultLogger = Logger<CerrLog, CerrLog, CoutLog, CoutLog, FileLog>;

} // trttl namespace
#endif // LOGGER_HPP