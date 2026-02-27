#pragma once

/**
 * V15 C++ Channel Detection System
 *
 * Main header that includes all V15 data structures.
 * Include this single header to access the complete V15 API.
 */

#include "types.hpp"
#include "channel.hpp"
#include "labels.hpp"
#include "sample.hpp"

// All V15 types are in the v15 namespace
namespace v15 {

/**
 * Version information
 */
constexpr const char* VERSION = "15.0.0";
constexpr int VERSION_MAJOR = 15;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace v15
