// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// #include <iostream>
// #include <type_traits>
// #include <string>
// #include <cstdint>

namespace py = pybind11;

// template <typename T>
// void check_type(const T& value) {
//     if (std::is_same<T, int32_t>::value) {
//         std::cout << "The type is int32_t" << std::endl;
//     } else if (std::is_same<T, uint32_t>::value) {
//         std::cout << "The type is uint32_t" << std::endl;
//     } else if (std::is_same<T, float>::value) {
//         std::cout << "The type is float" << std::endl;
//     } else if (std::is_same<T, double>::value) {
//         std::cout << "The type is double" << std::endl;
//     } else if (std::is_same<T, uint8_t>::value) {
//         std::cout << "The type is uint8_t" << std::endl;
//     } else if (std::is_same<T, int8_t>::value) {
//         std::cout << "The type is int8_t" << std::endl;
//     } else if (std::is_integral<T>::value) {
//         std::cout << "The type is another integral type" << std::endl;
//     } else if (std::is_floating_point<T>::value) {
//         std::cout << "The type is another floating-point type" << std::endl;
//     } else if (std::is_same<T, std::string>::value) {
//         std::cout << "The type is std::string" << std::endl;
//     } else {
//         std::cout << "The type is something else" << std::endl;
//     }
// }

namespace diskannpy
{

typedef uint32_t filterT;

typedef uint32_t StaticIdType;
typedef uint32_t DynamicIdType;

template <class IdType> using NeighborsAndDistances = std::pair<py::array_t<IdType>, py::array_t<float>>;

}; // namespace diskannpy
