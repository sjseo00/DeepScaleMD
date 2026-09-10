/**
 * yaml_parser.h
 *
 * Parses a MLIP YAML model spec file into a MlipModelSpec object.
 * Strict validation: unknown keys are rejected, missing required keys throw.
 *
 * Depends on yaml-cpp (libyaml-cpp-dev).
 * This header is included only by yaml_parser.cpp and tests.
 * MD core must NOT include this header.
 */

#pragma once
#ifndef MLIP_YAML_PARSER_H
#define MLIP_YAML_PARSER_H

#include "../runtime/model_specification.h"
#include <string>

namespace mlip {

/**
 * Parse a YAML file and return the validated MlipModelSpec.
 * Throws std::runtime_error with a descriptive message on any error.
 */
MlipModelSpec parse_model_spec(const std::string& yaml_path);

/**
 * Parse a YAML string directly (useful for tests).
 */
MlipModelSpec parse_model_spec_from_string(const std::string& yaml_text);

} // namespace mlip
#endif // MLIP_YAML_PARSER_H
