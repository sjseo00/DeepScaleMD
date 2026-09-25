
#pragma once
#ifndef MLIP_YAML_PARSER_H
#define MLIP_YAML_PARSER_H

#include "../runtime/model_specification.h"
#include <string>

namespace mlip {

MlipModelSpec parse_model_spec(const std::string& yaml_path);

MlipModelSpec parse_model_spec_from_string(const std::string& yaml_text);

} // namespace mlip
#endif // MLIP_YAML_PARSER_H
