#ifndef MINI_JSON_H
#define MINI_JSON_H

// Small strict JSON reader (RFC 8259: no comments, no trailing commas, no trailing text), used for the metadata.json
// and reference.json of a converted model folder. Numbers are doubles. No third-party dependency.

#include <string>
#include <vector>

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::string> keys;     // object members, in file order
    std::vector<JsonValue> values;

    // Last occurrence wins for a repeated key; nullptr when the key is absent.
    const JsonValue* find(const std::string& key) const {
        for (size_t i = keys.size(); i-- > 0;) {
            if (keys[i] == key) return &values[i];
        }
        return nullptr;
    }
};

const char* json_type_name(JsonValue::Type type);   // "a number", "an array", ...

// Throws std::runtime_error: "<source>: invalid JSON at line L, column C: <what>".
JsonValue parse_json(const std::string& text, const std::string& source);

#endif // MINI_JSON_H
