#include "MiniJson.h"

#include <locale>
#include <sstream>
#include <stdexcept>

const char* json_type_name(JsonValue::Type type) {
    switch (type) {
        case JsonValue::Type::Null: return "null";
        case JsonValue::Type::Bool: return "a boolean";
        case JsonValue::Type::Number: return "a number";
        case JsonValue::Type::String: return "a string";
        case JsonValue::Type::Array: return "an array";
        case JsonValue::Type::Object: return "an object";
    }
    return "?";
}

namespace {

class JsonParser {
public:
    JsonParser(const std::string& text, const std::string& source) : text_(text), source_(source) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value(0);
        skip_whitespace();
        if (pos_ != text_.size()) fail("unexpected text after the JSON value");
        return value;
    }

private:
    static constexpr int kMaxDepth = 64;

    [[noreturn]] void fail(const std::string& what) const {
        size_t line = 1, column = 1;
        for (size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw std::runtime_error(source_ + ": invalid JSON at line " + std::to_string(line) + ", column " +
                                 std::to_string(column) + ": " + what);
    }

    bool at_end() const { return pos_ >= text_.size(); }
    char peek() const { return at_end() ? '\0' : text_[pos_]; }

    void skip_whitespace() {
        while (!at_end() && (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
                             text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    void expect_literal(const char* literal) {
        for (const char* p = literal; *p; ++p) {
            if (peek() != *p) fail(std::string("expected '") + literal + "'");
            ++pos_;
        }
    }

    JsonValue parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting is too deep");
        if (at_end()) fail("unexpected end of file");
        JsonValue value;
        const char c = peek();
        if (c == '{') {
            parse_object(value, depth);
        } else if (c == '[') {
            parse_array(value, depth);
        } else if (c == '"') {
            value.type = JsonValue::Type::String;
            value.string = parse_string();
        } else if (c == 't') {
            expect_literal("true");
            value.type = JsonValue::Type::Bool;
            value.boolean = true;
        } else if (c == 'f') {
            expect_literal("false");
            value.type = JsonValue::Type::Bool;
            value.boolean = false;
        } else if (c == 'n') {
            expect_literal("null");
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            value.type = JsonValue::Type::Number;
            value.number = parse_number();
        } else {
            fail(std::string("unexpected character '") + c + "'");
        }
        return value;
    }

    void parse_object(JsonValue& value, int depth) {
        value.type = JsonValue::Type::Object;
        expect('{');
        skip_whitespace();
        if (peek() == '}') {
            ++pos_;
            return;
        }
        while (true) {
            skip_whitespace();
            if (peek() != '"') fail("expected a string key");
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            JsonValue member = parse_value(depth + 1);
            value.keys.push_back(std::move(key));
            value.values.push_back(std::move(member));
            skip_whitespace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return;
        }
    }

    void parse_array(JsonValue& value, int depth) {
        value.type = JsonValue::Type::Array;
        expect('[');
        skip_whitespace();
        if (peek() == ']') {
            ++pos_;
            return;
        }
        while (true) {
            skip_whitespace();
            value.array.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return;
        }
    }

    unsigned parse_hex4() {
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = peek();
            code <<= 4;
            if (c >= '0' && c <= '9') code |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') code |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') code |= static_cast<unsigned>(c - 'A' + 10);
            else fail("expected four hexadecimal digits after \\u");
            ++pos_;
        }
        return code;
    }

    static void append_utf8(std::string& out, unsigned code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (at_end()) fail("unterminated string");
            const char c = text_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) {
                --pos_;
                fail("control character inside a string");
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (at_end()) fail("unterminated escape");
            const char e = text_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code = parse_hex4();
                    if (code >= 0xD800 && code <= 0xDBFF) {
                        if (peek() != '\\' || pos_ + 1 >= text_.size() || text_[pos_ + 1] != 'u') {
                            fail("high surrogate without a low surrogate");
                        }
                        pos_ += 2;
                        const unsigned low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) fail("invalid low surrogate");
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    } else if (code >= 0xDC00 && code <= 0xDFFF) {
                        fail("low surrogate without a high surrogate");
                    }
                    append_utf8(out, code);
                    break;
                }
                default:
                    --pos_;
                    fail(std::string("invalid escape '\\") + e + "'");
            }
        }
    }

    double parse_number() {
        const size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            fail("invalid number");
        }
        if (peek() == '.') {
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number: digits expected after '.'");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number: digits expected in the exponent");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        std::istringstream stream(text_.substr(start, pos_ - start));
        stream.imbue(std::locale::classic());
        double number = 0.0;
        stream >> number;
        if (stream.fail()) fail("number out of range");
        return number;
    }

    const std::string& text_;
    const std::string& source_;
    size_t pos_ = 0;
};

}  // namespace

JsonValue parse_json(const std::string& text, const std::string& source) {
    return JsonParser(text, source).parse();
}
