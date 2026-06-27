// core/config.hpp — minimal first-party TOML reader (the project config/recipe/annotation
// format; no external dep). Supports comments, [dotted.sections], key = value with string/
// int/float/bool/array values. Flattened to "section.key" lookups. Enough for recipes,
// annotations, the scrolls registry, and stage Params. (A reflection-driven binder layers
// on top later.) See fenix-core-internals.
#pragma once

#include "core/error.hpp"
#include "core/types.hpp"

#include <charconv>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fenix {

class Config {
public:
    static Expected<Config> parse(const std::string& text) {
        Config c;
        std::string section;
        std::istringstream in(text);
        for (std::string line; std::getline(in, line);) {
            strip_comment(line);
            std::string t = trim(line);
            if (t.empty()) continue;
            if (t.front() == '[' && t.back() == ']') {
                section = trim(t.substr(1, t.size() - 2));
                continue;
            }
            const auto eq = t.find('=');
            if (eq == std::string::npos) return err(Errc::decode_error, "bad config line: " + t);
            std::string key = trim(t.substr(0, eq));
            std::string val = trim(t.substr(eq + 1));
            c.kv_[section.empty() ? key : section + "." + key] = val;
        }
        return c;
    }

    static Expected<Config> load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return err(Errc::not_found, "cannot open " + path);
        std::stringstream ss;
        ss << f.rdbuf();
        return parse(ss.str());
    }

    [[nodiscard]] bool has(const std::string& key) const { return kv_.contains(key); }

    [[nodiscard]] std::optional<std::string> get_string(const std::string& key) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return std::nullopt;
        return unquote(it->second);
    }
    [[nodiscard]] std::optional<f64> get_float(const std::string& key) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return std::nullopt;
        return to_double(it->second);
    }
    [[nodiscard]] std::optional<s64> get_int(const std::string& key) const {
        auto v = get_float(key);
        return v ? std::optional<s64>(static_cast<s64>(*v)) : std::nullopt;
    }
    [[nodiscard]] std::optional<bool> get_bool(const std::string& key) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return std::nullopt;
        if (it->second == "true") return true;
        if (it->second == "false") return false;
        return std::nullopt;
    }
    // Parse a [a, b, c] array into raw element strings (unquoted).
    [[nodiscard]] std::vector<std::string> get_array(const std::string& key) const {
        std::vector<std::string> out;
        auto it = kv_.find(key);
        if (it == kv_.end()) return out;
        std::string s = it->second;
        if (s.size() >= 2 && s.front() == '[' && s.back() == ']') s = s.substr(1, s.size() - 2);
        std::istringstream ss(s);
        for (std::string item; std::getline(ss, item, ',');) {
            std::string t = trim(item);
            if (!t.empty()) out.push_back(unquote(t));
        }
        return out;
    }

    [[nodiscard]] const std::unordered_map<std::string, std::string>& entries() const { return kv_; }

private:
    static std::string trim(std::string s) {
        usize a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        usize b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }
    static void strip_comment(std::string& s) {
        bool in_str = false;
        for (usize i = 0; i < s.size(); ++i) {
            if (s[i] == '"') in_str = !in_str;
            else if (s[i] == '#' && !in_str) {
                s = s.substr(0, i);
                return;
            }
        }
    }
    static std::string unquote(const std::string& s) {
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
        return s;
    }
    static std::optional<f64> to_double(const std::string& s) {
        f64 v = 0;
        const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
        if (ec == std::errc{}) return v;
        return std::nullopt;
    }

    std::unordered_map<std::string, std::string> kv_;
};

}  // namespace fenix
