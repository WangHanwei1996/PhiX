#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// ConfigFile
//
// Loads a JSONC file (JSON with // line comments) and exposes the parsed
// data as a nlohmann::json object.
//
// Comment stripping correctly ignores '//' that appear inside string values.
//
// Usage:
//   ConfigFile cfg("settings/settings.jsonc");
//   int  nx = cfg["mesh"]["nx"];
//   bool ok = cfg.has("mesh");
//
// The underlying json is accessible via cfg.data() for advanced queries.
// ---------------------------------------------------------------------------

class ConfigFile {
public:
    /// Load and parse a JSONC file.  Throws std::runtime_error on failure.
    explicit ConfigFile(const std::string& path);

    /// Construct from command-line arguments.
    /// argv[1] is used as the path if provided; otherwise defaultPath is used.
    /// Prints an error and calls std::exit(1) on failure.
    static ConfigFile fromArgs(int argc, char* argv[],
                               const std::string& defaultPath = "settings/settings.jsonc");

    /// Access a top-level key (read-only).
    const nlohmann::json& operator[](const std::string& key) const;

    /// Check whether a top-level key exists.
    bool has(const std::string& key) const;

    /// Direct access to the underlying json object (for nested / advanced use).
    const nlohmann::json& data() const { return data_; }

private:
    nlohmann::json data_;

    /// Strip // line comments from a single line, respecting quoted strings.
    static std::string stripLineComment(const std::string& line);
};

} // namespace IO
} // namespace PhiX
