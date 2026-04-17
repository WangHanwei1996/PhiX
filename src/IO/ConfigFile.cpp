#include "IO/ConfigFile.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// stripLineComment
//
// Walk the line character by character, tracking whether we are inside a
// double-quoted string (handles \" escape).  A '//' found outside a string
// marks the start of a comment — everything from that point is dropped.
// ---------------------------------------------------------------------------
std::string ConfigFile::stripLineComment(const std::string& line)
{
    bool in_string = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (in_string) {
            if (c == '\\') {
                ++i;            // skip escaped character (e.g. \")
            } else if (c == '"') {
                in_string = false;
            }
        } else {
            if (c == '"') {
                in_string = true;
            } else if (c == '/' && i + 1 < line.size() && line[i + 1] == '/') {
                return line.substr(0, i);   // drop comment tail
            }
        }
    }
    return line;
}

// ---------------------------------------------------------------------------
// fromArgs
// ---------------------------------------------------------------------------
ConfigFile ConfigFile::fromArgs(int argc, char* argv[], const std::string& defaultPath)
{
    const std::string path = (argc >= 2) ? argv[1] : defaultPath;
    try {
        return ConfigFile(path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n"
                  << "  Usage: " << argv[0] << " [path/to/settings.jsonc]\n";
        std::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ConfigFile::ConfigFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("ConfigFile: cannot open \"" + path + "\"");
    }

    std::ostringstream stripped;
    std::string line;
    while (std::getline(file, line)) {
        stripped << stripLineComment(line) << '\n';
    }

    try {
        data_ = nlohmann::json::parse(stripped.str());
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(
            "ConfigFile: JSON parse error in \"" + path + "\": " + e.what());
    }
}

// ---------------------------------------------------------------------------
// operator[]
// ---------------------------------------------------------------------------
const nlohmann::json& ConfigFile::operator[](const std::string& key) const
{
    return data_.at(key);   // throws nlohmann::json::out_of_range if missing
}

// ---------------------------------------------------------------------------
// has
// ---------------------------------------------------------------------------
bool ConfigFile::has(const std::string& key) const
{
    return data_.contains(key);
}

} // namespace IO
} // namespace PhiX
