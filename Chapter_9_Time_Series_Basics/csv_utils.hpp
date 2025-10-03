// csv_utils.hpp
// Tiny header-only CSV loader/saver for 1D numeric series.
// Intended for quick experiments; not a full CSV parser.

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace tinycsv {

// Trim helpers
inline std::string ltrim(std::string s) {
    size_t i = 0; while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
    return s.substr(i);
}
inline std::string rtrim(std::string s) {
    size_t i = s.size(); while (i > 0 && std::isspace((unsigned char)s[i-1])) --i;
    s.resize(i); return s;
}
inline std::string trim(std::string s) { return rtrim(ltrim(std::move(s))); }

// Split on common CSV delimiters
inline void split_fields(const std::string& line, std::vector<std::string>& out) {
    out.clear();
    std::string field;
    std::istringstream ss(line);
    // We accept comma/semicolon/tab/space as delimiters; treat any as separator.
    char c;
    while (ss.get(c)) {
        if (c==',' || c==';' || c=='\t' || c==' ') {
            if (!field.empty()) { out.emplace_back(std::move(field)); field.clear(); }
            // coalesce runs of separators
            while (ss.peek()==',' || ss.peek()==';' || ss.peek()=='\t' || ss.peek()==' ') ss.get();
        } else {
            field.push_back(c);
        }
    }
    if (!field.empty()) out.emplace_back(std::move(field));
}

// Detect if a token looks numeric (very permissive)
inline bool looks_number(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[i]=='+' || s[i]=='-') ++i;
    bool any=false, dot=false;
    for (; i<s.size(); ++i) {
        char c = s[i];
        if (std::isdigit((unsigned char)c)) { any=true; continue; }
        if (c=='.' && !dot) { dot=true; continue; }
        if (c=='e' || c=='E') return true; // scientific; assume ok
        return false;
    }
    return any;
}

// Read a single numeric column (1-based col_idx) from CSV.
// If header_row=true, first line is skipped if it doesn't parse as numeric.
inline std::vector<float> read_csv_column(const std::string& path, int col_idx_1based=1, bool header_row=true) {
    if (col_idx_1based <= 0) throw std::invalid_argument("col index must be >= 1");
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Failed to open: " + path);

    std::vector<float> data;
    std::string line;
    bool first = true;
    std::vector<std::string> fields;

    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0]=='#') continue;

        split_fields(line, fields);
        if ((int)fields.size() < col_idx_1based) continue;

        if (first && header_row) {
            // If the chosen column does NOT look numeric, skip it as header.
            if (!looks_number(fields[col_idx_1based-1])) { first=false; continue; }
        }
        first = false;

        try {
            data.push_back(std::stof(fields[col_idx_1based-1]));
        } catch (...) {
            // skip non-numeric rows quietly
        }
    }
    return data;
}

// Save one column: y[i]
inline void write_csv_1col(const std::string& path, const std::string& header,
                           const std::vector<float>& y) {
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to open for write: " + path);
    if (!header.empty()) o << header << "\n";
    for (size_t i=0; i<y.size(); ++i) {
        o << y[i] << "\n";
    }
}

// Save two columns: i, y[i] (or any two series of same length)
inline void write_csv_2col(const std::string& path, const std::string& h1, const std::string& h2,
                           const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size()!=b.size()) throw std::invalid_argument("length mismatch");
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to open for write: " + path);
    if (!h1.empty() || !h2.empty()) o << h1 << "," << h2 << "\n";
    for (size_t i=0; i<a.size(); ++i) {
        o << a[i] << "," << b[i] << "\n";
    }
}

// Save index + one column: i, y[i]
inline void write_csv_indexed(const std::string& path, const std::string& h1, const std::string& h2,
                              const std::vector<float>& y) {
    std::ofstream o(path);
    if (!o) throw std::runtime_error("Failed to open for write: " + path);
    if (!h1.empty() || !h2.empty()) o << h1 << "," << h2 << "\n";
    for (size_t i=0; i<y.size(); ++i) {
        o << i << "," << y[i] << "\n";
    }
}

} // namespace tinycsv
