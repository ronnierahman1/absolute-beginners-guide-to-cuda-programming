// csv_head.cpp — tiny CSV preview
// Usage:
//   g++ -O2 -std=c++17 csv_head.cpp -o csv_head
//   ./csv_head data.csv [--rows 10] [--sep auto|,|;|\t|" " ] [--no-header]
//   ./csv_head data.csv --rows 20 --sep auto
//
// Prints: detected separator, header (if present), column count, and first N rows.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <cctype>
#include <cstdlib>
using namespace std;

static char guess_sep(const string& line) {
    // Score common separators
    const vector<char> cands = {',',';','\t',' '};
    array<int, 256> cnt{}; for (unsigned char ch: line) cnt[ch]++;
    char best = ','; int bestc = -1;
    for (char c: cands) if (cnt[(unsigned char)c] > bestc) best = c, bestc = cnt[(unsigned char)c];
    return bestc <= 0 ? ',' : best;
}

static vector<string> split_line(const string& s, char sep) {
    vector<string> out;
    if (sep==' ') { // collapse spaces
        string tok; bool in=false;
        for (char ch: s) {
            if (isspace((unsigned char)ch)) { if (in) { out.push_back(tok); tok.clear(); in=false; } }
            else { tok.push_back(ch); in=true; }
        }
        if (!tok.empty()) out.push_back(tok);
        return out;
    }
    string tok; for (char ch: s) {
        if (ch==sep) { out.push_back(tok); tok.clear(); }
        else tok.push_back(ch);
    }
    out.push_back(tok);
    return out;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " file.csv [--rows N] [--sep auto|,|;|\\t|\" \"] [--no-header]\n";
        return 1;
    }
    string path = argv[1];
    int rows = 10;
    string sep_opt = "auto";
    bool no_header = false;

    for (int i=2;i<argc;i++) {
        string a = argv[i];
        auto need = [&](const char* flag){ if (i+1>=argc) { cerr<<"Missing value for "<<flag<<"\n"; exit(2);} return string(argv[++i]); };
        if (a=="--rows") rows = max(1, stoi(need("--rows")));
        else if (a=="--sep") sep_opt = need("--sep");
        else if (a=="--no-header") no_header = true;
    }

    ifstream f(path);
    if (!f) { cerr << "Failed to open: " << path << "\n"; return 1; }

    string first;
    if (!getline(f, first)) { cerr << "Empty file.\n"; return 0; }

    char sep = ',';
    if (sep_opt=="auto") sep = guess_sep(first);
    else if (sep_opt=="\\t") sep = '\t';
    else if (sep_opt==" ") sep = ' ';
    else if (sep_opt.size()==1) sep = sep_opt[0];
    else { cerr<<"Bad --sep option\n"; return 2; }

    // Determine header
    auto fields_first = split_line(first, sep);
    auto looks_number = [](const string& s)->bool {
        if (s.empty()) return false;
        char* end=nullptr; strtod(s.c_str(), &end);
        return end && *end=='\0';
    };
    bool header = !no_header;
    if (!no_header) {
        // If most fields look non-numeric, treat as header
        int numeric=0;
        for (auto& x: fields_first) if (looks_number(x)) numeric++;
        header = (numeric * 2 < (int)fields_first.size());
    }

    cout << "Detected separator: " << (sep=='\t' ? "\\t" : string(1,sep)) << "\n";
    cout << "Header present: " << (header ? "yes" : "no") << "\n";

    vector<string> header_row;
    if (header) {
        header_row = fields_first;
        cout << "Columns ("<<header_row.size()<<"):\n";
        for (size_t i=0;i<header_row.size();++i)
            cout << "  ["<<i+1<<"] " << header_row[i] << "\n";
    } else {
        cout << "Columns ("<<fields_first.size()<<")\n";
        // We'll print the first line as data
    }

    cout << "\nPreview (first " << rows << " rows):\n";
    if (!header) {
        // print first line as row 1
        for (size_t i=0;i<fields_first.size();++i) {
            if (i) cout << sep;
            cout << fields_first[i];
        }
        cout << "\n";
        rows--;
    }

    string line; int printed=0;
    while (printed<rows && getline(f, line)) {
        if (line.empty()) continue;
        cout << line << "\n";
        printed++;
    }
    return 0;
}
