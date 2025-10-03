// csv_to_plot.cpp — emit timestamp,value for plotting
// Usage:
//   g++ -O2 -std=c++17 csv_to_plot.cpp -o csv_to_plot
//   ./csv_to_plot in.csv out.csv [--sep auto|,|;|\t|" "] [--col-t 1] [--col-y 2]
//                                 [--no-header] [--to-unix]
// Examples:
//   ./csv_to_plot prices.csv prices_plot.csv --col-t 1 --col-y 5 --to-unix
//   ./csv_to_plot sensor.csv sensor_plot.csv --sep auto

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <cctype>
#include <algorithm>
using namespace std;

static char guess_sep(const string& line) {
    const vector<char> cands = {',',';','\t',' '};
    array<int, 256> cnt{}; for (unsigned char ch: line) cnt[ch]++;
    char best=','; int bestc=-1; for (char c: cands) if (cnt[(unsigned char)c]>bestc){best=c; bestc=cnt[(unsigned char)c];}
    return bestc<=0 ? ',' : best;
}
static vector<string> split_line(const string& s, char sep) {
    vector<string> out; if (sep==' ') { string tok; bool in=false; for (char ch: s){ if(isspace((unsigned char)ch)){ if(in){out.push_back(tok); tok.clear(); in=false;} } else { tok.push_back(ch); in=true; } } if(!tok.empty()) out.push_back(tok); return out; }
    string tok; for (char ch: s) { if (ch==sep) { out.push_back(tok); tok.clear(); } else tok.push_back(ch); } out.push_back(tok); return out;
}

static bool is_int(const string& s) {
    if (s.empty()) return false;
    size_t i=0; if (s[0]=='+'||s[0]=='-') i=1;
    if (i>=s.size()) return false;
    for (; i<s.size(); ++i) if (!isdigit((unsigned char)s[i])) return false;
    return true;
}

static bool parse_iso8601(const string& s, time_t& out_epoch) {
    // Accept forms like: 2024-09-12, 2024-09-12T15:42:10Z, 2024-09-12 15:42:10
    // We implement a permissive parser without timezone offsets (assume UTC if 'Z', else local).
    // Preferred for plotting: convert to epoch (UTC).
    tm t{}; int Y,M,D,h=0,m=0; double sec=0.0;
    char Z=' ';
    // Try full with time and optional 'T'/' ' and 'Z'
    // 1) YYYY-MM-DD[T ]hh:mm:ssZ?
    {
        char c1, c2, c3, c4;
        if (sscanf(s.c_str(), "%d%c%d%c%d%*[Tt ]%d%c%d%c%lf%c",
                   &Y,&c1,&M,&c2,&D,&h,&c3,&m,&c4,&sec,&Z) >= 5) {
            if (c1=='-'&&c2=='-'&&c3==':'&&c4==':') {
                t = tm{};
                t.tm_year = Y-1900; t.tm_mon = M-1; t.tm_mday = D;
                t.tm_hour = h; t.tm_min = m; t.tm_sec = (int)floor(sec);
                // Convert as UTC if Z present, else assume local time then convert to epoch
                // We will treat both as UTC to keep it simple for plotting.
                // Use timegm if available; fallback: setenv("TZ","UTC"), tzset(), then mktime.
                #if defined(_GNU_SOURCE) || defined(__USE_MISC)
                time_t epoch = timegm(&t);
                #else
                // Portable fallback: temporarily force TZ=UTC
                char* oldtz = getenv("TZ");
                setenv("TZ","UTC",1); tzset();
                time_t epoch = mktime(&t);
                if (oldtz) setenv("TZ", oldtz, 1); else unsetenv("TZ");
                tzset();
                #endif
                out_epoch = epoch + (time_t)0; // ignore fractional sec
                return true;
            }
        }
    }
    // 2) YYYY-MM-DD only
    {
        char c1, c2;
        if (sscanf(s.c_str(), "%d%c%d%c%d", &Y,&c1,&M,&c2,&D)==5 && c1=='-'&&c2=='-') {
            t = tm{}; t.tm_year = Y-1900; t.tm_mon = M-1; t.tm_mday = D;
            #if defined(_GNU_SOURCE) || defined(__USE_MISC)
            time_t epoch = timegm(&t);
            #else
            char* oldtz = getenv("TZ");
            setenv("TZ","UTC",1); tzset();
            time_t epoch = mktime(&t);
            if (oldtz) setenv("TZ", oldtz, 1); else unsetenv("TZ");
            tzset();
            #endif
            out_epoch = epoch;
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " in.csv out.csv [--sep auto|,|;|\\t|\" \"] [--col-t 1] [--col-y 2] [--no-header] [--to-unix]\n";
        return 1;
    }
    string in = argv[1], out = argv[2];
    string sep_opt = "auto";
    int col_t = 1, col_y = 2;
    bool no_header = false, to_unix = false;

    for (int i=3;i<argc;i++) {
        string a = argv[i];
        auto need = [&](const char* flag){ if (i+1>=argc) { cerr<<"Missing value for "<<flag<<"\n"; exit(2);} return string(argv[++i]); };
        if (a=="--sep") sep_opt = need("--sep");
        else if (a=="--col-t") col_t = max(1, stoi(need("--col-t")));
        else if (a=="--col-y") col_y = max(1, stoi(need("--col-y")));
        else if (a=="--no-header") no_header = true;
        else if (a=="--to-unix") to_unix = true;
    }

    ifstream f(in);
    if (!f) { cerr << "Failed to open: " << in << "\n"; return 1; }
    ofstream g(out);
    if (!g) { cerr << "Failed to write: " << out << "\n"; return 1; }

    string first;
    if (!getline(f, first)) { cerr << "Empty file.\n"; return 0; }
    char sep = ',';
    if (sep_opt=="auto") sep = guess_sep(first);
    else if (sep_opt=="\\t") sep = '\t';
    else if (sep_opt==" ") sep = ' ';
    else if (sep_opt.size()==1) sep = sep_opt[0];

    vector<string> fields = split_line(first, sep);
    auto join = [&](const vector<string>& v, char s){ string o; for (size_t i=0;i<v.size();++i){ if(i)o.push_back(s); o+=v[i]; } return o; };

    auto looks_number = [](const string& s)->bool{ if (s.empty()) return false; char* e=nullptr; strtod(s.c_str(), &e); return e && *e=='\0'; };

    bool header = !no_header;
    if (!no_header) {
        // If target columns don't look numeric (col_y) or col_t looks non-numeric, assume header
        bool ynum = (col_y <= (int)fields.size()) ? looks_number(fields[col_y-1]) : false;
        bool tnum = (col_t <= (int)fields.size()) ? looks_number(fields[col_t-1]) : false;
        header = !(ynum && (tnum || !tnum)); // be permissive; treat first row as header usually
    }

    // Write header for plot-friendly output
    g << "timestamp,value\n";

    if (!header) {
        // Process first line as data
        if ((int)fields.size() >= max(col_t,col_y)) {
            string ts = fields[col_t-1];
            string ys = fields[col_y-1];

            // Normalize timestamp if requested
            if (to_unix) {
                time_t ep=0;
                if (is_int(ts)) {
                    // If ms since epoch, reduce to sec
                    long long v = atoll(ts.c_str());
                    if (v > 100000000000LL) v /= 1000; // crude ms→s
                    g << v << "," << ys << "\n";
                } else if (parse_iso8601(ts, ep)) {
                    g << ep << "," << ys << "\n";
                } else {
                    // Fallback: skip
                }
            } else {
                g << ts << "," << ys << "\n";
            }
        }
    }

    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        auto cols = split_line(line, sep);
        if ((int)cols.size() < max(col_t,col_y)) continue;

        string ts = cols[col_t-1];
        string ys = cols[col_y-1];

        if (to_unix) {
            time_t ep=0;
            if (is_int(ts)) {
                long long v = atoll(ts.c_str());
                if (v > 100000000000LL) v /= 1000; // ms→s
                g << v << "," << ys << "\n";
            } else if (parse_iso8601(ts, ep)) {
                g << ep << "," << ys << "\n";
            } else {
                // skip unparseable
            }
        } else {
            g << ts << "," << ys << "\n";
        }
    }

    cerr << "Wrote: " << out << " (timestamp,value)\n";
    return 0;
}
