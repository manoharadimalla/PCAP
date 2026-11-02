#include "runway/instance_io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace runway {

SeparationMatrix get_default_separation_matrix() {
    SeparationMatrix mat;
    // Default matrix as specified in Section 4.2:
    // Rows: Leader (0: ARR-H, 1: ARR-M, 2: ARR-L, 3: DEP-H, 4: DEP-M, 5: DEP-L)
    // Cols: Follower (0: ARR-H, 1: ARR-M, 2: ARR-L, 3: DEP-H, 4: DEP-M, 5: DEP-L)
    const float raw[6][6] = {
        { 96.0f, 157.0f, 196.0f, 60.0f, 60.0f, 60.0f },
        { 60.0f,  69.0f, 131.0f, 60.0f, 60.0f, 60.0f },
        { 60.0f,  69.0f,  82.0f, 60.0f, 60.0f, 60.0f },
        { 75.0f,  75.0f,  75.0f, 60.0f, 94.0f, 118.0f },
        { 75.0f,  75.0f,  75.0f, 60.0f, 60.0f, 79.0f },
        { 75.0f,  75.0f,  75.0f, 60.0f, 60.0f, 60.0f }
    };
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            mat.set(r, c, raw[r][c]);
        }
    }
    mat.check_triangle_inequality();
    return mat;
}

bool load_separation_matrix(const std::string& path, SeparationMatrix& mat) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Warning: Cannot open separation file: " << path << ". Using default.\n";
        mat = get_default_separation_matrix();
        return false;
    }
    std::string line;
    int row = 0;
    while (std::getline(fin, line) && row < 6) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string token;
        int col = 0;
        while (std::getline(ss, token, ',') && col < 6) {
            try {
                mat.set(row, col, std::stof(token));
            } catch (...) {
                mat.set(row, col, 60.0f);
            }
            col++;
        }
        if (col == 6) {
            row++;
        }
    }
    mat.check_triangle_inequality();
    return (row == 6);
}

bool save_separation_matrix(const std::string& path, const SeparationMatrix& mat) {
    std::ofstream fout(path);
    if (!fout.is_open()) return false;
    fout << "# RunwayMC Separation Matrix (6x6)\n";
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            fout << mat.get(r, c) << (c + 1 < 6 ? "," : "\n");
        }
    }
    return true;
}

static int parse_op(const std::string& s) {
    if (s == "ARR" || s == "0" || s == "arr") return OP_ARR;
    return OP_DEP;
}

static int parse_wake(const std::string& s) {
    if (s == "H" || s == "0" || s == "Heavy" || s == "heavy") return WAKE_HEAVY;
    if (s == "M" || s == "1" || s == "Medium" || s == "medium") return WAKE_MEDIUM;
    return WAKE_LIGHT;
}

static int parse_prio(const std::string& s) {
    if (s == "High" || s == "high" || s == "1") return PRIO_HIGH;
    if (s == "Emergency" || s == "emergency" || s == "2") return PRIO_EMERGENCY;
    return PRIO_NORMAL;
}

bool load_instance(const std::string& path, AircraftData& data) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open instance file: " << path << "\n";
        return false;
    }
    data.clear();
    std::string line;
    // Check header
    if (!std::getline(fin, line)) return false;

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ',')) {
            parts.push_back(token);
        }
        if (parts.size() < 8) continue;

        Aircraft ac;
        ac.id = std::stoi(parts[0]);
        ac.op = parse_op(parts[1]);
        ac.wake = parse_wake(parts[2]);
        ac.cls = make_class(ac.op, ac.wake);
        ac.ready = std::stof(parts[3]);
        ac.target = std::stof(parts[4]);
        ac.latest = std::stof(parts[5]);
        ac.prio = parse_prio(parts[6]);
        ac.weight = std::stof(parts[7]);

        data.add(ac);
    }
    return (data.n > 0);
}

bool save_instance(const std::string& path, const AircraftData& data) {
    std::ofstream fout(path);
    if (!fout.is_open()) return false;
    fout << "id,op,wake,ready,target,latest,priority,weight\n";
    for (int i = 0; i < data.n; ++i) {
        std::string op_str = (data.op[i] == OP_ARR) ? "ARR" : "DEP";
        std::string wake_str = (data.wake[i] == WAKE_HEAVY) ? "H" : (data.wake[i] == WAKE_MEDIUM ? "M" : "L");
        std::string prio_str = (data.prio[i] == PRIO_NORMAL) ? "Normal" : (data.prio[i] == PRIO_HIGH ? "High" : "Emergency");
        fout << data.id[i] << ","
             << op_str << ","
             << wake_str << ","
             << data.ready[i] << ","
             << data.target[i] << ","
             << data.latest[i] << ","
             << prio_str << ","
             << data.weight[i] << "\n";
    }
    return true;
}

bool load_schedule(const std::string& path, std::vector<int>& perm) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    perm.clear();
    std::string token;
    while (fin >> token) {
        std::stringstream ss(token);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                try {
                    perm.push_back(std::stoi(item));
                } catch (...) {}
            }
        }
    }
    return !perm.empty();
}

bool save_schedule(const std::string& path, const std::vector<int>& perm) {
    std::ofstream fout(path);
    if (!fout.is_open()) return false;
    for (size_t i = 0; i < perm.size(); ++i) {
        fout << perm[i] << (i + 1 < perm.size() ? "," : "\n");
    }
    return true;
}

} // namespace runway
