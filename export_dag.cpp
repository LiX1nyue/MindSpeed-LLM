#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace std;

// 节点类型常量
const int NODE_TYPE_NONE = 0;
const int NODE_TYPE_COLLECTIVE = 1; // 集合通信节点
const int NODE_TYPE_COMPUTE = 2;    // 计算节点
const int NODE_TYPE_P2P = 3;        // 单对单通信节点

// 基类 Node
struct Node {
    int id;                 // 节点全局唯一编号 (1 ~ n)
    int node_type;          // 1: 集合通信, 2: 计算, 3: 单对单通信
    string name;            // 算子名称
    double ts;              // 开始时间 (微秒 us)
    double dur;             // 持续耗时 (微秒 us)
    long long pid;          // 进程号
    long long tid;          // 线程/轨道号
    long long stream_id;    // 硬件流编号
    vector<int> children;   // 子节点数组：指向的下一个(后序)节点编号

    Node() {
        id = 0;
        node_type = NODE_TYPE_NONE;
        ts = 0.0;
        dur = 0.0;
        pid = -1;
        tid = -1;
        stream_id = -1;
    }
};

// 1. 集合通信节点 (继承 Node)
struct CollectiveCommunicationNode : public Node {
    string comm_type;       // ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, ALL_TO_ALL, BROADCAST
    string data_type;       // 数据类型，如 BF16, FP16, FP32
    long long count;        // 元素个数
    long long data_bytes;   // 通信字节数
    double bandwidth_gbps;  // 估算通信带宽 (GB/s)
    string group_name;      // 通信域名称

    CollectiveCommunicationNode() : Node() {
        node_type = NODE_TYPE_COLLECTIVE;
        comm_type = "UNKNOWN";
        data_type = "UNKNOWN";
        count = 0;
        data_bytes = 0;
        bandwidth_gbps = 0.0;
        group_name = "NONE";
    }
};

// 2. 计算节点 (继承 Node)
struct ComputeNode : public Node {
    double compute_time_us; // 计算耗时 (微秒 us)
    string task_type;       // 硬件计算核类型，如 AI_CORE, AI_VECTOR_CORE

    ComputeNode() : Node() {
        node_type = NODE_TYPE_COMPUTE;
        compute_time_us = 0.0;
        task_type = "AI_CORE";
    }
};

// 3. 单对单通信节点 (继承 Node)
struct P2PCommunicationNode : public Node {
    string p2p_type;        // SEND 或 RECV
    string data_type;       // 数据类型
    long long count;        // 元素个数
    long long data_bytes;   // 通信字节数
    double bandwidth_gbps;  // 估算通信带宽 (GB/s)
    string group_name;      // 通信域名称

    P2PCommunicationNode() : Node() {
        node_type = NODE_TYPE_P2P;
        p2p_type = "UNKNOWN";
        data_type = "UNKNOWN";
        count = 0;
        data_bytes = 0;
        bandwidth_gbps = 0.0;
        group_name = "NONE";
    }
};

// 跨流同步信号记录 (由 flow 事件 s->f 或 Notify_Record->Notify_Wait 产生)
struct SyncPoint {
    string sync_id;
    long long pid;
    long long tid;
    long long stream_id;
    double ts;
    bool is_start; // true: 起点(s / Record), false: 终点(f / Wait)
};

// 三个全局 vector 暂存识别出的三类节点
vector<CollectiveCommunicationNode> g_coll_nodes;
vector<ComputeNode> g_comp_nodes;
vector<P2PCommunicationNode> g_p2p_nodes;

// 暂存跨流同步信号
vector<SyncPoint> g_sync_points;

// 元数据过滤集合：用于去重（锁定 Ascend Hardware 和 HCCL，拉黑 Plane 拆解小算子轨道）
set<long long> g_hw_pids;
set<long long> g_hccl_pids;
set<pair<long long, long long> > g_plane_tids;

// ============================================================================
// 基础字符串与简单 JSON 字段提取工具函数
// ============================================================================

string toLowerStr(const string& s) {
    string res = s;
    for (size_t i = 0; i < res.size(); ++i) {
        if (res[i] >= 'A' && res[i] <= 'Z') {
            res[i] = res[i] - 'A' + 'a';
        }
    }
    return res;
}

string removeUnderscoreAndDash(const string& s) {
    string res = "";
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '_' && s[i] != '-') {
            char c = s[i];
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
            res += c;
        }
    }
    return res;
}

// 在单行 JSON 对象中查找字段 "key": 后面紧跟的值起始下标
int findKeyValuePos(const string& line, const string& key) {
    string pattern = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = line.find(pattern, pos)) != string::npos) {
        size_t p = pos + pattern.size();
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
        if (p < line.size() && line[p] == ':') {
            p++;
            while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) p++;
            return (int)p;
        }
        pos += pattern.size();
    }
    return -1;
}

// 提取字符串类型字段（若为数字也转为字符串返回）
bool getJsonString(const string& line, const string& key, string& out_val) {
    int p = findKeyValuePos(line, key);
    if (p < 0 || p >= (int)line.size()) return false;
    if (line[p] == '"') {
        p++;
        string res = "";
        while (p < (int)line.size()) {
            if (line[p] == '\\' && p + 1 < (int)line.size()) {
                res += line[p + 1];
                p += 2;
            } else if (line[p] == '"') {
                out_val = res;
                return true;
            } else {
                res += line[p];
                p++;
            }
        }
        return false;
    } else {
        int start = p;
        while (p < (int)line.size() && line[p] != ',' && line[p] != '}' && line[p] != ' ' && line[p] != '\t') {
            p++;
        }
        string token = line.substr(start, p - start);
        if (token.empty() || token == "null" || token == "true" || token == "false") return false;
        out_val = token;
        return true;
    }
}

// 提取浮点数类型字段
bool getJsonDouble(const string& line, const string& key, double& out_val) {
    string s;
    if (!getJsonString(line, key, s)) return false;
    char* endptr = NULL;
    double v = strtod(s.c_str(), &endptr);
    if (endptr == s.c_str()) return false;
    out_val = v;
    return true;
}

// 提取长整型字段
bool getJsonLongLong(const string& line, const string& key, long long& out_val) {
    string s;
    if (!getJsonString(line, key, s)) return false;
    char* endptr = NULL;
    long long v = strtoll(s.c_str(), &endptr, 10);
    if (endptr == s.c_str()) return false;
    out_val = v;
    return true;
}

// 提取硬件流编号 stream_id
long long extractStreamId(const string& line, long long fallback_tid) {
    const char* keys[] = {
        "Stream Id", "stream_id", "stream id", "stream", "Stream",
        "Physic Stream Id", "physic_stream_id", "Physical Stream Id", "physical stream id"
    };
    int num_keys = 9;
    for (int i = 0; i < num_keys; ++i) {
        long long sid = -1;
        if (getJsonLongLong(line, keys[i], sid)) {
            return sid;
        }
    }
    return fallback_tid;
}

// 根据数据类型计算单个元素字节数
long long getDataTypeBytes(const string& dtype) {
    string low = toLowerStr(dtype);
    if (low.find("bf16") != string::npos || low.find("bfloat16") != string::npos ||
        low.find("fp16") != string::npos || low.find("float16") != string::npos ||
        low.find("int16") != string::npos) {
        return 2;
    }
    if (low.find("fp32") != string::npos || low.find("float32") != string::npos ||
        low == "float" || low.find("int32") != string::npos) {
        return 4;
    }
    if (low.find("fp64") != string::npos || low.find("double") != string::npos ||
        low.find("int64") != string::npos) {
        return 8;
    }
    if (low.find("int8") != string::npos || low.find("uint8") != string::npos) {
        return 1;
    }
    return 2; // 默认按 BF16/FP16 (2字节) 估算
}

// ============================================================================
// 一：翻译 JSON
// ============================================================================

// 1. 将 JSON 文件中的 traceEvents 条目翻译为单行字符串数组
bool translateJsonFileToLines(const string& filename, vector<string>& out_lines) {
    ifstream fin(filename.c_str());
    if (!fin.is_open()) {
        return false;
    }

    string content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    fin.close();

    size_t pos = content.find("\"traceEvents\"");
    if (pos == string::npos) {
        return false;
    }
    pos = content.find('[', pos);
    if (pos == string::npos) {
        return false;
    }

    int brace_depth = 0;
    bool in_string = false;
    bool escape = false;
    string current_obj = "";

    for (size_t i = pos + 1; i < content.size(); ++i) {
        char c = content[i];

        if (brace_depth == 0) {
            if (!in_string && c == ']') {
                break; // traceEvents 数组结束
            }
            if (!in_string && c == '{') {
                brace_depth = 1;
                current_obj = "{";
            }
            continue;
        }

        // 把换行和制表符替换为空格，保证单条记录在一行内
        if (!in_string && (c == '\n' || c == '\r' || c == '\t')) {
            if (!current_obj.empty() && current_obj[current_obj.size() - 1] != ' ') {
                current_obj += ' ';
            }
            continue;
        }

        current_obj += c;

        if (escape) {
            escape = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escape = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }

        if (!in_string) {
            if (c == '{') {
                brace_depth++;
            } else if (c == '}') {
                brace_depth--;
                if (brace_depth == 0) {
                    out_lines.push_back(current_obj);
                    current_obj = "";
                }
            }
        }
    }
    return true;
}

// 预扫描元数据 ("ph": "M")，识别 Ascend Hardware / HCCL 进程号以及需要去重拉黑的 Plane 轨道
void scanMetadataLine(const string& line) {
    string ph;
    if (!getJsonString(line, "ph", ph) || ph != "M") return;

    string meta_name;
    getJsonString(line, "name", meta_name);
    long long pid = -1, tid = -1;
    getJsonLongLong(line, "pid", pid);
    getJsonLongLong(line, "tid", tid);

    // 提取 args 里的 "name"
    size_t args_pos = line.find("\"args\"");
    string arg_name = "";
    if (args_pos != string::npos) {
        getJsonString(line.substr(args_pos), "name", arg_name);
    }

    if (meta_name == "process_name") {
        if (arg_name.find("Ascend Hardware") != string::npos) {
            g_hw_pids.insert(pid);
        } else if (arg_name.find("HCCL") != string::npos) {
            g_hccl_pids.insert(pid);
        }
    } else if (meta_name == "thread_name") {
        if (arg_name.find("Plane") == 0) {
            g_plane_tids.insert(make_pair(pid, tid));
        }
    }
}

// 收集跨流同步信号 (flow 事件 "ph": "s"/"f" 或 Notify_Record/Notify_Wait)
void collectSyncSignalFromLine(const string& line) {
    string ph;
    if (!getJsonString(line, "ph", ph)) return;

    double ts = 0.0;
    if (!getJsonDouble(line, "ts", ts)) return;

    long long pid = -1, tid = -1;
    getJsonLongLong(line, "pid", pid);
    getJsonLongLong(line, "tid", tid);
    long long sid = extractStreamId(line, tid);

    // 情况 1: 标准 flow 连线事件 ("ph": "s" 起点, "ph": "f" 终点)
    if (ph == "s" || ph == "f") {
        string flow_id;
        if (!getJsonString(line, "id", flow_id)) {
            getJsonString(line, "bind_id", flow_id);
        }
        if (!flow_id.empty()) {
            SyncPoint sp;
            sp.sync_id = "flow_" + flow_id;
            sp.pid = pid;
            sp.tid = tid;
            sp.stream_id = sid;
            sp.ts = ts;
            sp.is_start = (ph == "s");
            g_sync_points.push_back(sp);
        }
        return;
    }

    // 情况 2: 硬件流同步事件 Notify_Record / Notify_Wait (若有 wait_id / notify_id)
    if (ph == "X") {
        string name;
        getJsonString(line, "name", name);
        string low = toLowerStr(name);
        if (low.find("notify_record") != string::npos || low.find("notify_wait") != string::npos) {
            string notify_id;
            if (!getJsonString(line, "notify_id", notify_id)) {
                getJsonString(line, "event_id", notify_id);
            }
            if (!notify_id.empty()) {
                SyncPoint sp;
                sp.sync_id = "notify_" + notify_id;
                sp.pid = pid;
                sp.tid = tid;
                sp.stream_id = sid;
                sp.ts = ts;
                sp.is_start = (low.find("notify_record") != string::npos);
                g_sync_points.push_back(sp);
            }
        }
    }
}

// 判断函数 1：是否代表集合通信节点 (自动过滤重复的 Plane 小算子和 CPU 算子)
bool isCollectiveCommunicationNode(const string& line, CollectiveCommunicationNode& node) {
    string ph;
    if (!getJsonString(line, "ph", ph) || ph != "X") return false;

    double ts = 0.0, dur = 0.0;
    if (!getJsonDouble(line, "ts", ts)) return false;
    getJsonDouble(line, "dur", dur);

    string name = "", cat = "";
    getJsonString(line, "name", name);
    getJsonString(line, "cat", cat);
    string low_name = toLowerStr(name);
    string low_cat = toLowerStr(cat);

    // 排除 CPU 侧算子
    if (low_cat == "cpu_op") return false;

    long long pid = -1, tid = -1;
    getJsonLongLong(line, "pid", pid);
    getJsonLongLong(line, "tid", tid);

    // 如果存在 HCCL 独立进程，只保留 HCCL 进程下非 Plane 轨道的通信大算子，避免与硬件流重复计时
    if (!g_hccl_pids.empty()) {
        if (g_hccl_pids.find(pid) == g_hccl_pids.end()) return false;
        if (g_plane_tids.find(make_pair(pid, tid)) != g_plane_tids.end()) return false;
    }

    // 必须是 hcom 或 hccl 相关通信事件
    bool has_comm_mark = (low_name.find("hcom") != string::npos ||
                          low_name.find("hccl") != string::npos ||
                          low_cat.find("hccl") != string::npos ||
                          low_cat.find("communication") != string::npos);
    if (!has_comm_mark) return false;

    // 排除底层拆解微步骤 (Notify_Wait / Memcpy / Reduce_Inline / RDMASend 等) 和单对单通信 (Send / Recv)
    if (low_name.find("notify") != string::npos ||
        low_name.find("memcpy") != string::npos ||
        low_name.find("reduce_inline") != string::npos ||
        low_name.find("rdma") != string::npos ||
        low_name.find("send") != string::npos ||
        low_name.find("recv") != string::npos ||
        low_name.find("receive") != string::npos) {
        return false;
    }

    // 识别具体的集合通信类型
    string norm = removeUnderscoreAndDash(name);
    string comm_type = "";
    if (norm.find("allreduce") != string::npos) comm_type = "ALL_REDUCE";
    else if (norm.find("allgather") != string::npos) comm_type = "ALL_GATHER";
    else if (norm.find("reducescatter") != string::npos) comm_type = "REDUCE_SCATTER";
    else if (norm.find("alltoall") != string::npos) comm_type = "ALL_TO_ALL";
    else if (norm.find("broadcast") != string::npos) comm_type = "BROADCAST";
    else return false;

    node.name = name;
    node.ts = ts;
    node.dur = dur;
    node.pid = pid;
    node.tid = tid;
    node.stream_id = extractStreamId(line, tid);
    node.comm_type = comm_type;

    getJsonString(line, "data_type", node.data_type);
    getJsonLongLong(line, "count", node.count);
    getJsonString(line, "group_name", node.group_name);

    node.data_bytes = node.count * getDataTypeBytes(node.data_type);
    if (node.dur > 0.0 && node.data_bytes > 0) {
        // 字节 / 微秒 -> GB/s: (bytes / 1e9) / (dur / 1e6) = (bytes / dur) * 1e-3
        node.bandwidth_gbps = ((double)node.data_bytes / node.dur) * 1e-3;
    } else {
        getJsonDouble(line, "Bandwidth(GB/s)", node.bandwidth_gbps);
    }

    return true;
}

// 判断函数 2：是否代表 NPU 计算节点 (排除 CPU 下发算子和通信算子)
bool isComputeNode(const string& line, ComputeNode& node) {
    string ph;
    if (!getJsonString(line, "ph", ph) || ph != "X") return false;

    double ts = 0.0, dur = 0.0;
    if (!getJsonDouble(line, "ts", ts)) return false;
    getJsonDouble(line, "dur", dur);

    string name = "", cat = "";
    getJsonString(line, "name", name);
    getJsonString(line, "cat", cat);
    string low_name = toLowerStr(name);
    string low_cat = toLowerStr(cat);

    // 排除 CPU 下发算子和通信事件
    if (low_cat == "cpu_op") return false;
    if (low_name.find("hcom") != string::npos || low_name.find("hccl") != string::npos) return false;
    if (low_cat.find("hccl") != string::npos || low_cat.find("communication") != string::npos) return false;
    if (low_name.find("notify_") != string::npos || low_name.find("memcpy") != string::npos) return false;

    long long pid = -1, tid = -1;
    getJsonLongLong(line, "pid", pid);
    getJsonLongLong(line, "tid", tid);

    // 若存在 Ascend Hardware 进程标识，则只从硬件进程取计算节点
    if (!g_hw_pids.empty() && g_hw_pids.find(pid) == g_hw_pids.end()) {
        return false;
    }

    // 检查 Task Type（若存在则要求属于计算核类型）
    string task_type = "";
    if (getJsonString(line, "Task Type", task_type)) {
        if (task_type != "AI_CORE" && task_type != "AI_VECTOR_CORE" &&
            task_type != "MIX_AIC" && task_type != "MIX_AIV" && task_type != "AI_CPU") {
            return false;
        }
    } else {
        // 没有 Task Type 时，要求 cat 为 Kernel / operator / compute
        if (low_cat.find("kernel") == string::npos &&
            low_cat.find("operator") == string::npos &&
            low_cat.find("compute") == string::npos) {
            return false;
        }
        task_type = "AI_CORE";
    }

    node.name = name;
    node.ts = ts;
    node.dur = dur;
    node.compute_time_us = dur;
    node.pid = pid;
    node.tid = tid;
    node.stream_id = extractStreamId(line, tid);
    node.task_type = task_type;
    return true;
}

// 判断函数 3：是否代表 NPU 单对单通信节点 (Send / Recv)
bool isP2PCommunicationNode(const string& line, P2PCommunicationNode& node) {
    string ph;
    if (!getJsonString(line, "ph", ph) || ph != "X") return false;

    double ts = 0.0, dur = 0.0;
    if (!getJsonDouble(line, "ts", ts)) return false;
    getJsonDouble(line, "dur", dur);

    string name = "", cat = "";
    getJsonString(line, "name", name);
    getJsonString(line, "cat", cat);
    string low_name = toLowerStr(name);
    string low_cat = toLowerStr(cat);

    if (low_cat == "cpu_op") return false;

    long long pid = -1, tid = -1;
    getJsonLongLong(line, "pid", pid);
    getJsonLongLong(line, "tid", tid);

    // 若存在 HCCL 进程，只从 HCCL 非 Plane 轨道取，排除 Plane 轨道内的 RDMASend 等子步骤
    if (!g_hccl_pids.empty()) {
        if (g_hccl_pids.find(pid) == g_hccl_pids.end()) return false;
        if (g_plane_tids.find(make_pair(pid, tid)) != g_plane_tids.end()) return false;
    }

    bool has_comm_mark = (low_name.find("hcom") != string::npos ||
                          low_name.find("hccl") != string::npos ||
                          low_cat.find("hccl") != string::npos ||
                          low_cat.find("communication") != string::npos);
    if (!has_comm_mark) return false;

    // 排除 Plane 子步骤中的 RDMASend / Notify / Memcpy
    if (low_name.find("rdma") != string::npos ||
        low_name.find("notify") != string::npos ||
        low_name.find("memcpy") != string::npos) {
        return false;
    }

    string p2p_type = "";
    if (low_name.find("send") != string::npos) {
        p2p_type = "SEND";
    } else if (low_name.find("recv") != string::npos || low_name.find("receive") != string::npos) {
        p2p_type = "RECV";
    } else {
        return false;
    }

    node.name = name;
    node.ts = ts;
    node.dur = dur;
    node.pid = pid;
    node.tid = tid;
    node.stream_id = extractStreamId(line, tid);
    node.p2p_type = p2p_type;

    getJsonString(line, "data_type", node.data_type);
    getJsonLongLong(line, "count", node.count);
    getJsonString(line, "group_name", node.group_name);

    node.data_bytes = node.count * getDataTypeBytes(node.data_type);
    if (node.dur > 0.0 && node.data_bytes > 0) {
        node.bandwidth_gbps = ((double)node.data_bytes / node.dur) * 1e-3;
    } else {
        getJsonDouble(line, "Bandwidth(GB/s)", node.bandwidth_gbps);
    }

    return true;
}

// 2. 逐条读取单行字符串，依次调用 3 个判断函数，存入对应的 3 个 vector
void parseLinesToNodes(const vector<string>& lines) {
    // 先扫一遍元数据行 ("ph": "M")，以便准确去重
    for (size_t i = 0; i < lines.size(); ++i) {
        scanMetadataLine(lines[i]);
    }

    // 逐条判断节点类型并存入对应的 vector（此时 children 均留空）
    for (size_t i = 0; i < lines.size(); ++i) {
        const string& line = lines[i];

        // 记录可能存在的跨流同步信号
        collectSyncSignalFromLine(line);

        CollectiveCommunicationNode coll_node;
        if (isCollectiveCommunicationNode(line, coll_node)) {
            if (coll_node.stream_id >= 0) {
                g_coll_nodes.push_back(coll_node);
            }
            continue;
        }

        ComputeNode comp_node;
        if (isComputeNode(line, comp_node)) {
            if (comp_node.stream_id >= 0) {
                g_comp_nodes.push_back(comp_node);
            }
            continue;
        }

        P2PCommunicationNode p2p_node;
        if (isP2PCommunicationNode(line, p2p_node)) {
            if (p2p_node.stream_id >= 0) {
                g_p2p_nodes.push_back(p2p_node);
            }
            continue;
        }
    }
}

// ============================================================================
// 二：建边（复杂度 O((N + M) log N)，无平方级操作）
// ============================================================================

// 全局节点指针列表 (按开始时间排序后分配 id = 1 .. total_nodes)
vector<Node*> g_all_nodes;

bool cmpNodePtrByTime(const Node* a, const Node* b) {
    if (a->ts != b->ts) return a->ts < b->ts;
    return a->node_type < b->node_type;
}

// 辅助函数：在已按 ts 升序排列的节点列表 vec 中，二分查找 ts >= target_ts 的第一个节点
Node* findFirstNodeAfter(const vector<Node*>& vec, double target_ts) {
    int l = 0, r = (int)vec.size() - 1;
    int ans = -1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (vec[mid]->ts >= target_ts) {
            ans = mid;
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    return (ans != -1) ? vec[ans] : NULL;
}

// 辅助函数：在已按 ts 升序排列的节点列表 vec 中，二分查找 ts <= target_ts 的最后一个节点
Node* findLastNodeBefore(const vector<Node*>& vec, double target_ts) {
    int l = 0, r = (int)vec.size() - 1;
    int ans = -1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
        if (vec[mid]->ts <= target_ts) {
            ans = mid;
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return (ans != -1) ? vec[ans] : NULL;
}

void addDirectedEdge(Node* u, Node* v) {
    if (u == NULL || v == NULL) return;
    if (u->id == v->id) return;
    u->children.push_back(v->id);
}

void buildDagEdges() {
    // 1. 汇总全部三类节点指针，按开始时间 ts 排序并分配全局唯一 id (1 ~ N)
    g_all_nodes.clear();
    for (size_t i = 0; i < g_coll_nodes.size(); ++i) {
        g_all_nodes.push_back(&g_coll_nodes[i]);
    }
    for (size_t i = 0; i < g_comp_nodes.size(); ++i) {
        g_all_nodes.push_back(&g_comp_nodes[i]);
    }
    for (size_t i = 0; i < g_p2p_nodes.size(); ++i) {
        g_all_nodes.push_back(&g_p2p_nodes[i]);
    }

    sort(g_all_nodes.begin(), g_all_nodes.end(), cmpNodePtrByTime);
    for (size_t i = 0; i < g_all_nodes.size(); ++i) {
        g_all_nodes[i]->id = (int)(i + 1);
    }

    // 2. 按 stream_id 分别组织计算流节点列表与通信流节点列表
    //    因为 g_all_nodes 已经按 ts 排好序，依序 push_back 后每个桶天然按 ts 升序！
    map<long long, vector<Node*> > comp_streams; // 计算流: stream_id -> 计算节点数组
    map<long long, vector<Node*> > comm_streams; // 通信流: stream_id -> 通信节点数组(含集合通信与P2P)
    map<long long, vector<Node*> > any_streams;  // 任意流: stream_id -> 该流上的所有节点
    map<pair<long long, long long>, vector<Node*> > tid_tracks; // (pid, tid) -> 该轨道上的所有节点

    for (size_t i = 0; i < g_all_nodes.size(); ++i) {
        Node* cur = g_all_nodes[i];
        any_streams[cur->stream_id].push_back(cur);
        tid_tracks[make_pair(cur->pid, cur->tid)].push_back(cur);
        if (cur->node_type == NODE_TYPE_COMPUTE) {
            comp_streams[cur->stream_id].push_back(cur);
        } else {
            comm_streams[cur->stream_id].push_back(cur);
        }
    }

    // 3. 建立【同流后序边 (same_stream)】—— 时间复杂度 O(N)
    // (1) 计算节点 -> 同计算流的下一个计算节点 (index + 1)
    for (map<long long, vector<Node*> >::iterator it = comp_streams.begin(); it != comp_streams.end(); ++it) {
        const vector<Node*>& vec = it->second;
        for (size_t i = 0; i + 1 < vec.size(); ++i) {
            addDirectedEdge(vec[i], vec[i + 1]);
        }
    }
    // (2) 集合通信节点 / 单对单通信节点 -> 同通信流的下一个通信节点 (index + 1)
    for (map<long long, vector<Node*> >::iterator it = comm_streams.begin(); it != comm_streams.end(); ++it) {
        const vector<Node*>& vec = it->second;
        for (size_t i = 0; i + 1 < vec.size(); ++i) {
            addDirectedEdge(vec[i], vec[i + 1]);
        }
    }

    // 4. 建立【跨流后序边 (cross_stream)】—— 时间复杂度 O(K log N)
    //    配对相同 sync_id 的起点 (s / Record) 和终点 (f / Wait)
    map<string, SyncPoint> start_points;
    for (size_t i = 0; i < g_sync_points.size(); ++i) {
        if (g_sync_points[i].is_start) {
            start_points[g_sync_points[i].sync_id] = g_sync_points[i];
        }
    }

    for (size_t i = 0; i < g_sync_points.size(); ++i) {
        if (g_sync_points[i].is_start) continue;
        const SyncPoint& ed = g_sync_points[i];
        map<string, SyncPoint>::iterator it = start_points.find(ed.sync_id);
        if (it == start_points.end()) continue;
        const SyncPoint& st = it->second;

        // 优先按 stream_id 查找所在流，若流号未直接匹配则回退按 (pid, tid) 轨道查找
        const vector<Node*>* src_vec = NULL;
        if (any_streams.find(st.stream_id) != any_streams.end()) {
            src_vec = &any_streams[st.stream_id];
        } else if (tid_tracks.find(make_pair(st.pid, st.tid)) != tid_tracks.end()) {
            src_vec = &tid_tracks[make_pair(st.pid, st.tid)];
        }

        const vector<Node*>* dst_vec = NULL;
        if (any_streams.find(ed.stream_id) != any_streams.end()) {
            dst_vec = &any_streams[ed.stream_id];
        } else if (tid_tracks.find(make_pair(ed.pid, ed.tid)) != tid_tracks.end()) {
            dst_vec = &tid_tracks[make_pair(ed.pid, ed.tid)];
        }

        if (src_vec == NULL || dst_vec == NULL) continue;

        // 二分查找起点流上 <= st.ts 的最后一个节点，以及终点流上 >= ed.ts 的第一个节点
        Node* src_node = findLastNodeBefore(*src_vec, st.ts);
        Node* dst_node = findFirstNodeAfter(*dst_vec, ed.ts);
        if (src_node == NULL || dst_node == NULL) continue;
        if (src_node->stream_id == dst_node->stream_id) continue;

        // 情况 A: 计算节点 -> 触发集合通信节点 或 单对单 SEND 节点
        if (src_node->node_type == NODE_TYPE_COMPUTE &&
            (dst_node->node_type == NODE_TYPE_COLLECTIVE || dst_node->node_type == NODE_TYPE_P2P)) {
            addDirectedEdge(src_node, dst_node);
        }
        // 情况 B: 集合通信节点 -> 唤醒等待它数据的计算节点
        else if (src_node->node_type == NODE_TYPE_COLLECTIVE &&
                 dst_node->node_type == NODE_TYPE_COMPUTE) {
            addDirectedEdge(src_node, dst_node);
        }
        // 情况 C: 单对单 RECV 接收节点 -> 唤醒等待接收数据的计算节点
        else if (src_node->node_type == NODE_TYPE_P2P &&
                 dst_node->node_type == NODE_TYPE_COMPUTE) {
            P2PCommunicationNode* p2p = (P2PCommunicationNode*)src_node;
            if (p2p->p2p_type == "RECV") {
                addDirectedEdge(src_node, dst_node);
            }
        }
    }

    // 5. 对每个节点的 children 去重并升序排列
    for (size_t i = 0; i < g_all_nodes.size(); ++i) {
        vector<int>& ch = g_all_nodes[i]->children;
        sort(ch.begin(), ch.end());
        ch.erase(unique(ch.begin(), ch.end()), ch.end());
    }
}

// ============================================================================
// 三：输出 DAG 图到文件
// ============================================================================

bool writeDagToFile(const string& output_filename) {
    ofstream fout(output_filename.c_str());
    if (!fout.is_open()) {
        return false;
    }

    int total_nodes = (int)g_all_nodes.size();
    long long total_edges = 0;
    for (int i = 0; i < total_nodes; ++i) {
        total_edges += (long long)g_all_nodes[i]->children.size();
    }

    // 第一行：总点数 和 总边数
    fout << total_nodes << " " << total_edges << "\n";

    // 随后 n 行：一行完整表示一个节点的信息及其指向的所有子节点
    fout.setf(ios::fixed);
    fout.precision(3);

    for (int i = 0; i < total_nodes; ++i) {
        Node* base = g_all_nodes[i];
        fout << "id=" << base->id << " ";

        if (base->node_type == NODE_TYPE_COLLECTIVE) {
            CollectiveCommunicationNode* n = (CollectiveCommunicationNode*)base;
            fout << "type=COLLECTIVE "
                 << "name=\"" << n->name << "\" "
                 << "comm_type=" << n->comm_type << " "
                 << "stream=" << n->stream_id << " "
                 << "ts=" << n->ts << " "
                 << "dur=" << n->dur << " "
                 << "data_type=" << n->data_type << " "
                 << "count=" << n->count << " "
                 << "data_bytes=" << n->data_bytes << " "
                 << "bandwidth_gbps=" << n->bandwidth_gbps << " ";
        } else if (base->node_type == NODE_TYPE_COMPUTE) {
            ComputeNode* n = (ComputeNode*)base;
            fout << "type=COMPUTE "
                 << "name=\"" << n->name << "\" "
                 << "task_type=" << n->task_type << " "
                 << "stream=" << n->stream_id << " "
                 << "ts=" << n->ts << " "
                 << "compute_time_us=" << n->compute_time_us << " ";
        } else if (base->node_type == NODE_TYPE_P2P) {
            P2PCommunicationNode* n = (P2PCommunicationNode*)base;
            fout << "type=P2P "
                 << "name=\"" << n->name << "\" "
                 << "p2p_type=" << n->p2p_type << " "
                 << "stream=" << n->stream_id << " "
                 << "ts=" << n->ts << " "
                 << "dur=" << n->dur << " "
                 << "data_type=" << n->data_type << " "
                 << "count=" << n->count << " "
                 << "data_bytes=" << n->data_bytes << " "
                 << "bandwidth_gbps=" << n->bandwidth_gbps << " ";
        }

        // 输出该节点的后序子节点数组
        fout << "out_degree=" << base->children.size() << " children=[";
        for (size_t j = 0; j < base->children.size(); ++j) {
            if (j > 0) fout << ",";
            fout << base->children[j];
        }
        fout << "]\n";
    }

    fout.close();
    return true;
}

int main(int argc, char* argv[]) {
    string input_path = "view_trace.json";
    string output_path = "dag_output.txt";

    if (argc >= 2) {
        input_path = argv[1];
    }
    if (argc >= 3) {
        output_path = argv[2];
    }

    vector<string> json_lines;
    if (!translateJsonFileToLines(input_path, json_lines)) {
        cerr << "Error: Failed to read or parse traceEvents from " << input_path << endl;
        return 1;
    }

    parseLinesToNodes(json_lines);
    buildDagEdges();

    if (!writeDagToFile(output_path)) {
        cerr << "Error: Failed to write DAG output to " << output_path << endl;
        return 1;
    }

    cout << "Successfully exported DAG to " << output_path
         << " (Nodes: " << g_all_nodes.size()
         << ", Compute: " << g_comp_nodes.size()
         << ", Collective: " << g_coll_nodes.size()
         << ", P2P: " << g_p2p_nodes.size() << ")" << endl;
    return 0;
}
