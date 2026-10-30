#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <cmath>
#include <cerrno>
#include <limits>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

// Offline, dependency-free multi-rank DAG exporter.
// Input: <bundle>/rank0..rank7/{view_trace.json,profiler_metadata.json}
// Output: text DAG (N M followed by one node per line).

struct Json {
    enum Kind { NIL, BOOL, NUMBER, STRING, ARRAY, OBJECT } kind = NIL;
    string scalar;
    vector<Json> array;
    map<string, Json> object;
};

class JsonParser {
    const string &s;
    size_t p = 0;
    void ws() { while (p < s.size() && isspace(static_cast<unsigned char>(s[p]))) ++p; }
    void expect(char c) { ws(); if (p >= s.size() || s[p++] != c) throw runtime_error("invalid JSON"); }
    string str() {
        expect('"'); string out;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return out;
            if (c != '\\') { if(static_cast<unsigned char>(c)<32) throw runtime_error("control character in JSON string"); out += c; continue; }
            if (p >= s.size()) throw runtime_error("invalid JSON escape");
            char e = s[p++];
            if (e == '"' || e == '\\' || e == '/') out += e;
            else if (e == 'n') out += '\n';
            else if (e == 'r') out += '\r';
            else if (e == 't') out += '\t';
            else if (e == 'b') out += '\b';
            else if (e == 'f') out += '\f';
            else if (e == 'u') {
                auto hex4 = [&]() {
                    unsigned v = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (p >= s.size()) throw runtime_error("short unicode escape");
                        char h = s[p++];
                        int d = h >= '0' && h <= '9' ? h-'0' : h >= 'a' && h <= 'f' ? h-'a'+10 : h >= 'A' && h <= 'F' ? h-'A'+10 : -1;
                        if (d < 0) throw runtime_error("invalid unicode escape");
                        v = v * 16 + d;
                    }
                    return v;
                };
                unsigned u = hex4();
                if (u >= 0xd800 && u <= 0xdbff) {
                    if (p+2 > s.size() || s.substr(p,2) != "\\u") throw runtime_error("missing low surrogate");
                    p += 2; unsigned lo = hex4();
                    if (lo < 0xdc00 || lo > 0xdfff) throw runtime_error("invalid low surrogate");
                    u = 0x10000 + (u-0xd800)*1024 + lo-0xdc00;
                } else if (u >= 0xdc00 && u <= 0xdfff) throw runtime_error("unpaired surrogate");
                if (u < 0x80) out += char(u);
                else if (u < 0x800) { out += char(0xc0 | (u>>6)); out += char(0x80 | (u&63)); }
                else if (u < 0x10000) { out += char(0xe0 | (u>>12)); out += char(0x80 | ((u>>6)&63)); out += char(0x80 | (u&63)); }
                else { out += char(0xf0 | (u>>18)); out += char(0x80 | ((u>>12)&63)); out += char(0x80 | ((u>>6)&63)); out += char(0x80 | (u&63)); }
            } else throw runtime_error("invalid JSON escape");
        }
        throw runtime_error("unterminated JSON string");
    }
    Json value() {
        ws(); if (p >= s.size()) throw runtime_error("invalid JSON");
        if (s[p] == '"') { Json v; v.kind = Json::STRING; v.scalar = str(); return v; }
        if (s[p] == '{') return object();
        if (s[p] == '[') return array_value();
        size_t begin = p;
        while (p < s.size() && string(",]} \t\r\n").find(s[p]) == string::npos) ++p;
        string t = s.substr(begin, p - begin);
        Json v; v.scalar = t; v.kind = (t == "true" || t == "false") ? Json::BOOL :
                                      (t == "null" ? Json::NIL : Json::NUMBER);
        if (v.kind == Json::NUMBER) {
            size_t i = 0;
            if (i < t.size() && t[i] == '-') ++i;
            if (i == t.size()) throw runtime_error("invalid JSON number");
            if (t[i] == '0') ++i;
            else { if (t[i] < '1' || t[i] > '9') throw runtime_error("invalid JSON number"); while (i<t.size() && isdigit(t[i])) ++i; }
            if (i<t.size() && t[i]=='.') { size_t b=++i; while(i<t.size() && isdigit(t[i])) ++i; if(i==b) throw runtime_error("invalid fraction"); }
            if (i<t.size() && (t[i]=='e'||t[i]=='E')) { ++i; if(i<t.size() && (t[i]=='+'||t[i]=='-')) ++i; size_t b=i; while(i<t.size()&&isdigit(t[i])) ++i; if(i==b) throw runtime_error("invalid exponent"); }
            if (i != t.size()) throw runtime_error("invalid JSON number");
        }
        return v;
    }
    Json object() {
        Json v; v.kind = Json::OBJECT; expect('{'); ws();
        if (p < s.size() && s[p] == '}') { ++p; return v; }
        while (true) { string k = str(); expect(':'); v.object[k] = value(); ws();
            if (p < s.size() && s[p] == '}') { ++p; return v; } expect(','); }
    }
    Json array_value() {
        Json v; v.kind = Json::ARRAY; expect('['); ws();
        if (p < s.size() && s[p] == ']') { ++p; return v; }
        while (true) { v.array.push_back(value()); ws();
            if (p < s.size() && s[p] == ']') { ++p; return v; } expect(','); }
    }
public:
    explicit JsonParser(const string &input) : s(input) {}
    Json parse() { Json v = value(); ws(); if (p != s.size()) throw runtime_error("trailing JSON data"); return v; }
};

static string read_file(const string &path) {
    ifstream f(path.c_str()); if (!f) throw runtime_error("cannot open " + path);
    return string((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
}
static const Json *field(const Json &v, const string &key) {
    if (v.kind != Json::OBJECT) return nullptr;
    auto it = v.object.find(key); return it == v.object.end() ? nullptr : &it->second;
}
static string text(const Json *v, const string &fallback = "") {
    return v && (v->kind == Json::STRING || v->kind == Json::NUMBER || v->kind == Json::BOOL) ? v->scalar : fallback;
}
static long long integer(const Json *v, long long fallback = -1) {
    if (!v || (v->kind != Json::NUMBER && v->kind != Json::STRING)) return fallback;
    string token = text(v); char *end = nullptr; errno = 0;
    long long n = strtoll(token.c_str(), &end, 10);
    return !token.empty() && !errno && end && *end == '\0' ? n : fallback;
}
static double number(const Json *v, double fallback = 0) {
    if (!v || (v->kind != Json::NUMBER && v->kind != Json::STRING)) return fallback;
    string token = text(v); char *end = nullptr; errno = 0;
    double n = strtod(token.c_str(), &end);
    return !token.empty() && !errno && end && *end == '\0' && isfinite(n) ? n : fallback;
}
static string lower(string s) { for (char &c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c))); return s; }
static string norm(string s) { string r; for (char c : lower(s)) if (c != '_' && c != '-') r += c; return r; }
static string quote(const string &s) {
    string r = "\"";
    for (unsigned char c : s) {
        if(c=='"'||c=='\\') { r+='\\'; r+=char(c); }
        else if(c<32) { const char *hex="0123456789abcdef"; r+="\\u00"; r+=hex[c>>4]; r+=hex[c&15]; }
        else r+=char(c);
    }
    return r + "\"";
}

struct Record { int rank; double ts = 0, dur = 0; long long stream = -1, pid = -1, tid = -1; long long bytes = -1; };
struct Node {
    enum Type { COMPUTE, COLLECTIVE, P2P } type;
    int id = 0, rank = -1; string name, task_type, comm_type, p2p_type, data_type;
    double ts = 0, dur = 0, compute_time = 0, bandwidth = 0;
    long long stream = -1, pid = -1, tid = -1, count = 0, bytes = 0;
    vector<int> participants; vector<Record> records; set<int> children;
    set<int> expected;
    size_t source = 0;
    string track, group;
    bool active = true;
};

struct RankTrace {
    int rank; Json trace, metadata; vector<Node *> local;
    map<long long,string> processes;
    map<pair<long long,long long>,string> threads;
    map<string,set<int>> groups;
    map<size_t,Node *> event_nodes;
};

static const Json *events_of(const Json &root) {
    if (root.kind == Json::ARRAY) return &root;
    const Json *e = field(root, "traceEvents");
    if (!e || e->kind != Json::ARRAY) throw runtime_error("trace must be an event array or contain traceEvents");
    return e;
}

static long long stream_of(const Json &e, const Json *args, long long tid) {
    const char *keys[] = {"Physic Stream Id", "physic_stream_id", "Stream Id", "stream_id", "stream"};
    for (const char *k : keys) { if (const Json *v = field(e, k)) return integer(v, tid); if (args && field(*args, k)) return integer(field(*args, k), tid); }
    return tid;
}
static string process_name(const Json &e) {
    const Json *a = field(e, "args"); const Json *n = a ? field(*a, "name") : nullptr;
    return text(n, text(field(e, "process_name")));
}
static string task_type(const Json &e, const Json *args) {
    for (const char *k : {"Task Type", "task_type", "Accelerator Core", "accelerator_core"}) {
        if (const Json *v = field(e, k)) return text(v);
        if (args && field(*args, k)) return text(field(*args, k));
    }
    return "";
}
static bool is_compute(const Json &e, const string &proc, const string &name, const string &cat, const string &task) {
    if (text(field(e, "ph")) != "X" || proc.find("Ascend Hardware") == string::npos || lower(cat) == "cpu_op") return false;
    if (name.find("hcom") != string::npos || name.find("hccl") != string::npos) return false;
    if (lower(cat).find("communication") != string::npos || lower(name).find("memcpy") != string::npos || lower(name).find("notify") != string::npos) return false;
    return task == "AI_CORE" || task == "AI_VECTOR_CORE" || task == "MIX_AIC" || task == "MIX_AIV";
}
static string collective_type(const string &name) {
    string n = norm(name);
    if (n.find("allreduce") != string::npos) return "ALL_REDUCE";
    if (n.find("allgather") != string::npos) return "ALL_GATHER";
    if (n.find("reducescatter") != string::npos) return "REDUCE_SCATTER";
    if (n.find("alltoall") != string::npos) return "ALL_TO_ALL";
    if (n.find("broadcast") != string::npos) return "BROADCAST";
    return "";
}
static bool is_collective(const Json &e, const string &proc, const string &name) {
    if (text(field(e, "ph")) != "X") return false;
    string p = lower(proc), n = lower(name); if (p.find("communication") == string::npos && p.find("hccl") == string::npos) return false;
    if (n.find("send") != string::npos || n.find("recv") != string::npos || n.find("receive") != string::npos) return false;
    if (n.find("notify") != string::npos || n.find("memcpy") != string::npos || n.find("reduce_inline") != string::npos || n.find("rdma") != string::npos) return false;
    return !collective_type(name).empty();
}
static bool is_p2p(const Json &e, const string &proc, const string &name) {
    if (text(field(e, "ph")) != "X") return false;
    string p = lower(proc), n = lower(name); if (p.find("communication") == string::npos && p.find("hccl") == string::npos) return false;
    return (n.find("send") != string::npos || n.find("recv") != string::npos || n.find("receive") != string::npos) &&
           n.find("notify") == string::npos && n.find("memcpy") == string::npos && n.find("rdma") == string::npos;
}

static void validate_metadata(const Json &m, int rank) {
    const Json *args = field(m, "distributed_args");
    const Json *rv = args ? field(*args, "rank") : nullptr;
    const Json *wv = args ? field(*args, "world_size") : nullptr;
    if (rv && integer(rv) != rank) throw runtime_error("metadata rank mismatch for rank" + to_string(rank));
    if (wv && integer(wv) != 8) throw runtime_error("world_size is not 8 for rank" + to_string(rank));
    if (!rv || !wv) throw runtime_error("metadata requires distributed_args.rank and world_size for rank" + to_string(rank));
}

static string key_for(const Node &n) { return n.name; }
static long long dtype_bytes(const string &d) {
    string x = lower(d); if (x.find("fp64") != string::npos || x.find("int64") != string::npos || x.find("double") != string::npos) return 8;
    if (x == "float" || x.find("float32") != string::npos || x.find("fp32") != string::npos || x.find("int32") != string::npos) return 4;
    if (x.find("int8") != string::npos || x.find("uint8") != string::npos) return 1;
    if (x.find("bf16") != string::npos || x.find("float16") != string::npos || x.find("fp16") != string::npos || x.find("int16") != string::npos) return 2;
    return -1;
}

static void add_edge(Node *a, Node *b) { if (a && b && a != b) a->children.insert(b->id); }

static void collect_groups(const Json &v, map<string,set<int>> &groups, string key="") {
    if (v.kind == Json::ARRAY) { for (const Json &x:v.array) collect_groups(x,groups,key); return; }
    if (v.kind != Json::OBJECT) return;
    const Json *ranks = field(v,"global_ranks");
    if (!ranks) ranks = field(v,"group_global_rank");
    string name = text(field(v,"group_name"),key);
    if (ranks && ranks->kind==Json::ARRAY && !name.empty()) {
        set<int> members;
        for (const Json &r:ranks->array) {
            long long rank=integer(&r);
            if(rank<0||rank>=8) throw runtime_error("invalid group member for " + name);
            members.insert(int(rank));
        }
        if(!members.empty()) {
            if(groups.count(name)&&groups[name]!=members) throw runtime_error("conflicting metadata group " + name);
            groups[name]=members;
        }
    }
    for(const auto &kv:v.object) collect_groups(kv.second,groups,kv.first);
}

static string event_track(const Json &e) {
    const Json *args=field(e,"args");
    long long sid=stream_of(e,args,-1);
    string context=args ? text(field(*args,"Context Id"),text(field(*args,"context_id"),"0")) : "0";
    if(sid>=0) return "device:"+context+":"+to_string(sid);
    return "track:"+to_string(integer(field(e,"pid")))+":"+to_string(integer(field(e,"tid")));
}

static void metadata_index(RankTrace &rt) {
    for(const Json &e:events_of(rt.trace)->array) {
        if(text(field(e,"ph"))!="M") continue;
        long long pid=integer(field(e,"pid")), tid=integer(field(e,"tid"));
        string name=text(field(e,"name"));
        if(name=="process_name") rt.processes[pid]=process_name(e);
        if(name=="thread_name") rt.threads[{pid,tid}]=process_name(e);
    }
    for(const char *key:{"parallel_group_info","group_info"}) {
        const Json *v=field(rt.metadata,key); if(v) collect_groups(*v,rt.groups);
    }
}

// Keep edge provenance separately, including the original per-rank event records.
struct Evidence { Node *src; Node *dst; string reason; };
static vector<Evidence> evidence;
static void link(Node *a,Node *b,const string &why) {
    if(a && b && a!=b) evidence.push_back({a,b,why});
}

static string attribute(const Json &e, const string &key) {
    const Json *args = field(e, "args");
    return text(field(e, key), args ? text(field(*args, key)) : "");
}

static string context_of(const Json &e) {
    string context = attribute(e, "context_id");
    return context.empty() ? attribute(e, "Context Id") : context;
}

static void local_dependencies(RankTrace &rt, ostream &diag, double window, bool infer) {
    const auto &events=events_of(rt.trace)->array;
    map<string,vector<Node *>> tracks;
    map<pair<long long,long long>,vector<Node *>> display_tracks;
    map<string,vector<Node *>> connections;
    for(Node *n:rt.local) tracks[n->track].push_back(n);
    for(Node *n:rt.local) {
        display_tracks[{n->pid,n->tid}].push_back(n);
        string id=attribute(events[n->source],"connection_id");
        if(!id.empty()) connections[id].push_back(n);
    }
    for(auto &kv:tracks) {
        auto &v=kv.second;
        sort(v.begin(),v.end(),[](Node *a,Node *b){return a->ts!=b->ts?a->ts<b->ts:a->source<b->source;});
        for(size_t i=1;i<v.size();++i) {
            // A display thread is not a proven serial device stream.
            if(kv.first.find("device:")==0 && v[i-1]->ts+v[i-1]->dur<=v[i]->ts)
                link(v[i-1],v[i],"same_stream");
        }
    }
    for(auto &kv:display_tracks)
        sort(kv.second.begin(),kv.second.end(),[](Node *a,Node *b){return a->ts<b->ts;});
    // A connection identifies ownership, NOT a finish-before-start edge. Keep
    // launch and microstep events as evidence even though they are not public nodes.
    auto owner=[&](size_t index)->Node * {
        const Json &e=events[index];
        string id=attribute(e,"connection_id");
        if(id.empty() || !connections.count(id)) return nullptr;
        double t=number(field(e,"ts"),-1), end=t+number(field(e,"dur"));
        Node *result=nullptr;
        for(Node *n:connections[id]) {
            string context=context_of(e), nc=context_of(events[n->source]);
            if(!context.empty()&&!nc.empty()&&context!=nc) continue;
            string group=attribute(e,"group_name");
            if(!group.empty()&&!n->group.empty()&&group!=n->group) continue;
            if(t<0 || end<n->ts-window || t>n->ts+n->dur+window) continue;
            if(result && result!=n) {
                diag<<"rank="<<rt.rank<<" ambiguous_connection event="<<index<<"\n";
                return nullptr;
            }
            result=n;
        }
        return result;
    };
    auto locate=[&](size_t index,bool before)->Node * {
        auto exact=rt.event_nodes.find(index); if(exact!=rt.event_nodes.end()) return exact->second;
        const Json &e=events[index]; double t=number(field(e,"ts"),-1);
        if(Node *n=owner(index)) return n;
        if(t<0) return nullptr;
        // Flow markers often omit stream args. Resolve through their display
        // track without treating that track as a serial device stream.
        const vector<Node *> *nodes=nullptr;
        auto it=tracks.find(event_track(e));
        if(it!=tracks.end()) nodes=&it->second;
        else {
            auto dt=display_tracks.find({integer(field(e,"pid")),integer(field(e,"tid"))});
            if(dt!=display_tracks.end()) nodes=&dt->second;
        }
        if(!nodes) return nullptr;
        Node *found=nullptr;
        for(Node *n:*nodes) {
            if(n->ts<=t && t<n->ts+n->dur) return n;
            if(before && n->ts+n->dur<=t && t-(n->ts+n->dur)<=window) found=n;
            if(!before && n->ts>=t) return n->ts-t<=window?n:nullptr;
        }
        return found;
    };
    auto completion_link=[&](size_t start,size_t finish,const string &reason) {
        Node *a=locate(start,true), *b=locate(finish,false);
        if(!a||!b) {
            diag<<"rank="<<rt.rank<<" unresolved_dependency reason="<<reason
                <<" events="<<start<<","<<finish<<"\n";
            return;
        }
        if(a==b) return;
        // Partial collective progress and host launches must not become whole
        // operation completion dependencies. Zero tolerance: timestamps are us.
        if(a->ts+a->dur>b->ts) {
            diag<<"rank="<<rt.rank<<" noncompletion_dependency reason="<<reason
                <<" events="<<start<<","<<finish<<"\n";
            return;
        }
        link(a,b,reason);
        diag<<"rank="<<rt.rank<<" dependency_evidence reason="<<reason
            <<" events="<<start<<","<<finish<<" nodes="<<a->source<<","<<b->source<<"\n";
    };
    vector<size_t> order;
    for(size_t i=0;i<events.size();++i) if(number(field(events[i],"ts"),-1)>=0) order.push_back(i);
    stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return number(field(events[a],"ts"))<number(field(events[b],"ts"));});
    map<string,size_t> pending;
    for(size_t i:order) {
        const Json &e=events[i]; string ph=text(field(e,"ph"));
        if(ph!="s" && ph!="t" && ph!="f") continue;
        string id=text(field(e,"id"),text(field(e,"bind_id")));
        if(id.empty()) { diag<<"rank="<<rt.rank<<" unresolved flow event="<<i<<" missing id\n"; continue; }
        // name is descriptive and may differ between flow endpoints.
        string key=quote(text(field(e,"cat")))+quote(text(field(e,"scope")))+
                   quote(text(field(e,"bid")))+quote(id);
        if(ph=="s") {
            if(pending.count(key)) diag<<"rank="<<rt.rank<<" reused active flow id="<<quote(key)<<"\n";
            pending[key]=i;
        } else if(pending.count(key)) {
            string name=lower(text(field(e,"name")));
            string start_name=lower(text(field(events[pending[key]],"name")));
            bool launch=name=="hostop"||name=="hostopdevice"||start_name=="hostop"||start_name=="hostopdevice";
            if(launch) diag<<"rank="<<rt.rank<<" launch_flow events="<<pending[key]<<","<<i
                           <<" (ownership only; not a completion edge)\n";
            else completion_link(pending[key],i,"flow");
            if(ph=="f") pending.erase(key); else pending[key]=i;
        } else diag<<"rank="<<rt.rank<<" flow missing start event="<<i<<"\n";
    }
    for(const auto &kv:pending) diag<<"rank="<<rt.rank<<" flow missing finish id="<<quote(kv.first)<<"\n";
    // IDs are rank-local and reusable. Match within a bounded time window and
    // reject ambiguous active records instead of silently taking the last one.
    map<string,size_t> records;
    set<string> ambiguous_records;
    for(size_t i:order) {
        const Json &e=events[i]; const Json *a=field(e,"args"); if(!a) continue;
        string task=task_type(e,a), name=lower(text(field(e,"name")));
        bool notify=name.find("notify_record")!=string::npos||name.find("notify_wait")!=string::npos;
        bool record=task=="EVENT_RECORD"||name.find("notify_record")!=string::npos;
        bool wait=task=="EVENT_WAIT"||name.find("notify_wait")!=string::npos;
        if(!record&&!wait) continue;
        string id=attribute(e,notify?"notify_id":"event_id");
        if(id.empty()) id=attribute(e,"Event Id");
        if(id.empty()&&notify) id=attribute(e,"event_id");
        string context=context_of(e);
        if(id.empty()||context.empty()) {diag<<"rank="<<rt.rank<<" unresolved_event_sync event="<<i<<"\n";continue;}
        string key=(notify?"notify":"event")+quote(context)+quote(id);
        if(notify) {
            string group=attribute(e,"group_name"), src=attribute(e,"src_rank"), dst=attribute(e,"dst_rank");
            if(group.empty()||src.empty()||dst.empty()) {
                diag<<"rank="<<rt.rank<<" unresolved_notify event="<<i<<" missing group/peer context\n";
                continue;
            }
            key+=quote(group)+quote(src)+quote(dst);
        }
        if(record) {
            if(records.count(key) && number(field(e,"ts"))-number(field(events[records[key]],"ts"))<=window)
                ambiguous_records.insert(key);
            else ambiguous_records.erase(key);
            records[key]=i;
        } else if(records.count(key)) {
            size_t start=records[key];
            if(!ambiguous_records.count(key) && number(field(e,"ts"))-number(field(events[start],"ts"))<=window)
                completion_link(start,i,notify?"notify_record_wait":"event_record_wait");
            else diag<<"rank="<<rt.rank<<" unresolved_event_sync event="<<i<<" ambiguous or expired record\n";
            records.erase(key);
            ambiguous_records.erase(key);
        } else diag<<"rank="<<rt.rank<<" unresolved_event_sync event="<<i<<" missing record\n";
    }
    if(infer) {
        // Only infer within a demonstrated stream/launch chain, never select the
        // globally nearest compute. Connection ownership anchors hidden waits.
        for(size_t i:order) {
            const Json &e=events[i];
            Node *comm=owner(i);
            if(!comm||comm->type==Node::COMPUTE||rt.event_nodes.count(i)) continue;
            auto track=tracks.find(event_track(e));
            if(track==tracks.end()||track->first.find("device:")!=0) continue;
            string task=task_type(e,field(e,"args")), name=lower(text(field(e,"name")));
            bool wait=task=="EVENT_WAIT"||name.find("notify_wait")!=string::npos;
            double t=number(field(e,"ts")), end=t+number(field(e,"dur"));
            Node *candidate=nullptr;
            for(Node *n:track->second) {
                if(n==comm) continue;
                if(wait && n->ts>=max(end,comm->ts+comm->dur) && n->ts-end<=window) { candidate=n; break; }
                if(!wait && n->type==Node::COMPUTE && n->ts+n->dur<=t && t-(n->ts+n->dur)<=window)
                    candidate=n;
            }
            if(candidate) {
                if(wait) link(comm,candidate,"inferred_collective_completion");
                else if(candidate->ts+candidate->dur<=comm->ts)
                    link(candidate,comm,"inferred_compute_to_communication");
                diag<<"rank="<<rt.rank<<" inferred_anchor event="<<i<<" communication="<<comm->source<<"\n";
            }
        }
    }
    for(Node *n:rt.local) if(n->type==Node::P2P)
        diag<<"rank="<<rt.rank<<" unresolved_p2p source="<<n->source<<" name="<<quote(n->name)<<" (peer matching unavailable)\n";
}

int main(int argc, char **argv) {
    if (argc < 3) { cerr << "usage: " << argv[0] << " <profile_bundle> <dag_output> [--infer-dependencies] [--dependency-window-us N]\n"; return 2; }
    try {
        bool infer=false;
        double window=1000;
        for(int i=3;i<argc;++i) {
            string arg=argv[i];
            if(arg=="--infer-dependencies") infer=true;
            else if(arg=="--dependency-window-us" && i+1<argc) {
                string value=argv[++i]; char *end=nullptr;
                window=strtod(value.c_str(),&end);
                if(value.empty()||*end||!isfinite(window)||window<=0) throw runtime_error("invalid dependency window");
            } else { cerr<<"unknown or incomplete option: "<<arg<<"\n"; return 2; }
        }
        string root = argv[1], output = argv[2]; vector<RankTrace> traces; deque<Node> storage;
        map<string, vector<Node *> > collective_by_name;
        ofstream diag((output+".diagnostics.txt").c_str());
        if(!diag) throw runtime_error("cannot create diagnostics");
        diag << "Collective matching uses exact-name hypothesis; unresolved dependencies are listed below.\n";
        for (int rank = 0; rank < 8; ++rank) {
            string dir = root + "/rank" + to_string(rank);
            RankTrace rt;
            rt.rank=rank;
            rt.trace=JsonParser(read_file(dir + "/view_trace.json")).parse();
            rt.metadata=JsonParser(read_file(dir + "/profiler_metadata.json")).parse();
            validate_metadata(rt.metadata, rank);
            metadata_index(rt);
            const Json *events = events_of(rt.trace);
            for (size_t index=0;index<events->array.size();++index) {
                const Json &e=events->array[index];
                string name = text(field(e, "name")), cat = text(field(e, "cat"));
                const Json *args = field(e, "args"); long long pid = integer(field(e, "pid")), tid = integer(field(e, "tid"));
                string proc=rt.processes[pid];
                string thread=rt.threads[{pid,tid}];
                if(thread.find("Plane")==0) continue;
                double ts = number(field(e, "ts"), -1), dur = number(field(e, "dur")); string task = task_type(e, args);
                if (ts < 0) continue;
                if(dur<0) throw runtime_error("negative duration rank"+to_string(rank));
                Node n; n.rank = rank; n.name = name; n.ts = ts; n.dur = dur; n.pid = pid; n.tid = tid; n.stream = stream_of(e, args, tid);
                n.source=index; n.track=event_track(e);
                if (is_compute(e, proc, name, cat, task)) {
                    n.type = Node::COMPUTE; n.task_type = task; n.compute_time = dur; storage.push_back(n); rt.local.push_back(&storage.back());
                } else if (is_collective(e, proc, name)) {
                    n.type = Node::COLLECTIVE; n.comm_type = collective_type(name); n.data_type = text(field(args ? *args : e, "data_type"));
                    n.count = integer(field(args ? *args : e, "count"), -1);
                    long long unit=dtype_bytes(n.data_type);
                    if(unit>0 && n.count>numeric_limits<long long>::max()/unit) throw runtime_error("byte count overflow");
                    n.bytes = n.count >= 0 && unit>0 ? n.count * unit : -1;
                    n.group=text(field(args ? *args : e,"group_name"));
                    if(n.group.empty() && thread.find("Group ")==0) {
                        size_t end=thread.find(','); n.group=thread.substr(6,end==string::npos?string::npos:end-6);
                    }
                    if(rt.groups.count(n.group)) n.expected=rt.groups[n.group];
                    else diag<<"rank="<<rank<<" unresolved_group event="<<index<<" group="<<quote(n.group)<<"\n";
                    n.bandwidth = n.bytes && dur > 0 ? (static_cast<double>(n.bytes) / dur) * 1e-3 : 0; storage.push_back(n); rt.local.push_back(&storage.back());
                    collective_by_name[key_for(storage.back())].push_back(&storage.back());
                } else if (is_p2p(e, proc, name)) {
                    n.type = Node::P2P; n.p2p_type = lower(name).find("send") != string::npos ? "SEND" : "RECV"; n.data_type = text(field(args ? *args : e, "data_type"));
                    n.count = integer(field(args ? *args : e, "count"), -1); n.bytes=-1; storage.push_back(n); rt.local.push_back(&storage.back());
                }
                if(!rt.local.empty() && rt.local.back()->source==index) rt.event_nodes[index]=rt.local.back();
            }
            local_dependencies(rt,diag,window,infer);
            // Release raw JSON before reading the next rank; nodes have stable deque addresses.
            rt.trace=Json(); rt.metadata=Json();
            traces.push_back(std::move(rt));
        }
        // A collective name is the trace-only cross-rank identity. Reject repeated names on one rank.
        vector<Node> merged; map<Node *, Node *> replacement;
        for (auto &kv : collective_by_name) {
            set<int> ranks; for (Node *n : kv.second) ranks.insert(n->rank);
            if (ranks.size() != kv.second.size()) throw runtime_error("duplicate collective name on one rank: " + kv.first);
            for(Node *n:kv.second) {
                if(!n->expected.empty() && n->expected!=ranks) throw runtime_error("collective participants conflict with metadata: "+kv.first);
                if(n->comm_type!=kv.second.front()->comm_type) throw runtime_error("collective type mismatch");
            }
            Node m = *kv.second.front(); m.rank = -1; m.participants.assign(ranks.begin(), ranks.end()); m.records.clear(); m.ts = kv.second.front()->ts; m.dur = 0;
            double finish=0;
            for (Node *n : kv.second) {
                m.ts = min(m.ts, n->ts);
                finish=max(finish,n->ts+n->dur);
                Record record;
                record.rank = n->rank;
                record.ts = n->ts;
                record.dur = n->dur;
                record.stream = n->stream;
                record.pid = n->pid;
                record.tid = n->tid;
                record.bytes=n->bytes;
                m.records.push_back(record);
                replacement[n] = nullptr;
                n->active=false;
            }
            m.dur=finish-m.ts;
            merged.push_back(m);
        }
        // Replace collective records in per-rank lists with the merged node after stable storage is complete.
        for (Node &m : merged) storage.push_back(m);
        for (size_t i = 0; i < merged.size(); ++i) {
            Node *m = &storage[storage.size() - merged.size() + i];
            for (auto &kv : collective_by_name) if (kv.first == m->name) for (Node *old : kv.second) replacement[old] = m;
        }
        vector<Node *> all; for (Node &n : storage) if (n.active) all.push_back(&n);
        sort(all.begin(), all.end(), [](Node *a, Node *b) { return a->ts != b->ts ? a->ts < b->ts : a->name < b->name; });
        for (size_t i = 0; i < all.size(); ++i) all[i]->id = static_cast<int>(i + 1);
        for(const Evidence &e:evidence) {
            Node *a=replacement.count(e.src)?replacement[e.src]:e.src;
            Node *b=replacement.count(e.dst)?replacement[e.dst]:e.dst;
            add_edge(a,b);
            if(a!=b) diag<<"edge "<<a->id<<" -> "<<b->id<<" reason="<<e.reason<<" rank="<<e.src->rank<<" sources="<<e.src->source<<","<<e.dst->source<<"\n";
        }
        for(Node *n:all) if(n->type==Node::COLLECTIVE && n->children.empty())
            diag<<"collective_without_successor id="<<n->id<<" name="<<quote(n->name)<<"\n";
        // Validate the resulting graph is acyclic.
        map<int, int> indegree; for (Node *n : all) indegree[n->id] = 0;
        for (Node *n : all) for (int c : n->children) ++indegree[c];
        vector<int> q;
        for (auto &x : indegree) if (!x.second) q.push_back(x.first);
        size_t seen = 0; for (size_t i = 0; i < q.size(); ++i) { ++seen; Node *n = all.at(q[i]-1); for (int c : n->children) if (--indegree[c] == 0) q.push_back(c); }
        if (seen != all.size()) throw runtime_error("generated graph contains a cycle");
        ofstream out(output.c_str()); if (!out) throw runtime_error("cannot write " + output); long long edges = 0; for (Node *n : all) edges += n->children.size();
        out << all.size() << " " << edges << "\n"; out.setf(ios::fixed); out.precision(3);
        for (Node *n : all) {
            out << "id=" << n->id << " ";
            if (n->type == Node::COMPUTE) out << "type=COMPUTE rank=" << n->rank << " name=" << quote(n->name) << " task_type=" << n->task_type << " stream=" << n->stream << " ts=" << n->ts << " compute_time_us=" << n->compute_time;
            else if (n->type == Node::P2P) out << "type=P2P rank=" << n->rank << " name=" << quote(n->name) << " p2p_type=" << n->p2p_type << " stream=" << n->stream << " ts=" << n->ts << " dur=" << n->dur;
            else { out << "type=COLLECTIVE name=" << quote(n->name) << " comm_type=" << n->comm_type << " participants=["; for (size_t i=0;i<n->participants.size();++i) out << (i ? "," : "") << n->participants[i]; out << "] ts=" << n->ts << " span_us=" << n->dur << " participant_source=exact_name records=[";
                for(size_t i=0;i<n->records.size();++i) {
                    const Record &r=n->records[i];
                    out<<(i?",":"")<<"{rank="<<r.rank<<",ts="<<r.ts<<",dur_us="<<r.dur<<",stream="<<r.stream<<",pid="<<r.pid<<",tid="<<r.tid<<",bytes="<<r.bytes<<",estimated_bandwidth_GBps=";
                    if(r.bytes>=0&&r.dur>0) out<<r.bytes/r.dur/1000.0; else out<<"unknown";
                    out<<"}";
                } out<<"]";
            }
            out << " out_degree=" << n->children.size() << " children=["; bool first = true; for (int c : n->children) { if (!first) out << ","; first = false; out << c; } out << "]\n";
        }
        out.close(); if(!out) throw runtime_error("failed writing DAG output");
        cout << "Successfully exported multi-rank DAG to " << output << " (Nodes: " << all.size() << ", Edges: " << edges << ")\n";
    } catch (const exception &e) { cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}
