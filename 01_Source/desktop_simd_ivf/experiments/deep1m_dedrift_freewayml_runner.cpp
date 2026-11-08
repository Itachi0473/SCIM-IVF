#include "../adaptive_hierarchical_ivf.h"
#include "../ivf_flat.h"
#include "../simd_l2.h"
#include <limits>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
constexpr std::size_t kDimension = 96;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 512;
constexpr std::size_t kInitialRows = 500'000;
constexpr std::size_t kBatchRows = 5'000;
constexpr std::size_t kStages = 100;
constexpr std::size_t kQueries = 1'000;
constexpr std::uint32_t kSeed = 1234;
constexpr std::array<const char*, 9> kCases{"G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
using TruthRow = std::array<std::uint32_t, kTopK>;

class MappedFbin {
public:
    explicit MappedFbin(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open fbin: " + path.string());
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < 8) { close(); throw std::runtime_error("cannot get fbin size"); }
        bytes_ = static_cast<std::uint64_t>(size.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) { close(); throw std::runtime_error("cannot map fbin"); }
        mapped_ = static_cast<const std::uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!mapped_) { close(); throw std::runtime_error("cannot view fbin"); }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open fbin: " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) { close(); throw std::runtime_error("cannot stat fbin"); }
        bytes_ = static_cast<std::uint64_t>(st.st_size);
        mapped_ = static_cast<const std::uint8_t*>(::mmap(nullptr, static_cast<std::size_t>(bytes_), PROT_READ, MAP_SHARED, fd_, 0));
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; close(); throw std::runtime_error("cannot mmap fbin"); }
#endif
        std::uint32_t shape[2]{};
        std::memcpy(shape, mapped_, sizeof(shape));
        rows_ = shape[0]; dimension_ = shape[1];
        if (dimension_ != kDimension || bytes_ != 8ull + rows_ * dimension_ * sizeof(float)) {
            close(); throw std::runtime_error("fbin shape/size mismatch: " + path.string());
        }
    }
    ~MappedFbin() { close(); }
    MappedFbin(const MappedFbin&) = delete;
    MappedFbin& operator=(const MappedFbin&) = delete;
    const float* row(std::size_t id) const {
        if (id >= rows_) throw std::out_of_range("fbin row out of range");
        return reinterpret_cast<const float*>(mapped_ + 8) + id * dimension_;
    }
    std::size_t rows() const noexcept { return rows_; }
private:
    void close() noexcept {
#ifdef _WIN32
        if (mapped_) UnmapViewOfFile(mapped_);
        mapped_ = nullptr;
        if (mapping_) CloseHandle(mapping_);
        mapping_ = nullptr;
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
#else
        if (mapped_) ::munmap(const_cast<std::uint8_t*>(mapped_), static_cast<std::size_t>(bytes_));
        mapped_ = nullptr;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }
    const std::uint8_t* mapped_ = nullptr;
    std::uint64_t bytes_ = 0;
    std::size_t rows_ = 0, dimension_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open prepared truth: " + path.string());
    char magic[8]{};
    std::uint32_t shape[3]{};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!in || std::string(magic, sizeof(magic)) != "DLQTRTH1" ||
        shape[0] != kStages + 1 || shape[1] != kQueries || shape[2] != kTopK)
        throw std::runtime_error("prepared truth header mismatch: " + path.string());
    std::vector<TruthRow> truth((kStages + 1) * kQueries);
    in.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth.size() * sizeof(TruthRow)));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("prepared truth payload size mismatch: " + path.string());
    return truth;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t i = std::min(values.size() - 1,
        static_cast<std::size_t>(std::ceil(p * values.size()) - 1.0));
    return values[i];
}

// Uncompressed DeDrift-Hybrid, paper Section 5.1: one mean-center update,
// then joint k2-means on k largest plus k2-k smallest non-overlapping lists.
struct DeDriftPartition {
    std::vector<std::uint32_t> ids;
    std::vector<float> vectors;
    std::vector<float> center;
    std::vector<double> sum;
};
struct DeDriftMaintenance {
    std::size_t k=0, k2_raw=0, k2=0, vectors=0, reassigned=0;
    bool capped=false;
};
struct DeDriftSearch {
    std::vector<SearchResult> hits;
    std::size_t centers=0,vectors=0,lists=0;
};
class DeDriftHybrid {
public:
    DeDriftHybrid(const float* data,std::size_t rows):data_(data),rows_(rows){}
    void initialize(const std::vector<float>& centers,
                    const std::vector<std::vector<std::uint32_t>>& members) {
        if(centers.size()!=members.size()*kDimension || members.empty())
            throw std::invalid_argument("invalid shared initial partitions");
        parts_.resize(members.size());count_=0;
        for(std::size_t i=0;i<parts_.size();++i){
            parts_[i].center.assign(centers.begin()+i*kDimension,centers.begin()+(i+1)*kDimension);
            parts_[i].sum.assign(kDimension,0.);
            for(auto id:members[i])append(parts_[i],id);
        }
        validate();
    }
    void add(std::size_t begin,std::size_t count) {
        if(begin!=count_ || count>rows_-begin)throw std::invalid_argument("non-prefix insertion");
        for(std::size_t id=begin;id<begin+count;++id)append(parts_[nearest(data_+id*kDimension)],static_cast<std::uint32_t>(id));
    }
    DeDriftMaintenance maintain(std::size_t k,std::size_t stage,std::size_t iterations=10) {
        if(k==0||k>=parts_.size()||iterations==0)throw std::invalid_argument("invalid DeDrift parameters");
        // Aggregates produce exactly one fixed-membership mean update, without
        // a full-vector rescan. This matches the shared engine's center refresh.
        for(auto& p:parts_)if(!p.ids.empty())for(std::size_t d=0;d<kDimension;++d)
            p.center[d]=static_cast<float>(p.sum[d]/p.ids.size());
        std::vector<std::size_t> ascending(parts_.size());std::iota(ascending.begin(),ascending.end(),0);
        std::sort(ascending.begin(),ascending.end(),[&](auto a,auto b){return parts_[a].ids.size()==parts_[b].ids.size()?a<b:parts_[a].ids.size()<parts_[b].ids.size();});
        const std::size_t K=parts_.size();
        const double median=K%2?double(parts_[ascending[K/2]].ids.size()):
            .5*(parts_[ascending[K/2-1]].ids.size()+parts_[ascending[K/2]].ids.size());
        if(median<=0)throw std::runtime_error("DeDrift requires positive median partition size");
        std::vector<std::size_t> selected;std::vector<bool> used(K,false);std::size_t large_size=0;
        for(std::size_t i=0;i<k;++i){auto leaf=ascending[K-1-i];selected.push_back(leaf);used[leaf]=true;large_size+=parts_[leaf].ids.size();}
        DeDriftMaintenance report;report.k=k;report.k2_raw=static_cast<std::size_t>(std::ceil(large_size/median));
        report.k2=std::clamp(report.k2_raw,k,K);report.capped=report.k2!=report.k2_raw;
        for(auto leaf:ascending)if(selected.size()<report.k2&&!used[leaf]){selected.push_back(leaf);used[leaf]=true;}
        std::sort(selected.begin(),selected.end());
        std::vector<std::uint32_t> ids;std::vector<std::size_t> old_slots;std::vector<float> points;
        for(auto leaf:selected){auto& p=parts_[leaf];ids.insert(ids.end(),p.ids.begin(),p.ids.end());old_slots.insert(old_slots.end(),p.ids.size(),leaf);points.insert(points.end(),p.vectors.begin(),p.vectors.end());}
        report.vectors=ids.size();
        const auto local_seed=kSeed^(0x9e3779b9u*static_cast<std::uint32_t>(stage+1))^(0x85ebca6bu*static_cast<std::uint32_t>(report.k2));
        IvfFlat local(kDimension,report.k2,local_seed);local.train(points.data(),ids.size(),iterations);const auto& centers=local.centroids();
        std::vector<DeDriftPartition> replacement(report.k2);
        for(std::size_t i=0;i<report.k2;++i){replacement[i].center.assign(centers.begin()+i*kDimension,centers.begin()+(i+1)*kDimension);replacement[i].sum.assign(kDimension,0.);}
        for(std::size_t i=0;i<ids.size();++i){
            std::size_t best=0;float distance=simd_squared_l2(points.data()+i*kDimension,centers.data(),kDimension);
            for(std::size_t j=1;j<report.k2;++j){float d=simd_squared_l2(points.data()+i*kDimension,centers.data()+j*kDimension,kDimension);if(d<distance){best=j;distance=d;}}
            if(selected[best]!=old_slots[i])++report.reassigned;
            auto& p=replacement[best];p.ids.push_back(ids[i]);p.vectors.insert(p.vectors.end(),points.begin()+i*kDimension,points.begin()+(i+1)*kDimension);
            for(std::size_t d=0;d<kDimension;++d)p.sum[d]+=points[i*kDimension+d];
        }
        for(std::size_t i=0;i<report.k2;++i){if(replacement[i].ids.empty())throw std::runtime_error("local k-means produced an empty partition");parts_[selected[i]]=std::move(replacement[i]);}
        return report;
    }
    DeDriftSearch search(const float* query,std::size_t budget) const {
        DeDriftSearch out;out.centers=parts_.size();if(budget<out.centers+kTopK)throw std::invalid_argument("query budget too small");
        std::vector<std::pair<float,std::size_t>> order;order.reserve(parts_.size());
        for(std::size_t i=0;i<parts_.size();++i)order.emplace_back(simd_squared_l2(query,parts_[i].center.data(),kDimension),i);
        std::sort(order.begin(),order.end());out.hits.reserve(kTopK+1);
        auto better=[](const SearchResult& a,const SearchResult& b){return a.squared_l2==b.squared_l2?a.id<b.id:a.squared_l2<b.squared_l2;};
        const auto limit=budget-out.centers;
        for(auto [distance,leaf]:order){(void)distance;const auto& p=parts_[leaf];if(!p.ids.empty())++out.lists;
            for(std::size_t i=0;i<p.ids.size()&&out.vectors<limit;++i){SearchResult hit{p.ids[i],simd_squared_l2(query,p.vectors.data()+i*kDimension,kDimension)};
                auto at=std::lower_bound(out.hits.begin(),out.hits.end(),hit,better);if(out.hits.size()<kTopK||at!=out.hits.end()){out.hits.insert(at,hit);if(out.hits.size()>kTopK)out.hits.pop_back();}++out.vectors;}
            if(out.vectors==limit)break;
        }
        if(out.vectors!=limit||out.hits.size()!=kTopK)throw std::runtime_error("query budget not consumed");return out;
    }
    void validate() const {
        std::vector<bool> seen(count_,false);std::size_t found=0;
        for(const auto& p:parts_){if(p.vectors.size()!=p.ids.size()*kDimension)throw std::runtime_error("posting payload mismatch");for(auto id:p.ids){if(id>=count_||seen[id])throw std::runtime_error("invalid/duplicate ID");seen[id]=true;++found;}}
        if(found!=count_)throw std::runtime_error("missing vectors");
    }
    std::size_t size()const{return count_;}
    std::size_t nlist()const{return parts_.size();}
    const std::vector<DeDriftPartition>& partitions()const{return parts_;}
private:
    void append(DeDriftPartition& p,std::uint32_t id){if(id>=rows_)throw std::out_of_range("vector ID");p.ids.push_back(id);const float* v=data_+id*kDimension;p.vectors.insert(p.vectors.end(),v,v+kDimension);for(std::size_t d=0;d<kDimension;++d)p.sum[d]+=v[d];++count_;}
    std::size_t nearest(const float* x)const{std::size_t best=0;float score=simd_squared_l2(x,parts_[0].center.data(),kDimension);for(std::size_t i=1;i<parts_.size();++i){float d=simd_squared_l2(x,parts_[i].center.data(),kDimension);if(d<score){score=d;best=i;}}return best;}
    const float* data_;std::size_t rows_,count_=0;std::vector<DeDriftPartition> parts_;
};

void dedrift_self_test(){
    const std::array<std::size_t,8> sizes{30,28,27,26,24,23,22,20};std::vector<float> data(200*kDimension),centers(8*kDimension);
    std::vector<std::vector<std::uint32_t>> members(8);std::uint32_t id=0;
    for(std::size_t c=0;c<8;++c){for(std::size_t j=0;j<sizes[c];++j,++id){members[c].push_back(id);for(std::size_t d=0;d<kDimension;++d)data[id*kDimension+d]=float(c*10)+float((j*(d+1))%17)*.01f;}
        for(std::size_t d=0;d<kDimension;++d)centers[c*kDimension+d]=float(c*10)-1.f;}
    DeDriftHybrid index(data.data(),200);index.initialize(centers,members);auto r=index.maintain(4,1);index.validate();
    if(r.k2_raw!=5||r.k2!=5||r.vectors!=131||r.capped||index.nlist()!=8||index.size()!=200)throw std::runtime_error("hybrid selection/count self-test failed");
    for(std::size_t c:{4,5,6}){if(index.partitions()[c].ids!=members[c])throw std::runtime_error("untouched memberships changed");
        for(std::size_t d=0;d<kDimension;++d){double mean=0;for(auto v:members[c])mean+=data[v*kDimension+d];mean/=members[c].size();if(std::abs(index.partitions()[c].center[d]-mean)>1e-5)throw std::runtime_error("Lazy mean update failed");}}
    auto result=index.search(data.data(),208);std::vector<SearchResult> brute;for(std::size_t i=0;i<200;++i)brute.push_back({static_cast<std::int64_t>(i),simd_squared_l2(data.data(),data.data()+i*kDimension,kDimension)});
    std::sort(brute.begin(),brute.end(),[](auto a,auto b){return a.squared_l2==b.squared_l2?a.id<b.id:a.squared_l2<b.squared_l2;});
    for(std::size_t i=0;i<kTopK;++i)if(result.hits[i].id!=brute[i].id)throw std::runtime_error("exhaustive budget search self-test failed");
    std::cout<<"SELF_TEST_PASS: joint largest/smallest selection; fixed K; live-ID conservation; mean update; exact-budget search\n";
}

void run_dedrift_case(const std::filesystem::path& data_root,const std::filesystem::path& truth_root,
 const std::filesystem::path& output,const std::string& name,const std::vector<float>& centers,
 const std::vector<std::vector<std::uint32_t>>& members,const MappedFbin& reference,std::size_t k,
 std::size_t threads,std::size_t divisor,std::size_t max_stage,std::size_t query_count,std::ofstream& aggregate){
    MappedFbin data(data_root/name/"base.fbin");if(data.rows()!=kInitialRows+kStages*kBatchRows||std::memcmp(data.row(0),reference.row(0),kInitialRows*kDimension*sizeof(float))!=0)throw std::runtime_error("shared initial data mismatch in "+name);
    auto truth=read_truth(truth_root/(name+".u32bin"));auto folder=output/name;std::filesystem::create_directory(folder);
    std::ofstream stage_csv(folder/"stage_summary.csv");stage_csv<<"case,stage,active_vectors,nlist,k,k2_raw,k2_used,k2_capped,reclustered_vectors,reassigned_vectors,insert_ms,maintenance_ms,cumulative_update_s,budget_per_query,query_count,recall_at_10,mean_query_ms,p95_query_ms,mean_center_distances,mean_vector_distances,mean_total_distances\n";
    DeDriftHybrid index(data.row(0),data.rows());index.initialize(centers,members);double sum_recall=0,insert_total=0,maintenance_total=0,final_recall=0;std::size_t reclustered_total=0,capped_total=0;
    for(std::size_t stage=0;stage<=max_stage;++stage){double insert_ms=0,maintenance_ms=0;DeDriftMaintenance report;
        if(stage){auto start=std::chrono::steady_clock::now();index.add(kInitialRows+(stage-1)*kBatchRows,kBatchRows);insert_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();start=std::chrono::steady_clock::now();report=index.maintain(k,stage);maintenance_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
        index.validate();if(index.nlist()!=kInitialNlist||index.size()!=kInitialRows+stage*kBatchRows)throw std::runtime_error("population/K invariant failed");
        insert_total+=insert_ms;maintenance_total+=maintenance_ms;reclustered_total+=report.vectors;capped_total+=report.capped;
        std::ostringstream filename;filename<<"query_"<<std::setw(3)<<std::setfill('0')<<stage<<".fbin";MappedFbin queries(data_root/name/filename.str());if(queries.rows()!=kQueries)throw std::runtime_error("query shape mismatch");
        const auto budget=index.size()/divisor;struct Outcome{std::size_t hits=0,vectors=0;double ms=0;};std::vector<Outcome> outcomes(query_count);std::atomic<std::size_t> next{0};std::exception_ptr error;std::mutex mutex;
        auto worker=[&]{while(true){auto q=next.fetch_add(1);if(q>=query_count)return;try{auto start=std::chrono::steady_clock::now();auto found=index.search(queries.row(q),budget);auto& o=outcomes[q];o.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();o.vectors=found.vectors;const auto& exact=truth[stage*kQueries+q];for(auto hit:found.hits)if(std::find(exact.begin(),exact.end(),hit.id)!=exact.end())++o.hits;}catch(...){std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();return;}}};
        std::vector<std::thread> workers;for(std::size_t t=0;t<std::min(threads,query_count);++t)workers.emplace_back(worker);for(auto& t:workers)t.join();if(error)std::rethrow_exception(error);
        double hits=0,scanned=0;std::vector<double> ms;for(auto o:outcomes){hits+=o.hits;scanned+=o.vectors;ms.push_back(o.ms);}final_recall=hits/(query_count*kTopK);if(stage)sum_recall+=final_recall;
        stage_csv<<std::setprecision(12)<<name<<','<<stage<<','<<index.size()<<','<<index.nlist()<<','<<k<<','<<report.k2_raw<<','<<report.k2<<','<<report.capped<<','<<report.vectors<<','<<report.reassigned<<','<<insert_ms<<','<<maintenance_ms<<','<<(insert_total+maintenance_total)/1000<<','<<budget<<','<<query_count<<','<<final_recall<<','<<std::accumulate(ms.begin(),ms.end(),0.)/query_count<<','<<percentile(ms,.95)<<','<<index.nlist()<<','<<scanned/query_count<<','<<index.nlist()+scanned/query_count<<'\n';stage_csv.flush();
        std::cout<<"DeDrift-Hybrid case="<<name<<" stage="<<stage<<'/'<<max_stage<<" recall="<<final_recall<<" K="<<index.nlist()<<" k2="<<report.k2<<" reclustered="<<report.vectors<<std::endl;
    }
    aggregate<<std::setprecision(12)<<name<<",dedrift_hybrid,"<<k<<','<<max_stage<<','<<query_count<<','<<index.nlist()<<','<<(max_stage?sum_recall/max_stage:final_recall)<<','<<final_recall<<','<<insert_total/1000<<','<<maintenance_total/1000<<','<<(insert_total+maintenance_total)/1000<<','<<reclustered_total<<','<<capped_total<<'\n';aggregate.flush();
}
} // namespace

int main(int argc,char** argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){dedrift_self_test();return 0;}
    if(argc<5||argc>11){std::cerr<<"Usage: deep1m_dedrift_freewayml_runner DATA_ROOT INITIAL_SNAPSHOT TRUTH_ROOT OUTPUT [k=4] [threads=16] [budget_divisor=100] [case=all] [stages=100] [queries=1000]\n";return 2;}
    std::filesystem::path data_root=argv[1],snapshot_path=argv[2],truth_root=argv[3],output=argv[4];
    std::size_t k=argc>5?std::stoull(argv[5]):4,threads=argc>6?std::stoull(argv[6]):16,divisor=argc>7?std::stoull(argv[7]):100,max_stage=argc>9?std::stoull(argv[9]):100,nq=argc>10?std::stoull(argv[10]):1000;std::string selected=argc>8?argv[8]:"all";
    if(!k||k>=kInitialNlist||!threads||!divisor||max_stage>kStages||!nq||nq>kQueries)throw std::invalid_argument("invalid experiment parameters");
    std::vector<std::string> cases;for(auto name:kCases)if(selected=="all"||selected==name)cases.emplace_back(name);if(cases.empty())throw std::invalid_argument("unknown scenario");
    if(std::filesystem::is_directory(truth_root/"_prepared_truth"))truth_root/="_prepared_truth";
    if(std::filesystem::exists(output)&&(!std::filesystem::is_directory(output)||!std::filesystem::is_empty(output)))throw std::runtime_error("output must be new or empty");std::filesystem::create_directories(output);
    auto snapshot=AdaptiveHierarchicalIvf::load(snapshot_path.string());const auto& cfg=snapshot.config();if(snapshot.size()!=kInitialRows||snapshot.leaf_nlist()!=kInitialNlist||cfg.dimension!=kDimension||cfg.uint8_storage||cfg.lvq4_storage||cfg.lvq8_storage||cfg.angular_metric)throw std::runtime_error("invalid shared initial DEEP-1M snapshot");
    auto centers=snapshot.leaf_centroids();auto members=snapshot.leaf_vector_ids();MappedFbin reference(data_root/"G/base.fbin");
    std::ofstream manifest(output/"manifest.txt");manifest<<"method=DeDrift-Hybrid; uncompressed Section 5.1 adaptation\ndata_root="<<data_root.string()<<"\ninitial_snapshot="<<snapshot_path.string()<<"\ntruth_root="<<truth_root.string()<<"\nK=512 fixed; k="<<k<<"; local_kmeans_iterations=10; seed=1234\nmaintenance=once after every insertion batch; Lazy mean update then Split; stage0 has no maintenance\nquery_budget=N/"<<divisor<<" including 512 center distances; threads="<<threads<<"; no warmup or early stopping\nstages="<<max_stage<<"; queries_per_stage="<<nq<<"; case="<<selected<<"\nmean_recall=stages 1..last, excluding stage0\npayload=96-D FP32 as in existing DEEP-1M experiments; packed vectors in each posting\nLazy uses exact sufficient-statistic mean updates; memberships unchanged until Split\nk2=ceil(sum of k largest sizes / global median size); jointly recluster largest + smallest lists\nk2 safety cap=K; raw/used/cap flag logged for every stage\nconstruction, validation and query costs excluded from update cost\nno index snapshots, per-query logs, or returned-neighbor arrays saved\n";manifest.flush();
    std::ofstream all(output/"all_cases_summary.csv");all<<"case,method,k,update_stages,queries,final_K,mean_recall_at_10,final_recall_at_10,insert_s,maintenance_s,update_s,reclustered_vectors,k2_cap_events\n";
    for(auto name:cases)run_dedrift_case(data_root,truth_root,output,name,centers,members,reference,k,threads,divisor,max_stage,nq,all);
    std::ofstream(output/(max_stage==kStages&&nq==kQueries?"COMPLETE.txt":"SMOKE_COMPLETE.txt"))<<"completed_cases="<<cases.size()<<"\nmethod=DeDrift-Hybrid\nk="<<k<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<std::endl;return 1;}}

