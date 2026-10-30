// Research-only cached Qwen3.5 last-down training. No Python or decoder loader.
#include <torch/torch.h>
#include <torch/mps.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/Parallel.h>
#include <libproc.h>
#include <cerrno>
#include <cmath>
#include <CommonCrypto/CommonDigest.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <cstdlib>
#include <climits>

namespace fs = std::filesystem;
using Tensor = torch::Tensor;
using torch::indexing::Slice;
const char* ABI = "LUBAN_NATIVE_TRAINING_V1";
#ifdef LUBAN_REUSE_REPLAY_FORWARD
static_assert(LUBAN_REUSE_REPLAY_FORWARD == 1, "Replay reuse macro must be1");
constexpr bool REUSE_REPLAY_FORWARD = true;
#else
constexpr bool REUSE_REPLAY_FORWARD = false;
#endif
#ifdef LUBAN_RELEASE_CE_GRAPH
static_assert(LUBAN_RELEASE_CE_GRAPH == 1 && REUSE_REPLAY_FORWARD, "CE release requires replay reuse");
constexpr bool RELEASE_CE_GRAPH = true;
#else
constexpr bool RELEASE_CE_GRAPH = false;
#endif
void need(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> out; std::stringstream stream(text); std::string part;
  while (std::getline(stream, part, delimiter)) out.push_back(part);
  return out;
}
int64_t integer(const std::string& text) {
  need(std::regex_match(text, std::regex("[0-9]+")), "Unsigned integer required");
  size_t used=0; auto n=std::stoll(text,&used); need(used==text.size(), "Invalid integer"); return n;
}
std::string hashFile(const fs::path& path) {
  std::ifstream stream(path,std::ios::binary); need(bool(stream), "Cannot read SHA input");
  CC_SHA256_CTX state; CC_SHA256_Init(&state); char block[1024*1024];
  while (stream) { stream.read(block,sizeof(block)); auto count=stream.gcount(); if(count) CC_SHA256_Update(&state,block,CC_LONG(count)); }
  need(stream.eof(), "SHA stream failed"); unsigned char hash[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(hash,&state);
  std::ostringstream out; out<<std::hex<<std::setfill('0'); for(auto byte:hash) out<<std::setw(2)<<int(byte); return out.str();
}
fs::path safePath(const fs::path& path) { auto p=fs::absolute(path).lexically_normal(); need(fs::canonical(p)==p, "Symlink/noncanonical path rejected"); return p; }
struct Spec { std::string name,dtype,file,sha; std::vector<int64_t> shape; uint64_t bytes; };
struct Row { std::string id,role,sha; int64_t length,gold; };
struct Step { int64_t index; std::string role,ce,replay; };
struct Manifest {
  fs::path root; std::map<std::string,std::string> config; std::map<std::string,Spec> tensors;
  std::map<std::string,Row> rows; std::vector<Step> steps;
};
Manifest parse(const fs::path& root,const std::string& approved) {
  Manifest m; m.root=safePath(root); auto file=m.root/"manifest.tsv";
  need(hashFile(safePath(file))==approved,"Manifest SHA mismatch");
  std::ifstream stream(file); std::string line; need(bool(std::getline(stream,line))&&line==ABI,"Wrong native fixture ABI");
  std::set<std::string> files;
  while(std::getline(stream,line)) {
    auto p=split(line,'\t'); need(!p.empty(),"Empty manifest row");
    if(p[0]=="config") { need(p.size()==3&&m.config.emplace(p[1],p[2]).second,"Duplicate/malformed config"); }
    else if(p[0]=="tensor") {
      need(p.size()==7&&std::regex_match(p[1],std::regex("[a-z][a-z0-9_]*"))&&std::regex_match(p[4],std::regex("[a-z][a-z0-9_]*\\.raw")),"Malformed tensor/path");
      Spec s{p[1],p[2],p[4],p[6],{},uint64_t(integer(p[5]))};
      if(p[3]!="-") for(auto& dimension:split(p[3],',')) { auto n=integer(dimension);need(n>0,"Empty dimension");s.shape.push_back(n); }
      need(s.shape.size()<=3&&std::regex_match(s.sha,std::regex("[0-9a-f]{64}")),"Tensor shape/SHA invalid");
      need(m.tensors.emplace(s.name,s).second&&files.insert(s.file).second,"Duplicate tensor/file");
    } else if(p[0]=="row") {
      need(p.size()==6&&std::regex_match(p[1],std::regex("r[0-9]+"))&&(p[2]=="new"||p[2]=="replay")&&std::regex_match(p[3],std::regex("[0-9a-f]{64}")),"Malformed TRAIN row");
      Row r{p[1],p[2],p[3],integer(p[4]),integer(p[5])};need(r.length>=2&&r.gold>0&&r.gold<r.length&&m.rows.emplace(r.id,r).second,"Invalid complete TRAIN row");
    } else if(p[0]=="step") {
      need(p.size()==5&&(p[2]=="new"||p[2]=="replay"),"Malformed schedule");m.steps.push_back({integer(p[1]),p[2],p[3],p[4]});
    } else throw std::runtime_error("Unknown manifest record");
  }
  need(stream.eof(),"Manifest read failed");
  std::set<std::string> expected{"identity","device","steps","norm_eps","train_eos","checkpoint_steps"};
  for(auto& key:expected) need(m.config.count(key),"Missing fixed config");
  for(const auto& [key,value]:m.config)need(expected.count(key)||key=="training_offset"||key=="learning_rate"||key=="distillation_weight","Unexpected config");
  need(std::regex_match(m.config.at("identity"),std::regex("[0-9a-f]{64}")),"Base identity invalid");
  need(m.config.at("device")=="cpu"||m.config.at("device")=="mps","Device must CPU/MPS");
  need(m.config.at("norm_eps")=="1e-6","Actual Qwen norm eps required");
  auto steps=integer(m.config.at("steps"));
  need(steps>=1&&steps<=4096&&m.steps.size()==size_t(steps),"Native steps must1..4096");
  int64_t last=0;for(const auto& item:split(m.config.at("checkpoint_steps"),',')){auto n=integer(item);need(n>last&&n<=steps,"Invalid checkpoint order/range");last=n;}
  need(last==steps,"Final checkpoint required");
  auto offset=m.config.count("training_offset")?integer(m.config.at("training_offset")):0;
  need(offset<=10000&&offset+steps<=10000,"Training offset invalid");
  for(size_t i=0;i<m.steps.size();i++) {
    const auto& s=m.steps[i]; need(s.index==int64_t(i+1)&&m.rows.count(s.ce)&&m.rows.count(s.replay),"Schedule index/row invalid");
    need(s.role==((offset+i)%2==0?"new":"replay")&&m.rows.at(s.ce).role==s.role&&m.rows.at(s.replay).role=="replay","Alternating CE/KL actual TRAIN role differs");
  }
  return m;
}
struct ProcessLock {
  int fd=-1;
  ProcessLock(const std::string& identity,const std::string& device) {
    auto dir=fs::path("/tmp")/("luban-model-locks-"+std::to_string(getuid()));
    if(::mkdir(dir.c_str(),0700)!=0) need(errno==EEXIST,"Cannot create model lock directory");
    struct stat info;need(lstat(dir.c_str(),&info)==0&&S_ISDIR(info.st_mode)&&info.st_uid==getuid()&&(info.st_mode&0077)==0,"Unsafe model lock directory");
    auto path=dir/(identity+"-"+device+".lock");
    if(const char* inherited=std::getenv("LUBAN_MODEL_LOCK_FD")) {
      auto n=integer(inherited);need(n<=INT_MAX,"Inherited lock fd invalid");
      struct stat source{},target{};
      need(fstat(int(n),&source)==0&&lstat(path.c_str(),&target)==0&&S_ISREG(source.st_mode)&&S_ISREG(target.st_mode)&&source.st_dev==target.st_dev&&source.st_ino==target.st_ino&&source.st_uid==getuid()&&source.st_nlink==1&&(source.st_mode&0077)==0,"Inherited model lock does not match actual protected file");
      fd=::dup(int(n));
    } else fd=::open(path.c_str(),O_RDWR|O_CREAT|O_NOFOLLOW,0600);
    if(fd<0) throw std::runtime_error("Cannot open model process lock");
    if(fstat(fd,&info)!=0||!S_ISREG(info.st_mode)||info.st_uid!=getuid()||info.st_nlink!=1||(info.st_mode&0077)!=0||flock(fd,LOCK_EX|LOCK_NB)!=0) {
      ::close(fd);fd=-1;throw std::runtime_error("Unsafe or already-owned base/device model lock");
    }
  }
  ~ProcessLock(){if(fd>=0)::close(fd);}
};
torch::ScalarType scalar(const std::string& type) {
  if(type=="f32")return torch::kFloat32;if(type=="bf16")return torch::kBFloat16;
  if(type=="i64")return torch::kInt64;if(type=="u8")return torch::kUInt8;throw std::runtime_error("Unsupported tensor dtype");
}
void finite(const Tensor& t) { need(torch::isfinite(t).all().item<bool>(),"Nonfinite native tensor"); }
Tensor load(const Manifest& m,const std::string& name,const torch::Device& device) {
  need(m.tensors.count(name),"Missing tensor "+name);const auto& s=m.tensors.at(name);
  auto path=safePath(m.root/s.file); need(fs::file_size(path)==s.bytes&&hashFile(path)==s.sha,"Raw tensor bytes/SHA mismatch");
  auto dtype=scalar(s.dtype);uint64_t elements=1;
  for(auto d:s.shape){need(elements<=uint64_t(std::numeric_limits<int64_t>::max())/d,"Tensor dimension overflow");elements*=d;}
  auto width=c10::elementSize(dtype);need(elements<=std::numeric_limits<uint64_t>::max()/width&&elements*width==s.bytes,"Tensor exact byte count differs");
  Tensor cpu=torch::empty(s.shape,torch::TensorOptions().dtype(dtype).device(torch::kCPU));
  std::ifstream stream(path,std::ios::binary);stream.read(static_cast<char*>(cpu.data_ptr()),s.bytes);need(bool(stream),"Short raw tensor read");
  need(hashFile(path)==s.sha,"Raw tensor changed during load");finite(cpu);return cpu.to(device);
}
void sync(const torch::Device& device){if(device.type()==torch::kMPS)torch::mps::synchronize();}
struct Sample {Tensor x,residual,ids,labels;};
Sample sample(const Manifest& m,const Row& row,const torch::Device& device,int64_t input,int64_t hidden,int64_t vocab) {
  Sample s{load(m,row.id+"_x",device),load(m,row.id+"_residual",device),load(m,row.id+"_ids",device),load(m,row.id+"_labels",device)};
  need(s.x.sizes()==torch::IntArrayRef({1,row.length,input})&&s.residual.sizes()==torch::IntArrayRef({1,row.length,hidden}),"Full X/residual shape differs");
  need(s.ids.sizes()==torch::IntArrayRef({1,row.length})&&s.ids.scalar_type()==torch::kInt64&&s.labels.sizes()==s.ids.sizes()&&s.labels.scalar_type()==torch::kInt64,"IDs/labels dtype/shape differs");
  auto ids=s.ids.to(torch::kCPU),labels=s.labels.to(torch::kCPU);auto a=ids.accessor<int64_t,2>(),b=labels.accessor<int64_t,2>();auto eos=integer(m.config.at("train_eos"));
  bool started=false;int64_t gold=0;
  for(int64_t i=0;i<row.length;i++) {
    need(a[0][i]>=0&&a[0][i]<vocab,"Input token out of full vocabulary");
    if(b[0][i]==-100){need(!started,"Gold labels interrupted");continue;}
    need(i>=1&&b[0][i]==a[0][i],"Gold/causal prefix labels differ");started=true;++gold;
    need(i==row.length-1?b[0][i]==eos:b[0][i]!=eos,"Training tokenizer EOS contract differs");
  }
  need(started&&gold==row.gold&&b[0][row.length-1]==eos,"Missing complete gold+EOS");return s;
}
Tensor logits(const Sample& s,const Tensor& weight,const Tensor& norm,const Tensor& head,const Tensor& bias,const Tensor& headBias) {
  auto z=torch::nn::functional::linear(s.x,weight,bias)+s.residual;
  auto f=z.to(torch::kFloat32);auto normalized=f*torch::rsqrt(f.pow(2).mean(-1,true)+1e-6);
  normalized=(normalized*(1.0+norm.to(torch::kFloat32))).to(z.scalar_type());
  auto out=torch::nn::functional::linear(normalized,head,headBias);finite(out);return out;
}
Tensor answerLog(const Tensor& values,const Tensor& labels) {
  auto targets=labels.index({Slice(),Slice(1)});auto mask=targets.ne(-100);
  return torch::log_softmax(values.index({Slice(),Slice(0,-1)}).to(torch::kFloat32).index({mask}),-1);
}
std::string typeName(const Tensor& value){
  if(value.scalar_type()==torch::kFloat32)return "f32";if(value.scalar_type()==torch::kBFloat16)return "bf16";
  if(value.scalar_type()==torch::kInt64)return "i64";if(value.scalar_type()==torch::kUInt8)return "u8";throw std::runtime_error("Unsupported output dtype");
}
void save(const fs::path& output,const std::string& name,const Tensor& value,std::ofstream& listing) {
  auto cpu=value.detach().to(torch::kCPU).contiguous();auto path=output/(name+".raw");need(!fs::exists(path),"Output tensor already exists");
  std::ofstream stream(path,std::ios::binary);need(bool(stream),"Cannot write tensor output");
  stream.write(static_cast<const char*>(cpu.data_ptr()),cpu.nbytes());stream.close();need(bool(stream),"Short tensor output write");
  std::ostringstream shape;if(cpu.dim()==0)shape<<'-';else for(int64_t i=0;i<cpu.dim();i++){if(i)shape<<',';shape<<cpu.size(i);}
  listing<<"tensor\t"<<name<<'\t'<<typeName(cpu)<<'\t'<<shape.str()<<'\t'<<path.filename().string()<<'\t'<<cpu.nbytes()<<'\t'<<hashFile(path)<<'\n';listing.flush();
}
void nativeAdam(Tensor& master,Tensor& avg,Tensor& sq,int64_t& step,double learningRate) {
  // Match installed Torch2.8 optim/adam.py single-tensor noncapturable FP32 order.
  // Default C++ Adam instead uses mul/add for avg and is not this control.
  torch::NoGradGuard guard;++step;
  avg.lerp_(master.grad(),1-.9);
  sq.mul_(.999).addcmul_(master.grad(),master.grad(),1-.999);
  double bias1=1-std::pow(.9,double(step)),bias2=1-std::pow(.999,double(step));
  auto denom=(sq.sqrt()/std::sqrt(bias2)).add_(1e-6);
  master.addcdiv_(avg,denom,-learningRate/bias1);
}
std::pair<uint64_t,uint64_t> footprint() {
  rusage_info_v4 info{};
  need(proc_pid_rusage(getpid(),RUSAGE_INFO_V4,reinterpret_cast<rusage_info_t*>(&info))==0,"Darwin process footprint unavailable");
  return {info.ri_phys_footprint,info.ri_lifetime_max_phys_footprint};
}
int main(int argc,char** argv) {
  fs::path output;Clock::time_point all=Clock::now();bool ownsOutput=false;
  try {
    need(argc==7&&std::string(argv[1])=="--input"&&std::string(argv[3])=="--output"&&std::string(argv[5])=="--manifest-sha","Usage: --input fixture --output fresh --manifest-sha exactSHA");
    const uint16_t endian=1;need(*reinterpret_cast<const uint8_t*>(&endian)==1,"LE ABI only");
    auto executable=safePath(argv[0]);auto executableSHA=hashFile(executable);
    auto m=parse(argv[2],argv[6]);torch::Device device(m.config.at("device"));
    auto number=[&](const std::string& key,double fallback){if(!m.config.count(key))return fallback;size_t used=0;auto value=std::stod(m.config.at(key),&used);need(used==m.config.at(key).size()&&std::isfinite(value),"Invalid numeric config");return value;};
    double learningRate=number("learning_rate",1e-5),distillationWeight=number("distillation_weight",5);
    need(learningRate>=1e-8&&learningRate<=.01&&distillationWeight>=0&&distillationWeight<=100,"Native LR/KL range invalid");
    std::set<int64_t> checkpoints;for(const auto& item:split(m.config.at("checkpoint_steps"),','))checkpoints.insert(integer(item));
    ProcessLock lock(m.config.at("identity"),m.config.at("device"));
    at::set_num_threads(1);at::set_num_interop_threads(1);
    need(device.type()!=torch::kMPS||torch::mps::is_available(),"Actual MPS unavailable");
    output=fs::absolute(argv[4]).lexically_normal();need(safePath(output.parent_path())==output.parent_path()&&!fs::exists(output),"Fresh independent output required");
    need(output!=m.root&&output.string().find(m.root.string()+"/")!=0&&m.root.string().find(output.string()+"/")!=0,"Input/output roots must be disjoint");
    need(fs::create_directory(output),"Cannot create native output");ownsOutput=true;
    std::ofstream listing(output/"tensors.tsv");listing<<"LUBAN_NATIVE_TRAINING_OUTPUT_V1\n";
    auto loadStart=Clock::now();auto master=load(m,"master",device);auto reference=load(m,"reference",device);auto basis=load(m,"basis",device);
    auto candidate=load(m,"candidate",device),norm=load(m,"norm",device),head=load(m,"head",device);
    auto avg=load(m,"exp_avg",device),sq=load(m,"exp_avg_sq",device),entry=load(m,"adam_step",torch::kCPU);
    Tensor bias=m.tensors.count("down_bias")?load(m,"down_bias",device):Tensor();
    Tensor headBias=m.tensors.count("head_bias")?load(m,"head_bias",device):Tensor();
    need(master.dim()==2&&master.scalar_type()==torch::kFloat32&&reference.sizes()==master.sizes()&&reference.scalar_type()==torch::kFloat32,"FP32 full master/reference required");
    need(candidate.sizes()==master.sizes()&&(candidate.scalar_type()==torch::kBFloat16||candidate.scalar_type()==torch::kFloat32)&&torch::equal(master.to(candidate.scalar_type()),candidate),"Native candidate must exact master cast");
    need(norm.dim()==1&&norm.size(0)==master.size(0)&&(norm.scalar_type()==torch::kFloat32||norm.scalar_type()==candidate.scalar_type()),"Actual norm tensor dtype/shape required");
    need(head.dim()==2&&head.size(1)==master.size(0)&&head.scalar_type()==candidate.scalar_type(),"Native full vocab head required");
    need(basis.dim()==2&&basis.size(0)==master.size(1)&&basis.size(1)>0&&basis.size(1)<master.size(1)&&basis.scalar_type()==torch::kFloat32,"Replay FP32 basis must leave trainable complement");
    need(avg.sizes()==master.sizes()&&sq.sizes()==master.sizes()&&avg.scalar_type()==torch::kFloat32&&sq.scalar_type()==torch::kFloat32&&sq.ge(0).all().item<bool>(),"Adam moments invalid");
    need(entry.dim()==0&&entry.scalar_type()==torch::kFloat32&&entry.item<double>()>=0&&entry.item<double>()<=10000&&std::floor(entry.item<double>())==entry.item<double>(),"Actual integer Adam step invalid");
    auto originalStep=entry.item<int64_t>();if(originalStep==0)need(avg.eq(0).all().item<bool>()&&sq.eq(0).all().item<bool>(),"Fresh Adam moments must zero");
    need(!m.config.count("training_offset")||originalStep==integer(m.config.at("training_offset")),"Adam step and training offset differ");
    if(bias.defined())need(bias.dim()==1&&bias.size(0)==master.size(0)&&bias.scalar_type()==candidate.scalar_type(),"Actual down bias invalid");
    if(headBias.defined())need(headBias.dim()==1&&headBias.size(0)==head.size(0)&&headBias.scalar_type()==candidate.scalar_type(),"Actual head bias invalid");
    std::set<std::string> allowed{"master","reference","basis","candidate","norm","head","exp_avg","exp_avg_sq","adam_step"};if(bias.defined())allowed.insert("down_bias");if(headBias.defined())allowed.insert("head_bias");
    std::map<std::string,Sample> samples;
    for(const auto& [id,row]:m.rows){auto s=sample(m,row,device,master.size(1),master.size(0),head.size(0));need(s.x.scalar_type()==candidate.scalar_type()&&s.residual.scalar_type()==candidate.scalar_type(),"X/R native precision differs");samples.emplace(id,std::move(s));for(auto suffix:{"_x","_residual","_ids","_labels"})allowed.insert(id+suffix);}
    need(allowed.size()==m.tensors.size(),"Unknown tensor outside fixed cached tail");for(auto& name:allowed)need(m.tensors.count(name),"Missing declared tensor");
    uint64_t required=4*master.nbytes()+candidate.nbytes()+basis.nbytes()+4+1024*1024;
    for(const auto& step:m.steps)if(checkpoints.count(step.index)) {
      required+=5*master.nbytes()+4*candidate.nbytes()+4;
      required+=uint64_t(m.rows.at(step.ce).gold+2*m.rows.at(step.replay).gold)*head.size(0)*4;
    }
    need(fs::space(output).available>=required,"Insufficient space for all predeclared raw audit checkpoints");
    master.requires_grad_(true);int64_t stateStep=originalStep;
    sync(device);double loadSeconds=elapsed(loadStart);std::ofstream metrics(output/"trace.tsv");metrics<<std::setprecision(17)<<"step\tce_role\tce_row\treplay_row\tce\tkl\tpreclip_norm\tloss\twall_seconds\n";
    std::ofstream phases(output/"phases.jsonl");phases<<std::setprecision(17);
    int64_t newCount=0,replayCount=0,klCount=0,reusedStudentForwards=0,ceForwards=0,teacherForwards=0,studentForwards=0;
    for(const auto& step:m.steps) {
      sync(device);auto began=Clock::now();const auto& ceSample=samples.at(step.ce);const auto& replay=samples.at(step.replay);
      master.mutable_grad()=Tensor();auto weight=master.detach().to(candidate.scalar_type()).clone().requires_grad_(true);
      std::vector<Tensor> gradients;auto hook=weight.register_hook([&](const Tensor& g){gradients.push_back(g.detach().clone());return g;});
      const bool reuseStudentForward=REUSE_REPLAY_FORWARD&&step.role=="replay"&&step.ce==step.replay;
      auto phase=Clock::now();auto ceLogits=logits(ceSample,weight,norm,head,bias,headBias);++ceForwards;
      auto shifted=ceSample.labels.index({Slice(),Slice(1)});
      auto ce=torch::nn::functional::cross_entropy(ceLogits.index({Slice(),Slice(0,-1)}).to(torch::kFloat32).reshape({-1,head.size(0)}),shifted.reshape({-1}),torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
      // Keep the same full-sequence graph only for an identical cached REPLAY row.
      // CE and weighted KL still run two separate backwards, in that order.
      finite(ce);ce.backward({},reuseStudentForward);
      // KL needs the shared tail graph through ceLogits, never the CE-only loss branch.
      if(RELEASE_CE_GRAPH)ce=ce.detach();
      sync(device);double ceSeconds=elapsed(phase);phase=Clock::now();Tensor teacher;
      {torch::NoGradGuard noGrad;teacher=answerLog(logits(replay,reference.to(weight.scalar_type()),norm,head,bias,headBias),replay.labels).detach();++teacherForwards;}
      auto replayLogits=reuseStudentForward?ceLogits:logits(replay,weight,norm,head,bias,headBias);
      if(reuseStudentForward)++reusedStudentForwards;else ++studentForwards;
      auto student=answerLog(replayLogits,replay.labels);
      auto kl=torch::nn::functional::kl_div(student,teacher,torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean).log_target(true));
      finite(kl);(distillationWeight*kl).backward();weight.remove_hook(hook);sync(device);double klSeconds=elapsed(phase);phase=Clock::now();
      need(gradients.size()==2,"Exactly separate CE then KL backward required");finite(weight.grad());auto total=weight.grad().detach().clone();master.mutable_grad()=total.to(torch::kFloat32).clone();
      auto normValue=at::linalg_vector_norm(master.grad(),2.0);
      auto totalNorm=at::linalg_vector_norm(torch::stack({normValue}),2.0);finite(totalNorm);
      double preclip=totalNorm.item<double>();master.grad().mul_(torch::clamp_max(1.0/(totalNorm+1e-6),1.0));
      auto clipped=master.grad().detach().clone();nativeAdam(master,avg,sq,stateStep,learningRate);
      {torch::NoGradGuard noGrad;auto delta=master-reference;master.copy_(reference+delta-torch::matmul(torch::matmul(delta,basis),basis.t()));candidate=master.detach().to(candidate.scalar_type()).clone();}
      finite(master);finite(candidate);
      need(stateStep==originalStep+step.index,"Adam step monotonicity failed");finite(avg);finite(sq);sync(device);double updateSeconds=elapsed(phase);phase=Clock::now();
      std::ostringstream key;key<<"checkpoint"<<std::setw(6)<<std::setfill('0')<<step.index<<'_';auto prefix=key.str();
      bool checkpoint=checkpoints.count(step.index);
      if(checkpoint) {
        // Forward remains full-sequence/full-vocabulary. Only audit storage uses causal gold positions.
        auto ceMask=ceSample.labels.index({Slice(),Slice(1)}).ne(-100);
        auto ceGold=ceLogits.index({Slice(),Slice(0,-1)}).to(torch::kFloat32).index({ceMask});
        for(const auto& [name,value]:std::vector<std::pair<std::string,Tensor>>{{"ce_logits",ceGold},{"teacher_log_probs",teacher},{"student_log_probs",student},{"ce_grad",gradients[0]},{"kl_grad",gradients[1]},{"total_grad",total},{"clipped_grad",clipped},{"master",master},{"candidate",candidate},{"exp_avg",avg},{"exp_avg_sq",sq},{"adam_step",torch::tensor(float(stateStep))},{"delta",master-reference}})save(output,prefix+name,value,listing);
      }
      newCount+=step.role=="new";replayCount+=step.role=="replay";++klCount;
      phases<<"{\"step\":"<<step.index<<",\"reused_student_forward\":"<<(reuseStudentForward?"true":"false")<<",\"adam_step\":"<<stateStep<<",\"ce_role\":\""<<step.role<<"\",\"ce_row\":\""<<step.ce<<"\",\"kl_row\":\""<<step.replay<<"\",\"ce\":"<<ce.item<double>()<<",\"kl\":"<<kl.item<double>()<<",\"preclip_norm\":"<<preclip<<",\"ce_seconds\":"<<ceSeconds<<",\"kl_seconds\":"<<klSeconds<<",\"update_seconds\":"<<updateSeconds<<",\"raw_save_seconds\":"<<elapsed(phase)<<",\"whole_step_seconds\":"<<elapsed(began)<<"}\n";phases.flush();need(bool(phases),"Cannot save step phases");
      metrics<<step.index<<'\t'<<step.role<<'\t'<<step.ce<<'\t'<<step.replay<<'\t'<<ce.item<double>()<<'\t'<<kl.item<double>()<<'\t'<<preclip<<'\t'<<(ce.item<double>()+distillationWeight*kl.item<double>())<<'\t'<<elapsed(began)<<'\n';metrics.flush();need(bool(metrics),"Cannot save step trace");
    }
    for(const auto& [name,value]:std::vector<std::pair<std::string,Tensor>>{{"master",master},{"candidate",candidate},{"exp_avg",avg},{"exp_avg_sq",sq},{"adam_step",torch::tensor(float(stateStep))},{"reference",reference},{"basis",basis}})save(output,"final_"+name,value,listing);
    sync(device);
    need(hashFile(safePath(m.root/"manifest.tsv"))==argv[6],"Manifest changed during native training");
    for(const auto& [name,spec]:m.tensors)need(fs::file_size(safePath(m.root/spec.file))==spec.bytes&&hashFile(m.root/spec.file)==spec.sha,"Input tensor changed during native training");
    need(hashFile(executable)==executableSHA,"Native executable changed during training");
    auto memory=footprint();struct rusage usage;need(getrusage(RUSAGE_SELF,&usage)==0,"Cannot measure RSS");
    std::ostringstream summary;summary<<std::setprecision(17)<<"{\"manifest_sha256\":\""<<argv[6]<<"\",\"binary_sha256\":\""<<executableSHA<<"\",\"all_input_tensors_preserved\":true,\"input_tensor_count\":"<<m.tensors.size()<<",\"cpu_threads\":1,\"cpu_interop_threads\":1,\"memory_api\":\"Darwin proc_pid_rusage RUSAGE_INFO_V4\",\"abi\":\""<<ABI<<"\",\"worker\":\"C++LibTorch-autograd-aligned-single-tensor-Adam\",\"passed_execution\":true,\"equivalence_tested\":false,\"teacher_calls\":0,\"generation_calls\":0,\"publication\":false,\"device\":\""<<device.str()<<"\",\"steps\":"<<m.steps.size()<<",\"entry_adam_step\":"<<originalStep<<",\"final_adam_step\":"<<stateStep<<",\"new_ce_exposures\":"<<newCount<<",\"replay_ce_exposures\":"<<replayCount<<",\"replay_kl_exposures\":"<<klCount<<",\"reuse_replay_forward\":"<<(REUSE_REPLAY_FORWARD?"true":"false")<<",\"release_ce_graph\":"<<(RELEASE_CE_GRAPH?"true":"false")<<",\"reused_student_forwards\":"<<reusedStudentForwards<<",\"ce_forwards\":"<<ceForwards<<",\"teacher_forwards\":"<<teacherForwards<<",\"student_forwards\":"<<studentForwards<<",\"forward_total\":"<<(ceForwards+teacherForwards+studentForwards)<<",\"load_hash_transfer_seconds\":"<<loadSeconds<<",\"whole_worker_seconds\":"<<elapsed(all)<<",\"declared_raw_space_bytes\":"<<required<<",\"maxrss_bytes\":"<<usage.ru_maxrss<<",\"physical_footprint_bytes\":"<<memory.first<<",\"physical_footprint_peak_bytes\":"<<memory.second<<",\"mps_allocated_bytes\":"<<(device.type()==torch::kMPS?at::detail::getMPSHooks().getCurrentAllocatedMemory():0)<<",\"mps_driver_bytes\":"<<(device.type()==torch::kMPS?at::detail::getMPSHooks().getDriverAllocatedMemory():0)<<",\"adam_implementation\":\"ATen lerp/addcmul/sqrt/addcdiv matching installed Python single-tensor order\",\"rss_scope\":\"Darwin getrusage process high-water bytes\"}\n";
    std::ofstream report(output/"result.json");report<<summary.str();report.close();need(bool(report),"Cannot save native summary");std::cout<<summary.str();return 0;
  } catch(const std::exception& error) {
    std::cerr<<"Native training rejected: "<<error.what()<<'\n';
    if(ownsOutput){std::ofstream failure(output/"failure.txt");failure<<error.what()<<'\n';}
    return 1;
  }
}
