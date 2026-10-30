// Independent CPU harness for the shared terminal conversion and its exact ABI assets.
#include <torch/torch.h>
#include "EvolutionConversion.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
namespace fs=std::filesystem;
using namespace luban_conversion;
at::Tensor readRaw(const fs::path& path,std::vector<int64_t> shape){
 auto value=at::empty(shape,at::TensorOptions().dtype(at::kFloat));int fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW);require(fd>=0,"Missing CPU input");
 try{struct stat s;require(fstat(fd,&s)==0&&S_ISREG(s.st_mode)&&s.st_size==int64_t(value.nbytes()),"CPU input shape differs");size_t at=0;while(at<value.nbytes()){auto n=pread(fd,static_cast<char*>(value.data_ptr())+at,value.nbytes()-at,at);require(n>0,"CPU input truncated");at+=n;}}catch(...){close(fd);throw;}close(fd);return value;
}
void saveRaw(const fs::path& path,const at::Tensor& value){auto data=value.contiguous();int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(fd>=0,"Fresh CPU output required");try{size_t at=0;while(at<data.nbytes()){auto n=write(fd,static_cast<const char*>(data.const_data_ptr())+at,data.nbytes()-at);require(n>0,"CPU output write failed");at+=n;}require(fsync(fd)==0,"CPU output fsync failed");}catch(...){close(fd);throw;}close(fd);}
int main(int argc,char** argv){try{
 at::set_num_threads(1);at::set_num_interop_threads(1);
 if(argc==12&&std::string(argv[1])=="--convert"){
  const int64_t h=std::stoll(argv[6]),w=std::stoll(argv[7]),rank=std::stoll(argv[8]),positions=std::stoll(argv[9]),old=std::stoll(argv[10]);
  require(h>0&&w>0&&h*w<=12582912&&rank>0&&rank<w&&positions>old&&old>0&&positions*w<=10000000,"Conversion dimensions invalid");
  auto reference=readRaw(argv[2],{h,w}),master=readRaw(argv[3],{h,w}),basis=readRaw(argv[4],{w,rank}),inputs=readRaw(argv[5],{positions,w});
  const auto result=convert(master,reference,basis,inputs,old);const fs::path output=argv[11];require(fs::create_directory(output),"Fresh CPU result directory required");
  saveRaw(output/"candidate.raw",result.candidate);saveRaw(output/"master.raw",result.master);std::cout<<reportJSON(result,old,positions)<<'\n';return 0;
 }
 require(argc==3&&std::string(argv[1])=="--self-test","Use --self-test fresh-directory or --convert reference master basis inputs h w rank positions old fresh-output");
 const fs::path output=argv[2];require(fs::create_directory(output),"Fresh test directory required");
 torch::manual_seed(142);auto reference=torch::randn({3,160}).to(at::kBFloat16).to(at::kFloat),master=reference+.04*torch::randn({3,160});
 auto basis=std::get<0>(at::linalg_qr(torch::randn({160,1}),"reduced")).contiguous(),inputs=at::matmul(torch::randn({8,3}),torch::randn({3,160}));
 auto candidate=master.to(at::kBFloat16),originalMaster=master.clone(),originalCandidate=candidate.clone(),originalBasis=basis.clone();
 auto rng=at::detail::getDefaultCPUGenerator().get_state();
 require(!applyAt450(master,candidate,reference,basis,inputs,3,449,true,[](){}),"Step449 converted");require(master.equal(originalMaster)&&candidate.equal(originalCandidate),"Step449 mutated");
 int cancelled=0;for(const int stop:{1,5,12}){int count=0;try{applyAt450(master,candidate,reference,basis,inputs,3,450,true,[&](){if(++count==stop)throw std::runtime_error("cancelled");});throw std::runtime_error("Cancellation not reached");}catch(const std::exception&e){require(std::string(e.what())=="cancelled","Wrong cancellation failure");++cancelled;}require(master.equal(originalMaster)&&candidate.equal(originalCandidate),"Cancel committed partial conversion");}
 bool refused=false;try{applyAt450(master,candidate,reference,basis,inputs,3,450,false,[](){});}catch(...){refused=true;}require(refused,"Nonterminal450 accepted");
 Result result;require(applyAt450(master,candidate,reference,basis,inputs,3,450,true,[](){},&result),"Terminal450 did not convert");
 require(result.eligible&&master.to(at::kBFloat16).equal(candidate)&&basis.equal(originalBasis),"Invalid terminal state");
 require(rng.equal(at::detail::getDefaultCPUGenerator().get_state()),"Conversion advanced RNG");
 saveRaw(output/"converted_master.raw",master);require(readRaw(output/"converted_master.raw",{3,160}).to(at::kBFloat16).equal(candidate),"Saved master cannot recover candidate");
 // Tiny real-trainer inputs, derived once here; no model or product state is read.
 saveRaw(output/"reference.raw",reference);saveRaw(output/"master.raw",originalMaster);saveRaw(output/"candidate.raw",originalCandidate);saveRaw(output/"basis.raw",basis);saveRaw(output/"constraint_inputs.raw",inputs);
 saveRaw(output/"exp_avg.raw",torch::zeros_like(master));saveRaw(output/"exp_avg_sq.raw",torch::zeros_like(master));saveRaw(output/"adam_step.raw",torch::tensor(448.f));
 saveRaw(output/"norm.raw",torch::ones({3},at::kBFloat16));saveRaw(output/"head.raw",(torch::arange(24).reshape({8,3}).to(at::kFloat)/64).to(at::kBFloat16));
 saveRaw(output/"x.raw",(torch::arange(480).reshape({1,3,160}).remainder(17).to(at::kFloat)/128).to(at::kBFloat16));saveRaw(output/"residual.raw",torch::zeros({1,3,3},at::kBFloat16));
 std::cout<<"{\"terminal_boundary\":true,\"resume_bit_exact\":true,\"cancel_before_commit\":true,\"cancellation_cases\":"<<cancelled<<",\"rng_unchanged\":true,\"conversion\":"<<reportJSON(result,3,8)<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
