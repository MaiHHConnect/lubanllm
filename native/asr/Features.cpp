// Native CPU DSP only. The ASR model graph and transcription are not implemented by this probe.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cmath>
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int64_t number(const char* text){const std::string s(text);need(!s.empty()&&s.find_first_not_of("0123456789")==std::string::npos,"Canonical positive integer required");size_t end;auto n=std::stoll(s,&end);need(end==s.size()&&n>0&&n<=1440000,"Audio size limit");return n;}
at::Tensor read(int fd,std::vector<int64_t> shape){int64_t count=1;for(auto d:shape){need(d>0&&count<=1440000/d,"Audio tensor budget");count*=d;}struct stat s;need(fstat(fd,&s)==0&&S_ISREG(s.st_mode)&&s.st_nlink==1&&s.st_size==count*4&&(fcntl(fd,F_GETFL)&O_ACCMODE)==O_RDONLY,"Exact readonly inherited FD required");auto t=at::empty(shape,at::TensorOptions().dtype(at::kFloat).device(at::kCPU));size_t at=0;while(at<t.nbytes()){auto n=pread(fd,static_cast<char*>(t.data_ptr())+at,t.nbytes()-at,at);need(n>0,"Input truncated");at+=n;}need(at::isfinite(t).all().item<bool>(),"Nonfinite audio feature input");return t;}
void save(const std::filesystem::path& p,const at::Tensor& input){auto t=input.contiguous();int fd=open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);need(fd>=0,"Exclusive native output required");try{size_t at=0;while(at<t.nbytes()){auto n=write(fd,static_cast<const char*>(t.const_data_ptr())+at,t.nbytes()-at);need(n>0,"Output truncated");at+=n;}need(fsync(fd)==0,"Output fsync failed");}catch(...){close(fd);throw;}need(close(fd)==0,"Output close failed");}
int main(int argc,char** argv){try{
 need(argc==7,"frames rate output_frames new_output audioFD filtersFD required");const auto frames=number(argv[1]),rate=number(argv[2]),count=number(argv[3]);need(rate>=8000&&rate<=48000&&double(frames)/rate>=.05&&double(frames)/rate<=30,"Mono audio duration/rate contract");need(count<=480000&&count==std::nearbyint(double(frames)/rate*16000),"Original output frame rounding differs");
 at::set_num_threads(1);at::set_num_interop_threads(1);auto audio=read(number(argv[5]),{frames}),filters=read(number(argv[6]),{201,128});need(audio.abs().le(1).all().item<bool>(),"Audio outside unit range");
 if(rate!=16000)audio=at::upsample_linear1d(audio.view({1,1,frames}),at::IntArrayRef({count}),false,std::optional<double>{}).view({count});
 auto shortPadded=audio.numel()<8000?at::constant_pad_nd(audio,{0,8000-audio.numel()},0):audio;
 // Actual processor padding=True pads this single clip to its length, after min_length=8000.
 auto padded=shortPadded.unsqueeze(0),window=at::hann_window(400,at::TensorOptions().dtype(at::kFloat));
 auto stft=at::stft(padded,400,160,std::nullopt,window,true,"reflect",false,true,true,std::nullopt);
 auto magnitudes=stft.slice(-1,0,-1).abs().pow(2),mel=at::matmul(filters.transpose(0,1),magnitudes),log=at::clamp_min(mel,1e-10).log10();
 auto maximum=std::get<0>(log.max(2,true));maximum=std::get<0>(maximum.max(1,true));log=at::maximum(log,maximum-8);log=(log+4)/4;
 auto active=shortPadded.numel()/160,pad=(100-active%100)%100;auto mask=at::ones({1,active},at::TensorOptions().dtype(at::kInt));log=at::constant_pad_nd(log,{0,pad},0);mask=at::constant_pad_nd(mask,{0,pad},0);
 const auto output=std::filesystem::absolute(argv[4]);need(!std::filesystem::exists(output)&&std::filesystem::create_directory(output),"New output required");need(chmod(output.c_str(),0700)==0,"Private output required");save(output/"audio.raw",audio);save(output/"padded.raw",padded);save(output/"mel.raw",mel);save(output/"log.raw",log);save(output/"mask.raw",mask);
 std::cout<<"{\"passed\":true,\"frames\":"<<count<<",\"active_frames\":"<<active<<",\"pid\":"<<getpid()<<",\"CPU\":true,\"model_loaded\":false}\n";
 return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 2;}}
