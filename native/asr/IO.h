// SPDX-License-Identifier: Apache-2.0
// Single-owner research ASR input: numeric inherited model FD3, packed input FD4, lock FD5.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/InferenceMode.h>
#include <filesystem>
#include <map>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <signal.h>
using at::Tensor;using at::indexing::Slice;namespace fs=std::filesystem;
volatile sig_atomic_t stopped=0;
void onSignal(int){stopped=1;}
void need(bool v,const std::string& s){if(!v)throw std::runtime_error(s);}
void cancel(){need(!stopped,"ASR prefill interrupted");}
bool same(const struct stat& a,const struct stat& b){return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec&&a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec&&a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec&&a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec;}
struct Owner{
 int fd;std::string path;struct stat stat{};
 Owner(int inherited,const std::string& named):fd(dup(inherited)),path(named){struct stat n{};need(fd>=0&&fcntl(fd,F_GETFL)>=0&&(fcntl(fd,F_GETFL)&O_ACCMODE)==O_RDONLY&&fstat(fd,&stat)==0&&lstat(path.c_str(),&n)==0&&S_ISREG(stat.st_mode)&&stat.st_nlink==1&&stat.st_uid==getuid()&&same(stat,n),"Readonly inherited owner does not match named source");}
 ~Owner(){if(fd>=0)close(fd);}Owner(const Owner&)=delete;
 void check(){struct stat a{},b{};need(fstat(fd,&a)==0&&lstat(path.c_str(),&b)==0&&same(a,stat)&&same(a,b),"ASR whole owner changed");cancel();}
 void read(uint64_t offset,void* p,uint64_t bytes){need(offset<=uint64_t(stat.st_size)&&bytes<=uint64_t(stat.st_size)-offset,"Owner range outside source");uint64_t at=0;while(at<bytes){cancel();auto n=pread(fd,(char*)p+at,std::min(bytes-at,uint64_t(65536)),offset+at);if(n<0&&errno==EINTR)continue;need(n>0,"Owner range truncated");at+=n;}check();}

};
std::vector<std::string> fields(const std::string& text,char separator){std::vector<std::string> out;size_t start=0;for(;;){auto end=text.find(separator,start);out.push_back(text.substr(start,end-start));if(end==std::string::npos)break;start=end+1;}return out;}
uint64_t integer(const std::string& s){need(!s.empty()&&s.find_first_not_of("0123456789")==std::string::npos&&s==std::to_string(std::stoull(s)),"Canonical unsigned integer required");return std::stoull(s);}
