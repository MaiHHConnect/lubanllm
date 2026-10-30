// Standalone CPU fixture: actual source owners, expert bind/wake and router bind; no model load.
#include LUBAN_CPU_BASE_HEADER
#include LUBAN_EXPERT_RESOURCE_HEADER
#include LUBAN_ROUTER_RESOURCE_HEADER

int main(int argc, char** argv) {
  try {
    need(argc == 11, "Exact CPU permission fixture arguments required");
    at::set_num_threads(1); at::set_num_interop_threads(1); torch::NoGradGuard noGrad;
    std::array<Range,3> ranges;
    for (size_t i=0; i<3; i++) ranges[i] = {"bf16", argv[3+i], i==2?std::vector<int64_t>{16,64}:std::vector<int64_t>{64,16}, i*2048, 2048};
    ExpertResource expert(16,64,torch::kCPU); expert.bind(argv[1],argv[2],ranges);
    if (std::string(argv[10]) == "mutate") {
      const int fd=open(argv[1],O_WRONLY|O_NOFOLLOW); need(fd>=0,"Fixture mutation descriptor required");
      const char byte=42; need(pwrite(fd,&byte,1,0)==1&&fsync(fd)==0,"Fixture mutation failed"); close(fd);
    }
    expert.wake(); const auto input=at::ones({1,16},at::TensorOptions().dtype(at::kBFloat16));
    const auto first=expert.forward(input); const auto outputSHA=hashMemory(first.data_ptr(),first.nbytes());
    expert.drop(); expert.wake(); need(hashMemory(expert.forward(input).data_ptr(),first.nbytes())==outputSHA,"Cold wake output changed");
    RouterResource router(16,torch::kCPU);
    const std::array<Range,2> heads={Range{"f32",argv[8],{3,16},0,192},Range{"f32",argv[9],{3},192,12}};
    router.load(argv[6],argv[7],heads); router.check();
    const auto weight=router.projection->weight.detach().cpu().contiguous();
    std::cout<<"{\"passed\":true,\"CPU\":true,\"GPU\":false,\"model_loaded\":false,\"output_sha256\":"<<jsonString(outputSHA)
      <<",\"router_weight_sha256\":"<<jsonString(hashMemory(weight.data_ptr(),weight.nbytes()))<<",\"constructions\":"<<expert.constructions<<"}\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<"\n"; return 1; }
}
