// Shared resident tensors, no model loader and no Python linkage.
// QA graph and Adam order match native-image-luban-resident-v2.
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <ATen/DLConvertor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <cstring>
#include <memory>

using at::Tensor;
using torch::indexing::Slice;
struct ImageTrainer {
  // FunctionType callees borrow this archive CompilationUnit. Destroy the
  // executor first, then its owning module (reverse member order).
  torch::jit::Module module;
  std::unique_ptr<torch::jit::GraphExecutor> executor;
  std::vector<Tensor> parameters, masters, avg, sq;
  int64_t step=0, maximum;
  explicit ImageTrainer(const char* path,int64_t limit):maximum(limit) {
    TORCH_CHECK(limit>=7&&limit<=1024,"Qualified sequence limit required");
    torch::jit::GraphOptimizerEnabledGuard off(false);
    module=torch::jit::load(path,at::kCPU);
    TORCH_CHECK(module.named_parameters().size()==0&&module.named_buffers().size()==0,"Weight-free graph archive required");
    auto graph=module.get_method("forward").graph()->copy();
    TORCH_CHECK(graph->inputs().size()>0&&graph->inputs()[0]->uses().empty(),"Unused archive self only");graph->eraseInput(0);
    executor=std::make_unique<torch::jit::GraphExecutor>(graph,"image_qa");
  }
  Tensor forward(const std::vector<Tensor>& input) {
    torch::jit::GraphOptimizerEnabledGuard off(false);
    std::vector<torch::jit::IValue> args(input.begin(),input.end());
    for(auto& t:parameters)args.emplace_back(t);
    executor->run(args);TORCH_CHECK(args.size()==1&&args[0].isTensor(),"One full QA output required");auto logits=args[0].toTensor();
    TORCH_CHECK(logits.dim()==3&&logits.size(0)==1&&logits.size(1)==input[4].size(1)&&logits.size(2)==248320,"Original full QA vocabulary required");
    TORCH_CHECK(at::isfinite(logits).all().item<bool>(),"Nonfinite image QA logits");
    return logits;
  }
};
static thread_local std::string error;
template<class F> int checked(F f) {try{f();error.clear();return 0;}catch(const std::exception& e){error=e.what();return -1;}}
struct SharedTensor {DLManagedTensor* tensor;const int64_t* strides;int64_t rank;};
static Tensor take(void* pointer) {
  TORCH_CHECK(pointer,"DLPack tensor required");auto& source=*static_cast<SharedTensor*>(pointer);TORCH_CHECK(source.tensor&&source.rank>=0&&source.rank<=8,"Bound original tensor metadata required");
  auto t=at::fromDLPack(source.tensor);TORCH_CHECK(t.device().is_cpu()&&!t.is_meta()&&t.dim()==source.rank,"Resident CPU tensor required");
  // DLPack normalizes singleton-axis strides to 1. Preserve the original
  // convolution layout: a different singleton stride changes BF16 backward.
  std::vector<int64_t> strides;for(int64_t i=0;i<source.rank;i++){TORCH_CHECK(source.strides&&source.strides[i]>=0,"Original nonnegative strides required");strides.push_back(source.strides[i]);}
  return t.as_strided(t.sizes(),strides);
}
static std::vector<Tensor> inputs(SharedTensor* ptr) {std::vector<Tensor> out;for(int i=0;i<7;i++)out.push_back(take(&ptr[i]));return out;}
extern "C" {
const char* luban_image_error(){return error.c_str();}
void* luban_image_create(const char* path,int64_t maximum){ImageTrainer* p=nullptr;checked([&]{p=new ImageTrainer(path,maximum);});return p;}
void luban_image_close(void* p){delete static_cast<ImageTrainer*>(p);}
int luban_image_bind(void* p,void* tensor,int master){return checked([&]{
  auto& e=*static_cast<ImageTrainer*>(p);auto t=take(tensor);
  if(master){TORCH_CHECK(e.masters.size()<3&&t.scalar_type()==at::kFloat,"Three FP32 masters required");t=t.detach().set_requires_grad(true);e.masters.push_back(t);e.avg.push_back(at::zeros_like(t));e.sq.push_back(at::zeros_like(t));}
  e.parameters.push_back(t);
});}
int luban_image_step(void* p,SharedTensor* input,void* label,double lr,double* loss_out){return checked([&]{
  torch::jit::GraphOptimizerEnabledGuard off(false);
  auto& e=*static_cast<ImageTrainer*>(p);auto value=inputs(input);auto labels=take(label);auto ids=value[4];
  TORCH_CHECK(e.masters.size()==3&&std::isfinite(lr)&&lr>0&&lr<=.003,"Three masters and bounded learning rate required");
  TORCH_CHECK(ids.scalar_type()==at::kLong&&ids.dim()==2&&ids.size(0)==1&&ids.size(1)>=7&&ids.size(1)<=e.maximum&&labels.scalar_type()==at::kLong&&labels.sizes()==ids.sizes(),"Bounded complete QA required");
  bool gold=false;for(int64_t i=0;i<ids.size(1);i++){auto x=ids[0][i].item<int64_t>(),y=labels[0][i].item<int64_t>();TORCH_CHECK(x>=0&&x<248320,"Token out of range");if(y==-100){TORCH_CHECK(!gold,"Prefix mask only");}else {gold=true;TORCH_CHECK(y==x,"Complete answer target required");}}
  TORCH_CHECK(gold&&labels[0][-1].item<int64_t>()==248046,"Complete EOS target required");
  for(auto& t:e.masters)t.mutable_grad()=Tensor();
  auto logits=e.forward(value);auto loss=torch::nn::functional::cross_entropy(logits.index({Slice(),Slice(0,-1)}).to(at::kFloat).reshape({-1,248320}),labels.index({Slice(),Slice(1)}).reshape({-1}),torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
  TORCH_CHECK(at::isfinite(loss).item<bool>(),"Nonfinite QA loss");loss.backward();std::vector<Tensor> norms;
  for(auto& m:e.masters){TORCH_CHECK(m.grad().defined()&&at::isfinite(m.grad()).all().item<bool>(),"Nonfinite image gradient");norms.push_back(at::linalg_vector_norm(m.grad(),2.));}
  auto magnitude=at::linalg_vector_norm(at::stack(norms),2.);TORCH_CHECK(at::isfinite(magnitude).item<bool>(),"Nonfinite gradient norm");auto coefficient=at::clamp_max(1./(magnitude+1e-6),1.);
  torch::NoGradGuard guard;++e.step;
  for(size_t i=0;i<3;i++){auto& m=e.masters[i];m.grad().mul_(coefficient);e.avg[i].lerp_(m.grad(),1-.9);e.sq[i].mul_(.999).addcmul_(m.grad(),m.grad(),1-.999);auto denom=(e.sq[i].sqrt()/std::sqrt(1-std::pow(.999,double(e.step)))).add_(1e-6);m.addcdiv_(e.avg[i],denom,-lr/(1-std::pow(.9,double(e.step))));TORCH_CHECK(at::isfinite(m).all().item<bool>(),"Nonfinite updated image master");}
  *loss_out=loss.item<double>();
});}
int luban_image_greedy(void* p,SharedTensor* input,int64_t prefix,int64_t* tokens,int64_t* count,int(*cancel)()){return checked([&]{
  torch::jit::GraphOptimizerEnabledGuard off(false);
  auto& e=*static_cast<ImageTrainer*>(p);auto value=inputs(input);TORCH_CHECK(prefix>=1&&prefix<e.maximum&&prefix<=value[4].size(1),"Answer-free prefix required");
  value[4]=value[4].index({Slice(),Slice(0,prefix)});value[5]=value[5].index({Slice(),Slice(),Slice(0,prefix)});*count=0;torch::NoGradGuard guard;
  while(value[4].size(1)<e.maximum){if(cancel&&cancel())break;auto n=value[4].size(1);value[6]=at::full({1,1,n,n},-INFINITY,at::kBFloat16).triu(1);auto next=e.forward(value)[0][-1].argmax().item<int64_t>();tokens[(*count)++]=next;if(next==248046)break;value[4]=at::cat({value[4],at::tensor({next},at::kLong).reshape({1,1})},1);value[5]=at::cat({value[5],value[5].index({Slice(),Slice(),Slice(-1)})+1},-1);}
});}
int luban_image_save(void* p,const char* path,const char* identity){return checked([&]{
  auto& e=*static_cast<ImageTrainer*>(p);TORCH_CHECK(e.masters.size()==3&&strlen(identity)==64,"Bound checkpoint identity required");torch::serialize::OutputArchive out;out.write("step",at::tensor(e.step,at::kLong).reshape({}));out.write("identity",at::from_blob((void*)identity,{64},at::kByte));
  for(size_t i=0;i<3;i++){out.write("master"+std::to_string(i),e.masters[i]);out.write("avg"+std::to_string(i),e.avg[i]);out.write("sq"+std::to_string(i),e.sq[i]);}out.save_to(path);
});}
int luban_image_restore(void* p,const char* path,const char* identity,int64_t expected){return checked([&]{
  auto& e=*static_cast<ImageTrainer*>(p);TORCH_CHECK(e.masters.size()==3&&expected>=0,"Complete restore target required");torch::serialize::InputArchive in;in.load_from(path,at::kCPU);auto keys=in.keys();TORCH_CHECK(keys.size()==11,"Exact checkpoint schema required");Tensor id,step;in.read("identity",id);in.read("step",step);TORCH_CHECK(id.scalar_type()==at::kByte&&id.numel()==64&&strlen(identity)==64&&memcmp(id.data_ptr(),identity,64)==0&&step.scalar_type()==at::kLong&&step.dim()==0&&step.item<int64_t>()==expected,"Checkpoint identity/step mismatch");
  std::vector<Tensor> restored;for(size_t i=0;i<3;i++)for(auto name:{"master","avg","sq"}){Tensor t;in.read(std::string(name)+std::to_string(i),t);TORCH_CHECK(t.scalar_type()==at::kFloat&&t.sizes()==e.masters[i].sizes()&&at::isfinite(t).all().item<bool>()&&(std::string(name)!="sq"||t.ge(0).all().item<bool>()),"Invalid optimizer checkpoint");restored.push_back(t);}
  torch::NoGradGuard guard;for(size_t i=0;i<3;i++){e.masters[i].copy_(restored[i*3]);e.avg[i].copy_(restored[i*3+1]);e.sq[i].copy_(restored[i*3+2]);}e.step=expected;
});}
}
