// SPDX-License-Identifier: Apache-2.0
// Resident Qwen3-ASR graph; shared with the validated native training loop.
#include "IO.h"
#include <cmath>
#ifdef LUBAN_ASR_TINY
#ifdef LUBAN_ASR_TINY_HD128
constexpr int M=8,C=2,A=8,AH=2,AL=2,AF=12,H=8,QH=2,KVH=1,HD=128,TL=2,TF=12,V=32,AID=31;
#elif defined(LUBAN_ASR_TINY_WIDE_HEAD)
constexpr int M=8,C=2,A=8,AH=2,AL=2,AF=12,H=8,QH=2,KVH=1,HD=8,TL=2,TF=12,V=32,AID=31;
#else
constexpr int M=128,C=2,A=8,AH=2,AL=2,AF=12,H=8,QH=2,KVH=1,HD=4,TL=2,TF=12,V=151936,AID=151676;
#endif
#else
constexpr int M=128,C=480,A=896,AH=14,AL=18,AF=3584,H=1024,QH=16,KVH=8,HD=128,TL=28,TF=3072,V=151936,AID=151676;
#endif
struct Graph {
 std::map<std::string,Tensor> tensors;fs::path out;std::string prefix;std::vector<Tensor> keys=std::vector<Tensor>(TL),values=std::vector<Tensor>(TL);int64_t seen=0;Tensor tailX,tailResidual;
 explicit Graph(const std::map<std::string,Tensor>& input):tensors(input){}
 Tensor w(const std::string& key){auto actual=key=="lm_head.weight"&&!tensors.count(key)?"model.language_model.embed_tokens.weight":key;need(tensors.count(actual),"Missing "+actual);return tensors.at(actual);}
 void put(const std::string&,const Tensor&){}
 Tensor linear(const Tensor& x,const std::string& p,bool bias=true){return at::linear(x,w(p+".weight"),bias?w(p+".bias"):Tensor{});}
 Tensor ln(const Tensor& x,const std::string& p){return at::layer_norm(x,{x.size(-1)},w(p+".weight"),w(p+".bias"),1e-5,true);}
 Tensor rms(const Tensor& x,const std::string& p){return w(p+".weight")*(x*at::rsqrt(x.pow(2).mean(-1,true)+1e-6));}
 Tensor audio(){
  auto x=w("input");auto mask=w("mask");need(x.dim()==3&&x.size(0)==1&&x.size(1)==M&&x.size(2)%100==0,"Mel padded length contract");need(mask.sizes()==at::IntArrayRef({1,x.size(2)})&&mask.eq(0).logical_or(mask.eq(1)).all().item<bool>(),"Binary mask");auto chunks=x.size(2)/100;
  auto lengths=mask.view({chunks,100}).sum(1).to(at::kLong);need(mask.sum().item<int64_t>()>0,"Nonempty audio");auto zeroStarted=false;for(int64_t i=0;i<mask.numel();i++){if(mask.flatten()[i].item<float>()==0)zeroStarted=true;else need(!zeroStarted,"Right padded mask required");}
  x=x.view({1,M,chunks,100}).permute({0,2,1,3}).reshape({chunks,1,M,100});
  for(int i=1;i<=3;i++){auto p="model.audio_tower.conv2d"+std::to_string(i);x=at::gelu(at::conv2d(x,w(p+".weight"),w(p+".bias"),{2,2},{1,1},{1,1},1),"none");put("conv"+std::to_string(i),x);}
  auto steps=x.size(3);x=linear(x.permute({0,3,1,2}).contiguous().view({chunks,steps,C*((((M+1)/2+1)/2+1)/2)}),"model.audio_tower.conv_out",false);
  auto inv=at::exp(-std::log(10000.)/(A/2-1)*at::arange(A/2,at::kFloat));auto time=at::arange(13,at::kFloat).unsqueeze(1)*inv.unsqueeze(0);auto pos=at::cat({time.sin(),time.cos()},1);x+=pos.slice(0,0,steps);put("convpos",x);
  for(int i=0;i<3;i++)lengths=at::where(lengths>0,at::floor_divide(lengths-1,2)+1,at::zeros_like(lengths));
  auto valid=at::arange(steps).unsqueeze(0)<lengths.unsqueeze(1);x=at::index_select(x.reshape({-1,A}),0,at::nonzero(valid.flatten()).squeeze(-1));put("packed",x);put("postlength",lengths.to(at::kFloat));
  // Original inference windows: max per-chunk post-CNN length × (800/100).
  auto window=lengths.max().item<int64_t>()*8;need(window>0,"Valid inference window");std::vector<int64_t> windows;for(int64_t n=x.size(0);n>0;n-=std::min(window,n))windows.push_back(std::min(window,n));
  for(int i=0;i<AL;i++){cancel();auto p="model.audio_tower.layers."+std::to_string(i);auto residual=x;auto h=ln(x,p+".self_attn_layer_norm");auto q=linear(h,p+".self_attn.q_proj").view({-1,AH,A/AH}).transpose(0,1).unsqueeze(0);auto k=linear(h,p+".self_attn.k_proj").view({-1,AH,A/AH}).transpose(0,1).unsqueeze(0);auto v=linear(h,p+".self_attn.v_proj").view({-1,AH,A/AH}).transpose(0,1).unsqueeze(0);
   auto qs=q.split_with_sizes(windows,2),ks=k.split_with_sizes(windows,2),vs=v.split_with_sizes(windows,2);std::vector<Tensor> outputs;for(size_t j=0;j<qs.size();j++)outputs.push_back(at::scaled_dot_product_attention(qs[j],ks[j],vs[j],{},0,false,1/std::sqrt(double(A/AH))).transpose(1,2).contiguous());
   x=residual+linear(at::cat(outputs,1).reshape({-1,A}),p+".self_attn.out_proj");residual=x;h=ln(x,p+".final_layer_norm");x=residual+linear(at::gelu(linear(h,p+".fc1"),"none"),p+".fc2");put("audio_layer"+std::to_string(i),x);
  }
  x=ln(x,"model.audio_tower.ln_post");put("audio",x);x=linear(at::gelu(linear(x,"model.multi_modal_projector.linear_1"),"none"),"model.multi_modal_projector.linear_2");put("projector",x);return x;
 }
 Tensor rotate(const Tensor& x){auto parts=x.chunk(2,-1);return at::cat({-parts[1],parts[0]},-1);}
 Tensor initial(){auto a=audio(),ids=w("ids").to(at::kLong);need(ids.dim()==2&&ids.size(0)==1&&ids.ge(0).all().item<bool>()&&ids.lt(V).all().item<bool>(),"ASR IDs");auto h=at::embedding(w("model.language_model.embed_tokens.weight"),ids);auto mask=ids.eq(AID).unsqueeze(-1).expand_as(h);need(mask.sum().item<int64_t>()==a.numel(),"Audio placeholder count mismatch");h=h.masked_scatter(mask,a);put("embeds",h);return h;}
 Tensor text(Tensor h){auto T=h.size(1);auto inv=1./at::pow(1000000.,at::arange(0,HD,2,at::kLong).to(at::kFloat)/HD);auto freqs=at::matmul(inv.view({1,HD/2,1}),at::arange(seen,seen+T,at::kFloat).view({1,1,T})).transpose(1,2);auto emb=at::cat({freqs,freqs},-1);auto cos=emb.cos().unsqueeze(1),sin=emb.sin().unsqueeze(1);
  for(int i=0;i<TL;i++){cancel();auto p="model.language_model.layers."+std::to_string(i);auto residual=h;auto x=rms(h,p+".input_layernorm");auto q=rms(linear(x,p+".self_attn.q_proj",false).view({1,T,QH,HD}),p+".self_attn.q_norm").transpose(1,2);auto k=rms(linear(x,p+".self_attn.k_proj",false).view({1,T,KVH,HD}),p+".self_attn.k_norm").transpose(1,2);auto v=linear(x,p+".self_attn.v_proj",false).view({1,T,KVH,HD}).transpose(1,2);q=q*cos+rotate(q)*sin;k=k*cos+rotate(k)*sin;if(keys[i].defined()){k=at::cat({keys[i],k},2);v=at::cat({values[i],v},2);}keys[i]=k;values[i]=v;put("key"+std::to_string(i),k);put("value"+std::to_string(i),v);
   x=at::scaled_dot_product_attention(q,k,v,{},0,T>1,std::pow(double(HD),-.5),QH/KVH>1).transpose(1,2).contiguous().reshape({1,T,QH*HD});h=residual+linear(x,p+".self_attn.o_proj",false);residual=h;x=rms(h,p+".post_attention_layernorm");auto mlpInput=at::silu(linear(x,p+".mlp.gate_proj",false))*linear(x,p+".mlp.up_proj",false);if(i==TL-1){tailX=mlpInput.detach();tailResidual=residual.detach();}h=residual+linear(mlpInput,p+".mlp.down_proj",false);put("text_layer"+std::to_string(i),h);
  }
  h=rms(h,"model.language_model.norm");put("text",h);auto logits=linear(h,"lm_head",false);put("logits",logits);seen+=T;return logits;
 }
 std::vector<int64_t> generate(int limit,int eos1,int eos2){need(limit>0&&limit<=512&&eos1>=0&&eos1<V&&eos2>=0&&eos2<V,"Bounded greedy controls");auto logits=text(initial());std::vector<int64_t> tokens;for(int step=0;step<limit;step++){cancel();need(at::isfinite(logits).all().item<bool>(),"Finite ASR logits");auto id=logits[0][-1].argmax().item<int64_t>();tokens.push_back(id);if(id==eos1||id==eos2||step+1==limit)break;prefix="step"+std::to_string(step+1)+"-";auto next=at::full({1,1},id,at::kLong);auto h=at::embedding(w("model.language_model.embed_tokens.weight"),next);put("embeds",h);logits=text(h);}return tokens;}
};
