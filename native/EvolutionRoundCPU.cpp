#include "../.native-migration/research/native-qwen35-cache/Base.h"
#include "EvolutionRound.h"
int main(int argc,char**argv){try{
 need(argc==2,"CPU round output required");at::set_num_threads(1);at::set_num_interop_threads(1);torch::NoGradGuard noGrad;const auto root=fresh(argv[1]);
 EvolutionRoundRNG round(torch::kCPU);int rejects=0;auto reject=[&](const std::function<void()>& f){try{f();throw std::logic_error("Invalid round accepted");}catch(const std::logic_error&){throw;}catch(...){rejects++;}};
 reject([&]{round.restore();});reject([&]{round.checkpoint(root/"invalid");});
 seedNativeCPURNG(87);const auto before=captureNativeCPURNG();round.begin();reject([&]{round.begin();});need(fs::is_empty(root),"Begin wrote audit state");
 seedNativeCPURNG(42);auto expected=torch::randn({17});round.restore();need(before.equal(captureNativeCPURNG()),"Round RNG not restored");round.restore();need(before.equal(captureNativeCPURNG()),"Repeated cleanup changed RNG");need(fs::is_empty(root),"Restore wrote audit state");
 round.begin();seedNativeCPURNG(42);round.checkpoint(root/"checkpoint");auto actual=torch::randn({17});need(actual.equal(expected),"Checkpoint consumed RNG");
 std::set<std::string> files;for(const auto& entry:fs::directory_iterator(root/"checkpoint"))files.insert(entry.path().filename());need(files==std::set<std::string>{"cpu_rng.raw"},"Checkpoint RNG wrote extra state");round.restore();need(before.equal(captureNativeCPURNG())&&rejects==3,"Round negative or cleanup failed");
 durableText(root/"result.json","{\"passed\":true,\"CPU\":true,\"negative_controls\":3,\"restore_idempotent\":true,\"round_audit_files\":0,\"checkpoint_rng_files\":1}\n");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
