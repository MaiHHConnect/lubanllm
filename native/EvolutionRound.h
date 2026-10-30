#pragma once
// Python's fork_rng boundary, held in memory in the one resident worker.
struct EvolutionRoundRNG {
 Tensor cpu,mps;torch::Device device; bool begun=false,active=false;
 explicit EvolutionRoundRNG(torch::Device value=torch::kMPS):device(value){}
 void begin(){need(!active,"Evolution round already owns RNG");cpu=captureNativeCPURNG();if(device.is_mps())mps=captureNativeMPSRNG();begun=active=true;}
 void restore(){need(begun,"Evolution round RNG owner absent");applyNativeCPURNG(cpu);if(device.is_mps())applyNativeMPSRNG(mps);need(cpu.equal(captureNativeCPURNG())&&(!device.is_mps()||mps.equal(captureNativeMPSRNG())),"Evolution RNG restore failed");active=false;}
 void checkpoint(const fs::path& path){need(active,"Checkpoint RNG requires active evolution round");auto out=fresh(path);auto c=captureNativeCPURNG();durable(out/"cpu_rng.raw",c.data_ptr(),c.nbytes());if(device.is_mps()){auto m=captureNativeMPSRNG();durable(out/"mps_rng.raw",m.data_ptr(),m.nbytes());}syncDir(out);}
};
