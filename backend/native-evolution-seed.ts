/** Original post-baseline seed42 boundary; only checkpoint RNG is exported, no seven-state audit. */
import assert from 'node:assert/strict';
import {NativeEngine,assertNativeExpertScopeGuard} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {actualCoreInstallation} from '../.native-migration/research/native-unified-core-control/install.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {validateCore450Fresh,type Core450Fresh} from '../.native-migration/research/native-core450-runner/fresh.ts';
import {NativeCore450Owner} from '../.native-migration/research/native-core450-owner-controller/owner.ts';
import {captureNativeEvolutionTrainingRNG} from './native-evolution-ambient.ts';
type Guard=Parameters<Parameters<NativeEngine['withSourceQuiescent']>[0]>[0];
const seeded=new WeakSet<Core450Fresh>();
export async function seedNativeEvolutionFresh(engine:NativeEngine,guard:Guard,cap:Core450Fresh,owners:NativeCore450Owner){
 assertNativeExpertScopeGuard(engine,guard);validateCore450Fresh(engine,cap);assert(cap.approval.step===0&&!seeded.has(cap),'Fresh seed must occur once and never on resume');
 const installed=actualCoreInstallation(engine,guard,cap),peer=borrowNativeEnginePeer(engine,guard);assert(peer.worker.pid===installed.receipt.pid&&!peer.worker.isBranchEnabled);await cap.assertUnchanged();seeded.add(cap);
 await peer.worker.seedFresh();const rng=await captureNativeEvolutionTrainingRNG(engine,guard,owners);
 return{rng,receipt:Object.freeze({pid:peer.worker.pid,step:0,seed:42,cpu_rng_sha256:rng.cpu_rng.sha256,mps_rng_sha256:rng.mps_rng.sha256,state_sha256:installed.receipt.state_sha256,product_admitted:false})};
}
