"""Bind a recovered final down to unchanged Q4 records; screen native generation."""
import os
from pathlib import Path
import re
import time

import numpy as np
from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from capture_q4_paired_tails import BINARY, OUTPUT as CAPTURE, collect, read_frame_file
from diagnose_q4_task_candidate import diagnostic_result, ordinary_loaded, read
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import protection_generations
from q4_packed_format import PAYLOAD_BYTES, encode_header, projections, write_sparse
from q4_sequential_protection import check_proposal, protection_cases
from study_q4_column_protection import ROOT, SOURCE, MASK, frozen_scope
from validate_q4_paired_tail_proxy import ERROR_COLUMNS, LOGIT_COLUMNS, errors, logit_summary
from verify_q4_sensitive_source import CONTROLS

TRAINING = ROOT/'paired-q4-tail-recovery-v2'
OUTPUT = TRAINING/'screen'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_paired_tail_recovery_20261017.json'


def sha_text(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def local_artifact(value, directory):
    path = Path(value)
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or not path.is_relative_to(directory) or not path.is_file()):
        raise ValueError('regular artifact inside frozen training output required')
    return path


def candidate_contract(candidate, source_receipt, directory=TRAINING):
    if (candidate.get('schema') != 'q4-paired-tail-recovery-v1'
            or any(candidate.get(k) is not True for k in ('completed','trainingcompleted','originalbase'))
            or candidate.get('source_receipt') != source_receipt
            or not sha_text(candidate.get('core_file_sha256')) or candidate['core_file_sha256'] == '0'*64):
        raise ValueError('completed original-base recovery with exact source/core binding required')
    core = local_artifact(candidate['core_path'], directory)
    if not 8 < core.stat().st_size <= 32 << 20:
        raise ValueError('bounded native core safetensors required')
    proxy = candidate.get('final_proxy')
    if not isinstance(proxy, list) or [row.get('step') for row in proxy] != [0,1]:
        raise ValueError('exact final cal00 step0/1 proxy outputs required')
    for row in proxy:
        for kind in ('down','logits'):
            local_artifact(row[kind+'_path'], directory)
    return core


def source_payloads(path, manifest, mask):
    expected = projections(mask)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != 148+len(expected)*(8+PAYLOAD_BYTES):
        raise ValueError('complete original sparse carrier required')
    with path.open('rb') as stream:
        if stream.read(148) != encode_header(manifest, '0'*64, mask):
            raise ValueError('original coreNone sparse header required')
        for layer,kind,_,_ in expected:
            raw = stream.read(8+PAYLOAD_BYTES)
            if len(raw) != 8+PAYLOAD_BYTES:
                raise ValueError('truncated original projection')
            yield layer,kind,raw
        if stream.read(1):
            raise ValueError('trailing original payload')


def rebind_carrier(source, destination, core_sha, manifest=MANIFEST, mask=MASK):
    if not sha_text(core_sha) or core_sha == '0'*64 or source.resolve() == destination.resolve():
        raise ValueError('new core binding and separate carrier required')
    result = write_sparse(destination, manifest, core_sha, mask, source_payloads(source,manifest,mask))
    return {**result, 'source':str(source), 'core_file_sha256':core_sha,
            'format':'LQ4PK02', 'original_record_bytes_preserved':True,
            'scope':'same source record buffers streamed unchanged through existing validated writer; no carrier hash'}


def candidate_loaded(loaded, core_sha, expected_base, tensor_sha=None):
    if (loaded.get('base_identity') != IDENTITY or loaded.get('manifest_sha256') != MANIFEST
            or loaded.get('inference_precision') != 'mixed-q4-g16-packed'
            or loaded.get('core_loaded') is not True or loaded.get('core_file_sha256') != core_sha
            or not sha_text(loaded.get('core_tensor_sha256'))
            or (tensor_sha is not None and loaded.get('core_tensor_sha256') != tensor_sha)
            or loaded.get('retained_original_weight_bytes') != 0
            or tuple(loaded.get(k) for k in ('packed_version','residual_rank','active_residual_projections','quantized_projections')) != (2,0,0,32)
            or any(loaded.get(k) is not False for k in ('research_sequential','research_capture_inputs','research_intervention'))
            or any(loaded.get(k) != expected_base.get(k) for k in ('checkpoint_sha256','original_down_sha256'))):
        raise ValueError('actual recovered core with unchanged ordinary packed32 graph required')


def unchanged_upstream(actual, original):
    if (actual.shape != (10240,) or original.shape != (10240,)
            or not np.isfinite(actual).all() or not np.isfinite(original).all()
            or not np.array_equal(actual[:8192].view(np.uint32), original[:8192].view(np.uint32))):
        raise ValueError('final down update changed native upstream h/act')


def preflight():
    candidate = read(TRAINING/'result.json'); source = read(SOURCE/'result.json'); frozen_scope(source)
    training_audit = read(TRAINING/'independent-readback.json')
    if training_audit.get('audit_passed') is not True or training_audit.get('audit_completed') is not True or training_audit.get('directory') != str(TRAINING):
        raise ValueError('independently checked final recovery required')
    core = candidate_contract(candidate,source['receipt'])
    registration = read(REGISTRATION)
    controls, control_audit = read(CONTROLS/'result.json'), read(CONTROLS/'independent-readback.json')
    capture, audit = read(CAPTURE/'result.json'), read(CAPTURE/'independent-readback.json')
    receipt = read(Path(str(BINARY)+'.json'))
    if (registration.get('registered_before_run') is not True
            or registration.get('native_followup',{}).get('registered_before_run') is not True
            or controls.get('completed') is not True or controls.get('controls_passed') is not True
            or controls.get('source_receipt') != source['receipt']
            or control_audit.get('audit_passed') is not True or control_audit.get('old_bf16_four_exact') is not True
            or control_audit.get('new_v2_old20_exact') is not True
            or capture.get('completed') is not True or capture.get('receipt') != receipt
            or capture.get('binary') != str(BINARY) or capture.get('source_receipt') != source['receipt']
            or capture.get('source_plan') != source['plan'] or capture.get('core') is not None
            or audit.get('audit_completed') is not True or audit.get('audit_passed') is not True
            or audit.get('directory') != str(CAPTURE)):
        raise ValueError('registered recovery and audited original/native controls required')
    ordinary_loaded(controls['reports']['bf16']['loaded'],'bf16')
    specs = controls['cases']
    if len(specs) != 4 or {s['id'] for s in specs} != {
            'mmlu-2804','preservation-replay-train-olive-path','decode-holdout-unit-volume','seq-final-young-dog'}:
        raise ValueError('only the existing four exposed checks allowed')
    union = check_proposal(protection_cases(),controls['baseline'],controls['previous'],controls['previous'])
    if len(union['original_correct_ids']) != 13 or len(union['required_correct_ids']) != 14 or 'seq-train-copy-en' not in union['required_correct_ids']:
        raise ValueError('original13/prior14 correctness union changed')
    case = source['plan']['cases'][0]
    if case['id'] != 'cal-00' or case['split'] != 'calibration' or len(case['tokens']) != 26 or not case['forced_tokens']:
        raise ValueError('fixed original first calibration prefix required')
    return candidate, source, controls, capture, receipt, registration, core


def four_generations(worker, specs, tokenizer, label):
    rows = []
    for spec in specs:
        case, identifier = spec['case'],label+'-'+spec['id']
        raw = worker.request(case['tokens'],identifier,max_tokens=case['max_tokens'],eos_ids=case['eos_ids'],temperature=0,seed=42,reuse=False)
        if raw.get('event') != 'result' or raw.get('id') != identifier:
            raise ValueError('ordinary result identity changed')
        rows.append(diagnostic_result(spec,raw,tokenizer,248320))
        save(worker.out/'four.json',rows)
    return rows


def short_ordinary(worker, case, eos, identifier):
    raw = worker.request(case['tokens'],identifier,max_tokens=2,eos_ids=eos,temperature=0,seed=42,reuse=False)
    tokens = raw.get('tokens',[])
    if (not 1 <= len(tokens) <= 2 or raw.get('id') != identifier or raw.get('cached_prompt_tokens') != 0
            or raw.get('prompt_tokens') != len(case['tokens']) or raw.get('generated_tokens') != len(tokens)
            or any(t in eos for t in tokens[:-1]) or raw.get('complete') is not (tokens[-1] in eos)
            or raw.get('finish_reason') != ('stop' if tokens[-1] in eos else 'length')
            or (tokens[-1] not in eos and len(tokens) != 2)):
        raise ValueError('ordinary short control must retain natural stopping contract')
    return raw


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('mounted experiment volume required')
    candidate,source,controls,capture,receipt,registration,core = preflight()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    specs = controls['cases']
    for spec,recorded in zip(specs,controls['reports']['bf16']['four'],strict=True):
        checked = diagnostic_result(spec,recorded['native_result'],tokenizer,248320)
        if checked != recorded or checked['score']['passed'] is not True:
            raise ValueError('original BF16 four-case evidence changed')
    OUTPUT.mkdir(exist_ok=False)
    started = time.perf_counter(); lock = worker = None
    baseline = []; four = []; train = protection = reload = loaded = None
    protocol = {'schema':'q4-paired-recovery-native-screen-v1','training_directory':str(TRAINING),
        'training_result':candidate,'source':str(SOURCE),'source_receipt':source['receipt'],
        'binary':str(BINARY),'receipt':receipt,'core_path':str(core),'core_file_sha256':candidate['core_file_sha256'],
        'baseline_core':None,'baseline_reference':str(CONTROLS),'cases':specs,
        'seed':42,'temperature':0,'cache_reuse':False,'parameter_updates':0,
        'qualified':False,'product_adopted':False,'new_holdout_read':False,
        'full_regression_run':False,'final_holdout_run':False,'native_followup':registration['native_followup']}
    save(OUTPUT/'protocol.json',protocol)
    try:
        path = OUTPUT/'candidate.bin'
        artifact = rebind_carrier(SOURCE/'gptq-sequential.bin',path,candidate['core_file_sha256'])
        if artifact['quantized_projections'] != 32 or artifact['layer_mask'] != MASK:
            raise ValueError('fixed sparse32 payload scope changed')
        save(OUTPUT/'carrier.json',artifact)
        lock = ModelProcessLock(IDENTITY,'mps')
        baseline_dir = OUTPUT/'bf16'; baseline_dir.mkdir()
        worker = MixedWorker(BINARY,'bf16',baseline_dir,lock,precision='bf16',core=None)
        ordinary_loaded(worker.loaded,'bf16'); save(baseline_dir/'loaded.json',worker.loaded)
        baseline = four_generations(worker,specs,tokenizer,'baseline')
        for row,old in zip(baseline,controls['reports']['bf16']['four'],strict=True):
            if any(row[k] != old[k] for k in ('id','prompt_tokens','tokens','text','complete','score')):
                raise ValueError('new worker original BF16 baseline differs from exact old control')
        worker.close()
        if worker.process.returncode != 0:
            raise RuntimeError('original BF16 worker failed on close')
        worker = None
        directory = OUTPUT/'candidate'; (directory/'traces').mkdir(parents=True)
        worker = MixedWorker(BINARY,'candidate_mixed',directory,lock,precision='mixed-q4-g16-packed',
            calibration_profile=path,core=core,core_sha=candidate['core_file_sha256'],trace_directory=directory/'traces')
        loaded = worker.loaded
        candidate_loaded(loaded,candidate['core_file_sha256'],capture['reports']['q4']['loaded'],candidate.get('core_tensor_sha256'))
        save(directory/'loaded.json',loaded)
        case = source['plan']['cases'][0]; eos = source['plan']['eos_ids']
        short = {'tokens':case['tokens'],'forced_tokens':case['forced_tokens'][:1]}
        before = short_ordinary(worker,short,eos,'before-reload')
        metrics = []
        def inspect(step,frame):
            original = read_frame_file(CAPTURE/'q4/traces'/f'cal-00-step{step:04d}.tail.bf16',10240)
            unchanged_upstream(frame['tail'],original)
            proxy = candidate['final_proxy'][step]
            down = read_frame_file(proxy['down_path'],2048); logits = read_frame_file(proxy['logits_path'],248320)
            metrics.append({'step':step,'upstream_h_act_bitexact':True,
                'down':dict(zip(ERROR_COLUMNS,errors(frame['tail'][8192:],down).tolist(),strict=True)),
                'logits':dict(zip(LOGIT_COLUMNS,logit_summary(frame['logits'],logits).tolist(),strict=True)),
                'numerical_equivalence_claimed':False})
        trace = collect(worker,short,eos,'reload-cal00',inspect=inspect)
        after = short_ordinary(worker,short,eos,'after-reload')
        if any(before[k] != after[k] for k in ('tokens','complete','finish_reason')):
            raise ValueError('candidate ordinary output changed across trace')
        for step,event in enumerate(trace['events']):
            if len(before['tokens']) > step and before['tokens'][:step] == short['forced_tokens'][:step] and event['top1'] != before['tokens'][step]:
                raise ValueError('candidate trace differs from ordinary generation on same prefix')
        reload = {'before':before,'trace':trace,'after':after,'metrics':metrics,'ordinary_before_after_exact':True,
                  'proxy_comparison':'descriptive native-vs-proxy differences only; no tolerance gate or bitexact assumption'}
        save(OUTPUT/'reload.json',reload)
        four = four_generations(worker,specs,tokenizer,'four')
        passed = all(row['score']['passed'] for row in four)
        if passed:
            train = protection_generations(worker,protection_cases(),tokenizer,eos,'train',248320)
            protection = check_proposal(protection_cases(),controls['baseline'],controls['previous'],train)
        worker.close(); code = worker.process.returncode; worker = None
        if code != 0:
            raise RuntimeError('candidate worker failed on close')
        accepted = passed and protection is not None and protection['accepted'] is True
        save(OUTPUT/'result.json',{**protocol,'completed':True,'artifact':artifact,'candidate':str(path),
            'baseline_four':baseline,'loaded':loaded,'reload':reload,'four':four,'four_passed':passed,
            'train':train,'train_skipped':not passed,'protection':protection,'screen_passed':accepted,
            'full_regression_permitted':accepted,'full_regression_skipped':True,'final_holdout_skipped':True,
            'selected':{'carrier':str(path),'core':str(core),'core_file_sha256':candidate['core_file_sha256']} if accepted else None,
            'worker_exit_code':code,'elapsed_seconds':time.perf_counter()-started})
    except BaseException as error:
        save(OUTPUT/'failure.json',{'completed':False,'error':repr(error),'baseline_four':baseline,
            'loaded':loaded,'reload':reload,'four':four,'train':train,'protection':protection,
            'elapsed_seconds':time.perf_counter()-started})
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if lock is not None:
                lock.close()


if __name__ == '__main__':
    main()
