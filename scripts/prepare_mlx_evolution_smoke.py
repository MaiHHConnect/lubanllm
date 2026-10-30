#!/usr/bin/env python3
"""User-requested finite app evolution test; authorized product facts only.

TRAIN and VALID differ in wording. This is a limited paraphrase check, not a
claim of family-wide generalization. Proposals are screened for route coverage
before answers are generated; validation never enters gradient training.
"""
import argparse
import json
from pathlib import Path

IDENTITY='673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df'
FACTS={'name':'luban0','owner':'麦好火'}
PROMPTS={
'name':{
 'training':['请核对你的模型身份。只写模型名。','请说明你的模型身份。只写模型名。','介绍一下你的模型身份，只写模型名。','请确认你的模型身份，只写模型名。','请说出自己的模型身份，只写模型名。','你的模型身份是什么？只写模型名。','请给出你的官方模型身份，只写模型名。','请提供你的模型身份，只写模型名。','请说明你的模型名称，只写名称。','请核对你的模型名称，只写名称。','请确认你的模型名称，只写名称。','请介绍你的模型名称，只写名称。','请说出你的模型名称，只写名称。','请提供你的模型名称，只写名称。','请给出你的模型名称，只写名称。','请说明你是什么模型，只写名称。'],
 'validation':['模型身份核对：仅提交模型名称，不要自我介绍。','为模型档案填写名称字段：你叫什么？只填字段值。','把你的官方模型身份缩写为名称，其他内容省略。','正在对话的这个模型叫什么？只给名称即可。','核实模型身份时，名称这一栏应该填什么？仅回答这一栏。','请用模型名回答身份核验，省略品牌与基座。']},
'owner':{
 'training':['请核对你的品牌归属。只写品牌名。','请说明你的品牌归属。只写品牌名。','介绍一下你的品牌归属，只写品牌名。','请确认你的品牌归属，只写品牌名。','请说出自己的品牌归属，只写品牌名。','你的品牌归属是什么？只写品牌名。','请给出你的官方品牌归属，只写品牌名。','请提供你的品牌归属，只写品牌名。','请说明你的模型所属品牌，只写品牌名。','请核对你的模型所属品牌，只写品牌名。','请确认你的模型所属品牌，只写品牌名。','请介绍你的模型所属品牌，只写品牌名。','请说出你的模型所属品牌，只写品牌名。','请提供你的模型所属品牌，只写品牌名。','请给出你的模型所属品牌，只写品牌名。','请说明你属于哪个品牌，只写品牌名。'],
 'validation':['模型身份核对：仅提交所属品牌，不要自我介绍。','为模型档案填写品牌字段：你属于哪一方？只填字段值。','把你的官方模型身份缩写为所属品牌，其他内容省略。','正在对话的这个模型属于哪个品牌？只给品牌名即可。','核实模型身份时，所属品牌这一栏应该填什么？仅回答这一栏。','请用品牌名回答归属核验，省略型号与基座。']}}


def proposals():
    rows=[]
    for fact,splits in PROMPTS.items():
        for purpose,prompts in splits.items():
            for i,prompt in enumerate(prompts):
                answer=FACTS[fact]
                rows.append({'id':f'evolution-smoke-20261021-{fact}-{purpose}-{i}',
                    'prompt':prompt,'answer':answer,'source':'human','purpose':purpose,'approved':True,'training_allowed':True,
                    'provenance':{'base_identity':IDENTITY,'verified_by':'local requester','evolution_role':'new',
                        'split_family':f'20261021-field-{fact}-{purpose}',
                        'question_authorship':'assistant','direct_human_quote':False,
                        'authorization_reference':'User requested sufficient evolution test questions on 2026-10-21; established product facts luban0 and 麦好火',
                        'verification_evidence':'Exact field from user-authorized product identity; not a teacher response',
                        'test_batch':'mlx-evolution-smoke-20261021','fact':fact,
                        'messages':[{'role':'user','content':prompt}],
                        'generation_check':{'type':'exact','expected':answer,'max_new_tokens':64}}})
    assert len({r['prompt'] for r in rows})==len(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--routes',type=Path)
    args=parser.parse_args();rows=proposals()
    if args.routes:
        route=json.loads(args.routes.read_text());assert route['released'] and route['identity']==IDENTITY
        covered={r['id'] for r in route['rows'] if r['action']=='neural_expert'}
        selected=[]
        for fact in FACTS:
            candidates=[r for r in rows if r['provenance']['fact']==fact and r['purpose']=='training' and r['id'] in covered]
            assert len(candidates)>=8,(fact,'insufficient actual expert training coverage')
            selected+=candidates[:8]
        validation=[r for r in rows if r['purpose']=='validation' and r['id'] in covered]
        assert len(validation)>=4 and {r['provenance']['fact'] for r in validation[:4]}==set(FACTS)
        selected+=validation[:4]
        rows=selected
    assert not args.output.exists()
    args.output.write_text(json.dumps({'rows':rows,'teacher_calls':0,'answers_used_for_route_selection':False},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'training':sum(r['purpose']=='training' for r in rows),'validation':sum(r['purpose']=='validation' for r in rows)},ensure_ascii=False))


if __name__=='__main__':main()
