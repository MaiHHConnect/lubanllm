"""Finite synthetic continual-learning data, independent of model execution.

Facts recur across disjoint wording templates. TEST measures new wording for
seen facts, never transfer to unseen facts. No external source or teacher.
"""
import random
import string

DATASET_SEED = 20261024
TASK_IDS = ("A", "B", "C")
LABELS = ("Q", "R", "S", "T")
CONTEXTS = ("amber archive", "cobalt ledger", "violet register")
TEMPLATES = {
    "TRAIN": (
        "In the {context}, what is the mark of {entity}? give only the mark.",
        "For {entity}, give its assigned mark in the {context}. give only the mark.",
    ),
    "DEV": (
        "Which mark belongs to {entity} according to the {context}? give only the mark.",
    ),
    "TEST": (
        "Identify the mark recorded for {entity} in the {context}. give only the mark.",
        "Give the {context} mark associated with {entity}. give only the mark.",
    ),
}
SPLITS = {"TRAIN": "training", "DEV": "development", "TEST": "test"}


def _row(identifier, task, fact_id, split, prompt, answer, check):
    training = split == "training"
    return {
        "id": identifier, "task": task, "fact_id": fact_id, "split": split,
        "purpose": "training" if training else "validation",
        "prompt": prompt, "messages": [{"role": "user", "content": prompt}],
        "answer": answer, "check": check,
        "training_allowed": training,
        "provenance": {
            "source": "project-authored deterministic synthetic data",
            "dataset_seed": DATASET_SEED, "external_teacher": False,
            "training_authorization": "user-authorized local experiment" if training else "evaluation only",
        },
    }


def build_dataset():
    """Return fresh objects for the fixed, prespecified dataset seed."""
    rng = random.Random(DATASET_SEED)
    alphabet = "".join(letter for letter in string.ascii_uppercase if letter not in LABELS)
    entities = set()
    facts, groups = [], {key: [] for key in TEMPLATES}
    for task, context in zip(TASK_IDS, CONTEXTS):
        assigned = list(LABELS)
        rng.shuffle(assigned)
        for index, answer in enumerate(assigned):
            while True:
                entity = "".join(rng.sample(alphabet, 6))
                if entity not in entities:
                    entities.add(entity)
                    break
            fact_id = task + ":fact:" + str(index)
            facts.append({"id": fact_id, "task": task, "entity": entity, "answer": answer, "context": context})
            for group, templates in TEMPLATES.items():
                for template_index, template in enumerate(templates):
                    prompt = template.format(context=context, entity=entity)
                    groups[group].append(_row(
                        f"{fact_id}:{group.lower()}:{template_index}", task, fact_id,
                        SPLITS[group], prompt, answer,
                        {"type": "exact", "expected": answer, "max_new_tokens": 16},
                    ))
    protection_specs = (
        ("Compute 17 + 25. give only the integer.", "42", "integer", 42),
        ("Compute 63 - 28. give only the integer.", "35", "integer", 35),
        ("Compute 7 * 8. give only the integer.", "56", "integer", 56),
        ("Compute 81 / 9. give only the integer.", "9", "integer", 9),
        ("Convert 'cedar' to uppercase. give only the result.", "CEDAR", "exact", "CEDAR"),
        ("Convert 'violet' to uppercase. give only the result.", "VIOLET", "exact", "VIOLET"),
        ('Extract the value of key "value" from JSON {"value":73}. give only the JSON value.', "73", "json", 73),
        ('Extract the value of key "value" from JSON {"value":[2,5]}. give only the JSON value.', "[2,5]", "json", [2, 5]),
    )
    protection = []
    for index, (prompt, answer, validator, expected) in enumerate(protection_specs):
        check = {"type": "exact" if validator == "exact" else "semantic", "expected": expected,
                 "max_new_tokens": 32}
        if validator != "exact":
            check["validator"] = validator
        protection.append(_row(f"protection:{index}", "P", f"protection:{index}",
                               "protection", prompt, answer, check))
    return {
        "schema": "luban-continual-synthetic-data-v1", "seed": DATASET_SEED,
        "metadata": {"task_ids": list(TASK_IDS), "facts_per_task": 4,
                     "wording_templates_per_fact": {key: len(value) for key, value in TEMPLATES.items()}},
        "provenance": {"source": "project-authored deterministic synthetic data",
                       "external_teacher_calls": 0, "dataset_seed": DATASET_SEED},
        "license": {"origin": "project-authored synthetic data",
                    "authorization": "user authorized this local training experiment",
                    "external_license_claim": False},
        "scope": "new wording for seen synthetic facts; not unseen-fact transfer or a product learning round",
        "schedule_protocol": {
            "actual_adam_updates_per_stage": 64, "task_order": list(TASK_IDS),
            "current_only": "64 current rows; each current training row appears 8 times",
            "replay": "stage 1 identical; later 32 current plus 32 history; stage 3 history A/B 16 each",
            "budget_boundary": "same Adam count, unequal current exposure; not equal tokens or FLOPs",
            "stage_optimizer": "new stage worker inherits saved expert weights and resets Adam",
        },
        "facts": facts, "groups": groups, "protection": protection,
    }


def schedule(train_rows, stage, arm, seed):
    """Choose 64 training exposures without touching DEV/TEST or caller objects."""
    if type(stage) is not int or stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2 or 3")
    if arm not in ("current_only", "replay") or type(seed) is not int:
        raise ValueError("invalid arm or integer seed")
    if (not isinstance(train_rows, list) or not train_rows
            or any(not isinstance(row, dict) or row.get("split") != "training"
                   or row.get("purpose") != "training" or row.get("training_allowed") is not True
                   or row.get("task") not in TASK_IDS or not isinstance(row.get("id"), str)
                   for row in train_rows)):
        raise ValueError("schedule accepts authorized TRAIN rows only")
    if len({row["id"] for row in train_rows}) != len(train_rows):
        raise ValueError("duplicate training row IDs")
    by_task = {task: sorted([row for row in train_rows if row["task"] == task], key=lambda row: row["id"])
               for task in TASK_IDS}
    if any(len(rows) != 8 for rows in by_task.values()):
        raise ValueError("need exactly eight TRAIN rows for each task")

    def cycle(task, role, count):
        rows = list(by_task[task])
        random.Random(f"{seed}:{stage}:{role}:{task}").shuffle(rows)
        return [rows[index % len(rows)] for index in range(count)]

    current_task = TASK_IDS[stage - 1]
    if stage == 1 or arm == "current_only":
        return cycle(current_task, "current", 64)
    current = cycle(current_task, "current", 32)
    histories = [cycle(task, "history", 32 // (stage - 1)) for task in TASK_IDS[:stage - 1]]
    history = [rows[index] for index in range(len(histories[0])) for rows in histories]
    return [row for pair in zip(current, history) for row in pair]
