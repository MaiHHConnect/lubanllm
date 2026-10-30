/** Issued independent evidence; runtime data cannot add qualifications. */
import type {NativeEvolutionGraphReport} from './native-evolution-graph-qualification.ts';
export const VERIFIED_NATIVE_EVOLUTION_GRAPH_REPORTS:readonly NativeEvolutionGraphReport[]=Object.freeze([
  {
    "protocol": "native-evolution-independent-graph-v2",
    "identity": "673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df",
    "release_manifest_sha256": "14fbc4064cb1d3e142c1a46bcbff943383e8fc09fe6a6701dd7768d939ede47d",
    "graph_binary_sha256": "f7f5ae18a8b8a21c3522799dbc63b576d6eecccfe0c9f2c541eedcf292958223",
    "training_binary_sha256": "e83534c1de0c97f1da4fbb3f8fe340eaa351b2880a4ecac7d8c70034dcb7c93a",
    "graph_source_proof_sha256": "3a1555706df0c6f24f29316339513e7827858145baa3693b0eb568606e50bbd0",
    "training_source_proof_sha256": "d9f742586551ace02e12cf67de2defaa898d4722dead8c16f68edad6c80e8ae9",
    "inherited_math": {
      "evidence_id": "native-evolution-product-20261017",
      "graph_root_result_id": "math-real-20261017-v1/original-HF",
      "training_root_result_id": "math-real-20261017-v1/native-result",
      "full_pairs": [
        {
          "name": "accepted-down",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r000-x",
          "elements": 258048,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r000-residual",
          "elements": 86016,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r000-ids",
          "elements": 42,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r000-labels",
          "elements": 42,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r012-x",
          "elements": 159744,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r012-residual",
          "elements": 53248,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r012-ids",
          "elements": 26,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "prefix-r012-labels",
          "elements": 26,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-ce_logits",
          "elements": 993280,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-teacher_log_probs",
          "elements": 993280,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-student_log_probs",
          "elements": 993280,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-ce_grad",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-kl_grad",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-total_grad",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-clipped_grad",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-master",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-candidate",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-exp_avg",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-exp_avg_sq",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-adam_step",
          "elements": 1,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "update-delta",
          "elements": 12582912,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "final-cpu-rng",
          "elements": 5056,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        },
        {
          "name": "final-mps-rng",
          "elements": 44,
          "max_abs_error": 0,
          "allowed_max_abs_error": 0
        }
      ],
      "actual_original_HF": true,
      "actual_full_graph_backwards": true,
      "CE_KL_Adam_projection_steps": 1,
      "domains": [
        "forward",
        "CE",
        "KL",
        "gradient",
        "Adam",
        "projection"
      ],
      "unchanged_operator_sources": {
        "Training.cpp": {
          "previous": "dbec68f69884b01999c3dfca52a49d3068cd99cd3755771263c479a61bc5ca22",
          "current": "dbec68f69884b01999c3dfca52a49d3068cd99cd3755771263c479a61bc5ca22"
        },
        "Chunk.h": {
          "previous": "8718ba899900f12cc2ee661220c7930190a6245a204afe713ea6a613fc8ca09a",
          "current": "8718ba899900f12cc2ee661220c7930190a6245a204afe713ea6a613fc8ca09a"
        },
        "Training.v3.cpp/original-operator-body-before-terminal-overlay": {
          "previous": "d5936b2c4ccc022ef32c06ca3e885ea1d5271131cdbd59b622dd38fb4a6a532f",
          "current": "d5936b2c4ccc022ef32c06ca3e885ea1d5271131cdbd59b622dd38fb4a6a532f"
        }
      }
    },
    "observation": {
      "small": {
        "result_id": "ordinary-observation-cpu-UB0vKQ",
        "actual_execution": "small-native",
        "exit_code": 0,
        "cleanup_confirmed": true,
        "pairs": [
          {
            "name": "ordinary-versus-audit-CE",
            "elements": 1,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "resident-previous-repeat-CE",
            "elements": 1,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "resident-current-repeat-CE",
            "elements": 1,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          }
        ]
      },
      "actual": {
        "result_id": "math-real-20261017-v1/selection-round-source-v1+selection-exfat-effective-none-v3/original-HF-result",
        "actual_execution": "original-HF-real-model",
        "exit_code": 0,
        "cleanup_confirmed": true,
        "pairs": [
          {
            "name": "previous-head",
            "elements": 248320,
            "max_abs_error": 0.03125,
            "allclose_passed": true,
            "atol": 0.0009765625,
            "rtol": 0.0078125,
            "relative_l2": 0.00003329080837500219,
            "relative_l2_limit": 0.0078125
          },
          {
            "name": "previous-CE",
            "elements": 1,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "current-head",
            "elements": 248320,
            "max_abs_error": 0.0625,
            "allclose_passed": true,
            "atol": 0.0009765625,
            "rtol": 0.0078125,
            "relative_l2": 0.00006614541301210525,
            "relative_l2_limit": 0.0078125
          },
          {
            "name": "current-CE",
            "elements": 1,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "previous-head-v3-exact",
            "elements": 248320,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "current-head-v3-exact",
            "elements": 248320,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "round-restored-head-exact",
            "elements": 248320,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          }
        ]
      }
    },
    "tail": {
      "small": {
        "result_id": "ordinary-observation-cpu-UB0vKQ/tail",
        "actual_execution": "small-native",
        "exit_code": 0,
        "cleanup_confirmed": true,
        "pairs": [
          {
            "name": "tail_x",
            "elements": 160,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "tail_residual",
            "elements": 80,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          }
        ]
      },
      "actual": {
        "result_id": "math-real-20261017-v1/original-HF+selection-round-source-v1/tail",
        "actual_execution": "original-HF-real-model",
        "exit_code": 0,
        "cleanup_confirmed": true,
        "pairs": [
          {
            "name": "r000-tail_x",
            "elements": 258048,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r000-tail_residual",
            "elements": 86016,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r000-ids",
            "elements": 42,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r000-labels",
            "elements": 42,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r012-tail_x",
            "elements": 159744,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r012-tail_residual",
            "elements": 53248,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r012-ids",
            "elements": 26,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "r012-labels",
            "elements": 26,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          }
        ]
      }
    },
    "terminal_conversion": {
      "method": "gptq-bf16-refine-v1",
      "evidence_id": "terminal-conversion-independent-v1+native-refine-restore-v1",
      "training_binary_sha256": "e83534c1de0c97f1da4fbb3f8fe340eaa351b2880a4ecac7d8c70034dcb7c93a",
      "training_source_proof_sha256": "d9f742586551ace02e12cf67de2defaa898d4722dead8c16f68edad6c80e8ae9",
      "graph_binary_sha256": "f7f5ae18a8b8a21c3522799dbc63b576d6eecccfe0c9f2c541eedcf292958223",
      "shape": [
        2048,
        6144
      ],
      "passes": 2,
      "optimizer_steps_in_conversion": 0,
      "calibration": "known-old-and-actual-q4-replay-training",
      "new_validation_used": false,
      "independent_reference": {
        "result_id": "/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/identity-native450-v19/terminal-conversion-independent-v1/result.json",
        "execution": "real-shape-independent-python-reference",
        "candidate_tensor_sha256": "ae14867f1aaba5d0b6d8fd49abb551759c063184af53596bd29f574f99f0d6fb",
        "pairs": [
          {
            "name": "converted-candidate",
            "elements": 12582912,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          },
          {
            "name": "converted-master",
            "elements": 12582912,
            "max_abs_error": 0,
            "allowed_max_abs_error": 0
          }
        ]
      },
      "boundary": {
        "result_id": "build/native-evolution-conversion/cpu-proof.json",
        "step449_unmodified": true,
        "step450_applied_once": true,
        "seven_state_resume_equal": true,
        "reference_training_basis_preserved": true,
        "cancellation_before_commit": true,
        "parent_rng_restored": true
      },
      "actual_restore": {
        "result_id": "/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/identity-native450-v19/gptq-bf16-refine-retention-v1/result.json",
        "actual_execution": "native-real-model",
        "graph_binary_sha256": "f7f5ae18a8b8a21c3522799dbc63b576d6eecccfe0c9f2c541eedcf292958223",
        "candidate_tensor_sha256": "ae14867f1aaba5d0b6d8fd49abb551759c063184af53596bd29f574f99f0d6fb",
        "restored_tensor_sha256": "1706e50833fffb25da45ad958af043a688e5ef495d45f33fa4b1ab623c696719",
        "pid": 46049,
        "engine_closed": true,
        "lease_released": true,
        "product_pointers_unchanged": true
      }
    }
  }
]);
