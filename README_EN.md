# Luban Zero

**Application: public beta · Technical report: Research Preview**

[Website](https://ai.maihh.net/?lang=en) · [中文](README.md) · [Beta status](STATUS.md)

Luban Zero (luban0) is a continual-learning AI prototype for personal devices, starting from an approximately 2B language core. It explores local inference, routing, learning from authorized material, and forgetting control.

## Research Paper — Luban0

**Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts**

DOI: [10.5281/zenodo.23552845](https://doi.org/10.5281/zenodo.23552845)

The DOI identifies the archived preprint; repository files include subsequent experimental updates.

Status: **Technical Report / Preprint (not peer reviewed)**.

DOI: [10.5281/zenodo.23552845](https://doi.org/10.5281/zenodo.23552845)

The DOI identifies the archived preprint; repository files include subsequent experimental updates.

Status: **Technical Report / Preprint (not peer reviewed)**.

[📄 Read Paper](paper/Luban_Technical_Report.md) · [PDF](paper/Luban_Technical_Report.pdf) · [Chinese supplement](paper/Luban_Technical_Report_ZH.md) · [📊 Experimental Results](docs/EXPERIMENTS.md)

Report v0.1 (22 October 2026) describes the current MLX 4-bit architecture, expert learning, route-aware evaluation, net-gain admission, observed failures, and limitations. It is a **Technical Report — Research Preview / Work in Progress**, not submitted or peer reviewed. Author: vinson. Affiliation: 麦好火电商研究. Final manuscript review remains pending; no independent peer review is claimed. The report and sanitized summaries use [CC BY-NC-ND 4.0](LICENSE-REPORT.txt).

Current daily learning freezes the base and accepted core while training an expert candidate. The Evolution action uses this same expert-learning path. Adoption requires measured validation gains to exceed sampled forgetting, plus quality and persistence checks. Automatic specialization into many independent experts and sustained long-term improvement remain research directions.

A small study improved development-validation answers from 2/6 to 6/6 for two identity facts. Its historical protection pool was empty, so it does not demonstrate broad retention or generalization. The later FAQ study paused at18/20 actual rounds,2800 Adam updates,4 commits and14 quality rejections because no uncovered eligible independent validation records remained. The208/218/private219 phases are reported separately; this is neither20 completed rounds nor18 novel abilities. Read the [experiment summary](docs/EXPERIMENTS.md) and [reproduction protocol](docs/REPRODUCIBILITY.md) for conditions, instability and missing evaluation.

On 23 October, a separate 20/20-repetition isolated inference/lifecycle test passed: all 2,960 checked observations agreed; 80 actual updates included 20 checkpoint replays, with no adoption or teacher calls. This does not complete the earlier 18/20 learning study or establish new abilities or speedup. See the [experiment summary](docs/EXPERIMENTS.md).

## Product directions

- Local customer service and enterprise knowledge questions
- Adaptation to personal knowledge and work habits
- Simple automation tasks
- Studying continual learning and expert growth within finite resources

These are directions rather than completed capability promises. See [the current architecture](docs/ARCHITECTURE.md).

## Public scope

This repository publishes project information, the technical report, and sanitized aggregate results. Source code, personal data, complete raw research evidence, and internal workflow documents remain restricted. Publication grants no new source-code or model license; upstream components retain their own licenses.

The complete package includes the local base, metadata, speech and factory expert; the application-only GitHub asset is a separate advanced option. The complete installer is about 3.62 GB; keep 16 GB free before downloading. Application 0.3.218 public beta: [Complete installer (models included)](https://ai.maihh.net/downloads/Luban-0.3.218-macos-arm64-full.zip) · [Download the macOS beta](https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.218-beta.1) · [Quark Drive](https://pan.quark.cn/s/0c64deea0cd9). Requires macOS 26 or later and Apple Silicon; not notarized by Apple.
