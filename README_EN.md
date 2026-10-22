# Luban Zero

**Application: private beta · Technical report: Research Preview**

[Website](https://ai.maihh.net/?lang=en) · [中文](README.md) · [Beta status](STATUS.md)

Luban Zero (luban0) is a continual-learning AI prototype for personal devices, starting from an approximately 2B language core. It explores local inference, routing, learning from authorized material, and forgetting control.

## Research & Technical Report

**Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts**

[📄 Read Paper](paper/Luban_Technical_Report.md) · [PDF](paper/Luban_Technical_Report.pdf) · [Chinese supplement](paper/Luban_Technical_Report_ZH.md) · [📊 Experimental Results](docs/EXPERIMENTS.md)

Report v0.1 (22 October 2026) describes the current MLX 4-bit architecture, expert learning, route-aware evaluation, net-gain admission, observed failures, and limitations. It is a **Technical Report — Research Preview / Work in Progress**, not submitted or peer reviewed. Author: vinson. Affiliation: 麦好火电商研究. Final manuscript review remains pending; no independent peer review is claimed. The report and sanitized summaries use [CC BY 4.0](LICENSE-REPORT.txt).

Current daily learning freezes the base and accepted core while training an expert candidate. The Evolution action uses this same expert-learning path. Adoption requires measured validation gains to exceed sampled forgetting, plus quality and persistence checks. Automatic specialization into many independent experts and sustained long-term improvement remain research directions.

A small study improved development-validation answers from 2/6 to 6/6 for two identity facts. Its historical protection pool was empty, so it does not demonstrate broad retention or generalization. Read the [experiment summary](docs/EXPERIMENTS.md) for conditions and failures.

## Product directions

- Local customer service and enterprise knowledge questions
- Adaptation to personal knowledge and work habits
- Simple automation tasks
- Studying continual learning and expert growth within finite resources

These are directions rather than completed capability promises. See [the current architecture](docs/ARCHITECTURE.md).

## Public scope

This repository publishes project information, the technical report, and sanitized aggregate results. Source code, personal data, complete raw research evidence, and internal workflow documents remain restricted. Publication grants no new source-code or model license; upstream components retain their own licenses.

Publishing this report does not publish the application installer. Consult the [website](https://ai.maihh.net/?lang=en) and [Releases](https://github.com/MaiHHConnect/lubanllm/releases) for download availability.
