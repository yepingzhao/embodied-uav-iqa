# Documentation

## Code and architecture

| Document | Scope |
|---|---|
| [Codemap index](CODEMAPS/INDEX.md) | Package navigation and entry points |
| [Architecture](CODEMAPS/ARCHITECTURE.md) | Data flow and package boundaries |
| [File map](CODEMAPS/FILES.md) | Source files, scripts, and tests |
| [Module APIs](CODEMAPS/MODULES.md) | Module responsibilities and interfaces |
| [Offline inference framework](CODEMAPS/INFERENCE_FRAMEWORK.md) | Multi-GPU scheduling, checkpointing, and output contracts |

Development commands are in [AGENTS.md](../AGENTS.md). Training configurations
are in [configs/experiments/](../configs/experiments/).

## Literature notes

These notes summarize prior work. Their reported results belong to the cited
papers and do not establish results for this repository.

| Note | Source |
|---|---|
| [AirCopBench](<literature/AirCopBench A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning.md>) | arXiv:2511.11025 |
| [Embodied-IQA](<literature/Image Quality Assessment for Embodied AI.md>) | arXiv:2505.16815 |
| [EPD / MA-EIQA](<literature/Embodied Image Quality Assessment for Robotic Intelligence.md>) | arXiv:2412.18774 |
| [UAV embodied IQA literature review](<literature/低空无人机具身智能的图像质量评估-文献综述.md>) | Literature synthesis dated 2026-06-18 |

The imported notes retain their original titles and Obsidian references.
Some references and attachments belong to the original note collection and
are not included in this repository.

## Research analysis

- [Multi-view IQA baseline analysis](research/multi-view-iqa-baseline-report.md)

## Downloaded paper sources

| Directory | Contents |
|---|---|
| [arXiv-AirCopBench/](arXiv-AirCopBench/) | AirCopBench LaTeX, bibliography, and figures |
| [arXiv-CARLA-Air/](arXiv-CARLA-Air/) | CARLA-Air LaTeX and bibliography |
| [arXiv-Embodied-IQA/](arXiv-Embodied-IQA/) | Embodied-IQA LaTeX and bibliography |
