# Plan

## Pipeline Blocks

Represent the general building blocks to a pipeline. 

### General

These blocks are responsible for general tasks like:
- Planning
- Synthesis
- Critiques
Conventionally "textual" tasks will be left up to these tasks.

#### List of General Blocks

- [x] Plan generation block
- [ ] Critique block
- [ ] Improvement block
- [ ] Synthesis block

### Tool

These blocks are responsible for functional tasks and information gathering.

#### List of Tool Blocks

- [ ] Creative idea generation block
- [ ] Python excution block
- [ ] Web search block
- [ ] Wikipedia search block

### Routing

These blocks are responsible for routing the pipeline to different tools / tool blocks

#### List of Routing Blocks

- [ ] Tool routing block
- [ ] Response routing block (for longer responses)

## Pipeline Formation

The only requirements for pipeline formation are:

1. It starts with a planner block
2. It ends with a synthesis block

