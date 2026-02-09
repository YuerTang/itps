# Investigation: ITPS Paper and Codebase Analysis

**Date**: 2026-02-05
**Type**: System Architecture Archeology + Technical Exploration

## Task Description

Understand the Inference-Time Policy Steering (ITPS) framework by:
1. Reading and analyzing the research paper thoroughly
2. Exploring the codebase structure and implementation
3. Understanding the task/problem being solved
4. Documenting the relationship between paper concepts and code

## Background Context

ITPS is a framework that leverages human interactions to bias the generative sampling process at inference time, rather than fine-tuning the policy on interaction data.

## Task Checklist

- [x] Read and summarize the PDF paper
- [x] Explore codebase structure
- [x] Document core algorithms and methods
- [x] Understand the robotic manipulation tasks
- [x] Map paper concepts to code implementations
- [x] Document training and inference pipelines

---

## Paper Summary

See: [01-paper-summary.md](./01-paper-summary.md)

## Codebase Architecture

See: [02-codebase-architecture.md](./02-codebase-architecture.md)

## Paper-to-Code Mapping

See: [03-paper-code-mapping.md](./03-paper-code-mapping.md)

---

## Quick Overview

### What is ITPS?

**Inference-Time Policy Steering (ITPS)** is a framework that allows humans to guide pre-trained robotic policies *without fine-tuning*. Instead of retraining models on interaction data, ITPS modifies the *sampling process* at inference time to align generated trajectories with user intent.

### The Core Problem

Generative policies (like Diffusion Policy, ACT) trained via behavior cloning can:
- Execute multimodal, long-horizon tasks autonomously
- BUT humans are removed from the execution loop during inference
- Naive human intervention can cause **distribution shift** → constraint violations, execution failures

### The Solution

ITPS frames policy steering as **conditional sampling** from the learned distribution:
- Likelihood constraints from training keep outputs valid
- Conditional sampling aligns outputs with user objectives
- No modification to the pre-trained policy weights

### Three Types of Human Interaction

1. **Point Input**: Click a pixel → robot reaches that 3D location
2. **Sketch Input**: Draw a trajectory → robot follows that path
3. **Physical Correction**: Physically move the robot → policy continues from there

### Six Sampling Strategies

| Method | Description | Policy-Agnostic? |
|--------|-------------|------------------|
| Random Sampling (RS) | Baseline, no steering | Yes |
| Output Perturbation (OP) | Overwrite trajectory start, resample | Yes |
| Post-Hoc Ranking (PR) | Sample batch, pick closest to input | Yes |
| Biased Initialization (BI) | Initialize diffusion with corrupted input | Diffusion only |
| Guided Diffusion (GD) | Add alignment gradient during denoising | Diffusion only |
| **Stochastic Sampling (SS)** | MCMC sampling for product distribution | Diffusion only |

### Key Finding: Alignment vs. Constraint Satisfaction Trade-off

More aggressive steering → better alignment with user intent → BUT more distribution shift → more failures

**Stochastic Sampling (SS)** achieves the best trade-off by sampling from the *product* of the policy distribution and objective distribution (rather than their sum, like GD).

### Evaluation Tasks

1. **Maze2D** - Continuous motion alignment with sketch input
2. **Block Stacking** - Discrete task alignment in Isaac Sim
3. **Real Kitchen** - Real-world manipulation with point/physical input

