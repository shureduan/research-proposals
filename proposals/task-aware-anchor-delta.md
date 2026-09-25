# Task-Aware Anchor-Delta Representations for Robot Learning

**Status:** Early-stage research proposal / research direction document  
**Last updated:** 2026-09

---

## 1. Research Motivation

Robot manipulation datasets contain substantial visual redundancy. Between critical events such as gripper closure, object lift, contact establishment, and contact release, most frames carry little new information: the camera is stationary, the background is unchanged, and the scene evolves slowly relative to the recording frame rate.

This redundancy has practical costs throughout the pipeline, beyond storage alone:

- **Storage and bandwidth** — Dense video is expensive to store, transfer, and version.
- **Data-loading throughput** — The training pipeline spends a disproportionate amount of I/O and decoding time on frames with little task-relevant signal.
- **Quality of the training signal** — Many redundant, nearly identical transitions may dilute the gradient signal that matters for learning.

The initial intuition is simple: **If most of a robot video is redundant, why store all of it and feed all of it to the model?** Rather than proposing a general-purpose video codec, the project narrows this intuition to a concrete, falsifiable research question.

## 2. Core Research Question

> **Does robot policy learning actually require continuous, dense video observations, or can a sparse set of visual anchors plus task-relevant state deltas suffice?**

The target representation is formalized as:

```text
Z = { anchors, deltas, events }
```

Two directly testable conditions follow:

```text
Storage(Z)     ≪ Storage(V)
Performance(Z) ≈ Performance(V)
```

Here, `V` denotes the full dense video. This formulation deliberately permits partial or negative findings, such as an approach that works for high-level action recognition but fails for precise insertion. Such a result would still be valuable and publishable because it would identify *where* dense visual observations are actually needed.

## 3. Proposed Representation

### 3.1 Anchor–Delta Structure

At each time step, the representation is either a complete anchor frame or a lightweight delta relative to the current anchor:

```text
Δ_t →  continue using the current anchor A_k
     →  refresh the anchor when accumulated Δ can no longer explain the current state
```

The policy and the anchor-refresh trigger consume **the same** delta signal. This shared signal is the central design distinction from a pure keyframe-selection method (see Section 5):

```text
(A_k, Δ_t) → Policy
```

The delta `Δ_t` is not discarded after the frame-selection decision. It remains a first-class input to the policy and supplies continuous state information between anchors.

### 3.2 What Counts as a Delta

The distance function `D(z_t, z_last)`, which drives anchor selection and event detection, may combine:

- Pixel- or feature-space differences, such as CLIP/DINO embedding distances
- Optical-flow magnitude
- Changes in robot state, including end-effector pose and joint angles
- Gripper-state transitions (open ↔ closed), which offer a particularly clear, low-noise event signal
- Action-conditioned prediction error, or “surprise”: when a learned predictor cannot explain the current observation from the anchor, a refresh is forced

### 3.3 Event Categories

The representation distinguishes four classes of per-frame retained signals: periodic full anchors, local visual deltas, robot/object-state deltas, and “surprise” frames that are retained in full when a compact delta cannot explain them.

## 4. Research Landscape and Positioning

Several neighboring research directions address related but distinct questions. At present, no work combines **task-aware sparsification, action-conditioned deltas, and direct policy consumption of the sparse representation** in the same way.

| Direction | What is compressed | Relationship to this project |
|---|---|---|
| Conventional video coding (H.264/AV1) | Pixel residuals | Structurally similar to I-frames and P-frames, but redundancy is defined by pixel similarity rather than task relevance. |
| Video/Feature Coding for Machines (MPEG) | Video/neural features | Designed for general machine-vision tasks such as detection and segmentation, rather than sequential robot decisions. |
| Neural video tokenizers (e.g., Cosmos) | Spatiotemporal latents | Achieve high compression, but the latents are not interpretable and still represent data on a uniform time grid. |
| Robot-data curation (SCIZOR) | Redundant state–action pairs | The closest prior work: it removes low-value transitions and reports higher policy success with less data. **Key distinction:** SCIZOR discards transitions, whereas the proposed method compresses them into a retained delta signal that continuously carries information (see Section 4.1). |
| Keyframe/skill discovery (KISA, HYDRA) | Key states and action phases | Identifies important moments for task decomposition, but does not define a training-data representation and still depends on full dense video. |
| World models (V-JEPA 2) | Latent-state predictions | Predicts future latent states rather than defining a shared, storable data format. |
| Object-state modeling (SPOC, ORION) | Object/state changes | Semantically related, but not validated as a policy-training representation. |
| Action tokenization (FAST, PRISE) | Continuous action sequences | Addresses redundancy on the action side rather than in visual observations. |
| Video-generation acceleration (SKIP) | Sparse keyframes plus interpolation | Philosophically the closest: sparsity concentrates near events such as approach, contact, grasp, and release. However, it ultimately reconstructs dense video for a world model. The proposed representation is intended for direct consumption without reconstruction. |

### 4.1 Distinction from SCIZOR

Because SCIZOR is the closest prior work, the difference should be stated precisely:

- **SCIZOR asks:** Should this transition be kept or deleted? This is a binary decision.
- **This project asks:** Should this transition be stored in full or compressed into a lower-fidelity delta? This forms a continuum rather than a binary choice.

Data discarded by SCIZOR disappears entirely. In this project, a compressed delta continues to carry a signal into the policy. Whether that difference provides a measurable advantage on tasks sensitive to slow, precise motion, such as insertion, is an open empirical question. Therefore, SCIZOR-style similarity-based deletion is a required strong baseline throughout the project.

## 5. Why Keyframe Selection Alone Is Insufficient

An early version used the delta signal solely to **select** which frames to retain and discarded the delta afterward. This is a weaker research contribution: after selection, the policy still consumes ordinary video frames, and only the number of frames has been reduced.

The current design assigns the delta signal two functions:

1. **Continuous signal** — `Δ_t` is provided to the policy directly alongside the current anchor.
2. **Anchor-refresh trigger** — When accumulated `Δ_t` can no longer explain the current state, it triggers a new anchor.

This design allows the policy to consume `(A_k, Δ_t)` directly without reconstructing a dense frame sequence. That is the project's central claim to be tested.

## 6. Three Research Tracks

| Track | Description | Role in the project |
|---|---|---|
| **A — Task-aware frame sparsification** | Select time steps whose full frames are retained; train the policy on the selected full frames. | Entry experiment to test the redundancy hypothesis, not the primary novelty claim. |
| **B — Anchor + learned residual** | `z_t = f(z_anchor, Δ_{1:t}, a_{1:t})`; the policy directly consumes the anchor plus accumulated latent deltas without reconstruction. | **Core research contribution:** the main claim is tested here. |
| **C — Object-centered semantic deltas** | Scene-graph-style updates such as `ADD contact(...)` and `UPDATE pose(...)`. | A longer-term, higher-risk extension outside the initial scope. |

Track A serves as a gatekeeping experiment rather than a deliverable in itself. Resources are concentrated on Track B.

## 7. Datasets

### 7.1 Public Benchmarks for Algorithm Development and Baseline Comparisons

| Dataset | Task | Reference policy | Notes |
|---|---|---|---|
| LeRobot PushT | Push a T-shaped object into a target region | Diffusion Policy | Official baseline: approximately 65.4% success and 0.955 overlap |
| LeRobot xarm_push_medium | Single-arm xArm tabletop pushing | BC / Diffusion Policy | Approximately 20,000 frames |
| LeRobot xarm_lift_medium | Single-arm xArm grasping and lifting | BC / Diffusion Policy | Approximately 20,000 frames |
| LeRobot ALOHA Transfer Cube (Human) | Grasp and transfer a cube | ACT | Approximately 20,000 frames; closer to real manipulation than PushT |
| LeRobot ALOHA Insertion (Human) | Grasp a peg/socket and complete insertion | ACT | 50 episodes / approximately 25,000 frames; an official pretrained ACT checkpoint is available |

These datasets span a difficulty gradient from lower-precision PushT to contact-intensive insertion and offer standard baselines for comparison. They make it possible to begin the first two stages without collecting data.

### 7.2 Real Hardware in Stage Three

**Kinova Gen3 (six degrees of freedom) robot arm + Robotiq 2F-85 gripper.**

- The six-degree-of-freedom joint/end-effector state determines the dimensions of the robot-state delta.
- The two-finger parallel gripper supplies a clear, low-noise binary event signal (open/closed). This is one of the main triggers for anchor refresh and event annotation and is generally more reliable than detecting contact solely from vision.

## 8. Deliverable: A Data-Processing Pipeline Compatible with LeRobot

The anticipated output is more than a paper. It includes a usable pipeline: **raw robot video and logs go in; a dataset accessible through a standard LeRobot interface comes out.**

### 8.1 Design Constraints

The LeRobot dataset format (Parquet tables + MP4 video + an `info.json` feature schema) assumes dense frames distributed uniformly across time steps by default. The proposed representation is sparse and nonuniform, requiring a design choice:

- **Mode A, compatibility first:** Reconstruct anchor-plus-delta data into dense frames when writing and package the result as a standard dataset. This maximizes compatibility but loses the central claim: users receive an ordinary dense dataset with a smaller footprint.
- **Mode B, expose the sparse structure:** Extend the schema with explicit sparse fields and supply a dataset class exposing **both** a standard `__getitem__` interface (compatible with existing LeRobot training code, reconstructing on demand) **and** a `get_sparse_item` interface (directly exposing `(anchor, delta)` for the Track B policy).

**Mode B is selected.** It is the only choice that keeps the deliverable aligned with the research contribution. The dataset is both the object used to validate the research claim and the final product.

### 8.2 Draft Schema

| Field | Type | Description |
|---|---|---|
| `episode_index`, `frame_index`, `timestamp` | Standard fields | Retained unchanged for compatibility |
| `is_anchor` | bool | Whether this frame is a complete anchor |
| `anchor_ref_index` | int | Which anchor the current delta is relative to |
| `delta_repr` | Fixed-length vector | Learned latent delta; populated only on non-anchor rows |
| `event_flag` | Enum | 0 = ordinary delta; 1 = gripper event; 2 = contact event; 3 = surprise (prediction error exceeds threshold) |
| `observation.state` | Standard field | Six-degree-of-freedom pose/joint angles + one-dimensional gripper state; kept dense because storage cost is small |
| `action` | Standard field | Kept dense, as the action sequence itself is usually not redundant |

Only rows for which `is_anchor == True` are actually encoded into the video stream. The compact `delta_repr` is stored in a sidecar array rather than in the video track.

### 8.3 Known Cost

On-the-fly reconstruction through the standard interface adds per-batch computation and may partly offset I/O savings. It must be measured directly using the throughput metric in Section 9. If reconstruction dominates, the fallback is to reconstruct and cache the data once in advance.

## 9. Evaluation Metrics

Compression ratio alone is insufficient. The project tracks:

| Metric | Purpose |
|---|---|
| Storage size | Raw storage savings |
| Fraction of retained frames per episode | Degree of sparsification |
| DataLoader throughput | Whether training actually gets faster, rather than only the files getting smaller |
| Encoding/decoding overhead | Whether compression costs exceed its benefits |
| Action-prediction accuracy | Whether task-relevant information is preserved |
| Policy success rate | The primary downstream metric |
| Out-of-distribution success rate | Whether information needed for generalization was removed |
| Failure rate by task stage | Pinpoint where information cannot be discarded |
| Steps since the last anchor refresh versus performance degradation | **Track B specific:** whether policy performance declines systematically as deltas accumulate |

The main experimental result is a **performance-versus-retained-data-fraction** curve. Comparators include uniform sampling, random sampling, similarity-based keyframe selection, and SCIZOR-style transition deletion.

## 10. Roadmap

**Stage One — Test the redundancy hypothesis.**  
Select a small LeRobot dataset. Compare the proposed selector with three baselines: fixed-interval, random, and similarity-based keyframe selection. Measure action-phase or next-action prediction accuracy at different retention fractions. This is a go/no-go checkpoint.

**Stage Two — Test the core contribution.**  
Add object tracking and robot-state/action signals. Train a learned task-aware selector. Implement a Track B policy that consumes `(anchor, Δ_{1:t}, a_{1:t})` directly, without reconstruction. Compare against a dense-video policy on two to three tasks. Measure both open-loop prediction error and closed-loop drift/success degradation.

**Stage Three — Validate in a real setting.**  
Collect data on Kinova Gen3 + Robotiq 2F-85, prioritizing a fixed camera and tabletop tasks. Measure deployment success, out-of-distribution generalization, and failure modes. Complete the LeRobot-compatible Mode B pipeline. Submit to a robot-learning workshop first, and consider a main conference (CoRL/ICRA/IROS) if the results are strong.

## 11. Novelty Statement

None of the mechanisms is individually new: keyframes, residual coding, latent tokenization, transition filtering, and action tokenization all have counterparts in prior work (Section 4). The contribution lies in the specific combination and the empirical claim it tests:

> The project investigates whether robot learning truly requires dense visual observations and proposes an action-conditioned incremental representation that retains decision-relevant state transitions under aggressive temporal sparsification. The policy consumes that representation directly without reconstructing dense video.

## 12. Open Risks

- **Reconstruction may cancel the I/O savings** in the standard-interface path (Section 8.3).
- **Drift and error accumulation between anchor refreshes** are major failure modes for Track B and must be measured directly, rather than inferred from open-loop metrics alone.
- **Uniform/fixed-interval subsampling may already be a strong baseline in practice.** Many VLA pipelines sample below the camera frame rate; the method must outperform that baseline, not just full dense video.
- **SCIZOR-style deletion is a mandatory strong baseline.** The proposed advantage of retaining compressed deltas over deletion (Section 4.1) has not yet been validated empirically.
- Storage savings and computational savings are separate claims and must be reported separately (Section 9).
