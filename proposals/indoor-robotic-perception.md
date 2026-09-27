# Adaptive Radar History Selection and Visual Anchor Refresh for Indoor Robotic Perception

**Status:** Research proposal with an implementation and evaluation plan  
**Version:** 2.0 | September 2026  
**Initial task:** Sequential 3D detection of doors and obstacles  
**Core inputs:** 4D radar point clouds, RGB images, and causal motion estimates

---

## 1. Research Motivation

Indoor robots supporting building inspection, renovation, and emergency reconnaissance must interpret corridors, corners, and doorways as their viewpoints change. Current observations can be incomplete, while previous observations may become unreliable because of motion, occlusion, or changes in visibility.

Radar and vision have different costs and failure modes. Aggregating radar observations can improve geometric coverage, but excessive history increases processing and may introduce misalignment or moving-object trails. Visual features provide appearance information, but repeatedly encoding similar images can be expensive. A cached visual representation saves computation only while it remains useful for the current task.

The proposed research asks how these two sources of temporal information should be managed together. It connects earlier work on multimodal temporal windows and task-aware visual updates to a concrete perception problem. Its initial contribution would be a decision mechanism around an existing detector, supported by controlled experiments.

## 2. Core Research Question and Hypotheses

> Under a limited computation budget, can a robot choose how much radar history to use and when to refresh visual features while preserving reliable detection of doors and obstacles?

The principal objective is a measurable improvement in the detection–latency trade-off:

$$
\min_{\pi}\;\mathbb{E}[C_t(\pi)]
\quad\text{subject to}\quad
\operatorname{mAP}(\pi)\geq\operatorname{mAP}(B)-\epsilon,
$$

where $\pi$ is a temporal selection policy, $C_t$ is measured processing cost, and $B$ is a validation-selected reference configuration. Section 10 defines provisional acceptance targets.

| Hypothesis | Testable prediction | Evidence that would weaken it |
|---|---|---|
| H1: Useful radar history depends on observation and alignment quality. | Adaptive history improves detection at comparable latency relative to fixed windows and simple motion rules. | One fixed window performs equally well across conditions. |
| H2: Radar and motion cues help determine when visual features need updating. | A cross-modal trigger outperforms a visual-only trigger at comparable computation. | Anchor age or inexpensive image change alone explains the benefit. |
| H3: The two decisions benefit from coordination. | Joint selection improves the trade-off beyond independently optimized history and refresh policies. | Independent policies achieve equivalent results with less overhead. |

The project can produce useful findings if only H1 or H2 holds. A coordination contribution requires direct support for H3.

## 3. Research Position and Scope

Adaptive aggregation and feature reuse already have substantial precedents. The proposed work therefore concentrates on the task value of historical information and the cost of deciding whether to use it.

| Related work | Established direction | Role in this study |
|---|---|---|
| DoppDrive, ICCV 2025 | Doppler-based compensation and selection of radar aggregation durations | Required methodological comparison where the necessary radar fields are available. |
| Towards High Performance Video Object Detection, CVPR 2018 | Adaptive keyframes and temporal feature reuse | Basis for quality-driven visual refresh comparisons. |
| TARSS-Net, NeurIPS 2024 | Temporal relationships for radar semantic segmentation | Modeling reference; its segmentation network is not a directly comparable detector. |
| VLA-Cache, NeurIPS 2025 | Reuse of visual tokens in robot policy inference | Related efficiency principle; the proposed task evaluates 3D detection rather than manipulation policies. |
| M³Detection, 2025 preprint | Multi-frame radar–camera detection with feature reuse and trajectory reasoning | Close related work that must be considered when defining the contribution. |
| R4Det, CVPR 2026 | Radar–camera detection with pose-free temporal fusion | Strong external detector comparison and a test of whether improved temporal fusion reduces the need for the proposed selector. |

The candidate contribution is **a lightweight, causal decision module that coordinates radar history and visual refresh using observation quality, alignment reliability, and measured cost**. Its novelty remains a hypothesis to establish through literature review and experiments.

The initial scope is recorded-sequence perception. Navigation success, collision reduction, human–robot collaboration, and world-model training are later extensions requiring separate tasks and evidence.

## 4. Proposed Representation and Technical Route

### 4.1 Inputs, memory, and outputs

At radar timestamp $t$, the system receives the current point cloud $R_t$, the most recent RGB image $I_t$ captured no later than $t$, and a causal motion estimate. All selected radar history also ends at $t$.

The initial window candidates are **1, 2, 4, and 8 radar frames, including the current frame**. Their duration in seconds is recorded for each dataset; comparable-duration windows are used in cross-dataset experiments.

A visual anchor stores:

$$
A_k=\{F_k,\tau_k,\widehat T_k,K,E,q_k\},
$$

where $F_k$ is the image feature map, $\tau_k$ its timestamp, $\widehat T_k$ the estimated pose, $K,E$ the calibration, and $q_k$ a quality summary. The cache initially contains one anchor.

The decision is:

$$
a_t=(w_t,u_t),\qquad
u_t\in\{\text{reuse},\text{refresh},\text{disable visual input}\}.
$$

This gives 12 candidate actions. Disabling vision prevents an invalid anchor from being used when the current image also offers little usable evidence. The detector returns 3D boxes, class scores, and a record of the action taken.

### 4.2 Processing sequence

1. Compute inexpensive summaries from the current observations and cache metadata.
2. Choose a radar window and visual action.
3. Align only the selected radar history into the current coordinate frame.
4. Reuse or recompute visual features, or mask visual input.
5. Fuse the selected information and run the common 3D detector.
6. Log detections, cache age, selected history, quality indicators, and complete processing time.

No full current-image encoding is performed merely to decide whether that encoding is needed. Likewise, the selector must not aggregate every candidate radar window before making its decision.

### 4.3 Initial detector and feature reuse

The first implementation uses a PointPillars-style radar detector and a compact RGB encoder, initially ResNet-18 with a feature pyramid. Camera features sampled at projected radar locations are fused with point or pillar features before the common detection head. These are implementation starting points, to be checked against indoor object geometry during the baseline stage.

Radar history is transformed using estimated relative poses:

$$
\bar R_t(w)=\bigcup_{j=t-w+1}^{t}\widehat T_{t\leftarrow j}R_j.
$$

Each point retains its age. Doppler and signal-strength fields are included only when supplied and verified. Ego-motion compensation alone does not correct moving-object motion; this is explicitly tested in the external datasets.

For a current-frame radar location $p_t$, cached image features are sampled by projecting that location into the anchor camera:

$$
f^v_t(p_t)=\operatorname{sample}
\left(F_k,\Pi\!\left(K E\widehat T_{k\leftarrow t}p_t\right)\right).
$$

This projection uses radar geometry and relative pose without requiring a new dense depth estimate at every frame. It is only a correspondence approximation: old pixels can describe occluders or moving objects. Projection validity, anchor age, viewpoint change, and available geometric consistency checks therefore produce a validity mask. Newly exposed regions without trustworthy correspondence receive no cached visual feature.

A fusion gate weights valid visual evidence. The same gate is available to all comparable baselines; gating gains are evaluated separately from scheduling gains. The minimal radar-supported fusion design may miss objects with few or no radar returns. Such failures are reported, and R4Det provides a relevant stronger comparison.

### 4.4 Lightweight decision features

| Feature group | Example inputs | Intended use |
|---|---|---|
| Radar evidence | Current point count, spatial coverage, signal statistics, inexpensive current/previous-frame consistency | Estimate whether additional history could be useful. |
| Motion and alignment | Relative translation and rotation, odometry residuals, inlier ratio or a calibrated reliability proxy | Identify histories that may be difficult to align. |
| Visual observations | Brightness, saturation, blur, and grid-level changes from a small image thumbnail | Detect poor visibility or newly changed image regions cheaply. |
| Cache state | Anchor age, accumulated viewpoint change, valid projection coverage, previous detector uncertainty | Estimate whether old features remain useful. |
| Resource state | Current point count, action-cost estimates, recent processing time | Avoid actions whose expected cost exceeds the selected budget. |

For the prototype, thumbnails may be resized to approximately 96 × 54 pixels. All resizing, consistency checks, and motion processing count toward latency. Ground-truth boxes, future observations, and current detections that require the unselected full computation cannot enter the selector.

### 4.5 Motion estimation and fallback

Two evaluation tracks are maintained: a supplied-pose diagnostic track and a causal deployment-oriented track. The latter uses a verified radar–inertial or other available causal odometry front end, with its sensor dependencies and runtime reported. IMU orientation alone is insufficient to provide translation.

On initialization or after a sequence break, cached state is cleared. If alignment becomes unreliable, history is shortened and stale features are invalidated; a current image is encoded when useful, otherwise the system falls back to current radar. Maximum anchor age and quality thresholds are selected on validation sequences.

If reliable online alignment cannot be achieved, results remain a supplied-pose feasibility study rather than an online system claim.

## 5. Selector Learning and Implementation Stages

**Stage A: Rules and opportunity analysis.** Sweep fixed radar windows and periodic visual intervals, then test motion- and quality-based rules. Profile where computation is spent and whether the best action changes across conditions. An offline action oracle estimates the opportunity for adaptation; it is a diagnostic reference, not a deployable method.

**Stage B: A small learned selector.** Train the detector with varied history lengths, cache ages, and visual dropout so that it can handle all candidate inputs. Freeze its weights for the first scheduling comparison. On training sequences, sample reachable cache states from several causal schedules. Clone each state and evaluate alternative actions using training labels and measured action costs.

For each sampled state, an initial target action is:

$$
a_t^*=\arg\min_{a\in\mathcal A_t}
\left[\mathcal L_{\mathrm{det}}(a;y_t)
+\lambda\widehat C(a;s_t)\right].
$$

A small multilayer perceptron predicts action scores from the inexpensive state summary. The cost weight is selected on validation data. This one-step target does not capture every future consequence of refreshing a cache; deployment evaluation must therefore replay the selector's own evolving state over complete sequences.

**Stage C: Coordination.** Compare a joint 12-action selector with separately trained history and visual-action selectors given the same input summaries and comparable capacity. Only retain joint selection if it improves the measured trade-off after counting its overhead. Reinforcement learning and large temporal transformers are optional follow-ups, not prerequisites.

## 6. Candidate Datasets and Their Roles

Dataset descriptions below were checked against author or official sources in September 2026. They describe potential experiments; access and usable sequence coverage still require verification.

| Dataset | Relevant data | Proposed validation role | Access or interpretation limit |
|---|---|---|---|
| **Indoor FireRescue Radar (IFR)** | Approximately 27,000 synchronized multimodal frames across 10 buildings and 35 layouts; RGB, 4D radar, IMU, LiDAR, poses, and 3D boxes | Primary indoor door/obstacle detection; history selection, cache refresh, and held-out-building evaluation | Public demonstration subset; full data require a request. Confirm complete sequences and condition coverage before fixing the experiment. |
| **View-of-Delft (VoD)** | More than 8,600 synchronized automotive frames; radar, cameras, LiDAR, odometry, and 3D road-user labels with tracking IDs | Main external replication candidate: moving objects, different motion, and different radar characteristics | Academic access request required. Outdoor road-user results do not establish indoor deployment performance. |
| **TJ4DRadSet** | 7,757 annotated frames in 44 sequences; radar and 3D labels; the recorded dataset includes camera/LiDAR and varying illumination | Conditional replication under darkness and dynamic traffic; radar-history tests if only radar is available | The official repository currently warns that only complete 4D radar data are released. Confirm actual synchronized RGB access before committing to fusion or visual-refresh experiments. |
| **NTU4DRadLM** | Radar, RGB, thermal, IMU, LiDAR, and reference odometry across six outdoor trajectories | Optional diagnostic for motion estimation, alignment residuals, and cache behavior | Primarily a localization/mapping dataset. Do not assume detection boxes exist or report detection AP without added annotations. |

**Minimum empirical scope:** IFR plus one accessible external detection dataset. VoD is preferred for the second dataset; TJ4DRadSet is a conditional alternative. NTU4DRadLM diagnostics cannot replace the second detection experiment. Each dataset is trained and evaluated with its own classes; replication across datasets is distinct from zero-shot transfer.

### 6.1 Indoor task and label mapping

The first IFR experiment uses two evaluation classes: **door** and **obstacle**. The obstacle category combines desk/chair, cabinet, and waste-container labels, following the aggregation supported in the data description. Other annotated categories receive an explicit ignore policy. This subset does not cover every possible obstacle.

The mapping, detection range, box convention, and minimum class support are fixed during data preparation. A broader class-wise experiment follows only if label counts support it. Public sample performance is not used to claim generalization across buildings.

### 6.2 Temporal and spatial separation

Split by building before generating radar windows or caches. With sufficient access, reserve at least two buildings for validation and two for the main test; use remaining ordinary-building data for training. Rare conditions confined to one facility are separately reported rather than pooled into a broad robustness claim.

External experiments use sequence-disjoint partitions for temporal evaluation. If an official split has a different structure, publish both its conventional result and a separate temporal protocol. Never carry observations or cache state across train/test boundaries. Use sensor timestamps to exclude future camera frames even when the dataset supplies a nearest-time pairing.

## 7. Experimental Comparisons

### 7.1 Required controlled baselines

| ID | Configuration | Question answered |
|---|---|---|
| B0 | Radar-only detector with validation-selected fixed history | How much does visual evidence help? |
| B1 | RGB encoded every frame; validation-selected fixed radar history | Reference for full visual computation. |
| B2 | Fixed history plus periodic visual refresh | Can a simple schedule obtain the same savings? |
| B3 | Motion/quality rules for history and visual refresh | Is learning necessary? |
| B4 | Learned history; visual encoding every frame | Does adaptive radar history help independently? |
| B5 | Fixed history; visual-only refresh trigger | Does radar provide useful refresh evidence? |
| B6 | Fixed history; radar-informed visual trigger | Is cross-modal triggering useful independently? |
| B7 | Independently optimized history and refresh selectors | Strong reference for the coordination claim. |
| P | Joint history and visual-action selection | Proposed coordinated method. |

All scheduling comparisons use the same frozen detector, alignment method, image resolution, point-cap policy, fusion gate, and training data. A second experiment may fine-tune each policy-specific detector with equal training resources. Radar-only and external architectures are reported separately from these controlled comparisons.

Fixed refresh intervals initially include 1, 2, 4, and 8 processing steps. Sweep budget settings on validation sequences to build baseline trade-off curves; compare against the best eligible baseline at each operating point rather than a deliberately expensive configuration.

### 7.2 Related-method comparisons

DoppDrive must be assessed where Doppler and calibration fields support adaptation. Its inspected repository currently contains a README rather than a complete implementation; any local implementation must be labeled and its checks documented. If required inputs are unavailable, the limitation must be explicit and the novelty claim narrowed.

R4Det has an official implementation and is the preferred stronger radar–camera comparator. M³Detection is another close temporal-fusion comparator, subject to implementation availability. Their native architectures are evaluated as external references; swapping entire detectors is not evidence that the proposed scheduling mechanism caused an improvement.

### 7.3 Ablations and stress tests

The core ablations remove, one at a time, alignment reliability, radar cues in visual refresh, joint decision making, and adaptive visual weighting. Additional checks compare fixed-age expiry with the learned trigger and fixed point counts with naturally varying point counts.

Stress tests cover fast turns, newly visible regions, sparse radar returns, moving objects, visual degradation, and pose errors. Natural conditions and synthetic perturbations are reported separately. Darkened images or simulated pose noise are controlled sensitivity tests, not substitutes for real low-light recordings or measured odometry errors.

## 8. Evaluation Metrics and Measurement Protocol

| Metric | Definition and use |
|---|---|
| 3D detection AP | For the initial IFR task, report macro AP at 3D IoU 0.25 and 0.50, plus class-wise results. These are proposed study thresholds; also report the dataset's reference protocol where available. |
| Recall and precision | Select one score threshold per class on validation data targeting 90% precision. Freeze thresholds and report both achieved test precision and recall. An unattainable target is reported explicitly. |
| Geometric error | Center and yaw errors on matched boxes, with match coverage; useful for thin doors whose IoU can be unstable. |
| Mean and p95 latency | Complete causal processing time, including synchronization, motion estimation, selection, alignment, encoding, cache operations, fusion, and postprocessing. |
| Tail behavior | Worst-frame latency, deadline-miss fraction, and visual-refresh burst cost. |
| Resource use | Peak memory, visual encoding rate, selected history duration, retained point count, and selector overhead. |
| Condition-specific performance | Detection and latency by motion, illumination, point sparsity, visibility change, and anchor age; include sample counts. |

Latency is measured with batch size one on a named device, fixed software versions, GPU synchronization, and no future-frame batching. Benchmark complete sequential replay, including cache initialization, with at least three timing repetitions. Report model execution separately from a deployment-oriented measurement that also includes input decoding and transfer. Persistently precomputed test features cannot support a runtime claim.

The main detector and selector comparisons use at least three training seeds. Report paired differences and 95% intervals from resampling whole sequences, with building-aware aggregation where possible. Also show each held-out building individually; a small number of buildings cannot support strong population-wide statistical claims.

## 9. Deliverables and Reproducible Interface

The intended artifact is a perception pipeline that accepts synchronized sequences and exposes the temporal decisions alongside detection outputs.

| Record | Required fields |
|---|---|
| Input index | Dataset, building/sequence ID, sensor timestamps, calibration reference, label reference |
| Motion record | Pose estimate, estimation source, reliability proxy, sensor dependencies |
| Anchor state | Feature reference, timestamp, pose, age, validity coverage |
| Decision log | Radar window in frames and seconds, visual action, predicted action costs |
| Output | 3D boxes, classes, scores, runtime breakdown, memory statistics |

Expected deliverables are:

1. Dataset adapters, documented class mappings, and sequence-disjoint split manifests.
2. Fixed-policy, rule-based, independent-selector, and joint-selector implementations.
3. Benchmark scripts that generate detection–latency curves, per-condition results, and failure-case visualizations.
4. A technical report documenting the supported hypotheses, remaining failures, and computational costs.

Any release respects the source datasets' access terms. Dataset preparation code and split manifests can be shared without redistributing restricted sensor recordings.

## 10. Acceptance Criteria and Decision Gates

The following numbers are **provisional engineering targets**, not results or literature-derived guarantees. They must be preregistered after the data/profile pilot and before examining final test outcomes. Any change is versioned and justified by the pilot, never by test performance. AP differences below are percentage points on a 0–100 scale.

| Gate | Acceptance criterion | Response if unmet |
|---|---|---|
| G0: Data feasibility | Complete radar/RGB/calibration/label sequences support a building-disjoint IFR experiment with two held-out test buildings. Causal motion inputs and condition coverage are documented. | Continue a clearly labeled pilot; postpone the mature indoor generalization claim. |
| G1: Baseline readiness | Common detector converges across three seeds; fixed-history/refresh curves and a full latency breakdown are available. | Resolve detector, alignment, or class-mapping problems before adding a learned selector. |
| G2: Scheduling opportunity | Training/validation oracle analysis shows meaningful variation in preferred actions and enough avoidable cost to make G3 plausible. | Prefer the best simple schedule or narrow the project to one decision. |
| G3: Primary efficiency target | Relative to B1, reduce mean complete-processing latency by at least **15%**, lose no more than **1.0 pp** primary IFR mAP, and increase p95 latency by no more than **5%**. | Report the trade-off honestly; fewer encoder calls alone do not pass. |
| G4: Value beyond simple scheduling | Relative to the best validation-selected simple baseline B2/B3, either gain at least **1.0 pp mAP** at matched mean latency within **5%**, or save at least **10%** mean latency with at most **1.0 pp mAP** loss. | Retain a simple implementation and avoid a learned-scheduling superiority claim. |
| G5: Coordination contribution | P improves over B7 by at least **0.5 pp mAP** at matched latency within **5%**, or saves at least **5%** latency within **0.5 pp mAP**. The paired interval supports a positive gain. | Drop the coordination claim; report independently useful components. |
| G6: Detection guardrail | At the validation-selected thresholds, neither indoor class loses more than **2.0 pp recall** relative to B1. Report precision and all predefined stress strata alongside this check. | Reject that operating point for the primary efficiency claim. |
| G7: Replication | Repeat the central comparison on one external dataset with verified RGB and radar access, showing a consistent trade-off improvement or explaining its failure. | Restrict the conclusion to the demonstrated indoor setting. |

Passing G0–G4 and G6 supports the primary indoor method claim. G5 is additionally required for a coordination claim, and G7 for a broader replication claim. Report confidence intervals alongside all numerical gates; uncertain evidence does not become decisive merely because a mean crosses a target.

Project completion also permits a well-supported negative result: a reproducible benchmark identifying when simple policies suffice or when selector overhead eliminates savings. It does not imply that all method-success gates have passed.

## 11. Roadmap and Resource Plan

The proposed schedule assumes dataset access and one CUDA-capable research GPU. Runtime claims are limited to the measured hardware; a desktop/cloud result does not establish onboard real-time performance.

| Phase | Indicative duration | Main work | Exit artifact |
|---|---|---|---|
| 1. Access and feasibility | Weeks 1–2, excluding access delays | Verify fields, labels, timestamps, sequence continuity, and permitted use; profile a small pilot. | Data audit, split plan, and frozen initial protocol. |
| 2. Fixed and rule baselines | Weeks 3–5 | Train the shared detector; sweep windows and refresh intervals; inspect alignment and cost bottlenecks. | Baseline trade-off curves and G1/G2 decision. |
| 3. Individual selectors | Weeks 6–8 | Generate training action targets; train history and refresh selectors; run component ablations. | Evidence for H1 and H2. |
| 4. Coordination and robustness | Weeks 9–11 | Evaluate P against B7; conduct motion, visibility, and degradation tests. | G3–G6 assessment and failure analysis. |
| 5. External replication and release | Weeks 12–16 | Repeat the core experiment on an accessible external dataset; finalize artifacts. | Replication report, code, configurations, and technical manuscript. |

Compute planning proceeds from the measured pilot cost per epoch and per sequential replay. Initial runs use a compact backbone and short candidate set. Full multi-seed sweeps begin only after the fixed-policy experiments show an opportunity. Offline target enumeration and online inference costs are tracked separately; selector training overhead is included in the total research compute report.

## 12. Risks, Expected Outcomes, and Extensions

| Risk | Planned response |
|---|---|
| Full IFR or external RGB access is unavailable | Use accessible data for pipeline development and revise the experiment scope explicitly. |
| Cached features are geometrically misplaced or describe an occluder | Test correspondence validity, shorten cache lifetime, and compare against refresh or radar-only fallback. |
| Longer radar history creates moving-object trails | Preserve point ages, assess Doppler compensation, and analyze static and moving targets separately. |
| The lightweight fusion baseline is too weak | Establish a stronger detector before attributing gains to scheduling; compare with R4Det. |
| The selector costs more than it saves | Reduce decision features or retain fixed/rule-based policies. |
| Training action targets do not generalize to self-generated cache states | Train on states from several schedules and evaluate complete on-policy sequence replay. |
| Average AP hides missed doors or long latency spikes | Apply class recall guardrails and report tail latency and condition-specific results. |

The expected scientific outcome is evidence about **when historical radar remains useful, when cached visual information becomes insufficient, and whether coordinating those choices adds value**. The engineering outcome is a reproducible implementation with measured operating points and documented failure conditions. A publication is a dissemination objective, not an acceptance metric.

The work extends the temporal-window question in DWA and the update-timing question in Anchor–Delta. It tests those ideas in perception without assuming that results from robot manipulation automatically transfer to radar–camera detection.

Later work may study short-term prediction of perceptual state, interaction with a Unity environment, or navigation around temporary obstacles. World-model or navigation experiments would require action-conditioned sequences, independent simulation validation, and closed-loop measures such as goal success, collision frequency, and recovery time. These extensions follow successful perception validation and are outside the initial acceptance criteria.
