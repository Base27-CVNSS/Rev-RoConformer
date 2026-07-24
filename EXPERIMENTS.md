# 🧪 Rev‑RoConformer Experimental Protocol

**Purpose:** convert the architecture proposal into a falsifiable ASR study without inventing performance results.

---

## 1. Primary hypothesis

Rev‑RoConformer is valuable only if reversible activation reconstruction improves the practical training frontier of deep ASR encoders.

The principal hypothesis is:

\[
\boxed{
\text{At matched ASR quality, reversible depth reduces peak training memory enough to enable a larger useful depth or batch size.}
}
\]

This hypothesis may be rejected.

---

## 2. Required systems

| ID | Encoder | Position | Memory strategy | Purpose |
|---|---|---|---|---|
| B0 | Conformer | RelPos | Standard | classical reference |
| B1 | Conformer | RoPE | Standard | isolate RoPE |
| B2 | Rev-Conformer | RelPos | Reversible | isolate reversibility |
| **P** | **Rev‑RoConformer** | **RoPE** | **Reversible** | proposed intersection |
| C1 | RoPE-Conformer | RoPE | Gradient checkpointing | strongest direct memory baseline |

Optional contextual baselines:

- Efficient Conformer;
- Zipformer;
- Deep Sparse Conformer or an equivalent sparse/local-attention implementation.

These optional systems address different efficiency axes and should not replace B0/B1/B2/C1.

---

## 3. Dataset ladder

### Tier 0 — unit and synthetic tests

No speech dataset required.

Validate:

- exact algebraic inversion in FP64/FP32;
- numerical reconstruction in BF16/FP16;
- gradient equivalence versus ordinary autograd;
- deterministic handling of dropout/random state;
- scaling over 12, 24, 48, 72 and 96 blocks.

### Tier 1 — LibriSpeech train-clean-100

Use for rapid iteration and ablation.

Report:

- dev-clean/dev-other WER;
- peak allocated/reserved GPU memory;
- samples/s and audio-seconds/s;
- gradient/reconstruction diagnostics.

### Tier 2 — LibriSpeech 960 h

Main benchmark.

Report test-clean and test-other WER under identical decoding assumptions.

### Tier 3 — multilingual/cross-domain

Candidate datasets:

- AISHELL-1 for Mandarin;
- Common Voice for multilingual robustness;
- VoxPopuli where compute/data policy permits.

The multilingual stage is secondary to the core memory-scaling claim.

---

## 4. Acoustic frontend

Default:

- mono 16 kHz waveform;
- 80-bin log-Mel filterbank;
- 25 ms analysis window;
- 10 ms hop;
- utterance/global normalization according to the recipe;
- convolutional subsampling ×4.

A controlled ×8-subsampling ablation is recommended because attention cost depends strongly on frame count.

---

## 5. Proposed reversible block

Split the hidden state along channels:

\[
X=[x_1;x_2], \quad x_1,x_2\in\mathbb R^{B\times T\times d/2}.
\]

Forward:

\[
y_1=x_1+F(x_2)
\]

\[
y_2=x_2+G(y_1)
\]

with a candidate decomposition

\[
F(z)=\mathrm{FFN}_{1/2}(z)+\mathrm{RoPEMHSA}(z)
\]

and

\[
G(z)=\mathrm{ConvModule}(z)+\mathrm{FFN}_{1/2}(z).
\]

Inverse:

\[
x_2=y_2-G(y_1),\qquad x_1=y_1-F(x_2).
\]

The exact module ordering should be fixed before the primary benchmark and then treated as a preregistered architecture choice.

---

## 6. RoPE

For each head and each paired hidden dimension:

\[
R_t(\theta)=
\begin{bmatrix}
\cos(t\theta)&-\sin(t\theta)\\
\sin(t\theta)&\cos(t\theta)
\end{bmatrix}.
\]

Apply to queries and keys:

\[
Q'_t=R_tQ_t,\qquad K'_t=R_tK_t.
\]

Do not claim that RoPE removes quadratic attention complexity.

---

## 7. Memory propositions to test

For a conventional depth-\(L\) stack, the activation term scales approximately as

\[
M_{act,std}=\Theta(LBTd)
\]

under ordinary storage assumptions.

For an ideal reversible stack, the depth-dependent retained state can approach

\[
M_{act,rev}=\Theta(BTd),
\]

while real implementations also contain:

- temporary F/G recomputation tensors;
- attention/SDPA workspaces;
- frontend and decoder states;
- optimizer states;
- parameter gradients;
- CUDA allocator fragmentation;
- framework bookkeeping.

Therefore the benchmark must report **measured peak memory**, not only asymptotic formulas.

---

## 8. Three mandatory comparison regimes

### A. Iso-model

Hold constant:

- \(L\), \(d\), FFN ratio;
- attention heads;
- batch shape;
- utterance/sequence length distribution;
- optimizer;
- precision;
- decoder/loss.

Measure direct memory and throughput effects.

### B. Iso-memory

Give each model the same GPU memory budget, for example one 40 GB or 80 GB device class.

Allow batch size/depth to increase until a predefined safety margin below OOM.

Question:

> What useful model/batch can each method actually train under the same hardware limit?

### C. Iso-compute

Match total training FLOPs or measured GPU-hours as closely as practical.

Question:

> Does memory saving still produce a quality advantage once recomputation cost is charged fairly?

---

## 9. Depth sweep

At minimum:

\[
L\in\{12,24,48,72,96\}.
\]

A 100-layer result by itself is not a novelty target because Deep Sparse Conformer previously explored hundred-level Conformer depth.

For every depth report:

- parameter count;
- peak memory;
- throughput;
- reconstruction error;
- training stability;
- WER/CER.

---

## 10. Numerical reconstruction tests

For block \(\ell\):

\[
E_{rec}^{(\ell)}=\frac{\|X_\ell-\hat X_\ell\|_2}{\|X_\ell\|_2+\epsilon}.
\]

Also record maximum absolute error:

\[
E_{\infty}^{(\ell)}=\|X_\ell-\hat X_\ell\|_\infty.
\]

Test:

- FP32;
- BF16;
- FP16 where stable/supported;
- optional FP64 synthetic reference.

Track error both per block and after deep reverse reconstruction.

A reversible architecture that saves memory but corrupts gradients at practical precision fails the research objective.

---

## 11. Gradient-equivalence test

For a small deterministic network, compute gradients using:

1. ordinary forward/backward with stored activations;
2. reversible forward + reconstruction backward.

For each parameter \(p\), report

\[
E_{grad}(p)=\frac{\|g_p^{std}-g_p^{rev}\|_2}{\|g_p^{std}\|_2+\epsilon}.
\]

This test must pass before speech training begins.

---

## 12. Dropout/randomness

Stochastic layers create a hidden implementation problem: recomputation must reproduce the same stochastic transformation used in forward computation.

Valid approaches include:

- storing/restoring RNG state;
- deterministic/stateless masks keyed by layer and step;
- disabling dropout inside reconstructed functions during initial validation;
- using a reversible-autograd implementation that explicitly preserves stochastic state.

The chosen approach must be documented because incorrect dropout replay can silently invalidate gradients.

---

## 13. Attention backend matrix

Run the primary study first with a common backend across systems.

Recommended stages:

| Stage | Backend | Question |
|---|---|---|
| A | PyTorch SDPA/reference | correctness |
| B | FlashAttention-compatible exact attention | practical memory/throughput |
| C | optional local/sparse attention | separate long-sequence extension |

Never attribute FlashAttention savings to reversibility or vice versa.

---

## 14. Loss and decoder

Recommended first-paper objective:

\[
\mathcal L=\lambda\mathcal L_{CTC}+(1-\lambda)\mathcal L_{AED}.
\]

A Transducer version is a useful extension but should not be required for the first controlled study.

Keep decoder structure identical between paired baseline/proposed systems whenever possible.

---

## 15. Primary metrics

### Recognition

- WER;
- CER where appropriate.

### Memory

- `torch.cuda.max_memory_allocated()`;
- `torch.cuda.max_memory_reserved()`;
- external profiler peak if available;
- GB/sample and GB/audio-minute where meaningful.

### Speed

- optimizer steps/s;
- samples/s;
- audio-seconds/s;
- end-to-end training wall-clock;
- GPU-hours to target validation WER.

### Stability

- NaN/Inf events;
- gradient norm;
- reconstruction errors;
- loss divergence;
- failed runs/seeds.

---

## 16. Statistical protocol

For headline WER/memory conclusions:

- use at least 3 independent seeds where compute permits;
- report mean and standard deviation;
- retain all runs, including failures;
- keep data order and augmentation policy controlled between paired systems;
- publish exact configs and commit SHA.

Memory profiling should include warm-up and synchronized CUDA measurement.

---

## 17. Success conditions

A practical success could be declared when the proposed model shows all of the following:

1. **substantial measured peak-memory reduction** against standard RoPE-Conformer at matched architecture;
2. **competitive recognition quality** at matched model/compute budget;
3. **a superior iso-memory frontier**, such as a larger useful batch or depth;
4. stable reconstruction/gradients in the selected mixed precision;
5. recomputation overhead small enough that checkpointing does not dominate the Pareto frontier.

No numeric threshold is hard-coded as a result before experiments are run.

---

## 18. Failure conditions

The hypothesis should be considered weakened or rejected if:

- checkpointing achieves equal memory with materially better throughput;
- numerical reconstruction error causes instability at useful depth;
- larger reversible depth does not improve ASR quality;
- attention workspaces dominate memory so strongly that depth savings are negligible;
- WER degradation persists after fair tuning;
- complexity of the reversible implementation materially harms reproducibility.

Negative findings are valid research outcomes.

---

## 19. Result-table template

Do not fill this table without logs.

| Model | Layers | Params | Precision | Peak VRAM | audio-sec/s | test-clean WER | test-other WER | Rec. error |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Conformer RelPos | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a |
| RoPE-Conformer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a |
| Rev-Conformer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Rev‑RoConformer** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RoPE-Conformer + checkpointing | TBD | TBD | TBD | TBD | TBD | TBD | TBD | n/a |

---

## 20. Most important plots

1. **Peak VRAM vs encoder depth**
2. **WER vs peak VRAM**
3. **Throughput vs peak VRAM**
4. **WER vs GPU-hours**
5. **Reconstruction error vs depth**
6. **Peak VRAM vs acoustic sequence length**

Together these reveal whether the architecture actually changes the useful ASR scaling frontier.
