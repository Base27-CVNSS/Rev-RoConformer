<div align="center">

# 🎙️ Rev‑RoConformer

### *Reversible Depth × Rotary Geometry × Conformer Acoustics for Speech Recognition*

**A research preprint and open-source architecture proposal for memory-efficient deep Automatic Speech Recognition (ASR).**

[![Status](https://img.shields.io/badge/status-preprint%20%7C%20research%20proposal-6f42c1?style=for-the-badge)](#-research-status)
[![Literature cutoff](https://img.shields.io/badge/literature%20cutoff-2026--07--25-0a7ea4?style=for-the-badge)](RESEARCH_GAP.md)
[![Languages](https://img.shields.io/badge/paper-6%20languages-2ea44f?style=for-the-badge)](paper/index.html)
[![License](https://img.shields.io/badge/license-Apache--2.0-orange?style=for-the-badge)](LICENSE)

**English · Tiếng Việt · 简体中文 · Русский · Español · Français**

[📄 Read the multilingual preprint](paper/index.html) · [🔎 Research gap](RESEARCH_GAP.md) · [🧪 Benchmark protocol](EXPERIMENTS.md) · [📚 References](paper/references.bib)

</div>

---

## 💎 The idea

Rev‑RoConformer asks a focused research question:

> **Can a Conformer ASR encoder retain global self-attention, rotary positional geometry and local convolution while making training activation memory substantially less dependent on network depth through reversible additive coupling?**

The architectural signature is:

\[
\boxed{T^2 \;\oplus\; e^{it\Theta} \;\oplus\; \mathcal{C}_{local} \;\oplus\; R^{-1}R=I}
\]

It is intentionally symbolic rather than a single algebraic equality:

| Symbol | Meaning | What it does **not** claim |
|---|---|---|
| \(T^2\) | Full/global self-attention over acoustic frames | It does **not** mean information is lossless |
| \(e^{it\Theta}\) | RoPE as a rotation of query/key subspaces | It does **not** make attention linear-time |
| \(\mathcal{C}_{local}\) | Conformer-style local acoustic convolution | It does **not** replace long-range modeling |
| \(R^{-1}R=I\) | Reversible additive-coupling block | It does **not** imply \(R^2=I\) or that each submodule is involutory |

The project separates two efficiency problems that are often conflated:

\[
\boxed{\text{sequence-length bottleneck} \neq \text{depth-activation bottleneck}}
\]

Global attention remains quadratic in sequence length. Reversibility targets activation storage that otherwise grows with encoder depth.

---

## 🧭 Research status

> **Preprint status:** architecture proposal + theoretical efficiency analysis + reproducible experimental protocol.  
> **Empirical status:** WER, throughput and real GPU-memory results are **not yet reported** and must not be inferred from this repository.

This distinction is central to the project. We do **not** publish fabricated benchmark numbers.

### Literature-gap statement — cutoff 25 July 2026

A public-source search was performed using combinations of:

- `reversible Conformer automatic speech recognition`
- `reversible Conformer RoPE ASR`
- `rotary position embedding reversible Conformer speech`
- `additive coupling Conformer speech recognition`
- `Rev-RoConformer`

The survey included public results indexed through arXiv, ISCA Archive, ACL Anthology, GitHub and general scholarly/web search. We found prior work for **each ingredient separately**, including Conformer, RoPE-Conformer for ASR, reversible residual networks/Reformer, very deep sparse Conformers and efficient ASR encoders. We did **not identify a public work matching the exact proposed combination** of:

> **ASR Conformer local convolution + RoPE global attention + RevNet/Reformer-style additive-coupling reversible depth for activation reconstruction.**

This is a **search result, not a proof of non-existence**. A paper submission should phrase novelty as *“to the best of our public literature search up to 2026-07-25”*, never as an absolute claim that nobody has ever attempted it. See [RESEARCH_GAP.md](RESEARCH_GAP.md) for the audit trail.

---

## 🧠 Why this architecture is interesting

Modern ASR encoders need both:

- **global linguistic/contextual relationships**, and
- **local acoustic structure** such as phonetic transitions and nearby spectral patterns.

Conformer established a strong architecture by combining self-attention and convolution. RoPE gives attention a rotation-based positional geometry and has already been evaluated specifically for ASR. RevNet and Reformer showed that suitable residual structures can reconstruct intermediate activations during backward computation rather than storing every layer's state.

Rev‑RoConformer studies what happens when those ideas are combined **for depth scaling in ASR**.

---

## 🧩 Architecture

```mermaid
flowchart TB
    A[🎤 16 kHz waveform] --> B[Log-Mel / learned acoustic frontend]
    B --> C[Conv subsampling ×4 or ×8]
    C --> D{Split channels}
    D --> X1[x₁]
    D --> X2[x₂]

    X2 --> F[🌀 F: Norm → RoPE Global MHSA → FFN]
    X1 --> Y1[➕ y₁ = x₁ + F(x₂)]
    F --> Y1

    Y1 --> G[🎚️ G: Norm → Conformer Local Conv → FFN]
    X2 --> Y2[➕ y₂ = x₂ + G(y₁)]
    G --> Y2

    Y1 --> CAT[Concatenate]
    Y2 --> CAT
    CAT --> R[♻️ Repeat Rev-RoConformer × L]
    R --> H[Encoder representation]
    H --> CTC[CTC head]
    H --> DEC[Attention / Transducer decoder]
    CTC --> T[📝 Transcript]
    DEC --> T
```

The additive coupling block is

\[
y_1=x_1+F(x_2), \qquad y_2=x_2+G(y_1).
\]

Its input is reconstructed exactly in real arithmetic by

\[
x_2=y_2-G(y_1), \qquad x_1=y_1-F(x_2).
\]

Therefore

\[
R^{-1}(R(x))=x,
\]

without requiring \(F\), \(G\), attention or convolution to be individually invertible.

---

## 📐 What can already be proved

### Proposition 1 — depth-dependent activation storage

For a conventional stack of \(L\) encoder blocks, storing layer activations for backpropagation contributes approximately

\[
M_{act}=\mathcal O(LBTd),
\]

where \(B\) is batch size, \(T\) is sequence length and \(d\) is hidden width.

For an ideal reversible stack where intermediate layer inputs are reconstructed from later outputs, the **depth-dependent part** can approach

\[
M_{act,rev}=\mathcal O(BTd),
\]

plus transient workspaces, non-reversible boundaries, attention kernels, decoder states, optimizer states and implementation overhead.

This is the central theoretical efficiency claim of Rev‑RoConformer.

### Proposition 2 — RoPE is norm-preserving

Each two-dimensional RoPE rotation uses an orthogonal matrix

\[
R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix},
\qquad R(\phi)^TR(\phi)=I.
\]

Hence

\[
\|R(\phi)z\|_2=\|z\|_2.
\]

RoPE injects positional phase without inherently magnifying the query/key vector norm.

### Proposition 3 — reversibility does not remove quadratic attention compute

For full self-attention,

\[
QK^T \in \mathbb R^{T\times T}
\]

and the dominant arithmetic remains approximately

\[
\mathcal O(T^2d).
\]

FlashAttention can substantially improve exact-attention memory traffic and practical memory usage, but reversible depth does not turn global attention into a linear-time algorithm.

---

## ⚡ What must be demonstrated experimentally

The following are **hypotheses**, not existing results:

- lower peak training VRAM at matched architecture size;
- comparable WER/CER to a non-reversible RoPE-Conformer;
- larger trainable depth or batch size under the same GPU-memory budget;
- acceptable backward recomputation overhead;
- numerically stable reconstruction in BF16/FP32 at 24–96 layers;
- a favorable WER-versus-memory frontier.

The strongest paper would not merely claim “faster” or “better”. It would establish **when** reversibility helps and identify the point where sequence-length attention cost becomes dominant over depth activation cost.

---

## 🔬 Core research questions

1. **Memory:** How much peak training memory is saved at matched depth, width, batch and sequence length?
2. **Accuracy:** Does reversible coupling preserve ASR quality relative to equivalent Conformer/RoPE-Conformer baselines?
3. **Depth scaling:** Does memory saved by reversibility translate into useful additional depth?
4. **Numerics:** How does reconstruction error accumulate in FP32, BF16 and FP16?
5. **Throughput:** What recomputation penalty is paid per unit of memory saved?
6. **Long speech:** At what sequence length does global-attention compute dominate the memory benefits of reversible depth?

---

## 🧪 Required baselines

| ID | Encoder | Position encoding | Reversible depth |
|---|---|---|---|
| B0 | Conformer | Relative position | No |
| B1 | Conformer | RoPE | No |
| B2 | Rev-Conformer | Relative position | Yes |
| **P** | **Rev‑RoConformer** | **RoPE** | **Yes** |
| Ckpt | RoPE-Conformer | RoPE | Gradient checkpointing |

A serious evaluation must include **gradient checkpointing** because it is the most direct competing memory-for-compute strategy.

See [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 🎯 Evaluation modes

### Iso-model
Hold architecture, batch and sequence configuration fixed. Measure the direct cost/benefit of reversible computation.

### Iso-memory
Give all models the same GPU-memory budget. Measure maximum useful depth/batch size and downstream WER.

### Iso-compute
Match training compute or GPU-hours as closely as possible. Measure quality at a comparable computational budget.

Primary metrics:

`WER · CER · peak VRAM · activation memory · audio-sec/s · samples/s · GPU-hours · reconstruction error · max stable depth`

---

## 🌐 Six-language preprint

The complete HTML preprint is published in six editions:

| Language | File |
|---|---|
| 🇬🇧 English | [paper/en.html](paper/en.html) |
| 🇻🇳 Tiếng Việt | [paper/vi.html](paper/vi.html) |
| 🇨🇳 简体中文 | [paper/zh-CN.html](paper/zh-CN.html) |
| 🇷🇺 Русский | [paper/ru.html](paper/ru.html) |
| 🇪🇸 Español | [paper/es.html](paper/es.html) |
| 🇫🇷 Français | [paper/fr.html](paper/fr.html) |

The request named five languages while asking for six; **French is included as the sixth edition**.

Open [paper/index.html](paper/index.html) for the language switcher.

---

## 🗺️ Research roadmap

- **Phase 0 — Preprint & literature audit:** architecture, equations, falsifiable claims, public gap statement.
- **Phase 1 — Reversible core:** inverse tests, gradient equivalence, deterministic stochastic-state handling.
- **Phase 2 — Synthetic depth tests:** FP32/BF16 reconstruction through 12/24/48/72/96 blocks.
- **Phase 3 — LibriSpeech train-clean-100:** fast ablations and profiling.
- **Phase 4 — LibriSpeech 960 h:** principal benchmark.
- **Phase 5 — Cross-domain/multilingual:** Common Voice, AISHELL-1 and/or VoxPopuli.
- **Phase 6 — Sequence efficiency:** evaluate FlashAttention, subsampling ×8 and optional local/sparse attention as separate variables.

---

## 🧬 Evidence map

| Statement | Status |
|---|---|
| Conformer combines global attention and local convolution effectively for ASR | ✅ Established literature |
| RoPE has been evaluated successfully for ASR | ✅ Established literature |
| Reversible additive coupling can reconstruct activations for backward computation | ✅ Established literature |
| Very deep Conformers up to 100 encoder layers have been studied | ✅ Established literature |
| Exact Rev‑RoConformer combination identified in public literature before cutoff | 🔎 **Not found in our search** |
| Rev‑RoConformer reduces depth-dependent activation storage asymptotically | 📐 Derived from reversible coupling assumptions |
| Rev‑RoConformer improves WER | 🧪 Unproven — benchmark required |
| Rev‑RoConformer trains faster wall-clock | 🧪 Unproven; recomputation may make it slower |
| Rev‑RoConformer improves inference memory | ❌ Not a core claim |

---

## 📚 Anchor literature

- Gulati et al., **Conformer: Convolution-augmented Transformer for Speech Recognition** (2020).
- Gomez et al., **The Reversible Residual Network: Backpropagation Without Storing Activations** (2017).
- Kitaev et al., **Reformer: The Efficient Transformer** (2020).
- Su et al., **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021).
- Li, Xu & Zhang, **Conformer-based End-to-end Speech Recognition With Rotary Position Embedding** (2021).
- Wu, **Deep Sparse Conformer for Speech Recognition** (Interspeech 2022).
- Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022).
- Yao et al., **Zipformer: A Faster and Better Encoder for Automatic Speech Recognition** (ICLR 2024).
- Zhang et al., **Benchmarking Rotary Position Embeddings for Automatic Speech Recognition** (2025).

Machine-readable citations are in [paper/references.bib](paper/references.bib).

---

## 🛡️ Scientific integrity

This repository follows four rules:

1. **No invented benchmarks.** Tables remain marked `TBD` until experiments are run.
2. **No absolute novelty claim.** “Not found in our search” is not “mathematically proven never to exist”.
3. **No conflation of memory and compute.** Reversible depth, FlashAttention and sparse/local attention address different bottlenecks.
4. **Negative results are publishable.** If reversible ASR loses to checkpointing, the boundary itself is scientifically valuable.

---

## 🤝 Contributing

Contributions are welcome in literature review, reversible-autograd implementation, ASR recipes, numerical-stability testing, profiling and independent reproduction.

Before proposing a performance claim, include:

- hardware and software versions;
- exact configuration and seed;
- peak-memory measurement method;
- WER/CER decoding settings;
- baseline with comparable parameter/compute budget;
- raw logs or reproducible scripts.

---

## 📜 License

Repository code and project documentation are released under the **Apache License 2.0**, unless a subdirectory explicitly states otherwise. Third-party datasets, models and cited works retain their original licenses.

---

## ✨ Research philosophy

> **RoPE organizes relative geometry. Conformer organizes local and global acoustic structure. Reversible computation organizes depth. Efficient attention organizes sequence cost.**

Rev‑RoConformer is not presented as a solved ASR engine. It is a falsifiable research architecture designed to discover whether **memory-efficient depth scaling** is a useful missing axis in modern speech recognition.

<div align="center">

### ♻️ `Global context ⊕ Rotary geometry ⊕ Local acoustics ⊕ Reversible depth`

**Base27-CVNSS · Rev‑RoConformer Preprint Project · Literature cutoff: 2026‑07‑25**

</div>
