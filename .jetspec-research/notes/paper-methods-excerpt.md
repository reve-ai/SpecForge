# JetSpec — Methods excerpt (provided by user)

> Verbatim excerpt the user pasted from the JetSpec paper. Kept here because it concisely
> states the training recipe and architecture. Cross-check against
> `../sources/jetspec-paper-arxiv.md` for the authoritative full text.

**Models and Datasets.** We evaluate JetSpec on Qwen3-8B and Qwen3-30B-A3B, covering both
dense and MoE target models. Qwen3 supports both thinking and non-thinking modes, and we use
the non-thinking mode throughout our main evaluation for efficient decoding. For training
data, we curate 780K examples from the Nemotron Post-Training Dataset V2, including all
available coding and math splits, random samples from STEM and chat splits, and 20K
additional examples from CodeAlpaca. For regenerated training sequences, we apply the
corresponding chat template to each data type and continue generation with the target model.
We evaluate on math benchmarks including GSM8K, MATH-500, and AIME25; coding benchmarks
including HumanEval, MBPP, and LiveCodeBench; and open-ended conversational tasks including
MT-Bench. All ablation studies are conducted on the math split of Nemotron Post-Training
Dataset V2. For both DFlash and JetSpec, we perform a learning-rate sweep from 1e-4 to 1e-3
with five settings. We find that 3e-4 and 6e-4 generally perform best, with different tasks
favoring different choices. Since their overall performance is comparable, we report results
with 3e-4 by default. All draft-head training runs are conducted on 8 H100 GPUs with a micro
batch of 2.

For a fair comparison with DFlash, the JetSpec draft head is trained with block size 16,
corresponding to a maximum tree depth of 16, and uses fused hidden features extracted from
the frozen target model. During training, each block keeps the first token as the anchor and
replaces the remaining positions with mask-token embeddings. The causal head predicts all
masked future tokens in parallel, while each position can attend only to the prefix and
earlier positions within the same block, ensuring that the draft distribution follows the
autoregressive order. We sample random anchor positions from each sequence, use up to 512
anchors per example. An example of the attention mask with 3 anchors for 3 blocks is provided
in Figure 5.

JetSpec reuses intermediate representations from the frozen target model as draft-head
context. For Qwen3-8B, we extract hidden states from target layers {1, 9, 17, 25, 33} out of
the 36-layer target model, concatenate them along the channel dimension, and project the
resulting 5d feature back to hidden size d = 4096 through a bias-free linear layer followed by
RMSNorm. The draft head is implemented as a lightweight Qwen3-style decoder with 5 layers, 32
attention heads, 8 KV heads, head dimension 128, and MLP intermediate size 12288. In each
draft layer, the projected target feature is injected as contextual key/value states and
concatenated with the draft-token hidden states, allowing the causal draft head to condition
on rich target-model features while keeping the target model frozen.

For ablations, we consider DFlash-style exponential depth weighting. Since depth weighting can
implicitly bias the bidirectional DFlash head toward left-to-right prediction, in main results
reported in Table 1 and Table 2, we remove it when isolating the effect of causal masking
versus the native block-diffusion head design. For distillation runs, teacher logits are
obtained from the frozen target model and aligned with the student prediction positions; the
draft head is trained with a temperature-scaled soft-label distillation loss.
