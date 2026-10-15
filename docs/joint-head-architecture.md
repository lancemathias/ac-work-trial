# Joint schema head architecture

Architecture inspected on 2026-10-14 against Clef's `joint_schema_model.py`,
revision `0b331204bb13fbd2ca93a64df1956f5b55478ce5`, SHA-256
`0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
The implementation is shared across Noul, Choice and Score. “Multi-head” here
includes multi-head attention; there are no separate classifiers per task type.
Three baseline seeds are independent copies of this entire head, not an ensemble.

## Dimensions and data flow

Let T be input tokens, F schema fields, K the total allowed options, H backbone
hidden width, and W = 1,024. The random Qwen3.8-27B baseline uses H = 5,120;
the E1 Qwen3.5-9B configuration uses H = 4,096. Both use two routing layers,
four joint field layers, 16 attention heads (64 dimensions per head), FFN width
4,096, and dropout 0 throughout.

| Stage | Shape / layers | Operation |
| --- | --- | --- |
| Frozen backbone | `[B,T,H]` | Last hidden states; native causal/hybrid backbone retained; no gradients or LoRA |
| Input normalization | LayerNorm(H) | Applied to all hidden states before pooling/projection |
| Projections | Six independent bias-free Linear(H,1024) | Memory, question, option-question, global, option-context and option-lexical projections |
| Field type | Embedding(3,1024) | Noul=0, Choice=1, Score=2 |
| Option queries | `[K,1024]` | Sum projected option contextual mean, option lexical mean, and associated question mean |
| Evidence routing | 2 layers | Each option attends over all T projected token states, then residual FFN |
| Option summary | One `[1024]` vector per field | Softmax of option–projected-question dot products / sqrt(1024); weighted option sum; LayerNorm(1024) |
| Initial field vectors | `[F,1024]` | Projected question + normalized option summary + projected final token + type embedding |
| Joint field processing | 4 TransformerDecoderLayer blocks | Field self-attention, cross-attention to token memory, FFN; all with residual connections |
| Final representations | LayerNorm(1024) for fields and routed options | Shared across all field types |
| Residual scorer | Linear(4096,1024) → GELU → Dropout(0) → Linear(1024,1) | Concatenate field, option, elementwise product and absolute difference |
| Output | One logit per allowed option | Lexical prior + gated joint cosine/residual score; softmax within each field |

Question vectors are means over instruction-token spans; option contextual
vectors are means over option JSON spans (including IDs/descriptions). Lexical
vectors average the corresponding rows of the frozen backbone output embedding
matrix. The global vector is the final normalized sequence hidden state.

Each evidence-routing layer applies pre-LayerNorm to query and memory, then
MultiheadAttention(1024,16), residual addition, and a pre-LayerNorm FFN
`1024 → 4096 → 1024` with GELU and residual addition. Each joint field block
uses PyTorch TransformerDecoderLayer with `norm_first=True`, `batch_first=True`,
GELU, the same FFN dimensions, and both self- and cross-attention. No causal
mask is supplied to the head: field self-attention is bidirectional, and routing
and cross-attention see the entire memory. This does not remove the backbone's
causal mask. Standard Linear layers and attention projections have biases except
for the six explicitly bias-free input projections. LayerNorm uses affine scale
and bias with the PyTorch default epsilon 1e-5; GELU uses its default exact form.
There are no additional positional embeddings in this head.

## Exact scoring rule

For field f and option o, with normalized input-space question q, final-token g,
and lexical option vector e:

```
prior = exp(min(prior_logit_scale, log(100))) * cosine(e, q + g)
residual = MLP(concat(field, option, field * option, abs(field - option)))
joint = exp(min(joint_logit_scale, log(100))) * cosine(field, option) + residual
logit = prior + sigmoid(residual_gate) * joint
probabilities = softmax(logits / temperature)
```

All three scalar parameters initialize to zero: both exponential scales start
at 1 and the residual gate starts at 0.5. Other parameters use native PyTorch
constructor initialization. The untrained head therefore retains a pretrained
lexical prior. Confidence for argmax choice is its probability; there is no
separate learned confidence network. Score levels share the same categorical
scorer; an expected numerical score can be derived from their distribution.

The 27B baseline head has 128,056,324 parameters per seed. With H=4,096 the
same constructor has 121,762,820 parameters, derived by reducing six projection
matrices and the input LayerNorm. The backbone and output embedding matrix are
excluded from these counts.

## Precision and training distinction

| Component | Random 27B baseline | E1 9B training implementation |
| --- | --- | --- |
| Frozen backbone weights / outputs | BF16 | BF16; detached under no_grad |
| Head initialization / saved initial weights | CPU FP32 | FP32 |
| Head parameter storage | Cast to GPU BF16 for inference | GPU FP32, including saved trained head |
| Head forward | BF16 parameters/inputs | CUDA BF16 autocast; eligible operations use BF16, other operations follow autocast policies |
| Gradients / optimizer | None; inference mode | FP32 parameter gradients and AdamW moment buffers; no GradScaler |
| Loss | None | Cast logits to FP32 before cross-entropy |
| Probability calculation | Cast logits to FP32 before softmax; T=1 | FP32 logits/softmax for benchmark inference; separate calibration fits positive scalar T |
| Offline dev/calibration statistics | Python metrics from exported probabilities | FP64 logits for NLL, softmax and temperature search |

Autocast does not mean every activation or kernel accumulator is BF16. Exact
internal accumulation follows PyTorch/CUDA kernels; no per-kernel precision
trace was collected. TF32 matmul is disabled in both runners. Dropout is zero
in training and inference. Attention uses SDPA in the frozen backbone; the head
uses PyTorch MultiheadAttention/TransformerDecoderLayer dispatch.

Sources: `scripts/random_joint_baseline.py`, its archived executed runner and
run config under `outputs/baselines-20261014/random-joint-qwen38-27b/`; E1
`train_joint.py`, `configs/e1-qwen35-9b.json` and the vendored Clef source in the
training checkout. Precision differs between these protocols and must remain
part of every run's configuration.
