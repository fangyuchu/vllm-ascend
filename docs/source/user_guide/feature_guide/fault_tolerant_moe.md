# Fault-tolerant MoE on Ascend A3

The v2 model runner supports fault-tolerance `scale_down` with either
MoeDistribute V3 or CANN MegaMoe. Choose the backend when starting every worker;
changing environment variables on an already running worker does not switch
its loaded weights, communication buffers, or captured graphs.

| Backend | Environment | Additional configuration |
| --- | --- | --- |
| MoeDistribute V3 | `VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3=1` | `{"enable_fused_mc2": 0}` |
| CANN MegaMoe | `VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3=0` | `{"enable_fused_mc2": 2}` |

V3 retains priority if its environment switch is enabled. Mode `1` is the
legacy fused operator, which does not support fault-tolerance scale-down.
This branch normalizes mode `2` to `1` internally after enabling CANN MegaMoe;
use the values in the table on the command line.

Both backends require `VLLM_USE_V2_MODEL_RUNNER=1`, `--enable-expert-parallel`,
`--enable-fault-tolerance`, `--enable-eplb`, and enough redundant expert slots
to host every logical expert on the surviving ranks. Disable hierarchical
MC2 communication. All workers must use the same backend and compatible packages.

## MegaMoe requirements

- Ascend A3 and matching CANN operator binaries and `cann_ops_transformer`
  containing [ops-transformer PR #9368](https://gitcode.com/cann/ops-transformer/pull/9368).
  Updating only the Python wrapper does not update the operator ABI.
- Identical EP and MC2 rank sets. Without Elastic EP, their rank order must
  also match. Elastic EP initializes a graph-stable EP-to-MC2 expert-rank
  permutation before capture so middle-rank restoration can preserve MC2 slots.
- A model accepted by the existing MegaMoe dimension and quantization checks.
  The unmodified Qwen3-30B-A3B checkpoint has a 768-wide MoE intermediate
  dimension, which is not aligned to 512. It requires an operator build that
  explicitly supports this shape, such as the local A3 W8A8 768 exception;
  unmodified operators still reject it. Explicit MegaMoe fault-tolerance
  requests fail at startup for unsupported models instead of falling back.
- The existing expert reloaders support unquantized and W8A8 dynamic experts.
  Ordinary MegaMoe support for other quantization formats does not imply
  fault-tolerance reload support for those formats.

The first MegaMoe warmup creates the rank mask before graph capture and
registers its buffer with the fault manager. Missing mask APIs or allocation
errors stop initialization. Ordinary MegaMoe inference with fault tolerance
disabled does not require the new APIs.

## Recovery behavior

The fault manager keeps a CPU record of dead ranks for fault queries, so
querying a hung device is unnecessary. During recovery it updates the same
device mask storage captured by the graph. MegaMoe retains the original
physical expert capacity and ids; EPLB moves lost experts into surviving slots
and reloads their weights and packed scales in place. V3 continues to use its
existing dense expert ids and `elastic_info`.

After device reset, MegaMoe communication flags are cleared before inference
resumes, while cumulative dead-rank masks are preserved. Unmasking a rank
also clears stale communication flags; the caller must first stop outstanding
device work. Kernel masks do not replace device abort/reset when an operation
has already hung.

## Restore expansion with new-rank-only capture

Start **all** workers with `--enable-elastic-ep` in addition to the fault-tolerance
options above. This initializes graph-stable MegaMoe routing state before the
initial capture. The flag cannot be enabled on an already captured worker.
Both synchronous and asynchronous EPLB configurations are supported.

The MegaMoe fast path restores the original physical EP capacity after a fault,
for example 4 -> 3 -> 4. It currently requires W8A8, TP=PP=PCP=1, no LoRA or
speculative decoding, and the matching Python wrapper from
[ops-transformer PR #12345](https://gitcode.com/cann/ops-transformer/pull/12345).
That PR updates `SymmBuffer.context` in place and requires unchanged group size.
Other Elastic EP requests use the existing generic reconfiguration path; they
do not gain this graph-preserving behavior.

During preparation, existing workers drain model/EPLB collectives before
updating their context. Their physical MC2 slots and masks remain unchanged,
so they can continue serving the surviving ranks while new workers capture.
New workers mask existing ranks and route dummy tokens only to new physical
expert slots. Existing workers participate in the separate CPU DP metadata
synchronization protocol, without capturing their models again.

Commit installs bootstrap mappings matching the copied expert weights and
updates the EP-to-MC2 permutation in place. It retains old graphs, quantization
methods and captured tensor storage, attaches the buffers to the new fault
managers, clears communication flags, disables dummy routing and unmasks peers
before resuming EPLB. The same path handles middle and tail rank holes. The
operator's captured communication parameters must remain unchanged; a context
address or parameter change fails the transition.

The log `MegaMoe restore committed; existing graphs preserved` identifies this
path. Unit tests cover its state transitions and routing; full model restore
accuracy and repeated fault cycles still require validation on the target CANN
build. Passing fault scale-down alone does not establish restore correctness.

## Validation

The four-card fault-tolerance test covers removal of both a middle rank and
the last rank. It checks resumed serving and answers after expert reload.
For MegaMoe, set `MODEL_NAME` to a compatible W8A8 checkpoint and run:

```bash
pytest -sv tests/e2e/pull_request/four_card/fault_tolerance/test_fault_tolerance_e2e.py \
  -k 'scale_down and mega_moe'
```

The default Qwen3 model is skipped for MegaMoe because of its intermediate
dimension. A skipped hardware test is not evidence of successful recovery or
accuracy; validate both backends on the target CANN build before deployment.
