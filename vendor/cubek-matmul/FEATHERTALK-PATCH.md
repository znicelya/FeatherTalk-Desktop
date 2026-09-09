# Cubek 0.2.0: double-buffered K bounds

This is the published `cubek-matmul` 0.2.0 source, licensed under
`MIT OR Apache-2.0` as declared in its original Cargo manifest. Upstream:
<https://github.com/tracel-ai/cubek>, commit
`6b19493ee423554b263fb0c1ad7b7b5a0d96168e`, `crates/cubek-matmul`.
No dependency versions are upgraded. Cargo registry bookkeeping and the
package's standalone Cargo.lock are omitted.

The five modified files are `src/routines/double_unit.rs`,
`double_buffering.rs`, `ordered_double_buffering.rs`, `specialized.rs`, and
the DoubleVecMat routine in `vecmat_innerproduct.rs`.
Their `prepare` functions enable K bounds checks when K is not divisible by
**two** stages, preserving an explicitly enabled check in a forced blueprint.

Upstream's shared blueprint builder checks divisibility by one stage. These
double-buffered execution loops round the number of stages up to an even
number. For K=24 and a stage width of 8, they execute four stages while bounds
checks are disabled. The last eight products can therefore read the following
input row and output channel. SCRFD's 24-channel pointwise convolution exposed
this on Vulkan/SPIR-V. Single-stage routines keep their original predicate.

The worker regression `tests/wgpu_matmul.rs` forces the affected DoubleUnit
kernel with row-major left and column-major right operands of ones. Before
this correction, K=24 produces 32 instead of 24. It also covers a partial
stage and an even-stage control. A second test forces DoubleVecMat on two
384-element all-one inner products; the old implementation reads the next
batch and produces `[512, 384]` instead of `[384, 384]`.
The existing GPU SCRFD/PFLD fixture test
checks the complete face pipeline against its original pixel tolerances.

Remove this Cargo patch when an upstream version includes the equivalent
correction and both regressions pass with the replacement.
