# FeatherTalk patch to onnx-ir 0.21.0

This directory is the published `onnx-ir` crate version `0.21.0`, vendored
solely for deterministic SCRFD artifact generation.

The upstream parser incorrectly rejects a nonzero `AveragePool.ceil_mode`
before opset 19. ONNX's official historical schema defines that attribute in
AveragePool opset 11, and the tracked SCRFD opset 12 graph uses it. FeatherTalk
therefore changes the 2-D AveragePool validation threshold from 19 to 11 and
updates the associated comments and tests. No other production code is
modified.

Upstream package: <https://crates.io/crates/onnx-ir/0.21.0>

Official schema source:
<https://github.com/onnx/onnx/blob/main/onnx/defs/nn/old.cc>
