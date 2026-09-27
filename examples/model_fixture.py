#!/usr/bin/env python3
"""Regenerate the tiny hand-weighted ONNX integration fixture (not a speech model).

Run: uv run --with onnx==1.20.1 examples/model_fixture.py
ONNX is needed only to regenerate files; native Rust inference uses no Python.
"""

import json
from pathlib import Path
import struct
import wave

import onnx
from onnx import TensorProto, helper


def main():
    destination = Path(__file__).with_name("model-fixture")
    destination.mkdir(exist_ok=True)
    samples = 320
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["audio"], ["mean"], axes=[1], keepdims=1),
            helper.make_node("MatMul", ["mean", "weights"], ["logits"]),
        ],
        "nelly-harness-integration-fixture",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, samples])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor("weights", TensorProto.FLOAT, [1, 2], [-10.0, 10.0])],
    )
    model = helper.make_model(graph, producer_name="nelly-harness-fixture", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, destination / "weights.onnx")
    manifest = {
        "model_path": "weights.onnx",
        "sample_rate_hz": 16000,
        "chunk_samples": samples,
        "input_name": "audio",
        "output_name": "logits",
        "intra_threads": 1,
        "min_confidence": 0.8,
        "emit_on_change": True,
        "actions": [None, {"event": "prefetch", "call": {"tool": "notes_get", "topic": "demo", "name": "welcome"}}],
    }
    (destination / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Negative frame -> no-op; positive frame -> prefetch; positive frame again
    # -> duplicate suppressed. These are artificial DC signals, not speech.
    with wave.open(str(destination / "audio.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<" + "h" * (samples * 3), *([-24575] * samples + [24575] * (samples * 2))))
    print(destination)


if __name__ == "__main__":
    main()
