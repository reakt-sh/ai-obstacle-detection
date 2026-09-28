"""Patch the pinned Ultralytics TensorRT exporter for JetPack 7."""

from __future__ import annotations

from pathlib import Path


BUILDER_ORIGINAL = """    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
"""

BUILDER_PATCHED = """    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # Level 5 performs the most exhaustive tactic search. On Jetson Orin the
    # resulting FP16 engine is measurably faster while retaining identical
    # layer precision and output types.
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5
    workspace_bytes = int((workspace or 0) * (1 << 30))
"""

ORIGINAL = """    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
"""

PATCHED = """    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # TensorRT 10.16 on Orin has no INT8/FP32 tactic for the final
        # Conv+SiLU block in YOLO segmentation prototype heads. Allow FP16
        # fallback and constrain only that small block to FP16; the remaining
        # graph is still free to use INT8.
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            proto_layers = [
                network.get_layer(i)
                for i in range(network.num_layers)
                if "/proto/cv3/" in network.get_layer(i).name
            ]
            if proto_layers:
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                for layer in proto_layers:
                    layer.precision = trt.float16
                    for output_index in range(layer.num_outputs):
                        layer.set_output_type(output_index, trt.float16)
                LOGGER.info(
                    f"{prefix} forcing {len(proto_layers)} segmentation prototype layers to FP16 "
                    "for TensorRT INT8 compatibility"
                )
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
"""


def apply_patch(source: str) -> str:
    if source.count(BUILDER_ORIGINAL) != 1 or source.count(ORIGINAL) != 1:
        raise RuntimeError("Unexpected Ultralytics TensorRT exporter source; refusing an unsafe patch")
    return source.replace(BUILDER_ORIGINAL, BUILDER_PATCHED).replace(ORIGINAL, PATCHED)


def main() -> None:
    import ultralytics.utils.export.engine as engine

    path = Path(engine.__file__)
    path.write_text(apply_patch(path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Patched Ultralytics TensorRT exporter: {path}")


if __name__ == "__main__":
    main()
