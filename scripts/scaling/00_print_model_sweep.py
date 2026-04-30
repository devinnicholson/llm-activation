#!/usr/bin/env python
from __future__ import annotations

from scratch_llm.scaling.configs import MODEL_SWEEP


def main() -> None:
    for config in MODEL_SWEEP:
        config.validate()
        print(
            f"{config.name}: params~{config.parameter_estimate:,} "
            f"layers={config.num_layers} hidden={config.hidden_size} "
            f"heads={config.num_attention_heads} d_ff={config.intermediate_size}"
        )


if __name__ == "__main__":
    main()
