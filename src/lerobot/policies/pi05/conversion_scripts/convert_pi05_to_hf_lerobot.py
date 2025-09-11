import argparse
import os

import safetensors.torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def convert_pi0_checkpoint(checkpoint_dir: str, precision: str, tokenizer_id: str, output_path: str):
    openpi_model = safetensors.torch.load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    # openpi_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))

    # add prefix model. to all keys
    openpi_model = {f"model.{k}": v for k, v in openpi_model.items()}

    # # Transform keys to match LeRobot naming conventions.
    # transformations = [
    #     (
    #         re.compile(r"\.paligemma_with_expert\.paligemma\.lm_head"),
    #         ".paligemma_with_expert.paligemma.language_model.lm_head",
    #     ),
    #     (
    #         re.compile(r"\.paligemma_with_expert\.paligemma\.model\.language_model"),
    #         ".paligemma_with_expert.paligemma.language_model.model",
    #     ),
    #     (
    #         re.compile(r"\.paligemma_with_expert\.paligemma\.model\.vision_tower"),
    #         ".paligemma_with_expert.paligemma.vision_tower",
    #     ),
    #     (
    #         re.compile(
    #             r"\.paligemma_with_expert\.paligemma\.model\.multi_modal_projector"
    #         ),
    #         ".paligemma_with_expert.paligemma.multi_modal_projector",
    #     ),
    # ]

    # for pattern, replacement in transformations:
    #     openpi_model = {pattern.sub(replacement, k): v for k, v in openpi_model.items()}

    # Handle tied weights: lm_head.weight and embed_tokens.weight share memory
    lm_head_key = None
    embed_tokens_key = None
    for key in openpi_model.keys():
        if key.endswith(".paligemma_with_expert.paligemma.lm_head.weight"):
            lm_head_key = key
        elif key.endswith(".paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"):
            embed_tokens_key = key
        if lm_head_key and embed_tokens_key:
            break

    if lm_head_key and not embed_tokens_key:
        embed_tokens_key = lm_head_key.replace(".lm_head.weight", ".model.language_model.embed_tokens.weight")
        openpi_model[embed_tokens_key] = openpi_model[lm_head_key]
    elif embed_tokens_key and not lm_head_key:
        lm_head_key = embed_tokens_key.replace(".model.language_model.embed_tokens.weight", ".lm_head.weight")
        openpi_model[lm_head_key] = openpi_model[embed_tokens_key]

    if "pi05_base" in checkpoint_dir:
        pi05_config = PI05Config(
            empty_cameras=0,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
        )
    else:
        raise ValueError()

    pi05_model = PI05Policy(pi05_config)
    status = pi05_model.load_state_dict(openpi_model, strict=False)
    print(f"Missing keys: {status.missing_keys}")
    print("\n" * 4)
    print(f"Unexpected keys: {status.unexpected_keys}")
    assert len(status.unexpected_keys) == 0
    assert len(status.missing_keys) == 0

    pi05_model.save_pretrained(output_path, safe_serialization=True)

    # del pi05_model
    # PI05Policy.from_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        default="/raid/pablo/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params",
        type=str,
        help="Path to the ocdbt checkpoint",
    )

    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        type=str,
        help="Precision identifier for model conversion - should match the base checkpoint precision.",
    )
    # tokenizer is identical to paligemma, it appears

    parser.add_argument(
        "--tokenizer_hub_id",
        default="google/paligemma-3b-pt-224",
        type=str,
        help="Hub path to the tokenizer to save",
    )

    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save converted weights to",
    )

    args = parser.parse_args()
    convert_pi0_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        precision=args.precision,
        tokenizer_id=args.tokenizer_hub_id,
        output_path=args.output_path,
    )
