import torch
import argparse
from edge_sam import sam_model_registry
from edge_sam.utils.coreml import SamCoreMLModel
import onnx, onnxsim


parser = argparse.ArgumentParser(
    description="Export the EdgeSAM to ONNX models."
)

parser.add_argument(
    "checkpoint", type=str, help="The path to the EdgeSAM model checkpoint."
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--decoder",
    action="store_true",
    help="If set, export decoder, otherwise export encoder",
)

parser.add_argument(
    "--simplify",
    action="store_true",
    help="If set, run onnxsim",
)

parser.add_argument(
    "--upsample",
    action="store_true",
    help="If set, upsample output masks",
)

def export_encoder_to_onnx(sam, args):
    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    sam.forward = sam.forward_dummy_encoder

    traced_model = torch.jit.trace(sam, image_input)

    # Define the input names and output names
    input_names = ["image"]
    output_names = ["image_embeddings"]

    # Export the encoder model to ONNX format
    onnx_encoder_filename = args.checkpoint.replace('.pth', '_encoder.onnx')
    torch.onnx.export(
        traced_model,
        image_input,
        onnx_encoder_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,  # Use an appropriate ONNX opset version
        verbose=False
    )

    print(f"Exported ONNX encoder model to {onnx_encoder_filename}")

    return onnx_encoder_filename


def export_decoder_to_onnx(sam, args):
    sam_decoder = SamCoreMLModel(
        model=sam,
        use_stability_score=args.use_stability_score,
        upsample_masks=args.upsample,
    )
    sam_decoder.eval()

    if args.gelu_approximate:
        for n, m in sam.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    image_embeddings = torch.randn(1, embed_dim, *embed_size, dtype=torch.float)
    point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float)
    # can only support one box at a time
    boxes = torch.randint(low=0, high=4, size=(1, 1, 4), dtype=torch.float)

    point_valid = torch.ones((1, 1), dtype=torch.bool)
    boxes_valid = torch.ones((1, 1), dtype=torch.bool)

    # Define the input names and output names
    input_names = ["image_embeddings", "point_coords", "point_labels", "boxes", "point_valid", "boxes_valid"]
    output_names = ["scores", "masks"]

    # Export the decoder model to ONNX format
    onnx_decoder_filename = args.checkpoint.replace('.pth', '_decoder.onnx')
    torch.onnx.export(
        sam_decoder,
        (image_embeddings, point_coords, point_labels, boxes, point_valid, boxes_valid),
        onnx_decoder_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,  # Use an appropriate ONNX opset version
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
            # "boxes": {1: "num_boxes"},
        },
        verbose=False
    )

    print(f"Exported ONNX decoder model to {onnx_decoder_filename}")

    return onnx_decoder_filename

def simplify(f):
    model_onnx = onnx.load(f)  # load onnx model
    # onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if simplify:
        try:
            model_onnx, check = onnxsim.simplify(model_onnx) # , perform_optimization=False)
            assert check, 'Simplified ONNX model could not be validated'
        except Exception as e:
            print("onnxsim failure")
            exit()

    print("onnxsim success")
    onnx.save(model_onnx, f)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading model...")
    sam = sam_model_registry["edge_sam_dyt"](checkpoint=args.checkpoint, upsample_mode="bilinear")
    sam.eval()
    # breakpoint()
    if args.decoder:
        f = export_decoder_to_onnx(sam, args)
    else:
        f = export_encoder_to_onnx(sam, args)

    if args.simplify:
        simplify(f)