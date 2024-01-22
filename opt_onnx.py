from onnxsim import simplify
import onnxoptimizer as optimizer
import onnx

# Load your model
input_onnx = "grounded_quantized_real.onnx"
output_onnx = "onnx_files/grounded_simple.onnx"
opt_onnx = "onnx_files/grounded_opt_fused.onnx"

# Simplify
model_simplified, check = simplify(input_onnx)

# Save the simplified model
assert check, "Simplified ONNX model could not be validated"
model = model_simplified

# List of optimization passes
passes = ["eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout", "eliminate_nop_pad", "eliminate_nop_transpose", "eliminate_unused_initializer", "fuse_add_bias_into_conv", "fuse_bn_into_conv", "fuse_consecutive_concats", "fuse_consecutive_reduce_unsqueeze", "fuse_matmul_add_bias_into_gemm", "fuse_pad_into_conv", "fuse_transpose_into_gemm"]

# model = onnx.load(output_onnx)

# Apply the optimizations
optimized_model = optimizer.optimize(model, passes)

# Save the optimized model
onnx.save(optimized_model, opt_onnx)