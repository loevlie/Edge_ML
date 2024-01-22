import argparse
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import groundingdino.datasets.transforms as T
from groundingdino.util import get_tokenlizer
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
from groundingdino.util.slconfig import SLConfig
# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # max_batch_size=1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes) # # nbytes表示数组中的所有数据消耗掉的字节数
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, stream, bindings

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference. batch_size = 1
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([[1200, 800]]), # w, h,  max_size=1333
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    image_data = image.numpy().astype(np.float32).ravel()
    return image_pil, image_data

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def sig(x):
    return 1 / (1 + np.exp(-x))

def process_tensorrt_outputs(output_logits, output_boxes, text_prompt, tokenizer, box_threshold, text_threshold):

    # Filter outputs
    filt_mask = output_logits.max(dim=1)[0] > box_threshold
    logits_filt = output_logits[filt_mask]
    boxes_filt = output_boxes[filt_mask]

    # Generate phrases
    tokenized = tokenizer(text_prompt)
    pred_phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer) for logit in logits_filt]

    return boxes_filt, pred_phrases


def get_grounding_output(logits, boxes, image, caption, tokenizer, box_threshold, text_threshold, token_spans):
    # Assuming logits and boxes are already processed by TensorRT and reshaped to expected format
    if token_spans is None:
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        # Tokenize caption
        tokenized = tokenizer(caption)

        # Build prediction phrases
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
    else:
        # Given-phrase mode
        positive_maps = create_positive_map_from_span(
            tokenizer(caption),
            token_span=token_spans
        )

        logits_for_phrases = positive_maps @ logits.T
        all_boxes = []
        all_phrases = []
        for token_span, logit_phr in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            all_phrases.extend([phrase for _ in range(filt_mask.sum())])
        boxes_filt = torch.cat(all_boxes, dim=0)
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases

# def prepare_text_inputs(tokenizer, imgae, text_prompt, specical_tokens, max_text_len, max_seq_len=256):  

#     # Generate masks and IDs
#     (
#         text_self_attention_masks,
#         position_ids,
#         cate_to_token_mask_list,
#     )  = generate_masks_with_special_tokens_and_transfer_map(
#         tokenized, specical_tokens, tokenizer)

#     # Ensuring the shape doesn't exceed the model's max length
#     if text_self_attention_masks.shape[1] > max_text_len:
#         text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
#         position_ids = position_ids[:, :max_text_len]
#         tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
#         tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
#         tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

#     inputs = {}
#     # input_img = np.expand_dims(image, 0)
#     # inputs["img"] = input_img
#     inputs["input_ids"] = tokenized["input_ids"]
#     inputs["attention_mask"] = tokenized["attention_mask"]
#     inputs["token_type_ids"] = tokenized["token_type_ids"]
#     inputs["position_ids"] = position_ids
#     inputs["text_token_mask"] = text_self_attention_masks 

#     return inputs
def prepare_text_inputs(tokenizer, text_prompt, special_tokens, max_text_len):
    tokenized = tokenizer([text_prompt], padding="longest", return_tensors="pt")
    
    # Generate masks and IDs
    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens, tokenizer)

    # Ensure shapes don't exceed the model's max length
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "token_type_ids": tokenized["token_type_ids"],
        "position_ids": position_ids,
        "text_token_mask": text_self_attention_masks
    }


# Main execution
if __name__ == "__main__":
    import time
    from tqdm import tqdm
    parser = argparse.ArgumentParser("TensorRT Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--device", "-d",  type=str, default="CPU", help="set device, default: CPU")
    parser.add_argument("--tensorrt_path", "-tp", type=str, default="grounded_dino_swinT.trt", required=True, help="path and name of the converted TensorRT runtime file to save")
    args = parser.parse_args()
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    device = args.device
    tensorrt_path = args.tensorrt_path

    config_args = SLConfig.fromfile(args.config_file)
    text_encoder_type = config_args.text_encoder_type
    tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
    model_max_text_len = config_args.max_text_len  # Assuming this is defined in your config

    caption = text_prompt.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
        
    captions = [caption]

    trt.init_libnvinfer_plugins(None, "")

    # Initialize and load model
    trt_engine = load_engine(trt.Runtime(TRT_LOGGER), tensorrt_path)
    if trt_engine is None:
        sys.exit(1)

    with trt_engine.create_execution_context() as context:
        h_buffers, d_buffers, stream, bindings = allocate_buffers(trt_engine)

        # Preprocess image
        image_pil, image = load_image(image_path)
        input_img = np.expand_dims(image, 0)
        tokenized = tokenizer(captions, padding="longest", return_tensors="pt")
        # Prepare text inputs (once)
        special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        inputs = prepare_text_inputs(
            tokenizer, args.text_prompt, special_tokens, max_text_len=config_args.max_text_len)

        
        
        np.copyto(h_buffers[1].host, inputs["input_ids"].numpy().ravel())  # Input IDs
        np.copyto(h_buffers[2].host, inputs["attention_mask"].numpy().astype(np.bool_).ravel())  # Attention mask
        np.copyto(h_buffers[3].host, inputs["token_type_ids"].numpy().ravel())  # Token type IDs
        np.copyto(h_buffers[4].host, inputs["position_ids"].numpy().ravel())  # Position IDs
        np.copyto(h_buffers[5].host, inputs["text_token_mask"].numpy().astype(np.bool_).ravel())  # Text token mask
        # Run inference
        start_time = time.time()
        iterations = 50  # You can adjust the number of iterations

        for _ in tqdm(range(iterations)):
            # Preprocess image
            image_pil, image = load_image(image_path)
            input_img = np.expand_dims(image, 0)
            # Copy data to buffers
            np.copyto(h_buffers[0].host, input_img)  # Image data
            
            output_buffers = do_inference(context, bindings, h_buffers, d_buffers, stream, batch_size=1)
        
        time_it_took = (time.time() - start_time)/iterations
        print(f"time taken for inference (averaged): {time_it_took}")
        # Process outputs
        nq = 900
        logits = output_buffers[0].reshape(nq, 256) 
        logits = sig(logits)
        boxes = output_buffers[1].reshape(nq, 4)
        logits = torch.from_numpy(logits)
        boxes = torch.from_numpy(boxes) 
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        print("\n\nPRED_PHRASES: \n\n")
        print(pred_phrases)
        print("\n\n")
        
        # Visualization
        pred_dict = {
            "boxes": boxes_filt,
            "size": [image_pil.size[1], image_pil.size[0]],
            "labels": pred_phrases,
        }
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    
        # make dir
        os.makedirs(output_dir, exist_ok=True)
        image_with_box.save(os.path.join(output_dir, "pred_4.jpg"))

        end_time = time.time()
        print(f"Average Inference Time (TensorRT): {(end_time - start_time):.4f} seconds")