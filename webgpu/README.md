# I-3D_web — Custom WebGPU research branch

> **Branch isolation:** Everything under `webgpu/`, the WebGPU-serving Colab viewer changes, and this document are on **`I-3D_web` only**. Do not merge or cherry-pick into `main` without a new explicit user request. This research branch is not the CUDA/Colab production baseline.

The goal is to eventually execute I-3D's image → mesh → PBR texture → skeleton → pose → motion workflow using **browser-native WebGPU**, with original WGSL kernels, streamed model weights, quantization and a portable graph executor. The first implementation is a **working operator layer + WebGPU depth reconstruction viewer and numerical test harness**. It is *not* yet a browser inference port of Hi3D, Hunyuan Paint, TokenRig or MotionBricks.

## Launch in a browser

From the repository root on branch `I-3D_web`:

```bash
git clone --branch I-3D_web https://github.com/Apache0ne/I-3D.git
cd I-3D
python3 -m http.server 8000
# Chrome/Edge: http://localhost:8000/webgpu/
```

Use HTTPS or localhost. Choose **Run custom WGSL correctness tests** for GPU-vs-CPU checks and **Compute GPU surface** for a compute + render pipeline. For real depth maps, upload an 8-bit PNG, then choose **Unproject uploaded depth with WGSL**. This tool treats the PNG as a precomputed depth map; it does not hallucinate geometry or claim to run a depth model.

The Colab notebook from this branch is [here](../notebooks/I-3D_End_to_End_Colab.ipynb). It clones **I-3D_web**, starts the original viewer server, and embeds `/webgpu/` as a browser WebGPU lab. The **WebGPU GPU is your desktop/browser GPU, not the remote Colab CUDA GPU**. Colab model stages remain their original isolated Python backends.

## Original implemented WGSL operators

| Kernel file | Contract | Main target |
|---|---|---|
| `matmul.wgsl` | `A[M,K] @ B[K,N]` tiled f32 | Attention, feed-forward, projections |
| `q4_matvec.wgsl` | Signed 4-bit (two's complement), 8 weights/u32, per-row scales | Quantized model projections |
| `ternary_matvec.wgsl` | Packed 2-bit ternary {-1,0,+1}, 16 weights/u32; **I-3D format, not GGUF** | Low-bit projections |
| `softmax.wgsl` | Stable f32 row-wise softmax | Attention |
| `layernorm.wgsl` | Normalization with gamma, beta, epsilon | Transformers |
| `rmsnorm.wgsl` | Root-mean-square normalization | Transformer blocks |
| `silu.wgsl` | f32 SiLU | Diffusion UNet/MLP |
| `conv2d.wgsl` | f32 NHWC same-padding stride-1 convolution with bias | Vision / image diffusion |
| `alpha_composite.wgsl` | f32 **straight** RGBA alpha-over | Background removal and material previews |
| `depth_unproject.wgsl` | Camera intrinsics, positive depth → GPU XYZ+validity | Geometry |
| `normal_from_points.wgsl` | Grid adjacent-point cross-product surface normals | Geometry/materials |
| `skin_lbs.wgsl` | 4-influence linear blend skinning with column-major joint matrices | SOMA character animation |
| `grid_render.wgsl` | Direct WebGPU triangle-list rasterization of GPU-resident depth points | Geometry preview |

`ops.js` validates CPU shapes/types and exposes typed operator calls. `runtime.js` provides WGSL shader compilation/error reporting, cached compute pipelines, GPU-resident buffers, dispatch, uniforms, readback, device limits and cleanup. **A future graph executor should use resident GPU buffers throughout**, instead of calling these high-level wrappers with host readback after every operator. This is essential for competitive performance.

### Real hardware testing and correctness

```bash
cd webgpu
npm test
# CPU-only JS shape checks, signed I4 pack/depack and uniform layout checks.
```

The browser **Run custom WGSL correctness tests** button actually executes each compute pipeline on your browser's WebGPU GPU and compares its results to independent CPU references, then measures host-transfer-inclusive execution time. It checks irregular matmul sizes, Q4 signed nibble ordering, stable softmax, layernorm, RMSNorm, SiLU, alpha compositing, depth reprojection, normals and rigid/weighted skinning. **Those GPU checks have not been run by the assistant on a browser adapter.** Automated CI here validates JavaScript syntax and CPU-side contracts only.

`webgpu/package.json` has no runtime npm dependencies. To run the browser lab you only need a modern WebGPU browser and a localhost/HTTPS server.

## Source research and design choices

- [PrismML-Eng/Talk: readable `kernels.ts`](https://github.com/PrismML-Eng/Talk/blob/main/kernels.ts) is Prism ML's published browser engine for Ternary Bonsai 2. Its structure demonstrates GGUF loading, packed low-bit matrix operators, GPU residence and a browser UI. Its **ternary LLM kernels are not automatically compatible** with 3D diffusion or rigging. The published source also shows a native WebGPU compute-pass executor, GPU-resident bindings and 256-byte-aligned parameter slots. No Prism source was copied here; the current I-3D runtime has its own smaller explicit-dispatch API, not Prism's pooling/graph implementation.
- [Hugging Face Bonsai 2 WebGPU kernels Space](https://huggingface.co/spaces/webml-community/ternary-bonsai-2-webgpu-kernels/tree/main) contains the public browser demo (its HTML is a bundled source).
- [Hugging Face WebGPU kernels](https://huggingface.co/blog/webgpu-kernels) describes individually versioned WGSL contracts, reference test vectors and shape-dependent shader variants. We follow this *correctness before speed* architecture while writing original code.
- [ONNX Runtime Web WebGPU](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) is a potential **transitional** model graph backend. It isn't the core custom runtime and is not integrated in this branch.
- [m96-chan/web-xpu-ops](https://github.com/m96-chan/web-xpu-ops) documents a 3D/diffusion-relevant pattern of independent WGSL operators, host CPU references and per-device testing. No source copied.

### Next required model ports (not implemented)

1. **Model artifact contracts:** weight manifest, chunked `safetensors`/ONNX tensors, weight hash verification, quantization calibration, GPU memory budget and persistent caching. Different weights and licenses must be respected.
2. **Hi3D / Hunyuan-2mini:** image conditioning (DINO/CLIP), attention graphs, VAE/DiT, sparse voxel **3D convolution**, octree/mesh extraction, full denoiser loop, shape/unit parity tests against CUDA. Current 2D convolution is **not** a SpConv substitute.
3. **Hunyuan Paint 2.1:** multiview diffusion, UV unwrapping, CUDA rasterizer replacement and PBR bake; browser-side mesh library plus specialized WebGPU raster/texture kernels.
4. **TokenRig:** mesh encoder, joint/weight token decoder, Qwen component, topology/skinweight transfer, robust ROM pose tests.
5. **MotionBricks:** SOMA ONNX inference graph on WebGPU, temporal state and endpoint corrections, pose buffer layouts, verified 30-joint local transform order. Current LBS handles rendering, **not** model-generated animation.
6. **Performance:** GPU timestamp query when available, device-specific tiling, f16 optional feature, int8/fp8/fp4 calibration, residency and async copies, mixed precision and numerical drift acceptance bands.

**No end-to-end browser model inference or latency performance claim is made.** Browser WebGPU, third-party sources, original model checkpoints and licenses are distinct from the existing Colab CUDA pipeline.

## License

The new `webgpu/` operator/runtime implementation is independently written for I-3D_web. External Prism ML/Hugging Face code is used as design reference and **not vendored/copied**. The original I-3D vendor sources and model weights remain governed by their respective licenses.
