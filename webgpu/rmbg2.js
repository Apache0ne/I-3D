/**
 * BRIA RMBG-2.0 browser-stage contracts. No GPU code here: preparation and
 * transparent-PNG composition are deterministic and independently testable.
 * BRIA/kn4666: CC BY-NC 4.0. Keep source/licensing metadata with all exports.
 */
export const MODEL_SIZE=1024;
export const MAX_INPUT_PIXELS=40_000_000;
export const IMAGENET_MEAN=Object.freeze([0.485,0.456,0.406]);
export const IMAGENET_STD=Object.freeze([0.229,0.224,0.225]);
export const RMBG_PRESETS=Object.freeze({
  patched_fp16:{
    name:"RMBG 2.0 / patched WebGPU FP16 (recommended)",
    repo:"yamura4/RMBG-2.0-WebGPU",
    url:"https://huggingface.co/yamura4/RMBG-2.0-WebGPU/resolve/main/onnx/model_fp16.onnx",
    bytesApprox:513_000_000,
    output:"probability",
    expectedInput:"pixel_values",
    expectedOutput:"alphas",
    experimental:false,
  },
  kn4666_fp16:{
    name:"Your kn4666/RMBG 2.0 FP16 (unpatched, experimental)",
    repo:"kn4666/bria-rmbg-2.0-web",
    url:"https://huggingface.co/kn4666/bria-rmbg-2.0-web/resolve/main/onnx/model_fp16.onnx",
    bytesApprox:514_000_000,
    output:"auto",
    experimental:true,
  },
  kn4666_q4:{
    name:"Your kn4666/RMBG 2.0 Q4 (experimental, likely unsupported)",
    repo:"kn4666/bria-rmbg-2.0-web",
    url:"https://huggingface.co/kn4666/bria-rmbg-2.0-web/resolve/main/onnx/model_q4.onnx",
    bytesApprox:367_000_000,
    output:"auto",
    experimental:true,
  },
});

/** Input must be an ImageData-like 1024×1024 RGBA Uint8ClampedArray,
 * already resized by Canvas2D to MODEL_SIZE. Returns float32 NCHW RGB.
 */
export function toNormalizedNCHW(rgba,width,height){
 if(!(rgba instanceof Uint8ClampedArray)||rgba.length!==width*height*4)
   throw new TypeError("Expected RGBA Uint8ClampedArray width×height×4");
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1)
   throw new RangeError("Invalid input dimensions");
 const plane=width*height;
 const out=new Float32Array(plane*3);
 for(let i=0;i<plane;i++){
  for(let c=0;c<3;c++){
   // Canvas RGBA is 8-bit sRGB, matching common PyTorch ToTensor pipelines.
   out[c*plane+i]=(rgba[4*i+c]/255-IMAGENET_MEAN[c])/IMAGENET_STD[c];
  }
 }
 return out;
}
const sigmoid=x=> x>=0 ? 1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
/** Decode float32 ONNX [1,1,H,W] into grayscale alpha bytes.
 * Use explicit 'probability' or 'logit' whenever the model contract is known.
 * 'auto' is a heuristic for unknown exports only, and is disclosed in UI.
 */
export function predictionToAlpha(input,mode="probability"){
 if(!(input instanceof Float32Array) || !input.length)
   throw new TypeError("Output must be a nonempty float32 mask tensor");
 if(!["probability","logit","auto"].includes(mode))
   throw new RangeError("Mask interpretation must be probability/logit/auto");
 let isLogit=mode==="logit";
 if(mode==="auto"){
  for(let i=0;i<input.length;i++){
   if(input[i]<0||input[i]>1){isLogit=true;break;}
  }
 }
 const out=new Uint8ClampedArray(input.length);
 for(let i=0;i<input.length;i++){
   const v=input[i];
   if(!Number.isFinite(v))throw new Error("Model output contains nonfinite mask at "+i);
   const p=isLogit?sigmoid(v):Math.min(1,Math.max(0,v));
   out[i]=Math.round(p*255);
 }
 return {alpha:out,interpretation:isLogit?"logit(sigmoid)":"probability"};
}
export function rgbaFromAlpha(original,alpha) {
 if(!(original instanceof Uint8ClampedArray)||!(alpha instanceof Uint8ClampedArray)||
    original.length!==alpha.length*4)
   throw new TypeError("RGBA and alpha dimensions must match");
 const out=new Uint8ClampedArray(original);
 for(let i=0;i<alpha.length;i++){
   out[4*i+3]=Math.round(original[4*i+3]*alpha[i]/255);
 }
 return out;
}
export function grayscaleMaskRGBA(alpha){
 if(!(alpha instanceof Uint8ClampedArray))throw new TypeError("Expected Uint8ClampedArray alpha");
 const mask=new Uint8ClampedArray(alpha.length*4);
 for(let i=0;i<alpha.length;i++){
   const m=alpha[i]; const k=i*4;
   mask[k]=mask[k+1]=mask[k+2]=m;mask[k+3]=255;
 }
 return mask;
}
/** Ensures exactly one same-resolution mask; do not quietly reshape outputs. */
export function assertSingleMaskTensor(tensor){
 const dims=tensor?.dims;
 if(!Array.isArray(dims)||dims.length!==4||dims[0]!==1||dims[1]!==1 ||
    dims[2]!==MODEL_SIZE||dims[3]!==MODEL_SIZE ||
    !(tensor.data instanceof Float32Array) ||
    tensor.data.length!==MODEL_SIZE*MODEL_SIZE){
   throw new Error("Expected RMBG float32 output [1,1,1024,1024], got "
     +JSON.stringify(dims)+" dtype "+(tensor?.data?.constructor?.name||"unknown"));
 }
 return tensor.data;
}
