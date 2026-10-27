import {
  MODEL_SIZE,MAX_INPUT_PIXELS,RMBG_PRESETS,toNormalizedNCHW,predictionToAlpha,
  rgbaFromAlpha,grayscaleMaskRGBA,assertSingleMaskTensor
} from "./rmbg2.js";

const $=id=>document.getElementById(id);
const ORT_VERSION="1.31.1";
const ORT_ROOT=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_EDGE=MODEL_SIZE;
const status=s=>{$("status").textContent=s;};
const diagnostics=x=>{$("details").textContent=JSON.stringify(x,null,2);};
let ortPromise=null;
let session=null,sessionKey=null;
let originalBitmap=null,originalFileName="subject",originalWidth=0,originalHeight=0;
let currentOutputs=null;

async function ortModule(){
  if(!ortPromise){
    ortPromise=import(ORT_ROOT+"ort.webgpu.bundle.min.mjs").catch(error=>{
      ortPromise=null;throw new Error("ONNX Runtime Web could not load: "+error.message);
    });
  }
  const ort=await ortPromise;
  ort.env.wasm.wasmPaths=ORT_ROOT;
  ort.env.wasm.numThreads=1; // iframe proxy does not guarantee cross-origin isolation
  return ort;
}
function modelSelection(){
 const key=$("model").value;
 const config=RMBG_PRESETS[key];
 if(!config)throw new Error("Unknown BRIA model variant");
 return {key,config};
}
async function ensureSession(key,config){
 if(!navigator.gpu || !isSecureContext)
   throw new Error("WebGPU unavailable. Use Chrome/Edge on HTTPS/localhost with a supported GPU.");
 if(session&&sessionKey===key)return session;
 if(session){
   await session.release();
   session=null;sessionKey=null;
 }
 status("Loading ONNX Runtime Web "+ORT_VERSION+"\n"+
        "Downloading "+config.name+" (~"+Math.round(config.bytesApprox/1e6)+" MB)...\n"+
        "This runs on your browser GPU, not Colab. Large memory allocations are expected.");
 const ort=await ortModule();
 try{
   // The model URL stays in the host browser. No proxy, server-side Python,
   // silent CPU fallback, or pretend segmentation result is used.
   session=await ort.InferenceSession.create(config.url,{
     executionProviders:["webgpu"],
     graphOptimizationLevel:"all",
     logSeverityLevel:2,
   });
   sessionKey=key;
   if(!Array.isArray(session.inputNames)||session.inputNames.length!==1)
     throw new Error("Expected exactly one ONNX input; got "+JSON.stringify(session.inputNames));
   if(!Array.isArray(session.outputNames)||!session.outputNames.length)
     throw new Error("ONNX session has no named output tensors");
   return session;
 }catch(error){
   try{await session?.release();}catch{}
   session=null;sessionKey=null;
   throw new Error("BRIA ONNX WebGPU session initialization failed: "+error.message+
     (config.experimental? "\nThe unpatched kn4666 graph may exceed WebGPU shader binding limits. Try the patched preset.":""));
 }
}
function drawBitmap(canvas,bitmap){
  canvas.width=bitmap.width;canvas.height=bitmap.height;
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(bitmap,0,0);
  return ctx;
}
async function loadSource(){
 const file=$("photo").files?.[0];
 if(!file)throw new Error("Choose a PNG, JPEG or WEBP photo");
 if(!["image/png","image/jpeg","image/webp"].includes(file.type))
   throw new Error("Unsupported input type: "+file.type);
 originalBitmap?.close();
 originalBitmap=await createImageBitmap(file);
 originalWidth=originalBitmap.width;originalHeight=originalBitmap.height;
 if(!originalWidth||!originalHeight||originalWidth*originalHeight>MAX_INPUT_PIXELS){
   originalBitmap.close();originalBitmap=null;
   throw new Error("Input exceeds the 40 megapixel browser-side safety limit");
 }
 originalFileName=file.name.replace(/\.[^.]+$/,"").replace(/[^a-zA-Z0-9_-]/g,"_").slice(0,80)||"subject";
 drawBitmap($("original"),originalBitmap);
 return originalBitmap;
}
function encodePNGB(blobCanvas){
 return new Promise((resolve,reject)=>blobCanvas.toBlob(blob=>{
   if(blob)resolve(blob);else reject(new Error("PNG encoding failed"));
 },"image/png"));
}
async function createResults(prediction,mode,imageBitmap){
 const atWidth=MODEL_EDGE, atHeight=MODEL_EDGE;
 if(prediction.length!==atWidth*atHeight)throw new RangeError("Unexpected model mask size");
 const {alpha,interpretation}=predictionToAlpha(prediction,mode);
 const small=document.createElement("canvas");small.width=atWidth;small.height=atHeight;
 const smallCtx=small.getContext("2d",{willReadFrequently:true});
 smallCtx.putImageData(new ImageData(grayscaleMaskRGBA(alpha),atWidth,atHeight),0,0);
 const maskCanvas=$("mask");
 maskCanvas.width=originalWidth;maskCanvas.height=originalHeight;
 const maskCtx=maskCanvas.getContext("2d",{willReadFrequently:true});
 maskCtx.imageSmoothingEnabled=true;
 maskCtx.imageSmoothingQuality="high";
 maskCtx.clearRect(0,0,originalWidth,originalHeight);
 maskCtx.drawImage(small,0,0,originalWidth,originalHeight);
 const resized=maskCtx.getImageData(0,0,originalWidth,originalHeight).data;
 const fullAlpha=new Uint8ClampedArray(originalWidth*originalHeight);
 for(let i=0;i<fullAlpha.length;i++)fullAlpha[i]=resized[i*4];
 const cutoutCanvas=$("cutout");
 cutoutCanvas.width=originalWidth;cutoutCanvas.height=originalHeight;
 const cutoutCtx=cutoutCanvas.getContext("2d",{willReadFrequently:true});
 cutoutCtx.drawImage(imageBitmap,0,0);
 const source= cutoutCtx.getImageData(0,0,originalWidth,originalHeight);
 const resultRGBA=rgbaFromAlpha(source.data,fullAlpha);
 cutoutCtx.putImageData(new ImageData(resultRGBA,originalWidth,originalHeight),0,0);
 const [sourcePNG,cutoutPNG,maskPNG]=await Promise.all([
   encodePNGB($("original")),encodePNGB(cutoutCanvas),encodePNGB(maskCanvas)
 ]);
 return {sourcePNG,cutoutPNG,maskPNG,interpretation,width:originalWidth,height:originalHeight};
}
function download(blob,name){
 const url=URL.createObjectURL(blob);
 const a=document.createElement("a");a.href=url;a.download=name;
 document.body.append(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),30_000);
}
function lockAll(active){
 $("generate").disabled=active;
 $("sendWorkspace").disabled=active||!currentOutputs;
 $("downloadRGBA").disabled=active||!currentOutputs;
 $("downloadMask").disabled=active||!currentOutputs;
}
async function generate(){
 currentOutputs=null;lockAll(true);
 try {
   const bitmap=await loadSource();
   const {key,config}=modelSelection();
   const input=document.createElement("canvas");
   input.width=input.height=MODEL_EDGE;
   const ctx=input.getContext("2d",{willReadFrequently:true});
   ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality="high";
   ctx.drawImage(bitmap,0,0,MODEL_EDGE,MODEL_EDGE);
   const nchw=toNormalizedNCHW(
     ctx.getImageData(0,0,MODEL_EDGE,MODEL_EDGE).data,MODEL_EDGE,MODEL_EDGE);
   const s=await ensureSession(key,config);
   const ort=await ortModule();
   const first=s.inputNames[0];
   const output=config.expectedOutput||s.outputNames[0];
   if(!s.outputNames.includes(output))
     throw new Error("Required ONNX output "+output+" not available: "+s.outputNames.join(","));
   status("Running "+config.name+"\n"+
     "ONNX input: "+first+"; output: "+output+"\n"+
     "Image preprocessing: 1024×1024 RGB, ImageNet normalization.");
   const start=performance.now();
   const tensors=await s.run({
     [first]:new ort.Tensor("float32",nchw,[1,3,MODEL_EDGE,MODEL_EDGE])
   });
   const inferenceMs=performance.now()-start;
   const maskValues=assertSingleMaskTensor(tensors[output]);
   const mode=$("maskMode").value==="model"?config.output:$("maskMode").value;
   currentOutputs=await createResults(maskValues,mode,bitmap);
   currentOutputs.modelKey=key;
   currentOutputs.modelRepo=config.repo;
   diagnostics({
     model:config.repo,variant:key,ONNXInput:first,ONNXOutput:output,
     outputDimensions:tensors[output].dims,maskInterpretation:currentOutputs.interpretation,
     outputPixels:[currentOutputs.width,currentOutputs.height],
     inferenceMilliseconds:+inferenceMs.toFixed(2),
     sourcePNGBytes:currentOutputs.sourcePNG.size,
     cutoutPNGBytes:currentOutputs.cutoutPNG.size,
     maskPNGBytes:currentOutputs.maskPNG.size,
     executionProvider:"webgpu only (no silent CPU fallback)",
     warning:config.experimental?"Unpatched ONNX graph has not been verified on browser WebGPU.":null
   });
   status("BRIA model inference completed. Original, mask and RGBA PNG are ready.\n"+
       "Use 'Save into I-3D workspace' to hand them to the next pipeline stage.");
 }catch(error){
   diagnostics({error:error.message,stack:error.stack});
   status("Background removal failed:\n"+error.message);
   throw error;
 }finally{lockAll(false);}
}
$("generate").addEventListener("click",()=>generate().catch(console.error));
$("downloadRGBA").addEventListener("click",()=>{
 if(currentOutputs)download(currentOutputs.cutoutPNG,originalFileName+"_rgba.png");
});
$("downloadMask").addEventListener("click",()=>{
 if(currentOutputs)download(currentOutputs.maskPNG,originalFileName+"_mask.png");
});
$("photo").addEventListener("change",()=>{
 currentOutputs=null;lockAll(false);
 status("Image selected. Run BRIA segmentation to generate the cutout.");
});
$("model").addEventListener("change",()=>{
 currentOutputs=null;lockAll(false);
 const {config}=modelSelection();
 status(config.experimental
   ?"Experimental unpatched kn4666 export selected. The model may fail on WebGPU; no hidden CPU fallback."
   :"Patched WebGPU RMBG 2.0 model selected. The same BRIA weights, but fixed ONNX graph.");
});
$("sendWorkspace").addEventListener("click",async()=>{
 if(!currentOutputs)return;
 $("sendWorkspace").disabled=true;
 try {
   const artifacts=[
     ["source",currentOutputs.sourcePNG],
     ["mask",currentOutputs.maskPNG],
     ["cutout",currentOutputs.cutoutPNG],
   ];
   for(const [kind,blob] of artifacts){
     status("Saving "+kind+" PNG to I-3D viewer server...");
     const response=await fetch("/api/rmbg?kind="+kind,{
       method:"POST",
       headers:{
         "X-I3D-Action":"rmbg",
         "X-I3D-Model":currentOutputs.modelKey,
         "Content-Type":"image/png",
       },body:blob
     });
     let answer=null;
     try{answer=await response.json();}catch{}
     if(!response.ok||!answer?.ok){
       throw new Error("Workspace upload "+kind+" failed ("+response.status+"): "+
         (answer?.error||"Viewer server may not support the I-3D_web upload endpoint"));
     }
   }
   status("Saved 00_source_input.png, 00_alpha_mask.png and 00_background_removed.png "+
     "in the I-3D workspace. Continue the branch notebook's geometry cell.");
 }catch(error){status("Workspace save failed: "+error.message);}
 finally{$("sendWorkspace").disabled=false;}
});
status("Ready. BRIA RMBG 2.0 model is not loaded until you click Run. "+
       "For patched FP16, allow about 513 MB model download.");
