import assert from "node:assert/strict";
import {test} from "node:test";
import {
  MODEL_SIZE,RMBG_PRESETS,toNormalizedNCHW,predictionToAlpha,rgbaFromAlpha,
  grayscaleMaskRGBA,assertSingleMaskTensor
} from "../rmbg2.js";

test("BRIA web-model presets are real pinned Hugging Face ONNX paths",()=>{
  assert.match(RMBG_PRESETS.patched_fp16.url,/yamura4\/RMBG-2\.0-WebGPU\/resolve\/main\/onnx\/model_fp16\.onnx/);
  assert.match(RMBG_PRESETS.kn4666_fp16.url,/kn4666\/bria-rmbg-2\.0-web\/resolve\/main\/onnx\/model_fp16\.onnx/);
  assert.match(RMBG_PRESETS.kn4666_q4.url,/kn4666\/bria-rmbg-2\.0-web\/resolve\/main\/onnx\/model_q4\.onnx/);
  assert.equal(RMBG_PRESETS.patched_fp16.output,"probability");
  assert.equal(RMBG_PRESETS.kn4666_fp16.experimental,true);
});

test("ImageNet normalization uses true planar RGB layout, not RGBA interleaving",()=>{
  const input=new Uint8ClampedArray([255,0,128,255,0,255,64,255]);
  const out=toNormalizedNCHW(input,2,1);
  assert.equal(out.length,6);
  const near=(a,b)=>Math.abs(a-b)<1e-6;
  assert.ok(near(out[0],(1-.485)/.229));
  assert.ok(near(out[1],(0-.485)/.229));
  assert.ok(near(out[2],(0-.456)/.224));
  assert.ok(near(out[3],(1-.456)/.224));
  assert.ok(near(out[4],(128/255-.406)/.225));
  assert.ok(near(out[5],(64/255-.406)/.225));
});

test("BRIA output supports explicit probabilities and explicit sigmoid logits",()=>{
 const raw=new Float32Array([0,0.5,1]);
 assert.deepEqual([...predictionToAlpha(raw,"probability").alpha],[0,128,255]);
 assert.deepEqual([...predictionToAlpha(new Float32Array([-10,0,10]),"logit").alpha],[0,128,255]);
 assert.equal(predictionToAlpha(new Float32Array([-1,0,1]),"auto").interpretation,"logit(sigmoid)");
 assert.throws(()=>predictionToAlpha(new Float32Array([NaN]),"probability"),/nonfinite/);
});

test("RGBA foreground alpha is multiplied by source alpha, with no fake foreground",()=>{
 const rgb=new Uint8ClampedArray([200,100,30,255,30,70,200,128]);
 const out=rgbaFromAlpha(rgb,new Uint8ClampedArray([0,128]));
 assert.deepEqual([...out],[200,100,30,0,30,70,200,64]);
 assert.deepEqual([...grayscaleMaskRGBA(new Uint8ClampedArray([0,128]))],[0,0,0,255,128,128,128,255]);
});

test("reject output tensors with unexpected batch/channels/resolution/dtype",()=>{
 const data=new Float32Array(MODEL_SIZE*MODEL_SIZE);
 assert.equal(assertSingleMaskTensor({dims:[1,1,MODEL_SIZE,MODEL_SIZE],data}),data);
 assert.throws(()=>assertSingleMaskTensor({dims:[1,3,MODEL_SIZE,MODEL_SIZE],data}),/Expected RMBG/);
 assert.throws(()=>assertSingleMaskTensor({dims:[1,1,512,512],data}),/Expected RMBG/);
 assert.throws(()=>assertSingleMaskTensor({dims:[1,1,MODEL_SIZE,MODEL_SIZE],data:new Uint8Array(data.length)}),/Expected RMBG/);
});
