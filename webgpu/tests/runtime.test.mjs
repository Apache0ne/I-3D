import assert from "node:assert/strict";
import {test} from "node:test";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {packSignedI4,unpackSignedI4,quantizeRowsI4,I3DWebGPU} from "../ops.js";
import {params32} from "../runtime.js";

const here=path.dirname(fileURLToPath(import.meta.url));

test("signed Q4 packing roundtrip incl last partial word and negative weights",()=>{
 const rows=3,cols=13;
 const vals=new Int8Array(rows*cols);
 for(let i=0;i<vals.length;i++)vals[i]=i%16-8;
 const packed=packSignedI4(vals,rows,cols);
 assert.equal(packed.length,rows*Math.ceil(cols/8));
 assert.deepEqual(unpackSignedI4(packed,rows,cols),vals);
 assert.throws(()=>packSignedI4(Int8Array.from([8]),1,1),RangeError);
});

test("I4 quantizer packs valid integers and returns per-row scales",()=>{
 const w=Float32Array.from([0,0,0,0,1,1,-1,-1]);
 const q=quantizeRowsI4(w,2,4);
 assert.deepEqual(q.scales,Float32Array.from([1,1/7]));
 const ints=unpackSignedI4(q.packed,2,4);
 assert.deepEqual([...ints],[0,0,0,0,7,7,-7,-7]);
});

test("WGSL uniform writer matches u32/f32 offsets",()=>{
 const bytes=params32(["u32","u32","f32","f32","u32","u32"],
                     [42,7,1.25,-0.5,0,2]);
 const d=new DataView(bytes);
 assert.equal(bytes.byteLength,32);
 assert.equal(d.getUint32(0,true),42);
 assert.equal(d.getUint32(4,true),7);
 assert.equal(d.getFloat32(8,true),1.25);
 assert.equal(d.getFloat32(12,true),-0.5);
 assert.equal(d.getUint32(20,true),2);
 assert.throws(()=>params32(["u32"],[-1]),RangeError);
});

test("runtime checks malformed operator shapes before GPU allocations",async()=>{
 const ops=new I3DWebGPU({});
 await assert.rejects(ops.matmul(new Float32Array(3),new Float32Array(2),2,2,2),TypeError);
 await assert.rejects(ops.q4Matvec(new Uint32Array(1),new Float32Array(7),
                                  new Float32Array(2),2,7),TypeError);
 await assert.rejects(ops.depthUnproject(new Float32Array(4),2,2,0,2,1,1),RangeError);
});

test("all named shader files are present, no unbound imports",()=>{
 const kernels=["matmul","q4_matvec","softmax","layernorm","rmsnorm",
   "silu","gguf_quant_matvec","alpha_composite","depth_unproject","normal_from_points",
   "skin_lbs","conv2d","grid_render"];
 for(const name of kernels){
  const source=fs.readFileSync(path.join(here,"..","kernels",name+".wgsl"),"utf8");
  assert.match(source,/@(compute|vertex)/);
  assert.match(source,/@group\(0\)/);
  assert.ok(!source.includes("TODO: placeholder"),name);
 }
});
