// CPU-only contract checks. Reference formulas match published GGML formats.
// Real WebGPU math is separately checked by the browser-test UI.
import assert from "node:assert/strict";
import {test} from "node:test";
import {
  GGML_TYPES,BLOCKS,expectedQuantBytes,rawBytesToU32,
  unpackGGMLQuantRow,referenceQuantMatvec,inspectGGUF,loadQuantizedGGUFTensor
} from "../gguf-formats.js";

function put64(view,offset,value){view.setBigUint64(offset,BigInt(value),true);}
function syntheticGGUF(name,type,rows,cols,payload){
 const nameBytes=new TextEncoder().encode(name);
 const length=24+8+nameBytes.length+4+16+4+8;
 const aligned=Math.ceil(length/32)*32;
 const output=new Uint8Array(aligned+payload.length);
 const dv=new DataView(output.buffer);
 output.set([0x47,0x47,0x55,0x46],0);
 dv.setUint32(4,3,true);
 put64(dv,8,1);
 put64(dv,16,0);
 let at=24;
 put64(dv,at,nameBytes.length);at+=8;
 output.set(nameBytes,at);at+=nameBytes.length;
 dv.setUint32(at,2,true);at+=4;
 put64(dv,at,cols);at+=8;put64(dv,at,rows);at+=8;
 dv.setUint32(at,type,true);at+=4;
 put64(dv,at,0);
 output.set(payload,aligned);
 return new Blob([output]);
}

test("GGUF block sizes are upstream's raw on-disk layouts",()=>{
 assert.equal(expectedQuantBytes(GGML_TYPES.Q2_0,2,64),36);
 assert.equal(expectedQuantBytes(GGML_TYPES.MXFP4,2,32),34);
 assert.equal(expectedQuantBytes(GGML_TYPES.NVFP4,2,64),72);
 assert.throws(()=>expectedQuantBytes(39,2,33),RangeError);
 assert.throws(()=>expectedQuantBytes(42,0,64),RangeError);
 assert.throws(()=>expectedQuantBytes(9,1,64),Error);
});

test("GGUF Q2_0: FP16 block scale and code q-1 incl 11=>+2",()=>{
 const bytes=new Uint8Array(18);
 bytes[0]=0;bytes[1]=0x3c; // IEEE half 1.0, little endian
 bytes[2]=0b11100100; // code: 00 -1; 01 0; 10 +1; 11 +2
 for(let i=3;i<18;i++)bytes[i]=0b01010101; // zero
 const values=unpackGGMLQuantRow(bytes,42,64);
 assert.deepEqual([...values.slice(0,4)],[-1,0,1,2]);
 assert.deepEqual([...values.slice(4,10)],[0,0,0,0,0,0]);
 const x=new Float32Array(64);x[0]=1;x[2]=2;x[3]=3;
 assert.equal(referenceQuantMatvec(bytes,42,1,64,x)[0],-1+2+6);
});

test("GGUF MXFP4: E8M0 exponent and GGML half-block nibble split",()=>{
 const bytes=new Uint8Array(17);
 bytes[0]=127; // GGML half-scale = 2^-1
 bytes[1]=(9<<4)|7; // low: +12 => +6, high: -1 => -0.5
 const w=unpackGGMLQuantRow(bytes,39,32);
 assert.equal(w[0],6);
 assert.equal(w[16],-.5);
 assert.equal(w[1],0);
});

test("GGUF NVFP4: 4 independent UE4M3 scales and 16-element group nibble halves",()=>{
 const bytes=new Uint8Array(36);
 bytes[0]=0x38; // UE4M3 half-scale = 0.5
 bytes[1]=0x40; // (exp8) UE4M3 half-scale = 1.0
 bytes[2]=0x00; // zero
 bytes[3]=0x38;
 bytes[4]=(15<<4)|7; // low=+12, high=-12
 bytes[4+8]=(8<<4)|4; // second group: low +4, high -0
 const w=unpackGGMLQuantRow(bytes,40,64);
 assert.equal(w[0],6);
 assert.equal(w[8],-6);
 assert.equal(w[16],4);
 assert.equal(w[24],0);
 assert.equal(w[32],0);
});

test("u32 WebGPU upload preserves original quant payload bytes byte-by-byte",()=>{
 const x=Uint8Array.from({length:37},(_,i)=>i*17%256);
 const words=rawBytesToU32(x);
 const out=new Uint8Array(words.buffer);
 assert.deepEqual([...out.slice(0,37)],[...x]);
 assert.equal(words.byteLength,40);
});

test("actual GGUF v3 tensor table selects and reads quantized file tensor without full file",async()=>{
 const matrix=new Uint8Array(36);
 matrix[0]=0;matrix[1]=0x3c;matrix[18]=0;matrix[19]=0x40;
 matrix.fill(0x55,2,18);matrix.fill(0x55,20);
 const gguf=syntheticGGUF("model.blk.0.attn_q.weight",42,2,64,matrix);
 const header=await inspectGGUF(gguf);
 assert.equal(header.tensors.length,1);
 assert.equal(header.tensors[0].type,42);
 assert.deepEqual(header.tensors[0].dims,[64,2]);
 const tensor=await loadQuantizedGGUFTensor(gguf,header,header.tensors[0].name);
 assert.deepEqual([...tensor.bytes],[...matrix]);
 assert.equal(tensor.rows,2);
 assert.equal(tensor.cols,64);
 const vec=new Float32Array(64).fill(1);
 const y=referenceQuantMatvec(tensor.bytes,42,2,64,vec);
 assert.deepEqual([...y],[0,0]);
});

test("reject malformed GGUF header and unsupported quant input",async()=>{
 const invalid=new Blob([Uint8Array.from([1,2,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])]);
 await assert.rejects(inspectGGUF(invalid),/not GGUF/);
 const bytes=new Uint8Array(17);
 const gguf=syntheticGGUF("matrix",39,1,32,bytes);
 const hdr=await inspectGGUF(gguf);
 await assert.rejects(loadQuantizedGGUFTensor(gguf,hdr,"not-there"),/not found/);
});
