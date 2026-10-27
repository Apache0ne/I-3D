// Direct GGML/GGUF on-disk block format GEMV.
// Input is original file tensor bytes, merely zero-padded at END to 4 bytes
// so WebGPU can bind Uint32Array. There is no dequantized weight buffer.
// GGML_TYPE_MXFP4=39, GGML_TYPE_NVFP4=40, GGML_TYPE_Q2_0=42.
// Reference: ggml/src/ggml-common.h and ggml/src/ggml-quants.c.
// Tensor shape is GGUF [K,M]; each row contains whole quant blocks.
struct Config {rows:u32,cols:u32,format:u32,reserved:u32};
@group(0) @binding(0) var<storage,read> raw:array<u32>;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
@group(0) @binding(3) var<uniform> p:Config;
fn byte_at(offset:u32)->u32 {
 let word=raw[offset/4u];
 return (word>>((offset%4u)*8u))&255u;
}
fn half_scale(offset:u32)->f32 {
 let h=byte_at(offset)|(byte_at(offset+1u)<<8u);
 return unpack2x16float(h).x;
}
// GGML's kvalues_fp4 table, represented in doubled integer values.
fn e2m1_twice(code:u32)->f32 {
 let v=code&7u;
 var magnitude:f32=0.0;
 switch v {
  case 0u: {magnitude=0.0;}
  case 1u: {magnitude=1.0;}
  case 2u: {magnitude=2.0;}
  case 3u: {magnitude=3.0;}
  case 4u: {magnitude=4.0;}
  case 5u: {magnitude=6.0;}
  case 6u: {magnitude=8.0;}
  default: {magnitude=12.0;}
 }
 if((code&8u)!=0u){return -magnitude;}
 return magnitude;
}
fn ue4m3_half(code:u32)->f32 {
 if(code==0u || code==127u){return 0.0;}
 let exp=(code>>3u)&15u;let frac=code&7u;
 if(exp==0u){return f32(frac)*exp2(-10.0);}
 return (1.0+f32(frac)/8.0)*exp2(f32(exp)-8.0);
}
fn at_weight(row:u32,k:u32)->f32 {
 if(p.format==42u) {
  // Q2_0: 2-byte IEEE half d; 16 bytes, 4x2-bit/byte; 64 values.
  let start=(row*(p.cols/64u)+k/64u)*18u;
  let code=(byte_at(start+2u+(k%64u)/4u)>>((k%4u)*2u))&3u;
  return f32(i32(code)-1)*half_scale(start);
 }
 if(p.format==39u) {
  // MXFP4: 1 E8M0 scale byte + 16 FP4 E2M1 packed bytes; 32 values.
  // GGML's codes store the first 16 in low nibbles, last 16 in high.
  let start=(row*(p.cols/32u)+k/32u)*17u;
  let elem=k%32u;
  let pair=byte_at(start+1u+elem%16u);
  let nib=select(pair&15u,pair>>4u,elem>=16u);
  // GGML_E8M0_TO_FP32_HALF: 2^(exponent-128), with subnormal 0.
  return e2m1_twice(nib)*exp2(f32(byte_at(start))-128.0);
 }
 // NVFP4: 4 UE4M3 group scales (16/group) + 32 E2M1 bytes; 64 vals.
 let start=(row*(p.cols/64u)+k/64u)*36u;
 let elem=k%64u;let group=elem/16u;let local=elem%16u;
 let pair=byte_at(start+4u+group*8u+local%8u);
 let nib=select(pair&15u,pair>>4u,local>=8u);
 return e2m1_twice(nib)*ue4m3_half(byte_at(start+group));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let row=id.x;
 if(row>=p.rows){return;}
 var sum=0.0;
 for(var k:u32=0u;k<p.cols;k=k+1u){
  sum=sum+at_weight(row,k)*x[k];
 }
 out[row]=sum;
}
