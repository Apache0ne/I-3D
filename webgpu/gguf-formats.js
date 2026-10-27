/**
 * Direct file-compatible GGUF tensor reading for current upstream GGML types.
 * GGUF v3 container. Handles arbitrary KV entries / tensor directory before
 * selectively fetching one tensor. Never loads whole multi-GB checkpoints.
 *
 * Supported weight layouts: Q2_0=42, MXFP4=39, NVFP4=40.
 * All layouts/scale decoders follow ggml-common.h + ggml-quants.c at
 * https://github.com/ggml-org/llama.cpp ; not our own format.
 * Metadata array/string parser is bounded, never executes metadata.
 */
export const GGML_TYPES=Object.freeze({
  MXFP4:39,NVFP4:40,Q2_0:42
});
export const BLOCKS=Object.freeze({
  39:{name:"MXFP4",elements:32,bytes:17},
  40:{name:"NVFP4",elements:64,bytes:36},
  42:{name:"Q2_0",elements:64,bytes:18},
});
function positiveInt(x,label){
 if(!Number.isSafeInteger(x)||x<1)throw new RangeError(label+" must be a positive safe integer");
 return x;
}
export function expectedQuantBytes(type,rows,cols){
 positiveInt(rows,"rows");positiveInt(cols,"cols");
 const spec=BLOCKS[type];
 if(!spec)throw new Error("Unsupported GGML type "+type+"; Q2_0, MXFP4, NVFP4 only");
 if(cols%spec.elements)throw new RangeError(spec.name+" requires K divisible by "+spec.elements);
 const count=rows*(cols/spec.elements)*spec.bytes;
 if(!Number.isSafeInteger(count))throw new RangeError("Tensor byte count exceeds JS safe integer");
 return count;
}

/** Preserve every on-disk byte and its original block boundaries. Only add
 * zero padding after the FINAL byte for WebGPU u32-addressable storage.
 */
export function rawBytesToU32(bytes){
 if(!(bytes instanceof Uint8Array)||bytes.length<1)throw new TypeError("Expected nonempty Uint8Array");
 const words=new Uint32Array(Math.ceil(bytes.length/4));
 for(let i=0;i<bytes.length;i++)words[i>>2] |= bytes[i]<<((i&3)*8);
 return words;
}
export function q2F16(bits){
 let sign=(bits&0x8000)?-1:1,exp=(bits>>10)&31,man=bits&1023;
 if(exp===0)return sign*Math.pow(2,-14)*(man/1024);
 if(exp===31)return man?NaN:sign*Infinity;
 return sign*Math.pow(2,exp-15)*(1+man/1024);
}
const FP4_TWICE=Object.freeze([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]);
export function e8m0Half(byte) {return Math.pow(2,byte-128);}
export function ue4m3Half(byte){
 if(byte===0||byte===127)return 0;
 const exp=(byte>>3)&15,man=byte&7;
 return exp===0?man*Math.pow(2,-10):(1+man/8)*Math.pow(2,exp-8);
}
export function unpackGGMLQuantRow(raw,type,cols,row=0){
 if(!(raw instanceof Uint8Array))throw new TypeError("raw must be a Uint8Array");
 positiveInt(cols,"cols");
 if(!Number.isSafeInteger(row)||row<0)throw new RangeError("Negative row");
 const spec=BLOCKS[type];
 if(!spec||cols%spec.elements)throw new Error("Unknown GGML format or incomplete row");
 const nblocks=cols/spec.elements,rowBytes=nblocks*spec.bytes;
 if((row+1)*rowBytes>raw.length)throw new RangeError("Incomplete GGUF tensor payload");
 const out=new Float32Array(cols);
 const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);
 for(let b=0;b<nblocks;b++){
  const o=row*rowBytes+b*spec.bytes;
  if(type===42){
   const scale=q2F16(view.getUint16(o,true));
   for(let j=0;j<64;j++){
    const q=(raw[o+2+(j>>2)]>>((j&3)*2))&3;
    out[b*64+j]=(q-1)*scale;
   }
  }else if(type===39){
   const scale=e8m0Half(raw[o]);
   for(let j=0;j<32;j++){
    const code=(raw[o+1+(j%16)]>>(j<16?0:4))&15;
    out[b*32+j]=FP4_TWICE[code]*scale;
   }
  }else if(type===40){
   for(let group=0;group<4;group++){
    const scale=ue4m3Half(raw[o+group]);
    for(let j=0;j<16;j++){
      const code=(raw[o+4+group*8+(j%8)]>>(j<8?0:4))&15;
      out[b*64+group*16+j]=FP4_TWICE[code]*scale;
    }
   }
  }
 }
 return out;
}
export function referenceQuantMatvec(raw,type,rows,cols,input){
 if(!(input instanceof Float32Array)||input.length!==cols)throw new TypeError("Wrong input activation");
 if(!(raw instanceof Uint8Array)||raw.length!==expectedQuantBytes(type,rows,cols))
   throw new RangeError("Wrong GGUF tensor bytes");
 const out=new Float32Array(rows);
 for(let i=0;i<rows;i++){
   const weights=unpackGGMLQuantRow(raw,type,cols,i);
   let total=0;for(let k=0;k<cols;k++)total+=weights[k]*input[k];
   out[i]=total;
 }
 return out;
}

class Reader {
 constructor(buffer){
  this.view=new DataView(buffer);this.offset=0;
  this.text=new TextDecoder("utf-8",{fatal:true});
 }
 require(bytes){
  if(this.offset+bytes>this.view.byteLength)throw new RangeError("GGUF header incomplete");
 }
 u8(){this.require(1);return this.view.getUint8(this.offset++);}
 u16(){this.require(2);const v=this.view.getUint16(this.offset,true);this.offset+=2;return v;}
 u32(){this.require(4);const v=this.view.getUint32(this.offset,true);this.offset+=4;return v;}
 u64(){
  this.require(8);const v=this.view.getBigUint64(this.offset,true);this.offset+=8;
  if(v>BigInt(Number.MAX_SAFE_INTEGER))throw new RangeError("GGUF offset/length unsafe in JS");
  return Number(v);
 }
 i8(){const v=this.u8();return v>127?v-256:v;}
 i16(){const x=this.u16();return x>32767?x-65536:x;}
 i32(){return this.u32()|0;}
 i64(){this.require(8);const v=this.view.getBigInt64(this.offset,true);this.offset+=8;return v;}
 f32(){this.require(4);const v=this.view.getFloat32(this.offset,true);this.offset+=4;return v;}
 f64(){this.require(8);const v=this.view.getFloat64(this.offset,true);this.offset+=8;return v;}
 str(){
   const size=this.u64();
   if(size>16*1024*1024)throw new RangeError("Unreasonable GGUF string length");
   this.require(size);
   const out=this.text.decode(new Uint8Array(this.view.buffer,this.offset,size));
   this.offset+=size;return out;
 }
 scalar(type,depth=0){
  if(depth>3)throw new RangeError("GGUF recursive metadata too deep");
  switch(type){
   case 0:return this.u8();
   case 1:return this.i8();
   case 2:return this.u16();
   case 3:return this.i16();
   case 4:return this.u32();
   case 5:return this.i32();
   case 6:return this.f32();
   case 7:return this.u8()!==0;
   case 8:return this.str();
   case 9:{
     const inner=this.u32(),count=this.u64();
     if(count>5_000_000)throw new RangeError("GGUF array metadata too large");
     // Only need general.alignment; array contents are parsed but NOT retained.
     for(let i=0;i<count;i++)this.scalar(inner,depth+1);
     return null;
   }
   case 10:return this.u64();
   case 11:return this.i64();
   case 12:return this.f64();
   default:throw new Error("Unsupported GGUF metadata value type "+type);
  }
 }
}

/** Reads GGUF v3 header and tensor table. Re-attempts with larger file slices.
 * Limits header to 64 MiB; tensor payloads remain on disk until selected.
 */
export async function inspectGGUF(blob,{maxHeaderBytes=64*1024*1024}={}){
 if(!blob||typeof blob.slice!=="function"||!Number.isSafeInteger(blob.size))
   throw new TypeError("inspectGGUF requires a browser File/Blob");
 if(blob.size<24)throw new Error("GGUF file truncated");
 let length=Math.min(blob.size,1024*1024);
 while(true){
  const buf=await blob.slice(0,length).arrayBuffer();
  const r=new Reader(buf);
  try{
   const magic=r.u32();
   if(magic!==0x46554747)throw new Error("File is not GGUF");
   const version=r.u32();
   if(version!==3)throw new Error("Expected GGUF v3; got "+version);
   const tensorCount=r.u64(), kvCount=r.u64();
   if(tensorCount>1000000||kvCount>1000000)throw new RangeError("Unreasonable GGUF header entries");
   const metadata={};
   for(let i=0;i<kvCount;i++){
    const key=r.str(),type=r.u32(),value=r.scalar(type);
    if(key==="general.alignment")metadata.alignment=value;
   }
   const tensors=[];
   for(let i=0;i<tensorCount;i++){
    const name=r.str(),rank=r.u32();
    if(rank<1||rank>8)throw new Error("Invalid GGUF tensor rank");
    const dims=Array.from({length:rank},()=>r.u64());
    const type=r.u32(),offset=r.u64();
    tensors.push({name,dims,type,offset});
   }
   const alignment=metadata.alignment??32;
   if(!Number.isSafeInteger(alignment)||alignment<1||alignment>4096||(alignment&(alignment-1)))
     throw new RangeError("Invalid GGUF alignment");
   const dataOffset=Math.ceil(r.offset/alignment)*alignment;
   if(dataOffset>blob.size)throw new Error("GGUF tensor table beyond end of file");
   return {version,alignment,dataOffset,fileSize:blob.size,tensors};
  }catch(err){
   if(err instanceof RangeError&&err.message==="GGUF header incomplete"&&length<Math.min(blob.size,maxHeaderBytes)){
    length=Math.min(blob.size,maxHeaderBytes,length*2);
    continue;
   }
   throw err;
  }
 }
}
/** Select an existing quantized GGUF matrix [K,rows] from the same file.
 * This is NOT a full model loader or automatic tensor quantizer.
 */
export async function loadQuantizedGGUFTensor(blob,header,name,{maxTensorBytes=512*1024*1024}={}){
 const t=header.tensors.find(x=>x.name===name);
 if(!t)throw new Error("Tensor not found: "+name);
 if(t.dims.length!==2)throw new Error("Only 2D matrix tensors are supported");
 const [cols,rows]=t.dims;
 const n=expectedQuantBytes(t.type,rows,cols);
 if(n>maxTensorBytes)throw new RangeError("Tensor too large for current single-buffer WebGPU prototype: "+n);
 const begin=header.dataOffset+t.offset,end=begin+n;
 if(!Number.isSafeInteger(end)||end>blob.size)
   throw new Error("Tensor points beyond GGUF file boundary");
 const bytes=new Uint8Array(await blob.slice(begin,end).arrayBuffer());
 if(bytes.length!==n)throw new Error("Failed to read complete quantized tensor");
 return {...t,rows,cols,bytes};
}
