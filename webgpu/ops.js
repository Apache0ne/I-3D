import { WebGPURuntime, params32 } from "./runtime.js";
import {BLOCKS,expectedQuantBytes,rawBytesToU32} from "./gguf-formats.js";

/** Array shape & type guards keep malformed model tensors off the GPU. */
function check(input,length,label,type=Float32Array) {
  if (!(input instanceof type) || input.length!==length)
    throw new TypeError(`${label}: expected ${type.name}[${length}]`);
}
function dimensions(...values) {
  for(const value of values)if(!Number.isSafeInteger(value)||value<1||value>65535)
    throw new RangeError("Dimensions must be integers in [1,65535]");
}
const groups=(n,tile)=>[Math.ceil(n/tile),1,1];

/** Pack an integer weight array into row-padded, signed 4-bit values.
 * Layout is [rows,ceil(cols/8)] u32, eight nibbles per u32, little endian.
 */
export function packSignedI4(weights, rows, cols) {
  dimensions(rows,cols);
  if (!(weights instanceof Int8Array) || weights.length!==rows*cols)
    throw new TypeError("weights must be a dense signed Int8Array");
  const rowWords=Math.ceil(cols/8);
  const packed=new Uint32Array(rows*rowWords);
  for(let r=0;r<rows;r++){
    for(let k=0;k<cols;k++){
      const v=weights[r*cols+k];
      if(v<-8||v>7)throw new RangeError("Signed Q4 value outside [-8,7]");
      const word=r*rowWords+Math.floor(k/8);
      packed[word]|=(v&15)<<((k%8)*4);
    }
  }
  return packed;
}
export function unpackSignedI4(packed,rows,cols) {
  dimensions(rows,cols);
  const rowWords=Math.ceil(cols/8);
  check(packed,rows*rowWords,"packed",Uint32Array);
  const result=new Int8Array(rows*cols);
  for(let r=0;r<rows;r++) for(let k=0;k<cols;k++){
    const nib=(packed[r*rowWords+Math.floor(k/8)]>>>((k%8)*4))&15;
    result[r*cols+k]=nib>=8?nib-16:nib;
  }
  return result;
}

/** Optional per-row int4 quantization from f32 weights.
 * Quantization changes model predictions; calibrate real weights before use.
 */
export function quantizeRowsI4(weights,rows,cols) {
  dimensions(rows,cols);
  check(weights,rows*cols,"weights");
  const ints=new Int8Array(weights.length), scales=new Float32Array(rows);
  for(let r=0;r<rows;r++){
    let maxabs=0;
    for(let k=0;k<cols;k++) maxabs=Math.max(maxabs,Math.abs(weights[r*cols+k]));
    const scale=maxabs>0?maxabs/7:1;
    scales[r]=scale;
    for(let k=0;k<cols;k++) ints[r*cols+k]=Math.max(-8,Math.min(7,Math.round(weights[r*cols+k]/scale)));
  }
  return {packed:packSignedI4(ints,rows,cols),scales,rows,cols};
}

export class I3DWebGPU {
  constructor(runtime) {this.rt=runtime;}
  static async create(options) {return new I3DWebGPU(await WebGPURuntime.create(options));}
  info(){return this.rt.deviceInfo();}
  destroy(){this.rt.destroy();}

  /** Transfer + dispatch + readback. Model executors should instead use
   * rt.createTensor/dispatch to keep intermediate tensors on device.
   */
  async _run(name, inputs, uniform, outputLength, dispatch) {
    const allocated=[];
    try {
      for(const array of inputs)allocated.push(this.rt.createTensor(array,name+"_in"));
      const out=this.rt.output(outputLength,name+"_out");
      allocated.push(out);
      const p=this.rt.uniform(uniform);allocated.push(p);
      await this.rt.dispatch(name,[...allocated.slice(0,inputs.length),out,p],dispatch);
      return await this.rt.read(out,outputLength);
    }finally {this.rt.release(...allocated);}
  }

  async matmul(A,B,M,N,K) {
    dimensions(M,N,K);
    check(A,M*K,"A");check(B,K*N,"B");
    return this._run("matmul",[A,B],
      params32(["u32","u32","u32","u32"],[M,N,K,0]),
      M*N,[Math.ceil(N/8),Math.ceil(M/8),1]);
  }

  async q4Matvec(packed,vector,scales,rows,cols) {
    dimensions(rows,cols);
    const rowWords=Math.ceil(cols/8);
    check(packed,rows*rowWords,"packed",Uint32Array);
    check(vector,cols,"vector");check(scales,rows,"scales");
    return this._run("q4_matvec",[packed,vector,scales],
      params32(["u32","u32","u32","u32"],[rows,cols,rowWords,0]),
      rows,groups(rows,64));
  }

  /**
   * Consume an actual on-disk GGUF quant tensor without changing quantization:
   * - Q2_0 GGML type 42, 64-element blocks / 18 bytes
   * - MXFP4 type 39, 32-element blocks / 17 bytes
   * - NVFP4 type 40, 64-element blocks / 36 bytes
   *
   * The Uint8Array contains the EXACT GGUF tensor payload. It is only padded
   * to u32 storage alignment; WGSL decodes the original byte/scale layout.
   * High-level readback is for testing. Future graph executor must preserve
   * resident weights and activations across layers.
   */
  async ggufQuantMatvec(raw,vector,rows,cols,ggmlType) {
    if(!(raw instanceof Uint8Array))throw new TypeError("GGUF payload must be Uint8Array");
    dimensions(rows,cols);
    if(!BLOCKS[ggmlType])throw new Error("Unsupported GGML quantization type "+ggmlType);
    const expected=expectedQuantBytes(ggmlType,rows,cols);
    if(raw.length!==expected)throw new RangeError("GGUF quant tensor byte length: expected "+expected+", got "+raw.length);
    check(vector,cols,"input activations");
    const packed=rawBytesToU32(raw);
    return this._run("gguf_quant_matvec",[packed,vector],
      params32(["u32","u32","u32","u32"],[rows,cols,ggmlType,0]),
      rows,groups(rows,64));
  }

  async softmax(X,rows,cols) {
    dimensions(rows,cols);check(X,rows*cols,"X");
    return this._run("softmax",[X],
      params32(["u32","u32","u32","u32"],[rows,cols,0,0]),
      X.length,[rows,1,1]);
  }

  async layernorm(X,gamma,beta,rows,cols,eps=1e-5) {
    dimensions(rows,cols);check(X,rows*cols,"X");
    check(gamma,cols,"gamma");check(beta,cols,"beta");
    if(!(eps>0)) throw new RangeError("epsilon must be positive");
    return this._run("layernorm",[X,gamma,beta],
      params32(["u32","u32","f32","u32"],[rows,cols,eps,0]),
      X.length,[rows,1,1]);
  }

  async rmsnorm(X,gamma,rows,cols,eps=1e-5) {
    dimensions(rows,cols);check(X,rows*cols,"X");check(gamma,cols,"gamma");
    if(!(eps>0)) throw new RangeError("epsilon must be positive");
    return this._run("rmsnorm",[X,gamma],
      params32(["u32","u32","f32","u32"],[rows,cols,eps,0]),
      X.length,[rows,1,1]);
  }

  async silu(X) {
    if(!(X instanceof Float32Array)||!X.length)throw new TypeError("X must be nonempty Float32Array");
    return this._run("silu",[X],
      params32(["u32","u32","u32","u32"],[X.length,0,0,0]),X.length,groups(X.length,256));
  }

  async alphaComposite(fg,bg) {
    if(!(fg instanceof Float32Array)||fg.length<4||fg.length%4)throw new TypeError("fg must have RGBA stride 4");
    check(bg,fg.length,"bg");
    return this._run("alpha_composite",[fg,bg],
      params32(["u32","u32","u32","u32"],[fg.length/4,0,0,0]),
      fg.length,groups(fg.length/4,256));
  }

  async depthUnproject(depth,width,height,fx,fy,cx,cy) {
    dimensions(width,height);check(depth,width*height,"depth");
    if(!(fx>0&&fy>0)) throw new RangeError("Focal lengths must be positive");
    return this._run("depth_unproject",[depth],
      params32(["u32","u32","f32","f32","f32","f32","u32","u32"],
        [width,height,fx,fy,cx,cy,0,0]),
      depth.length*4,groups(depth.length,256));
  }

  async normalsFromPoints(points,width,height) {
    dimensions(width,height);check(points,width*height*4,"points");
    return this._run("normal_from_points",[points],
      params32(["u32","u32","u32","u32"],[width,height,0,0]),
      points.length,groups(width*height,256));
  }

  async conv2dNHWC(image,filter,bias,width,height,cin,cout,kh,kw) {
    dimensions(width,height,cin,cout,kh,kw);
    check(image,width*height*cin,"image");
    check(filter,kh*kw*cin*cout,"filter");
    check(bias,cout,"bias");
    const outputLength=width*height*cout;
    return this._run("conv2d",[image,filter,bias],
      params32(["u32","u32","u32","u32","u32","u32","u32","u32"],
        [width,height,cin,cout,kh,kw,Math.floor(kh/2),Math.floor(kw/2)]),
      outputLength,groups(outputLength,128));
  }

  async skinLBS(positions,jointIDs,weights,matrices,vertices,joints) {
    dimensions(vertices,joints);check(positions,vertices*4,"positions");
    check(jointIDs,vertices*4,"jointIDs",Uint32Array);
    check(weights,vertices*4,"weights");check(matrices,joints*16,"matrices");
    return this._run("skin_lbs",[positions,jointIDs,weights,matrices],
      params32(["u32","u32","u32","u32"],[vertices,joints,0,0]),
      vertices*4,groups(vertices,256));
  }
}
