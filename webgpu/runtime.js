/**
 * I-3D original browser WebGPU runtime: cached WGSL programs, explicit
 * resource lifetimes, GPU-resident tensors, optional host readback.
 *
 * No Torch/ONNX required for these individual operators. A model will only
 * run once its entire graph, weights and unsupported operators are ported.
 */
const STORAGE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const align4 = n => Math.max(4, Math.ceil(n / 4) * 4);

export class WebGPURuntime {
  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.programs = new Map();
    this.buffers = new Set();
    this.lost = null;
    device.lost.then(info => { this.lost = info; });
  }

  static async create({powerPreference = "high-performance"} = {}) {
    if (!globalThis.isSecureContext || !navigator.gpu) {
      throw new Error("WebGPU needs HTTPS or localhost and a WebGPU-enabled browser.");
    }
    const adapter = await navigator.gpu.requestAdapter({powerPreference});
    if (!adapter) throw new Error("No WebGPU adapter. Check browser and GPU settings.");
    // Avoid requiring shader-f16: f32 works on more GPUs (including mobile).
    const device = await adapter.requestDevice({requiredFeatures: []});
    return new WebGPURuntime(adapter, device);
  }

  deviceInfo() {
    const limits = this.device.limits;
    return {
      adapter: this.adapter.info ? {
        vendor:this.adapter.info.vendor,
        architecture:this.adapter.info.architecture,
        device:this.adapter.info.device,
        description:this.adapter.info.description,
      } : {},
      features:[...this.device.features],
      limits:{
        maxStorageBufferBindingSize:limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupsPerDimension:limits.maxComputeWorkgroupsPerDimension,
        maxBufferSize:limits.maxBufferSize,
      },
      shaderF16:this.device.features.has("shader-f16"),
    };
  }

  createTensor(input, label="tensor") {
    if (!(input instanceof Float32Array || input instanceof Uint32Array)) {
      throw new TypeError("Expected Float32Array or Uint32Array; input must use explicit dtype.");
    }
    if (input.byteLength === 0) throw new RangeError("Zero-sized GPU tensors are unsupported");
    const bytes = align4(input.byteLength);
    if (bytes > this.device.limits.maxStorageBufferBindingSize)
      throw new RangeError("Input exceeds maxStorageBufferBindingSize; segment the tensor.");
    const buffer = this.device.createBuffer({label,size:bytes,usage:STORAGE()});
    this.device.queue.writeBuffer(buffer,0,input);
    this.buffers.add(buffer);
    return buffer;
  }

  output(length, label="result") {
    if (!Number.isSafeInteger(length) || length<1) throw new RangeError("Output length must be positive");
    const bytes = align4(length*4);
    if (bytes > this.device.limits.maxStorageBufferBindingSize)
      throw new RangeError("Output exceeds maxStorageBufferBindingSize");
    const buffer = this.device.createBuffer({label,size:bytes,usage:STORAGE()});
    this.buffers.add(buffer);
    return buffer;
  }

  uniform(data) {
    if (!(data instanceof ArrayBuffer || ArrayBuffer.isView(data)))
      throw new TypeError("Uniform data must be an ArrayBuffer or typed array");
    const src = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);
    const size = Math.max(16,Math.ceil(src.byteLength/16)*16);
    const buffer = this.device.createBuffer({
      size,label:"I3D kernel uniform",usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(buffer,0,src);
    this.buffers.add(buffer);
    return buffer;
  }

  async pipeline(name) {
    if (this.lost) throw new Error("WebGPU device lost: " + this.lost.message);
    if (this.programs.has(name)) return this.programs.get(name);
    const task = (async()=>{
      const url = new URL("./kernels/"+name+".wgsl",import.meta.url);
      // All file names are controlled by I-3D ops, never user-provided URLs.
      if (!/^[a-z0-9_]+$/.test(name)) throw new Error("Invalid kernel name");
      const res = await fetch(url);
      if (!res.ok) throw new Error("Missing WGSL shader "+name+" ("+res.status+")");
      const source = await res.text();
      const module = this.device.createShaderModule({label:name,code:source});
      const info=await module.getCompilationInfo();
      const errors=info.messages.filter(m=>m.type==="error");
      if (errors.length)
        throw new Error("WGSL "+name+": "+errors.map(m=>m.lineNum+":"+m.linePos+" "+m.message).join("\n"));
      return await this.device.createComputePipelineAsync({
        label:"I3D_"+name,layout:"auto",compute:{module,entryPoint:"main"}
      });
    })();
    this.programs.set(name,task);
    try {return await task;}
    catch(err){this.programs.delete(name);throw err;}
  }

  /** GPU resident dispatch, no readback. Buffers in WGSL binding order. */
  async dispatch(name, bindings, workgroups) {
    if (this.lost) throw new Error("Device lost: "+this.lost.message);
    const pipeline = await this.pipeline(name);
    const groups=Array.isArray(workgroups)?workgroups:[workgroups,1,1];
    if (groups.some((n,i)=>!Number.isInteger(n)||n<1||n>this.device.limits.maxComputeWorkgroupsPerDimension)) {
      throw new RangeError("Invalid dispatch groups: "+JSON.stringify(groups));
    }
    const group = this.device.createBindGroup({
      layout:pipeline.getBindGroupLayout(0),
      entries:bindings.map((buffer,binding)=>({binding,resource:{buffer}}))
    });
    const encoder=this.device.createCommandEncoder({label:"I3D_"+name});
    const pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);
    pass.dispatchWorkgroups(groups[0],groups[1]||1,groups[2]||1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async read(buffer, length, kind="f32") {
    if (!Number.isSafeInteger(length)||length<1) throw new RangeError("Bad readback length");
    const bytes = length*4;
    const staging=this.device.createBuffer({
      size:align4(bytes),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST
    });
    try {
      const encoder=this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer,0,staging,0,bytes);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const src=staging.getMappedRange();
      const result=kind==="u32" ? new Uint32Array(src.slice(0,bytes)) : new Float32Array(src.slice(0,bytes));
      staging.unmap();
      return result;
    } finally { staging.destroy(); }
  }

  release(...buffers) {
    for(const b of buffers) {
      if(this.buffers.delete(b)) b.destroy();
    }
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();this.programs.clear();this.device.destroy();
  }
}

/** 16-byte WGSL uniform packing, mixed u32/f32 parameters. */
export function params32(types, values) {
  if (types.length !== values.length) throw new Error("Uniform type/value mismatch");
  const count=Math.ceil(types.length/4)*4;
  const bytes=new ArrayBuffer(count*4), view=new DataView(bytes);
  types.forEach((t,i)=>{
    const value=values[i];
    if(t==="u32"){
      if(!Number.isInteger(value)||value<0||value>4294967295)throw new RangeError("Invalid u32");
      view.setUint32(i*4,value,true);
    }else if(t==="f32"){
      if(!Number.isFinite(value))throw new RangeError("Invalid f32");
      view.setFloat32(i*4,value,true);
    }else throw new Error("Invalid uniform type: "+t);
  });
  return bytes;
}
