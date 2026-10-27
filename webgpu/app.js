import {I3DWebGPU} from "./ops.js";
import {runBrowserSuite} from "./browser-tests.js";
import {params32} from "./runtime.js";

const $=id=>document.getElementById(id);
const log=(message)=>{const pre=$("log");pre.textContent+=String(message)+"\n";pre.scrollTop=pre.scrollHeight;};
const status=(message)=>{$("status").textContent=message;};
let engine=null, renderer=null, currentDepth=null;

async function ensureEngine(){
 if(engine)return engine;
 status("Requesting WebGPU device");
 engine=await I3DWebGPU.create();
 const info=engine.info();
 $("device").textContent=JSON.stringify(info,null,2);
 status("WebGPU device ready");
 return engine;
}

class DepthGridRenderer {
 constructor(runtime,canvas) {
  this.rt=runtime;
  this.device=runtime.device;
  this.canvas=canvas;
  this.context=canvas.getContext("webgpu");
  if(!this.context)throw new Error("Canvas does not support WebGPU");
  this.format=navigator.gpu.getPreferredCanvasFormat();
  this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"});
  this.pipeline=null;
  this.buffers=[];
  this.count=0;this.width=0;this.height=0;
  this.yaw=0.5;this.pitch=0.32;this.zoom=1;
  this._bind=null;
 }
 async init() {
  const response=await fetch("./kernels/grid_render.wgsl");
  if(!response.ok)throw new Error("Grid rendering shader could not be loaded");
  const module=this.device.createShaderModule({code:await response.text()});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(e=>e.message).join("\n"));
  this.pipeline=await this.device.createRenderPipelineAsync({
   label:"I3D custom GPU geometry grid",
   layout:"auto",
   vertex:{module,entryPoint:"vs_main"},
   fragment:{module,entryPoint:"fs_main",targets:[{format:this.format}]},
   primitive:{topology:"triangle-list",cullMode:"none"},
  });
  this.camera=this.rt.uniform(params32(
    ["u32","u32","f32","f32","f32","f32","f32","f32"],
    [2,2,0,0,1,1,0,0]));
  this.buffers.push(this.camera);
 }
 async loadDepth(depth,width,height){
  if(width<2||height<2)throw new RangeError("Depth grid must be at least 2×2");
  if(depth.length!==width*height)throw new Error("Depth array dimensions mismatch");
  // Release previous resident geometry only after submitted work completes.
  await this.device.queue.onSubmittedWorkDone();
  const camera=this.camera;
  this.rt.release(...this.buffers.filter(b=>b!==camera));
  this.buffers=[camera];
  const d=this.rt.createTensor(depth,"depth meters");
  const p=this.rt.output(width*height*4,"reconstructed vertices");
  const n=this.rt.output(width*height*4,"generated normals");
  const c=this.rt.uniform(params32(["u32","u32","f32","f32","f32","f32","u32","u32"],
    [width,height,width*0.85,height*0.85,(width-1)/2,(height-1)/2,0,0]));
  const shape=this.rt.uniform(params32(["u32","u32","u32","u32"],[width,height,0,0]));
  this.buffers.push(d,p,n,c,shape);
  await this.rt.dispatch("depth_unproject",[d,p,c],[Math.ceil(depth.length/256),1,1]);
  await this.rt.dispatch("normal_from_points",[p,n,shape],[Math.ceil(depth.length/256),1,1]);
  this._bind=this.device.createBindGroup({
   layout:this.pipeline.getBindGroupLayout(0),
   entries:[{binding:0,resource:{buffer:p}},
            {binding:1,resource:{buffer:n}},
            {binding:2,resource:{buffer:this.camera}}]
  });
  this.width=width;this.height=height;
  this.count=(width-1)*(height-1)*6;
  this.resize();this.draw();
  $("gridInfo").textContent=`${width}×${height} depth, ${width*height} GPU points, ${this.count/3} GPU triangles`;
 }
 resize(){
  const dpr=Math.min(devicePixelRatio||1,2);
  const rect=this.canvas.getBoundingClientRect();
  const w=Math.max(1,Math.min(2048,Math.round(rect.width*dpr)));
  const h=Math.max(1,Math.min(2048,Math.round(rect.height*dpr)));
  if(this.canvas.width!==w)this.canvas.width=w;
  if(this.canvas.height!==h)this.canvas.height=h;
 }
 draw(){
  if(!this._bind||!this.count)return;
  this.resize();
  this.device.queue.writeBuffer(this.camera,0,params32(
    ["u32","u32","f32","f32","f32","f32","f32","f32"],
    [this.width,this.height,this.yaw,this.pitch,this.zoom,
     this.canvas.width/this.canvas.height,0,0]
  ));
  const encoder=this.device.createCommandEncoder();
  const view=this.context.getCurrentTexture().createView();
  const pass=encoder.beginRenderPass({
    colorAttachments:[{
      view,clearValue:{r:0.022,g:0.034,b:0.060,a:1.0},
      loadOp:"clear",storeOp:"store"
    }]
  });
  pass.setPipeline(this.pipeline);pass.setBindGroup(0,this._bind);
  pass.draw(this.count);pass.end();
  this.device.queue.submit([encoder.finish()]);
 }
 dispose(){
   this.rt.release(...this.buffers);this.buffers=[];
 }
}

const canvas=$("gpuCanvas");
let last=null;
canvas.addEventListener("pointerdown",e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener("pointermove",e=>{
 if(!last||!renderer)return;
 const dx=e.clientX-last[0],dy=e.clientY-last[1];
 renderer.yaw+=dx*0.006;
 renderer.pitch=Math.max(-1.45,Math.min(1.45,renderer.pitch+dy*0.006));
 last=[e.clientX,e.clientY];renderer.draw();
});
canvas.addEventListener("pointerup",()=>last=null);
canvas.addEventListener("pointercancel",()=>last=null);
canvas.addEventListener("wheel",e=>{
 e.preventDefault();if(!renderer)return;
 renderer.zoom=Math.max(.2,Math.min(5,renderer.zoom*Math.exp(e.deltaY*.001)));
 renderer.draw();
},{passive:false});
new ResizeObserver(()=>renderer?.draw()).observe(canvas);

function syntheticDepth(width,height,amplitude) {
 const values=new Float32Array(width*height);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const u=x/(width-1),v=y/(height-1);
  const distance=Math.hypot(u-.5,v-.5);
  values[y*width+x]=1.2+
   amplitude*Math.sin(u*Math.PI*5)*Math.cos(v*Math.PI*5)*Math.exp(-distance*1.1);
 }
 return values;
}

async function loadDepthFile(file){
 const bitmap=await createImageBitmap(file);
 const w=Math.min(bitmap.width,512),h=Math.min(bitmap.height,512);
 const surface=document.createElement("canvas");
 surface.width=w;surface.height=h;
 const ctx=surface.getContext("2d",{willReadFrequently:true});
 ctx.drawImage(bitmap,0,0,w,h);
 const rgba=ctx.getImageData(0,0,w,h).data;
 const depths=new Float32Array(w*h);
 for(let i=0;i<depths.length;i++)depths[i]=rgba[i*4+3]<128?0:0.3+rgba[i*4]/255*2.0;
 bitmap.close();return [depths,w,h];
}

async function createRenderer(){
 await ensureEngine();
 if(renderer)return renderer;
 renderer=new DepthGridRenderer(engine.rt,canvas);
 await renderer.init();
 return renderer;
}
function guardButton(id,fn){
 $(id).addEventListener("click",async()=>{
  $(id).disabled=true;
  try{await fn();}
  catch(e){status("ERROR: "+e.message);log("ERROR "+(e.stack||e.message));}
  finally{$(id).disabled=false;}
 });
}
guardButton("runTests",async()=>{
 await ensureEngine();
 $("log").textContent="";
 status("Running actual GPU kernels and CPU reference comparisons...");
 const start=performance.now();
 const result=await runBrowserSuite(m=>log(JSON.stringify(m)));
 status(`${result.count} WebGPU correctness tests passed in ${((performance.now()-start)/1000).toFixed(2)}s`);
});
guardButton("loadSynthetic",async()=>{
 const width=Number($("width").value),height=Number($("height").value);
 const amp=Number($("amp").value);
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<2||height<2||width*height>262144)
   throw new RangeError("Use a grid 2×2 through 512×512");
 if(!Number.isFinite(amp)||amp<0||amp>1.0)throw new RangeError("Amplitude must be between 0 and 1");
 const r=await createRenderer();
 await r.loadDepth(syntheticDepth(width,height,amp),width,height);
 status("Synthetic geometry computed and rendered by WebGPU");
});
guardButton("loadPNG",async()=>{
 const file=$("depthFile").files?.[0];
 if(!file)throw new Error("Select a grayscale PNG depth map first");
 const [depth,w,h]=await loadDepthFile(file);
 const r=await createRenderer();
 await r.loadDepth(depth,w,h);
 status("Image depth converted to GPU-resident 3D grid; not AI inference");
});
guardButton("screenshot",()=>{
 const a=document.createElement("a");
 a.download="i3d_webgpu_depth_preview.png";
 // Preserve drawing buffer is not guaranteed; use a second copy from current texture
 // Not all WebGPU contexts support toDataURL after presentation.
 a.href=canvas.toDataURL("image/png");a.click();
});
window.addEventListener("pagehide",()=>{renderer?.dispose();engine?.destroy();});
status("Ready. Run correctness tests or generate GPU geometry.");
