// Browser GPU correctness tests vs independently implemented CPU math.
// A passing CPU Node test NEVER counts as passing WebGPU tests.
import {I3DWebGPU,packSignedI4,unpackSignedI4,quantizeRowsI4,packTernary2,unpackTernary2} from "./ops.js";

const f = values => Float32Array.from(values);
function near(actual,reference,label,abs=2e-4,rel=2e-4){
 if(actual.length!==reference.length)throw new Error(label+" lengths differ");
 let maxError=0;
 for(let i=0;i<actual.length;i++){
   const a=actual[i],b=reference[i];const error=Math.abs(a-b);
   if(!Number.isFinite(a)||error>abs+rel*Math.abs(b))
     throw new Error(label+" mismatch at "+i+": GPU="+a+" CPU="+b+" diff="+error);
   maxError=Math.max(maxError,error);
 }
 return maxError;
}
const benchmark = async (fn, repeats=4) => {
  // Includes host uploads/readbacks; not a kernel-only GPU timestamp.
  await fn();
  const samples=[];
  for(let i=0;i<repeats;i++){const start=performance.now();await fn();samples.push(performance.now()-start);}
  samples.sort((a,b)=>a-b);
  return +samples[Math.floor(samples.length/2)].toFixed(3);
};

export async function runBrowserSuite(report=()=>{}) {
 const engine=await I3DWebGPU.create();
 const results=[];
 async function test(name,gpu,cpu,{abs=2e-4,rel=2e-4}={}){
  const start=performance.now();
  const expected=cpu();
  const actual=await gpu();
  const error=near(actual,expected,name,abs,rel);
  const latency=+(performance.now()-start).toFixed(3);
  const metric={name,ok:true,maxAbsError:error,elapsedMs:latency};
  results.push(metric);report(metric);
 }
 try{
  const M=19,N=11,K=13;
  const A=f(Array.from({length:M*K},(_,i)=>Math.sin(i*0.13)));
  const B=f(Array.from({length:K*N},(_,i)=>Math.cos(i*0.09)));
  await test("matmul 19×13×11",()=>engine.matmul(A,B,M,N,K),()=>{
   const out=new Float32Array(M*N);
   for(let m=0;m<M;m++)for(let n=0;n<N;n++){
    let v=0;for(let k=0;k<K;k++)v+=A[m*K+k]*B[k*N+n];out[m*N+n]=v;
   }return out;
  },{abs:5e-4});
  const rows=5,cols=13;
  const q=new Int8Array(rows*cols);
  for(let i=0;i<q.length;i++)q[i]=(i*7%16)-8;
  const packed=packSignedI4(q,rows,cols);
  const qback=unpackSignedI4(packed,rows,cols);
  if(qback.some((v,i)=>v!==q[i]))throw new Error("Q4 host packing mismatch");
  const scales=f([0.3,0.1,0.4,0.05,0.9]);
  const vector=f(Array.from({length:cols},(_,i)=>i/7-0.7));
  await test("signed Q4 GEMV unaligned K=13",()=>engine.q4Matvec(packed,vector,scales,rows,cols),()=>{
   const out=new Float32Array(rows);for(let r=0;r<rows;r++){
    let v=0;for(let k=0;k<cols;k++)v+=q[r*cols+k]*vector[k];
    out[r]=v*scales[r];
   }return out;
  });
  const trows=3,tcols=19;
  const ternary=new Int8Array(trows*tcols);
  for(let i=0;i<ternary.length;i++)ternary[i]=(i%3)-1;
  const ternaryPacked=packTernary2(ternary,trows,tcols);
  const roundtrip=unpackTernary2(ternaryPacked,trows,tcols);
  if(roundtrip.some((v,i)=>v!==ternary[i]))throw new Error("Ternary 2-bit host pack roundtrip mismatch");
  const tin=f(Array.from({length:tcols},(_,i)=>Math.cos(i*0.2)));
  const tscale=f([0.1,0.3,0.05]);
  await test("2-bit ternary GEMV K=19",()=>engine.ternaryMatvec(ternaryPacked,tin,tscale,trows,tcols),()=>{
    const o=new Float32Array(trows);
    for(let row=0;row<trows;row++){
      let v=0;for(let k=0;k<tcols;k++)v+=ternary[row*tcols+k]*tin[k];
      o[row]=v*tscale[row];
    }
    return o;
  });
  const x=f([1,2,3,4,-1,-2,-3,0.25,10,3,4,5,6,7]);
  await test("stable softmax 2×7",()=>engine.softmax(x,2,7),()=>{
   const o=new Float32Array(14);
   for(let r=0;r<2;r++){
    const a=Array.from(x.slice(r*7,r*7+7));const mx=Math.max(...a);
    const denom=a.reduce((s,v)=>s+Math.exp(v-mx),0);
    for(let j=0;j<7;j++)o[r*7+j]=Math.exp(a[j]-mx)/denom;
   }return o;
  });
  const gamma=f([0.9,0.5,1.1,1.2,-0.8,1.0,0.6]),beta=f([0,.1,.2,-.2,.3,.4,.5]);
  const refLayer=()=>{
    const o=new Float32Array(14);for(let r=0;r<2;r++){
      const vals=Array.from(x.slice(r*7,r*7+7));
      const mean=vals.reduce((a,b)=>a+b,0)/7;
      const variance=vals.reduce((a,b)=>a+(b-mean)**2,0)/7;
      for(let j=0;j<7;j++)o[r*7+j]=(vals[j]-mean)/Math.sqrt(variance+1e-5)*gamma[j]+beta[j];
    }return o;
  };
  await test("layernorm 2×7",()=>engine.layernorm(x,gamma,beta,2,7),refLayer,{abs:4e-4});
  await test("rmsnorm 2×7",()=>engine.rmsnorm(x,gamma,2,7),()=>{
    const o=new Float32Array(14);for(let r=0;r<2;r++){
      let ss=0;for(let j=0;j<7;j++)ss+=x[r*7+j]**2;
      for(let j=0;j<7;j++)o[r*7+j]=x[r*7+j]/Math.sqrt(ss/7+1e-5)*gamma[j];
    }return o;
  });
  await test("SiLU",()=>engine.silu(x),()=>f(Array.from(x,z=>z/(1+Math.exp(-z)))));
  const fg=f([1,.2,.3,.75, 0,.8,1,.5, .7,.4,.2,0,1,1,1,1]);
  const bg=f([0,1,0,1, 1,0,0,.5, 0,0,1,.2,0,0,0,0]);
  await test("straight-alpha RGBA composition",()=>engine.alphaComposite(fg,bg),()=>{
    const o=new Float32Array(fg.length);
    for(let i=0;i<fg.length;i+=4){
      const a=Math.max(0,Math.min(1,fg[i+3])),ba=Math.max(0,Math.min(1,bg[i+3]));
      const outa=a+ba*(1-a);
      for(let c=0;c<3;c++)o[i+c]=outa>0?(fg[i+c]*a+bg[i+c]*ba*(1-a))/outa:0;
      o[i+3]=outa;
    }return o;
  });
  const depth=f([1,1,1,1,1,1,1,1,0]);
  const pts=await engine.depthUnproject(depth,3,3,2,2,1,1);
  const cpuPts=new Float32Array(36);
  for(let i=0;i<depth.length;i++) {
    if(depth[i]<=0)continue;
    cpuPts[4*i]=((i%3)-1)*depth[i]/2;
    cpuPts[4*i+1]=(Math.floor(i/3)-1)*depth[i]/2;
    cpuPts[4*i+2]=depth[i];cpuPts[4*i+3]=1;
  }
  near(pts,cpuPts,"depth points");
  report({name:"depth unprojection 3×3",ok:true,maxAbsError:0});results.push({name:"depth unprojection 3×3",ok:true});
  await test("cross-product normals 3×3",()=>engine.normalsFromPoints(pts,3,3),()=>{
    const o=new Float32Array(36);
    for(let y=0;y<2;y++)for(let x=0;x<2;x++){
      const i=y*3+x,j=i+1,k=i+3;
      if(!pts[i*4+3]||!pts[j*4+3]||!pts[k*4+3])continue;
      const a=[0,1,2].map(c=>pts[j*4+c]-pts[i*4+c]);
      const b=[0,1,2].map(c=>pts[k*4+c]-pts[i*4+c]);
      const cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
      const len=Math.hypot(...cross);if(len>1e-10){for(let c=0;c<3;c++)o[i*4+c]=cross[c]/len;o[i*4+3]=1;}
    }return o;
  });
  const positions=f([1,2,3,1,2,3,4,1]);
  const ids=new Uint32Array([0,1,0,0,1,0,0,0]);
  const weights=f([.25,.75,0,0,1,0,0,0]);
  const identity=f([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
  const shifted=f([1,0,0,0,0,1,0,0,0,0,1,0,2,0,-1,1]);
  const joints=f([...identity,...shifted]);
  await test("30-joint-compatible skinning (2-joint test)",()=>engine.skinLBS(positions,ids,weights,joints,2,2),
    ()=>f([2.5,2,2.25,1,4,3,3,1]));
  const quant=quantizeRowsI4(f([0,-.2,.5,1,.6,-.9]),2,3);
  if(quant.packed.length!==2||quant.scales.length!==2)throw new Error("Quantization host smoke failed");
  const benchMs=await benchmark(()=>engine.matmul(A,B,M,N,K));
  const summary={ok:true,count:results.length,benchmarks:{smallMatmulHostRoundtripMedianMs:benchMs},
    adapter:engine.info()};
  report({name:"FINAL",...summary});return summary;
 }finally{engine.destroy();}
}
