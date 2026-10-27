// Linear blend skinning. Input 4-float vertices, 4 influences, column-major
// joint matrices (the same layout as glTF / Three.js boneMatrices).
struct Shape {vertices:u32,joints:u32,pad0:u32,pad1:u32};
@group(0) @binding(0) var<storage,read> positions:array<f32>;
@group(0) @binding(1) var<storage,read> joint_ids:array<u32>;
@group(0) @binding(2) var<storage,read> weights:array<f32>;
@group(0) @binding(3) var<storage,read> matrices:array<f32>;
@group(0) @binding(4) var<storage,read_write> output:array<f32>;
@group(0) @binding(5) var<uniform> shape:Shape;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let v=gid.x;if(v>=shape.vertices){return;}
 let k=v*4u;
 let p=vec4<f32>(positions[k],positions[k+1u],positions[k+2u],positions[k+3u]);
 var total=vec4<f32>(0.0);var sum=0.0;
 for(var j:u32=0u;j<4u;j=j+1u) {
   let idx=joint_ids[k+j];let weight=max(weights[k+j],0.0);
   if(idx>=shape.joints||weight==0.0){continue;}
   let m=idx*16u;
   let tp=vec4<f32>(
     dot(vec4<f32>(matrices[m],matrices[m+4u],matrices[m+8u],matrices[m+12u]),p),
     dot(vec4<f32>(matrices[m+1u],matrices[m+5u],matrices[m+9u],matrices[m+13u]),p),
     dot(vec4<f32>(matrices[m+2u],matrices[m+6u],matrices[m+10u],matrices[m+14u]),p),
     dot(vec4<f32>(matrices[m+3u],matrices[m+7u],matrices[m+11u],matrices[m+15u]),p)
   );
   total=total+tp*weight;sum=sum+weight;
 }
 let q=select(p,total/max(sum,1e-12),sum>0.0);
 output[k]=q.x;output[k+1u]=q.y;output[k+2u]=q.z;output[k+3u]=q.w;
}
