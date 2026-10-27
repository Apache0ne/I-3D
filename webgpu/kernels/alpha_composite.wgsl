// Straight-alpha RGBA over operation, both source and background are f32 RGBA.
struct Shape {pixels:u32,reserved0:u32,reserved1:u32,reserved2:u32};
@group(0) @binding(0) var<storage,read> fg:array<f32>;
@group(0) @binding(1) var<storage,read> bg:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
@group(0) @binding(3) var<uniform> shape:Shape;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let i=gid.x;if(i>=shape.pixels){return;}
 let k=i*4u;let a=clamp(fg[k+3u],0.0,1.0);
 let ba=clamp(bg[k+3u],0.0,1.0);let oa=a+ba*(1.0-a);
 for(var c=0u;c<3u;c=c+1u){
   let pre=fg[k+c]*a+bg[k+c]*ba*(1.0-a);
   out[k+c]=select(0.0,pre/max(oa,1e-12),oa>0.0);
 }
 out[k+3u]=oa;
}
