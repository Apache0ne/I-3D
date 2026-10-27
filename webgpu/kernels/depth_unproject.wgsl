// Depth -> camera-space 3D points (RGBA-like 4-float padded output).
// Inputs: positive metric depth; invalid/zero depth -> vec4(0).
struct Camera {width:u32,height:u32,fx:f32,fy:f32,cx:f32,cy:f32,pad0:u32,pad1:u32};
@group(0) @binding(0) var<storage,read> depth:array<f32>;
@group(0) @binding(1) var<storage,read_write> points:array<f32>;
@group(0) @binding(2) var<uniform> cam:Camera;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let i=gid.x;if(i>=cam.width*cam.height){return;}
 let z=depth[i];let k=i*4u;
 if(z<=0.0){points[k]=0.0;points[k+1u]=0.0;points[k+2u]=0.0;points[k+3u]=0.0;return;}
 let u=f32(i%cam.width);let v=f32(i/cam.width);
 points[k]=(u-cam.cx)*z/cam.fx;
 points[k+1u]=(v-cam.cy)*z/cam.fy;
 points[k+2u]=z;points[k+3u]=1.0;
}
