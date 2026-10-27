// Custom WebGPU triangle-list renderer for depth-unprojected point grids.
// Reads the SAME GPU-resident point + normal buffers written by WGSL compute.
// No Three.js/WebGL fallback and no CPU geometry reconstruction.
struct Camera {width:u32,height:u32,yaw:f32,pitch:f32,zoom:f32,aspect:f32,pad0:f32,pad1:f32};
@group(0) @binding(0) var<storage,read> positions:array<f32>;
@group(0) @binding(1) var<storage,read> normals:array<f32>;
@group(0) @binding(2) var<uniform> camera:Camera;
struct Vertex { @builtin(position) clip:vec4<f32>,@location(0) color:vec3<f32> };
@vertex fn vs_main(@builtin(vertex_index) id:u32)->Vertex{
 let cell=id/6u;let corner=id%6u;
 let x=cell%(camera.width-1u);let y=cell/(camera.width-1u);
 var dx=0u;var dy=0u;
 switch corner {
   case 1u, 3u, 4u: {dx=1u;}
   default: {}
 }
 switch corner {
   case 2u, 4u, 5u: {dy=1u;}
   default: {}
 }
 let index=(y+dy)*camera.width+(x+dx);
 let start=index*4u;
 let validity=positions[start+3u];
 var o:Vertex;
 if(validity<0.5){
   o.clip=vec4<f32>(3.0,3.0,0.5,1.0);
   o.color=vec3<f32>(0.0);
   return o;
 }
 let px=positions[start];let py=positions[start+1u];
 let pz=positions[start+2u]-1.2;
 let cy=cos(camera.yaw);let sy=sin(camera.yaw);
 let cp=cos(camera.pitch);let sp=sin(camera.pitch);
 let rx=cy*px+sy*pz;let rz=-sy*px+cy*pz;
 let ry=cp*py-sp*rz;let rzz=sp*py+cp*rz;
 o.clip=vec4<f32>(rx/(camera.zoom*camera.aspect),-ry/camera.zoom,
                   clamp(0.5+rzz*0.05,0.05,0.95),1.0);
 let n=vec3<f32>(normals[start],normals[start+1u],normals[start+2u]);
 o.color=vec3<f32>(0.15,0.29,0.50)+0.6*abs(n);
 return o;
}
@fragment fn fs_main(@location(0) color:vec3<f32>)->@location(0) vec4<f32>{
 return vec4<f32>(color,1.0);
}
