// Vertex normals for the organized point cloud (world/camera coords).
struct Shape {width:u32,height:u32,pad0:u32,pad1:u32};
@group(0) @binding(0) var<storage,read> points:array<f32>;
@group(0) @binding(1) var<storage,read_write> normals:array<f32>;
@group(0) @binding(2) var<uniform> shape:Shape;
fn point(i:u32)->vec3<f32>{let k=i*4u;return vec3<f32>(points[k],points[k+1u],points[k+2u]);}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let i=gid.x;let total=shape.width*shape.height;if(i>=total){return;}
 let x=i%shape.width;let y=i/shape.width;let k=i*4u;
 if(x+1u>=shape.width||y+1u>=shape.height ||
    points[k+3u]==0.0||points[(i+1u)*4u+3u]==0.0||
    points[(i+shape.width)*4u+3u]==0.0){
   normals[k]=0.0;normals[k+1u]=0.0;normals[k+2u]=0.0;normals[k+3u]=0.0;return;
 }
 let p=point(i);let dx=point(i+1u)-p;let dy=point(i+shape.width)-p;
 let n=cross(dx,dy);let len=length(n);
 if(len<=1e-10){normals[k]=0.0;normals[k+1u]=0.0;normals[k+2u]=0.0;normals[k+3u]=0.0;return;}
 let nn=n/len;normals[k]=nn.x;normals[k+1u]=nn.y;normals[k+2u]=nn.z;normals[k+3u]=1.0;
}
