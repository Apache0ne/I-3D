// f32 SiLU activation, common in diffusion/UNet/transformer MLP blocks.
struct Shape { count:u32, reserved0:u32, reserved1:u32, reserved2:u32 };
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read_write> y:array<f32>;
@group(0) @binding(2) var<uniform> shape:Shape;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let i=gid.x; if(i<shape.count) {let v=x[i];y[i]=v/(1.0+exp(-v));}
}
