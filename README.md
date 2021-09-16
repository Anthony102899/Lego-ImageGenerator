# Lego-ImageGenerator
## About The Project
The project is about automatically generate LEGO® Image by importing bit image.

## Project Member
- LYU, An
- Ding, Baizeng

# Appendix
## Appendix 1: LDaw Representation of LEGO part
When Line type of the block is 1, which is most of the bricks in project, the format shoube be like:  
1 <colour> x y z a b c d e f g h i <file>,  
- <colour> is a number representing the colour of the part. See the Colours section for allowable colour numbers.
- x y z is the x y z coordinate of the part.
- a b c d e f g h iis a top left 3x3 matrix of a standard 4x4 homogeneous transformation matrix. This represents the rotation and scaling of the part. The entire 4x4 3D transformation matrix would then take either of the following forms:  
  / a d g 0 \     / a b c x \  
  | b e h 0 |     | d e f y |  
  | c f i 0 |     | g h i z |  
  \ x y z 1 /     \ 0 0 0 1 /  
The above two forms are essentially equivalent, but note the location of the transformation portion (x, y, z) relative to the other terms.  
Formally, the transformed point (u', v', w') can be calculated from point (u, v, w) as follows:  
- u' = a*u + b*v + c*w + x  
- v' = d*u + e*v + f*w + y  
- w' = g*u + h*v + i*w + z  
- <file> is the filename of the sub-file referenced and must be a valid LDraw filename. Any leading and/or trailing whitespace must be ignored. Normal token separation is otherwise disabled for the filename value.  
