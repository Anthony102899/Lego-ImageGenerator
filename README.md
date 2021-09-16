# Lego-ImageGenerator
## About The Project
The project is about automatically generate LEGO® Image by importing bit image.

## Project Member
- LYU, An
- Ding, Baizeng

# Appendix
## Appendix 1: LDaw Representation of LEGO part
When Line type of the block is 1, which is most of the bricks in project, the format shoube be like:  
1 &lt;colour&gt; x y z a b c d e f g h i &lt;file&gt;,  
- &lt;colour&gt; is a number representing the colour of the part. See the Colours section for allowable colour numbers.
- x y z is the x y z coordinate of the part.
- a b c d e f g h iis a top left 3x3 matrix of a standard 4x4 homogeneous transformation matrix. This represents the rotation and scaling of the part. The entire 4x4 3D transformation matrix would then take either of the following forms:  
  / a d g 0 \ &nbsp;&nbsp; / a b c x \  
  | b e h 0 | &nbsp;&nbsp; &nbsp;| d e f y |  
  | c f i 0 | &nbsp;&nbsp; &nbsp;&nbsp;| g h i z |  
  \ x y z 1 / &nbsp;&nbsp; \ 0 0 0 1 /  

\begin{bmatrix}
a_{00}&a_{01}\\
a_{10}&a_{11}\\
\end{bmatrix}

The above two forms are essentially equivalent, but note the location of the transformation portion (x, y, z) relative to the other terms.  
Formally, the transformed point (u', v', w') can be calculated from point (u, v, w) as follows:  
    - u' = a*u + b*v + c*w + x  
    - v' = d*u + e*v + f*w + y  
    - w' = g*u + h*v + i*w + z  
- &lt;file&gt; is the filename of the sub-file referenced and must be a valid LDraw filename. Any leading and/or trailing whitespace must be ignored. Normal token separation is otherwise disabled for the filename value.  
