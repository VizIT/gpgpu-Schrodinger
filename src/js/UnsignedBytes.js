/**
 * Copyright 2015 Vizit Solutions
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */


function UnsignedBytes(gpgpUtility_)
{
  "use strict";

  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var positionHandle;
  var program;
  var textureCoordHandle;
  var textureHandle;

  /**
   * Compile shaders and link them into a program, then retrieve references to the
   * attributes and uniforms. The standard vertex shader, which simply passes on the
   * physical and texture coordinates, is used.
   *
   * @returns {WebGLProgram} The created program object.
   * @see {https://www.khronos.org/registry/webgl/specs/1.0/#5.6|WebGLProgram}
   */
  this.createProgram = function (gl)
  {
    var fragmentShaderSource;
    var program;

    // Note that the preprocessor requires the newlines.
    fragmentShaderSource = "#ifdef GL_FRAGMENT_PRECISION_HIGH\n"
                         + "precision highp float;\n"
                         + "#else\n"
                         + "precision mediump float;\n"
                         + "#endif\n"
                         + ""
                         + "uniform sampler2D texture;"
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "vec4 pack(float value)"
                         + "{"
                         + "  if (value == 0.0) return vec4(0, 0, 0, 0);"
                         + ""
                         + "  float exponent;"
                         + "  float mantissa;"
                         + "  vec4  result;"
                         + "  float sgn;"
                         + ""
                         + "  sgn = step(0.0, -value);"
                         + "  value = abs(value);"
                         + ""
                         + "  exponent =  floor(log2(value));"
                         + ""
                         + "  mantissa =  value*pow(2.0, -exponent)-1.0;"
                         + "  exponent =  exponent+127.0;"
                         + "  result   = vec4(0,0,0,0);"
                         + ""
                         + "  result.a = floor(exponent/2.0);"
                         + "  exponent = exponent - result.a*2.0;"
                         + "  result.a = result.a + 128.0*sgn;"
                         + ""
                         + "  result.b = floor(mantissa * 128.0);"
                         + "  mantissa = mantissa - result.b / 128.0;"
                         + "  result.b = result.b + exponent*128.0;"
                         + ""
                         + "  result.g =  floor(mantissa*32768.0);"
                         + "  mantissa = mantissa - result.g/32768.0;"
                         + ""
                         + "  result.r = floor(mantissa*8388608.0);"
                         + ""
                         + "  return result/255.0;"
                         + "}"
                         + ""
                         + "float unpack(vec4 texel)"
                         + "{"
                         + "  float exponent;"
                         + "  float mantissa;"
                         + "  float sgn;"
                         + "  float value;"
                         + ""
                         + "  /* sgn will be 0 or -1 */"
                         + "  sgn = -step(128.0, texel.a);"
                         + "  texel.a += 128.0*sgn;"
                         + ""
                         + "  exponent = step(128.0, texel.b);"
                         + "  texel.b -= exponent*128.0;"
                         + "  /* Multiple by 2 => left shift by one bit. */"
                         + "  exponent += 2.0*texel.a -127.0;"
                         + ""
                         + "  mantissa = texel.b*65536.0 + texel.g*256.0 + texel.r;"
                         + ""
                         + "  value = sgn * exp2(exponent)*(1.0 + mantissa * exp2(-23.0));"
                         + ""
                         + "  return value;"
                         + "}"
                         + ""
                         + "float computations(float value)"
                         + "{"
                         + "/* Computations go here*/"
                         + "}"
                         + ""
                         + "void main()"
                         + "{"
                         + "  vec4 texel;"
                         + "  float value;"
                         + ""
                         + "  texel = texture2D(texture, vTextureCoord);"
                         + "  value = unpack(texel);"
                         + ""
                         + "  value = computations(value);"
                         + ""
                         + "  gl_FragColor = pack(value);"
                         + "}";

    program            = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle     = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    textureHandle      = gl.getUniformLocation(program, "texture");

    return program;
  };

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in followon calculations.
   */
  this.done = function ()
  {
    gl.deleteProgram(program);
  };

  gpgpUtility = gpgpUtility_;
  gl          = gpgpUtility.getGLContext();
  program     = this.createProgram(gl);
};