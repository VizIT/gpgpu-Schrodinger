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


function ToUnsignedBytes(gpgpUtility_)
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
                         + "void main()"
                         + "{"
                         + "          vec4 data = texture2D(texture, vTextureCoord);"
                         + "          gl_FragColor = pack(data.r);"
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
   * Runs the program to do the actual work. On exit the framebuffer &amp;
   * texture are populated with the values drawn from the input texture,
   * but packed into an RGBA UNSIGNED_BYTE format. Use gl.readPixels to
   * retrieve texture values.
   */
  this.convert = function(width, height, texture)
  {
    gl.useProgram(program);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    //gl.uniform1f(widthHandle,  width);
    //gl.uniform1f(heightHandle, height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(textureHandle, 0);


    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  /**
   * Read back the i, j pixel and compare it with the expected value. The expected value
   * computation matches that in the fragment shader.
   * 
   * @param i {integer} the i index of the matrix.
   * @param j {integer} the j index of the matrix.
   */
  this.test = function(i, j)
  {
    var buffer;
    var expected;
    var floatingPoint;
    var passed;

    // One each for RGBA component of a pixel
    buffer = new Uint8Array(4);
    // Read a 1x1 block of pixels, a single pixel
    gl.readPixels(i,                // x-coord of lower left corner
                  j,                // y-coord of lower left corner
                  1,                // width of the block
                  1,                // height of the block
                  gl.RGBA,          // Format of pixel data.
                  gl.UNSIGNED_BYTE, // Data type of the pixel data, must match makeTexture
                  buffer);          // Load pixel data into buffer

    floatingPoint = new Float32Array(buffer.buffer);

    expected = i*1000.0 + j;

    passed   = (floatingPoint[0] === expected);

    if (!passed)
    {
      alert("Read " + floatingPoint[0] + " at (" + i + ", " + j + "), expected " + expected + ".");
    }

    return passed;
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