/**
 * Copyright 2016 Vizit Solutions
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

/**
 * Set initial values for the wave function texture, which will be
 * subsequently evolved in time.
 */
function SchrodingerInitializer(gpgpUtility_)
{
  "use strict";

  var framebuffer;
  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var kHandle;
  var lengthHandle;
  var positionHandle;
  var program;
  var textureCoordHandle;
  var x0Handle;
  var wHandle;

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
                         + "#define PI 3.1415926535897932384626433832795\n"
                         + ""
                         // The physical length of the grid in nm.
                         + "uniform float length;"
                         // center of the wave packet
                         + "uniform float x0;"
                         // width of the wave packet
                         + "uniform float w;"
                         // wave number
                         + "uniform float k;"
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "vec2 computePsi(float s)"
                         + "{"
                         + "  float x = length*s;"
                         // Generalization of gaussian width
                         + "  float alpha = w*w;"
                         + "  float deltaX = x-x0;"
                         // Normalization constant http://quantummechanics.ucsd.edu/ph130a/130_notes/node80.html
                         + "  float a = pow(2.0/(PI*alpha), 0.25);"
                         + "  float gaussian = a*exp(-deltaX*deltaX/alpha);"
                         + "  vec2  phase = vec2(cos(k*x), sin(k*x));"
                         + "  return gaussian*phase;"
                         + "}"
                         + ""
                         + "void main()"
                         + "{"
                         + "  gl_FragColor.rg = computePsi(vTextureCoord.s);"
                         + "}";

    program              = gpgpUtility.createProgram(null, fragmentShaderSource);

    positionHandle       = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle   = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    kHandle              = gpgpUtility.getUniformLocation(program, "k");
    lengthHandle         = gpgpUtility.getUniformLocation(program, "length");
    x0Handle             = gpgpUtility.getUniformLocation(program, "x0");
    wHandle              = gpgpUtility.getUniformLocation(program, "w");

    return program;
  };

  /**
   * Runs the program to do the actual work. On exit the framebuffer &amp;
   * texture are populated with the values computed in the fragment shader.
   * Use gl.readPixels to retrieve texture values.
   */
  this.initialize = function(framebuffer, length, k, x0, w)
  {
    var gl;

    gl = gpgpUtility.getComputeContext();

    gl.useProgram(program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.uniform1f(lengthHandle, length);
    gl.uniform1f(kHandle,      k);
    gl.uniform1f(x0Handle,     x0);
    gl.uniform1f(wHandle,      w);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in followon calculations.
   */
  this.done = function ()
  {
    gl.deleteProgram(program);
  };

  this.getPixels = function()
  {
      var buffer;

      // One each for RGBA component of each pixel
      buffer = new Float32Array(128*128*4);
      // Read a 1x1 block of pixels, a single pixel
      gl.readPixels(0,       // x-coord of lower left corner
                    0,       // y-coord of lower left corner
                    128,     // width of the block
                    128,     // height of the block
                    gl.RGBA, // Format of pixel data.
                    gl.FLOAT,// Data type of the pixel data, must match makeTexture
                    buffer); // Load pixel data into buffer

    return buffer;
  }

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
    var eps;
    var expected;
    var passed;

    // Error tollerance in calculations
    eps = 1.0E-20;

    // One each for RGBA component of a pixel
    buffer = new Float32Array(4);
    // Read a 1x1 block of pixels, a single pixel
    gl.readPixels(i,       // x-coord of lower left corner
                  j,       // y-coord of lower left corner
                  1,       // width of the block
                  1,       // height of the block
                  gl.RGBA, // Format of pixel data.
                  gl.FLOAT,// Data type of the pixel data, must match makeTexture
                  buffer); // Load pixel data into buffer

    expected = i*1000.0 + j;

    passed   = expected === 0.0 ? buffer[0] < eps : Math.abs((buffer[0] - expected)/expected) < eps;

    if (!passed)
    {
	alert("Read " + buffer[0] + " at (" + i + ", " + j + "), expected " + expected + ".");
    }

    return passed;
  };

  gpgpUtility = gpgpUtility_;
  gl          = gpgpUtility.getGLContext();
  program     = this.createProgram(gl);
};