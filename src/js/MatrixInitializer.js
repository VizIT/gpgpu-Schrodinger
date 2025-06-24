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

function MatrixInitializer(gpgpUtility_)
{
  "use strict";

  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var heightHandle;
  var positionHandle;
  var program;
  var textureCoordHandle;
  var widthHandle;

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
                         + "uniform float height;"
                         + "uniform float width;"
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "vec4 computeElement(float s, float t)"
                         + "{"
                         + "  float i = floor(width*s);"
                         + "  float j = floor(height*t);"
                         + "  return vec4(i*1000.0 + j, 0.0, 0.0, 0.0);"
                         + "}"
                         + ""
                         + "void main()"
                         + "{"
                         + "  gl_FragColor = computeElement(vTextureCoord.s, vTextureCoord.t);"
                         + "}";

    // Null first argument to createProgram => use the standard vertex shader
    program              = gpgpUtility.createProgram(null, fragmentShaderSource);

    // position and textureCoord are attributes from the standard vertex shader
    positionHandle       = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle   = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    // Height and width are the problem specific variables
    heightHandle         = gpgpUtility.getUniformLocation(program, "height");
    widthHandle          = gpgpUtility.getUniformLocation(program, "width");

    return program;
  };

  /**
   * Runs the program to do the actual work. On exit the framebuffer &amp;
   * texture are populated with the values computed in the fragment shader.
   * Later, use gl.readPixels to retrieve texture values.
   */
  this.initialize = function(width, height)
  {
    gl.useProgram(program);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     // The attribute
                           3,                  // The three (x,y,z) elements in each value
                           gl.FLOAT,           // The data type, so each position is three floating point numbers
                           gl.FALSE,           // Are values normalized - unused for float
                           20,                 // Stride, the spacing, in bytes, between beginnings of successive values
                           0);                 // Offset 0, data starts at the beginning of the array
    gl.vertexAttribPointer(textureCoordHandle, // The attribute
                           2,                  // The two (s,t) elements in each value
                           gl.FLOAT,           // The data type, so each position is two floating point numbers
                           gl.FALSE,           // Are values normalized - unused for float
                           20,                 // Stride, the spacing, in bytes, between beginnings of successive values
                           12);                // Offset 12 bytes, data starts after the positional data

    gl.uniform1f(widthHandle,  width);
    gl.uniform1f(heightHandle, height);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in follow-on calculations.
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

    // Error tolerance in calculations
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
}
