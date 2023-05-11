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


function Square128(gpgpUtility_)
{
  "use strict";

  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var pixels;
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
    fragmentShaderSource = //"#ifdef GL_FRAGMENT_PRECISION_HIGH\n"
                         "precision highp float;\n"
    //+ "#else\n"
    //+ "precision mediump float;\n"
    //+ "#endif\n"
                         + ""
                         + "uniform sampler2D m;"
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "vec4 computeElement(float s, float t)"
                         + "{"
                         + "  float i = floor(128.0*s);"
                         + "  float j = floor(128.0*t);"
                         + "  return vec4(i*1000.0 + j, 0.0, 0.0, 0.0);"
                         + "}"
                         + ""
                         + "void main()"
                         + "{"
                         + "  float i, j;"
                         + "  float value = 0.0;"
                         + ""
                         + "  i = vTextureCoord.x;"
                         + "  j = vTextureCoord.y;"
                         + ""
                         + "  for(float k=0.0; k<128.0; ++k)"
                         + "  {"
    //+ "    value = texture2D(m, vec2(i, j)).r;"
    //+ "    value = computeElement(i, j).r;"
    //+ "    value = texture2D(m, vec2(i, j)).r - computeElement(i, j).r;"
    + "    value +=  computeElement(i, k/128.0).r*computeElement(k/128.0, j).r;"
             // + "    value += texture2D(m, vec2(i, k/128.0)).r * texture2D(m, vec2(k/128.0, j)).r;"
                         + "  }"
                         + "  gl_FragColor.r = value;"
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
   * texture are populated with the square of the input matrix, m. Use
   * gl.readPixels to retrieve texture values.
   *
   * @param m        {WebGLTexture} A texture containing the elements of m.
   * @param mSquared {WebGLTexture} A texture to be incorporated into a fbo,
   *                                the target for our operations.
   */
  this.square = function(m, mSquared)
  {
    var m2FrameBuffer;

    // Create and bind a framebuffer
    m2FrameBuffer = gpgpUtility.attachFrameBuffer(mSquared);

    gl.useProgram(program);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, m);
    gl.uniform1i(textureHandle, 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  this.element = function(i, j)
  {
    return i*1000.0 + j;
  }

  /**
   * Read back the i, j pixel and compare it with the expected value. The expected value
   * computation matches that in the fragment shader.
   * 
   * @param i       {integer} the i index of the matrix element to be tested.
   * @param j       {integer} the j index of the matrix element to be tested.
   * @param display {HTMLTableElement} A table for test results.
   */
  this.test = function(i, j, display)
  {
    var buffer;
    var compare;
    var eps;
    var expected;
    var fromPixels;
    var passed;
    var ratio;
    var tableCell;
    var tableHeader;
    var tableRow;

    eps    = 1.0E-08;

    // One each for RGBA component of a pixel
    buffer = new Float32Array(4);
    // Read a 1x1 block of pixels, a single pixel
    gl.readPixels(i,                // x-coord of lower left corner
                  j,                // y-coord of lower left corner
                  1,                // width of the block
                  1,                // height of the block
                  gl.RGBA,          // Format of pixel data.
                  gl.FLOAT,         // Data type of the pixel data, must match makeTexture
                  buffer);          // Load pixel data into buffer

    compare    = 0.0;
    fromPixels = 0.0;

    for(var k=0.0; k<128.0; ++k)
    {
      compare     += this.element(i, k)*this.element(k, j);
    }

    ratio      = Math.abs((compare-buffer[0])/compare);

    passed     = ratio < eps;

    tableRow   = display.insertRow();
    // Coordinates column
    tableCell  = tableRow.insertCell();
    tableCell.appendChild(document.createTextNode("(" + i + ", " + j + ")"));
    // Found value column
    tableCell  = tableRow.insertCell();
    tableCell.appendChild(document.createTextNode(buffer[0]));
    // Expected value column
    tableCell  = tableRow.insertCell();
    tableCell.appendChild(document.createTextNode(compare));
    // Relative error
    tableCell  = tableRow.insertCell();
    tableCell.appendChild(document.createTextNode(ratio.toPrecision(2)));

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