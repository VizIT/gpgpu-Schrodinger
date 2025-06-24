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


function Matrix(gpgpUtility_, size_)
{
  "use strict";

  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var pixels;
  var positionHandle;
  var program;
  var size;
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
    var mSize;
    var program;

    // Note that the preprocessor requires the newlines.
    fragmentShaderSource ="#ifdef GL_FRAGMENT_PRECISION_HIGH\n"
                         + "precision highp float;\n"
                         + "#else\n"
                         + "precision mediump float;\n"
                         + "#endif\n"
                         + ""
                         // Ensure at least one decimal place, otherwise the compiler thinks it's an int
                         + "const float size = " + size.toFixed(1) + ";"
                         + ""
                         + "uniform sampler2D m;"
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "void main()"
                         + "{"
                         + "  float i, j, k;"
                         + "  float value1; "
                         + "  float value2;"
                         + "  float value = 0.0;"
                         + ""
                         + "  i = vTextureCoord.s;"
                         + "  j = vTextureCoord.t;"
                         + ""
                         + "  for(float l=0.0; l<size; ++l)"
                         + "  {"
                         + "    k = (l+0.5)/size;"
                         + "    value1 = texture2D(m, vec2(i, k)).r;"
                         + "    value2 = texture2D(m, vec2(k, j)).r;"
                         + "    value += value1*value2;"
                         + "  }"
                         + "  gl_FragColor.r = value;"
                         + "  gl_FragColor.g = value1;"
                         + "  gl_FragColor.b = value2;"
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
    // return i*1000.0 + j;
    return i + j;
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

    eps    = 1.0E-07;

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

    for(var k=0.0; k<size; ++k)
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

    if (!passed)
    {
      tableRow.classList.add("warn");
    }

    return passed;
  };

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in follow-on calculations.
   */
  this.done = function ()
  {
    gl.deleteProgram(program);
  };

  gpgpUtility = gpgpUtility_;
  size        = size_;

  gl          = gpgpUtility.getGLContext();
  program     = this.createProgram(gl);
}
