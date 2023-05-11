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

/**
 * Given a 1D texture containing wave function values, draw those values onto
 * a canvas with the texture coordinates along the x axis and the function
 * values along the y axis. Attach the canvas to the parent.
 */
function SchrodingerResults(gpgpUtility_, parent_, color_, yResolution_, psiMax_)
{
  "use strict";

  var color;
  var colorHandle;
  var gpgpUtility;
  var parent;
  var positionHandle;
  var potentialHandle;
  var program;
  var psiMax;
  var psiMaxHandle;
  var waveFunction;
  var waveFunctionHandle;
  var textureCoordHandle;
  var xResolution;
  var yResolution;
  var yResolutionHandle;

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
                         + "uniform vec4 color;"
                         + "// Y scale for the plot\n"
                         + " uniform float psiMax;"
                         + "// waveFunction.r is the real part waveFunction.g is the imiginary part.\n"
                         + "uniform sampler2D waveFunction;"
                         + "// The number of points along the y axis.\n"
                         + "uniform int yResolution;"
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "void main()"
                         + "{"
                         + "  float absPsi2;"
                         + "  float halfDeltaY;"
                         + "  vec2  psi;"
                         + "  float psiMax2;"
                         + ""
                         + "  psiMax2 = psiMax*psiMax;"
                         + "  halfDeltaY = 0.5/float(yResolution);"
                         + ""
                         + "  psi     = texture2D(waveFunction, vTextureCoord).rg;"
                         + "  absPsi2 = psi.r*psi.r + psi.g*psi.g;"
                         + ""
                         + "  gl_FragColor = texture2D(waveFunction, vTextureCoord);"
    //+ "  gl_FragColor = vec4(psiMax2*255.0, vTextureCoord.t*255.0, halfDeltaY*255.0, absPsi2*255.0);"
    //+ "  gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0) + color*step(psiMax2*(vTextureCoord.y-halfDeltaY), absPsi2)"
    //                     + "                 - color*step(psiMax2*(vTextureCoord.y+halfDeltaY), absPsi2);"
                         + "}";

    program            = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle     = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    potentialHandle    = gl.getUniformLocation(program, "potential");
    waveFunctionHandle = gl.getUniformLocation(program, "waveFunction");
    psiMaxHandle       = gl.getUniformLocation(program, "psiMax");
    colorHandle        = gl.getUniformLocation(program, "color");

    return program;
  };

  /**
   * Setup for rendering to the screen. Create a canvas, get a rendering context,
   * set uniforms.
   */
  this.setup = function(gpgpUtility, color, psiMax)
  {
    var gl;
    gl = gpgpUtility.getGLContext();

    gl.useProgram(program);

    gl.uniform4fv(colorHandle, new Float32Array(color));
    gl.uniform1f(psiMaxHandle, psiMax);
  }

  /**
   * Map the waveFunction texture onto a curve
   *
   * @param waveFunction {WebGLTexture} A xResolution by 1 texture containing the real
   *                                    and imaginary parts of the wave function.
   */
  this.show = function(waveFunction)
  {
    var gl;
    gl = gpgpUtility.getRenderingContext();

    gl.useProgram(program);

    // This time we will render to the screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.uniform1i(yResolutionHandle, yResolution);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, waveFunction);
    gl.uniform1i(waveFunctionHandle, 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  color        = color_;
  gpgpUtility  = gpgpUtility_;
  yResolution  = yResolution_;
  parent       = parent_;
  psiMax       = psiMax_;

  program      = this.createProgram(gpgpUtility.getGLContext());
  this.setup(gpgpUtility, color, psiMax);

}