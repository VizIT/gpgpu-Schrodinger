/**
 * Copyright 2016-2021 Vizit Solutions
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

"use strict";

/**
 * Given a 1D textures containing wave function and potential values, draw those
 * values onto a canvas with the texture coordinates along the x axis and the function
 * values along the y axis. Attach the canvas to the parent.
 */
function SchrodingerRenderer(gpgpUtility_, parent_, psiColor_, reColor_, imColor_, vColor_, xResolution_, yResolution_, E_, potential_, psiMax_, vMax_, width_=SchrodingerRenderer.DEFAULT_LINE_WIDTH)
{
  var E;
  var gpgpUtility;
  var width;
  // Color for the imaginary part of psi, 0,0,0,0 => nothing drawn.
  var imColor;
  var parent;
  var potential;
  var program;
  // psi*psi color
  var psiColor;
  var psiMax;
  // real psi component color, 0,0,0,0 => nothing drawn.
  var reColor;
  var time;
  // Color for the potential
  var vColor;
  // Max abs(v) for the potential
  var vMax;
  var xResolution;
  var yResolution;

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
    // Note that the preprocessor requires the newlines.
    const fragmentShaderSource = "#ifdef GL_FRAGMENT_PRECISION_HIGH\n"
                               + "precision highp float;\n"
                               + "#else\n"
                               + "precision mediump float;\n"
                               + "#endif\n"
                               + ""
                               // Energy of the initial wave packet
                               + "uniform float E;"
                               // Color for ImPsi: 0, 0, 0, 0 for no plot.
                               + "uniform vec4 imColor;"
                               // Psi*Psi color
                               + "uniform vec4 psiColor;"
                               // Y scale for the plot
                               + "uniform float psiMax;"
                               // Color for RePsi: 0.0, 0.0, 0.0, 0.0 for no plot.
                               + "uniform vec4 reColor;"
                               // waveFunction.r is the real part waveFunction.g is the imaginary part.
                               + "uniform sampler2D waveFunction;"
                               // Discrete representation of the potential function.
                               + "uniform sampler2D potential;"
                               // Color for the potential: 0.0, 0.0, 0.0, 0.0 for no plot
                               + "uniform vec4 vColor;"
                               + "uniform float vMax;"
                               // Number of points along the x axis
                               + "uniform int xResolution;"
                               // Number of points along the y axis.
                               + "uniform int yResolution;"
                               // Roughly corresponds to the rendered line width
                               + "uniform float width;"
                               + ""
                               + "varying vec2 vTextureCoord;"
                               + ""
                               /**
                                * Color pixels to represent the numerical values of a function on a grid. The function is
                                * confined to values in [-scale, +scale]. We also adjust the upper and lower bounds of the
                                * line as necessary to fill in discontinuities in the numerical values.
                                *
                                * @param {vec4} color  The color for a line of the given function
                                * @param {float} scale Possible values range from -scale to + scale
                                * @param {float} t     The t, vertical, texture coordinate of the fragment in question in
                                *                      the range [0, 1] bottom to top.
                                * @param {float} halfDeltaY 0.5/yResolution, a half step in the grid
                                * @param {float} width Roughly the line width in pixels
                                * @param {float} value The value of the function at the current position
                                * @param {float} previousValue
                                */
                               + "vec4 pixelColor(const vec4 color, const float scale, const float t, const float halfDeltaY,"
                               + "                const float width, const float value, const float previousValue)"
                               + "{"
                               + "  float scale2 = 2.0*scale;"
                               + ""
                               // The value for this t pixel in the s column.
                               + "  float pxvalue = scale2*t-scale;"
                               // We expect to begin fading in the color at the function value
                               // but adjust toward the previous value for continuity
                               + "  float loweredge = min(value, previousValue+scale2*halfDeltaY);"
                               // We expect to begin fading out the color at the function value
                               // but adjust toward the previous value for continuity
                               + "  float upperedge = max(value, previousValue-scale2*halfDeltaY);"
                               + ""
                               + "  return color*(smoothstep(loweredge - scale2*width*halfDeltaY,"
                               + "                           loweredge - scale2*halfDeltaY,"
                               + "                           pxvalue)"
                               + "                -smoothstep(upperedge + scale2*halfDeltaY,"
                               + "                            upperedge + scale2*width*halfDeltaY,"
                               + "                            pxvalue));"
                               + "}"
                               + ""
                               + "void main()"
                               + "{"
                               + "  vec4  background;"
                               + "  vec4  color;"
                               + "  float halfDeltaY;"
                               + "  vec2  psi;"
                               + "  vec2  psiPrevious;"
                               + "  float psiMax2;"
                               + "  float t;"
                               + "  float v;"
                               + "  float vPrevious;"
                               + ""
                               + "  halfDeltaY       = 0.5/float(yResolution);"
                               + "  psiMax2          = psiMax*psiMax;"
                               + ""
                               + "  psi              = texture2D(waveFunction, vTextureCoord).rg;"
                               + "  psiPrevious      = texture2D(waveFunction, vTextureCoord-vec2(1.0/float(xResolution), 0)).rg;"
                               + "  t                = vTextureCoord.t;"
                               + "  v                = texture2D(potential, vTextureCoord).r;"
                               // v from the previous texture element
                               + "  vPrevious        = texture2D(potential, vTextureCoord-vec2(1.0/float(xResolution), 0)).r;"
                               + ""
                               + "  background = pixelColor(psiColor, psiMax2, t, halfDeltaY, width,"
                               + "                          psi.r*psi.r + psi.g*psi.g,"
                               + "                          psiPrevious.r*psiPrevious.r + psiPrevious.g*psiPrevious.g);"
                               + ""
                               + "  color = pixelColor(reColor, psiMax2, t, halfDeltaY, width, psi.r, psiPrevious.r);"
                               + "  background = mix(background, color, color.a);"
                               + ""
                               + "  color = pixelColor(imColor, psiMax2, t, halfDeltaY, width, psi.g, psiPrevious.g);"
                               + "  background = mix(background, color, color.a);"
                               + ""
                               + "  color = pixelColor(vColor, vMax, t, halfDeltaY, width, v, vPrevious);"
                               + "  background = mix(background, color, color.a);"
                               + ""
                               + "  color = pixelColor(vColor, vMax, t, halfDeltaY, width, E, E);"
                               + "  background = mix(background, color, color.a);"
                               + ""
                               + "  gl_FragColor = background;"
                               + "}";

    const program            = gpgpUtility.createProgram(null, fragmentShaderSource);
    const positionHandle     = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    const textureCoordHandle = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);

    const EHandle            = gl.getUniformLocation(program, "E");
    const potentialHandle    = gl.getUniformLocation(program, "potential");
    const waveFunctionHandle = gl.getUniformLocation(program, "waveFunction");
    const psiMaxHandle       = gl.getUniformLocation(program, "psiMax");
    const psiColorHandle     = gl.getUniformLocation(program, "psiColor");
    const imColorHandle      = gl.getUniformLocation(program, "imColor");
    const reColorHandle      = gl.getUniformLocation(program, "reColor");
    const vMaxHandle         = gl.getUniformLocation(program, "vMax");
    const vColorHandle       = gl.getUniformLocation(program, "vColor");
    const xResolutionHandle  = gl.getUniformLocation(program, "xResolution");
    const yResolutionHandle  = gl.getUniformLocation(program, "yResolution");
    const widthHandle        = gl.getUniformLocation(program, "width");

    return {
      program:            program,
      positionHandle:     positionHandle,
      textureCoordHandle: textureCoordHandle,
      EHandle:            EHandle,
      potentialHandle:    potentialHandle,
      waveFunctionHandle: waveFunctionHandle,
      psiMaxHandle:       psiMaxHandle,
      psiColorHandle:     psiColorHandle,
      imColorHandle:      imColorHandle,
      reColorHandle:      reColorHandle,
      vMaxHandle:         vMaxHandle,
      vColorHandle:       vColorHandle,
      xResolutionHandle:  xResolutionHandle,
      yResolutionHandle:  yResolutionHandle,
      widthHandle:        widthHandle
    };
  };

  /**
   * Setup for rendering to the screen. Create a canvas, get a rendering context,
   * set uniforms.
   */
  this.setup = function(gpgpUtility, xResolution, yResolution, psiColor, reColor, imColor, E, potential, vColor, psiMax, vMax, width)
  {
    var gl;
    gl = gpgpUtility.getGLContext();

    gl.useProgram(program.program);

    gl.uniform1f(program.EHandle, E);
    gl.uniform4fv(program.imColorHandle, new Float32Array(imColor));
    gl.uniform4fv(program.psiColorHandle, new Float32Array(psiColor));
    gl.uniform1f(program.psiMaxHandle, psiMax);
    gl.uniform4fv(program.reColorHandle, new Float32Array(reColor));
    gl.uniform4fv(program.vColorHandle, new Float32Array(vColor));
    gl.uniform1f(program.vMaxHandle, vMax);
    gl.uniform1i(program.xResolutionHandle, xResolution);
    gl.uniform1i(program.yResolutionHandle, yResolution);
    gl.uniform1f(program.widthHandle, width);
  }

  this.setPsiMax = function(psiMax_)
  {
    var gl;
    gl = gpgpUtility.getGLContext();

    psiMax       = psiMax_;

    gl.useProgram(program.program);
    gl.uniform1f(program.psiMaxHandle, psiMax);
  }

  /**
   * Map the waveFunction texture onto a curve
   *
   * @param {WebGLTexture} waveFunction A xResolution by 1 texture containing the real
   *                                    and imaginary parts of the wave function.
   * @param {Number} t The time at which we are rendering the function
   */
  this.show = function(waveFunction, t)
  {
    var blending;
    var gl;

    time = t;
    gl = gpgpUtility.getRenderingContext();

    gl.useProgram(program.program);

    blending = gl.isEnabled(gl.BLEND);
    if (!blending)
    {
      gl.enable(gl.BLEND);
    }

    // This time we will render to the screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(program.positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(program.textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, potential);
    gl.uniform1i(program.potentialHandle, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, waveFunction);
    gl.uniform1i(program.waveFunctionHandle, 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    if (!blending)
    {
      gl.disable(gl.BLEND);
    }
  }

  this.getTime = function() {
    return time;
  }

  E            = E_;
  gpgpUtility  = gpgpUtility_;
  imColor      = imColor_;
  psiColor     = psiColor_;
  parent       = parent_;
  potential    = potential_;
  psiMax       = psiMax_;
  reColor      = reColor_;
  vColor       = vColor_;
  vMax         = vMax_;
  xResolution  = xResolution_;
  yResolution  = yResolution_;
  width        = width_;

  program      = this.createProgram(gpgpUtility.getGLContext());
  this.setup(gpgpUtility, xResolution, yResolution, psiColor, reColor, imColor, E, potential, vColor, psiMax, vMax, width);

}

SchrodingerRenderer.DEFAULT_LINE_WIDTH = 2.0;