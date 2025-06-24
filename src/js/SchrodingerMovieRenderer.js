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
 * Given a 1D textures containing wave function and potential values, draw those
 * values onto a canvas with the texture coordinates along the x-axis and the function
 * values along the y-axis. Attach the canvas to the parent.
 */
function SchrodingerRenderer(gpgpUtility_, parent_, psiColor_, reColor_, imColor_, vColor_, xResolution_, yResolution_, potential_, psiMax_, vMax_)
{
  "use strict";

  var gpgpUtility;
  // Color for the imaginary part of psi, 0,0,0,0 => nothing drawn.
  var imColor;
  var imColorHandle;
  var parent;
  var positionHandle;
  var potential;
  var potentialHandle;
  var program;
  // psi*psi color
  var psiColor;
  var psiColorHandle;
  var psiMax;
  var psiMaxHandle;
  // real psi component color, 0,0,0,0 => nothing drawn.
  var reColor;
  var reColorHandle;
  var waveFunction;
  var waveFunctionHandle;
  var textureCoordHandle;
  // Color for the potential
  var vColor;
  var vColorHandle;
  // Max abs(v) for the potential
  var vMax;
  var vMaxHandle;
  var xResolution;
  var xResolutionHandle;
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
                         + ""
                         + "varying vec2 vTextureCoord;"
                         + ""
                         + "vec4 blend(vec4 background, vec4 foreground)"
                         + "{"
                         + "  float alpha;"
                         + ""
                         + "  alpha = foreground.a;"
                         + "  background = foreground*alpha + background*(1.0-alpha);"
                         + ""
                         + "  return background;"
                         + "}"
                         + ""
                         + "vec4 pixelColor(vec4 color, float scale, float t, float halfDeltaY, float value, float previousValue)"
                         + "{"
                         + "  vec4  pixel;"
                         + "  float scale2;"
                         + ""
                         + "  scale2 = 2.0*scale;"
                         + ""
                         + "  pixel = color*(smoothstep(scale2*(t-2.0*halfDeltaY)-scale, scale2*(t-halfDeltaY)-scale, value)"
                         + "                   - smoothstep(scale2*(t+halfDeltaY)-scale, scale2*(t+2.0*halfDeltaY)-scale, value))"
                         // Fill in discontinuities in the function
                         + "        + color*(step(previousValue+scale2*halfDeltaY, value)"
                         + "                 *(step(previousValue, scale2*t-scale) - step(value, scale2*t-scale)))"
                         + "        + color*(step(value, previousValue+scale2*halfDeltaY)"
                         + "                 *(step(value, scale2*t-scale) - step(previousValue, scale2*t-scale)));"
                         + ""
                         + "  return pixel;"
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
                         + "  background = pixelColor(psiColor, psiMax2, t, halfDeltaY, psi.r*psi.r + psi.g*psi.g, psiPrevious.r*psiPrevious.r + psiPrevious.g*psiPrevious.g);"
                         + ""
                         + "  color = pixelColor(reColor, psiMax2, t, halfDeltaY, psi.r, psiPrevious.r);"
                         + "  background = blend(background, color);"
                         + ""
                         + "  color = pixelColor(imColor, psiMax2, t, halfDeltaY, psi.g, psiPrevious.g);"
                         + "  background = blend(background, color);"
                         + ""
                         + "  color = pixelColor(vColor, vMax, t, halfDeltaY, v, vPrevious);"
                         + "  background = blend(background, color);"
                         + ""
                         + "  gl_FragColor = background;"
                         + "}";

    program            = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle     = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);

    potentialHandle    = gl.getUniformLocation(program, "potential");
    waveFunctionHandle = gl.getUniformLocation(program, "waveFunction");
    psiMaxHandle       = gl.getUniformLocation(program, "psiMax");
    psiColorHandle     = gl.getUniformLocation(program, "psiColor");
    imColorHandle      = gl.getUniformLocation(program, "imColor");
    reColorHandle      = gl.getUniformLocation(program, "reColor");
    vMaxHandle         = gl.getUniformLocation(program, "vMax");
    vColorHandle       = gl.getUniformLocation(program, "vColor");
    xResolutionHandle  = gl.getUniformLocation(program, "xResolution");
    yResolutionHandle  = gl.getUniformLocation(program, "yResolution");

    return program;
  };

  /**
   * Setup for rendering to the screen. Create a canvas, get a rendering context,
   * set uniforms.
   */
  this.setup = function(gpgpUtility, xResolution, yResolution, psiColor, reColor, imColor, potential, vColor, psiMax, vMax)
  {
    var gl;
    gl = gpgpUtility.getGLContext();

    gl.useProgram(program);

    gl.uniform4fv(imColorHandle, new Float32Array(imColor));
    gl.uniform4fv(psiColorHandle, new Float32Array(psiColor));
    gl.uniform1f(psiMaxHandle, psiMax);
    gl.uniform4fv(reColorHandle, new Float32Array(reColor));
    gl.uniform4fv(vColorHandle, new Float32Array(vColor));
    gl.uniform1f(vMaxHandle, vMax);
    gl.uniform1i(xResolutionHandle, xResolution);
    gl.uniform1i(yResolutionHandle, yResolution);

  }

  this.setPsiMax = function(psiMax_)
  {
    var gl;
    gl = gpgpUtility.getGLContext();

    psiMax       = psiMax_;

    gl.useProgram(program);
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
    var blending;
    var gl;

    gl = gpgpUtility.getRenderingContext();

    gl.useProgram(program);

    blending = gl.isEnabled(gl.BLEND);
    if (!blending)
    {
      gl.enable(gl.BLEND);
    }

    // This time we will render to the screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, potential);
    gl.uniform1i(potentialHandle, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, waveFunction);
    gl.uniform1i(waveFunctionHandle, 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    if (!blending)
    {
      gl.disable(gl.BLEND);
    }
  }

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

  program      = this.createProgram(gpgpUtility.getGLContext());
  this.setup(gpgpUtility, xResolution, yResolution, psiColor, reColor, imColor, potential, vColor, psiMax, vMax);

}
