/**
 * Copyright 2015-2016 Vizit Solutions
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


function Schrodinger(gpgpUtility_, xResolution_, length_, dt_)
{
  "use strict";

  var dt;
  var dtHandle;
  var fbos;
  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var length;
  var lengthHandle;
  /** Real and imaginary components of the wave function at t - delta t */
  var oldWaveFunctionReHandle, oldWaveFunctionImHandle;
  var pixels;
  var positionHandle;
  var potential;
  var potentialHandle;
  var program;
  // Whether this is a zero or one step - determined which texture is the source, which is rendered to.
  var step;
  var textureCoordHandle;
  var textures;
  /** The wave function at t */
  var waveFunctionHandle;
  var xResolution;
  var xResolutionHandle;

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
                         // The delta-t for each timestep.
                         + "uniform float dt;"
                         // The physical length of the grid in nm.
                         + "uniform float length;"
                         // Real part of the wavefunction at t - delta t
                         + "uniform sampler2D oldWaveFunctionRe;"
                         // Imaginary part of the wavefunction at t - delta t
                         + "uniform sampler2D oldWaveFunctionIm;"
                         // The number of points along the x axis.
                         + "uniform int xResolution;"
                         + ""
                         // Real part of the wave function at time t.
                         + "uniform sampler2D waveFunctionRe;"
                         // Imaginary part of the wave function at time t.
                         + "uniform sampler2D waveFunctionIm;"
                         // Discrete representation of the potential function.
                         + "uniform sampler2D potential;"
                         + ""
                         // Vector to mix the real and imaginary parts in the wave function update.
                         + "const vec2 mixing = vec2(-1.0, +1.0);"
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
                         + "void main()"
                         + "{"
                         + "  float dx;"
                         + "  vec2  ds;"
                         + "  vec4  value;"
                         + ""
                         + "  dx    = length/float(xResolution);"
                         + "  ds    = vec2(1.2/float(xResolution), 0.0);"
                         + "  value = texture2D(waveFunction, vTextureCoord);"
                         + ""
                         + "  gl_FragColor.rg =  texture2D(oldWaveFunction, vTextureCoord).rg"
                         + "                     + ((texture2D(waveFunction, vTextureCoord+ds).gr"
                         + "                                  - 2.0*value.gr"
                         + "                                  + texture2D(waveFunction, vTextureCoord-ds).gr)/(dx*dx)"
                         + "                        - 2.0*texture2D(potential, vTextureCoord).r*value.gr)*mixing*dt;"
                         + "}";

    program               = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle        = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle    = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    dtHandle              = gl.getUniformLocation(program, "dt");
    oldWaveFunctionHandle = gl.getUniformLocation(program, "oldWaveFunction");
    potentialHandle       = gl.getUniformLocation(program, "potential");
    waveFunctionHandle    = gl.getUniformLocation(program, "waveFunction");
    xResolutionHandle     = gl.getUniformLocation(program, "xResolution");
    lengthHandle          = gl.getUniformLocation(program, "length");

    return program;
  };

  /**
   * Set up the initial values for textures. Two for values of the wave function,
   * and a third as a render target.
   */
  this.setInitialTextures = function(texture0, texture1, texture2)
  {
    textures[0] = texture0;
    fbos[0]     = gpgpUtility.attachFrameBuffer(texture0);
    textures[1] = texture1;
    fbos[1]     = gpgpUtility.attachFrameBuffer(texture1);
    textures[2] = texture2;
    fbos[2]     = gpgpUtility.attachFrameBuffer(texture2);
  }

  /**
   * Set the potential as a texture
   */
  this.setPotential = function(texture)
  {
    potential = texture;
  }

  /**
   * Runs the program to do the actual work. On exit the framebuffer &amp;
   * texture are populated with the next timestep of the wave function.
   * You can use gl.readPixels to retrieve texture values.
   */
  this.timestep = function()
  {
    var gl;

    gl = gpgpUtility.getComputeContext();

    gl.useProgram(program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[(step+2)%3]);

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.uniform1f(dtHandle,          dt);
    gl.uniform1i(xResolutionHandle, xResolution);
    gl.uniform1f(lengthHandle,      length);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[step]);
    gl.uniform1i(oldWaveFunctionHandle, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, textures[(step+1)%3]);
    gl.uniform1i(waveFunctionHandle, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, potential);
    gl.uniform1i(potentialHandle, 2);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Step cycles though 0, 1, 2
    // Controls cycling over old, current and render target uses of textures
    step = (step+1)%3;
  };

  /**
   * Retrieve the most recently rendered to texture.
   *
   * @returns {WebGLTexture} The texture used as the rendering target in the most recent
   *                         timestep.
   */
  this.getRenderedTexture = function()
  {
      return textures[(step+1)%3];
  }

  /**
   * Retrieve the two frambuffers that wrap the textures for the old and current wavefunctions in the
   * next timestep. Render to these FBOs in the initialization step.
   *
   * @returns {WebGLFramebuffer[]} The framebuffers wrapping the source textures for the next timestep.
   */
  this.getSourceFramebuffers = function()
  {
    var value = [];
    value[0] = fbos[step];
    value[1] = fbos[(step+1)%3];
    return value;
  }

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in follow-on calculations.
   */
  this.done = function ()
  {
    gl.deleteProgram(program);
  };

  dt          = dt_;
  gpgpUtility = gpgpUtility_;
  gl          = gpgpUtility.getGLContext();
  program     = this.createProgram(gl);
  fbos        = new Array(2);
  length      = length_;
  textures    = new Array(2);
  step        = 0;
  xResolution = xResolution_;
}