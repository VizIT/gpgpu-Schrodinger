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
  var pixels;
  var positionHandle;
  var potential;
  var potentialHandle;
  var program;
  // Whether this is a zero or one step - determined which texture is the source, which is rendered to.
  var step;
  var textureCoordHandle;
  var textures;
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
                         // The number of points along the x axis.
                         + "uniform int xResolution;"
                         + ""
                         // waveFunction.r is the real part waveFunction.g is the imaginary part.
                         + "uniform sampler2D waveFunction;"
                         // Discrete representation of the potential function.
                         + "uniform sampler2D potential;"
                         + ""
                         // Vector to mix the real and imaginary parts in the wave function update.
                         + "const vec2 mixing = vec2(-1.0, +1.0);"
                         + ""
                         + "varying vec2 vTextureCoord;"
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
                         + "  gl_FragColor.rg =  value.rg"
                         + "                     + ((texture2D(waveFunction, vTextureCoord+ds).gr"
                         + "                                  - 2.0*value.gr"
                         + "                                  + texture2D(waveFunction, vTextureCoord-ds).gr)/(2.0*(dx*dx))"
                         + "                        - texture2D(potential, vTextureCoord).r*value.gr)*mixing*dt;"
                         + "}";

    program            = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle     = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    dtHandle           = gl.getUniformLocation(program, "dt");
    potentialHandle    = gl.getUniformLocation(program, "potential");
    waveFunctionHandle = gl.getUniformLocation(program, "waveFunction");
    xResolutionHandle  = gl.getUniformLocation(program, "xResolution");
    lengthHandle       = gl.getUniformLocation(program, "length");

    return program;
  };

  /**
   * Setup the initial values for textures. 
   */
  this.setInitialTextures = function(texture0, texture1)
  {
    textures[0] = texture0;
    fbos[0]     = gpgpUtility.attachFrameBuffer(texture0);
    textures[1] = texture1;
    fbos[1]     = gpgpUtility.attachFrameBuffer(texture1);
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

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[step]);

    // Step switches back and forth between 0 and 1,
    // ping ponging the source & destination textures.
    step = (step+1)%2;

    gpgpUtility.getStandardVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.uniform1f(dtHandle,          dt);
    gl.uniform1i(xResolutionHandle, xResolution);
    gl.uniform1f(lengthHandle,      length);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[step]);
    gl.uniform1i(waveFunctionHandle, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, potential);
    gl.uniform1i(potentialHandle, 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  /**
   * Retrieve the most recently rendered to texture.
   *
   * @returns {WebGLTexture} The texture used as the rendering target in the most recent
   *                         timestep.
   */
  this.getRenderedTexture = function()
  {
      return textures[(step+1)%2];
  }

  /**
   * Retrieve the frambuffer that wraps the texture to be used as the source in the
   * next timestep. Render to this FBO in the initialization step.
   *
   * @returns {WebGLFramebuffer} The framebuffer wrapping the source texture for the next timestep.
   */
  this.getSourceFramebuffer = function()
  {
    return fbos[(step+1)%2];
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
