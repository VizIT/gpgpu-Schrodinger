/**
 * Copyright 2017 Vizit Solutions
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
function MurBoundary(gpgpUtility_, xResolution_, length_, dt_, vp_)
{
  "use strict";

  var dt;
  var dtHandle;
  /** WebGLRenderingContext */
  var gl;
  var gpgpUtility;
  var length;
  var lengthHandle;
  /** The wave function at t - delta t */
  var oldWaveFunctionHandle;
  var phaseVelocityHandle;
  var phaseVelocity;
  var pixels;
  var positionHandle;
  var potential;
  var potentialHandle;
  var program;
  // Whether this is a zero or one step - determined which texture is the source, which is rendered to.
  var step;
  var textureCoordHandle;
  var textures;
  var vertices;
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
                         // An estimate of the phase velocity at the boundary
                         + "uniform float vp;"
                         // At time t - delta t waveFunction.r is the real part waveFunction.g is the imaginary part.
                         + "uniform sampler2D oldWaveFunction;"
                         // The number of points along the x-axis.
                         + "uniform int xResolution;"
                         + ""
                         // At time t waveFunction.r is the real part waveFunction.g is the imaginary part.
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
                         + "  vec2  innerTextureCoord;"
                         + "  vec2  innerValue;"
                         + "  float offset;"
                         + "  vec4  value;"
                         + ""
                         + "  dx                = length/float(xResolution);"
                         + "  ds                = vec2(1.2/float(xResolution), 0.0);"
                         // On the left edge, compute the value of psi(x+dx), on the right edge, compute psi(x-dx)
                         + "  offset            = 1.0 - 2.0*step(0.5, vTextureCoord.s);"
                         + "  innerTextureCoord = vTextureCoord + offset*ds;"
                         + "  value             = texture2D(waveFunction, innerTextureCoord);"
                         + ""
                         // One step in from the edge, at t+&Delta;t
                         + "  innerValue        =  texture2D(oldWaveFunction, innerTextureCoord).rg"
                         + "                       + ((texture2D(waveFunction, innerTextureCoord+ds).gr"
                         + "                             - 2.0*value.gr"
                         + "                             + texture2D(waveFunction, innerTextureCoord-ds).gr)/(dx*dx)"
                         + "                          - 2.0*texture2D(potential, innerTextureCoord).r*value.gr)*mixing*dt;"
                         + ""
                         + "  gl_FragColor.rg = value.rg"
                         + "                    + ((vp*dt-dx)/(vp*dt+dx))*(innerValue"
                         + "                                               - texture2D(waveFunction, vTextureCoord).rg);"
                         + "}";

    program               = gpgpUtility.createProgram(null, fragmentShaderSource);
    positionHandle        = gpgpUtility.getAttribLocation(program,  "position");
    gl.enableVertexAttribArray(positionHandle);
    textureCoordHandle    = gpgpUtility.getAttribLocation(program,  "textureCoord");
    gl.enableVertexAttribArray(textureCoordHandle);
    dtHandle              = gl.getUniformLocation(program, "dt");
    oldWaveFunctionHandle = gl.getUniformLocation(program, "oldWaveFunction");
    phaseVelocityHandle   = gl.getUniformLocation(program, "vp");
    potentialHandle       = gl.getUniformLocation(program, "potential");
    waveFunctionHandle    = gl.getUniformLocation(program, "waveFunction");
    xResolutionHandle     = gl.getUniformLocation(program, "xResolution");
    lengthHandle          = gl.getUniformLocation(program, "length");

    return program;
  };

  /**
   * Read back the i, j pixel and display it to the console.
   * 
   * @param i       {integer} the i index of the texel to be tested.
   * @param j       {integer} the j index of the texel to be tested.
   */
  this.test = function(i, j)
  {
    var buffer;

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

    console.log(buffer);
  }

  /**
   * Return 1D boundary with texture coordinates for Mur boundary conditions.
   * One point at each end of the 1xn strip we are using in this simulation.
   * (-1, 0.5) as the left boundary, (1,0.5) as the right edge.
   *
   * @returns {Float32Array} A set of points and textures containing one point at
   *                         each end of the simulation.
   */
  this.getGeometry = function ()
  {
    // Sets of x,y,z(=0),s,t coordinates.
    return new Float32Array([-1.0+(0.5/xResolution), 0.0, 0.0, 0.0, 0.0,  // left edge
                              1.0, 0.0, 0.0, 1.0, 0.0]);// right edge
  };

  /**
   * Return vertices for the boundary. If they don't yet exist,
   * they are created and loaded with the appropriate geometry.
   * If they already exist, they are bound and returned.
   *
   * @returns {WebGLBuffer} A bound buffer containing the standard geometry.
   */
  this.getVertices = function ()
  {
    if (!vertices)
    {
      vertices = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
      gl.bufferData(gl.ARRAY_BUFFER, this.getGeometry(), gl.STATIC_DRAW);
    }
    else
    {
      gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
    }
    return vertices;
  };



  /**
   * Renders points at the boundary of the simulation to compute the boundary
   * values using the one way wave equations
   */
  this.render = function()
  {
    var gl;

    gl = gpgpUtility.getComputeContext();

    gl.useProgram(program);

    this.getVertices();

    gl.vertexAttribPointer(positionHandle,     3, gl.FLOAT, gl.FALSE, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);

    gl.uniform1f(dtHandle,            dt);
    gl.uniform1i(xResolutionHandle,   xResolution);
    gl.uniform1f(lengthHandle,        length);
    gl.uniform1f(phaseVelocityHandle, phaseVelocity);

    // Texture samplers, carried over from the main compute step
    gl.uniform1i(oldWaveFunctionHandle, 1);

    gl.uniform1i(waveFunctionHandle, 2);

    //this.test(1599, 0);

    gl.drawArrays(gl.POINTS, 0, 2);

    //this.test(0, 0);
  };

  /**
   * Invoke to clean up resources specific to this program. We leave the texture
   * and frame buffer intact as they are used in follow-on calculations.
   */
  this.done = function ()
  {
    gl.deleteProgram(program);
  };

  dt            = dt_;
  gpgpUtility   = gpgpUtility_;
  gl            = gpgpUtility.getGLContext();
  program       = this.createProgram(gl);
  length        = length_;
  phaseVelocity = vp_;
  xResolution   = xResolution_;
}
