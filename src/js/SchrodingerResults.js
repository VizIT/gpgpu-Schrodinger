/**
 * Copyright 2016-2025 Vizit Solutions
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
 * Given a 1D array containing wave function values, draw those values onto
 * a canvas with the array index along the x-axis and the function
 * values along the y-axis.
 */
class SchrodingerResults
{
  #device;
  #psiColor;
  #psiMax;
  #yResolution;
  #parametersBindGroup;
  #parametersBindGroupLayout;
  #plotParametersBuffer;
  #plotParametersBindGroupLayout;
  #plotParametersBindGroup;
  #vertexBuffer;
  #vertexBuffersDescriptor;
  #rendererShaderModule;
  #canvasID;
  #presentationFormat;
  #webGPUContext;

  /**
   * Build a Schrödinger wave function visualization with the given parameters.
   *
   * @param {GPUDevice} device       The device in use for the simulation, to allow buffer reuse.
   * @param {String} canvasID        The HTML ID for the canvas we render to.
   * @param {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters,
   *                                           carried over from the Schrödinger solver.
   * @param {GPUBindGroupLayout} parametersBindGroupLayout The bind group layout for the simulation parameters
   *                                                       from the Schrödinger simulation.
   * @param {Array<Number>} psiColor The r, g, b, a color for the wave function, 0, 0, 0 0, for no plot.
   * @param {Number} psiMax          The max psi value on the plot, the y-axis scale for the wave function plots.
   * @param {Number} yResolution     The number of pixels in the y direction.
   */
  constructor(device, canvasID, parametersBindGroup, parametersBindGroupLayout,
              psiColor, psiMax, yResolution)
  {
    this.#device = device;
    this.#canvasID = canvasID;
    this.#parametersBindGroup = parametersBindGroup;
    this.#parametersBindGroupLayout = parametersBindGroupLayout;
    this.#psiColor = psiColor;
    this.#psiMax = psiMax;
    this.#yResolution = yResolution;

    this.init();
  }

  init()
  {
    const rendererShader = `
        struct WaveFunctionParameters
        {
            dt: f32,              // The time step size
            xResolution: u32,     // The number of points along the x-axis, the number of elements in the array.
            length: f32,          // The full length for our simulation
        }
        
        struct PlotParameters
        {
            // Psi*Psi color
            psiColor: vec4f,
            // Y scale for the psi plot
            psiMax: f32,
            // Number of points along the y axis.
            yResolution: u32
        }
        
        // group 0 and 1, things that never change within a simulation.
        // The parameters for the simulation
        @group(0) @binding(0) var<storage, read> waveFunctionParameters: WaveFunctionParameters;
        // Plotting parameters, line colors, width, etc.
        @group(1) @binding(0) var<uniform> plotParameters : PlotParameters;
        
        //group 1, Wave function at t, changes on each iteration
        @group(2) @binding(0) var<storage, read> waveFunction : array<vec2f>;
        
        @fragment
        fn fs_main(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32>
        {
            let psiMax2          = plotParameters.psiMax*plotParameters.psiMax;
            // Remember, frag position ranges from 0.5 to xResolition-0.5,
            // see https://www.w3.org/TR/webgpu/#rasterization
            let index            = i32(fragPos.x);
            let psi              = waveFunction[index];
            let yResolution1     = 1.0/f32(plotParameters.yResolution);
            let adjustedPixel    = (f32(plotParameters.yResolution)-fragPos.y) * yResolution1;
            let absPsi2          = psi.r*psi.r + psi.g*psi.g;
            
            return plotParameters.psiColor*(smoothstep(psiMax2*(adjustedPixel-1.5*yResolution1),
                                                       psiMax2*(adjustedPixel-yResolution1),
                                                       absPsi2)
                                            - smoothstep(psiMax2*(adjustedPixel+yResolution1),
                                                         psiMax2*(adjustedPixel+1.5*yResolution1),
                                                         absPsi2));
        }
        
         @vertex
         fn vs_main(@location(0) inPos: vec3<f32>) -> @builtin(position) vec4f
         {
            return vec4(inPos, 1.0);
         }
    `;

    this.#rendererShaderModule = this.#device.createShaderModule({
      label: 'Schrodinger renderer shader',
      code: rendererShader
    });

    // A pair of triangles that cover the canvas in normalized device coordinates
    const vertexData = new Float32Array([
      -1.0,  1.0, 0.0, // upper left
      -1.0, -1.0, 0.0, // lower left
       1.0,  1.0, 0.0, // upper right
       1.0, -1.0, 0.0  // lower right
    ]);

    this.#vertexBuffer = this.#device.createBuffer({
      label: 'Position',
      mappedAtCreation: true,
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX
    });

    const vertexArrayBuffer = this.#vertexBuffer.getMappedRange();
    new Float32Array(vertexArrayBuffer).set(vertexData);
    this.#vertexBuffer.unmap();

    this.#vertexBuffersDescriptor = [{
      arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
      stepMode: 'vertex',
      attributes: [{
        shaderLocation: 0, // @location in shader
        offset: 0,
        format: 'float32x3'
      }]
    }];

    this.#plotParametersBindGroupLayout = this.#device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform"
        }
      }]
    });

    this.#plotParametersBuffer = this.#device.createBuffer({
      label: 'Plot Parameters',
      mappedAtCreation: true,
      size: 4*Float32Array.BYTES_PER_ELEMENT // psiColor
            + Float32Array.BYTES_PER_ELEMENT // psiMax
            + Uint32Array.BYTES_PER_ELEMENT  // yResolution
            + 8,                             // Required padding
      usage: GPUBufferUsage.UNIFORM
    });

    // Get the raw array buffer for the mapped GPU buffer
    const plotParametersArrayBuffer = this.#plotParametersBuffer.getMappedRange();

    let bytesSoFar = 0;
    new Float32Array(plotParametersArrayBuffer, bytesSoFar, 4).set(this.#psiColor);
    bytesSoFar += 4*Float32Array.BYTES_PER_ELEMENT;
    new Float32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#psiMax]);
    bytesSoFar += Float32Array.BYTES_PER_ELEMENT;
    new Uint32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#yResolution]);


    this.#plotParametersBuffer.unmap();

    this.#plotParametersBindGroup = this.#device.createBindGroup({
      layout: this.#plotParametersBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.#plotParametersBuffer
          }
        }
      ]});

    // Get a WebGPU context from the canvas and configure it
    const canvas = document.getElementById(this.#canvasID);
    this.#webGPUContext = canvas.getContext('webgpu');
    // This will be either rgba8unorm or bgra8unorm
    this.#presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.#webGPUContext.configure({
      device: this.#device,
      format: this.#presentationFormat,
      alphaMode: 'premultiplied'
    });

    return this;
  }

  /**
   * Render a wave function buffer from the schrodinger simulation.
   *
   * @param {GPUBuffer } waveFunctionBuffer
   */
  render(waveFunctionBuffer)
  {
    const waveFunctionBindGroupLayout = this.#device.createBindGroupLayout({
      label: "Wave function layout",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "read-only-storage"
        }
      }]
    });

    const waveFunctionBindGroup = this.#device.createBindGroup({
      layout: waveFunctionBindGroupLayout,
      entries: [{
        binding: 0,
        resource: {
          buffer: waveFunctionBuffer
        }
      }]
    });

    const pipelineLayout = this.#device.createPipelineLayout({
      bindGroupLayouts: [
        this.#parametersBindGroupLayout,     // Simulation parameters
        this.#plotParametersBindGroupLayout, // Plot parameters
        waveFunctionBindGroupLayout          // The wave function values
      ]
    });

    const pipeline = this.#device.createRenderPipeline({
      label: 'Render triangles to cover the rectangular canvas.',
      layout: pipelineLayout,
      primitive: {
        topology: "triangle-strip",
      },
      vertex: {
        module: this.#rendererShaderModule,
        entryPoint: 'vs_main',
        buffers: this.#vertexBuffersDescriptor
      },
      fragment: {
        module: this.#rendererShaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.#presentationFormat
        }]
      }
    });

    const commandEncoder = this.#device.createCommandEncoder();

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: this.#webGPUContext.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: [0.0, 0.0, 0.0, 0.0],
        storeOp: 'store',
      }]
    });
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, this.#parametersBindGroup);
    passEncoder.setBindGroup(1, this.#plotParametersBindGroup);
    passEncoder.setBindGroup(2, waveFunctionBindGroup);
    passEncoder.setVertexBuffer(0, this.#vertexBuffer);
    passEncoder.draw(4);
    passEncoder.end();

    const commandBuffer = commandEncoder.finish();
    this.#device.queue.submit([commandBuffer]);
  }
}

export {SchrodingerResults}