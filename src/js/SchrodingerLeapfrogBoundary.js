/**
 * Copyright 2025 Vizit Solutions
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
import {WebGPUCompute} from "./WebGPUCompute.js";

/**
 * @typedef {Number} Integer
 */

const WORKGROUP_SIZE = 64;

/**
 * An FDTD time evolver for the Schrödinger wave function. This version implements a staggered time approach,
 * and makes a copy of the wave function before computing the time step.
 *
 * @property {GPUDevice} #device The device as retrieved from the adaptor.
 * @property {Number} #dt The time step between the wave function and its updated version.
 * @property {Number} #length The physical length of this simulation.
 * @property {Integer} #xResolution The number of spatial steps in the wave function representation.
 * @property {Boolean} #running Whether the simulation is allowed to run. Setting this to false halts the simulation.
 * @property {Array<Number>} #potential An array of potential values, array elements correspond to physical locations
 *                                      just as the wave function arrays.
 * @property {GPUBuffer} #parametersBuffer The FDTD parameters buffer.
 * @property {GPUBindGroup} #parametersBindGroup The bind group for the parameters buffer.
 * @property {GPUBindGroupLayout} #parametersBindGroupLayout The bind group layout for the simulation parameters.
 * @property {GPUBuffer} #waveFunctionBuffer The wave function buffer.
 * @property {GPUBindGroup[]} #waveFunctionBindGroup THe bind group for the wave function buffer.
 * @property {GPUComputePipeline} #computePipeline The compute pipeline controlling some aspects of the shader execution.
 * @property {Boolean} #bcEnabled Whether to invoke the boundary value computations.
 * @property {LeapfrogMurBoundary} #boundary A MurBoundary instance, or similar boundary value class capable of
 *                                           working with the staggered time memory layout.
 * @property {Boolean} #debug A flag indicating whether this is a debugging instance.
 */
class Schrodinger
{
  #device;
  #dt;
  #length;
  #xResolution;
  #potential;
  #running;
  #parametersBuffer;
  #parametersBindGroup;
  #parametersBindGroupLayout;
  #waveFunctionBuffer;
  #oldWaveFunctionBuffer;
  #waveFunctionBindGroup;
  #waveFunctionBindGroupLayout;
  #imaginaryPartTimeStep;
  #realPartTimeStep
  // true => invoke boundary conditions - make sure boundary is set before setting this.
  #bcEnabled = false;
  // Renders the boundary at t+dt after the main t+dt rendering
  #boundary;
  #debug;

  /**
   * Build a Schrödinger equation integrator with the given parameters.
   *
   * @param {Number}        dt          The Δt between iterations of the wave function in natural units of time.
   * @param {Integer}       xResolution The size of the wave function arrays, the number of spatial steps
   *                                    on our 1D grid.
   * @param {Number}        length      The characteristic length for the problem in terms of natural units.
   * @param {Array<Number>} potential   An array of potential values, array elements correspond to physical locations
   *                                    just as the wave function arrays.
   * @param {Boolean}       debug       The debug option for this execution of the FDTD solver. Enabling this
   *                                    makes the wave function buffers copyable, potentially having a
   *                                    performance impact. Defaults to false.
   */
  constructor(dt, xResolution, length, potential, debug=false)
  {
    this.#dt = dt;
    this.#xResolution = xResolution;
    this.#length = length;
    this.#potential = potential;
    this.#debug = debug;
  }

  /**
   * Get the time step between wave function instances for FDTD time evolution.
   *
   * @returns {Number} The FDTD time step.
   */
  getTimeStep()
  {
    return this.#dt;
  }

  /**
   * Set the FDTD time step.
   *
   * @param {Number} dt The new time step between the wave function and its updated version.
   * @returns {Schrodinger}
   */
  setTimeStep(dt)
  {
    this.#dt = dt;
    return this;
  }

  /**
   * Get the number of array elements, or the number of spatial steps, in the wave function representation.
   *
   * @returns {Number} The number of spatial steps in the wave function representation.
   */
  getXResolution()
  {
    return this.#xResolution;
  }

  /**
   * Set the number of array elements, or the number of spatial steps, in the wave function representation.
   *
   * @param {Integer} xResolution The number of spatial steps in the wave function representation.
   * @returns {Schrodinger} This object with the new resolution set.
   */
  setXResolution(xResolution)
  {
    this.#xResolution = xResolution;
    return this;
  }

  /**
   * Get the physical length of this simulation.
   *
   * @returns {Number} The physical length of this simulation.
   */
  getLength()
  {
    return this.#length
  }

  /**
   * Get the physical length of this simulation.
   *
   * @param {Number} length The physical length of this simulation.
   * @returns {Schrodinger}
   */
  setLength(length)
  {
    this.#length = length;
    return this;
  }

  /**
   * Get the potential on our grid.
   *
   * @returns {Array<Number>} An array of potential values on our grid.
   */
  getPotential()
  {
    return this.#potential;
  }

  /**
   * Set the potential, V(x), used in the Schrödinger equation.
   *
   * @param {Array<Number>} potential An array of potential values on our grid.
   * @returns {Schrodinger}
   */
  setPotential(potential)
  {
    this.#potential = potential;
    return this;
  }

  /**
   * Set the boundary value delegate.
   *
   * @param {MurBoundary}   boundary    A MurBoundary instance, or similar boundary value class capable of
   *                                    working with the staggered time memory layout.
   */
  setBoundary(boundary)
  {
    this.#boundary = boundary;
    return this;
  }

  /**
   * Set whether ot not to use the boundary conditions. If true, the boundary conditions are
   * enforced, if false, they are ignored.
   *
   * @param {boolean} enabled True if boundary conditions are enabled, false if not.
   */
  setBCEnabled = function(enabled)
  {
    this.#bcEnabled = enabled;
    return this;
  }

  /**
   * Check whether boundary conditions are enabled. Returns true is they are enabled, false if not.
   *
   * @returns {boolean} True if boundary conditions are enabled, false if not.
   */
  isBCEnabled = function()
  {
    return bcEnabled;
  }

  /**
   * Get the bind group for the wave function parameters in the wave equation.
   *
   * @returns {GPUBindGroup} The bind group for the parameters buffer.
   * @see getParametersBindGroupLayout
   */
  getParametersBindGroup()
  {
    return this.#parametersBindGroup;
  }

  /**
   * Get the bind group layout for the schrodinger solver parameters.
   *
   * @returns {GPUBindGroupLayout} The bind group layout for the simulation parameters.
   * @see getParametersBindGroup
   */
  getParametersBindGroupLayout()
  {
    return this.#parametersBindGroupLayout;
  }

  /**
   * Get the wave function buffer for display or debugging.
   *
   * @returns {GPUBuffer} A wave function buffer.
   */
  getWaveFunctionBuffer()
  {
    return this.#waveFunctionBuffer;
  }

  /**
   * Get the FDTD parameters buffer for display or debugging.
   *
   * @returns {GPUBuffer} The FDTD parameters buffer.
   */
  getParametersBuffer()
  {
    return this.#parametersBuffer;
  }

  /**
   * Get the device we obtain resources from. Allows other classes to share resources.
   *
   * @returns {GPUDevice} The device in use for this simulation.
   */
  getDevice()
  {
    return this.#device;
  }

  /**
   * Async initialization of the object. JS does not allow async constructors, so the async initialization is here.
   * Invoke immediately after the constructor for a properly initialized object. Or get the object through
   * {@link getInstance}.
   *
   * @returns {Promise<Schrodinger>} A promise that resolves to the Schrodinger object.
   * @see getInstance
   */
  async init()
  {
    const timeStepShader = `
    struct Parameters {
        dt: f32,              // The time step, Δt.
        xResolution: u32,     // The number of points along the x-axis, the number of elements in the array.
        length: f32,          // The physical length for our simulation.
        potential: array<f32> // The potential the particle moves through.
    }

    // group 0, things that never change within a simulation.
    // The parameters for the simulation
    @group(0) @binding(0) var<storage, read> parameters: Parameters;
  
    // Group 1, Current and old wave function with Ψ_r at t and Ψ_i at t+Δt/2.
    @group(1) @binding(0) var<storage, read_write> waveFunction : array<vec2f>;
    @group(1) @binding(1) var<storage, read_write> oldWaveFunction: array<vec2f>;

    /**
     * Timestep the imaginary component of the wave function. For consistency and correctness we must
     * generate timesteps for the real part, then for the imaginary part of the wave function.
     */
    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn imaginaryPartTimeStep(@builtin(global_invocation_id) global_id : vec3u)
    {
      let index = global_id.x;
      // Skip invocations when work groups exceed the actual problem size
      if (index >= parameters.xResolution) {
        return;
      }
      // The potential and the wave function arrays have the same size.
      let dx = parameters.length / f32(parameters.xResolution-1);
      let dx22 = 2.0*dx*dx;
      
      let V = parameters.potential[index];
      let waveFunctionAtX = waveFunction[index];
      let waveFunctionAtXPlusDx = waveFunction[min(index+1, parameters.xResolution-1)];
      let waveFunctionAtXMinusDx = waveFunction[max(index-1, 0)];
    
      oldWaveFunction[index].y = waveFunctionAtX.y;
      waveFunction[index].y = waveFunctionAtX.y
                                        + ((waveFunctionAtXPlusDx.x - 2.0*waveFunctionAtX.x + waveFunctionAtXMinusDx.x)
                                            / dx22 - V*waveFunctionAtX.x) * parameters.dt;
    }
    
    /**
     * Timestep the real component of the wave function. For consistency and correctness we must
     * generate timesteps for the real part, then for the imaginary part of the wave function.
     */
    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn realPartTimeStep(@builtin(global_invocation_id) global_id : vec3u)
    {
      let index = global_id.x;
      // Skip invocations when work groups exceed the actual problem size
      if (index >= parameters.xResolution) {
        return;
      }
      // The potential and the wave function arrays have the same size.
      let dx = parameters.length / f32(parameters.xResolution-1);
      let dx22 = 2.0*dx*dx;
      
      let V = parameters.potential[index];
      let waveFunctionAtX = waveFunction[index];
      let waveFunctionAtXPlusDx = waveFunction[min(index+1, parameters.xResolution-1)];
      let waveFunctionAtXMinusDx = waveFunction[max(index-1, 0)];
    
      oldWaveFunction[index].x = waveFunctionAtX.x;
      waveFunction[index].x = waveFunctionAtX.x
                                    - ((waveFunctionAtXPlusDx.y - 2.0*waveFunctionAtX.y + waveFunctionAtXMinusDx.y)
                                        / dx22 - V*waveFunctionAtX.y) * parameters.dt;
    }
  `;

    const webgpuCompute = new WebGPUCompute();
    this.#device = await webgpuCompute.getDevice();

    const timeStepShaderModule = this.#device.createShaderModule({
      label: 'Schrodinger time step shader',
      code: timeStepShader
    });

    this.#parametersBindGroupLayout = this.#device.createBindGroupLayout({
      label: "Simulation parameters",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage"
          }
        }
      ]
    });

    this.#waveFunctionBindGroupLayout = this.#device.createBindGroupLayout({
      label: "Wave function data.",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage"
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage"
          }
        }
      ]
    });

    this.#imaginaryPartTimeStep = this.#device.createComputePipeline({
        label: "update imaginary part pipeline",
        layout: this.#device.createPipelineLayout({
        bindGroupLayouts: [this.#parametersBindGroupLayout, this.#waveFunctionBindGroupLayout]
      }),
      compute: {
        module: timeStepShaderModule,
        entryPoint: "imaginaryPartTimeStep"
      }
    });

    this.#realPartTimeStep = this.#device.createComputePipeline({
          label: "update real part pipeline",
          layout: this.#device.createPipelineLayout({
              bindGroupLayouts: [this.#parametersBindGroupLayout, this.#waveFunctionBindGroupLayout]
          }),
          compute: {
              module: timeStepShaderModule,
              entryPoint: "realPartTimeStep"
          }
      });

    this.#parametersBuffer = this.#device.createBuffer({
      label: "Parameters buffer",
      mappedAtCreation: true,
      size: Float32Array.BYTES_PER_ELEMENT                    // dt
          + Uint32Array.BYTES_PER_ELEMENT                     // xResolution
          + Float32Array.BYTES_PER_ELEMENT                    // length
          + this.#xResolution*Float32Array.BYTES_PER_ELEMENT, // potential
      usage:  this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
            // How we use this buffer, in the debug case we copy it to another buffer for reading
    });

    // Get the raw array buffer for the mapped GPU buffer
    const parametersArrayBuffer = this.#parametersBuffer.getMappedRange();

    let bytesSoFar = 0;
    new Float32Array(parametersArrayBuffer, bytesSoFar, 1).set([this.#dt]);
    bytesSoFar += Float32Array.BYTES_PER_ELEMENT;
    new Uint32Array(parametersArrayBuffer, bytesSoFar, 1).set([this.#xResolution]);
    bytesSoFar += Uint32Array.BYTES_PER_ELEMENT;
    new Float32Array(parametersArrayBuffer, bytesSoFar, 1).set([this.#length]);
    bytesSoFar += Float32Array.BYTES_PER_ELEMENT;
    if (this.#potential) {
      new Float32Array(parametersArrayBuffer, bytesSoFar, this.#xResolution).set(this.#potential);
    }

    // Unmap the buffer returning ownership to the GPU.
    this.#parametersBuffer.unmap();

    this.#parametersBindGroup = this.#device.createBindGroup({
      layout: this.#parametersBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.#parametersBuffer
          }
        }
      ]
    });

    // Wave function representations
    this.#waveFunctionBuffer = this.#device.createBuffer({
      label: "Wave function",
      size: 2*this.#xResolution*Float32Array.BYTES_PER_ELEMENT,
      usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
    });

    // Wave function representations
    this.#oldWaveFunctionBuffer = this.#device.createBuffer({
      label: "Old Wave function",
      size: 2*this.#xResolution*Float32Array.BYTES_PER_ELEMENT,
      usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
    });

    this.#waveFunctionBindGroup = this.#device.createBindGroup({
      label: "Wave function buffer binding",
      layout: this.#waveFunctionBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.#waveFunctionBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.#oldWaveFunctionBuffer
          }
        }
      ]
    });

    return this;
  }

  /**
   * Get an initialized instance of a Schrödinger equation integrator with the given parameters.
   *
   * @param {Number}        dt          The Δt between iterations of the wave function in natural units of time.
   * @param {Integer}       xResolution The size of the wave function arrays, the number of spatial steps
   *                                    on our 1D grid.
   * @param {Number}        length      The characteristic length for the problem in terms of natural units.
   * @param {Array<Number>} potential   An array of potential values, array elements correspond to physical locations
   *                                    just as the wave function arrays.
   * @param {Boolean}       debug       The debug option for this execution of the FDTD solver. Enabling this
   *                                    makes the wave function buffers copyable, potentially having a
   *                                    performance impact. Defaults to false.
   * @return Promise<Schrodinger> A promise that resolves to the requested Schrodinger instance.
   * @see #init
   */
  static async getInstance(dt, xResolution, length, potential, debug=false){
    const schrodinger = new Schrodinger(dt, xResolution, length, potential, debug);
    return schrodinger.init();
  }

  stop()
  {
    this.#running = false;
  }

  getWaveFunctionBindGroup() {
    return this.#waveFunctionBindGroup;
  }

  getWaveFunctionBindGroupLayout() {
    return this.#waveFunctionBindGroupLayout;
  }

  /**
   * Execute count iterations of the simulation. Each iteration consists of one update to the
   * real part of the wave function, and one update to the imaginary part of the wave function.
   *
   * @param {Integer} count The number of iterations to carry out.
   */
  step(count=21)
  {
    this.#running = true;

    // Recreate this because it can not be reused after finish is invoked.
    const commandEncoder = this.#device.createCommandEncoder();
    const workgroupCountX = Math.ceil(this.#xResolution / WORKGROUP_SIZE);
    for (let i=0; i<count && this.#running; i++)
    {
      const realPassEncoder = commandEncoder.beginComputePass();
      realPassEncoder.setPipeline(this.#realPartTimeStep);
      realPassEncoder.setBindGroup(0, this.#parametersBindGroup);
      realPassEncoder.setBindGroup(1, this.#waveFunctionBindGroup);
      realPassEncoder.dispatchWorkgroups(workgroupCountX);
      realPassEncoder.end();

      if (this.#bcEnabled) {
        this.#boundary.makeRealComputePass(commandEncoder, this.#waveFunctionBindGroup);
      }

      const imaginaryPassEncoder = commandEncoder.beginComputePass();
      imaginaryPassEncoder.setPipeline(this.#imaginaryPartTimeStep);
      imaginaryPassEncoder.setBindGroup(0, this.#parametersBindGroup);
      imaginaryPassEncoder.setBindGroup(1, this.#waveFunctionBindGroup);
      imaginaryPassEncoder.dispatchWorkgroups(workgroupCountX);
      imaginaryPassEncoder.end();

      if (this.#bcEnabled) {
        this.#boundary.makeImaginaryComputePass(commandEncoder, this.#waveFunctionBindGroup);
      }
    }

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.#device.queue.submit([gpuCommands]);
  }
}

export {Schrodinger}
