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

const workgroupSizeXLimitName =  "maxComputeWorkgroupSizeX";
const invocationsPerWorkgroupLimitName = "maxComputeInvocationsPerWorkgroup";

/**
 * @typedef {Number} Integer
 */

/**
 * An FDTD time evolver for the Schrödinger wave function with a workgroup size matching the
 * problem x resolution.
 *
 * @property {GPUDevice} #device The device as retrieved from the adaptor.
 * @property {Boolean} #invocationLimitAvailable Whether the requested invocation limit is available on this system.
 * @property {Boolean} #workgroupSizeLimitAvailable Whether the requested workgroup size limit is available on this system.
 * @property {Number} #dt The time step between the wave function and its updated version.
 * @property {Number} #length The physical length of this simulation.
 * @property {Integer} #xResolution The number of spatial steps in the wave function representation.
 * @property {Integer} #iterations The number of loop iterations to execute in the compute shader.
 * @property {Boolean} #running Whether the simulation is allowed to run. Setting this to false halts the simulation.
 * @property {GPUBuffer} #parametersBuffer The FDTD parameters buffer.
 * @property {GPUBindGroup} #parametersBindGroup The bind group for the parameters buffer.
 * @property {GPUBindGroupLayout} #parametersBindGroupLayout The bind group layout for the simulation parameters.
 * @property {GPUBuffer} #waveFunctionBuffer0 One of two wave function buffers.
 * @property {GPUBuffer} #waveFunctionBuffer1 One of two wave function buffers.
 * @property {GPUBindGroup[]} #waveFunctionBindGroup A pair of bind groups, used to ping pong the wave function buffers.
 * @property {GPUComputePipeline} #computePipeline The compute pipeline controlling some aspects of the shader execution.
 * @property {Boolean} #debug A flag indicating whether this is a debugging instance.
 */
class Schrodinger
{
  #device;
  #invocationLimitAvailable;
  #workgroupSizeLimitAvailable;
  #dt;
  #length;
  #xResolution;
  #iterations;
  #running;
  #parametersBuffer;
  #parametersBindGroup;
  #parametersBindGroupLayout;
  #waveFunctionBuffer0;
  #waveFunctionBuffer1;
  #waveFunctionBindGroup;
  #computePipeline;
  #debug;

  /**
   * Build a Schrödinger equation integrator with the given parameters.
   *
   * @param {Number}        dt          The Δt between iterations of the wave function in natural units of time.
   * @param {Integer}       xResolution The size of the wave function arrays, the number of spatial steps
   *                                    on our 1D grid.
   * @param {Number}        length      The characteristic length for the problem in terms of natural units.
   * @param {Integer}       iterations  The number of iterations executed by the step method.
   * @param {Boolean}       debug       The debug option for this execution of the FDTD solver. Enabling this
   *                                    makes the wave function buffers copyable, potentially having a
   *                                    performance impact. Defaults to false.
   */
  constructor(dt, xResolution, length, iterations, debug=false)
  {
    this.#dt = dt;
    this.#xResolution = xResolution;
    this.#length = length;
    this.#iterations = iterations;
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
   * Get whether the required invocation limit was available.
   *
   * @returns {Boolean} True if the available invocation limit is at least as large as xResolution,
   * false if not. Undefined if the object has not been initialized.
   */
  isInvocationLimitAvailable()
  {
    return this.#invocationLimitAvailable;
  }

  /**
   * Get whether the work groups X size limit was at least as large as the xResolution.
   *
   * @returns {Boolean} True if the X workgroup size limit is at least as large as the xResolution,
   * false if not. Is undefined if the object is not yet initialized.
   */
  isWorkgroupSizeLimitAvailable()
  {
    return this.#workgroupSizeLimitAvailable;
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
   * Get a wave function buffer for display, or debugging.
   *
   * @returns {GPUBuffer} A wave function buffer.
   */
  getWaveFunctionBuffer0()
  {
    return this.#waveFunctionBuffer0;
  }

  /**
   * Get a wave function buffer for display, or debugging.
   *
   * @returns {GPUBuffer} A wave function buffer.
   */
  getWaveFunctionBuffer1()
  {
    return this.#waveFunctionBuffer1;
  }

  /**
   * Get the FDTD parameters buffer for display, or debugging.
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
        dt: f32,          // The time step, Δt.
        xResolution: u32, // The number of points along the x-axis, the number of elements in the array.
        length: f32       // The physical length for our simulation.
    }

    // group 0, things that never change within a simulation.
    // The parameters for the simulation
    @group(0) @binding(0) var<storage, read> parameters: Parameters;
  
    //group 1, changes on each iteration
    // Initial wave function at t.
    @group(1) @binding(0) var<storage, read_write> waveFunction : array<vec2f>;
    // The updated wave function at t+Δt.
    @group(1) @binding(1) var<storage, read_write> updatedWaveFunction : array<vec2f>;
    
    override iterations = 250;
    override xResolution : u32 = 1024;
  
    @compute @workgroup_size(xResolution)
    fn timeSteps(@builtin(global_invocation_id) global_id : vec3u)
    {
      let index = global_id.x;
      // waveFunction and updatedWaveFunction have the same size.
      let dx = parameters.length / f32(xResolution-1);
      let dx22 = dx*dx*2.0;
      
      for (var i=0; i<iterations; i++)
      {
          var waveFunctionAtX = waveFunction[index];
          var waveFunctionAtXPlusDx = waveFunction[min(index+1, xResolution-1)];
          var waveFunctionAtXMinusDx = waveFunction[max(index-1, 0)];
    
          updatedWaveFunction[index].x = waveFunctionAtX.x
                      - ((waveFunctionAtXPlusDx.y - 2.0*waveFunctionAtX.y + waveFunctionAtXMinusDx.y)
                          / dx22) * parameters.dt;

          updatedWaveFunction[index].y = waveFunctionAtX.y
                      + ((waveFunctionAtXPlusDx.x - 2.0*waveFunctionAtX.x + waveFunctionAtXMinusDx.x)
                          / dx22) * parameters.dt;
          storageBarrier();
          
          waveFunctionAtX = updatedWaveFunction[index];
          waveFunctionAtXPlusDx = updatedWaveFunction[min(index+1, xResolution-1)];
          waveFunctionAtXMinusDx = updatedWaveFunction[max(index-1, 0)];
      
          waveFunction[index].x = waveFunctionAtX.x
                      - ((waveFunctionAtXPlusDx.y - 2.0*waveFunctionAtX.y + waveFunctionAtXMinusDx.y)
                          / dx22) * parameters.dt;

          waveFunction[index].y = waveFunctionAtX.y
                      + ((waveFunctionAtXPlusDx.x - 2.0*waveFunctionAtX.x + waveFunctionAtXMinusDx.x)
                          / dx22) * parameters.dt;
          storageBarrier();
      }
    }
  `;

    const deviceDescriptor = {
      requiredLimits: {
        maxComputeWorkgroupSizeX : this.#xResolution,
        maxComputeInvocationsPerWorkgroup: this.#xResolution
      }
    }

    const webgpuCompute = new WebGPUCompute(null, deviceDescriptor);

    this.#workgroupSizeLimitAvailable = await webgpuCompute.checkLimit(workgroupSizeXLimitName,
                                                                        this.#xResolution);
    this.#invocationLimitAvailable = await webgpuCompute.checkLimit(invocationsPerWorkgroupLimitName,
                                                                      this.#xResolution);

    if (this.#workgroupSizeLimitAvailable && this.#invocationLimitAvailable) {
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

      const waveFunctionBindGroupLayout = this.#device.createBindGroupLayout({
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

      this.#computePipeline = this.#device.createComputePipeline({
        label: "Tuned shader pipeline.",
        layout: this.#device.createPipelineLayout({
          bindGroupLayouts: [this.#parametersBindGroupLayout, waveFunctionBindGroupLayout]
        }),
        compute: {
          module: timeStepShaderModule,
          entryPoint: "timeSteps",
          constants: {
            iterations: this.#iterations,
            xResolution: this.#xResolution
          }
        }
      });

      this.#parametersBuffer = this.#device.createBuffer({
        label: "Parameters buffer",
        mappedAtCreation: true,
        size: Float32Array.BYTES_PER_ELEMENT                    // dt
            + Uint32Array.BYTES_PER_ELEMENT                     // xResolution
            + Float32Array.BYTES_PER_ELEMENT,                   // length
        usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
      });

      // Get the raw array buffer for the mapped GPU buffer
      const parametersArrayBuffer = this.#parametersBuffer.getMappedRange();

      var bytesSoFar = 0;
      new Float32Array(parametersArrayBuffer, bytesSoFar, 1).set([this.#dt]);
      bytesSoFar += Float32Array.BYTES_PER_ELEMENT;
      new Uint32Array(parametersArrayBuffer, bytesSoFar, 1).set([this.#xResolution]);
      bytesSoFar += Uint32Array.BYTES_PER_ELEMENT;
      new Float32Array(parametersArrayBuffer, bytesSoFar, 1).set([this.#length]);

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

      this.#waveFunctionBuffer0 = this.#device.createBuffer({
        label: "Wave function 0",
        size: 2 * this.#xResolution * Float32Array.BYTES_PER_ELEMENT, // A wave function representation
        usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
      });

      this.#waveFunctionBuffer1 = this.#device.createBuffer({
        label: "Wave function 1",
        size: 2 * this.#xResolution * Float32Array.BYTES_PER_ELEMENT, // A wave function representation
        usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
      });

      this.#waveFunctionBindGroup = this.#device.createBindGroup({
        layout: waveFunctionBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.#waveFunctionBuffer0
            }
          },
          {
            binding: 1,
            resource: {
              buffer: this.#waveFunctionBuffer1
            }
          }
        ]
      });
    }

    return this;
  }

  /**
   * Get an initialized instance of a Schrödinger equation integrator with the given parameters.
   *
   * @param {Number}        dt          The Δt between iterations of the wave function in natural units of time.
   * @param {Integer}       xResolution The size of the wave function arrays, the number of spatial steps
   *                                    on our 1D grid.
   * @param {Number}        length      The characteristic length for the problem in terms of natural units.
   * @param {Integer}       iterations  The number of iterations executed by the step method.
   * @param {Boolean}       debug       The debug option for this execution of the FDTD solver. Enabling this
   *                                    makes the wave function buffers copyable, potentially having a
   *                                    performance impact. Defaults to false.
   * @return Promise<Schrodinger> A promise that resolves to the requested Schrodinger instance.
   * @see #init
   */
  static async getInstance(dt, xResolution, length, iterations, debug=false){
    const schrodinger = new Schrodinger(dt, xResolution, length, iterations, debug);
    return schrodinger.init();
  }

  stop()
  {
    this.#running = false;
  }

  /**
   * Execute count iterations of the simulation.
   */
  step()
  {
    if (!this.#workgroupSizeLimitAvailable) {
      throw new Error(`${this.constructor.name}: Required ${workgroupSizeXLimitName} of ${this.#xResolution} is unavailable.`);
    }

    if (!this.#invocationLimitAvailable) {
      throw new Error(`${this.constructor.name}: Required ${invocationsPerWorkgroupLimitName} of ${this.#xResolution} is unavailable.`);
    }

    const commandEncoder = this.#device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.#computePipeline);
    passEncoder.setBindGroup(0, this.#parametersBindGroup);
    passEncoder.setBindGroup(1, this.#waveFunctionBindGroup);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.#device.queue.submit([gpuCommands]);
  }
}

export {Schrodinger}
