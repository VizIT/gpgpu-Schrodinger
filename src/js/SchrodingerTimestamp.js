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

const SECONDS_PER_NANOSECOND = 1.0E-09;

/**
 * A FDTD time evolver for the Schrödinger wave function. This time using timestamp queries
 * to time shader executions.
 *
 * @property {GPUDevice} #device The device as retrieved from the adaptor.
 * @property {Number} #dt The time step between the wave function and its updated version.
 * @property {Number} #length The physical length of this simulation.
 * @property {Integer} #xResolution The number of spatial steps in the wave function representation.
 * @property {Boolean} #running Whether the simulation is allowed to run. Setting this to false halts the simulation.
 * @property {GPUBuffer} #parametersBuffer The FDTD parameters buffer.
 * @property {GPUBindGroup} #parametersBindGroup The bind group for the parameters buffer.
 * @property {GPUBindGroupLayout} #parametersBindGroupLayout The bind group layout for the simulation parameters.
 * @property {GPUBuffer} #waveFunctionBuffer0 One of two wave function buffers.
 * @property {GPUBuffer} #waveFunctionBuffer1 One of two wave function buffers.
 * @property {GPUBindGroup[]} #waveFunctionBindGroup A pair of bind groups, used to ping pong the wave function buffers.
 * @property {GPUComputePipeline} #computePipeline The compute pipeline controlling some aspects of the shader execution.
 * @property {Boolean} #hasTimestampQuery A boolean indicating whether this system supports time stamp queries.
 * @property {GPUQuerySet} #timestampQueries The set of timestamp queries. Tracks the queries and their results.
 * @property {GPUBuffer} #timestampBuffer Query results are copied from the query set to this buffer.
 * @property {GPUBuffer} #timestampCopyBuffer Query results are then copied to this buffer where they can be mapped to main memory.
 * @property {HTMLElement} #timestampDisplay HTML element used to display timing data.
 * @property {Number} #maxDeltaT The max difference between time stamps, the max time taken by the compute shader.
 * @property {Boolean} #debug A flag indicating whether this is a debugging instance.
 */
class Schrodinger
{
  #device;
  #dt;
  #length;
  #xResolution;
  #running;
  #parametersBuffer;
  #parametersBindGroup;
  #parametersBindGroupLayout;
  #waveFunctionBuffer0;
  #waveFunctionBuffer1;
  #waveFunctionBindGroup = new Array(2);
  #computePipeline;
  #hasTimestampQuery;
  #timestampQueries;
  #timestampBuffer;
  #timestampCopyBuffer;
  #timestampDisplay;
  #maxDeltaT = 0n;
  #debug;

  /**
   * Build a Schrödinger equation integrator with the given parameters.
   *
   * @param {Number}       dt               The Δt between iterations of the wave function in natural units of time.
   * @param {Integer}      xResolution      The size of the wave function arrays, the number of spatial steps
   *                                        on our 1D grid.
   * @param {Number}       length           The characteristic length for the problem in terms of natural units.
   * @param {HTMLElement}  timestampDisplay The HTML element used to display shader timing data.
   * @param {Boolean}      debug            The debug option for this execution of the FDTD solver. Enabling this
   *                                        makes the wave function buffers copyable, potentially having a
   *                                        performance impact. Defaults to false.
   */
  constructor(dt, xResolution, length, timestampDisplay, debug=false)
  {
    this.#dt = dt;
    this.#xResolution = xResolution;
    this.#length = length;
    this.#timestampDisplay = timestampDisplay;
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
   * @param xResolution The number of spatial steps in the wave function representation.
   * @returns {Schrodinger}
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
    @group(1) @binding(0) var<storage, read> waveFunction : array<vec2f>;
    // The updated wave function at t+Δt.
    @group(1) @binding(1) var<storage, read_write> updatedWaveFunction : array<vec2f>;
  
    @compute @workgroup_size(64)
    fn timeStep(@builtin(global_invocation_id) global_id : vec3u)
    {
      let index = global_id.x;
      // Skip invocations when work groups exceed the actual problem size
      if (index >= parameters.xResolution) {
        return;
      }
      // waveFunction and updatedWaveFunction have the same size.
      let dx = parameters.length / f32(parameters.xResolution-1);
      let dx22 = dx*dx*2.0;
      let waveFunctionAtX = waveFunction[index];
      let waveFunctionAtXPlusDx = waveFunction[min(index+1, parameters.xResolution-1)];
      let waveFunctionAtXMinusDx = waveFunction[max(index-1, 0)];
    
      updatedWaveFunction[index].x = waveFunctionAtX.x
                                    - ((waveFunctionAtXPlusDx.y - 2.0*waveFunctionAtX.y + waveFunctionAtXMinusDx.y)
                                        / dx22) * parameters.dt;

      updatedWaveFunction[index].y = waveFunctionAtX.y
                                        + ((waveFunctionAtXPlusDx.x - 2.0*waveFunctionAtX.x + waveFunctionAtXMinusDx.x)
                                            / dx22) * parameters.dt;
    }
  `;

    const webgpuCompute = new WebGPUCompute();
    this.#hasTimestampQuery = await webgpuCompute.hasTimestampQuery();
    if (this.#hasTimestampQuery) {
      this.#device = await webgpuCompute.getTimestampDevice();
    } else {
      this.#device = await webgpuCompute.getDevice();
    }


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
            type: "read-only-storage"
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
        layout: this.#device.createPipelineLayout({
        bindGroupLayouts: [this.#parametersBindGroupLayout, waveFunctionBindGroupLayout]
      }),
      compute: {
        module: timeStepShaderModule,
        entryPoint: "timeStep"
      }
    });

    this.#parametersBuffer = this.#device.createBuffer({
      label: "Parameters buffer",
      mappedAtCreation: true,
      size: Float32Array.BYTES_PER_ELEMENT                    // dt
          + Uint32Array.BYTES_PER_ELEMENT                     // xResolution
          + Float32Array.BYTES_PER_ELEMENT,                   // length
      usage:  this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
            // How we use this buffer, in the debug case we copy it to another buffer for reading
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

    // Wave function representations
    this.#waveFunctionBuffer0 = this.#device.createBuffer({
      label: "Wave function 0",
      size: 2*this.#xResolution*Float32Array.BYTES_PER_ELEMENT,
      usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
    });

    this.#waveFunctionBuffer1 = this.#device.createBuffer({
      label: "Wave function 1",
      size: 2*this.#xResolution*Float32Array.BYTES_PER_ELEMENT,
      usage: this.#debug ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.STORAGE
    });

    this.#waveFunctionBindGroup[0] = this.#device.createBindGroup({
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

    this.#waveFunctionBindGroup[1] = this.#device.createBindGroup({
      layout: waveFunctionBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer:  this.#waveFunctionBuffer1
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.#waveFunctionBuffer0
          }
        }
      ]});

    if (this.#hasTimestampQuery) {
      // We expect to submit two queries, one at the top of the compute pass,
      // and the second at the bottom of the compute pass.
      this.#timestampQueries = await webgpuCompute.createQuerySet(WebGPUCompute.TIMESTAMP_QUERY_TYPE, 2);
      this.#timestampBuffer = this.#device.createBuffer({
        label: "Time stamp query buffer",
        size: this.#timestampQueries.count * BigInt64Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      // This is the buffer we map to a CPU side array buffer, it is of necessity the same size as the
      // query results buffer
      this.#timestampCopyBuffer = this.#device.createBuffer({
        label: "Time stamp mappable buffer",
        size: this.#timestampBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    return this;
  }

  /**
   * Get an initialized instance of a Schrödinger equation integrator with the given parameters.
   *
   * @param {Number}        dt               The Δt between iterations of the wave function in natural units of time.
   * @param {Integer}       xResolution      The size of the wave function arrays, the number of spatial steps
   *                                         on our 1D grid.
   * @param {Number}        length           The characteristic length for the problem in terms of natural units.
   * @param {HTMLElement}   timestampDisplay An HTML element where we can display the timestamp results.
   * @param {Boolean}       debug            The debug option for this execution of the FDTD solver. Enabling this
   *                                         makes the wave function buffers copyable, potentially having a
   *                                         performance impact. Defaults to false.
   * @return Promise<Schrodinger> A promise that resolves to the requested Schrodinger instance.
   * @see #init
   */
  static async getInstance(dt, xResolution, length, timestampDisplay, debug=false){
    const schrodinger = new Schrodinger(dt, xResolution, length, timestampDisplay, debug);
    return schrodinger.init();
  }

  stop()
  {
    this.#running = false;
  }

  /**
   * Execute count iterations of the simulation.
   *
   * @param {Integer} count The number of iterations to carry out.
   */
  async step(count=20)
  {
    this.#running = true;
    let deltaT = 0n;

    const workgroupCountX = Math.ceil(this.#xResolution / 64);
    for (let i=0; i<count && this.#running; i++)
    {
      // Created in the loop because it can not be reused after finish is invoked.
      const commandEncoder = this.#device.createCommandEncoder();

      const passEncoder = commandEncoder.beginComputePass({
        timestampWrites: {
          querySet: this.#timestampQueries,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1
        }
      });
      passEncoder.setPipeline(this.#computePipeline);
      passEncoder.setBindGroup(0, this.#parametersBindGroup);
      passEncoder.setBindGroup(1, this.#waveFunctionBindGroup[i%2]);
      passEncoder.dispatchWorkgroups(workgroupCountX);
      passEncoder.end();

      // Resolve the timestamp queries we set in the command stream.
      // https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/resolveQuerySet
      commandEncoder.resolveQuerySet(
          this.#timestampQueries,        // The GPUQuerySet
          0,                             // The first query
          this.#timestampQueries.count,  // The query count
          this.#timestampBuffer,         // GPUBuffer destination
          0);                            // Destination offset

      // Copy the query resolve buffer to a CPU mappable buffer
      // https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/copyBufferToBuffer
      commandEncoder.copyBufferToBuffer(
          this.#timestampBuffer,     // GPUBuffer we copy from
          0,                         // Start at the beginning of the buffer
          this.#timestampCopyBuffer, // GPUBuffer we copy to
          0,                         // Starting at the beginning of the destination
          this.#timestampBuffer.size // Copy the full contents of the source buffer
      );

      // Submit GPU commands.
      this.#device.queue.submit([commandEncoder.finish()]);
    }
    // Now we map this buffer to the CPU side.
    await this.#timestampCopyBuffer.mapAsync(GPUMapMode.READ);


    const timestampArrayBuffer = this.#timestampCopyBuffer.getMappedRange();
    const timestampNanoseconds = new BigInt64Array(timestampArrayBuffer);

    deltaT = timestampNanoseconds[1] - timestampNanoseconds[0];
    if (deltaT > this.#maxDeltaT) {
      this.#maxDeltaT = deltaT;
      this.#timestampDisplay.innerText = (Number(deltaT)*SECONDS_PER_NANOSECOND).toExponential(2);
    }

    this.#timestampCopyBuffer.unmap();

  }
}

export {Schrodinger}
