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

/**
 * Evaluate a free particle gaussian wave function. Providing values either for initializing
 * a numerical time evolution, or for exact values for a free particle evolved in time. This
 * version uses a staggered time in that Ψ_i is computed at t-Δt/2 and Ψ_r is computed at t+Δt/2.
 */
class SchrodingerFreeParticle
{
  #device;
  #parametersBindGroup;
  #parametersBindGroupLayout;
  #xResolution;
  #x0;
  #w;
  #k;
  #waveFunctionBuffer;
  #waveFunctionBindGroup;
  #wavePacketParametersBuffer;
  #wavePacketParametersBindGroupLayout;
  #wavePacketParametersBindGroup;
  #wavePacketShaderModule;
  #timeEvolvedPipeline;
  #timeBuffer;
  #timeWaveFunctionBuffer;
  #timeBindGroup;
  #timeBindGroupLayout;

  /**
   * Build a free particle object using the given WebGPU device, and with these wave packet parameters.
   *
   * @param {GPUDevice} device The device in use for the simulation, to allow buffer sharing.
   * @param {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters,
   *                                           carried over from the Schrödinger solver.
   * @param {GPUBindGroupLayout} parametersBindGroupLayout The bind group layout for the simulation parameters
   *                                                       from the Schrödinger simulation.
   * @param {Number} xResolution The number of data points on the grid in the x-direction.
   * @param {Number} x0 The center of the wave packet.
   * @param {Number} w The wave packet width.
   * @param {Number} k The wave number, momentum.
   */
  constructor(device,parametersBindGroup, parametersBindGroupLayout,
              xResolution, x0, w, k)
  {
    this.#device = device;
    this.#parametersBindGroup = parametersBindGroup;
    this.#parametersBindGroupLayout = parametersBindGroupLayout;
    this.#xResolution = xResolution;
    this.#x0 = x0;
    this.#w = w;
    this.#k = k;
  }

  getWaveFunctionBuffer() {
    return this.#waveFunctionBuffer;
  }

  /**
   * Setup shader, buffers, and set parameters. On exit, we are ready to issue commands and execute the
   * shader.
   */
  init()
  {
    const shaderSource = `
        const PI : f32 = 3.1415926535897932384626433832795;
        const fourthRoot2 = 1.1892071150027210667174999;
        const WORKGROUP_SIZE = 64;
                         
        struct Parameters {
            dt: f32,          // The time step size
            xResolution: u32, // The number of points along the x-axis, the number of elements in the array.
            length: f32       // The full length for our simulation
        }
        
        struct WavePacketParameters {
            x0: f32, // The center of the wave packet.
            w: f32,  // The width of the wave packet, in GeV^-1.
            k: f32   // The wave number, from particle momentum.
        }
        
        // group 0, parameters that never change over the course of a simulation
        @group(0) @binding(0) var<storage, read> parameters: Parameters;
        
        // Group 1, parameters describing the wave packet
        @group(1) @binding(0) var<storage, read> wavePacketParameters : WavePacketParameters;
        
        // For wave function evaluation at arbitrary times
        @group(2) @binding(0) var<storage, read> time: f32;
        @group(2) @binding(1) var<storage, read_write> timeWaveFunction : array<vec2f>;
                         
        /**
         * Compute a time evolving gaussian wave packet at x and at a specific point in time.
         *
         * globalID An unsigned in giving the thread id for this invocation.
         * t A float giving the current time.
         */
        fn computePsi(globalID: u32, t: f32) -> vec2f
        {
            let x = f32(globalID) * parameters.length / f32(parameters.xResolution-1);
            let w2 = wavePacketParameters.w * wavePacketParameters.w;
            // In our units, hbar = 1
            let p0 = wavePacketParameters.k;
            let deltaX = x - wavePacketParameters.x0;
            let theta = atan(2.0*t/w2)/2.0;
            let phi = -theta-p0*p0*t/2.0;
            let a = w2*w2 + 4.0*t*t;
            let b = (deltaX-p0*t)*(deltaX-p0*t);
            let c = phi + p0*deltaX + 2.0*t*b/a;

            let phase = vec2(cos(c), sin(c));            
            let magnitude = pow(2.0*w2/(PI*a), 0.25) * exp(-b*w2/a);

            return magnitude*phase;
        }
        
        /*
         * Populate an array with the wave function for the given time.
         */
        @compute @workgroup_size(WORKGROUP_SIZE)
        fn computeTimeValues(@builtin(global_invocation_id) global_id : vec3u)
        {
            let index = global_id.x;
            // Skip invocations when work groups exceed the actual problem size
            if (index >= parameters.xResolution) {
                return;
            }
            let halfDt = parameters.dt / 2.0;
            // Real part of the wave function at t.
            timeWaveFunction[index].x = computePsi(index, time).x;
            // Imiginary part of the wave function at t+dt/2, hence staggered time
            timeWaveFunction[index].y = computePsi(index, time+halfDt).y;
        }
    `;

    this.#wavePacketShaderModule = this.#device.createShaderModule({
      label: 'Wave packet shader',
      code: shaderSource
    });

    this.#wavePacketParametersBindGroupLayout = this.#device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage"
          }
        }
      ]
    });

    this.#wavePacketParametersBuffer = this.#device.createBuffer({
      label: 'Wave packet Parameters',
      mappedAtCreation: true,
      size: 3*Float32Array.BYTES_PER_ELEMENT, // x0, w, k are all Float 32
      usage: GPUBufferUsage.STORAGE
    });

    // Get the raw array buffer for the mapped GPU buffer
    const wavePacketParametersArrayBuffer = this.#wavePacketParametersBuffer.getMappedRange();

    let bytesSoFar = 0;
    new Float32Array(wavePacketParametersArrayBuffer, bytesSoFar, 1).set([this.#x0]);
    bytesSoFar += Float32Array.BYTES_PER_ELEMENT;
    new Float32Array(wavePacketParametersArrayBuffer, bytesSoFar, 1).set([this.#w]);
    bytesSoFar += Float32Array.BYTES_PER_ELEMENT;
    new Float32Array(wavePacketParametersArrayBuffer, bytesSoFar, 1).set([this.#k]);

    this.#wavePacketParametersBuffer.unmap()

    this.#wavePacketParametersBindGroup = this.#device.createBindGroup({
      layout: this.#wavePacketParametersBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.#wavePacketParametersBuffer
          }
        }
      ]});

    // A single floating point value holding the time value for the wave function.
    this.#timeBuffer = this.#device.createBuffer({
      label: "Time buffer",
      size: Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.#timeWaveFunctionBuffer = this.#device.createBuffer({
      label: "Time wave function",
      size: 2*this.#xResolution*Float32Array.BYTES_PER_ELEMENT,
      // we compute this in the shader, then copy it down to the CPU.
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
    });

    this.#timeBindGroupLayout = this.#device.createBindGroupLayout({
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

    this.#timeBindGroup = this.#device.createBindGroup({
      layout: this.#timeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.#timeBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: this.#timeWaveFunctionBuffer
          }
        }
      ]
    });

    this.#timeEvolvedPipeline = this.#device.createComputePipeline({
      layout: this.#device.createPipelineLayout({
        bindGroupLayouts: [
          this.#parametersBindGroupLayout, this.#wavePacketParametersBindGroupLayout,
          this.#timeBindGroupLayout
        ]
      }),
      compute: {
        module: this.#wavePacketShaderModule,
        entryPoint: "computeTimeValues"
      }
    });

    return this;
  };

  /**
   * Build a free particle instance using the given WebGPU device, and with these wave packet parameters.
   *
   * @param {GPUDevice} device The device in use for the simulation, to allow buffer sharing.
   * @param {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters,
   *                                           carried over from the Schrödinger solver.
   * @param {GPUBindGroupLayout} parametersBindGroupLayout The bind group layout for the simulation parameters
   *                                                       from the Schrödinger simulation.
   * @param {Number} xResolution The number of data points on the grid in the x-direction.
   * @param {Number} x0 The center of the wave packet.
   * @param {Number} w The wave packet width.
   * @param {Number} k The wave number, momentum.
   */
  static async getInstance(device,parametersBindGroup, parametersBindGroupLayout,
                           xResolution, x0, w, k){
    const initializer = new SchrodingerFreeParticle(device,parametersBindGroup,
        parametersBindGroupLayout, xResolution, x0, w, k);
    return initializer.init();
  }

  /**
   * Evaluate the free particle wave function at a given time. At the conclusion, the timeWaveFunctionBuffer
   * is populated with the wave function values.
   *
   * @param {Number}       time          The time at which to evaluate the wave function.
   * @param {GPUBindGroup} timeBindGroup The bind group containing the time and the time dependant wave function.
   * @see #getTimeWaveFunctionBuffer
   */
  doTimeWaveFunction(time, timeBindGroup) {

    this.#device.queue.writeBuffer(this.#timeBuffer, 0, new Float32Array([time]), 0, 1);
    const commandEncoder = this.#device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.#timeEvolvedPipeline);
    passEncoder.setBindGroup(0, this.#parametersBindGroup);
    passEncoder.setBindGroup(1, this.#wavePacketParametersBindGroup);
    passEncoder.setBindGroup(2, timeBindGroup);
    const workgroupCountX = Math.ceil(this.#xResolution / 64);
    passEncoder.dispatchWorkgroups(workgroupCountX);
    passEncoder.end();

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.#device.queue.submit([gpuCommands]);
  }

  getTimeWaveFunction(time) {
    if (!this.#waveFunctionBuffer) {
      this.#waveFunctionBuffer = this.#device.createBuffer({
        label: 'Time wave function',
        mappedAtCreation: false,
        // Real and imaginary parts to the wave function
        size: 2*this.#xResolution*Float32Array.BYTES_PER_ELEMENT,
        // we compute this in the shader, then copy it down to the CPU.
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
      });

      this.#waveFunctionBindGroup = this.#device.createBindGroup({
        layout: this.#timeBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.#timeBuffer
            }
          },
          {
            binding: 1,
            resource: {
              buffer: this.#waveFunctionBuffer
            }
          }
        ]
      });
    }
    this.doTimeWaveFunction(time, this.#waveFunctionBindGroup);
  }

  /**
   * Runs the compute shader. On exit the buffer is populated with the initial wave function values.
   * Ψ_r is computed at t, Ψ_i is computed at t+Δt/2 and with t=0.
   *
   * @param {Number}    dt                 The FDTD timestep.
   * @param {GPUBuffer} waveFunctionBuffer Data buffer to be initialized.
   */
  initialize(dt, waveFunctionBuffer)
  {
    const waveFunctionBindGroup = this.#device.createBindGroup({
      layout: this.#timeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.#timeBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: waveFunctionBuffer
          }
        }
      ]
    });
    this.doTimeWaveFunction(0.0, waveFunctionBindGroup);
  }
}

export {SchrodingerFreeParticle}