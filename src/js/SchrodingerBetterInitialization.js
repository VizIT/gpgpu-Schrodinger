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
 * Set initial values for the wave function buffer, which will be
 * subsequently evolved in time.
 */
class SchrodingerInitializer
{
  #device;
  #parametersBindGroup;
  #parametersBindGroupLayout;
  #xResolution;
  #x0;
  #w;
  #k;
  #wavePacketParametersBuffer;
  #wavePacketParametersBindGroupLayout;
  #wavePacketParametersBindGroup;
  #waveFunctionBindGroupLayout;
  #wavePacketShaderModule;
  #initialPipeline;
  #updatePipeline;

  /**
   * Build an initializer using the given WebGPU device, and with these wave packet parameters.
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

  getParametersBindGroup()
  {
    return this.#parametersBindGroup;
  }

  setParametersBindGroup(parametersBindGroup)
  {
    this.#parametersBindGroup = parametersBindGroup;
    return this;
  }

  getX0()
  {
    return this.#x0;
  }

  setX0(x0)
  {
    this.#x0 = x0;
  }

  getW()
  {
    return this.#w;
  }

  setW(w)
  {
    this.#w = w;
    return this;
  }

  getK()
  {
    return this.#k;
  }

  setK(k)
  {
    this.#k = k;
    return this;
  }

  /**
   *
   *
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
        
        // Initial wave function at t=0.
        @group(2) @binding(0) var<storage, read_write> waveFunction0 : array<vec2f>;
        @group(2) @binding(1) var<storage, read_write> waveFunction1 : array<vec2f>;
                         
        /**
         * Compute a time evolving Gaussian wave packet at x for a specific point in time.
         *
         * globalID An unsigned int giving the thread id for this invocation.
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
         * Populate initial arrays with a Gaussian free particle wave function.
         */
        @compute @workgroup_size(WORKGROUP_SIZE)
        fn computeInitialValues(@builtin(global_invocation_id) global_id : vec3u)
        {
            let index = global_id.x;
            // Skip invocations when work groups exceed the actual problem size
            if (index >= parameters.xResolution) {
                return;
            }
            waveFunction0[index] = computePsi(index, 0.0);
            waveFunction1[index] = computePsi(index, parameters.dt);
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

    this.#waveFunctionBindGroupLayout = this.#device.createBindGroupLayout({
      label: "Wave function layout",
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

    this.#initialPipeline = this.#device.createComputePipeline({
      layout: this.#device.createPipelineLayout({
        bindGroupLayouts: [
            this.#parametersBindGroupLayout, this.#wavePacketParametersBindGroupLayout,
            this.#waveFunctionBindGroupLayout
        ]
      }),
      compute: {
        module: this.#wavePacketShaderModule,
        entryPoint: "computeInitialValues"
      }
    });

    return this;
  };

  /**
   * Build an initializer using the given WebGPU device, and with these wave packet parameters.
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
    const initializer = new SchrodingerInitializer(device,parametersBindGroup,
        parametersBindGroupLayout, xResolution, x0, w, k);
    return initializer.init();
  }

  /**
   * Runs the compute shader. On exit the buffers are populated with the initial wave function values.
   *
   * @param {GPUBuffer} waveFunctionBuffer0 Data buffer to be initialized.
   * @param {GPUBuffer} waveFunctionBuffer1 Data buffer to be initialized.
   */
  initialize(waveFunctionBuffer0, waveFunctionBuffer1)
  {
    const waveFunctionBindGroup = this.#device.createBindGroup({
      layout: this.#waveFunctionBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: waveFunctionBuffer0
          }
        },
        {
          binding: 1,
          resource: {
            buffer: waveFunctionBuffer1
          }
        }
      ]});

    const commandEncoder = this.#device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.#initialPipeline);
    passEncoder.setBindGroup(0, this.#parametersBindGroup);
    passEncoder.setBindGroup(1, this.#wavePacketParametersBindGroup);
    passEncoder.setBindGroup(2, waveFunctionBindGroup);
    const workgroupCountX = Math.ceil(this.#xResolution / 64);
    passEncoder.dispatchWorkgroups(workgroupCountX);
    passEncoder.end();

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    this.#device.queue.submit([gpuCommands]);
  };

  /**
   * Invoke to clean up resources specific to this program.
   */
  done()
  {

  };
}

export {SchrodingerInitializer}