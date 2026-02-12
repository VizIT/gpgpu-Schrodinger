/**
 * Copyright 2026 Vizit Solutions
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
 * Implement the Mur boundary conditions, which force the boundaries to represent an outgoing wave.
 */
class MurBoundary
{
    #device;
    #boundaryValueShaderModule;
    #boundaryValueBindGroup;
    #boundaryValueParametersLayout;
    #boundaryValueParameters;
    #parametersBindGroup;
    #boundaryValueRealPipeline;
    #boundaryValueImaginaryPipeline;

  /**
   * Create a Mur boundary value object for use with the leapfrog Schrodinger implementation.
   *
   * @param {Schrodinger} schrodinger The Schrodinger FDTD instance.
   * @param {Number}      E           The characteristic energy of the wave or wave packet.
   * @param {Boolean}     debug       Indicates whether we are debugging this run, makes buffers copyable.
   */
    constructor(schrodinger, E, debug=false)
    {
        this.#device = schrodinger.getDevice();
        this.#parametersBindGroup = schrodinger.getParametersBindGroup();
        // Phase velocity = w/k = K+V/Sqrt(2mK), w/V=0, m=1
        const phaseVelocity = Math.sqrt(0.5*E);
        this.init(schrodinger, phaseVelocity, debug);
    }

  /**
   * Initialize the boundary shader, set up buffers and load parameters.
   *
   * @param {Schrodinger} schrodinger   The Schrodinger FDTD instance.
   * @param {Number}      phaseVelocity The characteristic phase velocity of the wave or wave packet.
   * @param {Boolean}     debug         Indicates whether we are debugging this run, makes buffers copyable.
   * @returns {MurBoundary}
   */
  init(schrodinger, phaseVelocity, debug)
    {
        const boundaryConditionsShader = `
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
          
          // Group 2, boundary value specific data.
          @group(2) @binding(0) var<uniform> phaseVelocity : f32;
          
          @compute @workgroup_size(2)
          fn recomputeRealBoundary(@builtin(global_invocation_id) global_id : vec3u)
          {
            // Index will be 0 at the left edge, and 1 at the right edge.
            let index = global_id.x;
            let sotrageBufferIndex = index*(parameters.xResolution-1);
            let offset = 1 - 2*index;
            let dx = parameters.length / f32(parameters.xResolution-1);
            
            waveFunction[sotrageBufferIndex].x = oldWaveFunction[sotrageBufferIndex+offset].x
                 + ((phaseVelocity*parameters.dt-dx)/(phaseVelocity*parameters.dt+dx))
                    *(waveFunction[sotrageBufferIndex+offset].x-oldWaveFunction[sotrageBufferIndex].x);
          }
          
          @compute @workgroup_size(2)
          fn recomputeImaginaryBoundary(@builtin(global_invocation_id) global_id : vec3u)
          {
            // Index will be 0 at the left edge, and 1 at the right edge.
            let index = global_id.x;
            let sotrageBufferIndex = index*(parameters.xResolution-1);
            let offset = 1 - 2*index;
            let dx = parameters.length / f32(parameters.xResolution-1);
            
            waveFunction[sotrageBufferIndex].y = oldWaveFunction[sotrageBufferIndex+offset].y
                 + ((phaseVelocity*parameters.dt-dx)/(phaseVelocity*parameters.dt+dx))
                    *(waveFunction[sotrageBufferIndex+offset].y-oldWaveFunction[sotrageBufferIndex].y);
          }
        `;

        this.#boundaryValueShaderModule = this.#device.createShaderModule({
            label: 'Mur Boundary Shader shader',
            code: boundaryConditionsShader
        });

        this.#boundaryValueParametersLayout = this.#device.createBindGroupLayout({
            label: "Mur Boundary parameters layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {}
                }
            ]
        });

        this.#boundaryValueParameters = this.#device.createBuffer({
            label: "Boundary parameters buffer",
            mappedAtCreation: true,
            size: Float32Array.BYTES_PER_ELEMENT,    // the single phase velocity parameter
            usage:  debug ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            // How we use this buffer, in the debug case we copy it to another buffer for reading
        });

        // Get the raw array buffer for the mapped GPU buffer
        const parametersArrayBuffer = this.#boundaryValueParameters.getMappedRange();

        new Float32Array(parametersArrayBuffer, 0, 1).set([phaseVelocity]);

        // Unmap the buffer returning ownership to the GPU.
        this.#boundaryValueParameters.unmap();

        this.#boundaryValueBindGroup = this.#device.createBindGroup({
            layout: this.#boundaryValueParametersLayout,
            entries: [{
              binding: 0,
              resource: {
                buffer: this.#boundaryValueParameters
              }
            }]
        });

        this.#boundaryValueRealPipeline = this.#device.createComputePipeline({
            label: "Real part update pipeline",
            layout: this.#device.createPipelineLayout({
                bindGroupLayouts: [
                  schrodinger.getParametersBindGroupLayout(),
                  schrodinger.getWaveFunctionBindGroupLayout(),
                  this.#boundaryValueParametersLayout
                ]
            }),
            compute: {
              module: this.#boundaryValueShaderModule,
              entryPoint: "recomputeRealBoundary"
            }
        });

      this.#boundaryValueImaginaryPipeline = this.#device.createComputePipeline({
        label: "Imaginary part update pipeline",
        layout: this.#device.createPipelineLayout({
          bindGroupLayouts: [
            schrodinger.getParametersBindGroupLayout(),
            schrodinger.getWaveFunctionBindGroupLayout(),
            this.#boundaryValueParametersLayout
          ]
        }),
        compute: {
          module: this.#boundaryValueShaderModule,
          entryPoint: "recomputeImaginaryBoundary"
        }
      });

        return this;
    }

  /**
   *
   * @param schrodinger
   * @param E
   * @param debug
   * @returns {MurBoundary}
   */
    static getInstance(schrodinger, E, debug)
    {
        const murBoundary = new MurBoundary(schrodinger, E, debug);
        return murBoundary.init();
    }

    /**
     * Append a compute pass to implement the boundary conditions.
     *
     * @param {GPUCommandEncoder} commandEncoder        The command encoder currently in use to collect GPU commands.
     * @param {GPUBindGroup}      waveFunctionBindGroup The bind group describing which wave function buffers are bound
     *                                                  to which indices.
     */
    makeRealComputePass(commandEncoder, waveFunctionBindGroup)
    {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.#boundaryValueRealPipeline);
        passEncoder.setBindGroup(0, this.#parametersBindGroup);
        passEncoder.setBindGroup(1, waveFunctionBindGroup);
        passEncoder.setBindGroup(2, this.#boundaryValueBindGroup);
        // We just need the one two thread workgroup, one for each edge.
        passEncoder.dispatchWorkgroups(1);
        passEncoder.end();
    }

  /**
   * Append a compute pass to implement the boundary conditions for the imaginary component of the wave function.
   *
   * @param {GPUCommandEncoder} commandEncoder The command encoder currently in use to collect GPU commands.
   * @param {GPUBindGroup} The bind group, describing which wave function buffers are bound to which indices,
   *                       is currently in use.
   */
  makeImaginaryComputePass(commandEncoder, waveFunctionBindGroup)
  {
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.#boundaryValueImaginaryPipeline);
    passEncoder.setBindGroup(0, this.#parametersBindGroup);
    passEncoder.setBindGroup(1, waveFunctionBindGroup);
    passEncoder.setBindGroup(2, this.#boundaryValueBindGroup);
    // We just need the one two thread workgroup, one for each edge.
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
  }
}

export {MurBoundary}