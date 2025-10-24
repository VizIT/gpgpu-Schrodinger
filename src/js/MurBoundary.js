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
class MurBoundary
{
    #schrodinger;
    #device;
    #boundaryValueShaderModule;
    #boundaryValueBindGroup;
    #boundaryValueParametersLayout;
    #boundaryValueParameters;
    #parametersBindGroup;
    #boundaryValuePipeline;
    #phaseVelocity;
    #debug;

    constructor(schrodinger, E, debug=false)
    {
        this.#schrodinger = schrodinger;
        this.#device = schrodinger.getDevice();
        this.#parametersBindGroup = schrodinger.getParametersBindGroup();
        // Phase velocity = w/k = K+V/Sqrt(2mK), w/V=0, m=1
        this.#phaseVelocity = Math.sqrt(0.5*E);
        this.#debug = debug;
    }

    init()
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
          
          // Group 1, changes on each iteration - the same as in the main solver to keep the same bindings.
          // Older wave function at t-Δt.
          @group(1) @binding(0) var<storage, read> oldWaveFunction : array<vec2f>;
          // Current wave function at t.
          @group(1) @binding(1) var<storage, read> waveFunction : array<vec2f>;
          // The updated wave function at t+Δt.
          @group(1) @binding(2) var<storage, read_write> updatedWaveFunction : array<vec2f>;
          
          // Group 2, boundary value specific data.
          @group(2) @binding(0) var<uniform> phaseVelocity : f32;
          
          @compute @workgroup_size(2)
          fn recomputeBoundary(@builtin(global_invocation_id) global_id : vec3u)
          {
            // Index will be 0 at the left edge, and 1 at the right edge.
            let index = global_id.x;
            let sotrageBufferIndex = index*(parameters.xResolution-1);
            let offset = 1 - 2*index;
            let dx = parameters.length / f32(parameters.xResolution-1);
            
            updatedWaveFunction[sotrageBufferIndex] = waveFunction[sotrageBufferIndex+offset]
                 + ((phaseVelocity*parameters.dt-dx)/(phaseVelocity*parameters.dt+dx))
                    *(updatedWaveFunction[sotrageBufferIndex+offset]-waveFunction[sotrageBufferIndex]);
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
            usage:  this.#debug ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            // How we use this buffer, in the debug case we copy it to another buffer for reading
        });

        // Get the raw array buffer for the mapped GPU buffer
        const parametersArrayBuffer = this.#boundaryValueParameters.getMappedRange();

        new Float32Array(parametersArrayBuffer, 0, 1).set([this.#phaseVelocity]);

        // Unmap the buffer returning ownership to the GPU.
        this.#boundaryValueParameters.unmap();

        this.#boundaryValueBindGroup = this.#device.createBindGroup({
            layout: this.#boundaryValueParametersLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.#boundaryValueParameters
                    }
                }
            ]
        });

        this.#boundaryValuePipeline = this.#device.createComputePipeline({
            layout: this.#device.createPipelineLayout({
                bindGroupLayouts: [
                                    this.#schrodinger.getParametersBindGroupLayout(),
                                    this.#schrodinger.getWavefunctionBindGroupLayout(),
                                    this.#boundaryValueParametersLayout
                                  ]
            }),
            compute: {
                module: this.#boundaryValueShaderModule,
                entryPoint: "recomputeBoundary"
            }
        });

        return this;
    }

    static getInstance(schrodinger, E, debug)
    {
        const murBoundary = new MurBoundary(schrodinger, E, debug);
        return murBoundary.init();
    }

    /**
     * Append a compute pass to implement the boundary conditions.
     *
     * @param {GPUCommandEncoder} commandEncoder The command encoder currently in use to collect GPU commands.
     * @param {GPUBindGroup} The bind group, describing which wave function buffers are bound to which indices,
     *                       is currently in use.
     */
    makeComputePass(commandEncoder, waveFunctionBindGroup)
    {
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.#boundaryValuePipeline);
        passEncoder.setBindGroup(0, this.#parametersBindGroup);
        passEncoder.setBindGroup(1, waveFunctionBindGroup);
        passEncoder.setBindGroup(2, this.#boundaryValueBindGroup);
        // We just need the one two thread workgroup, one for each edge.
        passEncoder.dispatchWorkgroups(1);
        passEncoder.end();
    }
}

export {MurBoundary}