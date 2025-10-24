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
 * Given 1D arrays containing wave function and potential values, draw those
 * values onto a canvas with array indices along the x-axis and the array values
 * along the y-axis.
 */
class SchrodingerRenderer
{
    #schrodinger;
    #device;

    // The plot parameters and their offsets for updating them.
    #reColor;
    #reColorOffset;
    #imColor;
    #imColorOffset;
    #psiColor;
    #psiColorOffset;
    #psiMax;
    #psiMaxOffset;
    #vColor;
    #vColorOffset;
    #vMax;
    #vMaxOffset;
    #E;
    #EOffset;
    #yResolution;
    #yResolutionOffset;
    #width;
    #widthOffset;

    #parametersBindGroup;
    #parametersBindGroupLayout;
    #plotParametersBuffer;
    #plotParametersBindGroup;
    #plotParametersLayout;
    #vertexBuffer;
    #vertexBuffersDescriptor;
    #rendererShaderModule;
    #canvasID;
    #canvas;
    #presentationFormat;
    #webGPUContext;

    /**
     * Build a Schrödinger wave function visualization with the given parameters.
     * @param {Schrodinger} schrodinger  A Schrodinger FDTD instance from which we retrieve the device
     *                                   and other parameters.
     * @param {GPUDevice} device         The device in use for the simulation, to allow buffer reuse.
     * @param {String} canvasID          The HTML ID for the canvas we render to.
     * @param {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters,
     *                                           carried over from the Schrödinger solver.
     * @param {GPUBindGroupLayout} parametersBindGroupLayout The bind group layout for the simulation parameters
     *                                                       from the Schrödinger simulation.
     *
     * @param {Array<Number>} reColor  The r, g, b, a color for the real part of the wave function, 0, 0, 0 0,
     *                                 for no plot.
     * @param {Array<Number>} imColor  The r, g, b, a color for the imaginary part of the wave function, 0, 0, 0 0,
     *                                 for no plot.
     * @param {Array<Number>} psiColor The r, g, b, a color for the wave function, 0, 0, 0 0, for no plot.
     * @param {Number} psiMax          The max psi value on the plot, the y-axis scale for the wave function plots.
     * @param {Array<Number>} vColor   The color for the potential function, v.
     * @param {Number} vMax            The max value for the potential on the plot, the y-axis scale for the
     *                                 potential plot.
     * @param {Number} E               The energy of the particle or wave function.
     * @param {Number} yResolution     The number of pixels in the y direction.
     * @param {Number} width           Roughly the width for renderer lines.
     */
    constructor(schrodinger, canvasID,
                reColor, imColor, psiColor, psiMax, vColor, vMax,
                E, yResolution, width)
    {
        this.#schrodinger = schrodinger;
        this.#device = schrodinger.getDevice();
        this.#canvasID = canvasID;
        this.#parametersBindGroup = schrodinger.getParametersBindGroup();
        this.#parametersBindGroupLayout = schrodinger.getParametersBindGroupLayout();
        this.#reColor = reColor;
        this.#imColor = imColor;
        this.#psiColor = psiColor;
        this.#psiMax = psiMax;
        this.#vColor = vColor;
        this.#vMax = vMax;
        this.#E = E;
        this.#yResolution = yResolution;
        this.#width = width;
    }

    /**
     * Get the simulation parameters bind group.
     *
     * @returns {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters.
     */
    getParametersBindGroup()
    {
        return this.#parametersBindGroup;
    }

    /**
     * Set the bind group for the Schrödinger equation parameters, carried over from the solver.
     *
     * @param {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters.
     */
    setParametersBindGroup(parametersBindGroup)
    {
        this.#parametersBindGroup = parametersBindGroup;
        return this;
    }

    /**
     * Get the color for the real part of the wave function.
     *
     * @returns {Array<Number>} The r, g, b, a color for the real part of the wave function.
     */
    getReColor()
    {
        return this.#reColor;
    }

    /**
     * Set the color for the real part of the wave function.
     *
     * @param {Array<Number>} reColor The r, g, b, a color for the real part of the wave function.
     * @returns {SchrodingerRenderer}
     */
    setReColor(reColor)
    {
        this.#reColor = reColor;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#reColorOffset, new Float32Array(reColor), 0, 4);
        return this;
    }

    /**
     * Get the color for the imaginary part of the wave function.
     *
     * @returns {Array<Number>} The r, g, b, a color for the imaginary part of the wave function.
     */
    getImColor()
    {
        return this.#imColor;
    }

    /**
     * Set the color for the imaginary part of the wave function.
     *
     * @param imColor {Array<Number>} The r, g, b, a color for the imaginary part of the wave function.
     * @returns {SchrodingerRenderer}
     */
    setImColor(imColor)
    {
        this.#imColor = imColor;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#imColorOffset, new Float32Array(imColor), 0, 4);
        return this;
    }

    /**
     * Get the color for the Ψ<sup>*</sup>Ψ plot.
     *
     * @returns {Array<Number>} The r, g, b, a color for the square of the wave function.
     */
    getPsiColor()
    {
        return this.#psiColor;
    }

    /**
     * Set the color for the Ψ<sup>*</sup>Ψ plot.
     *
     * @param {Array<Number>} psiColor The r, g, b, a color for the square of the wave function.
     * @returns {SchrodingerRenderer}
     */
    setPsiColor(psiColor)
    {
        this.#psiColor = psiColor;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#psiColorOffset, new Float32Array(psiColor), 0, 4);
        return this;
    }

    /**
     * Get the max value for Ψ values on this plot. This is the scale for psi, rePsi, and imPsi plots.
     *
     * @returns {Number} The scale for psi, rePsi, and imPsi plots.
     */
    getPsiMax()
    {
        return this.#psiMax;
    }

    /**
     * Set the max value for Ψ values on this plot. This is the scale for psi, rePsi, and imPsi plots.
     *
     * @param {Number} psiMax The scale for psi, rePsi, and imPsi plots.
     * @returns {SchrodingerRenderer}
     */
    setPsiMax(psiMax)
    {
        this.#psiMax = psiMax;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#psiMaxOffset, new Float32Array([psiMax]), 0, 1);
        return this;
    }

    /**
     * Get the color for the potential function plot.
     *
     * @returns {Array<Number>} The r, g, b, a color for the potential plot.
     */
    getVColor()
    {
        return this.#vColor;
    }

    /**
     * Set the color for the potential function plot.
     *
     * @param {Array<Number>} vColor The r, g, b, a color for the potential plot.
     * @returns {SchrodingerRenderer}
     */
    setVColor(vColor)
    {
        this.#vColor = vColor;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#vColorOffset, new Float32Array(vColor), 0, 4);
        return this;
    }

    /**
     * Get the expected max for the potential energy.
     *
     * @returns {Number} The maximum for the potential energy.
     */
    get vMax()
    {
        return this.#vMax;
    }

    /**
     * Sets the scale for the potential energy plot. Should slightly exceed the max potential energy.
     *
     * @param {Number} vMax The maximum for the potential energy.
     * @returns {SchrodingerRenderer}
     */
    setVMax(vMax)
    {
        this.#vMax = vMax;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#vMaxOffset, new Float32Array([vMax]), 0, 1);
        return this;
    }

    /**
     * Get the energy of the represented particle.
     *
     * @returns {Number} The energy of the particle.
     */
    getE()
    {
        return this.#E;
    }

    /**
     * Set the energy of the represented particle. This is drawn as a horizontal vColor line.
     *
     * @param {Number} E The energy of the particle.
     * @returns {SchrodingerRenderer}
     */
    setE(E)
    {
        this.#E = E;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#EOffset, new Float32Array([E]), 0, 1);
        return this;
    }

    /**
     * Get the number of pixels in the y-direction.
     *
     * @returns {Integer} The number of pixels along the y-axis.
     */
    getYResolution()
    {
        return this.#yResolution;
    }

    /**
     * Set the number of pixels along the y-axis.
     *
     * @param {Integer} yResolution The number of pixels in the y-direction.
     * @returns {SchrodingerRenderer}
     */
    setYResolution(yResolution)
    {
        this.#yResolution = yResolution;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#yResolutionOffset, new Float32Array([yResolution]), 0, 1);
        return this;
    }

    /**
     * Get the width for plotted lines.
     *
     * @returns {Number} The pixel width for plotted lines.
     */
    getWidth()
    {
        return this.#width;
    }

    /**
     * Set the width for plotted lines.
     *
     * @param {Number} width The pixel width for plotted lines.
     * @returns {SchrodingerRenderer}
     */
    setWidth(width)
    {
        this.#width = width;
        this.#device.queue.writeBuffer(this.#plotParametersBuffer, this.#widthOffset, new Float32Array([width]), 0, 1);
        return this;
    }

    /**
     * Get the plot parameters buffer for debugging.
     *
     * @returns {GPUBuffer} The plot parameters buffer.
     */
    getPlotParametersBuffer()
    {
        return this.#plotParametersBuffer;
    }

    loadPlotParameters(mappedBuffer)
    {
        // Get the raw array buffer for the mapped GPU buffer
        const plotParametersArrayBuffer = mappedBuffer.getMappedRange();

        let bytesSoFar = 0;
        this.#reColorOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 4).set(this.#reColor);
        bytesSoFar += 4*Float32Array.BYTES_PER_ELEMENT;

        this.#imColorOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 4).set(this.#imColor);
        bytesSoFar += 4*Float32Array.BYTES_PER_ELEMENT;

        this.#psiColorOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 4).set(this.#psiColor);
        bytesSoFar += 4*Float32Array.BYTES_PER_ELEMENT;

        this.#vColorOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 4).set(this.#vColor);
        bytesSoFar += 4*Float32Array.BYTES_PER_ELEMENT;

        this.#psiMaxOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#psiMax]);
        bytesSoFar += Float32Array.BYTES_PER_ELEMENT;

        this.#vMaxOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#vMax]);
        bytesSoFar += Float32Array.BYTES_PER_ELEMENT;

        this.#EOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#E]);
        bytesSoFar += Float32Array.BYTES_PER_ELEMENT;

        this.#yResolutionOffset = bytesSoFar;
        new Uint32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#yResolution]);
        bytesSoFar += Uint32Array.BYTES_PER_ELEMENT;

        this.#widthOffset = bytesSoFar;
        new Float32Array(plotParametersArrayBuffer, bytesSoFar, 1).set([this.#width]);

        mappedBuffer.unmap();
    }

    /**
     * Async initialization of the object.
     *
     * @returns {SchrodingerRenderer}
     */
    init()
    {
        const rendererShader = `
        struct WaveFunctionParameters
         {
            dt: f32,              // The time step size
            xResolution: u32,     // The number of points along the x-axis, the number of elements in the array.
            length: f32,          // The full length for our simulation
            potential: array<f32> // The potential the particle moves through
        }
        
        struct PlotParameters
        {
            // Color for RePsi: 0.0, 0.0, 0.0, 0.0 for no plot.
            reColor: vec4f,
            // Color for ImPsi: 0, 0, 0, 0 for no plot.
            imColor: vec4f,
            // Psi*Psi color
            psiColor: vec4f,
            // Color for the potential: 0.0, 0.0, 0.0, 0.0 for no plot
            vColor: vec4f,
            // Y scale for the psi plot
            psiMax: f32,
            // Y scale for the V plot.
            vMax: f32,
            // The energy of the particle or wave function.
            E: f32,
            // Number of points along the y axis.
            yResolution: u32,
            // Roughly corresponds to the rendered line width
            width: f32
        }
        
        // group 0 and 1, things that never change within a simulation.
        // The parameters for the simulation
        @group(0) @binding(0) var<storage, read> waveFunctionParameters: WaveFunctionParameters;
        // Plotting parameters, line colors, width, etc.
        @group(1) @binding(0) var<storage, read> plotParameters : PlotParameters;
    
        // Group 1, the wave function at t, changes on each invocation.
        @group(2) @binding(0) var<storage, read> waveFunction : array<vec2f>;
        
        /**
         * Color pixels to represent the numerical values of a function on a grid. The function is
         * confined to values in [-scale, +scale]. We also adjust the upper and lower bounds of the
         * line as necessary to fill in discontinuities in the numerical values.
         *
         * @param {vec4}    color         The color for a line of the given function.
         * @param {float}   scale         Possible values range from -scale to +scale.
         * @param {float}   fragY         The y, vertical, fragment shader coordinate of this pixel
         *                                in the range [0, yResolution-1] top to bottom.
         * @param {Integer} yResolution   The number of vertical pixels in the plot.
         * @param {float}   width         Roughly the line width in pixels.
         * @param {float}   value         The value of the function at the current position.
         * @param {float}   previousValue The previous function value.
         */
         fn pixelColor(color: vec4f, scale: f32, fragY: f32, yResolution: u32,
                       width: f32, value: f32, previousValue: f32) -> vec4f
         {
            // The total height runs from -scale to +scale
            let scale2 = 2.0*scale;
            let adjustedPixel = -fragY + (f32(yResolution-1)/2.0);
            // The function value for this pixel.
            let pxValue = scale2*adjustedPixel/f32(yResolution);
            
            // Begin fading in the color at the function value
            // but adjust toward the previous value for continuity
            let lowerEdge = min(value, previousValue+scale2/f32(yResolution));

            // Begin fading out the color at the function value
            // but adjust toward the previous value for continuity
            let upperEdge = max(value, previousValue-scale2/f32(yResolution));

            
            return color*(smoothstep(lowerEdge - scale2*width/f32(yResolution),
                                     lowerEdge - scale2/f32(yResolution),
                                     pxValue)
                          -smoothstep(upperEdge + scale2/f32(yResolution),
                                      upperEdge + scale2*width/f32(yResolution),
                                      pxValue));
         }
         
         @vertex
         fn vs_main(@location(0) inPos: vec3<f32>) -> @builtin(position) vec4f
         {
            return vec4(inPos, 1.0);
         }
         
         @fragment
         fn fs_main(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32>
         {
            let psiMax2          = plotParameters.psiMax*plotParameters.psiMax;
            // Remember, frag position ranges from 0.5 to xResolition-0.5,
            // see https://www.w3.org/TR/webgpu/#rasterization
            let index            = i32(fragPos.x);
            let previousIndex    = max(0, index-1);
            let psi              = waveFunction[index];
            let psiPrevious      = waveFunction[previousIndex];
            let v                = waveFunctionParameters.potential[index];
            let vPrevious        = waveFunctionParameters.potential[previousIndex];
            
            var background       = pixelColor(plotParameters.psiColor, psiMax2, fragPos.y,  plotParameters.yResolution,
                                                plotParameters.width, psi.r*psi.r + psi.g*psi.g,
                                                psiPrevious.r*psiPrevious.r + psiPrevious.g*psiPrevious.g);
                                                
            var color            = pixelColor(plotParameters.reColor, psiMax2, fragPos.y, plotParameters.yResolution,
                                                plotParameters.width, psi.r, psiPrevious.r);
            background           = mix(background, color, color.a);
            
            color                = pixelColor(plotParameters.imColor, psiMax2, fragPos.y, plotParameters.yResolution,
                                                plotParameters.width, psi.g, psiPrevious.g);
            background           = mix(background, color, color.a);
            
            color                = pixelColor(plotParameters.vColor, plotParameters.vMax, fragPos.y, plotParameters.yResolution,
                                                plotParameters.width, v, vPrevious);
            background           = mix(background, color, color.a);

            color                = pixelColor(plotParameters.vColor, plotParameters.vMax, fragPos.y, plotParameters.yResolution,
                                                plotParameters.width, plotParameters.E, plotParameters.E);
            background           = mix(background, color, color.a);
            return background;
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
                shaderLocation: 0, // position
                offset: 0,
                format: 'float32x3'
            }]
        }];


        this.#plotParametersLayout = this.#device.createBindGroupLayout({
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

        this.#plotParametersBuffer = this.#device.createBuffer({
            label: 'Plot Parameters',
            mappedAtCreation: true,
            size: 4*Float32Array.BYTES_PER_ELEMENT  // reColor
                + 4*Float32Array.BYTES_PER_ELEMENT  // imColor
                + 4*Float32Array.BYTES_PER_ELEMENT  // psiColor
                + 4*Float32Array.BYTES_PER_ELEMENT  // vColor
                + Float32Array.BYTES_PER_ELEMENT    // psiMax
                + Float32Array.BYTES_PER_ELEMENT    // vMax
                + Float32Array.BYTES_PER_ELEMENT    // E
                + Uint32Array.BYTES_PER_ELEMENT     // yResolution
                + Float32Array.BYTES_PER_ELEMENT    // width
                + 3*Float32Array.BYTES_PER_ELEMENT, // Required padding
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.loadPlotParameters(this.#plotParametersBuffer);

        this.#plotParametersBindGroup = this.#device.createBindGroup({
            layout: this.#plotParametersLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.#plotParametersBuffer
                    }
                }
            ]});

        // Get a WebGPU context from the canvas and configure it
        this.#canvas = document.getElementById(this.#canvasID);
        this.#webGPUContext = this.#canvas.getContext('webgpu');
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
     * Create and return an instance of a Schrödinger wave function visualization with the given parameters.
     *
     * @param {GPUDevice} device       The device in use for the simulation, to allow buffer reuse.
     * @param {String} canvasID        The HTML ID for the canvas we render to.
     * @param {GPUBindGroup} parametersBindGroup The bind group for the Schrödinger equation parameters,
     *                                           carried over from the Schrödinger solver.
     * @param {GPUBindGroupLayout} parametersBindGroupLayout The bind group layout for the simulation parameters
     *                                                       from the Schrödinger simulation.
     *
     * @param {Array<Number>} reColor  The r, g, b, a color for the real part of the wave function, 0, 0, 0 0,
     *                                 for no plot.
     * @param {Array<Number>} imColor  The r, g, b, a color for the imaginary part of the wave function, 0, 0, 0 0,
     *                                 for no plot.
     * @param {Array<Number>} psiColor The r, g, b, a color for the wave function, 0, 0, 0 0, for no plot.
     * @param {Number} psiMax          The max psi value on the plot, the y-axis scale for the wave function plots.
     * @param {Array<Number>} vColor   The color for the potential function, v.
     * @param {Number} vMax            The max value for the potential on the plot, the y-axis scale for the
     *                                 potential plot.
     * @param {Number} E               The energy of the particle or wave function.
     * @param {Number} yResolution     The number of pixels in the y direction.
     * @param {Number} width           Roughly the width for renderer lines.
     */
    static async getInstance(device, canvasID, parametersBindGroup, parametersBindGroupLayout,
                             reColor, imColor, psiColor, psiMax, vColor, vMax,
                             E, yResolution, width)
    {
        const schrodingerRenderer = new SchrodingerRenderer(
                        device, canvasID, parametersBindGroup, parametersBindGroupLayout,
                        reColor, imColor, psiColor, psiMax, vColor, vMax, E,
                        yResolution, width);
        return await schrodingerRenderer.init();
    }

    /**
     * The html canvas element that is our rendering target.
     *
     * @returns {HTMLCanvasElement} The html canvas element that is our rendering target.
     */
    getCanvas() {
        return this.#canvas;
    }

    /**
     * Render a wave function buffer from the schrodinger simulation.
     *
     * @param {GPUBuffer } waveFunctionBuffer
     */
    async render(waveFunctionBuffer)
    {
        const bindGroupLayout2 = this.#device.createBindGroupLayout({
            label: "Wave function layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "read-only-storage"
                    }
                }
            ]
        });

        const bindGroup2 = this.#device.createBindGroup({
            layout: bindGroupLayout2,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: waveFunctionBuffer
                    }
                }
            ]});

        const pipelineLayout = this.#device.createPipelineLayout({
            bindGroupLayouts: [this.#parametersBindGroupLayout, // Simulation parameters
                                this.#plotParametersLayout,         // Plot parameters
                                bindGroupLayout2]               // The wave function values
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
                    format: this.#presentationFormat,
                    blend: {
                        color: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha'
                        }
                    }
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
        passEncoder.setBindGroup(2, bindGroup2);
        passEncoder.setVertexBuffer(0, this.#vertexBuffer);
        passEncoder.draw(4);
        passEncoder.end();

        const commandBuffer = commandEncoder.finish();
        this.#device.queue.submit([commandBuffer]);
    }
}

export {SchrodingerRenderer}