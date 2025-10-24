import { WebGPUCompute } from "./WebGPUCompute.js";

class CopyBuffer {

    #device;

    /**
     * Construct a CopyBuffer object for the given options.
     *
     * @constructor
     */
    constructor() {}

    /**
     * Initialize the object, that is initialize the WebGPU api.
     *
     * @param {{Name: String, Value: String}} options An object containing
     * {@link https://www.w3.org/TR/webgpu/#dictdef-gpurequestadapteroptions|performance and fallback options}.
     * @returns {Promise<CopyBuffer>}
     */
    async init(options) {
        const webgpuCompute = new WebGPUCompute(options);
        this.#device = await webgpuCompute.getDevice();

        return this;
    }

    /**
     * Build a copy buffers WebGPU demo.
     *
     * @param {{Name: String, Value: String}} options An object containing
     * {@link https://www.w3.org/TR/webgpu/#dictdef-gpurequestadapteroptions|performance and fallback options}.
     * @returns {Promise<CopyBuffer>}
     */
    static async build(options){
        const copyBuffer = new CopyBuffer();
        return await copyBuffer.init(options);
    }

    /**
     * Route a Float32Array through the GPU.
     *
     * @param {Float32Array} dataset Data copied on the GPU.
     * @returns {Promise<Float32Array>}
     */
    async doCopy(dataset) {

        // Get a GPU buffer in a mapped state and an arrayBuffer for writing.
        const gpuWriteBuffer = this.#device.createBuffer({
            label: "my compute input buffer",
            mappedAtCreation: true,
            size: dataset.byteLength,
            usage: GPUBufferUsage.COPY_SRC
        });
        const arrayBuffer = gpuWriteBuffer.getMappedRange();

        // Write bytes to buffer.
        new Float32Array(arrayBuffer).set(dataset);

        // Unmap buffer so that it can be used later for copy.
        gpuWriteBuffer.unmap();

        // Get a GPU buffer for reading in an unmapped state.
        const gpuReadBuffer = this.#device.createBuffer({
            label: "my compute read buffer",
            size: dataset.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Encode commands for copying buffer to buffer.
        const copyEncoder = this.#device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
            gpuWriteBuffer,     // source buffer
            0,                  // source offset
            gpuReadBuffer,      // destination buffer
            0,                  // destination offset
            dataset.byteLength, // Count of bytes to be copied
        );

        // Submit copy command.
        const copyCommands = copyEncoder.finish({label: "GPU Compute Command Buffer"});
        this.#device.queue.submit([copyCommands]);

        // Read buffer.
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const copyArrayBuffer = gpuReadBuffer.getMappedRange();
        // Slice forces a copy of the data, so it is not lost when the gpu buffer is destroyed.
        const result = new Float32Array(copyArrayBuffer.slice());
        gpuWriteBuffer.destroy();
        gpuReadBuffer.destroy();
        return result;
    }
}

export { CopyBuffer }
