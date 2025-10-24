/**
 * Support for troubleshooting and diagnosing the Schrodinger FDTD integrator.
 */
class DumpSchrodinger
{
    #device;

    /**
     * Create an object to retrieve a storage buffer from the GPU, and perform some simple analysis.
     *
     * @param {GPUDevice} The GPUDevice in use by the simulation. This is necessary to access buffers
     * allocated from this device.
     * @constructor
     */
    constructor(device)
    {
        this.#device = device;
    }


    /**
     * Fetch current wave function values into a Float32AArray.
     *
     * @param {GPUBuffer} buffer Data buffer copied from the GPU.
     * @returns {Promise<Float32Array>}
     */
    async dumpWavefunction(buffer) {
        // Get a GPU buffer for reading.
        const gpuReadBuffer = this.#device.createBuffer({
            label: "wave function read buffer",
            size: buffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Encode commands for copying buffer to buffer.
        const copyEncoder = this.#device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
            buffer,        // source buffer
            0,             // source offset
            gpuReadBuffer, // destination buffer
            0,             // destination offset
            buffer.size    // Count of bytes to be copied (all of them)
        );

        // Submit copy command.
        const copyCommands = copyEncoder.finish({label: "GPU Compute Command Buffer"});
        this.#device.queue.submit([copyCommands]);

        // Read buffer.
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const copyArrayBuffer = gpuReadBuffer.getMappedRange();
        // Slice forces a copy of the data, so it is not lost when the gpu buffer is destroyed.
        const result = new Float32Array(copyArrayBuffer.slice());
        gpuReadBuffer.destroy();
        return result;
    }

    /**
     * Simpson's rule integration of the wave function. The expectation is that the value will be roughly 1,
     * and roughly constant over time. Theoretically, this is more accurate than the trapezoidal rule, but in
     * our case the difference was minor.
     *
     * @param {Float32Array} wavefunction An array containing the values of the wave function.
     * @param {Number} length The physical length of the simulation.
     * @returns {Number} The integral of the wave function over the length of the simulation.
     */
    simpson(wavefunction, length)
    {
        const nPoints = wavefunction.length/2;
        let sum = wavefunction[0]*wavefunction[0] + wavefunction[1]*wavefunction[1];


        for (let i=1; i<nPoints-2; i++) {
            const currentIndex = 2*i;
            sum += ((i%2+1)*2)*(wavefunction[currentIndex]*wavefunction[currentIndex]
                                + wavefunction[currentIndex+1]*wavefunction[currentIndex+1]);
        }
        const lastPointIndex = 2*(nPoints-1);
        sum += wavefunction[lastPointIndex]*wavefunction[lastPointIndex]
               + wavefunction[lastPointIndex+1]*wavefunction[lastPointIndex+1];

        return length/(3.0*(nPoints-1)) * sum;
    }

    /**
     * Simple trapezoidal integration of the wave function. The expectation is that the value will be roughly 1,
     * and roughly constant over time.
     *
     * @param {Float32Array} wavefunction An array containing the values of the wave function.
     * @param {Number}       length       The physical length of the simulation.
     * @returns {Number} The integral of the wave function over the length of the simulation.
     */
    integrateWaveFunction(wavefunction, length)
    {
        // Remember that each Ψ value contains both a real and imaginary part.
        const nIntervals = wavefunction.length/2.0 - 1.0;
        const dx = length / nIntervals;
        let sum = 0;
        for (let i=0; i<nIntervals; i++)
        {
            sum += ((wavefunction[2*i]*wavefunction[2*i] + wavefunction[2*i+1]*wavefunction[2*i+1]  // Ψᵢ
                     + wavefunction[2*i+2]*wavefunction[2*i+2] + wavefunction[2*i+3]*wavefunction[2*i+3]) // Ψᵢ₊₁
                /2.0) * dx;
        }
        return sum;
    }
}

export { DumpSchrodinger }
