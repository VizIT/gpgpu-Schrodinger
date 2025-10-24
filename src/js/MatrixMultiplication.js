import {WebGPUCompute} from "./WebGPUCompute.js";

/**
 * An example of the use of compute shaders for matrix multiplication.
 * Also provide a MathML representation of the multiplication.
 */
class MatrixMultiplier
{
    #bindGroupLayout;
    #computePipeline;
    #device;
    #shaderModule;

    /**
     * Construct a MatrixMultiplier class. This method is empty. The initialization is
     * delegated to the init method, which can be asynchronous.
     */
    constructor() {}

    /**
     * Initialize the MatrixMultiplier. Get a WebGPU device, the shader module, and bind group
     * layout.
     *
     * @returns {Promise<MatrixMultiplier>}
     */
    async init () {
        const webgpuCompute = new WebGPUCompute();
        this.#device = await webgpuCompute.getDevice();

        // ID, consumer and type for each buffer.
        this.#bindGroupLayout = this.#device.createBindGroupLayout({
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
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }
            ]
        });

        this.#shaderModule = this.#device.createShaderModule({
            label: 'matrix multiply shader',
            code: `
    struct Matrix {
      size: vec2u,         // The first two entries in the data are the matrix dimensions.
      elements: array<f32> // The last element of a struct can have an undefined size.
    }

    // Input, read only, matrices.
    @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
    @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
    // Output, writable, matrix.
    @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id : vec3u) {
      // Skip invocations when work groups exceed the actual problem size
      if (global_id.x >= firstMatrix.size.x || global_id.y >= secondMatrix.size.y) {
        return;
      }

      resultMatrix.size = vec2u(firstMatrix.size.x, secondMatrix.size.y);

      // Some constants in the loop
      let resultCell = vec2(global_id.x, global_id.y);
      let firstMatrixSizeY = firstMatrix.size.y;
      let secondMatrixSizeY = secondMatrix.size.y;
      let resultXOffset = resultCell.x * firstMatrixSizeY;
      var result = 0.0;
      for (var i = 0u; i < firstMatrixSizeY; i = i + 1u) {
        let a = i + resultXOffset;
        let b = resultCell.y + i * secondMatrixSizeY;
        result = result + firstMatrix.elements[a] * secondMatrix.elements[b];
      }

      let index = resultCell.y + resultCell.x * secondMatrixSizeY;
      resultMatrix.elements[index] = result;
    }
  `
        });

        this.#computePipeline = this.#device.createComputePipeline({
            layout: this.#device.createPipelineLayout({
                bindGroupLayouts: [this.#bindGroupLayout]
            }),
            compute: {
                module: this.#shaderModule,
                entryPoint: "main"
            }
        });

        return this;
    }

    /**
     * Convenience method to invoke the constructor and the init method to return an
     * object where you can invoke compute.
     *
     * @returns {Promise<MatrixMultiplier>}
     */
    static async getInstance(){
        const matrixMultiplier = new MatrixMultiplier();
        return await matrixMultiplier.init();
    }

    /**
     * Create a GPU buffer loaded with the size and elements of a matrix.
     *
     * @param {Array<Number>} matrixSize
     * @param {Array<Number>} matrixElements
     */
    makeMatrixBuffer(matrixSize, matrixElements)
    {
        // Allocate a mapped buffer for our data
        const matrixBuffer = this.#device.createBuffer({
            mappedAtCreation: true,
            size: matrixSize.length*Uint32Array.BYTES_PER_ELEMENT + matrixElements.length*Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE,
        });

        // Get the raw array buffer for the mapped gpu buffer.
        const matrixArrayBuffer = matrixBuffer.getMappedRange();

        // The first two elements of the buffer are the matrix rows and cols, a Uint32[2]
        new Uint32Array(matrixArrayBuffer, 0,matrixSize.length).set(matrixSize);
        // The remainder is the matrix elements
        new Float32Array(matrixArrayBuffer, matrixSize.length*Uint32Array.BYTES_PER_ELEMENT, matrixElements.length).set(matrixElements);

        // Unmap the buffer, making available for the GPU
        matrixBuffer.unmap();
        // Return this, we're going to need a reference to it.
        return matrixBuffer;
    }

    /**
     * Carry out a matrix multiplication on the gpu and return the results.
     *
     * @param {Array<Number>} firstMatrixSize      The rows, columns size of the first matrix.
     * @param {Array<Number>} firstMatrixElements  The row major elements of the first matrix.
     * @param {Array<Number>} secondMatrixSize     The rows, columns size of the second matrix.
     * @param {Array<Number>} secondMatrixElements The row major elements of the second matrix.
     * @returns {Promise<{size: Uint32Array, elements: Float32Array}>}
     */
    async compute(firstMatrixSize, firstMatrixElements, secondMatrixSize, secondMatrixElements) {


        const firstMatrixBuffer = this.makeMatrixBuffer(firstMatrixSize, firstMatrixElements);
        const secondMatrixBuffer = this.makeMatrixBuffer(secondMatrixSize, secondMatrixElements);

        // Result Matrix
        const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrixSize[0] * secondMatrixSize[1]);
        const resultMatrixBuffer = this.#device.createBuffer({
            size: resultMatrixBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });


        // And mapping from @binding to data
        const bindGroup = this.#device.createBindGroup({
            layout: this.#bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: firstMatrixBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: secondMatrixBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: resultMatrixBuffer
                    }
                }
            ]
        });

        const commandEncoder = this.#device.createCommandEncoder();

        // This encapsulates commands to a compute shader.
        const passEncoder = commandEncoder.beginComputePass();
        // Specify the compute shader and bind group layout.
        passEncoder.setPipeline(this.#computePipeline);
        // The @group part of the binding in the shader.
        passEncoder.setBindGroup(0, bindGroup);
        const workgroupCountX = Math.ceil(firstMatrixSize[0] / 8);
        const workgroupCountY = Math.ceil(secondMatrixSize[1] / 8);
        passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
        passEncoder.end();

        // Get a GPU buffer for reading, this time not mapped at creation.
        const gpuReadBuffer = this.#device.createBuffer({
            size: resultMatrixBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Copy the result buffer to a buffer that can be mapped and read.
        commandEncoder.copyBufferToBuffer(
            resultMatrixBuffer,    // source buffer
            0,                     // source offset
            gpuReadBuffer,         // destination buffer
            0,                     // destination offset
            resultMatrixBufferSize // number of bytes to copy
        );

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        this.#device.queue.submit([gpuCommands]);

        // Read buffer, waiting for the compute pipeline to finish.
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = gpuReadBuffer.getMappedRange();
        // Slice to create a local copy of the array, otherwise the data would be lot on unmap.
        const arrayBufferLocal = arrayBuffer.slice();
        // Unmap and destroy the buffers. We don't reuse them, so release them immediately.
        gpuReadBuffer.unmap();
        firstMatrixBuffer.destroy();
        secondMatrixBuffer.destroy();
        resultMatrixBuffer.destroy();
        gpuReadBuffer.destroy();
        return {size: new Uint32Array(arrayBufferLocal, 0, 2),
                elements: new Float32Array(arrayBufferLocal, 2*Uint32Array.BYTES_PER_ELEMENT, firstMatrixSize[0] * secondMatrixSize[1])};
    }
}

/**
 * Generate the MathML for firstMatrix x secondMatrix = resultMatrix
 */
class MatrixMultiplicationML
{
    /**
     * Construct a matrix MathML object for firstMatrix x secondMatrix = resultMatrix.
     *
     * @param {Array<Number>} firstMatrixSize      The rows, columns size of the first matrix.
     * @param {Array<Number>} firstMatrixElements  The row major elements of the first matrix.
     * @param {Array<Number>} secondMatrixSize     The rows, columns size of the second matrix.
     * @param {Array<Number>} secondMatrixElements The row major elements of the second matrix.
     * @param {Array<Number>} resultMatrixSize     The rows, columns size of the result matrix.
     * @param {Array<Number>} resultMatrixElements The row major elements of the result matrix.
     * @constructor
     */
    constructor(firstMatrixSize, firstMatrixElements, secondMatrixSize, secondMatrixElements, resultMatrixSize, resultMatrixElements)
    {
        this.firstMatrixSize = firstMatrixSize;
        this.firstMatrixElements = firstMatrixElements;
        this.secondMatrixSize = secondMatrixSize;
        this.secondMatrixElements = secondMatrixElements;
        this.resultMatrixSize = resultMatrixSize;
        this.resultMatrixElements = resultMatrixElements;
    }

    /**
     * Set the first matrix.
     *
     * @param {Array<Number>} firstMatrixSize     The rows, columns size of the matrix.
     * @param {Array<Number>} firstMatrixElements The row major elements of the first matrix.
     * @returns {MatrixMultiplicationML}          This object.
     */
    setFirstMatrix(firstMatrixSize, firstMatrixElements)
    {
        this.firstMatrixSize = firstMatrixSize;
        this.firstMatrixElements = firstMatrixElements;
        return this;
    }

    /**
     * Set the second matrix.
     *
     * @param secondMatrixSize           The rows, columns size of the matrix.
     * @param secondMatrixElements       The row major elements of the first matrix.
     * @returns {MatrixMultiplicationML} This object.
     */
    setSecondMatrix(secondMatrixSize, secondMatrixElements)
    {
        this.secondMatrixSize = secondMatrixSize;
        this.secondMatrixElements = secondMatrixElements;
        return this;
    }

    /**
     *
     * @param resultMatrixSize           The rows, columns size of the matrix.
     * @param resultMatrixElements       The row major elements of the matrix.
     * @returns {MatrixMultiplicationML} This object.
     */
    setResultMatrix(resultMatrixSize, resultMatrixElements)
    {
        this.resultMatrixSize = resultMatrixSize;
        this.resultMatrixElements = resultMatrixElements;
        return this;
    }

    /**
     * Create MathML for a matrix.
     *
     * @param matrixSize     The rows, columns size of the matrix.
     * @param matrixElements The row major elements of the matrix.
     * @returns {string}     A MathML representation of the matrix.
     */
    makeMatrixML(matrixSize, matrixElements)
    {
        let nRows = matrixSize[0];
        let nColumns = matrixSize[1];
        let mathMLMatrix = `
            <mrow>
                <mfenced open="(" close=")">
                    <mtable>`;
        for (let iRow = 0; iRow < nRows; iRow++)
        {
            mathMLMatrix += "<mtr>"
            for (let iColumn = 0; iColumn < nColumns; iColumn++)
            {
                mathMLMatrix += `<mtd><mn>${matrixElements[iRow * nColumns + iColumn]}</mn></mtd>`;
            }
            mathMLMatrix += "</mtr>\n";
        }
        mathMLMatrix += `
                </mtable>
            </mfenced>
        </mrow>`

        return mathMLMatrix;
    }

    /**
     * Generate a MathML representation of matrix multiplication.
     *
     * @returns {string} MathML representing the matrix multiplication.
     */
    toString()
    {
        let mathMLEquation = `<math display="block">\n`
        mathMLEquation += this.makeMatrixML(this.firstMatrixSize, this.firstMatrixElements);
        mathMLEquation += this.makeMatrixML(this.secondMatrixSize, this.secondMatrixElements);
        mathMLEquation += "<mo>=</mo>"
        mathMLEquation += this.makeMatrixML(this.resultMatrixSize, this.resultMatrixElements);
        mathMLEquation += '</math>';

        return mathMLEquation;
    }
}


export { MatrixMultiplier, MatrixMultiplicationML}
