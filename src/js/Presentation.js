import { WebGPUCompute } from "./WebGPUCompute.js";

/**
 * Presentation, a class to help out presenting status features and limits from WebGPU.
 */
class Presentation {

    #options;

    /**
     * Construct a WebGPU Presentation object for the given options.
     *
     * @param {{Name: String, Value: String}} options An object containing
     * {@link https://www.w3.org/TR/webgpu/#dictdef-gpurequestadapteroptions|performance and fallback options}.
     * @constructor
     */
    constructor(options) {
        this.#options = options;
    }

    /**
     * Adjust values using Ki, Mi, or Gi prefixes. This assumes that the values are an integer
     * number of k, m, or g.
     *
     * @param value The actual value.
     * @returns {string} The value as a string with appropriate units.
     */
    #normalizeValue(value) {
        let string;
        if (value > Prefix.giga) {
            string = (value / Prefix.giga).toFixed(0) + " GiB";
        } else if (value > Prefix.mega) {
            string = (value / Prefix.mega).toFixed(0) + " MiB";
        } else if (value > Prefix.kilo) {
            string = (value/Prefix.kilo).toFixed(0) + " KiB";
        } else {
            string = value.toString();
        }
        return string;
    }

    /**
     * Convert a value to a string, using Ki, Mi, or GiB when appropriate.
     *
     * @param value The initial value.
     * @param type The type of the value
     * @returns { string } The value converted to a string.
     */
    #stringify(value, type)
    {
        if (type === Units.bytes) {
            return this.#normalizeValue(value)
        } else {
            return value.toString();
        }
    }

    #makeLimitTableRow(limitName, localValue)
    {
      const row = document.createElement("tr");
      const fieldNameCell = document.createElement("td");
      const fieldName = document.createTextNode(limitName);
      fieldNameCell.append(fieldName);
      row.append(fieldNameCell);
      const currentValueCell = document.createElement("td");
      const defaultValueCell = document.createElement("td");
      let currentValue;
      let defaultValue;
      if (LimitFields[limitName]) {
        currentValue = this.#stringify(localValue, LimitFields[limitName][LimitFields.typeIndex]);
        defaultValue = this.#stringify(LimitFields[limitName][LimitFields.valueIndex], LimitFields[limitName][LimitFields.typeIndex]);
        defaultValueCell.append(document.createTextNode(defaultValue))
      } else {
        currentValue = this.#stringify(localValue);
        console.log("Unrecognized limit: " + limitName);
      }
      currentValueCell.append(document.createTextNode(currentValue));
      row.append(currentValueCell);
      row.append(defaultValueCell);

      return row;
    }

    /**
     * Get rows for a table describing the limits for this adapter.
     *
     * @param {String} tableBodyID Identifies the containing table.
     * @returns {Promise<HTMLElement>}
     */
    async getLimitsTableElements(tableBodyID) {
        const webgpuCompute = new WebGPUCompute(this.#options);
        const myAdapter = await webgpuCompute.getAdapter();
        const myLimits = myAdapter.limits;

        const tableBody = document.getElementById(tableBodyID)
        for (const limitName in myLimits) {
            tableBody.append(this.#makeLimitTableRow(limitName, myLimits[limitName]));
        }
        return tableBody;
    }

    /**
     * Get rows for a table describing the named limits for this adapter.
     *
     * @param {String} tableBodyID Identifies the containing table.
     * @param {...String} limitNames The names of the limits to be included in the table.
     * @returns {Promise<HTMLElement>}
     */
    async getSomeLimitsTableElements(tableBodyID, ...limitNames) {
        const webgpuCompute = new WebGPUCompute(this.#options);
        const myAdapter = await webgpuCompute.getAdapter();
        const myLimits = myAdapter.limits;

        const tableBody = document.getElementById(tableBodyID)
        for (const limitName of limitNames) {
            tableBody.append(this.#makeLimitTableRow(limitName, myLimits[limitName]));
        }
        return tableBody;
    }
}

/**
 * The type of each field.
 * Count such as the max number of vertex buffers.
 * Size such as the size of a workgroup.
 * Bytes such as the maz size of a vertex buffer.
 *
 * @type {{size: number, bytes: number, count: number}}
 */
const Units = {
    count: 1,
    size: 2,
    bytes: 3
}

const Prefix = {
    kilo : 1024,
    mega:  1024*1024,
    giga:  1024*1024*1024
}

const LimitFields = {
    'maxTextureDimension1D':                     [ Units.size ,      8192],
    'maxTextureDimension2D':                     [ Units.size ,      8192],
    'maxTextureDimension3D':                     [ Units.size ,      2048],
    'maxTextureArrayLayers':                     [ Units.size ,       256],
    'maxBindGroups':                             [ Units.count,         4],
    'maxBindGroupsPlusVertexBuffers':            [ Units.count,        24],
    'maxBindingsPerBindGroup':                   [ Units.count,      1000],
    'maxDynamicUniformBuffersPerPipelineLayout': [ Units.count,         8],
    'maxDynamicStorageBuffersPerPipelineLayout': [ Units.count,         4],
    'maxSampledTexturesPerShaderStage':          [ Units.count,        16],
    'maxSamplersPerShaderStage':                 [ Units.count,        16],
    'maxStorageBuffersPerShaderStage':           [ Units.count,         8],
    'maxStorageTexturesPerShaderStage':          [ Units.count,         4],
    'maxUniformBuffersPerShaderStage':           [ Units.count,        12],
    'maxUniformBufferBindingSize':               [ Units.bytes,     65536],
    'maxStorageBufferBindingSize':               [ Units.bytes, 134217728],
    'minUniformBufferOffsetAlignment':           [ Units.bytes,       256],
    'minStorageBufferOffsetAlignment':           [ Units.bytes,       256],
    'maxVertexBuffers':                          [ Units.count,         8],
    'maxBufferSize':                             [ Units.bytes, 268435456],
    'maxVertexAttributes':                       [ Units.count,        16],
    'maxVertexBufferArrayStride':                [ Units.bytes,      2048],
    'maxInterStageShaderComponents':             [ Units.count,        60],
    'maxInterStageShaderVariables':              [ Units.count,        16],
    'maxColorAttachments':                       [ Units.count,         8],
    'maxColorAttachmentBytesPerSample':          [ Units.bytes,        32],
    'maxComputeWorkgroupStorageSize':            [ Units.bytes,     16384],
    'maxComputeInvocationsPerWorkgroup':         [ Units.count,       256],
    'maxComputeWorkgroupSizeX':                  [ Units.size ,       256],
    'maxComputeWorkgroupSizeY':                  [ Units.size ,       256],
    'maxComputeWorkgroupSizeZ':                  [ Units.size ,        64],
    'maxComputeWorkgroupsPerDimension':          [ Units.bytes,     65535],
    typeIndex:                                   0,
    valueIndex:                                  1
}

export { Presentation, Units }