/**
 * Common utilities and functionality for getting started with WebGPU compute shaders.
 */
const POWER_PREFERENCE_NAME = "powerPreference";
const HIGH_PERFORMANCE_POWER_PREFERENCE = "high-performance";
const LOW_POWER_POWER_PREFERENCE = "low-power";
const FALLBACK_OPTION_NAME = "forceFallbackAdapter";
const TIMESTAMP_QUERY_FEATURE_NAME = "timestamp-query";
const TIMESTAMP_QUERY_TYPE = "timestamp";

/**
 * @typedef {Number} Integer
 */

class WebGPUCompute {

    #adapterOptions;
    #deviceDescriptor;
    #device;

    /**
     * Construct a WebGPU Compute object for the given options.
     *
     * @param {{Name: String, Value: String}|undefined} adapterOptions An object containing
     * {@link https://www.w3.org/TR/webgpu/#dictdef-gpurequestadapteroptions|performance and fallback options}. May be null.
     * @param {GPUDeviceDescriptor|undefined} deviceDescriptor A {@link https://www.w3.org/TR/webgpu/#gpudevicedescriptor|GPUDeviceDescriptor}
     * with required features and limits. May be null.
     * @constructor
     */
    constructor (adapterOptions, deviceDescriptor) {
        if (!navigator.gpu) {
            throw Error("WebGPU is not supported.");
        }
        this.#adapterOptions = adapterOptions;
        this.#deviceDescriptor = deviceDescriptor;
    }

    /**
     * Get the power preference option name.
     *
     * @returns {string} The name for the GPU power preference.
     */
    static get POWER_PREFERENCE_NAME() {
        return POWER_PREFERENCE_NAME;
    }

    /**
     * Get the high power preference.
     *
     * @returns {string} The high power preference option.
     */
    static get HIGH_PERFORMANCE_POWER_PREFERENCE() {
        return HIGH_PERFORMANCE_POWER_PREFERENCE;
    }

    /**
     * Get the low power preference.
     *
     * @returns {string} The low power preference option.
     */
    static get LOW_POWER_POWER_PREFERENCE() {
        return LOW_POWER_POWER_PREFERENCE;
    }

    /**
     * The force fallback options. Set to true to get the fallback adapter.
     *
     * @returns {string} The force fallback adapter option name.
     */
    static get FALLBACK_OPTION_NAME() {
        return FALLBACK_OPTION_NAME;
    }

    static get TIMESTAMP_QUERY_TYPE() {
        return TIMESTAMP_QUERY_TYPE;
    }

    /**
     * Get the adapter corresponding to the provided options. Or the default adapter
     * if no options are provided. Adapters can not generally be reused.
     * <a href = "https://developer.mozilla.org/en-US/docs/Web/API/GPUAdapter/requestDevice">Making
     * a requestDevice() call on a GPUAdapter where requestDevice() has already been called
     * generates a promise that fulfills with a device that is immediately lost</a>.
     *
     * @returns {Promise<GPUAdapter>} A promise that resolves to a GPUAdapter.
     */
    async getAdapter() {

        let adapter = await navigator.gpu.requestAdapter(this.#adapterOptions);
        if (!adapter) {
            if (this.#adapterOptions) {
                throw Error("Request WebGPU adapter failed with options: " + JSON.stringify(this.#adapterOptions));
            } else {
                throw Error("Request WebGPU adapter failed.");
            }
        }
        return adapter;
    }

    /**
     * Check whether the adaptor supports timestamp queries. Make this check before
     * setting up time stamp queries.
     *
     * @returns {Promise<boolean>} A promise that resolves to true if timestamp queries are supported,
     * false if not.
     */
    async hasTimestampQuery()
    {
        const adapter = await this.getAdapter();
        return adapter.features.has(TIMESTAMP_QUERY_FEATURE_NAME);
    }

    /**
     * Check a limit against a desired value. Return true if the limit is equal to or greater
     * than the desired value, or false if not.
     *
     * @param {String}  limitName   The name of the limit to be checked.
     * @param {Integer} targetValue The value to be checked against.
     * @return True if the target value is equal to or greater than the limit value,
     * false if not.
     */
    async checkLimit(limitName, targetValue)
    {
        const adapter = await this.getAdapter();
        const limitValue = adapter.limits[limitName];
        return limitValue >= targetValue;
    }

    /**
     * Get the limit associated with the given name.
     *
     * @param {String} limitName The name of the limit to be retrieved.
     * @returns {Promise<Integer>} A promise resolving to the limit value.
     */
    async getLimit(limitName)
    {
        const adapter = await this.getAdapter();
        return adaptor.limits[limitName];
    }

    /**
     * Get a device from the adapter using the given descriptor.
     *
     * @param {GPUDeviceDescriptor|undefined} deviceDescriptor
     * A {@link https://www.w3.org/TR/webgpu/#gpudevicedescriptor|GPUDeviceDescriptor} with required features
     * and limits. May be null.
     * @returns {Promise<GPUDevice>} A promise that resolves to a GPUDevice.
     */
    async getMyDevice(descriptor) {
        if (!this.#device) {
            const adapter = await this.getAdapter();
            this.#device = await adapter.requestDevice(descriptor);

            this.#device.addEventListener('uncapturederror', (event) => {
                console.error(`Uncaught WebGPU error from ${this.#device.label}: `, event.error);
            });
        }
        return this.#device;
    }

    /**
     * Get a device from the adapter using the provided descriptor.
     *
     * @returns {Promise<GPUDevice>} A promise that resolves to a GPUDevice.
     */
    async getDevice() {
        return this.getMyDevice(this.#deviceDescriptor)
    }

    /**
     * Get a device with timestamp queries enabled from the adapter. Any provided device features and limits are used,
     * with the addition of timestamp feature support.
     *
     * @returns {Promise<GPUDevice>} A promise that resolves to a GPUDevice.
     */
    async getTimestampDevice() {
        let descriptor;
        if (this.#deviceDescriptor) {
            descriptor = JSON.parse(JSON.stringify(this.#deviceDescriptor));

            if (descriptor.requiredFeatures)
            {
                if (!descriptor.requiredFeatures.includes(TIMESTAMP_QUERY_FEATURE_NAME))
                {
                    descriptor.requiredFeatures.push(TIMESTAMP_QUERY_FEATURE_NAME);
                }
            } else {
                descriptor.requiredFeatures = [TIMESTAMP_QUERY_FEATURE_NAME];
            }
        } else {
            descriptor = {
              requiredFeatures: [TIMESTAMP_QUERY_FEATURE_NAME]
            };
        }
        return this.getMyDevice(descriptor);
    }

    /**
     * Create a query set, for compute shaders usually timestamp queries.
     *
     * @param {String}  type  The type of query.
     * @param {Integer} count The number of queries.
     * @returns {GPUQuerySet} A query set containing queries of the given type.
     */
    createQuerySet(type, count) {
        return this.#device.createQuerySet({
            type: type,
            count: count,
        });
    }
}

export { WebGPUCompute }