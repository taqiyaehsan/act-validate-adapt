// pmv_cache.js
class PMVCache {
    constructor(cacheSize = 1000) {
        this.cache = new Map();
        this.maxSize = cacheSize;
    }

    _generateKey(temperature, humidity, airSpeed) {
        // Round to 1 decimal place to increase cache hits
        return `${temperature.toFixed(1)}_${humidity.toFixed(1)}_${airSpeed.toFixed(2)}`;
    }

    get(temperature, humidity, airSpeed) {
        return this.cache.get(this._generateKey(temperature, humidity, airSpeed));
    }

    set(temperature, humidity, airSpeed, value) {
        if (this.cache.size >= this.maxSize) {
            // Remove oldest entry
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(this._generateKey(temperature, humidity, airSpeed), value);
    }

    clear() {
        this.cache.clear();
    }
}

module.exports = PMVCache;