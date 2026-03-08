/**
 * AudioWorklet processor for reliable PCM capture.
 * Runs on a dedicated audio thread — never drops buffers regardless of main-thread load.
 * Converts Float32 → Int16 PCM at the native sample rate.
 */
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = new Float32Array(0);
        // Target ~4096 samples per message (matches old ScriptProcessor chunk size)
        this._chunkSize = 4096;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input.length) return true;

        const channelData = input[0]; // mono
        if (!channelData || channelData.length === 0) return true;

        // Accumulate samples
        const newBuf = new Float32Array(this._buffer.length + channelData.length);
        newBuf.set(this._buffer);
        newBuf.set(channelData, this._buffer.length);
        this._buffer = newBuf;

        // Flush chunks when we have enough
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.slice(0, this._chunkSize);
            this._buffer = this._buffer.slice(this._chunkSize);

            // Compute RMS for VAD
            let sumSq = 0;
            for (let i = 0; i < chunk.length; i++) sumSq += chunk[i] * chunk[i];
            const rms = Math.sqrt(sumSq / chunk.length);

            // Convert Float32 → Int16
            const i16 = new Int16Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            this.port.postMessage({ pcm: i16.buffer, rms }, [i16.buffer]);
        }

        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
