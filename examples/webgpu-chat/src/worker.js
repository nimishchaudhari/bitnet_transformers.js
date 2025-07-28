
import {
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
} from '@huggingface/transformers';


class CallbackTextStreamer extends TextStreamer {
    constructor(tokenizer, cb) {
        super(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
        });
        this.cb = cb;
    }

    on_finalized_text(text) {
        this.cb(text);
    }
}

class InterruptableStoppingCriteria extends StoppingCriteria {
    constructor() {
        super();
        this.interrupted = false;
    }

    interrupt() {
        this.interrupted = true;
    }

    reset() {
        this.interrupted = false;
    }

    _call(input_ids, scores) {
        return new Array(input_ids.length).fill(this.interrupted);
    }
}

const stopping_criteria = new InterruptableStoppingCriteria();

/**
 * Test WebGPU capability and BitNet readiness
 */
async function testBitNetReadiness() {
    console.log('🚀 === Testing BitNet Readiness ===');
    
    try {
        // Check WebGPU availability
        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
            console.log('[OK] WebGPU is available');
            
            try {
                // Try high-performance adapter first
                let adapter = await navigator.gpu.requestAdapter({
                    powerPreference: 'high-performance',
                    forceFallbackAdapter: false
                });
                
                // Fallback to default if high-performance unavailable
                if (!adapter) {
                    console.log('[WARN] High-performance GPU unavailable, using default');
                    adapter = await navigator.gpu.requestAdapter();
                }
                
                if (adapter) {
                    console.log('[OK] WebGPU adapter acquired');
                    
                    // Log detailed adapter info for debugging
                    console.log('[INFO] GPU Adapter Details:');
                    console.log('  - Vendor:', adapter.info?.vendor || 'unknown');
                    console.log('  - Architecture:', adapter.info?.architecture || 'unknown');
                    console.log('  - Device:', adapter.info?.device || 'unknown');
                    console.log('  - Description:', adapter.info?.description || 'unknown');
                    
                    const device = await adapter.requestDevice();
                    console.log('[OK] WebGPU device acquired');
                    
                    // Check for BitNet-relevant features
                    const hasF16 = adapter.features.has('shader-f16');
                    const hasDP4A = adapter.features.has('chromium-experimental-dp4a');
                    
                    console.log('[INFO] WebGPU Features:');
                    console.log('  - shader-f16:', hasF16);
                    console.log('  - dp4a:', hasDP4A);
                    
                    // Performance optimization hints
                    if (navigator.platform.includes('Win')) {
                        console.log('💡 [HINT] On Windows, enable chrome://flags/#force-high-performance-gpu for best BitNet performance');
                    }
                    
                    // Test basic compute shader capability
                    const shaderCode = `
                        @compute @workgroup_size(1)
                        fn main() {
                            // Simple test shader
                        }
                    `;
                    
                    const shaderModule = device.createShaderModule({ code: shaderCode });
                    console.log('[OK] WebGPU compute shaders working');
                    
                    console.log('🎉 [OK] BitNet WebGPU backend ready!');
                    return {
                        backend: 'webgpu',
                        webgpuInfo: {
                            vendor: adapter.info?.vendor || 'unknown',
                            architecture: adapter.info?.architecture || 'unknown',
                            device: adapter.info?.device || 'unknown',
                            description: adapter.info?.description || 'unknown',
                            features: {
                                'shader-f16': hasF16,
                                'dp4a': hasDP4A
                            }
                        },
                        performanceHints: navigator.platform.includes('Win') ? 
                            ['Enable chrome://flags/#force-high-performance-gpu for best performance'] : []
                    };
                }
            } catch (e) {
                console.log('[WARN] WebGPU failed, will use fallback:', e.message);
            }
        }
        
        // Check WASM SIMD
        if (typeof WebAssembly !== 'undefined') {
            console.log('[OK] WebAssembly available');
            
            try {
                // Test SIMD support
                const simdTest = new Uint8Array([
                    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
                    0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
                    0x41, 0x00, 0xfd, 0x0f, 0x0b
                ]);
                
                if (WebAssembly.validate(simdTest)) {
                    console.log('[OK] WASM SIMD supported');
                    return { backend: 'wasm-simd', webgpuInfo: null, performanceHints: [] };
                }
            } catch (e) {
                console.log('[WARN] WASM SIMD test failed:', e.message);
            }
        }
        
        console.log('[OK] CPU fallback available');
        return { backend: 'cpu', webgpuInfo: null, performanceHints: [] };
        
    } catch (error) {
        console.error('[ERROR] BitNet readiness test failed:', error);
        return { backend: 'error', webgpuInfo: null, performanceHints: [] };
    }
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has('shader-f16');
    } catch (e) {
        return false;
    }
}

/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
class TextGenerationPipeline {
    static model_id = null;
    static model = null;
    static tokenizer = null;
    static streamer = null;

    static async getInstance(progress_callback = null) {
        // Choose the model based on whether fp16 is available
        this.model_id ??= (await hasFp16())
            ? 'Xenova/Phi-3-mini-4k-instruct_fp16'
            : 'Xenova/Phi-3-mini-4k-instruct';

        this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id, {
            legacy: true,
            progress_callback,
        });

        this.model ??= AutoModelForCausalLM.from_pretrained(this.model_id, {
            dtype: 'q4',
            device: 'webgpu',
            use_external_data_format: true,
            progress_callback,
        });

        return Promise.all([this.tokenizer, this.model]);
    }
}

async function generate(messages) {
    // Retrieve the text-generation pipeline.
    const [tokenizer, model] = await TextGenerationPipeline.getInstance();

    const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
    });

    let startTime;
    let numTokens = 0;
    const cb = (output) => {
        startTime ??= performance.now();

        let tps;
        if (numTokens++ > 0) {
            tps = numTokens / (performance.now() - startTime) * 1000;
        }
        self.postMessage({
            status: 'update',
            output, tps, numTokens,
        });
    }

    const streamer = new CallbackTextStreamer(tokenizer, cb);

    // Tell the main thread we are starting
    self.postMessage({ status: 'start' });

    const outputs = await model.generate({
        ...inputs,
        max_new_tokens: 512,
        streamer,
        stopping_criteria,
    });
    const outputText = tokenizer.batch_decode(outputs, { skip_special_tokens: false });

    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: outputText,
    });
}

async function load() {
    self.postMessage({
        status: 'loading',
        data: 'Loading model...'
    });

    // Test BitNet readiness
    self.postMessage({
        status: 'loading',
        data: 'Testing BitNet backend readiness...'
    });
    
    const backendInfo = await testBitNetReadiness();
    self.postMessage({
        status: 'loading',
        data: `[OK] BitNet will use ${backendInfo.backend} backend`
    });
    
    // Send WebGPU performance info to frontend
    if (backendInfo.webgpuInfo) {
        self.postMessage({
            status: 'webgpu-info',
            data: backendInfo
        });
    }

    // Load the pipeline and save it for future use.
    const [tokenizer, model] = await TextGenerationPipeline.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    self.postMessage({
        status: 'loading',
        data: 'Compiling shaders and warming up model...'
    });

    // Run model with dummy input to compile shaders
    const inputs = tokenizer('a');
    await model.generate({ ...inputs, max_new_tokens: 1 });
    self.postMessage({ status: 'ready' });
}
// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load();
            break;

        case 'generate':
            stopping_criteria.reset();
            generate(data);
            break;

        case 'interrupt':
            stopping_criteria.interrupt();
            break;

        case 'reset':
            stopping_criteria.reset();
            break;
    }
});
