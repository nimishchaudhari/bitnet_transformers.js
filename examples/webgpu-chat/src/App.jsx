import { useEffect, useState, useRef } from 'react';

import Chat from './components/Chat';
import ArrowRightIcon from './components/icons/ArrowRightIcon';
import StopIcon from './components/icons/StopIcon';
import Progress from './components/Progress';

const IS_WEBGPU_AVAILABLE = !!navigator.gpu;
const STICKY_SCROLL_THRESHOLD = 120;

function App() {

  // Create a reference to the worker object.
  const worker = useRef(null);

  const textareaRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [progressItems, setProgressItems] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  // Inputs and outputs
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [tps, setTps] = useState(null);
  const [numTokens, setNumTokens] = useState(null);

  // BitNet WebGPU performance info
  const [webgpuInfo, setWebgpuInfo] = useState(null);
  const [bitnetBackend, setBitnetBackend] = useState(null);
  const [performanceHints, setPerformanceHints] = useState([]);

  function onEnter(message) {
    setMessages(prev => [
      ...prev,
      { "role": "user", "content": message },
    ]);
    setTps(null);
    setIsRunning(true);
    setInput('');
  }

  useEffect(() => {
    resizeInput();
  }, [input]);

  function onInterrupt() {
    // NOTE: We do not set isRunning to false here because the worker
    // will send a 'complete' message when it is done.
    worker.current.postMessage({ type: 'interrupt' });
  }

  function resizeInput() {
    if (!textareaRef.current) return;

    const target = textareaRef.current;
    target.style.height = 'auto';
    const newHeight = Math.min(Math.max(target.scrollHeight, 24), 200);
    target.style.height = `${newHeight}px`;
  }

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module'
      });
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case 'loading':
          // Model file start load: add a new progress item to the list.
          setStatus('loading');
          setLoadingMessage(e.data.data);
          break;

        case 'initiate':
          setProgressItems(prev => [...prev, e.data]);
          break;

        case 'progress':
          // Model file progress: update one of the progress items.
          setProgressItems(
            prev => prev.map(item => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data }
              }
              return item;
            })
          );
          break;

        case 'done':
          // Model file loaded: remove the progress item from the list.
          setProgressItems(
            prev => prev.filter(item => item.file !== e.data.file)
          );
          break;

        case 'ready':
          // Pipeline ready: the worker is ready to accept messages.
          setStatus('ready');
          break;

        case 'start': {
          // Start generation
          setMessages(prev => [...prev, { "role": "assistant", "content": "" }]);
        }
          break;

        case 'update': {
          // Generation update: update the output text.
          // Parse messages
          const { output, tps, numTokens } = e.data;
          setTps(tps);
          setNumTokens(numTokens)
          setMessages(prev => {
            const cloned = [...prev];
            const last = cloned.at(-1);
            cloned[cloned.length - 1] = { ...last, content: last.content + output };
            return cloned;
          });
        }
          break;

        case 'complete':
          // Generation complete: re-enable the "Generate" button
          setIsRunning(false);
          break;

        case 'webgpu-info':
          // WebGPU performance information from BitNet backend
          setWebgpuInfo(e.data.data.webgpuInfo);
          setBitnetBackend(e.data.data.backend);
          setPerformanceHints(e.data.data.performanceHints);
          break;
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener('message', onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener('message', onMessageReceived);
    };
  }, []);

  // Send the messages to the worker thread whenever the `messages` state changes.
  useEffect(() => {
    if (messages.filter(x => x.role === 'user').length === 0) {
      // No user messages yet: do nothing.
      return;
    }
    if (messages.at(-1).role === 'assistant') {
      // Do not update if the last message is from the assistant
      return;
    }
    setTps(null);
    worker.current.postMessage({ type: 'generate', data: messages });
  }, [messages, isRunning]);

  useEffect(() => {
    if (!chatContainerRef.current) return;
    if (isRunning) {
      const element = chatContainerRef.current;
      if (element.scrollHeight - element.scrollTop - element.clientHeight < STICKY_SCROLL_THRESHOLD) {
        element.scrollTop = element.scrollHeight;
      }
    }
  }, [messages, isRunning]);

  return (
    IS_WEBGPU_AVAILABLE
      ? (<div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900">

        {status === null && messages.length === 0 && (
          <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
            <div className="flex flex-col items-center mb-1 max-w-[280px] text-center">
              <img src="logo.png" width="100%" height="auto" className="block"></img>
              <h1 className="text-4xl font-bold mb-1">Phi-3 WebGPU</h1>
              <div className="flex items-center justify-center gap-2 mb-2">
                <h2 className="font-semibold">A private and powerful AI chatbot</h2>
                <span className="inline-flex items-center px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 text-xs font-medium">
                  🚀 BitNet Ready
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Enhanced with BitNet quantization support for faster inference
              </p>
            </div>

            <div className="flex flex-col items-center px-4">
              <p className="max-w-[514px] mb-4">
                <br />
                You are about to load <a href="https://huggingface.co/Xenova/Phi-3-mini-4k-instruct" target="_blank" rel="noreferrer" className="font-medium underline">Phi-3-mini-4k-instruct</a>,
                a 3.82 billion parameter LLM optimized for web inference. This demo includes <strong>BitNet quantization support</strong> for enhanced performance on compatible models.<br />
                <br />
                Everything runs directly in your browser using <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" className="underline">🤗&nbsp;Transformers.js</a> with WebGPU acceleration. Your conversations aren't sent to a server, and you can disconnect from the internet after loading!<br />
                <br />
                <div className="text-sm bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                  <strong className="text-blue-800 dark:text-blue-200">🚀 BitNet Features:</strong>
                  <ul className="mt-1 text-blue-700 dark:text-blue-300 text-left">
                    <li>• WebGPU performance optimization detection</li>
                    <li>• Multi-backend fallback (WebGPU → WASM → CPU)</li>
                    <li>• Real-time GPU adapter information</li>
                    <li>• 4x memory compression for compatible models</li>
                  </ul>
                </div>
              </p>

              <button
                className="border px-4 py-2 rounded-lg bg-blue-400 text-white hover:bg-blue-500 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
                onClick={() => {
                  worker.current.postMessage({ type: 'load' });
                  setStatus('loading');
                }}
                disabled={status !== null}
              >
                🚀 Load model with BitNet support
              </button>
            </div>
          </div>
        )}
        {status === 'loading' && (<>
          <div className="w-full max-w-[500px] text-left mx-auto p-4 bottom-0 mt-auto">
            <p className="text-center mb-1">{loadingMessage}</p>
            {progressItems.map(({ file, progress, total }, i) => (
              <Progress key={i} text={file} percentage={progress} total={total} />
            ))}
            
            {/* BitNet WebGPU Performance Info */}
            {webgpuInfo && (
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
                  🚀 BitNet WebGPU Backend ({bitnetBackend})
                </h4>
                <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                  <div><strong>GPU:</strong> {webgpuInfo.vendor} {webgpuInfo.device}</div>
                  <div><strong>Architecture:</strong> {webgpuInfo.architecture}</div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {webgpuInfo.features['shader-f16'] && (
                      <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 rounded text-xs">
                        ✓ shader-f16
                      </span>
                    )}
                    {webgpuInfo.features['dp4a'] && (
                      <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 rounded text-xs">
                        ✓ dp4a
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            {/* Performance Hints */}
            {performanceHints.length > 0 && (
              <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                <h4 className="font-semibold text-amber-800 dark:text-amber-200 mb-2">
                  💡 Performance Tips
                </h4>
                <ul className="text-sm text-amber-700 dark:text-amber-300 space-y-1">
                  {performanceHints.map((hint, i) => (
                    <li key={i}>• {hint}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </>)}

        {status === 'ready' && (<div
          ref={chatContainerRef}
          className="overflow-y-auto scrollbar-thin w-full flex flex-col items-center h-full"
        >
          <Chat messages={messages} />
          <div className="text-center space-y-2">
            <p className="text-sm min-h-6 text-gray-500 dark:text-gray-300">
              {tps && messages.length > 0 && (<>
                {!isRunning &&
                  <span>Generated {numTokens} tokens in {(numTokens / tps).toFixed(2)} seconds&nbsp;&#40;</span>}
                {<>
                  <span className="font-medium text-center mr-1 text-black dark:text-white">
                    {tps.toFixed(2)}
                  </span>
                  <span className="text-gray-500 dark:text-gray-300">tokens/second</span>
                </>}
                {!isRunning && <>
                  <span className="mr-1">&#41;.</span>
                  <span className="underline cursor-pointer" onClick={() => {
                    worker.current.postMessage({ type: 'reset' });
                    setMessages([]);
                  }}>Reset</span>
                </>}
              </>)}
            </p>
            
            {/* BitNet Backend Status */}
            {bitnetBackend && (
              <div className="text-xs text-gray-400 dark:text-gray-500">
                <span className="inline-flex items-center px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200">
                  🚀 BitNet: {bitnetBackend}
                  {webgpuInfo && webgpuInfo.features['shader-f16'] && <span className="ml-1">+ f16</span>}
                </span>
              </div>
            )}
          </div>
        </div>)}

        <div className="mt-2 border dark:bg-gray-700 rounded-lg w-[600px] max-w-[80%] max-h-[200px] mx-auto relative mb-3 flex">
          <textarea
            ref={textareaRef}
            className="scrollbar-thin w-[550px] dark:bg-gray-700 px-3 py-4 rounded-lg bg-transparent border-none outline-none text-gray-800 disabled:text-gray-400 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-400 disabled:placeholder-gray-200 resize-none disabled:cursor-not-allowed"
            placeholder="Type your message..."
            type="text"
            rows={1}
            value={input}
            disabled={status !== 'ready'}
            title={status === 'ready' ? "Model is ready" : "Model not loaded yet"}
            onKeyDown={(e) => {
              if (input.length > 0 && !isRunning && (e.key === "Enter" && !e.shiftKey)) {
                e.preventDefault(); // Prevent default behavior of Enter key
                onEnter(input);
              }
            }}
            onInput={(e) => setInput(e.target.value)}
          />
          {isRunning
            ? (<div className="cursor-pointer" onClick={onInterrupt}>
              <StopIcon
                className="h-8 w-8 p-1 rounded-md text-gray-800 dark:text-gray-100 absolute right-3 bottom-3"
              />
            </div>)
            : input.length > 0
              ? (<div className="cursor-pointer" onClick={() => onEnter(input)}>
                <ArrowRightIcon
                  className={`h-8 w-8 p-1 bg-gray-800 dark:bg-gray-100 text-white dark:text-black rounded-md absolute right-3 bottom-3`}
                />
              </div>)
              : (<div>
                <ArrowRightIcon
                  className={`h-8 w-8 p-1 bg-gray-200 dark:bg-gray-600 text-gray-50 dark:text-gray-800 rounded-md absolute right-3 bottom-3`}
                />
              </div>)
          }
        </div>

        <p className="text-xs text-gray-400 text-center mb-3">
          Disclaimer: Generated content may be inaccurate or false.
        </p>
      </div>)
      : (<div className="fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center">WebGPU is not supported<br />by this browser :&#40;</div>)
  )
}

export default App
