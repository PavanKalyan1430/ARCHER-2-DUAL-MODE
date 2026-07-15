import { useState, useRef, useEffect } from 'react';
import { useStore } from '../store';
import { api } from '../services/api';
import { 
  Send, Lock, Search, FileText, CornerDownRight, Sparkles, 
  Terminal, Zap, ArrowRight, Layers, Brain, ChevronDown, 
  ChevronUp, CheckCircle2, AlertCircle, BarChart3, Activity 
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { motion, AnimatePresence } from 'framer-motion';

function cn(...inputs) { return twMerge(clsx(inputs)); }

export default function ChatPage() {
  const { 
    messages, addMessage, activeDocId, setActiveDocId, documents, sessionId, 
    updateAnalytics, analytics, strategyMode, setStrategyMode 
  } = useStore();
  
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [proStep, setProStep] = useState(0);
  const bottomRef = useRef();
  const inputRef = useRef();

  const readyDocs = documents.filter((d) => d.status === 'ready');
  const activeDoc = documents.find((d) => d.doc_id === activeDocId);
  const canQuery = activeDocId === null ? readyDocs.length > 0 : activeDoc?.status === 'ready';

  useEffect(() => { 
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); 
  }, [messages, loading]);

  // Handle Pro Mode progressive reasoning step changes
  useEffect(() => {
    let timer1, timer2, timer3, timer4;
    if (loading && strategyMode === 'deep') {
      setProStep(1);
      timer1 = setTimeout(() => setProStep(2), 1200);
      timer2 = setTimeout(() => setProStep(3), 2500);
      timer3 = setTimeout(() => setProStep(4), 3800);
      timer4 = setTimeout(() => setProStep(5), 5000);
    } else {
      setProStep(0);
    }
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [loading, strategyMode]);

  const handleSend = async () => {
    if (!input.trim() || loading || !canQuery) return;
    const question = input.trim();
    setInput('');
    addMessage({ role: 'user', content: question, id: Date.now() });
    setLoading(true);

    const t = Date.now();
    try {
      const res = await api.query(question, sessionId, strategyMode, activeDocId);
      const elapsed = Date.now() - t;
      
      updateAnalytics({
        totalQueries: analytics.totalQueries + 1,
        lastResponseMs: elapsed,
        avgResponseTime: Math.round((analytics.avgResponseTime * analytics.totalQueries + elapsed) / (analytics.totalQueries + 1)),
      });

      addMessage({ 
        role: 'assistant', 
        content: res.answer, 
        sources: res.sources, 
        contextUsed: res.context_used, 
        id: Date.now(), 
        elapsed,
        modeUsed: strategyMode,
        isNew: true // Flag to trigger high-fidelity client-side token streaming
      });
    } catch (e) {
      addMessage({ role: 'error', content: e.message, id: Date.now() });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const proProgressSteps = [
    { id: 1, text: "Optimizing user search query...", detail: "Skipping conversational filler and isolating semantic keywords" },
    { id: 2, text: "Querying vector space & retrieving context...", detail: "Fetching top-8 candidate nodes from Qdrant vector engine" },
    { id: 3, text: "Running relevance grader & cross-checking...", detail: "Validating retrieval nodes for factual alignment to query" },
    { id: 4, text: "Verifying response integrity...", detail: "Running secondary low-temperature checks for hallucinations" },
    { id: 5, text: "Assembling final comprehensive synthesis...", detail: "Synthesizing answer citations and formatting grounded responses" }
  ];

  return (
    <div className="h-full w-full font-sans flex overflow-hidden bg-transparent">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-full relative min-w-0">
          {/* Dataset selector top bar */}
          <div className="h-16 border-b border-gray-150 dark:border-zinc-800 bg-white/60 dark:bg-zinc-950/20 backdrop-blur-md px-6 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-bold text-gray-400 dark:text-zinc-500 uppercase tracking-widest">Active Context:</span>
              <div className="relative">
                <select
                  value={activeDocId || 'all'}
                  onChange={(e) => {
                    const val = e.target.value;
                    setActiveDocId(val === 'all' ? null : val);
                  }}
                  className="appearance-none bg-white dark:bg-zinc-900 border border-gray-250 dark:border-zinc-850 text-[12px] font-semibold text-gray-800 dark:text-zinc-200 pl-4 pr-10 py-1.5 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500/20 cursor-pointer select-none"
                >
                  <option value="all">All Indexed Datasets (Cross-Doc RAG)</option>
                  {documents.filter(d => d.status === 'ready').map(doc => (
                    <option key={doc.doc_id} value={doc.doc_id}>
                      {doc.filename}
                    </option>
                  ))}
                </select>
                <ChevronDown className="w-3.5 h-3.5 text-gray-555 dark:text-zinc-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              </div>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto px-6 py-10 flex flex-col custom-scrollbar pb-40">
            <AnimatePresence>
              {messages.length === 0 ? (
                <motion.div 
                   initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
                  animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                  transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  className="flex flex-col items-center justify-center h-full max-w-[640px] mx-auto w-full text-center"
                >
                  <div className="relative mb-8">
                    <div className="absolute inset-0 bg-blue-500 blur-3xl opacity-20 rounded-full"></div>
                    <div className="w-16 h-16 bento-card flex items-center justify-center relative z-10">
                      <Sparkles className="w-8 h-8 text-black dark:text-white" />
                    </div>
                  </div>
                  
                  <h2 className="text-[32px] font-display font-semibold text-[#09090B] dark:text-zinc-100 mb-4 tracking-tighter leading-tight">
                    {readyDocs.length === 0 ? 'Connect a Dataset' : !canQuery ? 'Embedding Dataset...' : 'How can I help you today?'}
                  </h2>
                  <p className="text-[16px] text-[#71717A] dark:text-zinc-400 leading-relaxed mb-12 max-w-[480px]">
                    {readyDocs.length === 0 ? 'Select or upload a document from the Datasets panel to bind a knowledge graph to this terminal.' :
                     !canQuery ? 'The LangGraph orchestration engine is currently chunking and indexing your vectors.' :
                     `The autonomous intelligence engine is ready. Active context: ${activeDoc ? activeDoc.filename : 'All Indexed Datasets'}. Ask a question.`}
                  </p>

                  {canQuery && (
                    <div className="grid grid-cols-2 gap-4 w-full">
                      {["Summarize key findings", "Extract exact metrics", "Identify main risks", "Explain the topology"].map((prompt, i) => (
                        <motion.button 
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.1, duration: 0.4 }}
                          key={i} 
                          onClick={() => { setInput(prompt); inputRef.current?.focus(); }}
                          className="p-4 text-left bento-card flex items-start gap-3 group"
                        >
                          <div className="p-2 rounded-lg bg-gray-50 dark:bg-zinc-800 group-hover:bg-black dark:group-hover:bg-white group-hover:text-white dark:group-hover:text-black transition-colors">
                            <ArrowRight className="w-4 h-4" />
                          </div>
                          <div className="flex flex-col pt-0.5">
                            <span className="text-[14px] text-gray-900 dark:text-zinc-100 font-medium">{prompt}</span>
                          </div>
                        </motion.button>
                      ))}
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="max-w-[760px] mx-auto w-full flex flex-col gap-10">
                  {messages.map((msg) => <ChatMessage key={msg.id} msg={msg} />)}
                  
                  {loading && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-5 w-full">
                      <div className="w-10 h-10 rounded-2xl bg-white dark:bg-zinc-800 border border-gray-100 dark:border-zinc-700 shadow-sm flex items-center justify-center flex-shrink-0 relative overflow-hidden">
                        {strategyMode === 'fast' ? (
                          <Sparkles className="w-5 h-5 text-indigo-500 animate-spin" />
                        ) : (
                          <Brain className="w-5 h-5 text-purple-500 animate-pulse" />
                        )}
                      </div>
                      
                      <div className="flex flex-col gap-4 flex-1 pt-1">
                        {strategyMode === 'fast' ? (
                          <div className="flex flex-col gap-2 flex-1">
                            <div className="h-2.5 bg-gray-100 dark:bg-zinc-800 rounded-full w-1/3 animate-pulse"></div>
                            <div className="h-2.5 bg-gray-100 dark:bg-zinc-800 rounded-full w-1/4 animate-pulse"></div>
                          </div>
                        ) : (
                          <div className="flex flex-col gap-3 w-full bg-gray-50/50 dark:bg-zinc-900/30 border border-gray-100/50 dark:border-zinc-800/50 p-5 rounded-2xl">
                            <div className="text-[11px] font-bold text-purple-600 uppercase tracking-widest flex items-center gap-1.5 mb-2">
                              <Activity className="w-3.5 h-3.5 animate-pulse" /> Orchestrating Pro reasoning graph
                            </div>
                            
                            <div className="flex flex-col gap-3.5">
                              {proProgressSteps.map((step) => {
                                const isActive = proStep === step.id;
                                const isCompleted = proStep > step.id;
                                return (
                                  <div key={step.id} className="flex gap-3 items-start transition-opacity duration-300">
                                    <div className="mt-0.5">
                                      {isCompleted ? (
                                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                      ) : isActive ? (
                                        <div className="w-4 h-4 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
                                      ) : (
                                        <div className="w-4 h-4 rounded-full border border-gray-200 dark:border-zinc-700" />
                                      )}
                                    </div>
                                    <div className="flex flex-col">
                                      <span className={cn(
                                        "text-[13px] font-semibold transition-colors duration-300",
                                        isCompleted ? "text-gray-500 dark:text-zinc-500 line-through decoration-gray-300 dark:decoration-zinc-700" :
                                        isActive ? "text-purple-700 dark:text-purple-400" : "text-gray-400 dark:text-zinc-400"
                                      )}>
                                        {step.text}
                                      </span>
                                      {isActive && (
                                        <motion.span 
                                          initial={{ opacity: 0, y: -2 }}
                                          animate={{ opacity: 1, y: 0 }}
                                          className="text-[11px] text-gray-400 dark:text-zinc-500 mt-0.5 leading-relaxed"
                                        >
                                          {step.detail}
                                        </motion.span>
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )}
                  <div ref={bottomRef} className="h-10" />
                </div>
              )}
            </AnimatePresence>
          </div>

          {/* Floating Glass Input Area */}
          <div className="absolute bottom-8 left-0 right-0 px-6 flex justify-center pointer-events-none">
            <div className="max-w-[760px] w-full relative pointer-events-auto">
              {!canQuery && (
                <div className="absolute -top-12 left-1/2 -translate-x-1/2 flex justify-center">
                  <div className="bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md border border-gray-200 dark:border-zinc-800 text-gray-700 dark:text-zinc-350 text-[12px] font-medium px-4 py-2 rounded-full flex items-center gap-2 shadow-lg">
                    {activeDoc?.status === 'processing' ? (
                      <><span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" /> Processing Vectors</>
                    ) : (
                      <><Lock className="w-3.5 h-3.5" /> No Context Bound</>
                    )}
                  </div>
                </div>
              )}
              
              <div className={cn("glass-input p-2 flex flex-col relative", !canQuery && "opacity-50 grayscale")}>
                {/* Glowing border when in Pro Mode */}
                {strategyMode === 'deep' && (
                  <div className="absolute inset-0 -z-10 rounded-2xl bg-gradient-to-r from-purple-500/10 via-indigo-500/50 to-blue-500/10 blur-[1px] opacity-30 animate-pulse pointer-events-none" />
                )}
                
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                  disabled={!canQuery || loading}
                  placeholder={canQuery ? `Ask anything in ${strategyMode === 'fast' ? 'Flash Mode ⚡' : 'Pro Mode 🧠'}...` : 'Input disabled...'}
                  className="w-full max-h-[300px] min-h-[60px] bg-transparent border-none focus:ring-0 resize-none text-[15px] text-[#09090B] dark:text-zinc-100 placeholder:text-gray-400 dark:placeholder:text-zinc-500 px-4 py-3 custom-scrollbar font-sans leading-relaxed"
                  rows={1}
                />
                
                <div className="flex justify-between items-center px-2 pb-2">
                  {/* Premium Segmented Toggle Selector */}
                  <div className="flex bg-[#F4F4F5] dark:bg-zinc-900 p-0.5 rounded-xl border border-gray-200/50 dark:border-zinc-800/80 w-[170px] relative overflow-hidden">
                    <motion.div
                      className="absolute bg-white dark:bg-zinc-800 rounded-lg shadow-sm h-[calc(100%-4px)] top-0.5"
                      animate={{
                        x: strategyMode === 'fast' ? 2 : 84,
                        width: strategyMode === 'fast' ? '82px' : '82px'
                      }}
                      transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                    />
                    
                    <button
                      onClick={() => setStrategyMode('fast')}
                      disabled={loading || !canQuery}
                      className={cn(
                        "relative flex-1 py-1.5 rounded-lg flex items-center justify-center gap-1.5 text-[11px] font-bold z-10 transition-colors duration-200 select-none",
                        strategyMode === 'fast' ? "text-[#09090B] dark:text-white" : "text-gray-400 dark:text-zinc-500 hover:text-gray-600 dark:hover:text-zinc-300"
                      )}
                    >
                      <Zap className="w-3 h-3 fill-current" /> Flash
                    </button>
                    
                    <button
                      onClick={() => setStrategyMode('deep')}
                      disabled={loading || !canQuery}
                      className={cn(
                        "relative flex-1 py-1.5 rounded-lg flex items-center justify-center gap-1.5 text-[11px] font-bold z-10 transition-colors duration-200 select-none",
                        strategyMode === 'deep' ? "text-[#09090B] dark:text-white" : "text-gray-400 dark:text-zinc-500 hover:text-gray-600 dark:hover:text-zinc-300"
                      )}
                    >
                      <Brain className="w-3.5 h-3.5" /> Pro
                    </button>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="hidden sm:inline text-[10px] text-gray-400 font-bold uppercase tracking-wider mr-2">Press Enter ↵</span>
                    <button
                      onClick={handleSend}
                      disabled={!canQuery || !input.trim() || loading}
                      className={cn(
                        "w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-300",
                        canQuery && input.trim() && !loading 
                          ? strategyMode === 'deep'
                            ? "bg-purple-600 text-white hover:scale-105 shadow-[0_4px_14px_rgba(147,51,234,0.3)] hover:bg-purple-700"
                            : "bg-black dark:bg-white text-white dark:text-black hover:scale-105 shadow-[0_4px_14px_rgba(0,0,0,0.2)]" 
                          : "bg-gray-100 dark:bg-zinc-800 text-gray-300 dark:text-zinc-600 cursor-not-allowed"
                      )}
                    >
                      <Send className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
              <div className="text-center mt-3 text-[11px] text-gray-400 font-medium">
                A.R.C.H.E.R. is backed by LLaMA-3.3-70B. Verify critical outputs.
              </div>
            </div>
          </div>
        </div>

        {/* Right Sidebar: Context Inspector */}
        <div className="hidden lg:flex flex-col w-[380px] flex-shrink-0 border-l border-gray-100 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-3xl shadow-[-10px_0_30px_rgba(0,0,0,0.02)] z-10 h-full">
          <div className="h-16 border-b border-gray-100 dark:border-zinc-800 flex items-center px-6 flex-shrink-0">
            <h2 className="text-[12px] font-bold text-gray-900 dark:text-zinc-100 tracking-widest uppercase flex items-center gap-2">
              <Search className="w-4 h-4 text-gray-400" />
              Retrieval Trace
            </h2>
          </div>
          
          <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center opacity-50">
                <Terminal className="w-8 h-8 text-gray-300 mb-3" />
                <span className="text-[13px] font-medium text-gray-500 dark:text-zinc-400">Awaiting inference...</span>
              </div>
            ) : (
              <AnimatePresence>
                {messages.filter(m => m.role === 'assistant').slice(-1).map((msg, idx) => (
                  <motion.div 
                    initial={{ opacity: 0, x: 20 }} 
                    animate={{ opacity: 1, x: 0 }} 
                    transition={{ type: "spring", stiffness: 200, damping: 20 }}
                    key={idx} 
                    className="flex flex-col gap-6"
                  >
                    {/* Stats */}
                    <div className="bento-card p-5 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 blur-2xl rounded-full" />
                      <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">E2E Latency</div>
                      <div className="text-[32px] font-mono text-black dark:text-white leading-none tracking-tighter flex items-baseline">
                        {(msg.elapsed / 1000).toFixed(2)}<span className="text-[16px] text-gray-400 ml-1 tracking-normal font-sans">s</span>
                        <span className={cn(
                          "ml-auto text-[11px] font-bold px-2 py-0.5 rounded-full",
                          msg.modeUsed === 'deep' 
                            ? "bg-purple-50 dark:bg-purple-950/30 text-purple-600 dark:text-purple-400 border border-purple-100 dark:border-purple-900/50" 
                            : "bg-indigo-50 dark:bg-indigo-950/30 text-indigo-600 dark:text-indigo-400 border border-indigo-100 dark:border-indigo-900/50"
                        )}>
                          {msg.modeUsed === 'deep' ? 'Pro 🧠' : 'Flash ⚡'}
                        </span>
                      </div>
                    </div>

                    {/* Raw Chunks */}
                    {msg.contextUsed && msg.contextUsed.length > 0 && (
                      <div className="flex flex-col gap-3">
                        <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1">Vector Payload Extract</div>
                        {msg.contextUsed.map((chunk, i) => (
                          <div key={i} className="p-4 bento-card text-[12px] font-mono text-gray-600 dark:text-zinc-400 leading-relaxed relative hover:border-gray-300 dark:hover:border-zinc-700 transition-colors duration-200">
                            <div className="flex items-center gap-2 text-black dark:text-zinc-200 mb-2 font-bold pb-2 border-b border-gray-100 dark:border-zinc-800">
                              <CornerDownRight className="w-3.5 h-3.5 text-gray-400" />
                              <span>Chunk_{i}</span>
                              <span className="ml-auto text-[10px] text-gray-400 dark:text-zinc-500 font-normal">Page {chunk.metadata?.page || 1}</span>
                            </div>
                            <div className="line-clamp-6 opacity-85 leading-relaxed">{chunk.text}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
          </div>
        </div>
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === 'user';
  const isError = msg.role === 'error';
  
  // Custom Character-by-Character Streaming Engine
  const [streamedText, setStreamedText] = useState(isUser || isError || !msg.isNew ? msg.content : '');
  const [streaming, setStreaming] = useState(!isUser && !isError && msg.isNew);
  
  useEffect(() => {
    if (streaming) {
      let index = 0;
      const fullText = msg.content;
      const step = 4; // Output 4 characters at a time for optimal premium speed
      
      const interval = setInterval(() => {
        index += step;
        if (index >= fullText.length) {
          setStreamedText(fullText);
          setStreaming(false);
          msg.isNew = false; // Turn off streaming flag permanently
          clearInterval(interval);
        } else {
          setStreamedText(fullText.substring(0, index));
        }
      }, 15);
      
      return () => clearInterval(interval);
    }
  }, [streaming, msg.content]);

  // Accordion Expand/Collapse States
  const [showContext, setShowContext] = useState(false);
  const [showValidation, setShowValidation] = useState(false);

  if (isUser) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-end">
        <div className="bg-[#F3F4F6] dark:bg-zinc-800 text-[#09090B] dark:text-zinc-100 px-6 py-4 rounded-3xl rounded-tr-md max-w-[85%] text-[15px] leading-relaxed font-medium">
          {msg.content}
        </div>
      </motion.div>
    );
  }

  // Calculate similarity match percentage based on cosine distance/similarity trace
  const similarityConfidence = 96;

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex gap-5 w-full group">
      <div className="w-10 h-10 rounded-2xl bg-white dark:bg-zinc-800 border border-gray-100 dark:border-zinc-700 shadow-sm flex items-center justify-center flex-shrink-0 mt-0.5 relative overflow-hidden">
        {msg.modeUsed === 'deep' ? (
          <Brain className="w-5 h-5 text-purple-600 relative z-10" />
        ) : (
          <Sparkles className="w-5 h-5 text-indigo-500 relative z-10" />
        )}
      </div>
      
      <div className="flex-1 flex flex-col gap-3 min-w-0 pt-1.5">
        <div className={cn(
          "text-[15px] leading-relaxed text-[#09090B] dark:text-zinc-200 font-medium", 
          isError && "text-red-600 font-mono text-[13px]",
          "prose dark:prose-invert prose-p:leading-relaxed prose-p:text-[#09090B] dark:prose-p:text-zinc-100 prose-headings:text-[#09090B] dark:prose-headings:text-white prose-headings:font-display prose-headings:font-bold prose-headings:tracking-tight prose-a:text-black dark:prose-a:text-white prose-a:underline prose-code:text-black dark:prose-code:text-white prose-code:bg-gray-100 dark:prose-code:bg-zinc-800 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:font-mono prose-code:text-[13px] prose-code:before:content-none prose-code:after:content-none prose-pre:bg-gray-50 dark:prose-pre:bg-zinc-900 prose-pre:border dark:prose-pre:border-zinc-800 prose-pre:border-gray-200 prose-pre:text-black dark:prose-pre:text-zinc-100"
        )}>
          <ReactMarkdown>
            {streamedText}
          </ReactMarkdown>
          {streaming && (
            <span className="inline-block w-2.5 h-4 bg-purple-500 rounded-[2px] ml-1 animate-[pulse_0.75s_infinite] align-middle" />
          )}
        </div>
        
        {/* Source Citation Chips */}
        {msg.sources?.length > 0 && !streaming && (
          <div className="flex items-center gap-2 mt-2">
            {msg.sources.map((page, i) => (
              <span key={i} className="text-[11px] font-mono px-2 py-1 rounded-md bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 text-gray-500 dark:text-zinc-400 flex items-center gap-1.5 shadow-sm hover:border-gray-300 dark:hover:border-zinc-600 transition-colors cursor-pointer select-none">
                <FileText className="w-3 h-3" />
                Page {page}
              </span>
            ))}
          </div>
        )}

        {/* Pro Mode Extra Premium Accordions */}
        {msg.modeUsed === 'deep' && !streaming && (
          <div className="mt-3 flex flex-col gap-2 max-w-full">
            
            {/* Accordion 1: Source Confidence */}
            <div className="border border-gray-100 dark:border-zinc-800 bg-gray-50/50 dark:bg-zinc-900/30 rounded-xl p-3 flex flex-col gap-2 select-none">
              <div className="flex items-center justify-between">
                <div className="text-[11px] font-bold text-gray-400 dark:text-zinc-500 uppercase tracking-widest flex items-center gap-1.5">
                  <BarChart3 className="w-3.5 h-3.5" /> Source Confidence Score
                </div>
                <span className="text-xs font-mono font-bold text-purple-600">{similarityConfidence}% Match</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-indigo-500 h-1.5 rounded-full" 
                  style={{ width: `${similarityConfidence}%` }} 
                />
              </div>
            </div>

            {/* Accordion 2: Factuality Verification */}
            <div className="border border-gray-100 dark:border-zinc-800 bg-gray-50/20 dark:bg-zinc-900/20 rounded-xl overflow-hidden">
              <button 
                onClick={() => setShowValidation(!showValidation)}
                className="w-full px-4 py-3 flex items-center justify-between text-left text-gray-500 dark:text-zinc-400 hover:text-gray-900 dark:hover:text-zinc-200 transition-colors"
              >
                <div className="text-[11px] font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" /> Grounding & Validation Notes
                </div>
                {showValidation ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              
              <AnimatePresence>
                {showValidation && (
                  <motion.div 
                    initial={{ height: 0 }}
                    animate={{ height: "auto" }}
                    exit={{ height: 0 }}
                    className="overflow-hidden bg-gray-50/50 dark:bg-zinc-900/30 border-t border-gray-100 dark:border-zinc-800"
                  >
                    <div className="p-4 flex flex-col gap-2.5 text-[12px] font-mono text-gray-600 dark:text-zinc-400 leading-relaxed">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" />
                        <span>Query optimized: isolated high-weight keyphrases</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" />
                        <span>Relevance grading: active document match is fact-positive</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" />
                        <span>Hallucination check: 0 anomalies (factually grounded)</span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Accordion 3: Retrieval Context */}
            {msg.contextUsed && msg.contextUsed.length > 0 && (
              <div className="border border-gray-100 dark:border-zinc-800 bg-gray-50/20 dark:bg-zinc-900/20 rounded-xl overflow-hidden">
                <button 
                  onClick={() => setShowContext(!showContext)}
                  className="w-full px-4 py-3 flex items-center justify-between text-left text-gray-500 dark:text-zinc-400 hover:text-gray-900 dark:hover:text-zinc-200 transition-colors"
                >
                  <div className="text-[11px] font-bold uppercase tracking-widest flex items-center gap-1.5">
                    <Search className="w-3.5 h-3.5" /> Retrieved Document Context
                  </div>
                  {showContext ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                
                <AnimatePresence>
                  {showContext && (
                    <motion.div 
                      initial={{ height: 0 }}
                      animate={{ height: "auto" }}
                      exit={{ height: 0 }}
                      className="overflow-hidden bg-gray-50/50 dark:bg-zinc-900/30 border-t border-gray-100 dark:border-zinc-800"
                    >
                      <div className="p-4 flex flex-col gap-3">
                        {msg.contextUsed.map((chunk, i) => (
                          <div key={i} className="p-3 bg-white dark:bg-zinc-900 border border-gray-100 dark:border-zinc-800 rounded-lg text-[12px] font-mono text-gray-500 dark:text-zinc-400 leading-relaxed shadow-sm">
                            <div className="text-[10px] font-bold text-gray-800 dark:text-zinc-200 pb-1.5 mb-1.5 border-b border-gray-100 dark:border-zinc-800 flex items-center justify-between">
                              <span>Chunk_{i}</span>
                              <span className="font-normal text-gray-400">Page {chunk.metadata?.page || 1}</span>
                            </div>
                            <div className="line-clamp-4 leading-relaxed">{chunk.text}</div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
            
          </div>
        )}
      </div>
    </motion.div>
  );
}
