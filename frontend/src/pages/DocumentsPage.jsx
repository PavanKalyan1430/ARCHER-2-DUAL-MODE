import { useState, useRef, useCallback, useEffect } from 'react';
import { useStore } from '../store';
import { api } from '../services/api';
import { UploadCloud, FileText, Database, HardDrive, CheckCircle2, Clock, Trash2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { motion, AnimatePresence } from 'framer-motion';
import { CardSpotlight } from '../components/ui/card-spotlight';

function cn(...inputs) { return twMerge(clsx(inputs)); }

export default function DocumentsPage() {
  const { documents, addDocument, removeDocument } = useStore();
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deletingDocId, setDeletingDocId] = useState(null);
  const fileInputRef = useRef();

  const handleFiles = useCallback(async (files) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const ext = file.name.split('.').pop().toLowerCase();
      if (ext !== 'pdf' && ext !== 'docx' && ext !== 'doc') {
        alert(`File "${file.name}" is not supported. Only PDF and Word (.docx) documents are allowed.`);
        continue;
      }
      try {
        const res = await api.uploadDocument(file);
        const doc = { doc_id: res.doc_id, filename: file.name, status: 'processing', chunk_count: null };
        addDocument(doc);
      } catch (e) {
        alert(`Upload failed for "${file.name}": ` + e.message);
      }
    }
    setUploading(false);
  }, [addDocument]);

  const handleDelete = async (e, docId) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this document from the vector store?')) return;
    
    setDeletingDocId(docId);
    try {
      await api.deleteDocument(docId);
      removeDocument(docId);
    } catch (err) {
      alert('Failed to delete: ' + err.message);
    } finally {
      setDeletingDocId(null);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="w-full mx-auto p-8 lg:p-10 max-w-[1600px]">
      <header className="mb-10">
        <h1 className="text-[28px] font-display font-semibold text-[#111827] dark:text-white mb-2 tracking-tight">Dataset Intelligence</h1>
        <p className="text-[15px] text-[#6B7280] dark:text-zinc-400 max-w-2xl leading-relaxed">
          Ingest unstructured PDF and Word documents into the semantic vector space for agentic retrieval.
        </p>
      </header>

      {/* Enterprise Upload Zone */}
      <motion.div
        whileHover={{ scale: 1.005 }}
        whileTap={{ scale: 0.995 }}
        onClick={() => !uploading && fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFiles(e.dataTransfer.files); }}
        className={cn(
          "relative mb-10 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer bg-white dark:bg-zinc-900/30 shadow-sm",
          isDragging ? "border-blue-500 bg-blue-50/50 dark:bg-blue-950/20 shadow-md" : "border-gray-300 dark:border-zinc-700 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-blue-50/30 dark:hover:bg-blue-950/10",
          uploading && "opacity-60 cursor-not-allowed pointer-events-none"
        )}
      >
        <input ref={fileInputRef} type="file" accept=".pdf,.docx,.doc" multiple className="hidden" onChange={(e) => handleFiles(e.target.files)} />
        
        <div className="px-8 py-16 flex flex-col items-center justify-center text-center">
          <div className={cn(
            "w-20 h-20 rounded-2xl flex items-center justify-center mb-6 transition-all duration-300",
            isDragging ? "bg-blue-100 text-blue-600 scale-110 shadow-sm" : "bg-gray-50 dark:bg-zinc-800/40 text-gray-400 dark:text-zinc-500 border border-gray-100 dark:border-zinc-800"
          )}>
            <UploadCloud className="w-10 h-10" />
          </div>
          <h3 className="text-[18px] font-semibold text-[#111827] dark:text-white mb-2">Click or drag documents to upload</h3>
          <p className="text-[14px] text-gray-500 dark:text-zinc-400 font-medium">Maximum file size 1GB. PDF & Word formats only.</p>
          
          {uploading && (
            <div className="mt-8 flex items-center gap-3 text-sm font-semibold text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/30 px-5 py-2.5 rounded-xl border border-blue-100 dark:border-blue-900/50 shadow-sm">
              <svg className="animate-spin h-5 w-5 text-blue-600 dark:text-blue-450" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing vector embeddings...
            </div>
          )}
        </div>
      </motion.div>

      {/* Registry */}
      <div className="border border-zinc-200 dark:border-zinc-800 rounded-2xl overflow-hidden shadow-sm bg-white dark:bg-[#18181b]">
        <div className="px-6 py-5 border-b border-gray-200 dark:border-zinc-800 bg-gray-50 dark:bg-zinc-900/30 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-white dark:bg-zinc-800 rounded-md border border-gray-200 dark:border-zinc-700 shadow-sm">
              <HardDrive className="w-4 h-4 text-gray-600 dark:text-zinc-400" />
            </div>
            <h2 className="text-[14px] font-semibold text-[#111827] dark:text-white">Indexed Documents</h2>
          </div>
          <span className="text-[12px] font-semibold text-gray-600 dark:text-zinc-400 bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 px-3 py-1 rounded-full shadow-sm">
            {documents.length} Total
          </span>
        </div>
        
        {documents.length === 0 ? (
          <div className="p-10 bg-white dark:bg-zinc-900 flex justify-center">
            <CardSpotlight className="h-96 w-[600px]">
              <p className="text-2xl font-bold relative z-20 mt-2 text-white">
                Initialization Sequence
              </p>
              <div className="text-neutral-300 mt-4 relative z-20 text-[15px]">
                Follow these steps to deploy your context vector database:
                <ul className="list-none mt-6 space-y-4">
                  <Step title="Upload an unstructured PDF document" />
                  <Step title="Wait for LlamaIndex sentence splitting chunking" />
                  <Step title="BGE-Large model generates embeddings" />
                  <Step title="Qdrant DB indexes the semantic vectors" />
                </ul>
              </div>
              <p className="text-neutral-400 mt-10 relative z-20 text-[13px]">
                System requires at least one bound document to activate the LangGraph routing agent in the Chat terminal.
              </p>
            </CardSpotlight>
          </div>
        ) : (
          <div className="flex flex-col bg-white dark:bg-zinc-900">
            <AnimatePresence>
              {documents.map((doc, idx) => (
                <DocumentRow 
                  key={doc.doc_id}
                  doc={doc}
                  idx={idx}
                  isLast={idx === documents.length - 1}
                  onDelete={handleDelete}
                  deletingDocId={deletingDocId}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function DocumentRow({ doc, idx, isLast, onDelete, deletingDocId }) {
  const progress = doc.status === 'ready' ? 100 : doc.status === 'failed' ? 0 : (doc.progress ?? 5);

  let subtext = "";
  if (doc.status === 'processing') {
    if (progress < 25) subtext = "Parsing and chunking PDF pages...";
    else if (progress < 70) subtext = "Generating BGE-Large vector embeddings...";
    else subtext = "Indexing semantic vectors into Qdrant store...";
  } else if (doc.status === 'ready') {
    subtext = `${doc.chunk_count || 0} chunks successfully indexed`;
  } else if (doc.status === 'failed') {
    subtext = "Vector ingestion failed. Please try again.";
  }

  return (
    <motion.div 
      initial={{ opacity: 0, height: 0 }} 
      animate={{ opacity: 1, height: 'auto' }} 
      exit={{ opacity: 0, height: 0 }}
      className={cn(
        "p-5 flex flex-col md:flex-row md:items-center justify-between hover:bg-[#F4F7FB] dark:hover:bg-zinc-800/10 transition-colors group gap-4",
        !isLast && "border-b border-gray-100 dark:border-zinc-800"
      )}
    >
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <div className="w-12 h-12 rounded-xl bg-white dark:bg-zinc-800 border border-gray-200 dark:border-zinc-700 shadow-sm flex items-center justify-center shrink-0 group-hover:border-blue-300 dark:group-hover:border-blue-500/50 group-hover:shadow-md transition-all">
          <FileText className="w-6 h-6 text-gray-400 dark:text-zinc-500 group-hover:text-blue-500 transition-colors" />
        </div>
        <div className="flex flex-col flex-1 min-w-0">
          <h3 className="text-[15px] font-semibold text-[#111827] dark:text-zinc-100 leading-tight mb-1 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors truncate">
            {doc.filename}
          </h3>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-gray-500 dark:text-zinc-400 font-medium">
            <span className="bg-gray-100 dark:bg-zinc-800 px-2 py-0.5 rounded text-gray-600 dark:text-zinc-400 font-mono">
              ID: {doc.doc_id.split('-')[0]}
            </span>
            {subtext && (
              <>
                <span className="w-1 h-1 rounded-full bg-gray-300 dark:bg-zinc-700" />
                <span className={cn(
                  doc.status === 'processing' && "text-blue-600 dark:text-blue-400 animate-pulse",
                  doc.status === 'ready' && "text-gray-500 dark:text-zinc-400",
                  doc.status === 'failed' && "text-red-500 dark:text-red-400"
                )}>
                  {subtext}
                </span>
              </>
            )}
          </div>

          {/* Precision Embedding Loader (0% -> 100%) */}
          {doc.status === 'processing' && (
            <div className="w-full mt-3 flex items-center gap-3">
              <div className="flex-1 bg-gray-150 dark:bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                <motion.div 
                  initial={{ width: '0%' }}
                  animate={{ width: `${progress}%` }}
                  transition={{ ease: "easeOut", duration: 0.2 }}
                  className="bg-gradient-to-r from-blue-500 via-indigo-500 to-violet-500 h-full rounded-full"
                />
              </div>
              <span className="text-[11px] font-mono font-bold text-blue-600 dark:text-blue-400 min-w-[32px] text-right">
                {Math.round(progress)}%
              </span>
            </div>
          )}

          {doc.status === 'ready' && progress === 100 && (
            <div className="w-full mt-3 flex items-center gap-3">
              <div className="flex-1 bg-gray-150 dark:bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                <div 
                  className="bg-emerald-500 h-full rounded-full"
                  style={{ width: '100%' }}
                />
              </div>
              <span className="text-[11px] font-mono font-bold text-emerald-600 dark:text-emerald-400 min-w-[32px] text-right">
                100%
              </span>
            </div>
          )}
        </div>
      </div>
      
      <div className="flex items-center gap-4 self-end md:self-auto shrink-0">
        <StatusBadge status={doc.status} />
        <button 
          onClick={(e) => onDelete(e, doc.doc_id)}
          disabled={deletingDocId === doc.doc_id}
          className={cn(
            "p-2 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 dark:text-zinc-500 dark:hover:text-red-400 dark:hover:bg-red-950/20 transition-colors",
            deletingDocId === doc.doc_id && "opacity-50 cursor-not-allowed"
          )}
          title="Delete from Vector Store"
        >
          <Trash2 className={cn("w-4 h-4", deletingDocId === doc.doc_id && "animate-pulse")} />
        </button>
      </div>
    </motion.div>
  );
}

function StatusBadge({ status }) {
  if (status === 'ready') {
    return (
      <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-900/40 shadow-sm">
        <CheckCircle2 className="w-4 h-4" />
        <span className="text-[13px] font-semibold tracking-wide">Ready</span>
      </div>
    );
  }
  if (status === 'processing') {
    return (
      <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-blue-50 dark:bg-blue-950/20 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-900/40 shadow-sm">
        <Clock className="w-4 h-4 animate-spin text-blue-500 dark:text-blue-400" style={{ animationDuration: '3s' }} />
        <span className="text-[13px] font-semibold tracking-wide">Embedding</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-red-50 dark:bg-red-950/20 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-900/40 shadow-sm">
      <span className="w-2 h-2 rounded-full bg-red-500" />
      <span className="text-[13px] font-semibold tracking-wide">Failed</span>
    </div>
  );
}

const Step = ({ title }) => {
  return (
    <li className="flex gap-3 items-start">
      <CheckIcon />
      <p className="text-white font-medium">{title}</p>
    </li>
  );
};

const CheckIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="currentColor"
      className="h-5 w-5 text-blue-500 mt-0.5 shrink-0">
      <path stroke="none" d="M0 0h24v24H0z" fill="none" />
      <path
        d="M12 2c-.218 0 -.432 .002 -.642 .005l-.616 .017l-.299 .013l-.579 .034l-.553 .046c-4.785 .464 -6.732 2.411 -7.196 7.196l-.046 .553l-.034 .579c-.005 .098 -.01 .198 -.013 .299l-.017 .616l-.004 .318l-.001 .324c0 .218 .002 .432 .005 .642l.017 .616l.013 .299l.034 .579l.046 .553c.464 4.785 2.411 6.732 7.196 7.196l.553 .046l.579 .034c.098 .005 .198 .01 .299 .013l.616 .017l.642 .005l.642 -.005l.616 -.017l.299 -.013l.579 -.034l.553 -.046c4.785 -.464 6.732 -2.411 7.196 -7.196l.046 -.553l.034 -.579c.005 -.098 .01 -.198 .013 -.299l.017 -.616l.005 -.642l-.005 -.642l-.017 -.616l-.013 -.299l-.034 -.579l-.046 -.553c-.464 -4.785 -2.411 -6.732 -7.196 -7.196l-.553 -.046l-.579 -.034a28.058 28.058 0 0 0 -.299 -.013l-.616 -.017l-.318 -.004l-.324 -.001zm2.293 7.293a1 1 0 0 1 1.497 1.32l-.083 .094l-4 4a1 1 0 0 1 -1.32 .083l-.094 -.083l-2 -2a1 1 0 0 1 1.32 -1.497l.094 .083l1.293 1.292l3.293 -3.292z"
        fill="currentColor"
        strokeWidth="0" />
    </svg>
  );
};
