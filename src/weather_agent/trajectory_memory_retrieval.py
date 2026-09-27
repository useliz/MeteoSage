"""Optional local multilingual encoding. No provider, network, or policy decisions."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time

from ._serialization import canonical_json, identity, to_primitive
from .trajectory_memory import _bm25_advice_scores, _terms

PREPROCESSING = "minilm-window128-positive-bm25-equal-relative80-v2"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_FILES = frozenset({"config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"})


class LocalSemanticRetrieval:
    def __init__(self, model_path, revision, threshold, preprocessing=PREPROCESSING):
        if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not -1 <= threshold <= 1
                or preprocessing != PREPROCESSING):
            raise ValueError("Invalid local semantic retrieval configuration")
        self.path = Path(model_path)
        self.receipt = json.loads((self.path / "memory-model.json").read_text())
        if (not isinstance(self.receipt, dict) or self.receipt.get("model") != MODEL
                or self.receipt.get("revision") != revision):
            raise ValueError("Local Memory model revision does not match configuration")
        for name in ("torch", "transformers"):
            if importlib.util.find_spec(name) is None:
                raise ValueError("Semantic Memory requires the optional memory-semantic dependency group")
        files = self.receipt.get("files")
        if (not isinstance(files, list) or len(files) != len(MODEL_FILES)
                or any(not isinstance(item, dict) or not isinstance(item.get("name"), str)
                    or type(item.get("bytes")) is not int or item["bytes"] <= 0
                    or not isinstance(item.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in files)
                or {item["name"] for item in files} != MODEL_FILES):
            raise ValueError("Local Memory model receipt lacks complete file identities")
        self._verify_files()
        self.threshold = threshold
        self.metadata = {"backend": "semantic", "model": MODEL, "revision": revision,
            "preprocessing": preprocessing, "threshold": threshold, "relative_floor_ratio": 0.8,
            "positive_lexical_weight": 0.5,
            "files_identity": identity(self.receipt["files"])}
        self.model = self.tokenizer = None
        self.cache = {}

    def _verify_files(self):
        for item in self.receipt["files"]:
            path = self.path / item["name"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
                raise ValueError("Local Memory model file identity changed")

    def _load(self):
        if self.model is not None:
            return
        # Asset identity was checked at construction and remains immutable for this instance.
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True, trust_remote_code=False).eval().to("cpu")

    def _encode(self, texts, prefix):
        import torch
        self._load()
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        capacity = 128 - self.tokenizer.num_special_tokens_to_add() - len(prefix_ids)
        windows, spans, coverage = [], [], []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            start = len(windows)
            offset = 0
            while True:
                part = tokens[offset:offset+capacity]
                windows.append({"input_ids": self.tokenizer.build_inputs_with_special_tokens(prefix_ids + part)})
                if offset + capacity >= len(tokens):
                    break
                offset += capacity - 32
            spans.append((start, len(windows)))
            coverage.append({"input_tokens": len(tokens), "covered_tokens": len(tokens),
                "window_count": len(windows)-start})
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(windows), 16):
                batch = self.tokenizer.pad(windows[start:start+16], padding=True, return_tensors="pt")
                output = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"]
                pooled = output.masked_fill(~mask[..., None].bool(), 0).sum(1) / mask.sum(1)[:, None]
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        matrix = torch.cat(vectors)
        return [(matrix[a:b], c) for (a, b), c in zip(spans, coverage)]

    def score(self, query, advice):
        started = time.perf_counter()
        keys = [identity({"public": to_primitive(a), "encoder": self.metadata}) for a in advice]
        missing = {key: a for key, a in zip(keys, advice) if key not in self.cache}
        if missing:
            texts = [a.guidance + "\nApplicability: " + "\n".join(a.applicability) for a in missing.values()]
            self.cache.update(zip(missing, self._encode(texts, "")))
        query_vectors, query_coverage = self._encode([query], "")[0]
        scores = [float((query_vectors @ self.cache[key][0].T).max()) for key in keys]
        # Exact method terms complement multilingual similarity without another
        # encoder pass. Exceptions never become positive lexical matches.
        lexical = _bm25_advice_scores(_terms(query), tuple(
            replace(a, negative_conditions=()) for a in advice))
        maximum = max(lexical, default=0)
        fused = [(score + word_score / maximum) / 2 if maximum else score
                 for score, word_score in zip(scores, lexical, strict=True)]
        relative_floor = 0.8 * max(fused, default=-1)
        returned_scores = [score if score >= relative_floor else -1.0 for score in fused]
        return returned_scores, self.metadata | {"query_coverage": query_coverage,
            "raw_cosine_scores": {a.identity: score for a, score in zip(advice, scores)},
            "positive_lexical_scores": {a.identity: score for a, score in zip(advice, lexical)},
            "effective_floor": max(self.threshold, relative_floor),
            "entry_coverage": {a.identity: self.cache[key][1] for a, key in zip(advice, keys)},
            "encoded_entries": len(missing), "cached_entries": len(keys)-len(missing),
            "elapsed_seconds": time.perf_counter()-started}


CANDIDATE_PREPROCESSING = "minilm-window128-three-fields-context-dense24-positive-bm25-8-v1"


class LocalCandidateRetrieval:
    """Candidate discovery only. Complete text windows, no delivery threshold."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.metadata = {"backend":"candidate", "model":MODEL,
            "revision":encoder.metadata['revision'], "preprocessing":CANDIDATE_PREPROCESSING,
            "files_identity":encoder.metadata['files_identity'], "dense_limit":24, "lexical_limit":8}
        self.vectors, self.queries = OrderedDict(), OrderedDict()

    def candidates(self, query, advice, *, retrieval_context=None):
        started = time.perf_counter()
        advice = tuple(sorted(advice, key=lambda a:(a.identity,a.version)))
        texts = [canonical_json({"guidance":a.guidance, "applicability":a.applicability,
            "negative_conditions":a.negative_conditions}) for a in advice]
        keys = [identity({"text":text,"encoder":self.metadata}) for text in texts]
        missing = dict((key,text) for key,text in zip(keys,texts) if key not in self.vectors)
        encoding_started = time.perf_counter()
        if missing:
            self.vectors.update(zip(missing,self.encoder._encode(list(missing.values()),'')))
        encoding_seconds = time.perf_counter()-encoding_started
        query_text = canonical_json({'query':query,'public_working_context':retrieval_context or {}})
        query_key = identity({'text':query_text,'encoder':self.metadata})
        query_cached = query_key in self.queries
        if not query_cached:
            self.queries[query_key] = self.encoder._encode([query_text],'')[0]
        self.queries.move_to_end(query_key)
        query_vectors, query_coverage = self.queries[query_key]
        dense = [float((query_vectors @ self.vectors[key][0].T).max()) for key in keys]
        lexical = _bm25_advice_scores(_terms(query),tuple(replace(a,negative_conditions=()) for a in advice))
        dense_indices = sorted(range(len(advice)),key=lambda i:(-dense[i],advice[i].identity,advice[i].version))[:24]
        lexical_indices = sorted((i for i in range(len(advice)) if lexical[i]>0),
            key=lambda i:(-lexical[i],advice[i].identity,advice[i].version))[:8]
        chosen = sorted(set(dense_indices)|set(lexical_indices))
        result = tuple(advice[i] for i in chosen)
        trace = self.metadata | {'selected':[{"identity":a.identity,"version":a.version} for a in result],
            'candidate_scores':[{'identity':a.identity,'version':a.version,'cosine':dense[i],
                'positive_bm25':lexical[i],'dense_selected':i in dense_indices,'lexical_selected':i in lexical_indices,
                'coverage':self.vectors[keys[i]][1]} for i,a in enumerate(advice)],
            'encoded_entries':len(missing),'cached_entries':len(keys)-len(missing),
            'query_cached':query_cached,'query_coverage':query_coverage,'query_context_identity':query_key,
            'candidate_encoding_seconds':encoding_seconds,'elapsed_seconds':time.perf_counter()-started}
        # Bounded immutable content caches, with no run-specific search history.
        for key in keys:
            self.vectors.move_to_end(key)
        while len(self.vectors)>2048:
            self.vectors.popitem(last=False)
        while len(self.queries)>128:
            self.queries.popitem(last=False)
        return result, trace


RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_PREPROCESSING = "bge-pair2048-query512-fields-complete-v1"


class LocalRerankRetrieval:
    """Deployment-owned CPU models; all eligibility and run state stay in Core.

    A threshold of None is diagnostic only. Production configuration must supply
    the independently calibrated, finite logit threshold.
    """
    def __init__(self, model_path, revision, threshold, *, encoder=None, batch_size=4):
        if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or (threshold is not None and (isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)))
                or type(batch_size) is not int or not 1 <= batch_size <= 32):
            raise ValueError("Invalid rerank configuration")
        self.path = Path(model_path)
        receipt = json.loads((self.path / "memory-model.json").read_text())
        if receipt.get("model") != RERANK_MODEL or receipt.get("revision") != revision:
            raise ValueError("Reranker receipt differs from configured model")
        files = receipt.get("files", [])
        if (len(files) != len(MODEL_FILES) or {f.get("name") for f in files} != MODEL_FILES
                or any(type(f.get("bytes")) is not int or f["bytes"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256"))) for f in files)):
            raise ValueError("Incomplete reranker model receipt")
        # Exactly once at deployment construction; rank never hashes weights.
        for item in files:
            path = self.path / item["name"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
                raise ValueError("Reranker model asset identity changed")
        self.threshold = threshold
        self.metadata = {"backend": "rerank", "model": RERANK_MODEL, "revision": revision,
            "preprocessing": RERANK_PREPROCESSING, "files_identity": identity(files)}
        self.model = self.tokenizer = None
        self.encoder = encoder
        self.batch_size = batch_size
        from collections import OrderedDict
        self.vectors = OrderedDict()

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(4)
            self.tokenizer = AutoTokenizer.from_pretrained(self.path,
                trust_remote_code=False, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path,
                trust_remote_code=False, local_files_only=True,
                torch_dtype=torch.float32).eval().to("cpu")

    def _fit(self, text, budget):
        """Clip at Unicode text boundaries, retaining both ends and an omission marker."""
        encode = lambda value: self.tokenizer.encode(value, add_special_tokens=False,
            truncation=False, verbose=False)
        if len(encode(text)) <= budget:
            return text
        marker = "\n[omitted]\n"
        low, high = 0, len(text)
        while low < high:
            count = (low + high + 1) // 2
            candidate = text[:(count+1)//2] + marker + (text[-(count//2):] if count//2 else "")
            if len(encode(candidate)) <= budget:
                low = count
            else:
                high = count - 1
        return text[:(low+1)//2] + marker + (text[-(low//2):] if low//2 else "")

    def _query_text(self, query, context):
        if not context:
            return self._fit(query, 512)
        fields = [("Query", query), ("Task", context.get("task", "")),
            ("Notebook (interpretation, not evidence)", context.get("notebook", "")),
            ("Latest observation", context.get("observation", ""))]
        # All labels and separators count against the combined 512-token budget.
        budgets = [320, 64, 64, 64]
        if context.get("results"):
            fields.append(("Visible results", context["results"]))
            budgets = [256, 64, 64, 64, 64]
        if context.get("materials"):
            fields.append(("Visible materials", context["materials"]))
            budgets = [512 - 64 * (len(fields)-1)] + [64] * (len(fields)-1)
        empty = sum(64 for _, text in fields[1:] if not text)
        for i in range(1, len(fields)):
            if fields[i][1] and empty:
                budgets[i] += empty
                empty = 0
        parts = []
        for (label, value), budget in zip(fields, budgets, strict=True):
            if value:
                parts.append(self._fit(label + ": " + value, budget))
        return self._fit("\n".join(parts), 512)

    def _candidates(self, query, advice):
        if len(advice) <= 32:
            return set(range(len(advice))), {"small_package": True, "encoded_entries": 0}
        if self.encoder is None:
            raise OSError("Large-package retrieval requires the deployment encoder")
        texts = [a.guidance + "\nApplicability: " + "\n".join(a.applicability) for a in advice]
        keys = [identity({"text": t, "revision": self.encoder.metadata["revision"],
            "preprocessing": self.encoder.metadata["preprocessing"]}) for t in texts]
        missing = dict((k, t) for k, t in zip(keys, texts, strict=True) if k not in self.vectors)
        # Use vectors locally for this request even when its size exceeds the LRU.
        fresh = dict(zip(missing, self.encoder._encode(list(missing.values()), ""))) if missing else {}
        vectors = {k: self.vectors[k] if k in self.vectors else fresh[k] for k in keys}
        for k in keys:
            self.vectors[k] = vectors[k]
            self.vectors.move_to_end(k)
            if len(self.vectors) > 4096:
                self.vectors.popitem(last=False)
        q, _ = self.encoder._encode([query], "")[0]
        dense = [float((q @ vectors[k][0].T).max()) for k in keys]
        lexical = _bm25_advice_scores(_terms(query), tuple(replace(a, negative_conditions=()) for a in advice))
        order = lambda scores: sorted(range(len(advice)),
            key=lambda i: (-scores[i], advice[i].identity, advice[i].version))
        selected = set(order(dense)[:24]) | set([i for i in order(lexical) if lexical[i] > 0][:8])
        return selected, {"small_package": False, "encoded_entries": len(missing),
            "cached_entries": len(keys)-len(missing), "vector_cache_entries": len(self.vectors)}

    def score(self, query, advice, *, retrieval_context=None):
        import torch
        started = time.perf_counter()
        self.load()
        selected, candidate_trace = self._candidates(query, advice)
        query_text = self._query_text(query, retrieval_context)
        rows, pairs, indices = [], [], []
        for i, a in enumerate(advice):
            row = {"identity": a.identity, "version": a.version, "status": "not_selected"}
            rows.append(row)
            if i not in selected:
                continue
            document = ("Guidance: " + a.guidance + "\nApplicability: " + "\n".join(a.applicability)
                + "\nNegative conditions: " + "\n".join(a.negative_conditions))
            pair = self.tokenizer(query_text, document, truncation=False, verbose=False)
            row.update(model_input={"query_context": query_text, "advice": document},
                input_identity=identity({"query_context": query_text, "advice": document,
                    "preprocessing": RERANK_PREPROCESSING}), pair_tokens=len(pair["input_ids"]))
            if len(pair["input_ids"]) > 2048:
                row["status"] = "unscored_input_too_long"
            else:
                pairs.append(pair)
                indices.append(i)
        prepare_seconds = time.perf_counter() - started
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = self.tokenizer.pad(pairs[start:start+self.batch_size], padding=True, return_tensors="pt")
                logits = self.model(**batch, return_dict=True).logits.reshape(-1).float().tolist()
                for i, logit in zip(indices[start:start+self.batch_size], logits, strict=True):
                    if not math.isfinite(logit):
                        raise OSError("Reranker returned nonfinite logit")
                    rows[i].update(status="scored", logit=logit)
        elapsed = time.perf_counter() - started
        return [r.get("logit") for r in rows], self.metadata | candidate_trace | {
            "candidate_scores": rows, "coverage_complete": all(r["status"] != "unscored_input_too_long" for r in rows),
            "prepare_seconds": prepare_seconds, "score_seconds": elapsed-prepare_seconds,
            "elapsed_seconds": elapsed, "batch_size": self.batch_size}


class RerankRPC:
    """Host adapter: read deployment identity once, send only authorized public text."""
    def __init__(self, endpoint, revision, threshold):
        if (not isinstance(revision,str) or not re.fullmatch(r"[0-9a-f]{40}",revision)
                or isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold)):
            raise ValueError("Rerank requires a fixed revision and calibrated finite logit threshold")
        receipt = json.loads(Path(endpoint).read_text())
        metadata = receipt.get('model',{})
        if (receipt.get('schema_version')!='memory-rerank-ready-v1' or receipt.get('status')!='ready'
                or receipt.get('calibration',{}).get('rerank_logit_threshold')!=threshold
                or metadata.get('backend')!='rerank' or metadata.get('revision')!=revision
                or metadata.get('model')!=RERANK_MODEL or metadata.get('preprocessing')!=RERANK_PREPROCESSING
                or not re.fullmatch(r'[0-9a-f]{64}',str(metadata.get('files_identity')))
                or not isinstance(metadata.get('encoder'),dict)
                or metadata.get('encoder',{}).get('model')!=MODEL
                or metadata.get('encoder',{}).get('preprocessing')!=PREPROCESSING
                or any(not re.fullmatch(pattern,str(metadata.get('encoder',{}).get(key)))
                    for key,pattern in [('revision',r'[0-9a-f]{40}'),('files_identity',r'[0-9a-f]{64}')])
                or type(receipt.get('worker_pid')) is not int or receipt['worker_pid']<=0
                or any(not isinstance(receipt.get(k),str) or not Path(receipt[k]).is_absolute()
                    for k in ('socket_path','worker_path','python_path'))
                or (receipt.get('ssh_host') is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',receipt['ssh_host']))):
            raise ValueError('Invalid or mismatched rerank ready receipt')
        self.receipt = receipt
        self.metadata = metadata
        self.threshold = threshold

    def score(self, query, advice, *, retrieval_context=None):
        payload = {'query':query,'context':retrieval_context,'candidates':to_primitive(advice)}
        trace = _retrieval_rpc(self.receipt,self.metadata,'rank',payload)
        if any(trace.get(k)!=v for k,v in self.metadata.items() if k!='encoder'):
            raise OSError('Rerank score model identity mismatch')
        rows = trace.get('candidate_scores')
        if (not isinstance(rows,list) or len(rows)!=len(advice)
                or any(not isinstance(r,dict) or r.get('identity')!=a.identity or r.get('version')!=a.version
                    or r.get('status') not in {'scored','not_selected','unscored_input_too_long'}
                    or (r.get('status')=='scored' and (isinstance(r.get('logit'),bool)
                        or not isinstance(r.get('logit'),(int,float)) or not math.isfinite(r['logit'])))
                    or (r.get('status')!='scored' and 'logit' in r)
                    for r,a in zip(rows,advice,strict=True))):
            raise OSError('Rerank response candidate coverage mismatch')
        return [r.get('logit') for r in rows], trace


def _retrieval_rpc(receipt, metadata, operation, payload, timeout=30.0):
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or not 0<timeout<=30:
        raise OSError('Retrieval request has no usable time budget')
    import shlex
    import subprocess
    from uuid import uuid4
    from .trajectory_memory_retrieval_worker import MAX_BYTES, rank_rpc

    request = {'op':operation,'request_id':uuid4().hex,'request_identity':identity(payload),
        'payload':payload,'expires_at':time.time()+timeout}
    raw = json.dumps(request,ensure_ascii=False,allow_nan=False).encode()
    if len(raw)+1>MAX_BYTES:
        raise OSError('Rerank request exceeds 8 MiB')
    started = time.perf_counter()
    try:
        if receipt.get('ssh_host'):
            command = [receipt['python_path'], receipt['worker_path'],
                'rpc','--socket',receipt['socket_path'],'--timeout',str(timeout)]
            remote = 'env -i PATH=/usr/bin:/bin ' + shlex.join(command)
            result = subprocess.run(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10',
                receipt['ssh_host'],remote],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                timeout=timeout,check=True,env={'PATH':'/usr/bin:/bin','HOME':str(Path.home())})
            if len(result.stdout)>MAX_BYTES:
                raise OSError('Rerank response exceeds 8 MiB')
            response = json.loads(result.stdout)
        else:
            response = rank_rpc(receipt['socket_path'],request,timeout=timeout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        raise OSError('Rerank service unavailable: '+type(error).__name__) from error
    if (not isinstance(response,dict) or response.get('request_id')!=request['request_id']
            or response.get('request_identity')!=request['request_identity']
            or response.get('model')!=metadata
            or response.get('worker_pid')!=receipt['worker_pid']):
        raise OSError('Rerank response identity mismatch')
    if response.get('status')!='ok':
        raise OSError('Rerank service unavailable: '+str(response.get('reason','unknown')))
    trace = response.get('retrieval')
    if not isinstance(trace,dict):
        raise OSError('Retrieval response missing trace')
    return trace | {'worker_pid':response['worker_pid'],'queue_seconds':response.get('queue_seconds'),
        'rank_seconds':response.get('rank_seconds'),'rpc_elapsed_seconds':time.perf_counter()-started,
        'rpc_imports_heavy_modules':response.get('rpc_imports_heavy_modules')}


class CandidateRPC:
    """Light host client for the encoder-only worker profile."""
    def __init__(self, endpoint):
        receipt = json.loads(Path(endpoint).read_text())
        metadata = receipt.get('model',{})
        if (receipt.get('schema_version')!='memory-candidate-ready-v1' or receipt.get('status')!='ready'
                or receipt.get('profile')!='candidate' or metadata.get('backend')!='candidate'
                or metadata.get('model')!=MODEL or metadata.get('preprocessing')!=CANDIDATE_PREPROCESSING
                or metadata.get('dense_limit')!=24 or metadata.get('lexical_limit')!=8
                or not re.fullmatch(r'[0-9a-f]{40}',str(metadata.get('revision')))
                or not re.fullmatch(r'[0-9a-f]{64}',str(metadata.get('files_identity')))
                or type(receipt.get('worker_pid')) is not int or receipt['worker_pid']<=0
                or any(not isinstance(receipt.get(k),str) or not Path(receipt[k]).is_absolute()
                    for k in ('socket_path','worker_path','python_path'))
                or (receipt.get('ssh_host') is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',receipt['ssh_host']))):
            raise ValueError('Invalid candidate ready receipt')
        self.receipt, self.metadata = receipt, metadata

    def candidates(self, query, advice, *, retrieval_context=None, timeout=30.0):
        payload = {'query':query,'context':retrieval_context,'candidates':to_primitive(advice)}
        trace = _retrieval_rpc(self.receipt,self.metadata,'candidate',payload,timeout)
        if any(trace.get(k)!=v for k,v in self.metadata.items()):
            raise OSError('Candidate response model identity mismatch')
        selected = trace.get('selected')
        bindings = {(a.identity,a.version):a for a in advice}
        if (not isinstance(selected,list) or not 0<len(selected)<=min(32,len(advice))
                or any(not isinstance(a,dict) or set(a)!={'identity','version'}
                    or not isinstance(a['identity'],str) or not isinstance(a['version'],str)
                    or (a['identity'],a['version']) not in bindings for a in selected)
                or len({(a['identity'],a['version']) for a in selected})!=len(selected)):
            raise OSError('Candidate response binding mismatch')
        return tuple(bindings[(a['identity'],a['version'])] for a in selected), trace
