"""Run the LangGraph agent workflow headlessly (no Streamlit server).

This script injects a tiny fake `streamlit` module into sys.modules so the
repo modules that import `streamlit` during initialization don't fail. It
then imports the compiled graph from `src.aneeta.graph.workflow.get_graph`
and drives it the same way `app.py` does, but prints progress and the
final assistant output to stdout.

Usage examples (Windows cmd.exe):
  set LLM_MODEL=phi4-mini
  python scripts\run_agents_headless.py --question "Explain photosynthesis" --lang english

This intentionally tries to preserve the same logic as the Streamlit app
while avoiding any HTTP server.
"""
import sys
import os
import time
import types
import argparse
from contextlib import contextmanager


class _SessionState(dict):
    def clear(self):
        super().clear()


class _Status:
    def __init__(self, label="", state=None, expanded=False):
        self.label = label
        self.state = state
        self.expanded = expanded

    def update(self, label=None, state=None, expanded=None):
        if label is not None:
            self.label = label
        if state is not None:
            self.state = state
        if expanded is not None:
            self.expanded = expanded


class _FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        # Minimal session_state used by the repo
        self.session_state = _SessionState()

    def cache_resource(self, fn):
        # Very small no-op caching decorator: return function unchanged
        return fn

    @contextmanager
    def spinner(self, msg: str):
        # Simple console spinner context manager (no animation)
        print(f"[spinner] {msg}")
        try:
            yield
        finally:
            pass

    def status(self, label="", expanded=False):
        # Return a simple status-like object supporting .update()
        st = _Status(label=label, state=None, expanded=expanded)

        @contextmanager
        def _ctx():
            print(f"[status] {label}")
            try:
                yield st
            finally:
                pass

        return _ctx()

    # Helpful no-op UI methods used elsewhere
    def write(self, *args, **kwargs):
        print(*args)

    def markdown(self, *args, **kwargs):
        print(*args)

    def error(self, *args, **kwargs):
        print("ERROR:", *args)

    def stop(self):
        raise SystemExit(1)


def inject_fake_streamlit():
    """Insert a fake minimal `streamlit` module if real Streamlit isn't desired."""
    fake = _FakeStreamlit()
    sys.modules["streamlit"] = fake
    return fake


def consume_stream(response_stream):
    """Consume a response_stream returned by agents/chains.

    Returns a tuple (full_text, metadata) where metadata is a dict merged from
    any chunk that exposes response_metadata or usage_metadata attributes or
    dict-like metadata. This is best-effort: agents may yield plain strings.
    """
    parts = []
    metadata = {}
    try:
        # If it's an iterator/generator
        if hasattr(response_stream, "__iter__") and not isinstance(response_stream, (str, bytes)):
            for chunk in response_stream:
                # If chunk is an object with .content and metadata, try to extract
                try:
                    if hasattr(chunk, "content"):
                        c = getattr(chunk, "content")
                        parts.append(str(c))
                    else:
                        parts.append(str(chunk))

                    # Merge metadata if present
                    if hasattr(chunk, "response_metadata"):
                        metadata.setdefault("response_metadata", {}).update(getattr(chunk, "response_metadata") or {})
                    if hasattr(chunk, "usage_metadata"):
                        metadata.setdefault("usage_metadata", {}).update(getattr(chunk, "usage_metadata") or {})
                except Exception:
                    parts.append(str(chunk))
                # Print progressive chunks to stdout to simulate streaming
                print(parts[-1], end="", flush=True)
            print("")
        else:
            # Single value
            if hasattr(response_stream, "content"):
                parts.append(str(getattr(response_stream, "content")))
            else:
                parts.append(str(response_stream))
            # Try extract metadata
            if hasattr(response_stream, "response_metadata"):
                metadata.setdefault("response_metadata", {}).update(getattr(response_stream, "response_metadata") or {})
            if hasattr(response_stream, "usage_metadata"):
                metadata.setdefault("usage_metadata", {}).update(getattr(response_stream, "usage_metadata") or {})

        return "".join(parts), metadata
    except Exception as e:
        print("Failed to consume response stream:", e)
        return "", metadata


def main():
    parser = argparse.ArgumentParser(description="Run ANEETA LangGraph workflow headlessly (no Streamlit).")
    
    # Create mutually exclusive group for question input modes
    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument("--question", "-q", type=str, help="A single user question to send to the agents.")
    question_group.add_argument("--questions", type=int, help="Number of questions to randomly sample from ground-truth dataset for batch benchmarking.")
    
    parser.add_argument("--lang", "-l", default="english", help="User explanation language (lowercase).")
    parser.add_argument("--judge", action="store_true", help="Enable automatic judging against a ground-truth dataset (requires --gt-file).")
    parser.add_argument("--judge-llm", action="store_true", help="When --judge is set, use the LLM to assist scoring and explanation checks (slower, may be non-deterministic).")
    parser.add_argument("--gt-file", type=str, default=None, help="Path to ground-truth JSON/JSONL file containing solved MCQs (optional).")
    parser.add_argument("--recursion-limit", type=int, default=10, help="Recursion limit passed to graph.stream")
    parser.add_argument("--bench", "-b", type=int, default=0, help="Run benchmark iterations (N). If 0, run a single invocation.")
    parser.add_argument("--warmup", "-w", type=int, default=1, help="Number of warmup runs before measuring (default 1).")
    parser.add_argument("--csv", type=str, default=None, help="Path to write per-run metrics CSV (optional).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Alias for --csv (output file for metrics).")
    args = parser.parse_args()
    
    # Handle --output as alias for --csv
    if args.output and not args.csv:
        args.csv = args.output
    
    # Validate arguments
    if args.questions:
        if not args.gt_file:
            parser.error("--questions requires --gt-file to be specified for sampling questions")
        # Force judging on when using batch mode
        args.judge = True

    # Inject fake Streamlit BEFORE importing any repo modules that expect `streamlit`.
    fake_st = inject_fake_streamlit()

    # Ensure project root (repo) is on sys.path like app.py does
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Load environment variables from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Import minimal runtime pieces (these imports will use the fake streamlit shim)
    from langchain_core.messages import HumanMessage
    import json

    # Import the graph and global resources
    try:
        from src.aneeta.core.resources import llm, creative_llm, vector_stores, logs, embeddings
        from src.aneeta.graph.workflow import get_graph
        # import judge module if available
        from src.aneeta.eval.judge import score_answer as judge_score_answer
    except Exception as e:
        print("Failed to import ANEETA modules:", e)
        raise

    # Print the resource load logs (like the app's status block)
    print("Knowledge base logs:")
    for log in logs:
        print(" -", log)

    graph = get_graph()
    
    # Optionally load ground-truth dataset for judging
    gt_data = None
    gt_embeddings = None
    gt_questions = None
    if args.judge and args.gt_file:
        try:
            with open(args.gt_file, "r", encoding="utf-8") as fh:
                gt_data = json.load(fh)
                print(f"Loaded {len(gt_data)} ground-truth entries from {args.gt_file}")
        except Exception as e:
            print("Failed to load ground-truth file:", e)
            gt_data = None
    
    # Handle batch question mode: sample N questions from ground truth
    questions_to_run = []
    if args.questions:
        # Sample random questions from ground truth - use page_content field
        import random
        if gt_data and len(gt_data) > 0:
            # Filter records that have page_content
            valid_records = [rec for rec in gt_data if rec.get("page_content", "").strip()]
            
            if not valid_records:
                print("Error: No valid records with page_content found in ground-truth data")
                return
            
            sample_size = min(args.questions, len(valid_records))
            sampled_records = random.sample(valid_records, sample_size)
            
            # Use page_content as the question text
            for rec in sampled_records:
                q_text = rec.get("page_content", "").strip()
                if q_text:
                    # Truncate very long content to first 500 chars (usually contains the question)
                    questions_to_run.append(q_text[:500])
            
            print(f"Sampled {len(questions_to_run)} questions from ground-truth dataset")
        else:
            print("Error: No ground-truth data available for sampling")
            return
    else:
        # Single question mode
        questions_to_run = [args.question]
    
    # Compute GT embeddings for fuzzy matching (only if we have judge mode enabled)
    if args.judge and gt_data:
        try:
            # Extract all page_content as questions for embedding
            gt_questions = [rec.get("page_content", "").strip()[:500] for rec in gt_data if rec.get("page_content", "").strip()]
            if gt_questions and embeddings is not None:
                print(f"\n{'='*60}")
                print(f"Computing embeddings for {len(gt_questions)} ground-truth questions...")
                print(f"Embedding function: {embeddings.__class__.__name__}")
                print(f"Sample texts to embed (first 2):")
                for i, q in enumerate(gt_questions[:2]):
                    sample_text = (q[:80] + "...") if len(q) > 80 else q
                    print(f"  [{i+1}] {sample_text}")
                print(f"{'='*60}")
                
                # Batch embedding with progress feedback
                batch_size = 32  # Process in batches to avoid memory issues
                total_batches = (len(gt_questions) + batch_size - 1) // batch_size
                gt_embeddings = []
                
                for batch_idx in range(0, len(gt_questions), batch_size):
                    batch_end = min(batch_idx + batch_size, len(gt_questions))
                    batch = gt_questions[batch_idx:batch_end]
                    current_batch = (batch_idx // batch_size) + 1
                    
                    print(f"[Batch {current_batch}/{total_batches}] Embedding {batch_idx+1}-{batch_end}/{len(gt_questions)}...", end="", flush=True)
                    try:
                        batch_embeddings = embeddings.embed_documents(batch)
                        gt_embeddings.extend(batch_embeddings)
                        print(f" ✓ ({len(batch_embeddings)} embeddings)")
                    except Exception as batch_err:
                        print(f" ✗ ERROR: {str(batch_err)[:100]}")
                        raise
                
                print(f"\n✓ Embeddings computed successfully: {len(gt_embeddings)} embeddings")
                if gt_embeddings:
                    print(f"  Embedding dimension: {len(gt_embeddings[0]) if isinstance(gt_embeddings[0], list) else 'unknown'}")
        except Exception as e:
            import traceback
            print(f"\n✗ WARNING: Failed to compute GT embeddings")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"  Full traceback:")
            traceback.print_exc()
            gt_embeddings = None
            gt_questions = None

    node_to_status = {
        "agent_router": "Routing to the correct agent...",
        "teacher_vectordb_router": "Determining the subject...",
        "teacher_agent": "Engaging the Teacher Agent...",
        "mcq_question_solver_agent": "Engaging the MCQ Solver Agent...",
        "trainer_agent": "Preparing your interactive quiz...",
        "mentor_agent": "Engaging the Mentor Agent...",
        "general_query_agent": "Thinking about your general query..."
    }

    import csv
    import uuid
    from datetime import datetime, timezone
    import json

    def percentile(values, p):
        if not values:
            return None
        vs = sorted(values)
        k = (len(vs) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(vs) - 1)
        if f == c:
            return vs[int(k)]
        d0 = vs[f] * (c - k)
        d1 = vs[c] * (k - f)
        return d0 + d1

    def run_once(question_text: str):
        """Run the graph once and return (metrics_dict, full_response_text)."""
        start = time.time()
        final_state_local = {}
        try:
            # Ensure input_data matches expected type for graph.stream
            # Replace State with _SessionState for compatibility
            # Construct a State-compatible dictionary
            # Refine State-compatible dictionary with valid placeholders
            input_data = {
                "messages": [HumanMessage(content=question_text)],
                "user_explanation_language": args.lang.lower(),
                "agent_routing": "mcq_question_solver",  # Example routing
                "teacher_vectordb_routing": "biology",  # Example routing
                "response_stream": (m for m in [])  # Empty generator as placeholder
            }
            
            # Debug: Log input preparation
            debug_enabled = os.getenv("DEBUG_HEADLESS", "0") == "1"
            if debug_enabled:
                print(f"[DEBUG] Input prepared: messages={len(input_data.get('messages', []))} items")
            
            for s in graph.stream(input=input_data):
                node_name_local = list(s.keys())[-1]
                if node_name_local in node_to_status:
                    # Minimal progress indicator
                    print(f"[node] {node_to_status[node_name_local]}")
                final_state_local = s
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Log detailed error information
            print(f"\n[ERROR] Exception during graph.stream()")
            print(f"  Type: {error_type}")
            print(f"  Message: {error_msg}")
            print(f"  Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    print(f"    {line}")
            
            return {"success": 0, "error_message": error_msg, "error_type": error_type}, ""

        if not final_state_local:
            return {"success": 0, "error_message": "No final state produced"}, ""

        last_node = next(reversed(final_state_local))
        payload = final_state_local[last_node]

        if "response_stream" not in payload:
            return {"success": 0, "error_message": "No response_stream in payload", "agent": last_node}, ""

        # Consume response and capture metadata if any
        t0 = time.time()
        try:
            full_text, meta = consume_stream(payload["response_stream"])
        except Exception as stream_err:
            print(f"\n[ERROR] Failed to consume response stream")
            print(f"  Message: {str(stream_err)}")
            return {"success": 0, "error_message": f"Stream error: {str(stream_err)}"}, ""
        
        t1 = time.time()

        wall_ms = int((t1 - start) * 1000)
        llm_ms = None
        input_tokens = None
        output_tokens = None
        total_tokens = None

        # Try to pull provider timing/token metadata if present
        resp_meta = meta.get("response_metadata") if isinstance(meta, dict) else None
        usage_meta = meta.get("usage_metadata") if isinstance(meta, dict) else None
        if resp_meta:
            # Common key used earlier: total_duration (ms)
            try:
                if "total_duration" in resp_meta:
                    llm_ms = int(resp_meta.get("total_duration"))
                elif "duration_ms" in resp_meta:
                    llm_ms = int(resp_meta.get("duration_ms"))
            except Exception:
                pass
        if usage_meta:
            try:
                input_tokens = int(usage_meta.get("input_tokens")) if usage_meta.get("input_tokens") is not None else None
                output_tokens = int(usage_meta.get("output_tokens")) if usage_meta.get("output_tokens") is not None else None
                total_tokens = int(usage_meta.get("total_tokens")) if usage_meta.get("total_tokens") is not None else None
            except Exception:
                pass

        metrics = {
            "run_id": str(uuid.uuid4()),
            # timezone-aware UTC timestamp
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": os.getenv("LLM_MODEL") or "",
            "agent": last_node,
            "question": (question_text[:500] + "...") if len(question_text) > 500 else question_text,
            "wall_ms": wall_ms,
            "llm_ms": llm_ms,
            "stream_first_token_ms": None,  # Time to first token
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": None,  # Output tokens per second
            "retrieval_ms": None,
            "retrieved_doc_count": None,
            # Fuzzy GT fields
            "nearest_gt_question": None,
            "nearest_similarity": None,
            "nearest_gt_index": None,
            # Fields added for judge output (may remain None)
            "predicted_text": (full_text[:1000] + "...") if len(full_text) > 1000 else full_text,
            "predicted_label": None,
            "deterministic_match": None,
            "correct": None,
            "score": None,
            "explanation_score": None,
            "clarity": None,  # Clarity score from LLM judge (1-5)
            "reasoning": None,  # Reasoning score from LLM judge (1-5)
            "hallucination": None,
            "hallucinated_claims": None,  # List of hallucinated claims
            "judge_method": None,  # "deterministic" or "llm"
            "judge_error": None,  # Error message if judging failed
            "success": 1,
            "error_message": "",
        }
        
        # Calculate tokens per second if we have output tokens and llm_ms
        if output_tokens is not None and llm_ms is not None and llm_ms > 0:
            metrics["tokens_per_second"] = round((output_tokens / llm_ms) * 1000, 2)

        # If judge requested and ground-truth dataset was loaded, attempt to score
        try:
            # Pull retrieval metadata from payload if agent attached it
            if isinstance(payload, dict) and payload.get("retrieval_metadata"):
                rm = payload.get("retrieval_metadata") or {}
                # Fix type issues for metrics retrieval
                metrics["retrieval_ms"] = int(rm.get("retrieval_ms", 0))
                metrics["retrieved_doc_count"] = int(rm.get("retrieved_doc_count", 0))

            if args.judge and gt_data and isinstance(gt_data, list):
                # Try to find a matching question in gt_data. Matching strategy: exact question_text equality,
                # or substring match. The gt entries are expected to contain keys: question_text, options, correct_label, solution_text
                matched = None
                for rec in gt_data:
                    q = rec.get("question_text") or rec.get("question") or rec.get("question_text_raw")
                    if not q:
                        continue
                    if q.strip() == question_text.strip():
                        matched = rec
                        break
                if not matched:
                    # fallback: substring
                    for rec in gt_data:
                        q = rec.get("question_text") or rec.get("question") or rec.get("question_text_raw")
                        if not q:
                            continue
                        if q.strip() and q.strip() in question_text:
                            matched = rec
                            break

                if matched:
                    try:
                        jr = judge_score_answer(matched, full_text, use_llm=args.judge_llm)
                        metrics.update({
                            "predicted_label": jr.get("mapped_label"),
                            "deterministic_match": jr.get("deterministic_match"),
                            "correct": jr.get("correct"),
                            "score": jr.get("score"),
                            "explanation_score": jr.get("explanation_score"),
                            "clarity": jr.get("clarity"),
                            "reasoning": jr.get("reasoning"),
                            "hallucination": jr.get("hallucination"),
                            "hallucinated_claims": str(jr.get("hallucinated_claims", [])) if jr.get("hallucinated_claims") else None,
                            "judge_method": jr.get("judge_metadata", {}).get("method"),
                            "judge_error": jr.get("judge_metadata", {}).get("llm_error"),
                        })
                    except Exception as e:
                        metrics["judge_error"] = str(e)
                else:
                    # No exact or substring match; try embedding-based fuzzy matching if GT embeddings are available
                    try:
                        if gt_embeddings and gt_questions:
                            # embed the question
                            try:
                                q_emb = embeddings.embed_documents([question_text])[0]
                            except Exception:
                                q_emb = None
                            if q_emb is not None:
                                import math
                                def cosine(a, b):
                                    da = sum(x * y for x, y in zip(a, b))
                                    na = math.sqrt(sum(x * x for x in a))
                                    nb = math.sqrt(sum(y * y for y in b))
                                    if na == 0 or nb == 0:
                                        return 0.0
                                    return da / (na * nb)

                                sims = [cosine(q_emb, ge) for ge in gt_embeddings]
                                # top 5
                                idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:5]
                                print("No exact GT match found. Nearest GT candidates (top 5):")
                                for i in idxs:
                                    print(f" - score={sims[i]:.3f} question={gt_questions[i][:120]}")
                                best_idx = idxs[0] if idxs else None
                                best_score = sims[best_idx] if best_idx is not None else 0.0
                                # Always record the nearest candidate info for debugging/CSV
                                if best_idx is not None:
                                    metrics["nearest_gt_question"] = gt_questions[best_idx]
                                    metrics["nearest_similarity"] = float(best_score)
                                    metrics["nearest_gt_index"] = int(best_idx)
                                else:
                                    metrics["nearest_gt_question"] = None
                                    metrics["nearest_similarity"] = None
                                    metrics["nearest_gt_index"] = None

                                MATCH_THRESHOLD = float(os.getenv("GT_MATCH_THRESHOLD", "0.70"))
                                # Fix type issues for ground-truth matching
                                # Ensure best_idx is an integer before indexing
                                if best_idx is not None and isinstance(best_idx, int) and 0 <= best_idx < len(gt_data):
                                    if best_score >= MATCH_THRESHOLD:
                                        matched = gt_data[best_idx]
                                        try:
                                            jr = judge_score_answer(matched, full_text, use_llm=args.judge_llm)
                                            metrics.update({
                                                "predicted_label": jr.get("mapped_label"),
                                                "deterministic_match": jr.get("deterministic_match"),
                                                "correct": jr.get("correct"),
                                                "score": jr.get("score"),
                                                "explanation_score": jr.get("explanation_score"),
                                                "clarity": jr.get("clarity"),
                                                "reasoning": jr.get("reasoning"),
                                                "hallucination": jr.get("hallucination"),
                                                "hallucinated_claims": str(jr.get("hallucinated_claims", [])) if jr.get("hallucinated_claims") else None,
                                                "judge_method": jr.get("judge_metadata", {}).get("method"),
                                                "judge_error": jr.get("judge_metadata", {}).get("llm_error"),
                                            })
                                        except Exception as e:
                                            metrics["judge_error"] = str(e)
                    except Exception:
                        pass

        except Exception:
            # Non-fatal; continue returning run metrics
            pass

        return metrics, full_text

    # Batch benchmarking mode: run each question once
    if args.questions:
        rows = []
        print(f"\n{'='*60}")
        print(f"Running benchmarks on {len(questions_to_run)} questions...")
        print(f"{'='*60}\n")
        
        debug_enabled = os.getenv("DEBUG_HEADLESS", "0") == "1"
        if debug_enabled:
            print("[DEBUG MODE ENABLED]")
        
        for idx, question in enumerate(questions_to_run, 1):
            print(f"\n=== Question {idx}/{len(questions_to_run)} ===")
            print(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}")
            
            if debug_enabled:
                print(f"[DEBUG] Processing question {idx}...")
                print(f"[DEBUG] Question length: {len(question)} chars")
            
            try:
                metrics, text = run_once(question)
                rows.append(metrics)
                
                # Print quick status
                if metrics.get("success"):
                    status = "✓" if metrics.get("correct") else "✗"
                    wall_time = metrics.get('wall_ms', 0)
                    tokens = metrics.get('total_tokens', 0)
                    print(f"Status: {status} | Wall: {wall_time:.0f}ms | Tokens: {tokens}")
                    
                    if debug_enabled:
                        print(f"[DEBUG] Success: wall_ms={wall_time}, tokens={tokens}")
                else:
                    error_msg = metrics.get('error_message', 'Unknown')[:80]
                    print(f"Status: ERROR - {error_msg}")
                    
                    if debug_enabled:
                        print(f"[DEBUG] Error: {metrics.get('error_message', 'Unknown')}")
                        if 'error_type' in metrics:
                            print(f"[DEBUG] Error type: {metrics.get('error_type')}")
            except Exception as q_err:
                print(f"Status: EXCEPTION - {str(q_err)[:80]}")
                if debug_enabled:
                    import traceback
                    print(f"[DEBUG] Exception during question {idx}:")
                    traceback.print_exc()
                rows.append({
                    "success": 0,
                    "error_message": f"Exception: {str(q_err)}",
                })
        
        # Write CSV
        if args.csv:
            keys = [
                "run_id", "timestamp_utc", "model", "agent", "question", 
                # Timing metrics
                "wall_ms", "llm_ms", "stream_first_token_ms",
                # Token metrics
                "input_tokens", "output_tokens", "total_tokens", "tokens_per_second",
                # Prediction + judge fields
                "predicted_text", "predicted_label", "deterministic_match", 
                "correct", "score", "explanation_score",
                "clarity", "reasoning", "hallucination", "hallucinated_claims",
                "judge_method", "judge_error",
                "success", "error_message",
            ]
            with open(args.csv, "w", newline="", encoding="utf-8") as csvfile:
                import csv
                writer = csv.DictWriter(csvfile, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"\nMetrics written to {args.csv}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        success_runs = [r for r in rows if r.get("success") == 1]
        error_count = len(rows) - len(success_runs)
        
        print(f"Total runs: {len(rows)}")
        print(f"Successful: {len(success_runs)}")
        print(f"Errors: {error_count}")
        
        if success_runs:
            # Filter and convert to numeric types, ensuring no None values
            walls = [float(r.get("wall_ms")) for r in success_runs if r.get("wall_ms") is not None and isinstance(r.get("wall_ms"), (int, float))]
            tokens = [int(r.get("total_tokens")) for r in success_runs if r.get("total_tokens") is not None and isinstance(r.get("total_tokens"), (int, float))]
            tps_vals = [float(r.get("tokens_per_second")) for r in success_runs if r.get("tokens_per_second") is not None and isinstance(r.get("tokens_per_second"), (int, float))]
            
            if walls:
                print(f"\nTiming (ms):")
                print(f"  Mean: {sum(walls)/len(walls):.2f}")
                print(f"  Min:  {min(walls):.2f}")
                print(f"  Max:  {max(walls):.2f}")
            
            if tokens:
                print(f"\nTokens:")
                print(f"  Mean: {sum(tokens)/len(tokens):.2f}")
                print(f"  Total: {sum(tokens)}")
            
            if tps_vals:
                print(f"\nTokens/sec:")
                print(f"  Mean: {sum(tps_vals)/len(tps_vals):.2f}")
            
            # Judge statistics
            if args.judge:
                correct_count = sum(1 for r in success_runs if r.get("correct") == 1)
                scores = [float(r["score"]) for r in success_runs if "score" in r and r["score"] is not None and isinstance(r.get("score"), (int, float))]
                exp_scores = [float(r["explanation_score"]) for r in success_runs if "explanation_score" in r and r["explanation_score"] is not None and isinstance(r.get("explanation_score"), (int, float))]
                
                print(f"\nAccuracy:")
                print(f"  Correct: {correct_count}/{len(success_runs)} ({100*correct_count/len(success_runs):.1f}%)")
                
                if scores:
                    print(f"\nScores:")
                    print(f"  Mean: {sum(scores)/len(scores):.2f}")
                
                if exp_scores:
                    print(f"\nExplanation scores:")
                    print(f"  Mean: {sum(exp_scores)/len(exp_scores):.2f}")
                
                # LLM judge metrics
                clarity_vals = [float(r["clarity"]) for r in success_runs if "clarity" in r and r["clarity"] is not None and isinstance(r.get("clarity"), (int, float))]
                reasoning_vals = [float(r["reasoning"]) for r in success_runs if "reasoning" in r and r["reasoning"] is not None and isinstance(r.get("reasoning"), (int, float))]
                
                if clarity_vals:
                    print(f"\nClarity (1-5):")
                    print(f"  Mean: {sum(clarity_vals)/len(clarity_vals):.2f}")
                
                if reasoning_vals:
                    print(f"\nReasoning (1-5):")
                    print(f"  Mean: {sum(reasoning_vals)/len(reasoning_vals):.2f}")
        
        if error_count > 0:
            print("\n--- Error Analysis ---")
            error_msgs = {}
            for r in rows:
                if r.get("success") != 1:
                    msg = r.get("error_message", "Unknown error")
                    error_msgs[msg] = error_msgs.get(msg, 0) + 1
            for msg, count in sorted(error_msgs.items(), key=lambda x: x[1], reverse=True):
                print(f"  [{count}x] {msg[:80]}")
        
        print("="*60)
        return

    # If benchmarking requested, run warmups + iterations and write CSV
    if args.bench and args.bench > 0:
        rows = []
        question = questions_to_run[0]  # Use first question for repeated benchmarking
        print(f"Running {args.warmup} warmup runs...")
        for i in range(max(0, args.warmup)):
            _m, _ = run_once(question)

        print(f"Running {args.bench} benchmark runs and collecting metrics...")
        for i in range(args.bench):
            print(f"\n=== Iteration {i+1}/{args.bench} ===")
            metrics, text = run_once(question)
            rows.append(metrics)

        # Write CSV if requested
        if args.csv:
            keys = [
                "run_id", "timestamp_utc", "model", "agent", "question", 
                # Timing metrics
                "wall_ms", "llm_ms", "stream_first_token_ms",
                # Token metrics
                "input_tokens", "output_tokens", "total_tokens", "tokens_per_second",
                # Prediction + judge fields
                "predicted_text", "predicted_label", "deterministic_match", 
                "correct", "score", "explanation_score", 
                "clarity", "reasoning", "hallucination", "hallucinated_claims",
                "judge_method", "judge_error",
                # Retrieval metrics
                "retrieval_ms", "retrieved_doc_count",
                # Nearest GT fuzzy match fields
                "nearest_gt_question", "nearest_similarity", "nearest_gt_index",
                # Status
                "success", "error_message"
            ]
            try:
                with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=keys)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow({k: r.get(k, "") for k in keys})
                print(f"Wrote per-run metrics to {args.csv}")
            except Exception as e:
                print("Failed to write CSV:", e)

        # Aggregate percentiles and print summary
        wall_values = [r["wall_ms"] for r in rows if r.get("wall_ms") is not None]
        llm_values = [r["llm_ms"] for r in rows if r.get("llm_ms") is not None]
        token_values = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
        tps_values = [r["tokens_per_second"] for r in rows if r.get("tokens_per_second") is not None]
        success_count = sum(1 for r in rows if r.get("success") == 1)
        error_count = len(rows) - success_count

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total runs: {len(rows)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {error_count}")
        
        def fmt_pct(values, p):
            v = percentile(values, p)
            return f"{v:.1f}" if v is not None else "n/a"

        print("\n--- Timing Metrics ---")
        print(f"wall_ms: p50={fmt_pct(wall_values, 50)}ms  p90={fmt_pct(wall_values, 90)}ms  p99={fmt_pct(wall_values, 99)}ms")
        if llm_values:
            print(f"llm_ms:  p50={fmt_pct(llm_values, 50)}ms  p90={fmt_pct(llm_values, 90)}ms  p99={fmt_pct(llm_values, 99)}ms")
        
        print("\n--- Token Metrics ---")
        if token_values:
            print(f"total_tokens: p50={fmt_pct(token_values, 50)}  p90={fmt_pct(token_values, 90)}  p99={fmt_pct(token_values, 99)}")
        if tps_values:
            print(f"tokens/sec:   p50={fmt_pct(tps_values, 50)}  p90={fmt_pct(tps_values, 90)}  p99={fmt_pct(tps_values, 99)}")
        
        # Judge metrics summary if applicable
        if args.judge:
            correct_count = sum(1 for r in rows if r.get("correct") == True)
            incorrect_count = sum(1 for r in rows if r.get("correct") == False)
            judged_count = correct_count + incorrect_count
            
            print("\n--- Judge Metrics ---")
            print(f"Judged: {judged_count}/{len(rows)}")
            if judged_count > 0:
                accuracy = (correct_count / judged_count) * 100
                print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{judged_count})")
            
            # Explanation quality metrics
            clarity_values = [r["clarity"] for r in rows if r.get("clarity") is not None]
            reasoning_values = [r["reasoning"] for r in rows if r.get("reasoning") is not None]
            exp_score_values = [r["explanation_score"] for r in rows if r.get("explanation_score") is not None]
            
            if clarity_values:
                print(f"Clarity (1-5):  avg={sum(clarity_values)/len(clarity_values):.2f}")
            if reasoning_values:
                print(f"Reasoning (1-5): avg={sum(reasoning_values)/len(reasoning_values):.2f}")
            if exp_score_values:
                print(f"Explanation score (0-1): avg={sum(exp_score_values)/len(exp_score_values):.3f}")
            
            hallucination_count = sum(1 for r in rows if r.get("hallucination") == True)
            if hallucination_count > 0:
                print(f"Hallucinations detected: {hallucination_count}/{len(rows)}")
        
        # Error analysis
        if error_count > 0:
            print("\n--- Error Analysis ---")
            error_msgs = {}
            for r in rows:
                if r.get("success") != 1:
                    msg = r.get("error_message", "Unknown error")
                    error_msgs[msg] = error_msgs.get(msg, 0) + 1
            for msg, count in sorted(error_msgs.items(), key=lambda x: x[1], reverse=True):
                print(f"  [{count}x] {msg[:80]}")
        
        print("="*60)
        return

    # Single-run fallback (existing behavior)
    print("Starting headless agent run...")
    question = questions_to_run[0]
    metrics, full_text = run_once(question)
    if metrics.get("success"):
        print("\n---\nFinished. wall_ms=", metrics.get("wall_ms"))
    else:
        print("Run failed:", metrics.get("error_message"))
    # If judge fields present, print a concise judgement summary
    if args.judge:
        print("\n" + "="*60)
        print("JUDGE SUMMARY")
        print("="*60)
        print(f" Predicted label: {metrics.get('predicted_label')}")
        print(f" Deterministic match: {metrics.get('deterministic_match')}")
        print(f" Correct: {metrics.get('correct')}")
        print(f" Score: {metrics.get('score')}")
        print(f" Explanation score: {metrics.get('explanation_score')}")
        if metrics.get('clarity') is not None:
            print(f" Clarity (1-5): {metrics.get('clarity')}")
        if metrics.get('reasoning') is not None:
            print(f" Reasoning (1-5): {metrics.get('reasoning')}")
        print(f" Hallucination: {metrics.get('hallucination')}")
        if metrics.get('hallucinated_claims'):
            print(f" Hallucinated claims: {metrics.get('hallucinated_claims')}")
        print(f" Judge method: {metrics.get('judge_method')}")
        if metrics.get('judge_error'):
            print(f" Judge error: {metrics.get('judge_error')}")
        print("="*60)


if __name__ == "__main__":
    main()
