"""Taxonomy auto-refinement via monthly batch analysis.

Samples tagged sessions, clusters embeddings, proposes taxonomy changes
via Gemini Batch API, validates proposals, and stores reports for
human approval.
"""

import json
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Default Batch API polling interval and timeout
_BATCH_POLL_INTERVAL = 60  # seconds
_BATCH_TIMEOUT = 21600     # 6 hours


class TaxonomyRefiner:
    """Monthly taxonomy refinement pipeline."""

    def __init__(
        self,
        api_key: str,
        recordings_dir: str,
        config_dir: str,
        rag_classifier,
        sample_ratio: float = 0.2,
        proposal_model: str = "gemini-2.5-flash",
        validation_model: str = "gemini-2.5-flash-lite",
    ) -> None:
        self._api_key = api_key
        self._recordings_dir = recordings_dir
        self._config_dir = config_dir
        self._rag = rag_classifier
        self._sample_ratio = max(0.01, min(1.0, sample_ratio))
        self._proposal_model = proposal_model
        self._validation_model = validation_model

        # Paths
        self._taxonomy_path = os.path.join(config_dir, "tag_taxonomy.json")
        self._pending_path = os.path.join(config_dir, "pending_taxonomy.json")
        self._reports_dir = os.path.join(config_dir, "taxonomy_reports")

        # Runtime state
        self._running = False
        self._phase: Optional[str] = None
        self._last_run: Optional[str] = None
        self._last_error: Optional[str] = None

        # Lazy genai client
        self._genai_client = None

    # ------------------------------------------------------------------
    # Genai client
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._genai_client is None:
            from google import genai
            self._genai_client = genai.Client(api_key=self._api_key)
        return self._genai_client

    # ------------------------------------------------------------------
    # Batch API helpers (follows batch_corrector.py pattern)
    # ------------------------------------------------------------------

    def _submit_batch(self, model: str, requests: list, display_name: str = "") -> str:
        """Submit inline requests to Gemini Batch API. Returns batch name."""
        client = self._get_client()
        batch_result = client.batches.create(
            model=model,
            src=requests,
            config={"display_name": display_name or "taxonomy-refiner"},
        )
        return batch_result.name

    def _poll_batch(self, batch_name: str, timeout: int = _BATCH_TIMEOUT,
                    interval: int = _BATCH_POLL_INTERVAL):
        """Poll batch job until completion. Returns batch_job object."""
        client = self._get_client()
        start = time.time()
        while time.time() - start < timeout:
            job = client.batches.get(name=batch_name)
            state = job.state.name if hasattr(job.state, "name") else str(job.state)
            elapsed = int(time.time() - start)
            log.info("Batch poll: %s state=%s (elapsed=%ds)", batch_name, state, elapsed)
            if state == "JOB_STATE_SUCCEEDED":
                return job
            if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                raise RuntimeError(f"Batch job {state}: {batch_name}")
            time.sleep(interval)
        raise TimeoutError(f"Batch job timed out after {timeout}s: {batch_name}")

    def _extract_batch_responses(self, batch_job) -> list:
        """Extract text responses from completed batch job."""
        responses = []
        if hasattr(batch_job, "dest") and batch_job.dest:
            if hasattr(batch_job.dest, "inlined_responses") and batch_job.dest.inlined_responses:
                for inline_resp in batch_job.dest.inlined_responses:
                    text = None
                    if hasattr(inline_resp, "response") and inline_resp.response:
                        if hasattr(inline_resp.response, "text"):
                            text = inline_resp.response.text
                        elif hasattr(inline_resp.response, "candidates"):
                            cands = inline_resp.response.candidates
                            if cands and cands[0].content and cands[0].content.parts:
                                text = cands[0].content.parts[0].text
                    responses.append(text)
        return responses

    # ------------------------------------------------------------------
    # numpy k-means
    # ------------------------------------------------------------------

    @staticmethod
    def _kmeans(embeddings: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means clustering using cosine similarity. Returns labels."""
        n = len(embeddings)
        if n == 0:
            return np.array([], dtype=int)
        k = min(k, n)
        if k <= 1:
            return np.zeros(n, dtype=int)

        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=k, replace=False)
        centroids = embeddings[indices].copy()
        labels = np.zeros(n, dtype=int)

        # Pre-normalize embeddings
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        e_norms[e_norms == 0] = 1
        e_normed = embeddings / e_norms

        for _ in range(max_iter):
            # Normalize centroids
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            c_norms[c_norms == 0] = 1
            c_normed = centroids / c_norms

            sims = e_normed @ c_normed.T  # (n, k)
            new_labels = np.argmax(sims, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = embeddings[mask].mean(axis=0)

        return labels

    # ------------------------------------------------------------------
    # Step 1: Stratified sampling
    # ------------------------------------------------------------------

    def _step1_sample(self) -> list[dict]:
        """Collect tagged sessions with stratified sampling by conversation_type."""
        self._phase = "sampling"
        recordings = Path(self._recordings_dir)
        if not recordings.is_dir():
            return []

        # Collect all tagged sessions grouped by conversation_type
        by_type: dict[str, list[dict]] = {}
        for day_dir in sorted(recordings.iterdir()):
            if not day_dir.is_dir():
                continue
            for time_dir in sorted(day_dir.iterdir()):
                if not time_dir.is_dir():
                    continue
                transcript_path = time_dir / "transcript.json"
                if not transcript_path.exists():
                    continue
                try:
                    with open(transcript_path, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                session = data.get("session", {})
                if not session.get("session_tagged"):
                    continue
                tags = session.get("tags", {})
                conv_type = tags.get("conversation_type", "")
                if not conv_type:
                    continue

                segments = data.get("segments", [])
                text = _extract_plain_text(segments)
                if len(text) < 10:
                    continue

                session_id = f"{day_dir.name}/{time_dir.name}"
                entry = {
                    "session_id": session_id,
                    "text": text,
                    "current_tags": {
                        "conversation_type": conv_type,
                        "outcomes": tags.get("outcomes", []),
                        "domains": tags.get("domains", []),
                        "importance": tags.get("importance", "低"),
                    },
                }
                by_type.setdefault(conv_type, []).append(entry)

        # Stratified sampling
        rng = np.random.RandomState(int(datetime.now().timestamp()) % (2**31))
        samples = []
        for conv_type, entries in by_type.items():
            n = max(1, math.ceil(len(entries) * self._sample_ratio))
            n = min(n, len(entries))
            chosen_idx = rng.choice(len(entries), size=n, replace=False)
            for idx in chosen_idx:
                samples.append(entries[idx])

        log.info("Step 1: sampled %d sessions from %d types (total %d tagged)",
                 len(samples), len(by_type),
                 sum(len(v) for v in by_type.values()))
        return samples

    # ------------------------------------------------------------------
    # Step 2: Cluster samples
    # ------------------------------------------------------------------

    def _get_embeddings_for_samples(self, samples: list[dict]) -> np.ndarray:
        """Get embeddings from RAG index or compute new ones."""
        # Build lookup from RAG index
        rag_lookup: dict[str, np.ndarray] = {}
        if self._rag and self._rag._embeddings is not None:
            with self._rag._lock:
                for i, meta in enumerate(self._rag._metadata):
                    rag_lookup[meta["session_id"]] = self._rag._embeddings[i]

        embeddings = []
        need_embed: list[tuple[int, str]] = []  # (index, text)

        for i, sample in enumerate(samples):
            sid = sample["session_id"]
            if sid in rag_lookup:
                embeddings.append(rag_lookup[sid])
            else:
                embeddings.append(None)
                need_embed.append((i, sample["text"]))

        # Batch embed missing ones
        if need_embed:
            log.info("Step 2: embedding %d samples not in RAG index", len(need_embed))
            batch_size = 100
            for start in range(0, len(need_embed), batch_size):
                chunk = need_embed[start:start + batch_size]
                texts = [t for _, t in chunk]
                try:
                    vecs = self._rag.embed_texts_batch(texts) if self._rag else self._embed_texts(texts)
                    for j, (idx, _) in enumerate(chunk):
                        embeddings[idx] = np.array(vecs[j], dtype=np.float32)
                except Exception as e:
                    log.error("Step 2: batch embed error: %s", e)
                    for idx, text in chunk:
                        try:
                            vec = self._rag.embed_text(text) if self._rag else self._embed_text(text)
                            embeddings[idx] = np.array(vec, dtype=np.float32)
                        except Exception:
                            # Use zero vector as fallback
                            embeddings[idx] = np.zeros(768, dtype=np.float32)

        return np.array(embeddings, dtype=np.float32)

    def _embed_text(self, text: str) -> list[float]:
        """Fallback embed single text if no RAG classifier."""
        client = self._get_client()
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return list(result.embeddings[0].values)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Fallback batch embed if no RAG classifier."""
        client = self._get_client()
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

    def _step2_cluster(self, samples: list[dict]) -> list[dict]:
        """Cluster samples by embedding similarity."""
        self._phase = "clustering"
        if not samples:
            return []

        embeddings = self._get_embeddings_for_samples(samples)

        # Determine k
        n = len(samples)
        k = max(3, int(math.sqrt(n)))
        k = min(k, 20, n)

        labels = self._kmeans(embeddings, k)

        # Build cluster summaries
        # Normalize embeddings for coherence calculation
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        e_norms[e_norms == 0] = 1
        e_normed = embeddings / e_norms

        clusters = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if not mask.any():
                continue

            cluster_indices = np.where(mask)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = cluster_embeddings.mean(axis=0)

            # Representative texts: closest to centroid
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid_normed = centroid / c_norm
                cluster_normed = e_normed[cluster_indices]
                sims_to_centroid = cluster_normed @ centroid_normed
                top_indices = np.argsort(sims_to_centroid)[::-1][:3]
            else:
                top_indices = list(range(min(3, len(cluster_indices))))

            representative_texts = []
            for idx in top_indices:
                orig_idx = cluster_indices[idx]
                text = samples[orig_idx]["text"][:200]
                representative_texts.append(text)

            # Domain and type distributions
            domain_counter = Counter()
            type_counter = Counter()
            for idx in cluster_indices:
                tags = samples[idx]["current_tags"]
                type_counter[tags.get("conversation_type", "")] += 1
                for d in tags.get("domains", []):
                    domain_counter[d] += 1

            # Coherence: mean pairwise cosine similarity within cluster
            coherence = 0.0
            if len(cluster_indices) > 1:
                cluster_norm = e_normed[cluster_indices]
                sim_matrix = cluster_norm @ cluster_norm.T
                n_c = len(cluster_indices)
                # Mean of upper triangle (excluding diagonal)
                total = 0.0
                count = 0
                for i in range(n_c):
                    for j in range(i + 1, n_c):
                        total += sim_matrix[i, j]
                        count += 1
                coherence = total / count if count > 0 else 0.0
            else:
                coherence = 1.0

            clusters.append({
                "cluster_id": cluster_id,
                "size": int(mask.sum()),
                "representative_texts": representative_texts,
                "current_domains": dict(domain_counter.most_common(10)),
                "current_types": dict(type_counter.most_common(10)),
                "coherence_score": round(float(coherence), 4),
            })

        log.info("Step 2: %d clusters from %d samples (k=%d)", len(clusters), n, k)
        return clusters

    # ------------------------------------------------------------------
    # Step 3: Propose taxonomy changes
    # ------------------------------------------------------------------

    def _build_proposal_prompt(self, clusters: list[dict], taxonomy: dict) -> str:
        """Build the prompt for taxonomy proposal generation."""
        cluster_summary = []
        for c in clusters:
            cluster_summary.append(
                f"クラスタ {c['cluster_id']} (件数={c['size']}, 凝集度={c['coherence_score']}):\n"
                f"  代表テキスト: {c['representative_texts']}\n"
                f"  conversation_type 分布: {c['current_types']}\n"
                f"  domain 分布: {c['current_domains']}"
            )

        return f"""あなたはタクソノミー(分類体系)の専門家です。
以下のセッションクラスタ分析結果と現行タクソノミーを比較し、改善提案を作成してください。

## 現行タクソノミー
```json
{json.dumps(taxonomy, ensure_ascii=False, indent=2)}
```

## クラスタ分析結果
{chr(10).join(cluster_summary)}

## 指示
- 各クラスタの代表テキストと現行タクソノミーのカテゴリを比較
- 以下の改善が必要か判断:
  - **split**: 1つのカテゴリに異質なセッションが混在 → 分割
  - **merge**: 複数カテゴリの内容が重複 → 統合
  - **new**: 既存カテゴリに当てはまらないクラスタ → 新規追加
  - **remove**: 使用されていないカテゴリ → 削除
- 変更が不要な場合は no_change: true を返す

## 出力形式 (JSON)
```json
{{
  "no_change": false,
  "changes": [
    {{
      "type": "split|merge|new|remove",
      "axis": "conversation_type|outcomes|domains",
      "description": "変更理由",
      "before": ["変更前カテゴリ"],
      "after": ["変更後カテゴリ"]
    }}
  ],
  "proposed_taxonomy": {{
    "conversation_type": ["..."],
    "outcomes": ["..."],
    "domains": ["..."],
    "keyword_anchors": {{}},
    "domain_descriptions": {{}}
  }}
}}
```

JSON のみを返してください。説明文は不要です。"""

    def _generate_content(self, model: str, prompt: str) -> str:
        """Direct (non-batch) API call for single request."""
        client = self._get_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text if hasattr(response, "text") else ""

    def _step3_propose(self, clusters: list[dict], taxonomy: dict) -> dict:
        """Generate taxonomy proposal via direct Gemini API call."""
        self._phase = "proposing"
        if not clusters:
            return {"no_change": True, "changes": [], "proposed_taxonomy": taxonomy}

        prompt = self._build_proposal_prompt(clusters, taxonomy)

        # Single request: use direct API (much faster than Batch API)
        log.info("Step 3: generating proposal via direct API (%s)", self._proposal_model)
        try:
            response_text = self._generate_content(self._proposal_model, prompt)
        except Exception as e:
            log.error("Step 3: direct API call failed: %s", e)
            return {"no_change": True, "changes": [], "proposed_taxonomy": taxonomy}

        if not response_text:
            log.warning("Step 3: empty response from proposal API")
            return {"no_change": True, "changes": [], "proposed_taxonomy": taxonomy}

        return self._parse_proposal_json(response_text, taxonomy)

    def _parse_proposal_json(self, text: str, fallback_taxonomy: dict) -> dict:
        """Parse LLM response JSON for proposal."""
        try:
            # Try to extract JSON from markdown code blocks or raw text
            cleaned = text.strip()
            if "```json" in cleaned:
                start = cleaned.index("```json") + 7
                end = cleaned.index("```", start)
                cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                start = cleaned.index("```") + 3
                end = cleaned.index("```", start)
                cleaned = cleaned[start:end].strip()

            result = json.loads(cleaned)

            # Validate required fields
            if not isinstance(result, dict):
                raise ValueError("Response is not a dict")
            result.setdefault("no_change", False)
            result.setdefault("changes", [])
            result.setdefault("proposed_taxonomy", fallback_taxonomy)

            return result
        except (json.JSONDecodeError, ValueError) as e:
            log.error("Step 3: failed to parse proposal JSON: %s", e)
            return {"no_change": True, "changes": [], "proposed_taxonomy": fallback_taxonomy}

    # ------------------------------------------------------------------
    # Step 4: Validate proposal
    # ------------------------------------------------------------------

    def _build_validation_prompts(self, samples: list[dict],
                                  proposed_taxonomy: dict) -> list[dict]:
        """Build per-session classification requests for validation."""
        taxonomy_json = json.dumps(proposed_taxonomy, ensure_ascii=False)
        requests = []
        for sample in samples:
            text = sample["text"][:500]  # Truncate for validation
            prompt = f"""以下のテキストを新しいタクソノミーで分類してください。

## タクソノミー
{taxonomy_json}

## テキスト
{text}

## 出力形式 (JSON)
{{"conversation_type": "...", "outcomes": ["..."], "domains": ["..."], "importance": "高|中|低"}}

JSON のみを返してください。"""
            requests.append({
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            })
        return requests

    def _step4_validate(self, samples: list[dict], proposed_taxonomy: dict) -> dict:
        """Validate proposed taxonomy by re-classifying samples."""
        self._phase = "validating"
        if not samples:
            return {
                "accuracy_vs_current": 1.0,
                "improved_count": 0, "degraded_count": 0, "unchanged_count": 0,
                "per_axis_scores": {},
            }

        requests = self._build_validation_prompts(samples, proposed_taxonomy)

        log.info("Step 4: submitting validation batch (%d requests)", len(requests))
        batch_name = self._submit_batch(self._validation_model, requests,
                                        display_name="taxonomy-validation")

        log.info("Step 4: polling batch job %s", batch_name)
        batch_job = self._poll_batch(batch_name)

        responses = self._extract_batch_responses(batch_job)
        return self._score_validation(samples, responses, proposed_taxonomy)

    def _score_validation(self, samples: list[dict], responses: list,
                          proposed_taxonomy: dict) -> dict:
        """Compare validation responses with current tags."""
        improved = 0
        degraded = 0
        unchanged = 0
        type_match = 0
        domain_match = 0
        total = 0

        valid_types = set(str(x) for x in proposed_taxonomy.get("conversation_type", [])
                         if isinstance(x, str))
        valid_domains = set(str(x) for x in proposed_taxonomy.get("domains", [])
                           if isinstance(x, str))

        for i, sample in enumerate(samples):
            if i >= len(responses) or not responses[i]:
                unchanged += 1
                total += 1
                continue

            try:
                text = responses[i].strip()
                if "```json" in text:
                    start = text.index("```json") + 7
                    end = text.index("```", start)
                    text = text[start:end].strip()
                elif "```" in text:
                    start = text.index("```") + 3
                    end = text.index("```", start)
                    text = text[start:end].strip()
                new_tags = json.loads(text)
                if not isinstance(new_tags, dict):
                    raise ValueError("not a dict")
            except (json.JSONDecodeError, ValueError, TypeError):
                unchanged += 1
                total += 1
                continue

            current = sample["current_tags"]
            total += 1

            # Check conversation_type match
            new_ct = new_tags.get("conversation_type", "")
            if isinstance(new_ct, list):
                new_ct = new_ct[0] if new_ct else ""
            cur_ct = current.get("conversation_type", "")
            ct_valid = isinstance(new_ct, str) and new_ct in valid_types

            # Check domains match (ensure list of strings)
            new_domains = new_tags.get("domains", [])
            if not isinstance(new_domains, list):
                new_domains = [new_domains] if isinstance(new_domains, str) else []
            new_domains = [str(d) for d in new_domains if isinstance(d, str)]
            cur_domains = set(str(d) for d in current.get("domains", []) if isinstance(d, str))
            new_domains_set = set(new_domains)
            domains_valid = all(d in valid_domains for d in new_domains)

            if ct_valid:
                type_match += 1
            if domains_valid and new_domains:
                domain_match += 1

            # Simple heuristic: if new classification is valid in proposed taxonomy
            # and differs from current, count as improved; if invalid, degraded
            if ct_valid and domains_valid:
                if new_ct != cur_ct or new_domains_set != cur_domains:
                    improved += 1
                else:
                    unchanged += 1
            elif not ct_valid or not domains_valid:
                degraded += 1
            else:
                unchanged += 1

        accuracy = (type_match / total) if total > 0 else 0.0

        return {
            "accuracy_vs_current": round(accuracy, 4),
            "improved_count": improved,
            "degraded_count": degraded,
            "unchanged_count": unchanged,
            "total_validated": total,
            "per_axis_scores": {
                "conversation_type": round(type_match / total, 4) if total > 0 else 0.0,
                "domains": round(domain_match / total, 4) if total > 0 else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Pipeline orchestrator
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full refinement pipeline (Steps 1-4).

        Returns a report dict. This is a blocking call (runs in executor).
        """
        if self._running:
            return {"error": "Already running"}

        self._running = True
        self._last_error = None
        started = datetime.now()
        report = {
            "started_at": started.isoformat(),
            "finished_at": None,
            "status": "running",
            "sample_count": 0,
            "cluster_count": 0,
            "proposal": None,
            "validation": None,
        }

        try:
            # Load current taxonomy
            taxonomy = self._load_taxonomy()

            # Step 1: Sample
            samples = self._step1_sample()
            report["sample_count"] = len(samples)
            if not samples:
                report["status"] = "no_data"
                report["finished_at"] = datetime.now().isoformat()
                self._save_report(report)
                return report

            # Step 2: Cluster
            clusters = self._step2_cluster(samples)
            report["cluster_count"] = len(clusters)
            report["clusters"] = [
                {k: v for k, v in c.items() if k != "representative_texts"}
                for c in clusters
            ]

            # Step 3: Propose
            proposal = self._step3_propose(clusters, taxonomy)
            report["proposal"] = {
                "no_change": proposal.get("no_change", False),
                "changes": proposal.get("changes", []),
            }

            if proposal.get("no_change"):
                report["status"] = "no_change"
                report["finished_at"] = datetime.now().isoformat()
                self._save_report(report)
                return report

            # Step 4: Validate
            validation = self._step4_validate(samples, proposal["proposed_taxonomy"])
            report["validation"] = validation

            # Save pending proposal
            pending = {
                "created_at": datetime.now().isoformat(),
                "proposed_taxonomy": proposal["proposed_taxonomy"],
                "changes": proposal.get("changes", []),
                "validation": validation,
                "sample_count": len(samples),
                "cluster_count": len(clusters),
            }
            self._save_pending(pending)

            report["status"] = "pending_approval"
            report["finished_at"] = datetime.now().isoformat()
            self._save_report(report)
            log.info("Taxonomy refinement complete: %d changes proposed", len(proposal.get("changes", [])))
            return report

        except Exception as e:
            log.error("Taxonomy refinement failed: %s", e, exc_info=True)
            self._last_error = str(e)
            report["status"] = "error"
            report["error"] = str(e)
            report["finished_at"] = datetime.now().isoformat()
            self._save_report(report)
            return report

        finally:
            self._running = False
            self._phase = None
            self._last_run = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Approval / Rejection
    # ------------------------------------------------------------------

    def get_pending_proposal(self) -> Optional[dict]:
        """Get the current pending taxonomy proposal, if any."""
        if not os.path.exists(self._pending_path):
            return None
        try:
            with open(self._pending_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def approve_proposal(self) -> dict:
        """Approve the pending proposal and update taxonomy."""
        pending = self.get_pending_proposal()
        if not pending:
            return {"error": "No pending proposal"}

        proposed = pending.get("proposed_taxonomy", {})
        if not proposed:
            return {"error": "Invalid proposal: no taxonomy"}

        # Load current taxonomy and update
        taxonomy = self._load_taxonomy()

        # Update axes
        for key in ("conversation_type", "outcomes", "domains"):
            if key in proposed and isinstance(proposed[key], list):
                taxonomy[key] = proposed[key]

        # Update keyword_anchors if provided
        if "keyword_anchors" in proposed and isinstance(proposed["keyword_anchors"], dict):
            taxonomy["keyword_anchors"] = proposed["keyword_anchors"]

        # Update domain_descriptions if provided
        if "domain_descriptions" in proposed and isinstance(proposed["domain_descriptions"], dict):
            taxonomy["domain_descriptions"] = proposed["domain_descriptions"]

        # Save updated taxonomy
        with open(self._taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, ensure_ascii=False, indent=2)

        # Remove pending
        self._remove_pending()

        # Update report status
        self._update_latest_report_status("approved")

        log.info("Taxonomy proposal approved and applied")
        return {"status": "approved", "taxonomy": taxonomy}

    def reject_proposal(self) -> dict:
        """Reject the pending proposal."""
        pending = self.get_pending_proposal()
        if not pending:
            return {"error": "No pending proposal"}

        self._remove_pending()
        self._update_latest_report_status("rejected")

        log.info("Taxonomy proposal rejected")
        return {"status": "rejected"}

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def get_reports(self) -> list[dict]:
        """Get list of past refinement reports (newest first)."""
        if not os.path.isdir(self._reports_dir):
            return []

        reports = []
        for fname in sorted(os.listdir(self._reports_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self._reports_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    report = json.load(f)
                report["filename"] = fname
                reports.append(report)
            except (json.JSONDecodeError, OSError):
                continue
        return reports

    def get_status(self) -> dict:
        """Get current refiner status."""
        return {
            "running": self._running,
            "phase": self._phase,
            "last_run": self._last_run,
            "last_error": self._last_error,
            "has_pending": os.path.exists(self._pending_path),
        }

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _load_taxonomy(self) -> dict:
        """Load current taxonomy from config."""
        if not os.path.exists(self._taxonomy_path):
            return {}
        try:
            with open(self._taxonomy_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_pending(self, data: dict) -> None:
        """Save pending taxonomy proposal."""
        with open(self._pending_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _remove_pending(self) -> None:
        """Remove pending taxonomy file."""
        if os.path.exists(self._pending_path):
            os.remove(self._pending_path)

    def _save_report(self, report: dict) -> None:
        """Save report to monthly file."""
        os.makedirs(self._reports_dir, exist_ok=True)
        month = datetime.now().strftime("%Y-%m")
        path = os.path.join(self._reports_dir, f"{month}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def _update_latest_report_status(self, status: str) -> None:
        """Update the status field in the latest report."""
        if not os.path.isdir(self._reports_dir):
            return
        files = sorted(os.listdir(self._reports_dir), reverse=True)
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self._reports_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    report = json.load(f)
                report["status"] = status
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                return
            except (json.JSONDecodeError, OSError):
                continue


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _extract_plain_text(segments: list[dict]) -> str:
    """Extract plain text from segments."""
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)
