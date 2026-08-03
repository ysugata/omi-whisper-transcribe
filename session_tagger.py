"""Session-level tagging module using Gemini API.

Tags each recording session with 3-axis Japanese taxonomy:
- conversation_type (会話の種類): single-select
- outcomes (会話の成果物): multi-select
- domains (事業領域): multi-select
Plus entities, summary, decision_memo, quality, importance.
Uses OpenAI-compatible API (same as LLMCorrector).
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default taxonomy (hardcoded fallback)
# ---------------------------------------------------------------------------

_DEFAULT_TAXONOMY = {
    "version": 1,
    "conversation_type": [
        "商談", "社内ミーティング", "1on1", "アイデア出し",
        "顧客ヒアリング", "採用面接", "雑談・移動中メモ",
        "外部セミナー", "電話", "独り言・メモ", "その他",
    ],
    "outcomes": [
        "意思決定あり", "TODO発生", "方針転換", "情報共有のみ",
        "問題提起", "アイデア創出", "感情共有", "なし",
    ],
    "domains": [
        "水中超音波", "物販", "SNS集客", "観光業", "ダクト設備", "投資", "経営", "AI", "個人",
    ],
    "importance": ["高", "中", "低"],
}

_VALID_QUALITY = {"clean", "partial", "noisy", "fragmented"}


def _load_taxonomy(path: str) -> dict:
    """Load taxonomy from JSON file, falling back to defaults."""
    if not path or not os.path.exists(path):
        return dict(_DEFAULT_TAXONOMY)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "conversation_type" in data:
            return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load taxonomy from %s: %s", path, e)
    return dict(_DEFAULT_TAXONOMY)


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """あなたは音声文字起こしテキストの分類・要約者です。
セッションの全発話テキストが入力されます。以下の3軸で分類し、JSON で返してください。

出力フォーマット (JSON のみ、他のテキストは不要):
{{
  "conversation_type": "種類",
  "outcomes": ["成果物1", "成果物2"],
  "domains": ["事業領域1"],
  "entities": ["Entity1", "Entity2"],
  "summary": "日本語で1-2文の要約",
  "decision_memo": "何が決まったか (80文字以内、なければnull)",
  "quality": "clean|partial|noisy|fragmented",
  "importance": "高|中|低"
}}

### Axis 1: conversation_type (会話の種類、1つだけ選択、必須)
{conversation_type_list}

### Axis 2: outcomes (会話の成果物、1〜4つ選択、必須)
{outcomes_list}

### Axis 3: domains (事業領域、0〜3つ選択、任意)
{domains_list}

#### domains 選択ルール:
- 録音者自身の事業活動に関連する話題のみ選択する
- TV・ラジオ・店舗BGM等の外部音声は、たまたま事業キーワードが含まれていてもドメインに含めない
- unknown話者が大半の場合、外部メディアの音声である可能性が高いため、ドメイン選択は慎重に（確信がなければ空配列）
- 文字起こし品質が低い（noisy/fragmented）場合も、キーワードだけでドメインを推定しない
- 短い発話（3セグメント以下）で文脈が不十分な場合、ドメインは空配列を推奨
- 「経営」: 特定事業に属さない全社横断の意思決定・財務・組織の話題のみ
- 「AI」: AI技術そのものが主題の話題のみ。既存事業のAI活用は事業ラインを選択
- 「個人」: 業務外の個人的な話題（家族、健康、趣味等）
- 複数事業にまたがる場合は該当する全事業を選択（最大3つ）
- 該当するドメインがない場合は空配列 [] を返す
- キーワード走査結果が提示された場合、候補ドメインを優先的に検討してください。ただし候補外のドメインも正当な理由があれば選択可能です

### entities
テキスト中に登場する固有名詞 (人名、地名、サービス名、技術名など) をそのまま抽出。なければ空配列。

### summary
日本語で1-2文。何について話していたかの概要。

### decision_memo
「意思決定あり」を outcomes に含めた場合のみ、何が決まったかを日本語80文字以内で記述。該当なしなら null。

### quality (英語 enum、1つ)
clean (明瞭で完全), partial (部分的に不明瞭), noisy (ノイズ多い), fragmented (断片的)

### importance (日本語、1つ)
{importance_list}

#### importance 判定基準:
- 高: 事業判断に直結する内容、重要な意思決定
- 中: 参考情報、進捗確認、定例的な打ち合わせ
- 低: 雑談、移動中のメモ、個人的な話題

### 話者情報の解釈
入力テキストに [話者情報] と [名前] / [unknown] のプレフィックスがある場合:
- [自分] = 録音者本人の発話
- [名前(自分)] = 同上（名前付き）
- [名前] = 登録済みの他者の発話
- [unknown] = 未登録話者の発話（TV・ラジオ・通行人等の外部音声の可能性あり）
- unknown比率が高い場合、会話の種類は「独り言・メモ」ではなく実際の文脈で判断
- 全セグメントがunknownの場合、外部メディア（TV等）の音声が主体の可能性が高い

### タグ数制限
conversation_type (1) + outcomes (1-4) + domains (0-3) の合計が5以下になるようにしてください。"""


class SessionTagger:
    """Tags a session with 3-axis Japanese taxonomy."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0,
        max_tokens: int = 768,
        timeout: int = 15,
        taxonomy_path: str = "",
        prefilter_enabled: bool = True,
        prefilter_min_matches: int = 1,
        rag_classifier=None,
        reasoning_effort: str = "minimal",
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._client = None
        self._taxonomy_path = taxonomy_path
        self._taxonomy = _load_taxonomy(taxonomy_path)
        self._prefilter_enabled = prefilter_enabled
        self._prefilter_min_matches = max(1, prefilter_min_matches)

        self._rag = rag_classifier

        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                log.info(f"SessionTagger initialized: model={model}")
            except Exception as e:
                log.error(f"SessionTagger init failed: {e}")
                self._enabled = False

    def reload_taxonomy(self) -> None:
        """Reload taxonomy from file."""
        self._taxonomy = _load_taxonomy(self._taxonomy_path)
        log.info("SessionTagger: taxonomy reloaded (%d types, %d outcomes, %d domains)",
                 len(self._taxonomy.get("conversation_type", [])),
                 len(self._taxonomy.get("outcomes", [])),
                 len(self._taxonomy.get("domains", [])))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def taxonomy(self) -> dict:
        return self._taxonomy

    def _build_system_prompt(self) -> str:
        """Build the system prompt with taxonomy values injected."""
        t = self._taxonomy
        # Build domains list with descriptions and keyword anchors if available
        descriptions = t.get("domain_descriptions", {})
        anchors = t.get("keyword_anchors", {})
        domains = t.get("domains", [])
        if descriptions:
            domain_lines = []
            for d in domains:
                desc = descriptions.get(d, "")
                if desc:
                    domain_lines.append(f'- "{d}": {desc}')
                else:
                    domain_lines.append(f'- "{d}"')
                kw = anchors.get(d)
                if kw:
                    domain_lines.append(f'  関連キーワード: {", ".join(kw)}')
            domains_list = "\n".join(domain_lines)
        else:
            domains_list = ", ".join(f'"{v}"' for v in domains)

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            conversation_type_list=", ".join(f'"{v}"' for v in t.get("conversation_type", [])),
            outcomes_list=", ".join(f'"{v}"' for v in t.get("outcomes", [])),
            domains_list=domains_list,
            importance_list=", ".join(f'"{v}"' for v in t.get("importance", ["高", "中", "低"])),
        )
        return prompt

    @staticmethod
    def _compute_speaker_stats(segments: list[dict]) -> dict:
        """Compute speaker statistics from segments."""
        total = len(segments)
        if total == 0:
            return {
                "total": 0, "user": 0, "unknown": 0, "unknown_ratio": 0.0,
                "has_voice_tags": False, "voice_quality": 0.0,
            }
        has_voice = any(s.get("person_name") for s in segments)
        user = sum(1 for s in segments if s.get("is_user"))
        unknown = sum(1 for s in segments if s.get("person_name") == "unknown")
        return {
            "total": total,
            "user": user,
            "unknown": unknown,
            "unknown_ratio": unknown / total if total > 0 else 0.0,
            "has_voice_tags": has_voice,
            "voice_quality": SessionTagger._compute_voiceprint_quality(segments),
        }

    @staticmethod
    def _compute_voiceprint_quality(segments: list[dict]) -> float:
        """Estimate how well the voiceprint model is identifying speakers
        in this session.

        Returns a 0..1 score based on the distribution of `speaker_similarity`
        values across segments. High p75 similarity → voiceprint clearly
        identifies the user. Near-zero p75 → voiceprint output is essentially
        noise (e.g., stale enrollment, wrong model, no enrolled speakers).

        Mapping (cosine similarity → quality):
          p75 >= 0.50  → 1.0  (definitely working)
          p75 == 0.25  → 0.0  (noise floor)
          linearly interpolated between
        """
        sims = [
            s.get("speaker_similarity") for s in segments
            if s.get("speaker_similarity") is not None
        ]
        if not sims:
            return 0.0
        sims_sorted = sorted(sims)
        p75_idx = max(0, int(round(len(sims_sorted) * 0.75)) - 1)
        p75 = sims_sorted[p75_idx]
        # Map [0.25, 0.50] → [0, 1]
        if p75 <= 0.25:
            return 0.0
        if p75 >= 0.50:
            return 1.0
        return (p75 - 0.25) / 0.25

    @staticmethod
    def _compute_external_audio_ratio(
        stats: dict,
        audio_scene: dict | None,
    ) -> float | None:
        """Decide the external_audio_ratio for a session by blending the
        voiceprint signal with the PANNs scene classification.

        Logic:
          - voice_quality high (>= 0.7): trust voiceprint → unknown_ratio
          - voice_quality low (<= 0.3):  trust PANNs → 1.0 if media else 0.0
          - in-between: weighted blend so the more reliable signal dominates

        This is robust against stale/broken voiceprint enrollments — when the
        cosine similarities are essentially noise, we fall back to PANNs and
        avoid the catastrophic "tag everything as TV" failure mode.

        Returns None if neither signal is available.
        """
        has_voice = stats.get("has_voice_tags", False)
        voice_quality = stats.get("voice_quality", 0.0)
        unknown_ratio = stats.get("unknown_ratio", 0.0)

        # Extract PANNs media signal
        pann_media = None
        if audio_scene:
            scene_en = audio_scene.get("scene_label_en")
            if scene_en in ("media", "conversation", "telephone", "silence", "noise"):
                pann_media = 1.0 if scene_en == "media" else 0.0

        # Neither signal available
        if not has_voice and pann_media is None:
            return None

        # Voiceprint unavailable → trust PANNs alone
        if not has_voice:
            return pann_media

        # PANNs unavailable → trust voiceprint alone
        if pann_media is None:
            return round(unknown_ratio, 2)

        # Both available — blend based on voiceprint quality
        # voice_quality 1.0 → fully voiceprint, 0.0 → fully PANNs
        w_voice = max(0.0, min(1.0, voice_quality))
        w_pann = 1.0 - w_voice
        blended = w_voice * unknown_ratio + w_pann * pann_media
        return round(blended, 2)

    @staticmethod
    def _apply_monologue_heuristic(tags: dict, stats: dict) -> dict:
        """Override conversation_type based on speaker ratio.

        If voice tags show user_ratio >= 0.8 (mostly self),
        override '雑談・移動中メモ' → '独り言・メモ'.
        If user_ratio < 0.4 AND user > 0, override '独り言・メモ' → '雑談・移動中メモ'.
        Requires user > 0 to ensure voice tagging is actually working.
        """
        if not stats["has_voice_tags"] or stats["total"] == 0:
            return tags
        if stats["user"] == 0:
            return tags
        user_ratio = stats["user"] / stats["total"]
        ct = tags.get("conversation_type", "")
        if ct == "雑談・移動中メモ" and user_ratio >= 0.8:
            tags["conversation_type"] = "独り言・メモ"
            log.debug("Monologue heuristic: 雑談→独り言 (user_ratio=%.2f)", user_ratio)
        elif ct == "独り言・メモ" and user_ratio < 0.4:
            tags["conversation_type"] = "雑談・移動中メモ"
            log.debug("Monologue heuristic: 独り言→雑談 (user_ratio=%.2f)", user_ratio)
        return tags

    def _build_input(self, segments: list[dict], audio_scene: dict | None = None) -> str:
        """Build session text from segments with speaker info."""
        stats = self._compute_speaker_stats(segments)

        lines = []
        # Add audio scene preamble if available
        if audio_scene and audio_scene.get("scene_label"):
            scene_conf = int(audio_scene.get("scene_confidence", 0) * 100)
            scene_line = f"[音響環境 (PANNs推定)] シーン: {audio_scene['scene_label']} ({scene_conf}%)"
            if audio_scene.get("spatial_label"):
                spatial_conf = int(audio_scene.get("spatial_confidence", 0) * 100)
                scene_line += f", 空間: {audio_scene['spatial_label']} ({spatial_conf}%)"
            lines.append(scene_line)

            # Noise sources (top 3)
            noise_sources = audio_scene.get("noise_sources", [])
            if noise_sources:
                ns_str = ", ".join(
                    f"{ns['label']}({int(ns['confidence'] * 100)}%)" for ns in noise_sources[:3]
                )
                lines.append(f"  ノイズ源: {ns_str}")

            # Events (top 3)
            events = audio_scene.get("events", [])
            if events:
                ev_str = ", ".join(
                    f"{ev['label']}({int(ev['confidence'] * 100)}%)" for ev in events[:3]
                )
                lines.append(f"  検出イベント: {ev_str}")

            # Speaker count estimate
            sc = audio_scene.get("speaker_count")
            if sc and sc.get("estimate"):
                lines.append(f"  話者数推定: {sc['label']}({int(sc['confidence'] * 100)}%)")

            # Music detection
            music = audio_scene.get("music")
            if music and music.get("detected"):
                music_str = f"  音楽検出: あり({int(music['confidence'] * 100)}%)"
                if music.get("type_label"):
                    music_str += f" [{music['type_label']}]"
                lines.append(music_str)

            # Quality issues
            quality = audio_scene.get("quality_issues", [])
            if quality:
                q_str = ", ".join(
                    f"{q['label']}({int(q['confidence'] * 100)}%)" for q in quality[:3]
                )
                lines.append(f"  品質: {q_str}")

            lines.append("")

        # Add speaker statistics preamble if voice tags exist
        if stats["has_voice_tags"]:
            preamble = (
                f"[話者情報] 全{stats['total']}発話: "
                f"自分={stats['user']}, unknown={stats['unknown']}"
            )
            if stats["unknown_ratio"] >= 0.8:
                preamble += " ※ unknown話者が大半 — TV・ラジオ等の外部音声の可能性が高い"
            lines.append(preamble)
            lines.append("")

        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            person = seg.get("person_name", "")
            if person and person != "unknown":
                is_user = seg.get("is_user", False)
                label = f"{person}(自分)" if is_user else person
                lines.append(f"[{label}] {text}")
            elif person == "unknown":
                lines.append(f"[unknown] {text}")
            else:
                lines.append(text)
        return "\n".join(lines)

    def _scan_domain_keywords(self, text: str) -> dict:
        """Scan text for keyword_anchors and return candidate domains.

        Returns {"candidates": [{"domain": str, "hits": int, "total": int, "score": float}],
                 "non_candidates": [str]}
        """
        anchors = self._taxonomy.get("keyword_anchors", {})
        domains = self._taxonomy.get("domains", [])
        if not anchors:
            return {"candidates": [], "non_candidates": list(domains)}

        text_lower = text.lower()
        candidates = []
        non_candidates = []

        for domain in domains:
            kw_list = anchors.get(domain)
            if not kw_list:
                non_candidates.append(domain)
                continue
            hits = sum(1 for kw in kw_list if kw.lower() in text_lower)
            if hits >= self._prefilter_min_matches:
                candidates.append({
                    "domain": domain,
                    "hits": hits,
                    "total": len(kw_list),
                    "score": round(hits / len(kw_list), 2),
                })
            else:
                non_candidates.append(domain)

        candidates.sort(key=lambda c: c["hits"], reverse=True)
        return {"candidates": candidates, "non_candidates": non_candidates}

    def _format_prefilter_hint(self, scan_result: dict) -> str:
        """Format keyword scan result as a hint string for the user message."""
        candidates = scan_result.get("candidates", [])
        non_candidates = scan_result.get("non_candidates", [])

        if candidates:
            cand_parts = [f"{c['domain']}({c['hits']}hit)" for c in candidates]
            hint = f"\n\n[キーワード走査結果] 候補: {', '.join(cand_parts)}"
            if non_candidates:
                hint += f" / 非候補: {', '.join(non_candidates)}"
            return hint
        else:
            return "\n\n[キーワード走査結果] 候補なし — 全ドメインから判定してください"

    def _parse_response(self, raw: str) -> Optional[dict]:
        """Parse JSON response from LLM."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"SessionTagger: invalid JSON response: {raw[:200]}")
            return None

        if not isinstance(data, dict):
            log.warning(f"SessionTagger: response is not a dict")
            return None

        return data

    def _validate_tags(self, tags: dict) -> dict:
        """Validate and normalize tag values against taxonomy."""
        t = self._taxonomy
        result = {}

        valid_conv_types = set(t.get("conversation_type", []))
        valid_outcomes = set(t.get("outcomes", []))
        valid_domains = set(t.get("domains", []))
        valid_importance = set(t.get("importance", ["高", "中", "低"]))

        # conversation_type: single-select, required
        conv_type = str(tags.get("conversation_type", "")).strip()
        if conv_type not in valid_conv_types:
            conv_type = "その他" if "その他" in valid_conv_types else (
                list(valid_conv_types)[0] if valid_conv_types else "その他"
            )
        result["conversation_type"] = conv_type

        # outcomes: multi-select, required, 1-4
        outcomes_raw = tags.get("outcomes", [])
        if isinstance(outcomes_raw, list):
            outcomes = [str(o).strip() for o in outcomes_raw if str(o).strip() in valid_outcomes]
        else:
            outcomes = []
        if not outcomes:
            outcomes = ["なし"] if "なし" in valid_outcomes else []
        result["outcomes"] = outcomes[:4]

        # domains: multi-select, optional, 0-3
        domains_raw = tags.get("domains", [])
        if isinstance(domains_raw, list):
            domains = [str(d).strip() for d in domains_raw if str(d).strip() in valid_domains]
        else:
            domains = []
        result["domains"] = domains[:3]

        # 5-tag limit enforcement: conv_type(1) + outcomes + domains <= 5
        total_tags = 1 + len(result["outcomes"]) + len(result["domains"])
        while total_tags > 5 and result["domains"]:
            result["domains"].pop()
            total_tags -= 1
        while total_tags > 5 and len(result["outcomes"]) > 1:
            result["outcomes"].pop()
            total_tags -= 1

        # entities: list of strings (preserve case)
        entities = tags.get("entities", [])
        if isinstance(entities, list):
            result["entities"] = [str(e).strip() for e in entities[:20] if e]
        else:
            result["entities"] = []

        # summary: string (Japanese)
        summary = str(tags.get("summary", "")).strip()
        result["summary"] = summary[:200] if summary else ""

        # decision_memo: string or None (80 chars max)
        dm = tags.get("decision_memo")
        if dm and isinstance(dm, str) and dm.strip():
            result["decision_memo"] = dm.strip()[:80]
        else:
            result["decision_memo"] = None

        # quality: enum (English, internal use)
        quality = str(tags.get("quality", "clean")).lower().strip()
        result["quality"] = quality if quality in _VALID_QUALITY else "clean"

        # importance: Japanese
        importance = str(tags.get("importance", "低")).strip()
        result["importance"] = importance if importance in valid_importance else "低"

        return result

    def tag_session_rag_only(self, segments: list[dict], audio_scene: dict | None = None) -> Optional[dict]:
        """Try RAG classification only (no LLM API call).

        Returns tags if RAG is confident, None otherwise.
        Used by batch pipeline to separate RAG-handled sessions from LLM-needed ones.
        """
        if not self._enabled or not segments:
            return None

        if self._rag and self._rag.is_ready():
            try:
                from rag_classifier import RagClassifier
                plain_text = RagClassifier._extract_plain_text(segments)
                rag_result = self._rag.classify(plain_text)
            except Exception as e:
                log.warning("RAG classify failed: %s", e)
                return None

            if rag_result and rag_result["confident"]:
                tags = self._validate_tags(rag_result["tags"])
                tags["tagged_at"] = datetime.now().isoformat()
                tags["tag_version"] = 2
                tags["tag_source"] = "rag"
                tags["rag_similarity"] = round(rag_result["top_similarity"], 4)
                stats = self._compute_speaker_stats(segments)
                ext_ratio = self._compute_external_audio_ratio(stats, audio_scene)
                tags["external_audio_ratio"] = ext_ratio
                if ext_ratio is not None and ext_ratio >= 0.8:
                    tags["domains"] = []
                if stats["has_voice_tags"] and stats.get("voice_quality", 0.0) >= 0.5:
                    tags = self._apply_monologue_heuristic(tags, stats)
                tags.setdefault("summary", "")
                tags.setdefault("entities", [])
                tags.setdefault("decision_memo", None)
                log.info("SessionTagger[RAG]: type=%s, domains=%s, sim=%.3f",
                         tags["conversation_type"], tags["domains"], rag_result["top_similarity"])
                return tags

        return None

    def tag_session(self, segments: list[dict], audio_scene: dict | None = None) -> Optional[dict]:
        """Tag a session's segments and return tags dict.

        Uses RAG classification first if available and confident.
        Falls back to LLM on ambiguous results or when RAG is not ready.
        Returns None on error (API failure, invalid response, etc.).
        """
        if not self._enabled:
            return None

        if not segments:
            return None

        session_text = self._build_input(segments, audio_scene=audio_scene)
        if not session_text or len(session_text) < 10:
            log.debug("SessionTagger: session text too short, skipping")
            return None

        # --- Phase 1: RAG classification ---
        rag_result = None
        if self._rag and self._rag.is_ready():
            try:
                from rag_classifier import RagClassifier
                plain_text = RagClassifier._extract_plain_text(segments)
                rag_result = self._rag.classify(plain_text)
            except Exception as e:
                log.warning("RAG classify failed: %s", e)

        if rag_result and rag_result["confident"]:
            tags = self._validate_tags(rag_result["tags"])
            tags["tagged_at"] = datetime.now().isoformat()
            tags["tag_version"] = 2
            tags["tag_source"] = "rag"
            tags["rag_similarity"] = round(rag_result["top_similarity"], 4)
            # Apply speaker-derived metadata
            stats = self._compute_speaker_stats(segments)
            ext_ratio = self._compute_external_audio_ratio(stats, audio_scene)
            tags["external_audio_ratio"] = ext_ratio
            if ext_ratio is not None and ext_ratio >= 0.8:
                tags["domains"] = []
            # Monologue heuristic: user_ratio >= 0.8 → 独り言・メモ
            # (only meaningful when voiceprint is reliable)
            if stats["has_voice_tags"] and stats.get("voice_quality", 0.0) >= 0.5:
                tags = self._apply_monologue_heuristic(tags, stats)
            # RAG does not produce summary/entities/decision_memo
            tags.setdefault("summary", "")
            tags.setdefault("entities", [])
            tags.setdefault("decision_memo", None)
            log.info("SessionTagger[RAG]: type=%s, domains=%s, sim=%.3f",
                     tags["conversation_type"], tags["domains"], rag_result["top_similarity"])
            return tags

        # --- Phase 2: LLM fallback ---
        # Domain prefilter: scan keywords and inject hint into user message
        prefilter_hint = ""
        if self._prefilter_enabled:
            scan_result = self._scan_domain_keywords(session_text)
            prefilter_hint = self._format_prefilter_hint(scan_result)
            if scan_result["candidates"]:
                log.debug("SessionTagger prefilter: %s",
                          ", ".join(c["domain"] for c in scan_result["candidates"]))

        try:
            system_prompt = self._build_system_prompt()
            kwargs = dict(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": session_text + prefilter_hint},
                ],
            )
            if self._model.startswith("gpt-"):
                # gpt-5 系: max_tokens/temperature は拒否される。思考トークンは
                # 出力課金なので reasoning_effort で最小化 (llm_pipeline と同様)。
                kwargs["max_completion_tokens"] = self._max_tokens
                kwargs["reasoning_effort"] = self._reasoning_effort
            else:
                kwargs["temperature"] = self._temperature
                kwargs["max_tokens"] = self._max_tokens
            response = self._client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            log.debug(f"SessionTagger raw response: {raw[:300]}")

            parsed = self._parse_response(raw)
            if parsed is None:
                return None

            tags = self._validate_tags(parsed)
            tags["tagged_at"] = datetime.now().isoformat()
            tags["tag_version"] = 2
            tags["tag_source"] = "llm"

            # Add speaker-derived metadata
            stats = self._compute_speaker_stats(segments)
            ext_ratio = self._compute_external_audio_ratio(stats, audio_scene)
            tags["external_audio_ratio"] = ext_ratio
            # Force empty domains for external audio sessions
            if ext_ratio is not None and ext_ratio >= 0.8:
                tags["domains"] = []

            log.info(
                f"SessionTagger[LLM]: type={tags['conversation_type']}, "
                f"outcomes={tags['outcomes']}, "
                f"domains={tags['domains']}, "
                f"summary={tags['summary'][:50]}"
            )
            return tags

        except Exception as e:
            log.warning(f"SessionTagger API error: {e}")
            return None

    def build_batch_input(self, segments: list[dict], audio_scene: dict | None = None) -> tuple:
        """Build (system_prompt, user_message) for Batch API submission.

        Returns ("", "") if input is too short.
        """
        session_text = self._build_input(segments, audio_scene=audio_scene)
        if not session_text or len(session_text) < 10:
            return ("", "")

        # Domain prefilter: scan keywords and inject hint
        if self._prefilter_enabled:
            scan_result = self._scan_domain_keywords(session_text)
            session_text += self._format_prefilter_hint(scan_result)

        system_prompt = self._build_system_prompt()
        return (system_prompt, session_text)

    def validate_and_save_tags(self, raw_json: str, segments: list[dict] | None = None, audio_scene: dict | None = None) -> Optional[dict]:
        """Parse and validate raw JSON from Batch API response.

        Returns validated tags dict with tagged_at, or None on parse error.
        """
        parsed = self._parse_response(raw_json)
        if parsed is None:
            return None
        tags = self._validate_tags(parsed)
        tags["tagged_at"] = datetime.now().isoformat()
        tags["tag_version"] = 2

        # Add speaker-derived metadata if segments provided
        if segments:
            stats = self._compute_speaker_stats(segments)
            ext_ratio = self._compute_external_audio_ratio(stats, audio_scene)
            tags["external_audio_ratio"] = ext_ratio
            # Force empty domains for external audio sessions
            if ext_ratio is not None and ext_ratio >= 0.8:
                tags["domains"] = []
        else:
            tags["external_audio_ratio"] = None

        return tags

    # ------------------------------------------------------------------
    # Batch API support
    # ------------------------------------------------------------------

    def build_batch_request(
        self,
        segments: list[dict],
        audio_scene: dict | None = None,
    ) -> dict | None:
        """Build a Gemini Batch API request for a single session.

        Returns a dict suitable for genai.Client.batches.create(src=[...]),
        or None if the session should be skipped (RAG confident, too short, etc.).
        """
        if not self._enabled or not segments:
            return None

        session_text = self._build_input(segments, audio_scene=audio_scene)
        if not session_text or len(session_text) < 10:
            return None

        # If RAG is confident, no need for Batch API
        if self._rag and self._rag.is_ready():
            try:
                from rag_classifier import RagClassifier
                plain_text = RagClassifier._extract_plain_text(segments)
                rag_result = self._rag.classify(plain_text)
                if rag_result and rag_result["confident"]:
                    return None  # RAG handles this session
            except Exception:
                pass

        prefilter_hint = ""
        if self._prefilter_enabled:
            scan_result = self._scan_domain_keywords(session_text)
            prefilter_hint = self._format_prefilter_hint(scan_result)

        system_prompt = self._build_system_prompt()
        user_message = session_text + prefilter_hint

        return {
            "contents": [{
                "parts": [{"text": user_message}],
                "role": "user",
            }],
            "config": {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "response_mime_type": "application/json",
            },
        }

    def parse_batch_response(
        self,
        raw_text: str,
        segments: list[dict],
        audio_scene: dict | None = None,
    ) -> dict | None:
        """Parse a Batch API response and return finalized tags.

        Same post-processing as tag_session (validation, speaker stats, etc.).
        """
        parsed = self._parse_response(raw_text)
        if parsed is None:
            return None

        tags = self._validate_tags(parsed)
        tags["tagged_at"] = datetime.now().isoformat()
        tags["tag_version"] = 2
        tags["tag_source"] = "llm"

        stats = self._compute_speaker_stats(segments)
        ext_ratio = self._compute_external_audio_ratio(stats, audio_scene)
        tags["external_audio_ratio"] = ext_ratio
        if ext_ratio is not None and ext_ratio >= 0.8:
            tags["domains"] = []

        if stats["has_voice_tags"] and stats.get("voice_quality", 0.0) >= 0.5:
            tags = self._apply_monologue_heuristic(tags, stats)

        return tags
