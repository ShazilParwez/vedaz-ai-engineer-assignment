import argparse
import json
import logging
import re
import statistics
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Conversation:
    """Dataclass to hold conversation metadata during processing."""
    id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    is_valid: bool = True
    validation_error: str = ""
    flags: List[Dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    word_count: int = 0
    assistant_text: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


class Validator:
    """Handles basic structural validation of conversation data."""
    
    @staticmethod
    def validate_structure(conv_data: Dict[str, Any]) -> Tuple[bool, str, int, int, str]:
        """
        Validates chat structure and extracts basic metrics.
        Returns: (is_valid, error_msg, turn_count, word_count, combined_assistant_text)
        """
        if "id" not in conv_data or "messages" not in conv_data:
            return False, "Missing 'id' or 'messages' key", 0, 0, ""
        
        messages = conv_data["messages"]
        if not messages:
            return False, "Empty messages list", 0, 0, ""
            
        if messages[0].get("role") != "system":
            return False, "First message must be from 'system'", 0, 0, ""
            
        turn_count = len(messages)
        word_count = 0
        assistant_text_parts = []
        
        # Check alternating roles starting from index 1 (user, assistant, user, assistant...)
        expected_role = "user"
        for i in range(1, turn_count):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role != expected_role:
                return False, f"Expected role '{expected_role}' at index {i}, got '{role}'", 0, 0, ""
                
            word_count += len(content.split())
            if role == "assistant":
                assistant_text_parts.append(content)
                
            expected_role = "assistant" if expected_role == "user" else "user"
            
        return True, "", turn_count, word_count, " ".join(assistant_text_parts)


class SafetyChecker:
    """Detects violations of Vedaz safety policies using regex heuristics."""
    
    def __init__(self):
        # Dictionary mapping violation categories to regex patterns (English, Hindi, Hinglish)
        self.rules = {
            "death_prediction": re.compile(
                r"(die|death|maut|mrityu|marne|mar jaoge|jaan chali|life will end|lifespan is short)", 
                re.IGNORECASE
            ),
            "illness_prediction": re.compile(
                r"(cancer|disease|heart attack|bimari|gambhir rog|illness is guaranteed|bimar padoge)", 
                re.IGNORECASE
            ),
            "guarantees": re.compile(
                r"(100% guarantee|guaranteed|likh ke deta|pukka|definitely|certainty)", 
                re.IGNORECASE
            ),
            "money_guarantee": re.compile(
                r"(crorepati|lakhpati|rich soon|paisa hi paisa|ameer ban|lottery lag|financial certainty)", 
                re.IGNORECASE
            ),
            "marriage_pregnancy": re.compile(
                r"(shadi hogi hi|bachha hoga hi|definitely marry|guaranteed pregnancy|son is guaranteed)", 
                re.IGNORECASE
            ),
            "fear_selling": re.compile(
                r"(life ruined|tabah|barbaad|khatam|cursed|shrap|dosh lagega|kaal sarp.*tabah|manglik.*barbaad)", 
                re.IGNORECASE
            ),
            "expensive_rituals": re.compile(
                r"(pay \d+|51000|mehenga upay|buy gemstone|kharcha karna|black magic|kaala jadoo|tantrik puja|curses)", 
                re.IGNORECASE
            )
        }

    def check(self, assistant_text: str) -> List[Dict[str, str]]:
        """Scans assistant text for safety violations and returns offending excerpts."""
        flags = []
        for category, pattern in self.rules.items():
            match = pattern.search(assistant_text)
            if match:
                # Find the surrounding context (up to 30 chars around the match)
                start = max(0, match.start() - 30)
                end = min(len(assistant_text), match.end() + 30)
                context = assistant_text[start:end].replace('\n', ' ')
                flags.append({
                    "rule": category, 
                    "offending_text": f"...{context}..."
                })
        return flags


class DuplicateDetector:
    """Detects exact and semantic near-duplicates."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.model = None  # Lazy loading

    def _get_hash(self, text: str) -> str:
        """Returns MD5 hash for exact matching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def detect(self, conversations: List[Conversation]) -> List[str]:
        """Detects exact and semantic duplicates. Returns list of duplicate IDs to remove."""
        logger.info("Starting duplicate detection...")
        duplicates_to_remove = set()
        
        # 1. Exact Duplicates
        seen_hashes = {}
        for conv in conversations:
            if not conv.is_valid:
                continue
            
            conv_str = json.dumps(conv.raw_data.get("messages", []), sort_keys=True)
            conv_hash = self._get_hash(conv_str)
            
            if conv_hash in seen_hashes:
                logger.warning(f"Exact duplicate: {conv.id} is duplicate of {seen_hashes[conv_hash]}")
                duplicates_to_remove.add(conv.id)
            else:
                seen_hashes[conv_hash] = conv.id

        # 2. Semantic Duplicates using sentence-transformers
        valid_convs = [c for c in conversations if c.is_valid and c.id not in duplicates_to_remove]
        
        if not valid_convs:
            return list(duplicates_to_remove)

        logger.info("Loading sentence-transformers model for semantic deduplication...")
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        texts = [c.assistant_text for c in valid_convs]
        logger.info(f"Computing embeddings for {len(texts)} conversations...")
        embeddings = self.model.encode(texts)
        
        logger.info("Computing cosine similarities...")
        sim_matrix = cosine_similarity(embeddings)
        
        # Check upper triangular matrix
        for i in range(len(valid_convs)):
            for j in range(i + 1, len(valid_convs)):
                if sim_matrix[i][j] > self.threshold:
                    id1 = valid_convs[i].id
                    id2 = valid_convs[j].id
                    if id2 not in duplicates_to_remove: 
                        logger.warning(f"Semantic near-duplicate (sim: {sim_matrix[i][j]:.3f}): {id2} ~ {id1}")
                        duplicates_to_remove.add(id2)
                        
        return list(duplicates_to_remove)


class StatsCalculator:
    """Computes basic statistics on dataset elements."""
    
    @staticmethod
    def compute(conversations: List[Conversation]) -> Dict[str, Any]:
        """Computes descriptive statistics on valid conversations."""
        valid_convs = [c for c in conversations if c.is_valid]
        
        if not valid_convs:
            return {"error": "No valid conversations to analyze"}

        word_counts = [c.word_count for c in valid_convs]
        turn_counts = [c.turn_count for c in valid_convs]
        
        return {
            "total_conversations": len(valid_convs),
            "avg_word_count": round(statistics.mean(word_counts), 2),
            "min_word_count": min(word_counts),
            "max_word_count": max(word_counts),
            "avg_turns": round(statistics.mean(turn_counts), 2),
            "min_turns": min(turn_counts),
            "max_turns": max(turn_counts)
        }


def process_dataset(input_file: str, sim_threshold: float, test_size: float, random_seed: int):
    """Main orchestration function."""
    logger.info(f"Reading dataset from {input_file}...")
    conversations: List[Conversation] = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                data = json.loads(line)
                conv = Conversation(id=data.get("id", "unknown"), raw_data=data)
                
                # Validation
                is_valid, err_msg, turns, words, asst_text = Validator.validate_structure(data)
                conv.is_valid = is_valid
                conv.validation_error = err_msg
                conv.turn_count = turns
                conv.word_count = words
                conv.assistant_text = asst_text
                
                conversations.append(conv)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        return

    # Safety Check
    logger.info("Running safety checks...")
    safety_checker = SafetyChecker()
    flagged_chats = []
    
    for conv in conversations:
        if conv.is_valid:
            flags = safety_checker.check(conv.assistant_text)
            if flags:
                conv.flags = flags
                conv.is_valid = False # Exclude from valid split
                conv.validation_error = "Safety violation"
                flagged_chats.append({
                    "id": conv.id,
                    "violations": flags
                })
                logger.warning(f"Safety violation in {conv.id}: {[f['rule'] for f in flags]}")

    # Duplicate Detection
    dup_detector = DuplicateDetector(threshold=sim_threshold)
    duplicate_ids = dup_detector.detect(conversations)
    for conv in conversations:
        if conv.id in duplicate_ids:
            conv.is_valid = False
            conv.validation_error = "Duplicate"

    # Compute Statistics on the remaining clean dataset
    stats = StatsCalculator.compute([c for c in conversations if c.is_valid])
    
    # Write report
    report_path = "checker_report.txt"
    logger.info(f"Writing report to {report_path}...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Vedaz Dataset Validation Report ===\n\n")
        f.write(f"Total processed: {len(conversations)}\n")
        f.write(f"Valid and Clean: {stats.get('total_conversations', 0)}\n")
        f.write(f"Duplicates removed: {len(duplicate_ids)}\n")
        f.write(f"Safety Flags: {len(flagged_chats)}\n\n")
        
        f.write("--- Statistics (Clean Data) ---\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
            
        f.write("\n--- Malformed/Invalid Structure ---\n")
        for c in conversations:
            if not c.is_valid and c.validation_error not in ["Duplicate", "Safety violation"]:
                f.write(f"{c.id}: {c.validation_error}\n")
                
    # Write Flagged JSON
    flagged_path = "flagged_chats.json"
    logger.info(f"Writing flagged chats to {flagged_path}...")
    with open(flagged_path, "w", encoding="utf-8") as f:
        json.dump(flagged_chats, f, indent=2, ensure_ascii=False)

    # Train/Test Split
    clean_data = [c.raw_data for c in conversations if c.is_valid]
    
    if len(clean_data) > 1:
        logger.info(f"Splitting {len(clean_data)} items into train/test (test_size={test_size})...")
        train_data, test_data = train_test_split(
            clean_data, test_size=test_size, random_state=random_seed
        )
        
        with open("train.jsonl", "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        with open("test.jsonl", "w", encoding="utf-8") as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Successfully wrote train.jsonl and test.jsonl")
    else:
        logger.error("Not enough clean data to split into train and test sets.")


def main():
    parser = argparse.ArgumentParser(description="Vedaz AI Data Checker")
    parser.add_argument("--input", required=True, help="Input .jsonl file")
    parser.add_argument(
        "--sim_threshold", type=float, default=0.85, 
        help="Cosine similarity threshold for near-duplicates"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, 
        help="Proportion of data for test set"
    )
    parser.add_argument(
        "--seed", type=int, default=42, 
        help="Random seed for train/test split"
    )
    
    args = parser.parse_args()
    process_dataset(args.input, args.sim_threshold, args.test_size, args.seed)

if __name__ == "__main__":
    main()
