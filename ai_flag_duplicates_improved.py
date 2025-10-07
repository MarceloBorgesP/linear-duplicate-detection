import json
import openai
import os
import chromadb
from chromadb.utils import embedding_functions
import requests
import re
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
from dotenv import load_dotenv
import csv

# Load environment variables from .env file
load_dotenv()

# Only import tqdm if available (optional dependency)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Set up logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Reduce noise from HTTP libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration settings for the duplicate detection system."""
    openai_api_key: str
    linear_api_key: str
    linear_team_id: str
    chroma_db_path: Path
    chroma_collection_name: str = "linear_bugs"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"  # More cost-effective than gpt-4-1106-preview
    issues_json_path: Path = None
    confidence_threshold: int = 65
    max_candidates: int = 10
    dry_run: bool = False
    rate_limit_delay: float = 0.1  # Delay between API calls in seconds
    batch_size: int = 100  # For ChromaDB operations
    output_file: Path = None  # File to write actions instead of Linear API
    
    @classmethod
    def from_env(cls, issues_json_path: str = None, dry_run: bool = False) -> 'Config':
        """Create config from environment variables."""
        openai_key = os.environ.get('CHROMA_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        linear_key = os.environ.get('LINEAR_OAUTH_ACCESS_TOKEN') or os.environ.get('LINEAR_API_KEY')
        
        if not openai_key or not linear_key:
            raise ValueError(
                "API keys must be set as environment variables:\n"
                "- CHROMA_OPENAI_API_KEY or OPENAI_API_KEY\n"
                "- LINEAR_OAUTH_ACCESS_TOKEN or LINEAR_API_KEY"
            )
        
        # Use cross-platform paths - save in current working directory
        default_db_path = Path.cwd() / ".chroma" / "bug-bounty"
        default_issues_path = Path.cwd() / "linear_issues.json"
        
        return cls(
            openai_api_key=openai_key,
            linear_api_key=linear_key,
            linear_team_id=os.environ.get('LINEAR_TEAM_ID', 'd9c4b135-c5f6-4eec-a504-085f1c4e13dc'),
            chroma_db_path=Path(os.environ.get('CHROMA_DB_PATH', default_db_path)),
            issues_json_path=Path(issues_json_path) if issues_json_path else default_issues_path,
            dry_run=dry_run,
            confidence_threshold=int(os.environ.get('CONFIDENCE_THRESHOLD', '65')),
            max_candidates=int(os.environ.get('MAX_CANDIDATES', '10')),
            output_file=Path.cwd() / "linear_actions_output.csv"
        )

class LinearAPI:
    """Handles all interactions with the Linear GraphQL API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_url = "https://api.linear.app/graphql"
        self.headers = {
            "Authorization": self.config.linear_api_key,  # Linear expects just the API key, no "Bearer" prefix
            "Content-Type": "application/json"
        }
        self._request_count = 0
        self._actions_log = []  # Store actions to write to file

    def write_actions_to_file(self):
        """Write all logged actions to a CSV file."""
        if not self._actions_log:
            logger.info("No actions to write to file")
            return
        
        # Ensure output directory exists
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['issue_identifier', 'matches_summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for action in self._actions_log:
                writer.writerow(action)
        
        logger.info(f"Wrote {len(self._actions_log)} actions to {self.config.output_file}")

    def write_single_action_to_file(self, action: Dict[str, Any]):
        """Write a single action to the CSV file immediately."""
        # Ensure output directory exists
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists to determine if we need to write header
        file_exists = self.config.output_file.exists()
        
        with open(self.config.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['issue_identifier', 'matches_summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(action)
        
        logger.debug(f"Wrote action to file for {action['issue_identifier']}")

    def initialize_output_file(self):
        """Initialize the output file by clearing it and writing the header."""
        # Ensure output directory exists
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear the file and write header
        with open(self.config.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['issue_identifier', 'matches_summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        logger.info(f"Initialized output file: {self.config.output_file}")

    def write_duplicate_summary(self, issue_identifier: str, duplicates: List[Dict[str, Any]]):
        """Write a summary of duplicate matches for an issue."""
        if not duplicates:
            matches_summary = "0 - No duplicates found"
        else:
            # Format: "[total matches] - ([issue identifier 1]: [confidence], [issue identifier 2]: [confidence], ...)"
            match_details = []
            for dup in duplicates:
                match_details.append(f"{dup['id']}: {dup['confidence']}")
            
            matches_summary = f"{len(duplicates)} - ({', '.join(match_details)})"
        
        action = {
            'issue_identifier': issue_identifier,
            'matches_summary': matches_summary
        }
        
        self._actions_log.append(action)
        self.write_single_action_to_file(action)
        logger.info(f"Wrote duplicate summary for {issue_identifier}: {matches_summary}")

    def _make_request(self, query: str) -> Dict[str, Any]:
        """Helper to make requests to the Linear API with rate limiting."""
        self._request_count += 1
        
        if self._request_count > 1:
            time.sleep(self.config.rate_limit_delay)
        
        try:
            response = requests.post(self.api_url, json={"query": query}, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return {}
                
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Linear API: {e}")
            return {}

    def get_internal_id(self, identifier: str, all_issues: List[Dict[str, Any]]) -> Optional[str]:
        """Finds the internal Linear ID for a given public identifier."""
        for issue in all_issues:
            if issue.get('identifier') == identifier:
                return issue.get('id')
        
        logger.warning(f"Could not find internal ID for identifier: {identifier}")
        return None

    def get_or_create_label_id(self, label_name: str) -> Optional[str]:
        """Gets the ID of a label by name, creating it if it doesn't exist."""
        # COMMENTED OUT: Original Linear API call
        # if self.config.dry_run:
        #     logger.info(f"[DRY RUN] Would get/create label: {label_name}")
        #     return "dry-run-label-id"
        
        # # First, try to get existing label
        # query = f'''
        # query {{
        #     team(id: "{self.config.linear_team_id}") {{
        #         labels {{
        #             nodes {{
        #                 id
        #                 name
        #             }}
        #         }}
        #     }}
        # }}
        # '''
        
        # data = self._make_request(query)
        # if data and 'data' in data and 'team' in data['data'] and data['data']['team']:
        #     for label in data['data']['team']['labels']['nodes']:
        #         if label['name'] == label_name:
        #             logger.debug(f"Found existing label: {label_name} (ID: {label['id']})")
        #             return label['id']

        # # If not found, create it
        # logger.info(f"Creating new label: {label_name}")
        # mutation = f'''
        # mutation {{
        #     labelCreate(input: {{
        #         name: "{label_name}",
        #         teamId: "{self.config.linear_team_id}"
        #     }}) {{
        #         label {{
        #             id
        #         }}
        #         success
        #     }}
        # }}
        # '''
        
        # data = self._make_request(mutation)
        # if data and 'data' in data and 'labelCreate' in data['data'] and data['data']['labelCreate']['success']:
        #     label_id = data['data']['labelCreate']['label']['id']
        #     logger.info(f"Successfully created label: {label_name} (ID: {label_id})")
        #     return label_id
            
        # logger.error(f"Failed to create label: {label_name}")
        # return None
        
        # NEW: Return a mock label ID for file output
        mock_label_id = f"mock-label-{label_name.lower().replace(' ', '-')}"
        logger.info(f"[FILE OUTPUT] Would get/create label: {label_name} (mock ID: {mock_label_id})")
        return mock_label_id

    def add_label_to_issue(self, issue_id: str, label_id: str, issue_identifier: str = None):
        """Adds a label to an issue, preserving existing labels."""
        # COMMENTED OUT: Original Linear API call
        # if self.config.dry_run:
        #     logger.info(f"[DRY RUN] Would add label to issue {issue_identifier or issue_id}")
        #     return
        
        # # Get current labels
        # query = f'''
        # query {{
        #     issue(id: "{issue_id}") {{
        #         labels {{
        #             nodes {{
        #                 id
        #             }}
        #         }}
        #     }}
        # }}
        # '''
        
        # data = self._make_request(query)
        # if not (data and 'data' in data and 'issue' in data['data']):
        #     logger.error(f"Failed to get current labels for issue {issue_id}")
        #     return

        # current_label_ids = [label['id'] for label in data['data']['issue']['labels']['nodes']]
        
        # if label_id not in current_label_ids:
        #     current_label_ids.append(label_id)
        #     mutation = f'''
        #     mutation {{
        #         issueUpdate(id: "{issue_id}", input: {{
        #             labelIds: {json.dumps(current_label_ids)}
        #         }}) {{
        #             success
        #         }}
        #     }}
        #     '''
            
        #     data = self._make_request(mutation)
        #     if data and 'data' in data and 'issueUpdate' in data['data'] and data['data']['issueUpdate']['success']:
        #         logger.info(f"Successfully added label to issue {issue_identifier or issue_id}")
        #     else:
        #         logger.error(f"Failed to add label to issue {issue_identifier or issue_id}")
        # else:
        #     logger.debug(f"Label already exists on issue {issue_identifier or issue_id}")
        
        # Log the action (no longer writing to CSV as we use summary approach)
        logger.info(f"[FILE OUTPUT] Would add label {label_id} to issue {issue_identifier or issue_id}")

    def add_comment_to_issue(self, issue_id: str, comment: str, issue_identifier: str = None):
        """Adds a comment to a Linear issue."""
        # COMMENTED OUT: Original Linear API call
        # if self.config.dry_run:
        #     logger.info(f"[DRY RUN] Would add comment to issue {issue_identifier or issue_id}:\n{comment[:100]}...")
        #     return
        
        # safe_comment = json.dumps(comment)
        # mutation = f'''
        # mutation {{
        #     commentCreate(input: {{
        #         issueId: "{issue_id}",
        #         body: {safe_comment}
        #     }}) {{
        #         success
        #     }}
        # }}
        # '''
        
        # data = self._make_request(mutation)
        # if data and 'data' in data and 'commentCreate' in data['data'] and data['data']['commentCreate']['success']:
        #     logger.info(f"Successfully added comment to issue {issue_identifier or issue_id}")
        # else:
        #     logger.error(f"Failed to add comment to issue {issue_identifier or issue_id}")
        
        # Log the action (no longer writing to CSV as we use summary approach)
        logger.info(f"[FILE OUTPUT] Would add comment to issue {issue_identifier or issue_id}:\n{comment[:100]}...")

    def link_issues(self, source_issue_id: str, related_issue_id: str, 
                   source_identifier: str = None, related_identifier: str = None):
        """Links one issue to another as related, without changing ticket status."""
        # COMMENTED OUT: Original Linear API call
        # if self.config.dry_run:
        #     logger.info(f"[DRY RUN] Would link {source_identifier or source_issue_id} as duplicate of {related_identifier or related_issue_id}")
        #     return
        
        # mutation = f'''
        # mutation {{
        #     issueRelationCreate(input: {{
        #         issueId: "{source_issue_id}",
        #         relatedIssueId: "{related_issue_id}",
        #         type: related
        #     }}) {{
        #         success
        #     }}
        # }}
        # '''
        
        # data = self._make_request(mutation)
        # if data and 'data' in data and 'issueRelationCreate' in data['data'] and data['data']['issueRelationCreate']['success']:
        #     logger.info(f"Successfully linked {source_identifier or source_issue_id} as duplicate of {related_identifier or related_issue_id}")
        # else:
        #     logger.error(f"Failed to link issues")
        
        # Log the action (no longer writing to CSV as we use summary approach)
        logger.info(f"[FILE OUTPUT] Would link {source_identifier or source_issue_id} as duplicate of {related_identifier or related_issue_id}")


class ChromaDBManager:
    """Manages ChromaDB operations for ticket embeddings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.openai_api_key,
            model_name=config.embedding_model
        )
        self.client = None
        self.collection = None
    
    def initialize(self):
        """Initialize ChromaDB client and collection."""
        # Ensure the directory exists
        self.config.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.config.chroma_collection_name,
            embedding_function=self.openai_ef
        )
        logger.info(f"ChromaDB initialized at {self.config.chroma_db_path}")
    
    def get_existing_ids(self) -> set:
        """Get all existing IDs in the collection."""
        try:
            # Get all IDs by querying with a large limit
            result = self.collection.get(limit=10000)  # Adjust if you have more tickets
            return set(result['ids']) if result['ids'] else set()
        except Exception as e:
            logger.warning(f"Could not get existing IDs: {e}")
            return set()
    
    def seed_issues(self, issues: List[Dict[str, Any]], update_existing: bool = False):
        """Seeds the ChromaDB with Linear issues."""
        if not self.collection:
            self.initialize()
        
        existing_ids = self.get_existing_ids()
        logger.info(f"Found {len(existing_ids)} existing tickets in ChromaDB")
        
        documents = []
        metadatas = []
        ids = []
        
        for issue in issues:
            if not (issue.get('title') and issue.get('identifier')):
                continue
            
            identifier = issue['identifier']
            
            # Skip if already exists and not updating
            if identifier in existing_ids and not update_existing:
                continue
            
            # Combine title and description for better matching
            description = issue.get('description', '')
            document = f"{issue['title']}\n{description}" if description else issue['title']
            
            documents.append(document)
            metadatas.append({
                'linear_id': identifier,
                'created_at': issue.get('createdAt', ''),
                'state': issue.get('state', {}).get('name', '') if isinstance(issue.get('state'), dict) else ''
            })
            ids.append(identifier)
        
        if not ids:
            logger.info("No new issues to add to ChromaDB")
            return
        
        # Add in batches to handle large datasets
        total_added = 0
        with tqdm(total=len(ids), desc="Adding issues to ChromaDB") as pbar:
            for i in range(0, len(ids), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(ids))
                
                try:
                    if update_existing:
                        # Update existing entries
                        self.collection.upsert(
                            documents=documents[i:batch_end],
                            metadatas=metadatas[i:batch_end],
                            ids=ids[i:batch_end]
                        )
                    else:
                        # Add new entries only
                        self.collection.add(
                            documents=documents[i:batch_end],
                            metadatas=metadatas[i:batch_end],
                            ids=ids[i:batch_end]
                        )
                    
                    total_added += (batch_end - i)
                    pbar.update(batch_end - i)
                    
                except Exception as e:
                    logger.error(f"Error adding batch {i}-{batch_end}: {e}")
        
        logger.info(f"Successfully added/updated {total_added} issues in ChromaDB")
    
    def find_similar_tickets(self, ticket_text: str, ticket_id: str, new_ticket_created_at: str = None) -> List[Dict[str, Any]]:
        """Queries ChromaDB to find the most similar tickets that were created BEFORE the current ticket."""
        if not self.collection:
            self.initialize()
        
        try:
            query_results = self.collection.query(
                query_texts=[ticket_text],
                n_results=self.config.max_candidates + 20,  # Get more to account for date filtering
                include=["metadatas", "distances"]
            )
            
            candidates = []
            if not query_results['metadatas'] or not query_results['metadatas'][0]:
                return []
            
            # Parse the new ticket's creation date for comparison
            new_ticket_date = None
            if new_ticket_created_at:
                try:
                    new_ticket_date = datetime.fromisoformat(new_ticket_created_at.replace('Z', '+00:00'))
                except Exception as e:
                    logger.warning(f"Could not parse creation date for {ticket_id}: {new_ticket_created_at}")
            
            for i, meta in enumerate(query_results['metadatas'][0]):
                match_id = meta.get('linear_id')
                if match_id and match_id != ticket_id:
                    # CRITICAL: Filter out tickets created after the current ticket (for bug bounty program)
                    if new_ticket_date and meta.get('created_at'):
                        try:
                            candidate_date = datetime.fromisoformat(meta['created_at'].replace('Z', '+00:00'))
                            if candidate_date >= new_ticket_date:
                                logger.debug(f"Filtered out {match_id} - created after {ticket_id}")
                                continue
                        except Exception as e:
                            logger.warning(f"Could not parse creation date for candidate {match_id}: {meta.get('created_at')}")
                            # If we can't parse the date, skip this candidate to be safe
                            continue
                    
                    distance = query_results['distances'][0][i] if 'distances' in query_results else None
                    candidates.append({
                        'id': match_id,
                        'distance': distance,
                        'metadata': meta
                    })
            
            # Return up to max_candidates after filtering
            filtered_candidates = candidates[:self.config.max_candidates]
            if len(candidates) > len(filtered_candidates):
                logger.info(f"Date filtering reduced candidates from {len(query_results['metadatas'][0])} to {len(candidates)}, returning top {len(filtered_candidates)}")
            
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []


class DuplicateAnalyzer:
    """Handles LLM-based duplicate analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def analyze_duplicates(self, new_ticket: Dict[str, Any], candidates: List[Dict[str, Any]], 
                         all_issues_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Uses an LLM to analyze candidate tickets and determine if they are duplicates."""
        if not candidates:
            return []
        
        # Build candidate details
        candidate_details = []
        for i, c in enumerate(candidates):
            issue_details = all_issues_map.get(c['id'], {})
            candidate_details.append({
                'index': i + 1,
                'id': c['id'],
                'title': issue_details.get('title', 'N/A'),
                'description': issue_details.get('description', 'N/A'),
                'distance': c.get('distance', 'N/A')
            })
        
        prompt = self._build_prompt(new_ticket, candidate_details)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert bug triage assistant with deep knowledge of software engineering and issue tracking."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more consistent results
            )
            
            output = response.choices[0].message.content
            logger.debug(f"LLM Response:\n{output}")
            
            return self._parse_llm_response(output)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []
    
    def _build_prompt(self, new_ticket: Dict[str, Any], candidate_details: List[Dict[str, Any]]) -> str:
        """Builds the prompt for the LLM."""
        candidates_text = "\n\n".join([
            f"{c['index']}. ID: {c['id']} (similarity distance: {c['distance']:.3f})\n"
            f"Title: {c['title']}\n"
            f"Description: {c['description']}"
            for c in candidate_details
        ])
        
        return f"""You are an expert bug triage assistant. For each candidate ticket below, determine if it is a duplicate of the new ticket.

Consider these types of duplicates:
1. Exact duplicates - Same issue, possibly reported by different users
2. Root cause duplicates - Different symptoms but same underlying problem
3. Pattern duplicates - Same type of issue affecting different parts of the system

For each candidate, respond EXACTLY in this format on a single line:
[ID]: [Yes/No], [confidence 0-100], [explanation]

Example responses:
BUG-123: Yes, 90, Both tickets describe the same API timeout issue with identical error messages and stack traces
BUG-456: No, 20, Different feature area and unrelated error type

New Ticket:
ID: {new_ticket['identifier']}
Title: {new_ticket['title']}
Description: {new_ticket.get('description', 'No description provided')}

Candidate Tickets:
{candidates_text}

Analyze each candidate:"""
    
    def _parse_llm_response(self, output: str) -> List[Dict[str, Any]]:
        """Parse the LLM response with improved error handling."""
        duplicates = []
        
        # Split the output into lines and process each one
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Pattern to match: ID: Yes/No, confidence, explanation (single line)
            pattern = r'^(\d+\.\s*)?([A-Z0-9]+-\d+):\s*(Yes|No),\s*(\d+),\s*(.+)$'
            match = re.match(pattern, line, re.IGNORECASE)
            
            if match:
                # Extract components (group 1 is optional numbering, group 2 is ticket ID)
                ticket_id = match.group(2)
                is_duplicate = match.group(3)
                confidence_str = match.group(4)
                explanation = match.group(5)
                
                if is_duplicate.lower() == 'yes':
                    try:
                        confidence = int(confidence_str)
                        if confidence >= self.config.confidence_threshold:
                            duplicates.append({
                                'id': ticket_id.upper(),
                                'confidence': confidence,
                                'explanation': explanation.strip()
                            })
                            logger.debug(f"Found duplicate: {ticket_id} (confidence: {confidence}%)")
                    except ValueError:
                        logger.warning(f"Could not parse confidence for {ticket_id}: {confidence_str}")
        
        return duplicates


def load_all_issues(filepath: Path) -> List[Dict[str, Any]]:
    """Loads all Linear issues from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            issues = json.load(f)
            logger.info(f"Loaded {len(issues)} issues from {filepath}")
            return issues
    except FileNotFoundError:
        logger.error(f"Issues file not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return []


def process_ticket(ticket_id: str, config: Config, linear_api: LinearAPI, 
                  chroma_manager: ChromaDBManager, analyzer: DuplicateAnalyzer,
                  all_issues: List[Dict[str, Any]], all_issues_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main processing logic for a single ticket.
    Returns a summary of actions taken.
    """
    logger.info(f"Processing ticket: {ticket_id}")
    
    # Get ticket details
    new_ticket = all_issues_map.get(ticket_id)
    if not new_ticket:
        logger.error(f"Ticket {ticket_id} not found in loaded issues")
        return {'status': 'error', 'message': 'Ticket not found'}
    
    # Prepare ticket text for similarity search
    new_ticket_text = f"{new_ticket.get('title', '')}\n{new_ticket.get('description', '')}"
    if not new_ticket_text.strip():
        logger.warning(f"Ticket {ticket_id} has no title or description")
        return {'status': 'skipped', 'message': 'No content to analyze'}
    
    # Find similar candidates
    candidates = chroma_manager.find_similar_tickets(new_ticket_text, ticket_id, new_ticket.get('createdAt'))
    logger.info(f"Found {len(candidates)} potential duplicate candidates")
    
    if not candidates:
        # No similar tickets found
        internal_id = linear_api.get_internal_id(ticket_id, all_issues)
        if internal_id and not config.dry_run:
            comment = "🤖 AI Duplicate Detection: No potential duplicates were found for this ticket."
            linear_api.add_comment_to_issue(internal_id, comment, ticket_id)
        
        # Write duplicate summary to CSV (no duplicates found)
        linear_api.write_duplicate_summary(ticket_id, [])
        return {'status': 'completed', 'duplicates_found': 0}
    
    # Analyze with LLM
    duplicates = analyzer.analyze_duplicates(new_ticket, candidates, all_issues_map)
    logger.info(f"LLM identified {len(duplicates)} duplicates above confidence threshold")
    
    # Take action in Linear
    internal_id = linear_api.get_internal_id(ticket_id, all_issues)
    if not internal_id:
        logger.error(f"Could not find internal ID for {ticket_id}")
        return {'status': 'error', 'message': 'Internal ID not found'}
    
    # Write duplicate summary to CSV
    linear_api.write_duplicate_summary(ticket_id, duplicates)
    
    if duplicates:
        # Add label
        label_id = linear_api.get_or_create_label_id('ai-flagged-duplicate')
        if label_id:
            linear_api.add_label_to_issue(internal_id, label_id, ticket_id)
        
        # Process each duplicate
        for dup in duplicates:
            related_internal_id = linear_api.get_internal_id(dup['id'], all_issues)
            if related_internal_id:
                # Link issues
                linear_api.link_issues(internal_id, related_internal_id, ticket_id, dup['id'])
                
                # Add detailed comment
                comment = (
                    f"🤖 **AI Duplicate Detection**\n\n"
                    f"This ticket appears to be a duplicate of **{dup['id']}**.\n\n"
                    f"**Confidence:** {dup['confidence']}%\n\n"
                    f"**Analysis:** {dup['explanation']}\n\n"
                    f"*Please review this assessment. If incorrect, you can unlink the issues.*"
                )
                linear_api.add_comment_to_issue(internal_id, comment, ticket_id)
            else:
                logger.warning(f"Could not find internal ID for duplicate {dup['id']}")
    else:
        # No duplicates found above threshold
        comment = (
            f"🤖 **AI Duplicate Detection**\n\n"
            f"Analyzed {len(candidates)} similar tickets but found no duplicates "
            f"above the {config.confidence_threshold}% confidence threshold.\n\n"
            f"*This is an automated analysis. Feel free to manually link related issues if needed.*"
        )
        linear_api.add_comment_to_issue(internal_id, comment, ticket_id)
    
    return {
        'status': 'completed',
        'duplicates_found': len(duplicates),
        'candidates_analyzed': len(candidates)
    }


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Find and flag duplicate tickets in Linear using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial setup - seed ChromaDB with all tickets
  python %(prog)s --seed
  
  # Process specific tickets
  python %(prog)s --tickets BUG-123 BUG-456
  
  # Process all tickets (bulk backfill)
  python %(prog)s --all
  
  # Dry run to see what would happen
  python %(prog)s --tickets BUG-123 --dry-run
  
  # Update existing ChromaDB entries
  python %(prog)s --seed --update-existing
"""
    )
    
    parser.add_argument('--seed', action='store_true', 
                       help="Seed the ChromaDB with issues from the JSON file")
    parser.add_argument('--update-existing', action='store_true',
                       help="Update existing entries when seeding (use with --seed)")
    parser.add_argument('--tickets', nargs='+', 
                       help="List of ticket identifiers to process (e.g., BUG-123 BUG-456)")
    parser.add_argument('--all', action='store_true', 
                       help="Process all tickets from the JSON file")
    parser.add_argument('--issues-file', type=str,
                       help="Path to the Linear issues JSON file")
    parser.add_argument('--dry-run', action='store_true',
                       help="Run without making changes to Linear")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Set logging level")
    parser.add_argument('--log-file', type=str,
                       help="Log to file in addition to console")
    parser.add_argument('--confidence-threshold', type=int,
                       help="Override default confidence threshold (0-100)")
    parser.add_argument('--output-file', type=str,
                       help="Path to output CSV file for actions (default: linear_actions_output.csv)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Create configuration
        config = Config.from_env(
            issues_json_path=args.issues_file,
            dry_run=args.dry_run
        )
        
        # Override output file if provided
        if args.output_file:
            config.output_file = Path(args.output_file)
        
        # Override confidence threshold if provided
        if args.confidence_threshold is not None:
            if 0 <= args.confidence_threshold <= 100:
                config.confidence_threshold = args.confidence_threshold
            else:
                logger.error("Confidence threshold must be between 0 and 100")
                return 1
        
        if config.dry_run:
            logger.info("🏃 Running in DRY RUN mode - no changes will be made to Linear")
        
        logger.info(f"📄 Actions will be written to: {config.output_file}")
        
        # Load issues
        all_issues = load_all_issues(config.issues_json_path)
        if not all_issues:
            logger.error("No issues loaded. Exiting.")
            return 1
        
        # Initialize components
        chroma_manager = ChromaDBManager(config)
        
        # Handle seeding
        if args.seed:
            logger.info("Starting ChromaDB seeding process...")
            chroma_manager.seed_issues(all_issues, update_existing=args.update_existing)
            logger.info("✅ Seeding completed successfully")
            return 0
        
        # Determine which tickets to process
        tickets_to_process = []
        if args.tickets:
            tickets_to_process = args.tickets
        elif args.all:
            tickets_to_process = [issue['identifier'] for issue in all_issues]
        
        if not tickets_to_process:
            logger.error("No tickets specified. Use --tickets, --all, or --seed.")
            parser.print_help()
            return 1
        
        # Initialize remaining components
        linear_api = LinearAPI(config)
        analyzer = DuplicateAnalyzer(config)
        all_issues_map = {issue['identifier']: issue for issue in all_issues}
        
        # Initialize output file
        linear_api.initialize_output_file()
        
        # Process tickets
        logger.info(f"Processing {len(tickets_to_process)} tickets...")
        
        # Summary statistics
        total_processed = 0
        total_duplicates = 0
        errors = []
        
        # Use tqdm for progress bar if processing many tickets
        iterator = tqdm(tickets_to_process) if len(tickets_to_process) > 5 else tickets_to_process
        
        for ticket_id in iterator:
            try:
                result = process_ticket(
                    ticket_id, config, linear_api, chroma_manager, 
                    analyzer, all_issues, all_issues_map
                )
                
                if result['status'] == 'completed':
                    total_processed += 1
                    total_duplicates += result.get('duplicates_found', 0)
                elif result['status'] == 'error':
                    errors.append((ticket_id, result.get('message', 'Unknown error')))
                
                # Add a small delay between tickets to avoid rate limits
                if len(tickets_to_process) > 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {ticket_id}: {e}", exc_info=True)
                errors.append((ticket_id, str(e)))
        
        # Actions are now written incrementally, but we can still log the summary
        logger.info(f"All actions have been written incrementally to {config.output_file}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SUMMARY")
        logger.info("="*50)
        logger.info(f"Total tickets processed: {total_processed}")
        logger.info(f"Total duplicates found: {total_duplicates}")
        
        if errors:
            logger.warning(f"Errors encountered: {len(errors)}")
            for ticket_id, error in errors:
                logger.warning(f"  - {ticket_id}: {error}")
        
        logger.info("✅ Processing completed")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 