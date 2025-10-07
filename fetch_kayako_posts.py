#!/usr/bin/env python3
"""
Script to fetch Kayako conversation posts for Linear issues with 'Kayako' label.

This script:
1. Filters Linear issues with 'Kayako' label from linear_issues.json
2. Parses conversation IDs from issue titles (8-digit numbers in square brackets)
3. Fetches all posts from Kayako API for each conversation
4. Saves the mapping of Linear issues to Kayako posts
"""

import json
import re
import requests
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import os
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for Kayako API access."""
    kayako_base_url: str
    kayako_email: str
    kayako_password: str
    linear_issues_file: Path
    output_file: Path
    rate_limit_delay: float = 0.5  # Delay between API calls in seconds
    
    @classmethod
    def from_env(cls, linear_issues_file: str = "linear_issues.json", 
                 output_file: str = "kayako_posts_mapping.json", 
                 dry_run: bool = False) -> 'Config':
        """Create config from environment variables."""
        kayako_base_url = os.getenv('KAYAKO_BASE_URL')
        kayako_email = os.getenv('KAYAKO_EMAIL')
        kayako_password = os.getenv('KAYAKO_PASSWORD')
        
        # For dry run, we don't need API credentials
        if not dry_run and not all([kayako_base_url, kayako_email, kayako_password]):
            raise ValueError(
                "Missing required environment variables: "
                "KAYAKO_BASE_URL, KAYAKO_EMAIL, KAYAKO_PASSWORD"
            )
        
        return cls(
            kayako_base_url=kayako_base_url or "https://example.kayako.com",
            kayako_email=kayako_email or "dummy@example.com",
            kayako_password=kayako_password or "dummy-password",
            linear_issues_file=Path(linear_issues_file),
            output_file=Path(output_file)
        )

class KayakoAPI:
    """Handles interactions with Kayako API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        
        # Setup Basic HTTP Authentication
        import base64
        credentials = f"{config.kayako_email}:{config.kayako_password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        # Setup headers with Basic Auth
        self.session.headers.update({
            'Authorization': f'Basic {encoded_credentials}',
            'Accept': 'application/json'
        })
    
    def get_conversation_posts(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all posts for a conversation from Kayako API.
        
        Args:
            conversation_id: The Kayako conversation ID
            
        Returns:
            List of post dictionaries
        """
        url = urljoin(
            self.config.kayako_base_url,
            f"/api/v1/cases/{conversation_id}/posts.json"
        )
        
        try:
            logger.info(f"Fetching posts for conversation {conversation_id}")
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('data', [])
            
            logger.info(f"Retrieved {len(posts)} posts for conversation {conversation_id}")
            return posts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching posts for conversation {conversation_id}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response for conversation {conversation_id}: {e}")
            return []
    
    def make_request_with_retry(self, url: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make a request with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * self.config.rate_limit_delay
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    return None

def load_linear_issues(filepath: Path) -> List[Dict[str, Any]]:
    """Load Linear issues from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            issues = json.load(f)
            logger.info(f"Loaded {len(issues)} issues from {filepath}")
            return issues
    except FileNotFoundError:
        logger.error(f"Linear issues file not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        return []

def filter_kayako_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter issues that have the 'Kayako' label."""
    kayako_issues = []
    
    for issue in issues:
        labels = issue.get('labels', {}).get('nodes', [])
        label_names = [label.get('name', '') for label in labels]
        
        if 'Kayako' in label_names:
            kayako_issues.append(issue)
    
    logger.info(f"Found {len(kayako_issues)} issues with 'Kayako' label")
    return kayako_issues

def extract_conversation_id(title: str) -> Optional[str]:
    """
    Extract conversation ID from issue title.
    
    Looks for 8-digit numbers in square brackets like [60211605].
    
    Args:
        title: The issue title
        
    Returns:
        The conversation ID string or None if not found
    """
    # Pattern to match 8-digit numbers in square brackets
    pattern = r'\[(\d{8})\]'
    match = re.search(pattern, title)
    
    if match:
        return match.group(1)
    return None

def fetch_all_kayako_posts(kayako_issues: List[Dict[str, Any]], 
                          kayako_api: KayakoAPI) -> Dict[str, Any]:
    """
    Fetch Kayako posts for all issues with conversation IDs.
    
    Args:
        kayako_issues: List of Linear issues with Kayako label
        kayako_api: KayakoAPI instance
        
    Returns:
        Dictionary mapping Linear issue identifiers to their Kayako posts
    """
    results = {}
    total_issues = len(kayako_issues)
    
    for i, issue in enumerate(kayako_issues, 1):
        identifier = issue.get('identifier', '')
        title = issue.get('title', '')
        
        logger.info(f"Processing issue {i}/{total_issues}: {identifier}")
        
        # Extract conversation ID from title
        conversation_id = extract_conversation_id(title)
        
        if not conversation_id:
            logger.warning(f"No conversation ID found in title for {identifier}: '{title}'")
            results[identifier] = {
                'linear_issue': issue,
                'conversation_id': None,
                'posts': [],
                'error': 'No conversation ID found in title'
            }
            continue
        
        logger.info(f"Found conversation ID {conversation_id} for {identifier}")
        
        # Fetch posts from Kayako
        posts = kayako_api.get_conversation_posts(conversation_id)
        
        # Store results
        results[identifier] = {
            'linear_issue': issue,
            'conversation_id': conversation_id,
            'posts': posts,
            'posts_count': len(posts)
        }
        
        # Add delay between requests to respect rate limits
        if i < total_issues:
            time.sleep(kayako_api.config.rate_limit_delay)
    
    return results

def save_results(results: Dict[str, Any], output_file: Path):
    """Save the results to a JSON file."""
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to {output_file}")
        
        # Print summary
        total_issues = len(results)
        issues_with_posts = sum(1 for r in results.values() if r.get('posts_count', 0) > 0)
        total_posts = sum(r.get('posts_count', 0) for r in results.values())
        
        logger.info(f"Summary:")
        logger.info(f"  Total issues processed: {total_issues}")
        logger.info(f"  Issues with posts: {issues_with_posts}")
        logger.info(f"  Total posts fetched: {total_posts}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fetch Kayako conversation posts for Linear issues with 'Kayako' label",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables Required:
  KAYAKO_BASE_URL    - Base URL for Kayako API (e.g., https://your-domain.kayako.com)
  KAYAKO_EMAIL       - Kayako account email address
  KAYAKO_PASSWORD    - Kayako account password

Examples:
  # Use default files
  python %(prog)s
  
  # Specify custom files
  python %(prog)s --issues-file custom_issues.json --output-file custom_output.json
  
  # Dry run to see what would be processed
  python %(prog)s --dry-run
"""
    )
    
    parser.add_argument('--issues-file', type=str, default='linear_issues.json',
                       help='Path to Linear issues JSON file')
    parser.add_argument('--output-file', type=str, default='kayako_posts_mapping.json',
                       help='Path to output JSON file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without making API calls')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    
    args = parser.parse_args()
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Create configuration
        config = Config.from_env(
            linear_issues_file=args.issues_file,
            output_file=args.output_file,
            dry_run=args.dry_run
        )
        
        # Load Linear issues
        all_issues = load_linear_issues(config.linear_issues_file)
        if not all_issues:
            logger.error("No issues loaded. Exiting.")
            return 1
        
        # Filter Kayako issues
        kayako_issues = filter_kayako_issues(all_issues)
        if not kayako_issues:
            logger.warning("No issues with 'Kayako' label found.")
            return 0
        
        if args.dry_run:
            logger.info("DRY RUN - Would process the following issues:")
            for issue in kayako_issues:
                identifier = issue.get('identifier', '')
                title = issue.get('title', '')
                conversation_id = extract_conversation_id(title)
                logger.info(f"  {identifier}: {title} -> {conversation_id or 'NO CONVERSATION ID'}")
            return 0
        
        # Initialize Kayako API
        kayako_api = KayakoAPI(config)
        
        # Fetch all posts
        logger.info("Starting to fetch Kayako posts...")
        results = fetch_all_kayako_posts(kayako_issues, kayako_api)
        
        # Save results
        save_results(results, config.output_file)
        
        logger.info("✅ Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
