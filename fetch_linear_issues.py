#!/usr/bin/env python3
"""
Fetch all Linear issues and save them to a JSON file.
This script uses the Linear GraphQL API to fetch all issues from your workspace.
"""

import json
import requests
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def fetch_linear_issues(api_key: str) -> List[Dict[str, Any]]:
    """Fetch all issues from Linear using their GraphQL API."""
    
    api_url = "https://api.linear.app/graphql"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    
    all_issues = []
    has_more = True
    after_cursor = None
    
    # GraphQL query to fetch issues with pagination
    def build_query(after: Optional[str] = None) -> str:
        after_clause = f', after: "{after}"' if after else ''
        combined_filter = ', filter: {or: [{project: {name: {eq: "Support"}}}, {labels: {name: {eq: "Support"}}}]}'
        
        return f"""
        query {{
            issues(first: 100{after_clause}{combined_filter}) {{
                pageInfo {{
                    hasNextPage
                    endCursor
                }}
                nodes {{
                    id
                    identifier
                    title
                    description
                    state {{
                        name
                        type
                    }}
                    priority
                    createdAt
                    updatedAt
                    assignee {{
                        name
                        email
                    }}
                    creator {{
                        name
                        email
                    }}
                    team {{
                        id
                        name
                        key
                    }}
                    labels {{
                        nodes {{
                            name
                        }}
                    }}
                    project {{
                        name
                    }}
                    url
                }}
            }}
        }}
        """
    
    print("Fetching issues from Linear...")
    page_count = 0
    
    while has_more:
        page_count += 1
        print(f"Fetching page {page_count}...", end="", flush=True)
        
        query = build_query(after_cursor)
        
        try:
            response = requests.post(api_url, json={"query": query}, headers=headers)
            
            # Debug the response if there's an error
            if response.status_code != 200:
                print(f"\nAPI Response Status: {response.status_code}")
                print(f"Response Headers: {dict(response.headers)}")
                try:
                    error_data = response.json()
                    print(f"Error Response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Response Text: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                print(f"\nError from Linear API: {data['errors']}")
                break
            
            if 'data' not in data or 'issues' not in data['data']:
                print("\nUnexpected response format from Linear API")
                break
            
            issues_data = data['data']['issues']
            issues = issues_data.get('nodes', [])
            all_issues.extend(issues)
            
            page_info = issues_data.get('pageInfo', {})
            has_more = page_info.get('hasNextPage', False)
            after_cursor = page_info.get('endCursor')
            
            print(f" fetched {len(issues)} issues")
            
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching data from Linear: {e}")
            break
    
    print(f"\nTotal issues fetched: {len(all_issues)}")
    return all_issues

def save_issues_to_file(issues: List[Dict[str, Any]], output_path: Path):
    """Save issues to a JSON file in the expected format."""
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with nice formatting
    with open(output_path, 'w') as f:
        json.dump(issues, f, indent=2, default=str)
    
    print(f"Saved {len(issues)} issues to {output_path}")

def main():
    """Main function to fetch and save Linear issues."""
    # Get API key from environment
    api_key = os.environ.get('LINEAR_OAUTH_ACCESS_TOKEN') or os.environ.get('LINEAR_API_KEY')
    
    if not api_key:
        print("Error: LINEAR_OAUTH_ACCESS_TOKEN not found in environment variables or .env file")
        print("\nTo get your Linear API token:")
        print("1. Go to Linear Settings > API")
        print("2. Create a new personal API key")
        print("3. Add it to your .env file as: LINEAR_OAUTH_ACCESS_TOKEN=your-token-here")
        return 1
    
    # Filter for Support project OR Support label
    print("Filtering issues for 'Support' project OR 'Support' label")
    
    # Determine output path - save in current working directory
    output_path = Path.cwd() / "linear_issues.json"
    
    # Allow custom output path via command line argument
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    
    print(f"Output file: {output_path}")
    print("-" * 50)
    
    # Fetch issues
    issues = fetch_linear_issues(api_key)
    
    if issues:
        # Save to file
        save_issues_to_file(issues, output_path)
        
        # Print summary statistics
        print("\nSummary:")
        print(f"- Total issues: {len(issues)}")
        
        # Count by state
        state_counts = {}
        for issue in issues:
            state = issue.get('state', {}).get('name', 'Unknown')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        print("\nIssues by state:")
        for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {state}: {count}")
        
        # Count by team
        team_counts = {}
        for issue in issues:
            team = issue.get('team', {}).get('name', 'Unknown')
            team_counts[team] = team_counts.get(team, 0) + 1
        
        if len(team_counts) > 1:
            print("\nIssues by team:")
            for team, count in sorted(team_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {team}: {count}")
        
        print(f"\n✅ Successfully fetched and saved all Linear issues!")
        print(f"\nYou can now run the duplicate detection script:")
        print(f"  python ai_flag_duplicates_improved.py --seed")
        return 0
    else:
        print("\n❌ No issues were fetched. Please check your API token and network connection.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 