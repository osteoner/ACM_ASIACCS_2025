import re
import pandas as pd
import psycopg2
import json
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import numpy as np

def clean_text(text: str) -> str:
    """Clean and preprocess text data"""
    if pd.isna(text) or text == "":
        return ""
        
    text = re.sub(r"\\\\", "\n", text)
    quote_pattern = r'\*\*\*QUOTE\*\*\*([\s\S]*?)\*\*\*QUOTE\*\*\*'
    link_pattern = r'\*\*\*LINK\*\*\*([\s\S]*?)\*\*\*LINK\*\*\*'
    img_pattern = r'\*\*\*IMG\*\*\*([\s\S]*?)\*\*\*IMG\*\*\*'
    citing_pattern = r'\*\*\*CITING\*\*\*([\s\S]*?)\*\*\*CITING\*\*\*'
    iframe_pattern = r'\*\*\*IFRAME\*\*\*([\s\S]*?)\*\*\*IFRAME\*\*\*'
    attachment_pattern = r'\*\*\*ATTACHMENT\*\*\*([\s\S]*?)\*\*\*ATTACHMENT\*\*\*'
    
    text = re.sub(quote_pattern, 'QUOTE', text)
    text = re.sub(link_pattern, 'LINK', text)
    text = re.sub(img_pattern, 'IMG', text)
    text = re.sub(citing_pattern, 'CITING', text)
    text = re.sub(iframe_pattern, 'IFRAME', text)
    text = re.sub(attachment_pattern, 'ATTACHMENT', text)
    
    url_pattern = r'http\S+'
    text = re.sub(url_pattern, 'HTTPURL', text, flags=re.DOTALL)
    
    text = text.strip()
    text = re.sub(r'\n', ' ', text)
    text = text.lower()
    
    punctuation = r"[^\w\s'-]"
    text = re.sub(punctuation, ' ', text)
    emoji_pattern = r"[^a-z0-9\s']+"
    text = re.sub(emoji_pattern, ' ', text)
    
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&.+?;", " ", text)
    text = re.sub(r'\S{100,}', '[ENCODED_DATA]', text)
        
    return text.strip()

def process_user_batch(user_batch, posts_dict, threads_dict):
    """Process a batch of users and return their SUP sequences"""
    results = []
    
    for _, row in user_batch.iterrows():
        uid = row['id']
        name = clean_text(row['username'])
        reputation = row['reputation']

        # Get user's threads and posts using pre-built dictionaries
        user_threads = threads_dict.get(uid, pd.DataFrame())
        user_posts = posts_dict.get(uid, pd.DataFrame())
        
        # Clean and sort data
        if not user_threads.empty:
            user_threads = user_threads.copy()
            user_threads['name'] = user_threads['name'].apply(clean_text)
            user_threads = user_threads.sort_values(by='created_on').head(20)
        
        if not user_posts.empty:
            user_posts = user_posts.copy()
            user_posts['content'] = user_posts['content'].apply(clean_text)
            user_posts = user_posts.sort_values(by='created_on').head(20)

        # Create metadata
        metadata = f"[M]{name}[SEP]{reputation}[SEP]{len(user_threads)}[SEP]{len(user_posts)}"

        # Create sections
        if not user_threads.empty:
            threads_section = "[T]" + "[SEP]".join(user_threads['name'])
        else:
            threads_section = "[T]"
            
        if not user_posts.empty:
            replies_section = "[R]" + "[SEP]".join(user_posts['content'])
        else:
            replies_section = "[R]"

        # Compose SUP string
        sequence = metadata + threads_section + replies_section

        results.append({
            "user_id": uid,
            "username": name,
            "sup_sequence": sequence
        })
    
    return results

def create_user_data_dicts(posts, threads):
    """Create dictionaries for faster user data lookup"""
    print("Creating data dictionaries for faster lookup...")
    
    # Group posts by creator_id
    posts_dict = {}
    if not posts.empty:
        posts_grouped = posts.groupby('creator_id')
        posts_dict = {uid: group for uid, group in posts_grouped}
    
    # Group threads by creator_id
    threads_dict = {}
    if not threads.empty:
        threads_grouped = threads.groupby('creator_id')
        threads_dict = {uid: group for uid, group in threads_grouped}
    
    return posts_dict, threads_dict

def filter_active_users(members, posts_dict, threads_dict, min_threads=5, min_posts=10):
    """Filter users who have at least min_threads threads OR min_posts posts"""
    print(f"Filtering users with at least {min_threads} threads OR {min_posts} posts...")
    
    filtered_members = []
    
    for _, row in members.iterrows():
        uid = row['id']
        
        # Count user's threads and posts
        thread_count = len(threads_dict.get(uid, pd.DataFrame()))
        post_count = len(posts_dict.get(uid, pd.DataFrame()))
        
        # Keep user if they meet the minimum threshold for threads OR posts
        if thread_count >= min_threads or post_count >= min_posts:
            filtered_members.append(row)
    
    filtered_df = pd.DataFrame(filtered_members)
    print(f"Filtered from {len(members)} to {len(filtered_df)} active users")
    
    return filtered_df

def generate_sup_json(conn, start_date, end_date, min_threads=5, min_posts=10, n_processes=None):
    """Generate SUP JSON with multiprocessing support and user activity filtering"""
    
    if n_processes is None:
        n_processes = mp.cpu_count() - 1
    
    print(f"Using {n_processes} processes")
    
    # Load data with optimized queries
    print("Loading members data...")
    members = pd.read_sql("SELECT id, username, reputation FROM members", conn)

    print("Loading posts data...")
    posts_query = f"""
        SELECT creator_id, content, created_on
        FROM posts
        WHERE created_on >= '{start_date}' AND created_on <= '{end_date}'
        ORDER BY creator_id, created_on
    """
    posts = pd.read_sql(posts_query, conn)

    print("Loading threads data...")
    threads_query = f"""
        SELECT id, creator_id, name, created_on
        FROM threads
        WHERE created_on >= '{start_date}' AND created_on <= '{end_date}'
        ORDER BY creator_id, created_on
    """
    threads = pd.read_sql(threads_query, conn)

    # Filter members based on their posts and threads activity
    active_user_ids = set(posts['creator_id'].unique()) | set(threads['creator_id'].unique())
    members = members[members['id'].isin(active_user_ids)]
    print(f"Users with activity in date range: {len(members)}")
    
    # Create dictionaries for faster lookup
    posts_dict, threads_dict = create_user_data_dicts(posts, threads)
    
    # Apply activity filter (at least min_threads threads OR min_posts posts)
    members = filter_active_users(members, posts_dict, threads_dict, min_threads, min_posts)
    
    if members.empty:
        print("No users meet the activity criteria!")
        return []
    
    # Split members into batches for multiprocessing
    batch_size = max(1, len(members) // n_processes)
    member_batches = np.array_split(members, n_processes)
    
    print(f"Processing {len(members)} users in {len(member_batches)} batches...")
    
    # Create partial function with shared data
    process_func = partial(process_user_batch, 
                          posts_dict=posts_dict, 
                          threads_dict=threads_dict)
    
    # Process batches in parallel
    results = []
    with mp.Pool(processes=n_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_func, member_batches),
            total=len(member_batches),
            desc="Processing batches"
        ))
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
    
    print(f"Generated SUP sequences for {len(results)} users")
    return results

def generate_sup_json_optimized_single_process(conn, start_date, end_date, min_threads=5, min_posts=10):
    """Optimized single-process version for comparison with user activity filtering"""
    
    print("Loading data...")
    members = pd.read_sql("SELECT id, username, reputation FROM members", conn)

    # Use more efficient queries with indexes
    posts_query = f"""
        SELECT creator_id, content, created_on
        FROM posts
        WHERE created_on >= '{start_date}' AND created_on <= '{end_date}'
        ORDER BY creator_id, created_on
    """
    posts = pd.read_sql(posts_query, conn)

    threads_query = f"""
        SELECT id, creator_id, name, created_on
        FROM threads
        WHERE created_on >= '{start_date}' AND created_on <= '{end_date}'
        ORDER BY creator_id, created_on
    """
    threads = pd.read_sql(threads_query, conn)

    # Filter members and create lookup dictionaries
    active_user_ids = set(posts['creator_id'].unique()) | set(threads['creator_id'].unique())
    members = members[members['id'].isin(active_user_ids)]
    print(f"Users with activity in date range: {len(members)}")
    
    posts_dict, threads_dict = create_user_data_dicts(posts, threads)
    
    # Apply activity filter
    members = filter_active_users(members, posts_dict, threads_dict, min_threads, min_posts)
    
    if members.empty:
        print("No users meet the activity criteria!")
        return []
    
    # Process all users in single batch
    results = process_user_batch(members, posts_dict, threads_dict)
    
    print(f"Generated SUP sequences for {len(results)} users")
    return results

