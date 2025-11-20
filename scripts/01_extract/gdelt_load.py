"""Load GDELT GKG sentiment data from BigQuery and aggregate to 15-minute intervals.

Queries Google's public GDELT BigQuery dataset, filters for relevant topics/themes,
aggregates sentiment to 15-minute intervals, and saves to project BigQuery table.

GDELT GKG (Global Knowledge Graph) contains:
- Tone: Overall sentiment (-10 to +10, negative = bad news)
- Polarity: Strength of sentiment  
- Themes: Topics mentioned in articles
- Updated every 15 minutes

Public Dataset: `gdelt-bq.gdeltv2.gkg`
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import yaml
from typing import List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
TABLE_NAME = os.environ.get('BQ_GDELT_TABLE', 'gdelt_sentiment')
STAGING_TABLE_NAME = os.environ.get('BQ_GDELT_STAGING_TABLE', 'gdelt_sentiment_staging')

# GDELT public BigQuery dataset
GDELT_PROJECT = 'gdelt-bq'
GDELT_DATASET = 'gdeltv2'
GDELT_TABLE = 'gkg_partitioned'  # Use partitioned table for 28x cost reduction

# Default topics to track
DEFAULT_TOPICS = [
    'FED',  # Federal Reserve
    'FEDERAL_RESERVE',
    'INTEREST_RATE',
    'INFLATION',
    'RECESSION',
]


def build_gdelt_query(
    start_date: str,
    end_date: str,
    topics: List[str]
) -> str:
    """
    Build BigQuery SQL to fetch and aggregate GDELT GKG sentiment data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)  
        topics: List of topics/themes to filter for
    
    Returns:
        SQL query string
    """
    # Convert topics to UPPER and underscore format (GDELT format)
    topic_patterns = [f"'%{topic.upper().replace(' ', '_')}%'" for topic in topics]
    topic_conditions = ' OR '.join([f"V2Themes LIKE {pattern}" for pattern in topic_patterns])
    
    query = f"""
    WITH filtered_articles AS (
        SELECT
            -- Parse timestamp and round to 15-minute intervals
            TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS article_timestamp,
            TIMESTAMP_TRUNC(
                TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING))),
                HOUR
            ) + INTERVAL DIV(EXTRACT(MINUTE FROM TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING)))), 15) * 15 MINUTE AS interval_timestamp,
            
            -- Parse V2Tone: tone,positive,negative,polarity,activity,self_ref,word_count
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64) AS positive_score,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS negative_score,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(3)] AS FLOAT64) AS polarity,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(6)] AS INT64) AS word_count,
            
            V2Themes,
            SourceCommonName
            
        FROM `{GDELT_PROJECT}.{GDELT_DATASET}.{GDELT_TABLE}`
        
        WHERE
            -- Partition filter (reduces cost by 28x - only scans relevant days)
            _PARTITIONTIME >= TIMESTAMP("{start_date}")
            AND _PARTITIONTIME <= TIMESTAMP("{end_date}")
            
            -- Date range filter (DATE is INT64 in format YYYYMMDDHHMMSS)
            AND DATE >= {start_date.replace("-", "")}000000
            AND DATE <= {end_date.replace("-", "")}235959
            
            -- Topic filter (must mention at least one topic)
            AND ({topic_conditions})
            
            -- Valid tone data
            AND V2Tone IS NOT NULL
            AND LENGTH(V2Tone) > 0
    ),
    
    aggregated AS (
        SELECT
            interval_timestamp AS timestamp,
            
            -- Weighted averages (weight by word count)
            SUM(tone * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_tone,
            SUM(positive_score * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_positive,
            SUM(negative_score * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_negative,
            SUM(polarity * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_polarity,
            
            -- Counts
            COUNT(*) AS num_articles,
            COUNT(DISTINCT SourceCommonName) AS num_sources,
            SUM(COALESCE(word_count, 0)) AS total_word_count,
            
            -- Min/max for quality checks
            MIN(tone) AS min_tone,
            MAX(tone) AS max_tone
            
        FROM filtered_articles
        
        GROUP BY interval_timestamp
        ORDER BY interval_timestamp
    )
    
    SELECT * FROM aggregated
    """
    
    return query


def resample_to_frequency(
    df: pd.DataFrame,
    target_frequency: str,
    method: str = 'mean'
) -> pd.DataFrame:
    """Resample 15-minute GDELT data to target frequency.
    
    Args:
        df: DataFrame with 15-minute intervals
        target_frequency: Target frequency ('15m', '1h', '4h', '1d', '1w')
        method: Aggregation method ('mean', 'median', 'first', 'last')
    
    Returns:
        Resampled DataFrame
    """
    if target_frequency == '15m':
        return df  # Already at native frequency
    
    print(f"\n📊 Resampling from 15m to {target_frequency}...")
    
    # Set timestamp as index for resampling
    df_resampled = df.set_index('timestamp')
    
    # Define aggregation rules
    agg_rules = {
        'weighted_avg_tone': method,
        'weighted_avg_positive': method,
        'weighted_avg_negative': method,
        'weighted_avg_polarity': method,
        'num_articles': 'sum',
        'num_sources': 'sum',
        'total_word_count': 'sum',
        'min_tone': 'min',
        'max_tone': 'max'
    }
    
    # Resample
    df_resampled = df_resampled.resample(target_frequency).agg(agg_rules)
    
    # Reset index
    df_resampled = df_resampled.reset_index()
    
    # Remove rows with no data
    df_resampled = df_resampled[df_resampled['num_articles'] > 0]
    
    print(f"   Original intervals: {len(df):,}")
    print(f"   Resampled intervals: {len(df_resampled):,}")
    print(f"   Aggregation method: {method}")
    
    return df_resampled


def clear_frequency_data(client: bigquery.Client, table_id: str, frequency: str) -> None:
    """Delete all data for a specific frequency."""
    print(f"\n🗑️  Clearing existing data for frequency: {frequency}")
    print(f"   Table: {table_id}")
    
    delete_query = f"""
    DELETE FROM `{table_id}`
    WHERE frequency = '{frequency}'
    """
    
    try:
        job = client.query(delete_query)
        job.result()
        print(f"   ✅ Cleared all {frequency} data")
    except Exception as e:
        print(f"   ⚠️  Clear failed (table may not exist): {e}")


def fetch_gdelt_sentiment(
    start_date: str,
    end_date: str,
    topics: List[str] = None,
    project_id: str = None
) -> pd.DataFrame:
    """
    Fetch and aggregate GDELT sentiment data from BigQuery.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        topics: List of topics to filter for (default: DEFAULT_TOPICS)
        project_id: GCP project ID (for billing)
    
    Returns:
        DataFrame with aggregated sentiment by 15-minute intervals
    """
    if topics is None:
        topics = DEFAULT_TOPICS
    
    if project_id is None:
        project_id = PROJECT_ID
    
    print(f"\n{'='*80}")
    print(f"  FETCHING GDELT SENTIMENT DATA")
    print(f"{'='*80}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Topics: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")
    print(f"Source: {GDELT_PROJECT}.{GDELT_DATASET}.{GDELT_TABLE}")
    
    # Build query
    query = build_gdelt_query(start_date, end_date, topics)
    
    # Execute query
    print(f"\nExecuting BigQuery query...")
    client = bigquery.Client(project=project_id)
    
    try:
        df = client.query(query).to_dataframe()
        
        if df.empty:
            print("\n⚠️  No sentiment data found for specified date range and topics")
            return df
        
        print(f"\n✅ Query complete!")
        print(f"   Intervals: {len(df)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Total articles: {df['num_articles'].sum():,.0f}")
        print(f"   Total sources: {df['num_sources'].sum():,.0f}")
        print(f"   Avg sentiment: {df['weighted_avg_tone'].mean():.3f}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error querying GDELT: {e}")
        raise


def save_to_bigquery(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    dry_run: bool = False,
    auto_merge: bool = True,
    flush_existing: bool = False,
    purge_all: bool = False,
    start_date: str = None,
    end_date: str = None
) -> None:
    """
    Save sentiment data to BigQuery using staging + MERGE pattern.
    
    Args:
        df: Sentiment DataFrame
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        dry_run: If True, skip actual save
        auto_merge: If True, automatically merge staging to main
        flush_existing: If True, delete existing data for date range first
        purge_all: If True, delete all data from table first
        start_date: Start date for flush (YYYY-MM-DD)
        end_date: End date for flush (YYYY-MM-DD)
    """
    if df.empty:
        print("\nNo data to save")
        return
    
    if dry_run:
        print("\n🔍 DRY RUN: Would save to BigQuery")
        print(f"   Staging: {project_id}.{dataset_id}.{STAGING_TABLE_NAME}")
        print(f"   Main: {project_id}.{dataset_id}.{TABLE_NAME}")
        print(f"   Rows: {len(df)}")
        return
    
    client = bigquery.Client(project=project_id)
    
    # Table references
    staging_table_id = f"{project_id}.{dataset_id}.{STAGING_TABLE_NAME}"
    main_table_id = f"{project_id}.{dataset_id}.{TABLE_NAME}"
    
    # Handle purge (delete all data)
    if purge_all:
        purge_all_data(client, main_table_id)
        if not df.empty:
            print("\n  Continuing with data load after purge...")
        else:
            return
    
    # Handle flush (delete existing data for date range)
    if flush_existing and start_date and end_date:
        flush_existing_data(client, main_table_id, start_date, end_date)
    
    # Schema
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("frequency", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("weighted_avg_tone", "FLOAT"),
        bigquery.SchemaField("weighted_avg_positive", "FLOAT"),
        bigquery.SchemaField("weighted_avg_negative", "FLOAT"),
        bigquery.SchemaField("weighted_avg_polarity", "FLOAT"),
        bigquery.SchemaField("num_articles", "INTEGER"),
        bigquery.SchemaField("num_sources", "INTEGER"),
        bigquery.SchemaField("total_word_count", "INTEGER"),
        bigquery.SchemaField("min_tone", "FLOAT"),
        bigquery.SchemaField("max_tone", "FLOAT"),
        bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP"),
    ]
    
    # Add derived columns for consistency with tickers table
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    # Only set frequency if not already set (resampling sets it in main())
    if 'frequency' not in df.columns:
        df['frequency'] = '15m'  # GDELT native 15-minute interval
    df['ingestion_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    print(f"\n{'='*80}")
    print(f"  SAVING TO BIGQUERY")
    print(f"{'='*80}")
    print(f"Staging: {staging_table_id}")
    
    # Load to staging table (overwrite)
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
    )
    
    job = client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
    job.result()  # Wait for completion
    
    print(f"✓ Loaded {len(df)} rows to staging")
    
    # Create main table if it doesn't exist
    try:
        client.get_table(main_table_id)
        print(f"Main table exists: {main_table_id}")
    except:
        print(f"Creating main table: {main_table_id}")
        table = bigquery.Table(main_table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="date"
        )
        table.clustering_fields = ["frequency"]
        client.create_table(table)
    
    # MERGE staging into main table
    merge_query = f"""
    MERGE `{main_table_id}` T
    USING `{staging_table_id}` S
    ON T.timestamp = S.timestamp AND T.frequency = S.frequency
    WHEN MATCHED THEN
        UPDATE SET
            date = S.date,
            weighted_avg_tone = S.weighted_avg_tone,
            weighted_avg_positive = S.weighted_avg_positive,
            weighted_avg_negative = S.weighted_avg_negative,
            weighted_avg_polarity = S.weighted_avg_polarity,
            num_articles = S.num_articles,
            num_sources = S.num_sources,
            total_word_count = S.total_word_count,
            min_tone = S.min_tone,
            max_tone = S.max_tone,
            ingestion_timestamp = S.ingestion_timestamp
    WHEN NOT MATCHED THEN
        INSERT (
            timestamp,
            date,
            frequency,
            weighted_avg_tone,
            weighted_avg_positive,
            weighted_avg_negative,
            weighted_avg_polarity,
            num_articles,
            num_sources,
            total_word_count,
            min_tone,
            max_tone,
            ingestion_timestamp
        )
        VALUES (
            timestamp,
            date,
            frequency,
            weighted_avg_tone,
            weighted_avg_positive,
            weighted_avg_negative,
            weighted_avg_polarity,
            num_articles,
            num_sources,
            total_word_count,
            min_tone,
            max_tone,
            ingestion_timestamp
        )
    """
    
    print(f"Merging staging → main table...")
    merge_job = client.query(merge_query)
    merge_job.result()
    
    if auto_merge:
        print(f"Merging staging → main table...")
        merge_job = client.query(merge_query)
        merge_job.result()
        print(f"✓ MERGE complete")
        print(f"\n✅ Data saved to {main_table_id}")
    else:
        print(f"\n⚠️  Skipped auto-merge (--no-merge flag)")
        print(f"   Data in staging: {staging_table_id}")
        print(f"   Run MERGE manually to update main table")


def load_config(config_path: str) -> dict:
    """Load GDELT config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def flush_existing_data(
    client: bigquery.Client,
    table_id: str,
    start_date: str,
    end_date: str
) -> None:
    """Delete existing data for date range."""
    print(f"\n🗑️  FLUSHING existing data...")
    print(f"   Table: {table_id}")
    print(f"   Range: {start_date} to {end_date}")
    
    delete_query = f"""
    DELETE FROM `{table_id}`
    WHERE DATE(timestamp) >= '{start_date}'
      AND DATE(timestamp) <= '{end_date}'
    """
    
    try:
        job = client.query(delete_query)
        job.result()
        print(f"   ✓ Flushed existing data")
    except Exception as e:
        print(f"   ⚠️  Flush failed (table may not exist): {e}")


def purge_all_data(client: bigquery.Client, table_id: str) -> None:
    """Delete ALL data from table."""
    print(f"\n❌ PURGING all data...")
    print(f"   Table: {table_id}")
    
    response = input("   ⚠️  Delete ALL sentiment data? [y/N]: ")
    if response.lower() != 'y':
        print("   Cancelled")
        return
    
    try:
        client.query(f"TRUNCATE TABLE `{table_id}`").result()
        print(f"   ✓ Purged all data")
    except Exception as e:
        print(f"   ⚠️  Purge failed: {e}")


def get_latest_timestamp(client: bigquery.Client, table_id: str) -> Optional[str]:
    """Get latest timestamp from table."""
    query = f"SELECT MAX(DATE(timestamp)) as max_date FROM `{table_id}`"
    
    try:
        result = client.query(query).to_dataframe()
        if not result.empty and result['max_date'].iloc[0] is not None:
            return result['max_date'].iloc[0].strftime('%Y-%m-%d')
    except Exception as e:
        print(f"   ⚠️  Could not get latest timestamp: {e}")
    
    return None



def main():
    parser = argparse.ArgumentParser(
        description='Fetch GDELT GKG sentiment data from BigQuery and aggregate to 15-minute intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Fetch using config
  python gdelt_load.py --config configs/gdelt.yaml
  
  # Fetch last 7 days
  python gdelt_load.py --start-date 2025-11-11 --end-date 2025-11-18
  
  # Custom topics
  python gdelt_load.py --start-date 2025-11-01 --end-date 2025-11-18 \
    --topics SPY FEDERAL_RESERVE INFLATION
  
  # Flush and reload
  python gdelt_load.py --start-date 2025-11-11 --end-date 2025-11-18 --flush
  
  # Top-up with new data
  python gdelt_load.py --config configs/gdelt.yaml --top-up
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gdelt.yaml',
        help='Path to GDELT YAML config file'
    )
    parser.add_argument(
        '--start-date',
        '--start',
        dest='start_date',
        type=str,
        required=False,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        '--end',
        dest='end_date',
        type=str,
        required=False,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--topics',
        nargs='+',
        default=None,
        help=f'Topics/themes to track (default from config)'
    )
    parser.add_argument(
        '--flush',
        action='store_true',
        help='Delete existing data for date range before loading'
    )
    parser.add_argument(
        '--purge',
        action='store_true',
        help='Delete ALL data - requires confirmation'
    )
    parser.add_argument(
        '--top-up',
        action='store_true',
        help='Auto-detect latest and fetch new data only'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Skip auto-merge staging to main'
    )
    parser.add_argument(
        '--skip-bigquery',
        action='store_true',
        help='Skip BigQuery upload'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Clear all existing data for this frequency before loading'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be saved without actually saving'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = None
    topics = args.topics
    start_date = args.start_date
    end_date = args.end_date
    
    if args.config:
        try:
            config = load_config(args.config)
            print(f"📋 Loaded config: {args.config}")
            
            if not topics and 'topics' in config:
                topics = config['topics']
                print(f"  Topics: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")
            
            if not start_date and 'date_range' in config:
                start_date = config['date_range'].get('start_date')
            
            if not end_date and 'date_range' in config:
                end_date_cfg = config['date_range'].get('end_date')
                end_date = datetime.now().strftime('%Y-%m-%d') if end_date_cfg is None else end_date_cfg
            
            # Read frequency from nested config (aggregation.frequency)
            if 'aggregation' in config and 'frequency' in config['aggregation']:
                frequency = config['aggregation']['frequency']
                print(f"  Frequency: {frequency}")
            
        except FileNotFoundError:
            if args.config != 'configs/gdelt.yaml':
                print(f"⚠️  Config not found: {args.config}")
    
    if not topics:
        topics = DEFAULT_TOPICS
    
    # Set default frequency if not set
    if 'frequency' not in locals():
        frequency = '1d'  # Default to native GDELT frequency
    
    # Handle modes
    if args.purge:
        start_date = start_date or '2000-01-01'
        end_date = end_date or '2000-01-01'
    elif args.top_up:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            print(f"📅 Top-up to: {end_date}")
        if not start_date:
            start_date = '2016-01-01'
    else:
        if not start_date or not end_date:
            print("❌ Error: --start-date and --end-date required")
            print("   Or use --config to load from file")
            sys.exit(1)
    
    # Validate dates
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end < start and not args.top_up:
            print("Error: end_date must be >= start_date")
            sys.exit(1)
            
        # Warn if date range is very large (expensive query)
        days = (end - start).days
        if days > 90:
            print(f"\n⚠️  Large date range ({days} days) - this query may be expensive")
            response = input("   Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Cancelled")
                sys.exit(0)
                
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    # Validate project ID
    if not PROJECT_ID and not args.skip_bigquery:
        print("\nError: GCP_PROJECT_ID environment variable not set")
        print("Either set it in .env or use --skip-bigquery flag")
        sys.exit(1)
    
    # Handle purge
    if args.purge:
        if not PROJECT_ID:
            print("❌ Error: GCP_PROJECT_ID required for purge")
            sys.exit(1)
        client = bigquery.Client(project=PROJECT_ID)
        purge_all_data(client, f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}")
        if args.skip_bigquery:
            print("\n✨ Purge complete!")
            sys.exit(0)
    
    # Handle top-up
    if args.top_up:
        if not PROJECT_ID:
            print("❌ Error: GCP_PROJECT_ID required for top-up")
            sys.exit(1)
        client = bigquery.Client(project=PROJECT_ID)
        latest = get_latest_timestamp(client, f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}")
        if latest:
            start_date = latest
            print(f"✨ Top-up from {start_date} to {end_date}")
        else:
            print(f"⚠️  No data, fetching: {start_date} to {end_date}")
    
    # Fetch sentiment data (15m intervals from BigQuery)
    df = fetch_gdelt_sentiment(
        start_date=start_date,
        end_date=end_date,
        topics=topics,
        project_id=PROJECT_ID
    )
    
    if df.empty:
        print("\n⚠️  No data collected, exiting")
        sys.exit(0)
    
    # Resample to target frequency if needed
    if frequency != '15m':
        df = resample_to_frequency(df, frequency, method='mean')
        # Update frequency column to match resampled frequency
        df['frequency'] = frequency
        # Recalculate date column after resampling
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Save to BigQuery (unless skipped)
    if not args.skip_bigquery:
        # Handle --reload: clear all data for this frequency first
        if args.reload and not args.dry_run:
            client = bigquery.Client(project=PROJECT_ID)
            main_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
            clear_frequency_data(client, main_table_id, frequency)
        
        save_to_bigquery(
            df,
            PROJECT_ID,
            DATASET_ID,
            dry_run=args.dry_run,
            auto_merge=not args.no_merge,
            flush_existing=args.flush,
            purge_all=False,
            start_date=start_date,
            end_date=end_date
        )
    
    print("\n" + "="*80)
    print("  ✨ DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
