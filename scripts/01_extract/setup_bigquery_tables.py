"""Setup BigQuery dataset and tables for Polygon data ingestion.

Creates the raw_dataset with raw_ohlcv and raw_ohlcv_staging tables.
Includes frequency column for tracking data granularity (daily, hourly, 5m, etc.).
"""
import os
import sys
from google.cloud import bigquery

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
TABLE_NAME = os.environ.get('BQ_TABLE', 'raw_ohlcv')
STAGING_TABLE_NAME = os.environ.get('BQ_STAGING_TABLE', 'raw_ohlcv_staging')
INDICATORS_TABLE = os.environ.get('BQ_INDICATORS_TABLE', 'technical_indicators')
INDICATORS_STAGING_TABLE = os.environ.get('BQ_INDICATORS_STAGING_TABLE', 'technical_indicators_staging')
SYNTHETIC_INDICATORS_TABLE = os.environ.get('BQ_SYNTHETIC_TABLE', 'synthetic_indicators')
SYNTHETIC_INDICATORS_STAGING_TABLE = os.environ.get('BQ_SYNTHETIC_STAGING_TABLE', 'synthetic_indicators_staging')
GDELT_TABLE = os.environ.get('BQ_GDELT_TABLE', 'gdelt_sentiment')
GDELT_STAGING_TABLE = os.environ.get('BQ_GDELT_STAGING_TABLE', 'gdelt_sentiment_staging')
GDELT_METADATA_TABLE = os.environ.get('BQ_GDELT_METADATA_TABLE', 'gdelt_query_metadata')

# OHLCV table schema with frequency column
OHLCV_SCHEMA = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED", description="Ticker symbol (e.g., SPY, SPY)"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="Exact timestamp of the bar"),
    bigquery.SchemaField("date", "DATE", mode="REQUIRED", description="Date (for partitioning and queries)"),
    bigquery.SchemaField("frequency", "STRING", mode="REQUIRED", description="Data frequency: daily, hourly, 5m, etc."),
    bigquery.SchemaField("open", "FLOAT64", mode="NULLABLE", description="Opening price"),
    bigquery.SchemaField("high", "FLOAT64", mode="NULLABLE", description="Highest price"),
    bigquery.SchemaField("low", "FLOAT64", mode="NULLABLE", description="Lowest price"),
    bigquery.SchemaField("close", "FLOAT64", mode="NULLABLE", description="Closing price"),
    bigquery.SchemaField("volume", "FLOAT64", mode="NULLABLE", description="Trading volume"),
    bigquery.SchemaField("vwap", "FLOAT64", mode="NULLABLE", description="Volume weighted average price"),
    bigquery.SchemaField("transactions", "INTEGER", mode="NULLABLE", description="Number of transactions"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE", description="When this row was ingested"),
]

# Synthetic indicators table schema (wide format - computed from raw OHLCV)
SYNTHETIC_INDICATORS_SCHEMA = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED", description="Ticker symbol"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="Exact timestamp"),
    bigquery.SchemaField("date", "DATE", mode="REQUIRED", description="Date (for partitioning)"),
    bigquery.SchemaField("frequency", "STRING", mode="REQUIRED", description="Data frequency"),
    # Price-based indicators
    bigquery.SchemaField("returns", "FLOAT64", mode="NULLABLE", description="Simple returns (close to close)"),
    bigquery.SchemaField("returns_5", "FLOAT64", mode="NULLABLE", description="5-period returns"),
    bigquery.SchemaField("log_returns", "FLOAT64", mode="NULLABLE", description="Log returns"),
    bigquery.SchemaField("volatility_5", "FLOAT64", mode="NULLABLE", description="5-period rolling volatility"),
    bigquery.SchemaField("volatility_10", "FLOAT64", mode="NULLABLE", description="10-period rolling volatility"),
    bigquery.SchemaField("volatility_20", "FLOAT64", mode="NULLABLE", description="20-period rolling volatility"),
    # Momentum indicators
    bigquery.SchemaField("momentum_5", "FLOAT64", mode="NULLABLE", description="5-period price momentum"),
    bigquery.SchemaField("momentum_10", "FLOAT64", mode="NULLABLE", description="10-period price momentum"),
    bigquery.SchemaField("roc_5", "FLOAT64", mode="NULLABLE", description="5-period rate of change"),
    bigquery.SchemaField("roc_10", "FLOAT64", mode="NULLABLE", description="10-period rate of change"),
    # Volume indicators
    bigquery.SchemaField("volume_ma_5", "FLOAT64", mode="NULLABLE", description="5-period volume moving average"),
    bigquery.SchemaField("volume_ma_10", "FLOAT64", mode="NULLABLE", description="10-period volume moving average"),
    bigquery.SchemaField("volume_ratio", "FLOAT64", mode="NULLABLE", description="Current volume / 20-period avg volume"),
    bigquery.SchemaField("obv", "FLOAT64", mode="NULLABLE", description="On-Balance Volume"),
    # Price range indicators
    bigquery.SchemaField("atr_14", "FLOAT64", mode="NULLABLE", description="14-period Average True Range"),
    bigquery.SchemaField("high_low_ratio", "FLOAT64", mode="NULLABLE", description="(High - Low) / Close"),
    # Bollinger Bands
    bigquery.SchemaField("bb_upper_20", "FLOAT64", mode="NULLABLE", description="20-period Bollinger Band upper (2 std)"),
    bigquery.SchemaField("bb_middle_20", "FLOAT64", mode="NULLABLE", description="20-period Bollinger Band middle (SMA)"),
    bigquery.SchemaField("bb_lower_20", "FLOAT64", mode="NULLABLE", description="20-period Bollinger Band lower (2 std)"),
    bigquery.SchemaField("bb_width", "FLOAT64", mode="NULLABLE", description="Bollinger Band width"),
    # Simple Moving Averages
    bigquery.SchemaField("sma_10", "FLOAT64", mode="NULLABLE", description="10-period Simple Moving Average"),
    bigquery.SchemaField("sma_20", "FLOAT64", mode="NULLABLE", description="20-period Simple Moving Average"),
    bigquery.SchemaField("sma_50", "FLOAT64", mode="NULLABLE", description="50-period Simple Moving Average"),
    bigquery.SchemaField("sma_200", "FLOAT64", mode="NULLABLE", description="200-period Simple Moving Average"),
    # Exponential Moving Averages
    bigquery.SchemaField("ema_12", "FLOAT64", mode="NULLABLE", description="12-period Exponential Moving Average"),
    bigquery.SchemaField("ema_26", "FLOAT64", mode="NULLABLE", description="26-period Exponential Moving Average"),
    # MACD indicators
    bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE", description="MACD line (12-26 EMA difference)"),
    bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE", description="MACD signal line (9-period EMA of MACD)"),
    bigquery.SchemaField("macd_hist", "FLOAT64", mode="NULLABLE", description="MACD histogram (MACD - signal)"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE", description="When this row was computed"),
]

# Technical indicators table schema (wide format)
INDICATORS_SCHEMA = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED", description="Ticker symbol"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="Exact timestamp"),
    bigquery.SchemaField("date", "DATE", mode="REQUIRED", description="Date (for partitioning)"),
    bigquery.SchemaField("frequency", "STRING", mode="REQUIRED", description="Data frequency"),
    # Simple Moving Averages
    bigquery.SchemaField("sma_10", "FLOAT64", mode="NULLABLE", description="SMA with 10 period window"),
    bigquery.SchemaField("sma_20", "FLOAT64", mode="NULLABLE", description="SMA with 20 period window"),
    bigquery.SchemaField("sma_50", "FLOAT64", mode="NULLABLE", description="SMA with 50 period window"),
    bigquery.SchemaField("sma_200", "FLOAT64", mode="NULLABLE", description="SMA with 200 period window"),
    # Exponential Moving Averages
    bigquery.SchemaField("ema_10", "FLOAT64", mode="NULLABLE", description="EMA with 10 period window"),
    bigquery.SchemaField("ema_20", "FLOAT64", mode="NULLABLE", description="EMA with 20 period window"),
    bigquery.SchemaField("ema_50", "FLOAT64", mode="NULLABLE", description="EMA with 50 period window"),
    bigquery.SchemaField("ema_200", "FLOAT64", mode="NULLABLE", description="EMA with 200 period window"),
    # Relative Strength Index
    bigquery.SchemaField("rsi_14", "FLOAT64", mode="NULLABLE", description="RSI with 14 period"),
    # MACD
    bigquery.SchemaField("macd_value", "FLOAT64", mode="NULLABLE", description="MACD value (12, 26, 9)"),
    bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE", description="MACD signal line"),
    bigquery.SchemaField("macd_histogram", "FLOAT64", mode="NULLABLE", description="MACD histogram"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE", description="When this row was ingested"),
]

# GDELT sentiment table schema (15-minute aggregated sentiment from GKG)
GDELT_SCHEMA = [
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="15-minute interval timestamp"),
    bigquery.SchemaField("date", "DATE", mode="REQUIRED", description="Date (for partitioning and queries)"),
    bigquery.SchemaField("frequency", "STRING", mode="REQUIRED", description="Data frequency: 15m (GDELT native interval)"),
    bigquery.SchemaField("topic_group_id", "STRING", mode="REQUIRED", description="Topic group ID (e.g., inflation_prices, fed_policy)"),
    bigquery.SchemaField("weighted_avg_tone", "FLOAT", mode="NULLABLE", description="Sentiment tone (-10 to +10, weighted by word count)"),
    bigquery.SchemaField("weighted_avg_positive", "FLOAT", mode="NULLABLE", description="Positive sentiment score (weighted)"),
    bigquery.SchemaField("weighted_avg_negative", "FLOAT", mode="NULLABLE", description="Negative sentiment score (weighted)"),
    bigquery.SchemaField("weighted_avg_polarity", "FLOAT", mode="NULLABLE", description="Sentiment polarity/strength (weighted)"),
    bigquery.SchemaField("num_articles", "INTEGER", mode="NULLABLE", description="Number of articles in this interval"),
    bigquery.SchemaField("num_sources", "INTEGER", mode="NULLABLE", description="Number of unique news sources"),
    bigquery.SchemaField("total_word_count", "INTEGER", mode="NULLABLE", description="Total word count across all articles"),
    bigquery.SchemaField("min_tone", "FLOAT", mode="NULLABLE", description="Minimum tone in interval (for quality checks)"),
    bigquery.SchemaField("max_tone", "FLOAT", mode="NULLABLE", description="Maximum tone in interval (for quality checks)"),
    bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP", mode="NULLABLE", description="When this data was ingested"),
]

# GDELT query metadata table schema (tracks data lineage and query configurations)
GDELT_METADATA_SCHEMA = [
    bigquery.SchemaField("query_id", "STRING", mode="REQUIRED", description="Unique ID for this query/load"),
    bigquery.SchemaField("topic_group_id", "STRING", mode="REQUIRED", description="Human-readable topic group ID (e.g., inflation_prices, fed_policy)"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED", description="When this data was loaded"),
    bigquery.SchemaField("start_date", "DATE", mode="REQUIRED", description="Start date of query range"),
    bigquery.SchemaField("end_date", "DATE", mode="REQUIRED", description="End date of query range"),
    bigquery.SchemaField("topics", "STRING", mode="REPEATED", description="List of topics queried"),
    bigquery.SchemaField("config_file", "STRING", mode="NULLABLE", description="Path to config file used"),
    bigquery.SchemaField("num_records", "INTEGER", mode="NULLABLE", description="Number of records loaded"),
    bigquery.SchemaField("num_articles", "INTEGER", mode="NULLABLE", description="Total number of articles processed"),
    bigquery.SchemaField("frequency", "STRING", mode="NULLABLE", description="Data frequency (15m, 1h, etc.)"),
    bigquery.SchemaField("notes", "STRING", mode="NULLABLE", description="Optional notes about this load"),
]


def create_dataset(client: bigquery.Client, dataset_id: str) -> bool:
    """Create BigQuery dataset if it doesn't exist."""
    try:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        dataset.description = "Inflation prediction - Raw OHLCV data from Polygon.io"
        
        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"✅ Dataset {dataset_id} ready")
        return True
    except Exception as e:
        print(f"❌ Error creating dataset: {str(e)}")
        return False


def create_table(client: bigquery.Client, table_id: str, schema: list, partition_field: str = "date", clustering_fields: list = ["ticker", "frequency"]) -> bool:
    """Create BigQuery table if it doesn't exist."""
    try:
        table = bigquery.Table(table_id, schema=schema)
        
        # Partition by date for better query performance
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
        
        # Cluster by ticker and frequency for even better performance
        table.clustering_fields = clustering_fields
        
        table = client.create_table(table, exists_ok=True)
        print(f"✅ Table {table_id} ready")
        print(f"   - Partitioned by: {partition_field} (daily)")
        print(f"   - Clustered by: {', '.join(clustering_fields)}")
        return True
    except Exception as e:
        print(f"❌ Error creating table: {str(e)}")
        return False


def create_staging_table(client: bigquery.Client, table_id: str, schema: list, clustering_fields: list = ["ticker", "frequency"]) -> bool:
    """Create non-partitioned staging table to avoid partition modification quota limits.
    
    Staging tables are temporary and don't need partitioning. This avoids hitting
    the 1,500 partition modifications per day limit when loading many tickers.
    """
    try:
        table = bigquery.Table(table_id, schema=schema)
        
        # NO partitioning for staging tables (avoids quota limits)
        # Cluster only for better merge performance
        table.clustering_fields = clustering_fields
        
        table = client.create_table(table, exists_ok=True)
        print(f"✅ Staging table {table_id} ready")
        print(f"   - NOT partitioned (avoids quota limits)")
        print(f"   - Clustered by: {', '.join(clustering_fields)}")
        return True
    except Exception as e:
        print(f"❌ Error creating staging table: {str(e)}")
        return False


def verify_tables(client: bigquery.Client, dataset_id: str, main_table: str, staging_table: str):
    """Verify that tables exist and show their properties."""
    print("\n" + "=" * 80)
    print("   Verification")
    print("=" * 80)
    print()
    
    for table_name in [main_table, staging_table]:
        table_id = f"{dataset_id}.{table_name}"
        try:
            table = client.get_table(table_id)
            print(f"✅ {table_id}")
            print(f"   Rows: {table.num_rows:,}")
            print(f"   Size: {table.num_bytes / (1024*1024):.2f} MB")
            print(f"   Columns: {len(table.schema)}")
            print(f"   Created: {table.created}")
            print()
        except Exception as e:
            print(f"❌ {table_id}: {str(e)}")
            print()


def main():
    """Main setup function."""
    print("=" * 80)
    print("   Inflation prediction - BigQuery Setup")
    print("=" * 80)
    print()
    
    # Validate configuration
    if not PROJECT_ID:
        print("❌ Error: GCP_PROJECT_ID not set in environment")
        print("   Please set it in your .env file or environment")
        sys.exit(1)
    
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"OHLCV tables: {TABLE_NAME}, {STAGING_TABLE_NAME}")
    print(f"Polygon indicator tables: {INDICATORS_TABLE}, {INDICATORS_STAGING_TABLE}")
    print(f"Synthetic indicator tables: {SYNTHETIC_INDICATORS_TABLE}, {SYNTHETIC_INDICATORS_STAGING_TABLE}")
    print(f"GDELT sentiment tables: {GDELT_TABLE}, {GDELT_STAGING_TABLE}")
    print()
    
    # Initialize client
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print("✅ Connected to BigQuery")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to BigQuery: {str(e)}")
        print("   Make sure GOOGLE_APPLICATION_CREDENTIALS is set")
        sys.exit(1)
    
    # Create dataset
    print("Creating dataset...")
    dataset_id = f"{PROJECT_ID}.{DATASET_ID}"
    if not create_dataset(client, dataset_id):
        sys.exit(1)
    print()
    
    # Create OHLCV main table
    print("Creating OHLCV main table...")
    main_table_id = f"{dataset_id}.{TABLE_NAME}"
    if not create_table(client, main_table_id, OHLCV_SCHEMA):
        sys.exit(1)
    print()
    
    # Create OHLCV staging table (non-partitioned to avoid quota limits)
    print("Creating OHLCV staging table...")
    staging_table_id = f"{dataset_id}.{STAGING_TABLE_NAME}"
    if not create_staging_table(client, staging_table_id, OHLCV_SCHEMA):
        sys.exit(1)
    print()
    
    # Create technical indicators main table
    print("Creating technical indicators main table...")
    indicators_table_id = f"{dataset_id}.{INDICATORS_TABLE}"
    if not create_table(client, indicators_table_id, INDICATORS_SCHEMA):
        sys.exit(1)
    print()
    
    # Create technical indicators staging table (non-partitioned to avoid quota limits)
    print("Creating technical indicators staging table...")
    indicators_staging_id = f"{dataset_id}.{INDICATORS_STAGING_TABLE}"
    if not create_staging_table(client, indicators_staging_id, INDICATORS_SCHEMA):
        sys.exit(1)
    print()
    
    # Create synthetic indicators main table
    print("Creating synthetic indicators main table...")
    synthetic_indicators_table_id = f"{dataset_id}.{SYNTHETIC_INDICATORS_TABLE}"
    if not create_table(client, synthetic_indicators_table_id, SYNTHETIC_INDICATORS_SCHEMA):
        sys.exit(1)
    print()
    
    # Create synthetic indicators staging table (non-partitioned to avoid quota limits)
    print("Creating synthetic indicators staging table...")
    synthetic_indicators_staging_id = f"{dataset_id}.{SYNTHETIC_INDICATORS_STAGING_TABLE}"
    if not create_staging_table(client, synthetic_indicators_staging_id, SYNTHETIC_INDICATORS_SCHEMA):
        sys.exit(1)
    print()
    
    # Create GDELT sentiment main table
    print("Creating GDELT sentiment main table...")
    gdelt_table_id = f"{dataset_id}.{GDELT_TABLE}"
    if not create_table(client, gdelt_table_id, GDELT_SCHEMA, partition_field="date", clustering_fields=["topic_group_id", "frequency"]):
        sys.exit(1)
    print()
    
    # Create GDELT sentiment staging table (non-partitioned to avoid quota limits)
    print("Creating GDELT sentiment staging table...")
    gdelt_staging_id = f"{dataset_id}.{GDELT_STAGING_TABLE}"
    if not create_staging_table(client, gdelt_staging_id, GDELT_SCHEMA, clustering_fields=["topic_group_id", "frequency"]):
        sys.exit(1)
    print()
    
    # Create GDELT query metadata table (tracks data lineage)
    print("Creating GDELT query metadata table...")
    gdelt_metadata_id = f"{dataset_id}.{GDELT_METADATA_TABLE}"
    if not create_table(client, gdelt_metadata_id, GDELT_METADATA_SCHEMA, partition_field="load_timestamp", clustering_fields=["topic_group_id"]):
        sys.exit(1)
    print()
    
    # Verify all tables
    print("Verifying all tables...")
    verify_tables(client, dataset_id, TABLE_NAME, STAGING_TABLE_NAME)
    verify_tables(client, dataset_id, INDICATORS_TABLE, INDICATORS_STAGING_TABLE)
    verify_tables(client, dataset_id, SYNTHETIC_INDICATORS_TABLE, SYNTHETIC_INDICATORS_STAGING_TABLE)
    verify_tables(client, dataset_id, GDELT_TABLE, GDELT_STAGING_TABLE)
    
    print("=" * 80)
    print("   Setup Complete! ✅")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run tickers_load_polygon.py to fetch data")
    print("  2. Use extract_tickers.py to export to local parquet")
    print()
    print("Examples:")
    print("  # Fetch raw OHLCV data")
    print("  python scripts/01_extract/tickers_load_polygon.py \\")
    print("    --tickers SPY \\")
    print("    --start 2024-01-01 \\")
    print("    --end 2024-12-31 \\")
    print("    --frequency daily")
    print()
    print("  # Verify data (excluding weekend gaps for stock market data)")
    print("  python scripts/01_extract/tickers_verify_polygon.py \\")
    print("    --ticker SPY \\")
    print("    --frequency daily \\")
    print("    --check-gaps \\")
    print("    --exclude-weekends")
    print()
    print("  # Fetch technical indicators")
    print("  python scripts/01_extract/tickers_load_polygon.py \\")
    print("    --tickers SPY \\")
    print("    --start 2024-01-01 \\")
    print("    --end 2024-12-31 \\")
    print("    --frequency daily \\")
    print("    --data-type indicators")
    print()
    print("  # Compute synthetic indicators from raw OHLCV")
    print("  python scripts/01_extract/tickers_load_synthetic.py \\")
    print("    --tickers SPY \\")
    print("    --start 2024-01-01 \\")
    print("    --end 2024-12-31 \\")
    print("    --frequency daily")
    print()


if __name__ == '__main__':
    main()