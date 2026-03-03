# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import argparse
import gzip
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from joblib import Parallel, delayed
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("TS_HAYSTACK_DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))


# ---------------------------
# Constants
# ---------------------------

CAPTURE24_ZIP_PATH = os.path.join(DATA_DIR, "capture24.zip")
CAPTURE24_DATA_DIR = os.path.join(DATA_DIR, "capture24")
CAPTURE24_DOWNLOAD_URL = (
    "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001"
    "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"
)


# ---------------------------
# Helper Functions
# ---------------------------

def download_capture24_zip() -> None:
    """Download the Capture-24 dataset ZIP file (~6.9GB)."""
    os.makedirs(os.path.dirname(CAPTURE24_ZIP_PATH), exist_ok=True)
    print(f"Downloading capture24.zip (~6.9GB) to {CAPTURE24_ZIP_PATH}...")
    print("This may take a while...")

    with tqdm(total=6.9e9, unit="B", unit_scale=True, unit_divisor=1024,
              miniters=1, desc="Downloading capture24.zip") as pbar:
        urllib.request.urlretrieve(
            CAPTURE24_DOWNLOAD_URL,
            filename=CAPTURE24_ZIP_PATH,
            reporthook=lambda b, bsize, tsize: pbar.update(bsize)
        )
    print(f"Download complete: {CAPTURE24_ZIP_PATH}")


def get_sensor_data_dir(downsample_hz: int = 100) -> str:
    """
    Get path for sensor data directory based on sampling frequency.

    Args:
        downsample_hz: Sampling frequency in Hz (default: 100)

    Returns:
        Path to sensor data directory
    """
    return os.path.join(CAPTURE24_DATA_DIR, f"sensor_data_{downsample_hz}hz")


def is_data_ready(downsample_hz: int = 100) -> bool:
    """
    Check if Parquet files exist for the given sampling frequency.

    Args:
        downsample_hz: Sampling frequency in Hz (default: 100)

    Returns:
        True if all required files exist
    """
    participants_path = os.path.join(CAPTURE24_DATA_DIR, "participants.parquet")
    label_mappings_path = os.path.join(CAPTURE24_DATA_DIR, "label_mappings.parquet")
    sensor_data_dir = get_sensor_data_dir(downsample_hz)
    sensor_data_exists = os.path.isdir(sensor_data_dir) and len(list(Path(sensor_data_dir).glob("pid=P*/data.parquet"))) > 0

    return (
        os.path.exists(participants_path)
        and os.path.exists(label_mappings_path)
        and sensor_data_exists
    )


def parse_timestamp(timestamp_str: str) -> int:
    """Convert datetime string to Unix timestamp in milliseconds."""
    dt = pd.to_datetime(timestamp_str)
    return int(dt.timestamp() * 1000)


def process_participant_file(
    zip_path: str,
    csv_gz_name: str,
    pid: str,
    downsample_hz: int = 100
) -> None:
    """
    Extract and convert a single participant's gzipped CSV to Parquet.

    Args:
        zip_path: Path to the ZIP file
        csv_gz_name: Name of the .csv.gz file in the ZIP
        pid: Participant ID (e.g., "P001")
        downsample_hz: Target sampling frequency in Hz (default: 100 = no downsampling)
    """
    print(f"Processing {pid}...")

    # Create participant directory
    sensor_data_dir = get_sensor_data_dir(downsample_hz)
    pid_dir = os.path.join(sensor_data_dir, f"pid={pid}")
    os.makedirs(pid_dir, exist_ok=True)

    output_path = os.path.join(pid_dir, "data.parquet")

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"  {pid} already exists, skipping")
        return

    # Open ZIP file and extract gzipped CSV
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        with zip_file.open(csv_gz_name) as gz_file:
            with gzip.open(gz_file, 'rt') as csv_file:
                # Read CSV in chunks to manage memory
                chunk_size = 100000
                chunks = []

                # Use tqdm for chunk processing progress
                chunk_iterator = pd.read_csv(csv_file, chunksize=chunk_size)
                for chunk in tqdm(chunk_iterator, desc=f"  {pid} chunks", leave=False, unit="chunk"):
                    # Convert timestamp to milliseconds
                    chunk['timestamp_ms'] = chunk['time'].apply(parse_timestamp)

                    # Select and rename columns, making explicit copy to avoid pandas warnings
                    chunk = chunk[['timestamp_ms', 'x', 'y', 'z', 'annotation']].copy()

                    # Convert to float32 for storage efficiency
                    chunk['x'] = chunk['x'].astype('float32')
                    chunk['y'] = chunk['y'].astype('float32')
                    chunk['z'] = chunk['z'].astype('float32')

                    # Downsample if requested (100Hz is the original frequency)
                    if downsample_hz < 100:
                        downsample_factor = 100 // downsample_hz
                        chunk = chunk.iloc[::downsample_factor].reset_index(drop=True)

                    chunks.append(chunk)

                # Combine all chunks
                df = pd.concat(chunks, ignore_index=True)

                # Write to Parquet with snappy compression
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table,
                    output_path,
                    compression='snappy',
                    use_dictionary=True
                )

                sampling_info = f" at {downsample_hz}Hz" if downsample_hz < 100 else ""
                print(f"  {pid}: {len(df):,} samples written ({len(chunks)} chunks){sampling_info}")


def extract_and_convert_to_parquet(
    max_participants: Optional[int] = None,
    downsample_hz: int = 100,
    n_jobs: int = 1
) -> None:
    """
    Extract Capture-24 ZIP and convert all participant data to Parquet format.

    Args:
        max_participants: Optional limit on number of participants to process (for testing)
        downsample_hz: Target sampling frequency in Hz (default: 100 = no downsampling)
        n_jobs: Number of parallel jobs for processing participants (default: 1)

    Structure created:
        data/capture24/
        ├── participants.parquet
        ├── label_mappings.parquet
        └── sensor_data_{downsample_hz}hz/
            ├── pid=P001/data.parquet
            ├── pid=P002/data.parquet
            └── ...
    """
    if not os.path.exists(CAPTURE24_ZIP_PATH):
        print(f"Capture-24 ZIP not found at {CAPTURE24_ZIP_PATH}, downloading...")
        download_capture24_zip()

    os.makedirs(CAPTURE24_DATA_DIR, exist_ok=True)
    sensor_data_dir = get_sensor_data_dir(downsample_hz)
    os.makedirs(sensor_data_dir, exist_ok=True)

    print(f"Extracting and converting Capture-24 dataset from {CAPTURE24_ZIP_PATH}")

    with zipfile.ZipFile(CAPTURE24_ZIP_PATH, 'r') as zip_file:
        # Get list of all files in ZIP
        all_files = zip_file.namelist()

        # Extract metadata files if not already present
        metadata_path = os.path.join(CAPTURE24_DATA_DIR, "participants.parquet")
        if not os.path.exists(metadata_path):
            print("Extracting metadata.csv...")
            metadata_csv = [f for f in all_files if f.endswith('metadata.csv')][0]
            with zip_file.open(metadata_csv) as f:
                metadata_df = pd.read_csv(f)
                table = pa.Table.from_pandas(metadata_df)
                pq.write_table(table, metadata_path, compression='snappy')
            print(f"  Metadata saved: {len(metadata_df)} participants")

        # Extract label mappings if not already present
        labels_path = os.path.join(CAPTURE24_DATA_DIR, "label_mappings.parquet")
        if not os.path.exists(labels_path):
            print("Extracting annotation-label-dictionary.csv...")
            labels_csv = [f for f in all_files if f.endswith('annotation-label-dictionary.csv')][0]
            with zip_file.open(labels_csv) as f:
                labels_df = pd.read_csv(f)
                # Rename columns to snake_case
                labels_df.columns = [
                    'annotation',
                    'label_willetts_specific_2018',
                    'label_willetts_met_2018',
                    'label_doherty_specific_2018',
                    'label_willetts_2018',
                    'label_doherty_2018',
                    'label_walmsley_2020'
                ]
                table = pa.Table.from_pandas(labels_df)
                pq.write_table(table, labels_path, compression='snappy')
            print(f"  Label mappings saved: {len(labels_df)} annotations")

    # Process all participant CSV.gz files
    participant_files = sorted([f for f in all_files if f.endswith('.csv.gz')])

    # Limit number of participants if specified
    if max_participants is not None:
        participant_files = participant_files[:max_participants]
        print(f"\nProcessing {len(participant_files)} participant files (limited for testing)...")
    else:
        print(f"\nProcessing {len(participant_files)} participant files...")

    # Process participants in parallel
    Parallel(n_jobs=n_jobs)(
        delayed(process_participant_file)(
            CAPTURE24_ZIP_PATH,
            csv_gz_name,
            os.path.basename(csv_gz_name).replace('.csv.gz', ''),
            downsample_hz
        )
        for csv_gz_name in tqdm(participant_files, desc="Processing participants")
    )

    print("\nCapture-24 dataset extraction complete!")


def ensure_capture24_data(
    max_participants: Optional[int] = None,
    downsample_hz: int = 100,
    n_jobs: int = 1,
    overwrite: bool = False
) -> None:
    """
    Main entry point: ensure Capture-24 data is extracted and converted to Parquet.

    Args:
        max_participants: Optional limit on number of participants to process (for testing)
        downsample_hz: Target sampling frequency in Hz (default: 100 = no downsampling)
        n_jobs: Number of parallel jobs for processing participants (default: 1)
        overwrite: Force re-extraction even if data exists (default: False)

    If Parquet files already exist and overwrite is False, does nothing.
    Otherwise, extracts from ZIP and converts to Parquet format.
    """
    if is_data_ready(downsample_hz) and not overwrite:
        print(f"Capture-24 Parquet data already exists at {downsample_hz}Hz")
        return

    if overwrite:
        print("Overwrite flag set, re-extracting from ZIP...")
    else:
        print("Capture-24 Parquet data not found, extracting from ZIP...")

    extract_and_convert_to_parquet(max_participants=max_participants, downsample_hz=downsample_hz, n_jobs=n_jobs)


# ---------------------------
# Data Loading Functions
# ---------------------------

def load_participants() -> pl.DataFrame:
    """Load participant metadata as Polars DataFrame."""
    participants_path = os.path.join(CAPTURE24_DATA_DIR, "participants.parquet")
    return pl.read_parquet(participants_path)


def load_label_mappings() -> pl.DataFrame:
    """Load annotation-to-label mappings as Polars DataFrame."""
    labels_path = os.path.join(CAPTURE24_DATA_DIR, "label_mappings.parquet")
    return pl.read_parquet(labels_path)


def load_participant_sensor_data(
    pid: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    downsample_hz: int = 100
) -> pl.DataFrame:
    """
    Load sensor data for a single participant.

    Args:
        pid: Participant ID (e.g., "P001")
        start_ms: Optional start timestamp in milliseconds (inclusive)
        end_ms: Optional end timestamp in milliseconds (exclusive)
        downsample_hz: Sampling frequency in Hz (default: 100)

    Returns:
        Polars DataFrame with columns: timestamp_ms, x, y, z, annotation
    """
    sensor_data_dir = get_sensor_data_dir(downsample_hz)
    participant_path = os.path.join(sensor_data_dir, f"pid={pid}", "data.parquet")

    if not os.path.exists(participant_path):
        raise FileNotFoundError(f"Data for {pid} not found at {participant_path}")

    # Use lazy evaluation for efficient filtering
    query = pl.scan_parquet(participant_path)

    if start_ms is not None:
        query = query.filter(pl.col("timestamp_ms") >= start_ms)

    if end_ms is not None:
        query = query.filter(pl.col("timestamp_ms") < end_ms)

    return query.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and convert Capture-24 dataset to Parquet format"
    )
    parser.add_argument(
        '--max-participants', '-n',
        type=int,
        default=None,
        help='Maximum number of participants to process (for testing)'
    )
    parser.add_argument(
        '--downsample-hz', '-d',
        type=int,
        default=100,
        help='Target sampling frequency in Hz (default: 100)'
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Force re-extraction even if data exists'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Capture-24 Dataset Extraction")
    print("=" * 60)

    if args.max_participants is not None:
        print(f"Limiting extraction to {args.max_participants} participants for testing")

    if args.downsample_hz < 100:
        print(f"Downsampling to {args.downsample_hz}Hz")

    if args.n_jobs > 1:
        print(f"Using {args.n_jobs} parallel jobs")

    print()

    # Extract and convert data
    ensure_capture24_data(
        max_participants=args.max_participants,
        downsample_hz=args.downsample_hz,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite
    )

    # Quick verification
    print("\nVerifying extraction...")
    participants = load_participants()
    print(f"Loaded {len(participants)} participants")

    labels = load_label_mappings()
    print(f"Loaded {len(labels)} annotation mappings")

    print("\nTesting sensor data load for P001...")
    df = load_participant_sensor_data("P001")
    print(f"P001 has {len(df):,} samples")
    print(f"  Columns: {df.columns}")
    print(f"  First 5 rows:\n{df.head()}")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
