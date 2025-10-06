# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, io, re
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Configuration ===
# Global configuration
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Plot configuration
DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100

# Display configuration
MAX_RESULT_DISPLAY_LENGTH = 300

class ModelConfig:
    """Configuration class for different models."""
    
    def __init__(self, model_name: str, model_url: str, model_print_name: str, 
                 # QueryUnderstandingTool parameters
                 query_understanding_temperature: float = 0.1,
                 query_understanding_max_tokens: int = 5,
                 # CodeGenerationAgent parameters
                 code_generation_temperature: float = 0.2,
                 code_generation_max_tokens: int = 1024,
                 # ReasoningAgent parameters
                 reasoning_temperature: float = 0.2,
                 reasoning_max_tokens: int = 1024,
                 # DataInsightAgent parameters
                 insights_temperature: float = 0.2,
                 insights_max_tokens: int = 512,
                 reasoning_false: str = "detailed thinking off",
                 reasoning_true: str = "detailed thinking on"):
        self.MODEL_NAME = model_name
        self.MODEL_URL = model_url
        self.MODEL_PRINT_NAME = model_print_name
        
        # Function-specific LLM parameters
        self.QUERY_UNDERSTANDING_TEMPERATURE = query_understanding_temperature
        self.QUERY_UNDERSTANDING_MAX_TOKENS = query_understanding_max_tokens
        self.CODE_GENERATION_TEMPERATURE = code_generation_temperature
        self.CODE_GENERATION_MAX_TOKENS = code_generation_max_tokens
        self.REASONING_TEMPERATURE = reasoning_temperature
        self.REASONING_MAX_TOKENS = reasoning_max_tokens
        self.INSIGHTS_TEMPERATURE = insights_temperature
        self.INSIGHTS_MAX_TOKENS = insights_max_tokens
        self.REASONING_FALSE = reasoning_false
        self.REASONING_TRUE = reasoning_true

# Predefined model configurations
MODEL_CONFIGS = {
    "llama-3-1-nemotron-ultra-v1": ModelConfig(
        model_name="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        model_url="https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1",
        model_print_name="NVIDIA Llama 3.1 Nemotron Ultra 253B v1",
        # QueryUnderstandingTool
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=5,
        # CodeGenerationAgent
        code_generation_temperature=0.2,
        code_generation_max_tokens=1024,
        # ReasoningAgent
        reasoning_temperature=0.6,
        reasoning_max_tokens=1024,
        # DataInsightAgent
        insights_temperature=0.2,
        insights_max_tokens=512,
        reasoning_false="detailed thinking off",
        reasoning_true="detailed thinking on"
    ),
    "llama-3-3-nemotron-super-v1-5": ModelConfig(
        model_name="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        model_url="https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5",
        model_print_name="NVIDIA Llama 3.3 Nemotron Super 49B v1.5",
        # QueryUnderstandingTool
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=5,
        # CodeGenerationAgent
        code_generation_temperature=0.0,
        code_generation_max_tokens=1024,
        # ReasoningAgent
        reasoning_temperature=0.6,
        reasoning_max_tokens=2048,
        # DataInsightAgent
        insights_temperature=0.2,
        insights_max_tokens=512,
        reasoning_false="/no_think",
        reasoning_true=""
    )
}

# Default configuration (can be changed via environment variable or UI)
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama-3-1-nemotron-ultra-v1")
Config = MODEL_CONFIGS.get(DEFAULT_MODEL, MODEL_CONFIGS["llama-3-1-nemotron-ultra-v1"])

# Initialize OpenAI client with configuration
if not API_KEY:
    st.error("""
    **API Key Required**
    
    Please set your NVIDIA API key as an environment variable:
    
    **Option 1: Set environment variable**
    ```bash
    export NVIDIA_API_KEY=your_nvidia_api_key_here
    ```
    
    **Option 2: Set OPENAI_API_KEY (alternative)**
    ```bash
    export OPENAI_API_KEY=your_nvidia_api_key_here
    ```
    
    Get your API key from: https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1?integrate_nim=true&hosted_api=true&modal=integrate-nim
    """)
    st.stop()

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def get_current_config():
    """Get the current model configuration based on session state."""
    # Always return the current model from session state
    if "current_model" in st.session_state:
        return MODEL_CONFIGS[st.session_state.current_model]
    
    return MODEL_CONFIGS[DEFAULT_MODEL]

# === UTILITY FUNCTIONS (Available in code execution) ==================

def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare two datasets and return a summary DataFrame with key statistics.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        on_columns: Optional list of columns to compare. If None, compares all common numeric columns.
    
    Returns:
        DataFrame with comparison statistics (count, mean, median, std, min, max) for each dataset
    """
    if on_columns is None:
        # Find common numeric columns
        numeric_cols_df1 = df1.select_dtypes(include=[np.number]).columns
        numeric_cols_df2 = df2.select_dtypes(include=[np.number]).columns
        on_columns = list(set(numeric_cols_df1) & set(numeric_cols_df2))
    
    comparison_data = []
    
    for col in on_columns:
        if col in df1.columns and col in df2.columns:
            stats = {
                'Column': col,
                'df1_count': df1[col].count(),
                'df2_count': df2[col].count(),
                'df1_mean': df1[col].mean() if pd.api.types.is_numeric_dtype(df1[col]) else None,
                'df2_mean': df2[col].mean() if pd.api.types.is_numeric_dtype(df2[col]) else None,
                'df1_median': df1[col].median() if pd.api.types.is_numeric_dtype(df1[col]) else None,
                'df2_median': df2[col].median() if pd.api.types.is_numeric_dtype(df2[col]) else None,
                'df1_std': df1[col].std() if pd.api.types.is_numeric_dtype(df1[col]) else None,
                'df2_std': df2[col].std() if pd.api.types.is_numeric_dtype(df2[col]) else None,
                'mean_diff': (df1[col].mean() - df2[col].mean()) if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]) else None,
            }
            comparison_data.append(stats)
    
    return pd.DataFrame(comparison_data)


def find_differences(df1: pd.DataFrame, df2: pd.DataFrame, 
                     key_columns: Optional[List[str]] = None,
                     threshold: float = 0.01,
                     max_examples: int = 100) -> dict:
    """
    Find differences between two datasets and determine if they are identical.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        key_columns: Columns to use as keys for matching rows. If None, checks complete equality.
        threshold: Relative threshold for considering numeric values different (default 1%)
        max_examples: Maximum number of concrete difference examples to return
    
    Returns:
        Dictionary containing:
        - 'identical': Boolean indicating if datasets are completely identical
        - 'shape_diff': Difference in dimensions
        - 'columns_only_in_df1': Columns unique to df1
        - 'columns_only_in_df2': Columns unique to df2
        - 'common_columns': Columns present in both
        - 'dtype_differences': Columns with different data types
        - 'value_differences': Statistics and concrete examples of value differences
        - 'summary': Human-readable summary of findings
    """
    result = {
        'identical': False,
        'shape_diff': {
            'df1_shape': df1.shape,
            'df2_shape': df2.shape,
            'row_diff': df1.shape[0] - df2.shape[0],
            'col_diff': df1.shape[1] - df2.shape[1]
        },
        'columns_only_in_df1': list(set(df1.columns) - set(df2.columns)),
        'columns_only_in_df2': list(set(df2.columns) - set(df1.columns)),
        'common_columns': list(set(df1.columns) & set(df2.columns)),
        'dtype_differences': {},
        'value_differences': {},
        'summary': ''
    }
    
    # Quick check for complete identity
    try:
        if df1.shape == df2.shape and set(df1.columns) == set(df2.columns):
            # Reorder df2 columns to match df1
            df2_ordered = df2[df1.columns]
            if df1.equals(df2_ordered):
                result['identical'] = True
                result['summary'] = "✓ Datasets are IDENTICAL - same shape, columns, and all values match exactly."
                return result
    except Exception:
        pass  # Continue with detailed comparison
    
    # Check dtype differences for common columns
    for col in result['common_columns']:
        if df1[col].dtype != df2[col].dtype:
            result['dtype_differences'][col] = {
                'df1_dtype': str(df1[col].dtype),
                'df2_dtype': str(df2[col].dtype)
            }
    
    # Find concrete value-level differences
    if result['common_columns']:
        diff_details = []
        
        if key_columns and all(k in result['common_columns'] for k in key_columns):
            # Use key columns for row matching
            merged = df1.merge(df2, on=key_columns, how='outer', indicator=True, suffixes=('_df1', '_df2'))
            
            result['value_differences']['only_in_df1'] = int(len(merged[merged['_merge'] == 'left_only']))
            result['value_differences']['only_in_df2'] = int(len(merged[merged['_merge'] == 'right_only']))
            result['value_differences']['in_both'] = int(len(merged[merged['_merge'] == 'both']))
            
            # Find value differences in matched rows
            matched_rows = merged[merged['_merge'] == 'both']
            value_cols = [c for c in result['common_columns'] if c not in key_columns]
            
            for col in value_cols[:10]:  # Check up to 10 columns
                if f'{col}_df1' in matched_rows.columns and f'{col}_df2' in matched_rows.columns:
                    # Compare values
                    diff_mask = matched_rows[f'{col}_df1'] != matched_rows[f'{col}_df2']
                    # Also check for NaN differences
                    nan_diff = matched_rows[f'{col}_df1'].isna() != matched_rows[f'{col}_df2'].isna()
                    diff_mask = diff_mask | nan_diff
                    
                    if diff_mask.any():
                        diff_count = diff_mask.sum()
                        examples = matched_rows[diff_mask].head(max_examples)
                        diff_details.append({
                            'column': col,
                            'different_count': int(diff_count),
                            'examples': examples[[*key_columns, f'{col}_df1', f'{col}_df2']].to_dict('records')
                        })
        else:
            # Position-based comparison (no key columns)
            if df1.shape == df2.shape:
                df2_ordered = df2[df1.columns] if set(df1.columns) == set(df2.columns) else df2
                
                # Track all differences across rows
                rows_with_diffs = set()
                
                # First pass: identify which rows have differences
                for col in result['common_columns']:
                    if col in df2_ordered.columns:
                        diff_mask = df1[col] != df2_ordered[col]
                        nan_df1 = df1[col].isna()
                        nan_df2 = df2_ordered[col].isna()
                        diff_mask = diff_mask & ~(nan_df1 & nan_df2)
                        
                        if diff_mask.any():
                            rows_with_diffs.update(diff_mask[diff_mask].index.tolist())
                
                # Second pass: collect detailed differences for each column
                for col in result['common_columns']:
                    if col in df2_ordered.columns:
                        diff_mask = df1[col] != df2_ordered[col]
                        nan_df1 = df1[col].isna()
                        nan_df2 = df2_ordered[col].isna()
                        diff_mask = diff_mask & ~(nan_df1 & nan_df2)
                        
                        if diff_mask.any():
                            diff_count = diff_mask.sum()
                            diff_indices = diff_mask[diff_mask].index.tolist()[:max_examples]
                            examples = []
                            for idx in diff_indices:
                                examples.append({
                                    'row_number': int(idx),
                                    'column': col,
                                    'df1_value': str(df1.loc[idx, col]),
                                    'df2_value': str(df2_ordered.loc[idx, col]),
                                    'difference': f"Row {idx}: {col} differs - df1='{df1.loc[idx, col]}' vs df2='{df2_ordered.loc[idx, col]}'"
                                })
                            diff_details.append({
                                'column': col,
                                'different_count': int(diff_count),
                                'total_rows': len(df1),
                                'percentage': round(100 * diff_count / len(df1), 2),
                                'examples': examples
                            })
                
                # Add summary of affected rows
                result['value_differences']['total_rows_with_differences'] = len(rows_with_diffs)
                result['value_differences']['affected_rows'] = sorted(list(rows_with_diffs))[:50]  # First 50 row numbers
            
            result['value_differences']['position_based'] = True
        
        result['value_differences']['concrete_differences'] = diff_details
    
    # Build summary
    summary_parts = []
    if result['shape_diff']['row_diff'] != 0 or result['shape_diff']['col_diff'] != 0:
        summary_parts.append(f"Shape differs: df1{df1.shape} vs df2{df2.shape}")
    if result['columns_only_in_df1']:
        summary_parts.append(f"Columns only in df1: {result['columns_only_in_df1']}")
    if result['columns_only_in_df2']:
        summary_parts.append(f"Columns only in df2: {result['columns_only_in_df2']}")
    if result['dtype_differences']:
        summary_parts.append(f"Data type differences in {len(result['dtype_differences'])} columns")
    if result['value_differences'].get('concrete_differences'):
        total_diff_cols = len(result['value_differences']['concrete_differences'])
        total_rows_diff = result['value_differences'].get('total_rows_with_differences', 0)
        if total_rows_diff > 0:
            summary_parts.append(f"Value differences in {total_diff_cols} columns across {total_rows_diff} rows")
        else:
            summary_parts.append(f"Value differences found in {total_diff_cols} columns")
    
    if not summary_parts:
        result['summary'] = "Datasets appear identical or very similar"
    else:
        result['summary'] = "✗ DIFFERENCES FOUND: " + " | ".join(summary_parts)
    
    return result


def create_diff_report(df1: pd.DataFrame, df2: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       show_all_columns: bool = False) -> pd.DataFrame:
    """
    Create a detailed difference report showing exact row-by-row comparisons.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame  
        columns: Specific columns to check (None = all common columns)
        show_all_columns: If True, include all columns in output; if False, only show differing values
    
    Returns:
        DataFrame with columns: [row_number, column_name, df1_value, df2_value, match_status]
        showing all differences found between datasets
    """
    if df1.shape != df2.shape:
        raise ValueError(f"Datasets must have same shape. df1: {df1.shape}, df2: {df2.shape}")
    
    # Determine which columns to compare
    common_columns = list(set(df1.columns) & set(df2.columns))
    if columns:
        check_columns = [col for col in columns if col in common_columns]
    else:
        check_columns = common_columns
    
    # Reorder df2 to match df1 column order
    df2_ordered = df2[df1.columns] if set(df1.columns) == set(df2.columns) else df2
    
    # Collect all differences
    diff_records = []
    
    for col in check_columns:
        if col not in df2_ordered.columns:
            continue
            
        for idx in range(len(df1)):
            val1 = df1.iloc[idx][col]
            val2 = df2_ordered.iloc[idx][col]
            
            # Check if values differ (handling NaN properly)
            is_different = False
            if pd.isna(val1) and pd.isna(val2):
                is_different = False  # Both NaN = same
            elif pd.isna(val1) or pd.isna(val2):
                is_different = True  # One NaN, one not = different
            else:
                is_different = val1 != val2
            
            if is_different or show_all_columns:
                diff_records.append({
                    'row_number': idx,
                    'column_name': col,
                    'df1_value': val1,
                    'df2_value': val2,
                    'match_status': '✓ Match' if not is_different else '✗ DIFFER',
                    'difference_type': _classify_difference(val1, val2) if is_different else 'identical'
                })
    
    if not diff_records:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['row_number', 'column_name', 'df1_value', 'df2_value', 
                                    'match_status', 'difference_type'])
    
    diff_df = pd.DataFrame(diff_records)
    
    # Sort by row number, then column name
    diff_df = diff_df.sort_values(['row_number', 'column_name']).reset_index(drop=True)
    
    return diff_df


def _classify_difference(val1, val2) -> str:
    """Helper function to classify the type of difference between two values."""
    if pd.isna(val1) and not pd.isna(val2):
        return 'NaN vs Value'
    elif not pd.isna(val1) and pd.isna(val2):
        return 'Value vs NaN'
    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return 'Numeric difference'
    elif isinstance(val1, str) and isinstance(val2, str):
        return 'Text difference'
    else:
        return 'Type mismatch'


def plot_dual(df1: pd.DataFrame, df2: pd.DataFrame, 
              column: str, 
              plot_type: str = 'bar',
              title: Optional[str] = None,
              labels: tuple = ('df1', 'df2'),
              figsize: tuple = None) -> plt.Figure:
    """
    Create a single plot comparing data from two DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        column: Column name to plot (must exist in both DataFrames)
        plot_type: Type of plot - 'bar', 'line', 'hist', 'box', 'scatter'
        title: Plot title (auto-generated if None)
        labels: Tuple of labels for (df1, df2)
        figsize: Figure size tuple (default: (6, 4))
    
    Returns:
        matplotlib Figure object
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    
    # Validate that column exists in both dataframes
    if column not in df1.columns:
        raise ValueError(f"Column '{column}' not found in df1")
    if column not in df2.columns:
        raise ValueError(f"Column '{column}' not found in df2")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if title is None:
        title = f'Comparison: {column}'
    
    if plot_type == 'bar':
        # Aggregate data for bar chart
        if pd.api.types.is_numeric_dtype(df1[column]) and pd.api.types.is_numeric_dtype(df2[column]):
            x = ['Mean', 'Median', 'Max', 'Min']
            df1_vals = [df1[column].mean(), df1[column].median(), df1[column].max(), df1[column].min()]
            df2_vals = [df2[column].mean(), df2[column].median(), df2[column].max(), df2[column].min()]
            
            x_pos = np.arange(len(x))
            width = 0.35
            
            ax.bar(x_pos - width/2, df1_vals, width, label=labels[0], alpha=0.8)
            ax.bar(x_pos + width/2, df2_vals, width, label=labels[1], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x)
            ax.set_ylabel(column)
        else:
            # For categorical, show value counts
            counts1 = df1[column].value_counts().head(10)
            counts2 = df2[column].value_counts().head(10)
            
            # Get all unique categories from both datasets
            all_categories = sorted(set(list(counts1.index) + list(counts2.index)))[:10]
            x_pos = np.arange(len(all_categories))
            width = 0.35
            
            vals1 = [counts1.get(cat, 0) for cat in all_categories]
            vals2 = [counts2.get(cat, 0) for cat in all_categories]
            
            ax.bar(x_pos - width/2, vals1, width, label=labels[0], alpha=0.8)
            ax.bar(x_pos + width/2, vals2, width, label=labels[1], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.set_ylabel('Count')
    
    elif plot_type == 'line':
        # Use index for x-axis
        ax.plot(df1.index, df1[column].values, label=labels[0], alpha=0.8, marker='o', markersize=3)
        ax.plot(df2.index, df2[column].values, label=labels[1], alpha=0.8, marker='s', markersize=3)
        ax.set_xlabel('Index')
        ax.set_ylabel(column)
    
    elif plot_type == 'hist':
        # Create histogram with shared bins
        all_data = pd.concat([df1[column].dropna(), df2[column].dropna()])
        bins = np.histogram_bin_edges(all_data, bins=30)
        
        ax.hist(df1[column].dropna(), alpha=0.6, label=labels[0], bins=bins, edgecolor='black')
        ax.hist(df2[column].dropna(), alpha=0.6, label=labels[1], bins=bins, edgecolor='black')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
    
    elif plot_type == 'box':
        data = [df1[column].dropna(), df2[column].dropna()]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        # Color the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel(column)
    
    elif plot_type == 'scatter':
        # For scatter, we'll plot index vs value for both
        ax.scatter(df1.index, df1[column], alpha=0.6, label=labels[0], s=30)
        ax.scatter(df2.index, df2[column], alpha=0.6, label=labels[1], s=30)
        ax.set_xlabel('Index')
        ax.set_ylabel(column)
    
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be 'bar', 'line', 'hist', 'box', or 'scatter'")
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig

# ------------------  QueryUnderstandingTool ---------------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    current_config = get_current_config()
    
    # Prepend the instruction to the query
    full_prompt = f"""You are a query classifier. Your task is to determine if a user query is requesting a data visualization.

IMPORTANT: Respond with ONLY 'true' or 'false' (lowercase, no quotes, no punctuation).

Classify as 'true' ONLY if the query explicitly asks for:
- A plot, chart, graph, visualization, or figure
- To "show" or "display" data visually
- To "create" or "generate" a visual representation
- Words like: plot, chart, graph, visualize, show, display, create, generate, draw

Classify as 'false' for:
- Data analysis without visualization requests
- Statistical calculations, aggregations, filtering, sorting
- Questions about data content, counts, summaries
- Requests for tables, dataframes, or text results

User query: {query}"""
    
    messages = [
        {"role": "system", "content": current_config.REASONING_FALSE},
        {"role": "user", "content": full_prompt}
    ]
    
    response = client.chat.completions.create(
        model=current_config.MODEL_NAME,
        messages=messages,
        temperature=current_config.QUERY_UNDERSTANDING_TEMPERATURE,
        max_tokens=current_config.QUERY_UNDERSTANDING_MAX_TOKENS  # We only need a short response
    )
    
    # Extract the response and convert to boolean

    intent_response = response.choices[0].message.content.strip().lower()

    return intent_response == "true"

# === CodeGeneration TOOLS ============================================


# ------------------  CodeWritingTool ---------------------------------
def CodeWritingTool(cols_info: dict, query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    
    datasets_desc = []
    for df_name, cols in cols_info.items():
        datasets_desc.append(f"{df_name} with columns: {', '.join(cols)}")
    
    datasets_text = "\n    ".join(datasets_desc)

    return f"""

    Given DataFrame(s):
    {datasets_text}

    Write Python code (pandas **only**, no plotting) to answer: 
    "{query}"

    Rules
    -----
    1. Use pandas operations on available DataFrames ({', '.join(cols_info.keys())}).
    2. Rely only on the columns in each DataFrame.
    3. You can compare, merge, or analyze DataFrames separately or together.
    4. Assign the final result to `result`.
    5. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    6. Do not include any explanations, comments, or prose outside the code block.
    7. Do **not** read files, fetch data, or use Streamlit.
    8. Do **not** import any libraries (pandas is already imported as pd, numpy as np).
    9. Handle missing values (`dropna`) before aggregations.

    Available Utility Functions
    -----
    **compare_datasets(df1, df2, on_columns=None)** - Returns DataFrame with comparison statistics (count, mean, median, std, mean_diff) for numeric columns
    
    **find_differences(df1, df2, key_columns=None, max_examples=100)** - Returns dict with:
      - 'identical': Boolean (True if datasets are completely identical)
      - 'summary': Human-readable summary with row/column counts
      - 'value_differences': Dict containing total_rows_with_differences, affected_rows list, and concrete_differences
      - Best for: Quick overview and checking if datasets are identical
    
    **create_diff_report(df1, df2, columns=None, show_all_columns=False)** - Returns detailed DataFrame with:
      - Columns: row_number, column_name, df1_value, df2_value, match_status, difference_type
      - Shows exact row-by-row, column-by-column differences in tabular format
      - Best for: Pinpointing specific variations, especially for columns like "Cost" or "Date"
      - Can filter to specific columns or show all rows
    
    **plot_dual()** - For plotting only (use in plot queries)

    Examples
    -----
    ```python
    # Single dataset analysis
    result = df1.groupby("some_column")["a_numeric_col"].mean().sort_values(ascending=False)
    
    # Quick comparison using utility function
    result = compare_datasets(df1, df2, on_columns=['sales', 'profit'])
    
    # Quick check if identical
    result = find_differences(df1, df2)
    
    # Detailed row-by-row difference report (returns clean DataFrame)
    result = create_diff_report(df1, df2)
    
    # Check specific columns only (e.g., Cost, Date)
    result = create_diff_report(df1, df2, columns=['Cost', 'Date', 'Sales'])
    
    # Show all rows including matches (for full comparison)
    result = create_diff_report(df1, df2, show_all_columns=True)
    
    # Find differences with key matching (for datasets with ID columns)
    result = find_differences(df1, df2, key_columns=['id'], max_examples=50)
    
    # Manual comparison
    result = pd.DataFrame({{'df1_mean': df1['column'].mean(), 'df2_mean': df2['column'].mean()}})
    ```

    """


# ------------------  PlotCodeGeneratorTool ---------------------------
def PlotCodeGeneratorTool(cols_info: dict, query: str) -> str:

    """Generate a prompt for the LLM to write pandas + matplotlib code for a plot based on the query and columns."""
    
    datasets_desc = []
    for df_name, cols in cols_info.items():
        datasets_desc.append(f"{df_name} with columns: {', '.join(cols)}")
    
    datasets_text = "\n    ".join(datasets_desc)

    return f"""

    Given DataFrame(s):
    {datasets_text}

    Write Python code using pandas **and matplotlib** (as plt) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Rely only on the columns in each DataFrame.
    3. You can plot data from multiple DataFrames for comparison.
    4. Assign the final result (DataFrame, Series, scalar *or* matplotlib Figure) to a variable named `result`.
    5. Create only ONE relevant plot. Set `figsize={DEFAULT_FIGSIZE}`, add title/labels.
    6. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
    7. Do not include any explanations, comments, or prose outside the code block.
    8. Handle missing values (`dropna`) before plotting/aggregations.

    Available Utility Functions
    -----
    **plot_dual(df1, df2, column, plot_type='bar', title=None, labels=('df1', 'df2'))** - Create comparison plot
      - plot_type options: 'bar', 'line', 'hist', 'box', 'scatter'
      - Returns a matplotlib Figure with both datasets visualized together
      - Handles both numeric and categorical columns appropriately
    **compare_datasets(df1, df2, on_columns=None)** - Get comparison statistics (for analysis)
    **find_differences(df1, df2, key_columns=None, max_examples=100)** - Check if identical, get exact row numbers and concrete differences
    **create_diff_report(df1, df2, columns=None, show_all_columns=False)** - Get clean DataFrame showing all differences row-by-row

    Examples
    -----
    ```python
    # Easy dual plotting with utility function
    result = plot_dual(df1, df2, 'sales', plot_type='bar', title='Sales Comparison')
    
    # Histogram comparison with custom labels
    result = plot_dual(df1, df2, 'revenue', plot_type='hist', labels=('Q1 2024', 'Q2 2024'))
    
    # Box plot comparison
    result = plot_dual(df1, df2, 'price', plot_type='box', labels=('Before', 'After'))
    
    # Line plot for trends
    result = plot_dual(df1, df2, 'temperature', plot_type='line')
    
    # Manual comparison plot (if needed)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df1['date'], df1['value'], label='Dataset 1')
    ax.plot(df2['date'], df2['value'], label='Dataset 2')
    ax.legend()
    ax.set_title('Comparison')
    result = fig
    ```

    """
  

# === CodeGenerationAgent ==============================================

def CodeGenerationAgent(query: str, dataframes: dict, chat_context: Optional[str] = None):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""

    should_plot = QueryUnderstandingTool(query)
    
    # Build column info for all available dataframes
    cols_info = {df_name: df.columns.tolist() for df_name, df in dataframes.items()}

    prompt = PlotCodeGeneratorTool(cols_info, query) if should_plot else CodeWritingTool(cols_info, query)

    # Prepend the instruction to the query
    context_section = f"\nConversation context (recent user turns):\n{chat_context}\n" if chat_context else ""

    full_prompt = f"""You are a senior Python data analyst who writes clean, efficient code. 
    Solve the given problem with optimal pandas operations. Be concise and focused. 
    Your response must contain ONLY a properly-closed ```python code block with no explanations before or after (starts with ```python and ends with ```). 
    Ensure your solution is correct, handles edge cases, and follows best practices for data analysis. 
    You can work with multiple datasets and perform comparisons, merges, or analyze them separately.
    If the latest user request references prior results ambiguously (e.g., "it", "that", "same groups"), infer intent from the conversation context and choose the most reasonable interpretation. {context_section}{prompt}"""

    current_config = get_current_config()

    messages = [
        {"role": "system", "content": current_config.REASONING_FALSE},
        {"role": "user", "content": full_prompt}
    ]

    response = client.chat.completions.create(
        model=current_config.MODEL_NAME,
        messages=messages,
        temperature=current_config.CODE_GENERATION_TEMPERATURE,
        max_tokens=current_config.CODE_GENERATION_MAX_TOKENS
    )

    full_response = response.choices[0].message.content

    code = extract_first_code_block(full_response)
    return code, should_plot, ""

# === ExecutionAgent ====================================================

def ExecutionAgent(code: str, dataframes: dict, should_plot: bool):
    """Executes the generated code in a controlled environment and returns the result or error message."""
    
    # Set default DPI for all figures
    plt.rcParams["figure.dpi"] = DEFAULT_DPI
    
    # Set up execution environment with all necessary modules
    # matplotlib is always available since utility functions need it
    env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "io": io,
        # Add utility functions for dataset comparison
        "compare_datasets": compare_datasets,
        "find_differences": find_differences,
        "create_diff_report": create_diff_report,
        "plot_dual": plot_dual
    }
    
    # Add all dataframes to the environment
    for df_name, df in dataframes.items():
        env[df_name] = df
    
    try:
        # Execute the code in the environment
        exec(code, {}, env)
        result = env.get("result", None)
        
        # If no result was assigned, return the last expression
        if result is None:
            # Try to get the last executed expression
            if "result" not in env:
                return "No result was assigned to 'result' variable"
        
        return result
    except Exception as exc:
        return f"Error executing code: {exc}"

# === ReasoningCurator TOOL =========================================
def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:MAX_RESULT_DISPLAY_LENGTH]

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2–3 concise sentences what this tells about the data (no mention of charts).'''
    return prompt

# === ReasoningAgent (streaming) =========================================
def ReasoningAgent(query: str, result: Any):
    """Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    current_config = get_current_config()
    prompt = ReasoningCurator(query, result)

    # Streaming LLM call
    response = client.chat.completions.create(
        model=current_config.MODEL_NAME,
        messages=[
            {"role": "system", "content": current_config.REASONING_TRUE},
            {"role": "user", "content": "You are an insightful data analyst. " + prompt}
        ],
        temperature=current_config.REASONING_TEMPERATURE,
        max_tokens=current_config.REASONING_MAX_TOKENS,
        stream=True
    )

    # Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token

            # Simple state machine to extract <think>...</think> as it streams
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )

    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

# === DataFrameSummary TOOL (pandas only) =========================================
def DataFrameSummaryTool(dataframes: dict) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame(s)."""
    
    dataset_descriptions = []
    for df_name, df in dataframes.items():
        dataset_descriptions.append(f"""
        {df_name}: {len(df)} rows, {len(df.columns)} columns
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}
        """)
    
    datasets_text = "\n".join(dataset_descriptions)
    
    if len(dataframes) == 1:
        prompt = f"""
        Given a dataset:
        {datasets_text}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    else:
        prompt = f"""
        Given multiple datasets:
        {datasets_text}

        Provide:
        1. A brief description of what each dataset contains
        2. 3-4 possible data analysis questions, including potential comparisons or relationships between the datasets
        Keep it concise and focused."""
    
    return prompt

# === DataInsightAgent (upload-time only) ===============================

def DataInsightAgent(dataframes: dict) -> str:
    """Uses the LLM to generate a brief summary and possible questions for the uploaded dataset(s)."""
    current_config = get_current_config()
    prompt = DataFrameSummaryTool(dataframes)
    try:
        response = client.chat.completions.create(
            model=current_config.MODEL_NAME,
            messages=[
                {"role": "system", "content": current_config.REASONING_FALSE},
                {"role": "user", "content": "You are a data analyst providing brief, focused insights. " + prompt}
            ],
            temperature=current_config.INSIGHTS_TEMPERATURE,
            max_tokens=current_config.INSIGHTS_MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise Exception(f"Error generating dataset insights: {exc}")

# === Helpers ===========================================================

def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Main Streamlit App ===============================================

def main():
    st.set_page_config(
        layout="wide",
        page_title="DataChat AI • NVIDIA Powered",
        page_icon="🤖"
    )
    
    # Futuristic Blue & White CSS Styling
    st.markdown("""
    <style>
        /* Import Modern Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1420 100%);
            color: #e8f1ff;
        }
        
        /* Top Header Bar */
        header[data-testid="stHeader"] {
            background: linear-gradient(135deg, #1a1f3a 0%, #0f1629 100%) !important;
            border-bottom: 1px solid rgba(59, 130, 246, 0.3) !important;
        }
        
        /* Custom App Title in Top Bar */
        .app-title-bar {
            background: linear-gradient(135deg, #1a1f3a 0%, #0f1629 100%);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(59, 130, 246, 0.3);
            margin: -1rem -1rem 2rem -1rem;
        }
        
        .app-title-bar h1 {
            color: #e8f1ff;
            font-size: 1.8rem;
            font-weight: 600;
            margin: 0;
            text-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        }
        
        .app-title-bar .subtitle {
            color: #cbd5e1;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 600;
            letter-spacing: -0.5px;
            text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1629 0%, #1a1f3a 100%);
            border-right: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(59, 130, 246, 0.6);
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 500;
            letter-spacing: 0.3px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
            transform: translateY(-2px);
        }
        
        /* Select Box */
        [data-baseweb="select"] {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
            border-radius: 8px !important;
        }
        
        [data-baseweb="select"] > div {
            color: #e8f1ff !important;
        }
        
        /* Labels and Text */
        label, .stMarkdown, p {
            color: #cbd5e1 !important;
        }
        
        /* File uploader text */
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small {
            color: #cbd5e1 !important;
        }
        
        /* Chat Input */
        .stChatInput > div > div > input {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 12px;
            color: #e8f1ff;
            padding: 12px 20px;
            font-size: 14px;
        }
        
        .stChatInput > div > div > input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }
        
        /* Chat Messages */
        [data-testid="stChatMessageContent"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 16px;
            backdrop-filter: blur(10px);
        }
        
        /* User Message */
        [data-testid="stChatMessage"][data-testid*="user"] {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.1) 100%);
        }
        
        /* Assistant Message */
        [data-testid="stChatMessage"][data-testid*="assistant"] {
            background: rgba(15, 23, 42, 0.4);
        }
        
        /* DataFrames */
        [data-testid="stDataFrame"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Code Blocks */
        pre, code {
            background: rgba(10, 14, 39, 0.8) !important;
            border: 1px solid rgba(59, 130, 246, 0.2) !important;
            border-radius: 8px !important;
            color: #cbd5e1 !important;
        }
        
        /* Details/Accordion */
        details {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 10px;
            padding: 12px 16px;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        
        details:hover {
            border-color: rgba(59, 130, 246, 0.5);
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
        }
        
        details summary {
            cursor: pointer;
            font-weight: 500;
            color: #60a5fa;
            user-select: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        details[open] summary {
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        details pre {
            margin: 0;
            padding: 12px;
            font-size: 13px;
            line-height: 1.6;
        }
        
        /* Thinking Block */
        .thinking {
            border-left: 3px solid #3b82f6;
            background: rgba(59, 130, 246, 0.05);
        }
        
        /* Code Block */
        .code {
            border-left: 3px solid #10b981;
            background: rgba(16, 185, 129, 0.05);
        }
        
        /* Links */
        a {
            color: #60a5fa;
            text-decoration: none;
            transition: all 0.2s ease;
        }
        
        a:hover {
            color: #3b82f6;
            text-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: #3b82f6 transparent transparent transparent !important;
        }
        
        /* Info/Warning/Error Boxes */
        .stAlert {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        
        /* Markdown */
        .stMarkdown {
            color: #cbd5e1;
        }
        
        /* SVG Icons Color */
        svg {
            filter: drop-shadow(0 0 4px rgba(59, 130, 246, 0.3));
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #3b82f6, #2563eb);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #2563eb, #1d4ed8);
        }
        
        /* Expander */
        [data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 10px;
        }
        
        /* Metric */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 16px;
        }
        
        /* Success Message */
        .stSuccess {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10b981;
        }
        
        /* Custom Header with Glow */
        .header-glow {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
            border-radius: 16px;
            margin-bottom: 30px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .header-glow h1 {
            font-size: 2.5rem;
            margin: 0;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Add custom app title bar
    st.markdown("""
        <div class="app-title-bar">
            <h1>🤖 DataChat AI</h1>
            <div class="subtitle">Intelligent Data Analysis Powered by NVIDIA</div>
        </div>
    """, unsafe_allow_html=True)
    
    if "plots" not in st.session_state:
        st.session_state.plots = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = DEFAULT_MODEL

    left, right = st.columns([3,7])

    with left:
        st.header("Data Analysis Agent")
        
        # Model selector
        available_models = list(MODEL_CONFIGS.keys())
        model_display_names = {key: MODEL_CONFIGS[key].MODEL_PRINT_NAME for key in available_models}
        
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            format_func=lambda x: model_display_names[x],
            index=available_models.index(st.session_state.current_model)
        )
        
        # Get current config for the "Powered by" text - use selected_model for immediate updates
        display_config = MODEL_CONFIGS[selected_model]

        st.markdown(f"<p style='color: #cbd5e1; font-size: 0.9em;'>Powered by <a href='{display_config.MODEL_URL}'>{display_config.MODEL_PRINT_NAME}</a></p>", unsafe_allow_html=True)
        
        # Two file uploaders
        st.markdown("### Upload Dataset(s)")
        file1 = st.file_uploader("Dataset 1 (df1)", type=["csv"], key="csv_uploader_1")
        file2 = st.file_uploader("Dataset 2 (df2) - Optional", type=["csv"], key="csv_uploader_2")
        
        # Update configuration if model changed
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            # Get the updated config for the new model
            new_config = MODEL_CONFIGS[selected_model]

            # Clear chat history when model changes
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "plots" in st.session_state:
                st.session_state.plots = []

            # Regenerate insights immediately if we have data and file is present
            if "dataframes" in st.session_state and len(st.session_state.dataframes) > 0:
                with st.spinner("Generating dataset insights with new model …"):
                    try:
                        st.session_state.insights = DataInsightAgent(st.session_state.dataframes)
                        st.success(f"Insights updated with {new_config.MODEL_PRINT_NAME}")
                    except Exception as e:
                        st.error(f"Error updating insights: {str(e)}")
                        # Clear old insights if regeneration fails
                        if "insights" in st.session_state:
                            del st.session_state.insights
                st.rerun()  # Force UI update to show new insights
        
        
        # Clear data if files are removed (but not during model change). Keep chat history.
        if not file1 and "dataframes" in st.session_state:
            del st.session_state.dataframes
            del st.session_state.current_files
            if "insights" in st.session_state:
                del st.session_state.insights
            st.rerun()
        
        # Process uploaded files
        current_files = (file1.name if file1 else None, file2.name if file2 else None)
        files_changed = st.session_state.get("current_files") != current_files
        
        if file1 and files_changed:
            dataframes = {}
            dataframes["df1"] = pd.read_csv(file1)
            
            if file2:
                dataframes["df2"] = pd.read_csv(file2)
            
            st.session_state.dataframes = dataframes
            st.session_state.current_files = current_files
            st.session_state.messages = []
            
            # Generate insights with the current model
            with st.spinner("Generating dataset insights …"):
                try:
                    st.session_state.insights = DataInsightAgent(dataframes)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
        
        # Ensure insights exist even if they weren't generated properly
        elif file1 and "insights" not in st.session_state and "dataframes" in st.session_state:
            with st.spinner("Generating dataset insights …"):
                try:
                    st.session_state.insights = DataInsightAgent(st.session_state.dataframes)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
        
        # Display data and insights (always execute if we have data)
        if "dataframes" in st.session_state:
            st.markdown("### Dataset Insights")
            
            # Display insights with model attribution
            if "insights" in st.session_state and st.session_state.insights:
                # Show preview of each dataset
                for df_name, df in st.session_state.dataframes.items():
                    st.markdown(f"**{df_name}** ({len(df)} rows × {len(df.columns)} columns)")
                    st.dataframe(df.head(3))
                
                st.markdown(st.session_state.insights)
                # Get current config dynamically for model attribution
                current_config_left = get_current_config()
                st.markdown(f"*<span style='color: #cbd5e1; font-style: italic;'>Generated with {current_config_left.MODEL_PRINT_NAME}</span>*", unsafe_allow_html=True)
            else:
                st.warning("No insights available.")
        else:
            st.info("Upload at least one CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data") 
        if "dataframes" in st.session_state:
            # Get current config dynamically for the right column
            current_config_right = get_current_config()
            dataset_names = ", ".join(st.session_state.dataframes.keys())
            st.markdown(f"*<span style='color: #cbd5e1; font-style: italic;'>Using {current_config_right.MODEL_PRINT_NAME} | Datasets: {dataset_names}</span>*", unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Manual clear chat control
        clear_col1, clear_col2 = st.columns([8,2])
        with clear_col2:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.plots = []
                st.rerun()

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Display plot at fixed size
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if "dataframes" in st.session_state:  # allow chat when we have data loaded
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Working …"):
                    # Build brief chat context from the last few user messages
                    recent_user_turns = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-3:]
                    context_text = "\n".join(recent_user_turns[:-1]) if len(recent_user_turns) > 1 else None
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.dataframes, context_text)
                    result_obj = ExecutionAgent(code, st.session_state.dataframes, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build assistant response
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                else:
                    header = f"Result: {result_obj}"

                # Show only reasoning thinking in Model Thinking (collapsed by default)
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="7.5 4.21 12 6.81 16.5 4.21"/><polyline points="7.5 19.79 7.5 14.6 3 12"/><polyline points="21 12 16.5 14.6 16.5 19.79"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg> Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Code accordion with proper HTML <pre><code> syntax highlighting
                code_html = (
                    '<details class="code">'
                    '<summary><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg> View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                # Combine thinking, explanation, and code accordion
                assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()

if __name__ == "__main__":
    main() 